"""
Visualization module for cryptocurrency predictions.
Creates candlestick charts and comparison plots.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from datetime import datetime
import logging
import config

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT
)
logger = logging.getLogger(__name__)


class CandlestickVisualizer:
    """Creates visualizations for cryptocurrency candlestick data."""
    
    def __init__(self):
        """Initialize visualizer with styling."""
        try:
            plt.style.use('seaborn-v0_8-darkgrid')
        except:
            try:
                plt.style.use('seaborn-darkgrid')
            except:
                plt.style.use('default')
        
        self.colors = {
            'bullish': '#26a69a',
            'bearish': '#ef5350',
            'predicted_bullish': '#4dd0e1',
            'predicted_bearish': '#ff7043',
            'original': '#1976d2'
        }
    
    def plot_candlestick(self, df: pd.DataFrame, title: str = None, 
                        save_path: str = None, figsize: tuple = (15, 8)):
        """
        Plot candlestick chart with distinction between real and predicted candles.
        
        Args:
            df: DataFrame with OHLCV and is_predicted columns
            title: Chart title
            save_path: Path to save figure
            figsize: Figure size tuple
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        if title:
            ax.set_title(title, fontsize=16, fontweight='bold')
        
        # Separate real and predicted candles
        real_df = df[~df['is_predicted']].copy() if 'is_predicted' in df.columns else df.copy()
        pred_df = df[df['is_predicted']].copy() if 'is_predicted' in df.columns else pd.DataFrame()
        
        # Plot real candles
        self._draw_candles(ax, real_df, is_predicted=False)
        
        # Plot predicted candles
        if not pred_df.empty:
            self._draw_candles(ax, pred_df, is_predicted=True)
        
        # Format x-axis
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Price', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Add legend
        real_patch = mpatches.Patch(color=self.colors['original'], label='Real Candles')
        pred_patch = mpatches.Patch(color=self.colors['predicted_bullish'], label='Predicted Candles')
        ax.legend(handles=[real_patch, pred_patch], loc='upper left')
        
        # Rotate x-axis labels
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved candlestick chart to {save_path}")
        
        plt.show()
    
    def _draw_candles(self, ax, df: pd.DataFrame, is_predicted: bool = False):
        """
        Draw candlesticks on axis.
        
        Args:
            ax: Matplotlib axis
            df: DataFrame with OHLCV data
            is_predicted: Whether these are predicted candles
        """
        if df.empty:
            return
        
        # Reset index to get sequential positions
        df = df.reset_index(drop=True)
        
        width = 0.6
        width2 = 0.05
        
        for idx, row in df.iterrows():
            open_price = row['open']
            high_price = row['high']
            low_price = row['low']
            close_price = row['close']
            
            # Determine color
            if close_price >= open_price:
                color = self.colors['predicted_bullish'] if is_predicted else self.colors['bullish']
                body_height = close_price - open_price
                body_bottom = open_price
            else:
                color = self.colors['predicted_bearish'] if is_predicted else self.colors['bearish']
                body_height = open_price - close_price
                body_bottom = close_price
            
            # Draw high-low line (wick)
            ax.plot([idx, idx], [low_price, high_price], color=color, linewidth=1, alpha=0.8)
            
            # Draw body rectangle
            alpha = 0.6 if is_predicted else 0.9
            rect = Rectangle((idx - width/2, body_bottom), width, body_height,
                           facecolor=color, edgecolor=color, alpha=alpha, linewidth=1)
            ax.add_patch(rect)
        
        # Set x-axis limits
        ax.set_xlim(-1, len(df))
    
    def plot_gap_comparison(self, df_with_gap: pd.DataFrame, 
                           df_completed: pd.DataFrame,
                           gap_start_idx: int,
                           gap_length: int,
                           symbol: str,
                           save_path: str = None):
        """
        Plot comparison between data with gap and completed data.
        
        Args:
            df_with_gap: Original data with gap
            df_completed: Data with predicted candles
            gap_start_idx: Index where gap starts
            gap_length: Length of gap
            symbol: Symbol name
            save_path: Path to save figure
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
        
        # Context window
        context_size = 24
        start_idx = max(0, gap_start_idx - context_size)
        end_idx = min(len(df_completed), gap_start_idx + gap_length + context_size)
        
        # Plot original with gap
        ax1.set_title(f'{symbol} - Original Data (with {gap_length}h gap)', 
                     fontsize=14, fontweight='bold')
        original_window = df_with_gap.iloc[start_idx:end_idx].copy()
        self._plot_simple_candles(ax1, original_window)
        
        # Highlight gap area
        ax1.axvspan(gap_start_idx - start_idx - 0.5, 
                   gap_start_idx - start_idx + gap_length - 0.5,
                   alpha=0.2, color='red', label='Gap')
        ax1.legend()
        
        # Plot completed data
        ax2.set_title(f'{symbol} - Completed Data (AI predictions)', 
                     fontsize=14, fontweight='bold')
        completed_window = df_completed.iloc[start_idx:end_idx].copy()
        
        # Mark predicted candles
        completed_window['plot_idx'] = range(len(completed_window))
        real_candles = completed_window[~completed_window['is_predicted']]
        pred_candles = completed_window[completed_window['is_predicted']]
        
        # Plot both
        self._plot_simple_candles(ax2, real_candles, color_base='blue')
        self._plot_simple_candles(ax2, pred_candles, color_base='orange')
        
        # Highlight predicted area
        ax2.axvspan(gap_start_idx - start_idx - 0.5,
                   gap_start_idx - start_idx + gap_length - 0.5,
                   alpha=0.2, color='green', label='AI Predictions')
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved comparison plot to {save_path}")
        
        plt.show()
    
    def _plot_simple_candles(self, ax, df: pd.DataFrame, color_base: str = 'blue'):
        """Simple candlestick plot for comparison."""
        if df.empty:
            return
        
        df = df.reset_index(drop=True)
        
        for idx, row in df.iterrows():
            color = 'green' if row['close'] >= row['open'] else 'red'
            if color_base == 'orange':
                color = '#ff9800' if row['close'] >= row['open'] else '#f44336'
            
            # Wick
            ax.plot([idx, idx], [row['low'], row['high']], color=color, linewidth=1)
            
            # Body
            body_height = abs(row['close'] - row['open'])
            body_bottom = min(row['open'], row['close'])
            rect = Rectangle((idx - 0.3, body_bottom), 0.6, body_height,
                           facecolor=color, edgecolor=color, alpha=0.7)
            ax.add_patch(rect)
        
        ax.set_xlim(-1, len(df))
        ax.grid(True, alpha=0.3)
        ax.set_ylabel('Price')
    
    def plot_multiple_gaps(self, df: pd.DataFrame, symbol: str,
                          max_gaps: int = 5, save_path: str = None):
        """
        Plot multiple gap predictions in a grid.
        
        Args:
            df: DataFrame with completed data
            symbol: Symbol name
            max_gaps: Maximum number of gaps to plot
            save_path: Path to save figure
        """
        # Find gap regions
        df = df.sort_values('open_time').reset_index(drop=True)
        gaps = []
        in_gap = False
        gap_start = None
        
        for idx, row in df.iterrows():
            if row['is_predicted'] and not in_gap:
                in_gap = True
                gap_start = idx
            elif not row['is_predicted'] and in_gap:
                in_gap = False
                gaps.append((gap_start, idx - gap_start))
        
        if not gaps:
            logger.warning("No gaps found in data")
            return
        
        # Limit to max_gaps
        gaps = gaps[:max_gaps]
        num_gaps = len(gaps)
        
        # Create grid
        cols = 2
        rows = (num_gaps + 1) // 2
        fig, axes = plt.subplots(rows, cols, figsize=(16, 5*rows))
        if num_gaps == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle(f'{symbol} - AI Gap Predictions', fontsize=16, fontweight='bold')
        
        for idx, (gap_start, gap_length) in enumerate(gaps):
            row = idx // cols
            col = idx % cols
            ax = axes[row, col]
            
            # Get window around gap
            context = 12
            start = max(0, gap_start - context)
            end = min(len(df), gap_start + gap_length + context)
            window = df.iloc[start:end].copy()
            
            # Plot
            self._plot_simple_candles(ax, window[~window['is_predicted']], 'blue')
            self._plot_simple_candles(ax, window[window['is_predicted']], 'orange')
            
            # Highlight gap
            ax.axvspan(gap_start - start - 0.5, 
                      gap_start - start + gap_length - 0.5,
                      alpha=0.2, color='green')
            
            # Get confidence
            gap_window = df.iloc[gap_start:gap_start+gap_length]
            avg_conf = gap_window['prediction_confidence'].mean()
            
            ax.set_title(f'Gap {idx+1}: {gap_length}h (conf: {avg_conf:.2f})', fontsize=10)
            ax.grid(True, alpha=0.3)
        
        # Hide empty subplots
        for idx in range(num_gaps, rows * cols):
            row = idx // cols
            col = idx % cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved multiple gaps plot to {save_path}")
        
        plt.show()
    
    def plot_confidence_distribution(self, df: pd.DataFrame, 
                                    save_path: str = None):
        """
        Plot distribution of prediction confidence scores.
        
        Args:
            df: DataFrame with prediction_confidence column
            save_path: Path to save figure
        """
        if 'prediction_confidence' not in df.columns:
            logger.warning("No confidence scores found")
            return
        
        predicted = df[df['is_predicted']]
        if predicted.empty:
            logger.warning("No predictions found")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        ax1.hist(predicted['prediction_confidence'], bins=30, 
                color='steelblue', alpha=0.7, edgecolor='black')
        ax1.axvline(predicted['prediction_confidence'].mean(), 
                   color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {predicted["prediction_confidence"].mean():.3f}')
        ax1.set_xlabel('Confidence Score')
        ax1.set_ylabel('Count')
        ax1.set_title('Prediction Confidence Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Box plot by gap length
        if 'gap_length' in predicted.columns:
            gap_lengths = sorted(predicted['gap_length'].unique())
            data = [predicted[predicted['gap_length'] == gl]['prediction_confidence'].values 
                   for gl in gap_lengths]
            
            ax2.boxplot(data, labels=gap_lengths)
            ax2.set_xlabel('Gap Length (hours)')
            ax2.set_ylabel('Confidence Score')
            ax2.set_title('Confidence by Gap Length')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved confidence distribution to {save_path}")
        
        plt.show()


if __name__ == '__main__':
    visualizer = CandlestickVisualizer()
    logger.info("Visualizer module ready")