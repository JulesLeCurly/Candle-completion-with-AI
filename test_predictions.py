"""
Test script to visualize model predictions on test data.
Shows real vs predicted candles with visual comparison.
"""

import pandas as pd
import numpy as np
import logging
import os
import argparse
from tensorflow import keras
import config
from feature_engineering import FeatureEngineer
from dataset_builder import DatasetBuilder
from predictor import GapPredictor
from visualizer import CandlestickVisualizer
from model import AttentionLayer, ohlc_constraint_loss, mae_metric, mape_metric

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT
)
logger = logging.getLogger(__name__)


def load_trained_model(model_path: str = None):
    """Load trained model with custom objects."""
    model_path = model_path or config.MODEL_PATH
    
    if not os.path.exists(model_path):
        logger.error(f"Model not found at {model_path}")
        return None
    
    logger.info(f"Loading model from {model_path}")
    model = keras.models.load_model(
        model_path,
        custom_objects={
            'AttentionLayer': AttentionLayer,
            'ohlc_constraint_loss': ohlc_constraint_loss,
            'mae_metric': mae_metric,
            'mape_metric': mape_metric
        }
    )
    return model


def test_prediction_visualization(symbol: str = 'BTC/USDT', 
                                  num_examples: int = 3):
    """
    Test predictions and create visualizations.
    
    Args:
        symbol: Symbol to test
        num_examples: Number of gap predictions to visualize
    """
    logger.info("=" * 80)
    logger.info(f"TESTING PREDICTIONS FOR {symbol}")
    logger.info("=" * 80)
    
    # Load model
    model = load_trained_model()
    if model is None:
        return
    
    # Load feature engineer
    feature_engineer = FeatureEngineer()
    norm_params_path = os.path.join(config.MODEL_DIR, 'normalization_params.pkl')
    if os.path.exists(norm_params_path):
        feature_engineer.load_normalization_params(norm_params_path)
    
    # Load data
    safe_symbol = symbol.replace('/', '_')
    primary_file = f"{config.RAW_DATA_DIR}/{safe_symbol}_{config.PRIMARY_EXCHANGE}.csv"
    secondary_file = f"{config.RAW_DATA_DIR}/{safe_symbol}_{config.SECONDARY_EXCHANGE}.csv"
    
    if not os.path.exists(primary_file) or not os.path.exists(secondary_file):
        logger.error(f"Data files not found for {symbol}")
        return
    
    logger.info(f"Loading data for {symbol}")
    df_primary = pd.read_csv(primary_file)
    df_secondary = pd.read_csv(secondary_file)
    
    df_primary['open_time'] = pd.to_datetime(df_primary['open_time'])
    df_secondary['open_time'] = pd.to_datetime(df_secondary['open_time'])
    
    # Process features
    logger.info("Processing features")
    df_primary_proc, df_secondary_proc = feature_engineer.process_symbol(
        df_primary, df_secondary, symbol, fit=False
    )
    
    # Initialize predictor
    predictor = GapPredictor(model, feature_engineer)
    
    # Detect gaps
    gaps = predictor.detect_real_gaps(df_primary_proc)
    
    if not gaps:
        logger.warning(f"No gaps found in {symbol}")
        
        # Create synthetic gap for demonstration
        logger.info("Creating synthetic gap for demonstration")
        dataset_builder = DatasetBuilder()
        synthetic_gaps = dataset_builder.create_synthetic_gaps(df_primary_proc, gap_ratio=0.02)
        
        if not synthetic_gaps:
            logger.error("Could not create synthetic gaps")
            return
        
        gaps = [(idx, df_primary_proc.iloc[idx]['open_time'], length) 
                for idx, length in synthetic_gaps[:num_examples]]
    
    # Limit to num_examples
    gaps = gaps[:num_examples]
    
    # Initialize visualizer
    visualizer = CandlestickVisualizer()
    
    # Process each gap
    for i, (gap_idx, gap_start, gap_length) in enumerate(gaps):
        logger.info(f"\nProcessing gap {i+1}/{len(gaps)}: {gap_length}h at index {gap_idx}")
        
        # Create data with gap
        df_with_gap = df_primary_proc.copy()
        gap_end_idx = gap_idx + gap_length
        original_gap_data = df_with_gap.iloc[gap_idx:gap_end_idx].copy()
        
        # Predict gap
        predictions = predictor.predict_gap(
            df_primary_proc,
            df_secondary_proc,
            gap_idx,
            gap_start,
            gap_length,
            symbol
        )
        
        # Denormalize predictions
        predictions = feature_engineer.denormalize(predictions, symbol, config.OHLCV_FEATURES)
        
        # Post-process
        context_before = df_primary_proc.iloc[max(0, gap_idx-10):gap_idx]
        context_after = df_primary_proc.iloc[gap_end_idx:gap_end_idx+10]
        predictions = predictor.post_process_predictions(
            predictions, context_before, context_after
        )
        
        # Create comparison visualization
        logger.info(f"Creating visualization for gap {i+1}")
        
        # Denormalize context
        context_before_denorm = feature_engineer.denormalize(
            context_before.copy(), symbol, config.OHLCV_FEATURES
        )
        context_after_denorm = feature_engineer.denormalize(
            context_after.copy(), symbol, config.OHLCV_FEATURES
        )
        original_gap_denorm = feature_engineer.denormalize(
            original_gap_data.copy(), symbol, config.OHLCV_FEATURES
        )
        
        # Combine for visualization
        window_size = 24
        start_idx = max(0, gap_idx - window_size)
        end_idx = min(len(df_primary_proc), gap_end_idx + window_size)
        
        # Create comparison
        fig_title = f"{symbol} - Gap {i+1}: {gap_length}h prediction"
        
        # Plot 1: Original vs Predicted
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
        
        # Top: Show context + real gap data
        window_real = df_primary_proc.iloc[start_idx:end_idx].copy()
        window_real = feature_engineer.denormalize(window_real, symbol, config.OHLCV_FEATURES)
        window_real['is_predicted'] = False
        
        visualizer._plot_simple_candles(ax1, window_real, 'blue')
        ax1.axvspan(gap_idx - start_idx - 0.5, gap_end_idx - start_idx - 0.5,
                   alpha=0.2, color='yellow', label='Actual Data (hidden from model)')
        ax1.set_title(f'{fig_title} - Ground Truth', fontsize=14, fontweight='bold')
        ax1.legend()
        
        # Bottom: Show context + predictions
        window_pred = pd.concat([
            context_before_denorm,
            predictions,
            context_after_denorm
        ]).reset_index(drop=True)
        
        window_pred['is_predicted'] = False
        if len(context_before_denorm) > 0:
            pred_start = len(context_before_denorm)
            pred_end = pred_start + len(predictions)
            window_pred.loc[pred_start:pred_end-1, 'is_predicted'] = True
        
        visualizer._plot_simple_candles(ax2, window_pred[~window_pred['is_predicted']], 'blue')
        visualizer._plot_simple_candles(ax2, window_pred[window_pred['is_predicted']], 'orange')
        
        ax2.axvspan(len(context_before_denorm) - 0.5, 
                   len(context_before_denorm) + gap_length - 0.5,
                   alpha=0.2, color='green', label='AI Predictions')
        ax2.set_title(f'{fig_title} - Model Predictions', fontsize=14, fontweight='bold')
        ax2.legend()
        
        plt.tight_layout()
        
        # Save
        save_path = os.path.join(config.LOG_DIR, f'prediction_test_{safe_symbol}_gap{i+1}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved visualization to {save_path}")
        plt.show()
        
        # Calculate error metrics
        mae = np.mean(np.abs(original_gap_denorm['close'].values - predictions['close'].values))
        mape = np.mean(np.abs((original_gap_denorm['close'].values - predictions['close'].values) / 
                             original_gap_denorm['close'].values)) * 100
        
        logger.info(f"Gap {i+1} Metrics:")
        logger.info(f"  MAE (close): {mae:.4f}")
        logger.info(f"  MAPE (close): {mape:.2f}%")
        logger.info(f"  Avg Confidence: {predictions['prediction_confidence'].mean():.3f}")
    
    logger.info("\n" + "=" * 80)
    logger.info("TESTING COMPLETED")
    logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Test and visualize predictions')
    
    parser.add_argument(
        '--symbol',
        type=str,
        default='BTC/USDT',
        help='Symbol to test (default: BTC/USDT)'
    )
    
    parser.add_argument(
        '--num-examples',
        type=int,
        default=3,
        help='Number of examples to visualize (default: 3)'
    )
    
    args = parser.parse_args()
    
    test_prediction_visualization(args.symbol, args.num_examples)


if __name__ == '__main__':
    main()