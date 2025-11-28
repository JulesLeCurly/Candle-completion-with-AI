"""
Prediction module for completing missing cryptocurrency data.
Detects real gaps, generates predictions, and applies post-processing.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
import config
from feature_engineering import FeatureEngineer

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT
)
logger = logging.getLogger(__name__)


class GapPredictor:
    """Predicts and fills gaps in cryptocurrency data."""
    
    def __init__(self, model, feature_engineer: FeatureEngineer):
        """
        Initialize predictor.
        
        Args:
            model: Trained Keras model
            feature_engineer: FeatureEngineer instance with fitted parameters
        """
        self.model = model
        self.feature_engineer = feature_engineer
    
    def detect_real_gaps(self, df: pd.DataFrame) -> List[Tuple[int, datetime, int]]:
        """
        Detect actual gaps in the data.
        
        Args:
            df: DataFrame with 'open_time' column
            
        Returns:
            List of tuples (index, gap_start_time, gap_length)
        """
        df = df.sort_values('open_time').reset_index(drop=True)
        gaps = []
        
        for i in range(len(df) - 1):
            current_time = df.loc[i, 'open_time']
            next_time = df.loc[i + 1, 'open_time']
            expected_next = current_time + timedelta(hours=1)
            
            if next_time > expected_next:
                gap_length = int((next_time - expected_next).total_seconds() / 3600)
                
                if gap_length <= config.MAX_GAP_LENGTH:
                    gaps.append((i, expected_next, gap_length))
                else:
                    logger.warning(f"Gap too large ({gap_length}h) at {expected_next}, skipping")
        
        logger.info(f"Detected {len(gaps)} fillable gaps")
        return gaps
    
    def prepare_context(self, df_primary: pd.DataFrame, 
                       df_secondary: pd.DataFrame,
                       gap_index: int,
                       gap_length: int) -> Dict[str, np.ndarray]:
        """
        Prepare context data for prediction.
        
        Args:
            df_primary: Primary exchange data
            df_secondary: Secondary exchange data
            gap_index: Index where gap starts
            gap_length: Length of gap in hours
            
        Returns:
            Dictionary of model inputs
        """
        # Get context window before gap
        context_start = max(0, gap_index - config.LOOKBACK_WINDOW)
        context_end = gap_index
        
        primary_context = df_primary.iloc[context_start:context_end]
        
        # Find corresponding time in secondary exchange
        gap_start_time = df_primary.iloc[gap_index]['open_time']
        gap_end_time = gap_start_time + timedelta(hours=gap_length)
        
        # Get secondary data including gap period
        secondary_mask = (
            (df_secondary['open_time'] >= primary_context.iloc[0]['open_time']) &
            (df_secondary['open_time'] < gap_end_time)
        )
        secondary_context = df_secondary[secondary_mask]
        
        # Extract features
        feature_cols = config.OHLCV_FEATURES + config.DERIVED_FEATURES + config.TEMPORAL_FEATURES
        
        primary_features = primary_context[feature_cols].values
        secondary_features = secondary_context[feature_cols].values
        
        # Pad to required length
        primary_padded = self._pad_context(primary_features, config.LOOKBACK_WINDOW)
        secondary_padded = self._pad_context(
            secondary_features, 
            config.LOOKBACK_WINDOW + gap_length
        )
        
        return {
            'primary_context': primary_padded.reshape(1, -1, primary_padded.shape[-1]),
            'secondary_context': secondary_padded.reshape(1, -1, secondary_padded.shape[-1]),
            'gap_length': np.array([[gap_length]])
        }
    
    def _pad_context(self, data: np.ndarray, target_length: int) -> np.ndarray:
        """Pad or truncate context to target length."""
        if len(data) >= target_length:
            return data[-target_length:]
        else:
            padding = np.zeros((target_length - len(data), data.shape[1]))
            return np.vstack([padding, data])
    
    def predict_gap(self, df_primary: pd.DataFrame,
                   df_secondary: pd.DataFrame,
                   gap_index: int,
                   gap_start_time: datetime,
                   gap_length: int,
                   symbol: str) -> pd.DataFrame:
        """
        Predict missing candles for a gap.
        
        Args:
            df_primary: Primary exchange data
            df_secondary: Secondary exchange data
            gap_index: Index where gap starts
            gap_start_time: Start time of gap
            gap_length: Length of gap in hours
            symbol: Symbol identifier
            
        Returns:
            DataFrame with predicted candles
        """
        logger.info(f"Predicting {gap_length}h gap at {gap_start_time} for {symbol}")
        
        # Prepare context
        context = self.prepare_context(df_primary, df_secondary, gap_index, gap_length)
        
        # Add symbol encoding
        symbol_encoder = {s: i for i, s in enumerate(config.SYMBOLS)}
        context['symbol_id'] = np.array([[symbol_encoder.get(symbol, 0)]])
        
        # Make prediction
        prediction = self.model.predict(context, verbose=0)
        
        # Extract only the predicted gap length (not full max_gap_length)
        prediction = prediction[0, :gap_length, :]
        
        # Denormalize
        pred_df = pd.DataFrame(
            prediction,
            columns=config.OHLCV_FEATURES
        )
        pred_df['symbol'] = symbol
        pred_df = self.feature_engineer.denormalize(pred_df, symbol, config.OHLCV_FEATURES)
        
        # Add timestamps
        timestamps = [gap_start_time + timedelta(hours=i) for i in range(gap_length)]
        pred_df['open_time'] = timestamps
        pred_df['close_time'] = [t + timedelta(hours=1) for t in timestamps]
        
        # Mark as predicted
        pred_df['is_predicted'] = True
        pred_df['gap_length'] = gap_length
        pred_df['source_exchange'] = 'predicted'
        
        return pred_df
    
    def validate_candle(self, candle: pd.Series) -> Tuple[bool, str]:
        """
        Validate OHLC constraints for a candle.
        
        Args:
            candle: Series with OHLCV values
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        errors = []
        
        # Check OHLC relationships
        if candle['high'] < max(candle['open'], candle['close']):
            errors.append("high < max(open, close)")
        
        if candle['low'] > min(candle['open'], candle['close']):
            errors.append("low > min(open, close)")
        
        if candle['high'] < candle['low']:
            errors.append("high < low")
        
        # Check for negative or zero values
        if candle['volume'] <= 0:
            errors.append("volume <= 0")
        
        if any(candle[col] <= 0 for col in ['open', 'high', 'low', 'close']):
            errors.append("negative price values")
        
        if errors:
            return False, "; ".join(errors)
        return True, ""
    
    def correct_candle(self, candle: pd.Series, context: pd.DataFrame) -> pd.Series:
        """
        Correct invalid candle using context.
        
        Args:
            candle: Series with potentially invalid OHLCV values
            context: DataFrame with surrounding candles for reference
            
        Returns:
            Corrected candle
        """
        corrected = candle.copy()
        
        # Ensure high is maximum
        corrected['high'] = max(
            corrected['high'],
            corrected['open'],
            corrected['close']
        )
        
        # Ensure low is minimum
        corrected['low'] = min(
            corrected['low'],
            corrected['open'],
            corrected['close']
        )
        
        # Ensure positive volume
        if corrected['volume'] <= 0:
            if not context.empty:
                corrected['volume'] = context['volume'].median()
            else:
                corrected['volume'] = 1.0
        
        # Check price continuity with context
        if not context.empty:
            last_close = context.iloc[-1]['close']
            max_deviation = last_close * config.MAX_PRICE_DEVIATION
            
            # Clip prices to reasonable range
            for price_col in ['open', 'high', 'low', 'close']:
                corrected[price_col] = np.clip(
                    corrected[price_col],
                    last_close - max_deviation,
                    last_close + max_deviation
                )
        
        return corrected
    
    def post_process_predictions(self, predictions: pd.DataFrame,
                                context_before: pd.DataFrame,
                                context_after: pd.DataFrame) -> pd.DataFrame:
        """
        Apply post-processing to predictions.
        
        Args:
            predictions: DataFrame with predicted candles
            context_before: Candles before the gap
            context_after: Candles after the gap
            
        Returns:
            Post-processed predictions
        """
        processed = predictions.copy()
        num_invalid = 0
        
        for idx in processed.index:
            candle = processed.loc[idx]
            is_valid, error_msg = self.validate_candle(candle)
            
            if not is_valid:
                num_invalid += 1
                logger.debug(f"Invalid candle at {candle['open_time']}: {error_msg}")
                
                # Get local context
                local_context = context_before.tail(5)
                
                # Correct candle
                corrected = self.correct_candle(candle, local_context)
                processed.loc[idx] = corrected
        
        if num_invalid > 0:
            logger.info(f"Corrected {num_invalid}/{len(processed)} invalid candles")
        
        # Calculate confidence scores
        processed['prediction_confidence'] = self._calculate_confidence(
            processed, context_before, context_after
        )
        
        return processed
    
    def _calculate_confidence(self, predictions: pd.DataFrame,
                             context_before: pd.DataFrame,
                             context_after: pd.DataFrame) -> np.ndarray:
        """
        Calculate confidence scores for predictions.
        
        Args:
            predictions: Predicted candles
            context_before: Context before gap
            context_after: Context after gap
            
        Returns:
            Array of confidence scores (0-1)
        """
        scores = np.ones(len(predictions))
        
        if context_before.empty:
            return scores * 0.5
        
        # Check price continuity
        last_price = context_before.iloc[-1]['close']
        
        for i, row in predictions.iterrows():
            price_change = abs(row['close'] - last_price) / last_price
            
            # Lower confidence for large price changes
            if price_change > config.MAX_PRICE_DEVIATION:
                scores[i] *= 0.7
            
            # Check volume consistency
            if not context_before.empty:
                median_volume = context_before['volume'].median()
                volume_ratio = row['volume'] / median_volume if median_volume > 0 else 1
                
                if volume_ratio < config.MIN_VOLUME_RATIO or volume_ratio > config.MAX_VOLUME_RATIO:
                    scores[i] *= 0.8
            
            last_price = row['close']
        
        return scores
    
    def complete_symbol_data(self, df_primary: pd.DataFrame,
                            df_secondary: pd.DataFrame,
                            symbol: str) -> pd.DataFrame:
        """
        Complete all gaps for a symbol.
        
        Args:
            df_primary: Primary exchange data with gaps
            df_secondary: Secondary exchange complete data
            symbol: Symbol identifier
            
        Returns:
            Completed DataFrame
        """
        logger.info(f"Completing data for {symbol}")
        
        # Detect gaps
        gaps = self.detect_real_gaps(df_primary)
        
        if not gaps:
            logger.info(f"No gaps found for {symbol}")
            df_primary['is_predicted'] = False
            df_primary['prediction_confidence'] = 1.0
            return df_primary
        
        # Sort by time
        df_primary = df_primary.sort_values('open_time').reset_index(drop=True)
        
        completed_parts = []
        last_idx = 0
        
        for gap_idx, gap_start, gap_length in gaps:
            # Add data before gap
            completed_parts.append(df_primary.iloc[last_idx:gap_idx])
            
            # Predict gap
            predictions = self.predict_gap(
                df_primary,
                df_secondary,
                gap_idx,
                gap_start,
                gap_length,
                symbol
            )
            
            # Post-process
            context_before = df_primary.iloc[max(0, gap_idx-10):gap_idx]
            context_after = df_primary.iloc[gap_idx:gap_idx+10]
            
            predictions = self.post_process_predictions(
                predictions,
                context_before,
                context_after
            )
            
            completed_parts.append(predictions)
            last_idx = gap_idx
        
        # Add remaining data
        completed_parts.append(df_primary.iloc[last_idx:])
        
        # Combine all parts
        completed = pd.concat(completed_parts, ignore_index=True)
        
        # Mark original data
        completed['is_predicted'] = completed['is_predicted'].fillna(False)
        completed['prediction_confidence'] = completed['prediction_confidence'].fillna(1.0)
        
        logger.info(f"Completed {symbol}: filled {len(gaps)} gaps")
        return completed


if __name__ == '__main__':
    logger.info("Predictor module ready")
