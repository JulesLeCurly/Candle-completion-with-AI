"""
Dataset builder module - FIXED VERSION
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple
import config

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT
)
logger = logging.getLogger(__name__)


class DatasetBuilder:
    """Builds training datasets with synthetic gaps for model training."""
    
    def __init__(self, symbol_encoder: Dict[str, int] = None):
        self.symbol_encoder = symbol_encoder or self._create_symbol_encoder()
        self.samples = []
        self.labels = []
        self.metadata = []
    
    def _create_symbol_encoder(self) -> Dict[str, int]:
        return {symbol: idx for idx, symbol in enumerate(config.SYMBOLS)}
    
    def create_synthetic_gaps(self, df: pd.DataFrame, 
                             gap_ratio: float = None) -> List[Tuple[int, int]]:
        gap_ratio = gap_ratio or config.SYNTHETIC_GAP_RATIO
        total_points = len(df)
        num_gaps = int(total_points * gap_ratio / config.MAX_GAP_LENGTH)
        
        gaps = []
        used_indices = set()
        
        for _ in range(num_gaps):
            gap_length = np.random.randint(
                config.MIN_SYNTHETIC_GAP,
                config.MAX_SYNTHETIC_GAP + 1
            )
            
            max_start = total_points - gap_length - config.LOOKBACK_WINDOW
            if max_start <= config.LOOKBACK_WINDOW:
                continue
            
            start_idx = np.random.randint(config.LOOKBACK_WINDOW, max_start)
            
            gap_range = set(range(start_idx, start_idx + gap_length))
            if gap_range.intersection(used_indices):
                continue
            
            gaps.append((start_idx, gap_length))
            used_indices.update(gap_range)
        
        logger.info(f"Created {len(gaps)} synthetic gaps")
        return gaps
    
    def create_sample(self, df_primary: pd.DataFrame, 
                     df_secondary: pd.DataFrame,
                     gap_start: int, 
                     gap_length: int,
                     symbol: str) -> Dict:
        """Create a single training sample - FIXED to include ALL features."""
        
        # Context from primary exchange (before gap)
        context_start = gap_start - config.LOOKBACK_WINDOW
        context_end = gap_start
        primary_context = df_primary.iloc[context_start:context_end]
        
        # Context from secondary exchange (before + during gap)
        secondary_context_end = gap_start + config.MAX_GAP_LENGTH
        secondary_context = df_secondary.iloc[context_start:secondary_context_end]
        
        # Target: primary exchange data during gap
        target = df_primary.iloc[gap_start:gap_start + gap_length]
        
        # FIXED: Include ALL features (OHLCV + derived + temporal)
        feature_cols = []
        for col in config.OHLCV_FEATURES + config.DERIVED_FEATURES + config.TEMPORAL_FEATURES:
            if col in df_primary.columns:
                feature_cols.append(col)
        
        logger.debug(f"Using {len(feature_cols)} features: {feature_cols}")
        
        # Extract values
        primary_values = primary_context[feature_cols].values
        secondary_values = secondary_context[feature_cols].values
        target_values = target[config.OHLCV_FEATURES].values
        
        # Pad primary context to LOOKBACK_WINDOW
        if len(primary_values) < config.LOOKBACK_WINDOW:
            padding = np.zeros((config.LOOKBACK_WINDOW - len(primary_values), len(feature_cols)))
            primary_values = np.vstack([padding, primary_values])
        elif len(primary_values) > config.LOOKBACK_WINDOW:
            primary_values = primary_values[-config.LOOKBACK_WINDOW:]
        
        # Pad secondary context to LOOKBACK_WINDOW + MAX_GAP_LENGTH
        expected_secondary_len = config.LOOKBACK_WINDOW + config.MAX_GAP_LENGTH
        if len(secondary_values) < expected_secondary_len:
            padding = np.zeros((expected_secondary_len - len(secondary_values), len(feature_cols)))
            secondary_values = np.vstack([padding, secondary_values])
        elif len(secondary_values) > expected_secondary_len:
            secondary_values = secondary_values[-expected_secondary_len:]
        
        sample = {
            'primary_context': primary_values,
            'secondary_context': secondary_values,
            'target': target_values,
            'gap_length': gap_length,
            'symbol_id': self.symbol_encoder.get(symbol, 0),
            'gap_position': np.arange(gap_length) / gap_length
        }
        
        return sample
    
    def build_dataset(self, df_primary: pd.DataFrame, 
                     df_secondary: pd.DataFrame,
                     symbol: str,
                     gaps: List[Tuple[int, int]] = None) -> Tuple[np.ndarray, np.ndarray, List]:
        """Build complete dataset for a symbol."""
        
        if gaps is None:
            gaps = self.create_synthetic_gaps(df_primary)
        
        samples = []
        labels = []
        metadata = []
        
        for gap_start, gap_length in gaps:
            try:
                sample = self.create_sample(
                    df_primary, 
                    df_secondary,
                    gap_start,
                    gap_length,
                    symbol
                )
                
                # Flatten contexts for model input
                primary_flat = sample['primary_context'].flatten()
                secondary_flat = sample['secondary_context'].flatten()
                
                # Metadata features
                meta_features = np.array([
                    sample['gap_length'],
                    sample['symbol_id']
                ])
                
                # Combine all features
                X = np.concatenate([primary_flat, secondary_flat, meta_features])
                y = sample['target']
                
                samples.append(X)
                labels.append(y)
                metadata.append({
                    'gap_start': gap_start,
                    'gap_length': gap_length,
                    'symbol': symbol,
                    'symbol_id': sample['symbol_id']
                })
                
            except Exception as e:
                logger.warning(f"Error creating sample at {gap_start}: {e}")
                continue
        
        if not samples:
            logger.warning(f"No samples created for {symbol}")
            return np.array([]), np.array([]), []
        
        X_array = np.array(samples)
        y_padded = self._pad_labels(labels)
        
        logger.info(f"Built {len(samples)} samples for {symbol} with input shape {X_array.shape}")
        return X_array, y_padded, metadata
    
    def _pad_labels(self, labels: List[np.ndarray]) -> np.ndarray:
        """Pad labels to max gap length."""
        max_gap = config.MAX_GAP_LENGTH
        num_features = len(config.OHLCV_FEATURES)
        padded = np.zeros((len(labels), max_gap, num_features))
        
        for i, label in enumerate(labels):
            padded[i, :len(label), :] = label
        
        return padded
    
    def temporal_split(self, X: np.ndarray, y: np.ndarray, 
                      metadata: List) -> Dict[str, Dict]:
        """Split data temporally (no random shuffle)."""
        n_samples = len(X)
        
        train_end = int(n_samples * config.TRAIN_SPLIT)
        val_end = int(n_samples * (config.TRAIN_SPLIT + config.VALIDATION_SPLIT))
        
        splits = {
            'train': {
                'X': X[:train_end],
                'y': y[:train_end],
                'metadata': metadata[:train_end]
            },
            'val': {
                'X': X[train_end:val_end],
                'y': y[train_end:val_end],
                'metadata': metadata[train_end:val_end]
            },
            'test': {
                'X': X[val_end:],
                'y': y[val_end:],
                'metadata': metadata[val_end:]
            }
        }
        
        logger.info(f"Split: train={len(splits['train']['X'])}, "
                   f"val={len(splits['val']['X'])}, "
                   f"test={len(splits['test']['X'])}")
        
        return splits
    
    def build_multi_symbol_dataset(self, data_dict: Dict[str, Tuple]) -> Dict[str, Dict]:
        """Build combined dataset from multiple symbols."""
        all_X = []
        all_y = []
        all_metadata = []
        
        expected_input_size = None
        
        for symbol, (df_primary, df_secondary) in data_dict.items():
            X, y, metadata = self.build_dataset(df_primary, df_secondary, symbol)
            
            if len(X) > 0:
                if expected_input_size is None:
                    expected_input_size = X.shape[1]
                    logger.info(f"Expected input size: {expected_input_size}")
                
                if X.shape[1] != expected_input_size:
                    logger.warning(f"Skipping {symbol}: input size mismatch {X.shape[1]} != {expected_input_size}")
                    continue
                
                all_X.append(X)
                all_y.append(y)
                all_metadata.extend(metadata)
        
        if not all_X:
            logger.error("No data to build dataset")
            return {}
        
        X_combined = np.vstack(all_X)
        y_combined = np.vstack(all_y)
        
        logger.info(f"Combined dataset shape: X={X_combined.shape}, y={y_combined.shape}")
        
        splits = self.temporal_split(X_combined, y_combined, all_metadata)
        
        logger.info(f"Built multi-symbol dataset with {len(X_combined)} total samples")
        return splits