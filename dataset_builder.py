"""
Dataset builder module for creating training/validation/test datasets.
Handles synthetic gap generation, sample creation, and temporal splitting.
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
        """
        Initialize dataset builder.
        
        Args:
            symbol_encoder: Dictionary mapping symbols to integer IDs
        """
        self.symbol_encoder = symbol_encoder or self._create_symbol_encoder()
        self.samples = []
        self.labels = []
        self.metadata = []
    
    def _create_symbol_encoder(self) -> Dict[str, int]:
        """Create symbol to integer encoding."""
        return {symbol: idx for idx, symbol in enumerate(config.SYMBOLS)}
    
    def create_synthetic_gaps(self, df: pd.DataFrame, 
                             gap_ratio: float = None) -> List[Tuple[int, int]]:
        """
        Create synthetic gaps by selecting random positions to mask.
        
        Args:
            df: DataFrame with complete data
            gap_ratio: Ratio of data to mask
            
        Returns:
            List of (start_idx, gap_length) tuples
        """
        gap_ratio = gap_ratio or config.SYNTHETIC_GAP_RATIO
        total_points = len(df)
        num_gaps = int(total_points * gap_ratio / config.MAX_GAP_LENGTH)
        
        gaps = []
        used_indices = set()
        
        for _ in range(num_gaps):
            # Random gap length
            gap_length = np.random.randint(
                config.MIN_SYNTHETIC_GAP,
                config.MAX_SYNTHETIC_GAP + 1
            )
            
            # Random start position (ensure context window exists)
            max_start = total_points - gap_length - config.LOOKBACK_WINDOW
            if max_start <= config.LOOKBACK_WINDOW:
                continue
            
            start_idx = np.random.randint(config.LOOKBACK_WINDOW, max_start)
            
            # Check if gap overlaps with existing gaps
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
        """
        Create a single training sample.
        
        Args:
            df_primary: Primary exchange data
            df_secondary: Secondary exchange data
            gap_start: Index where gap starts
            gap_length: Length of gap
            symbol: Symbol identifier
            
        Returns:
            Dictionary containing sample data
        """
        # Context from primary exchange (before gap)
        context_start = gap_start - config.LOOKBACK_WINDOW
        context_end = gap_start
        primary_context = df_primary.iloc[context_start:context_end]
        
        # Context from secondary exchange (before + during gap, padded to max)
        secondary_context_end = gap_start + config.MAX_GAP_LENGTH
        secondary_context = df_secondary.iloc[context_start:secondary_context_end]
        
        # Target: primary exchange data during gap
        target = df_primary.iloc[gap_start:gap_start + gap_length]
        
        # Extract feature columns - ONLY OHLCV + derived (not temporal)
        feature_cols = []
        for col in config.OHLCV_FEATURES + config.DERIVED_FEATURES:
            if col in df_primary.columns:
                feature_cols.append(col)
        
        # Get values and ensure proper shape
        primary_values = primary_context[feature_cols].values
        secondary_values = secondary_context[feature_cols].values
        target_values = target[config.OHLCV_FEATURES].values
        
        # Ensure primary context has correct length
        if len(primary_values) < config.LOOKBACK_WINDOW:
            padding = np.zeros((config.LOOKBACK_WINDOW - len(primary_values), len(feature_cols)))
            primary_values = np.vstack([padding, primary_values])
        elif len(primary_values) > config.LOOKBACK_WINDOW:
            primary_values = primary_values[-config.LOOKBACK_WINDOW:]
        
        # Ensure secondary context has correct length (FIXED to LOOKBACK + MAX_GAP)
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
            'gap_position': np.arange(gap_length) / gap_length  # Normalized position
        }
        
        return sample
    
    def build_dataset(self, df_primary: pd.DataFrame, 
                     df_secondary: pd.DataFrame,
                     symbol: str,
                     gaps: List[Tuple[int, int]] = None) -> Tuple[np.ndarray, np.ndarray, List]:
        """
        Build complete dataset for a symbol.
        
        Args:
            df_primary: Primary exchange data
            df_secondary: Secondary exchange data
            symbol: Symbol identifier
            gaps: Pre-defined gaps or None for synthetic generation
            
        Returns:
            Tuple of (X, y, metadata)
        """
        if gaps is None:
            gaps = self.create_synthetic_gaps(df_primary)
        
        samples = []
        labels = []
        metadata = []
        
        # Calculate fixed feature size
        num_ohlcv_features = len(config.OHLCV_FEATURES)
        num_derived_features = len(config.DERIVED_FEATURES)
        num_temporal_features = len(config.TEMPORAL_FEATURES)
        
        primary_context_size = config.LOOKBACK_WINDOW * (num_ohlcv_features + num_derived_features + num_temporal_features)
        secondary_context_size = (config.LOOKBACK_WINDOW + config.MAX_GAP_LENGTH) * (num_ohlcv_features + num_derived_features + num_temporal_features)
        meta_features_size = 2
        
        fixed_input_size = primary_context_size + secondary_context_size + meta_features_size
        
        for gap_start, gap_length in gaps:
            try:
                sample = self.create_sample(
                    df_primary, 
                    df_secondary,
                    gap_start,
                    gap_length,
                    symbol
                )
                
                # Pad primary context to fixed size
                primary_padded = self._pad_to_size(
                    sample['primary_context'],
                    (config.LOOKBACK_WINDOW, num_ohlcv_features + num_derived_features + num_temporal_features)
                )
                
                # Pad secondary context to fixed size
                secondary_padded = self._pad_to_size(
                    sample['secondary_context'],
                    (config.LOOKBACK_WINDOW + config.MAX_GAP_LENGTH, num_ohlcv_features + num_derived_features + num_temporal_features)
                )
                
                # Flatten contexts
                primary_flat = primary_padded.flatten()
                secondary_flat = secondary_padded.flatten()
                
                # Add metadata features
                meta_features = np.array([
                    sample['gap_length'],
                    sample['symbol_id']
                ])
                
                # Combine all features with fixed size
                X = np.concatenate([primary_flat, secondary_flat, meta_features])
                
                # Ensure X has correct size
                if len(X) != fixed_input_size:
                    logger.warning(f"Input size mismatch: {len(X)} != {fixed_input_size}")
                    continue
                
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
        
        # Stack samples (should all have same size now)
        X_array = np.array(samples)
        y_padded = self._pad_labels(labels)
        
        logger.info(f"Built {len(samples)} samples for {symbol} with input shape {X_array.shape}")
        return X_array, y_padded, metadata
    
    def _pad_sequences(self, sequences: List[np.ndarray]) -> np.ndarray:
        """Pad sequences to same length."""
        max_len = max(len(seq) for seq in sequences)
        padded = np.zeros((len(sequences), max_len))
        
        for i, seq in enumerate(sequences):
            padded[i, :len(seq)] = seq
        
        return padded
    
    def _pad_to_size(self, array: np.ndarray, target_shape: tuple) -> np.ndarray:
        """
        Pad or truncate array to target shape.
        
        Args:
            array: Input array
            target_shape: Desired shape (timesteps, features)
            
        Returns:
            Padded/truncated array
        """
        current_shape = array.shape
        
        # Handle 1D arrays
        if len(current_shape) == 1:
            if len(array) >= target_shape[0] * target_shape[1]:
                return array[:target_shape[0] * target_shape[1]].reshape(target_shape)
            else:
                padded = np.zeros(target_shape[0] * target_shape[1])
                padded[:len(array)] = array
                return padded.reshape(target_shape)
        
        # Handle 2D arrays
        padded = np.zeros(target_shape)
        
        rows_to_copy = min(current_shape[0], target_shape[0])
        cols_to_copy = min(current_shape[1], target_shape[1])
        
        padded[:rows_to_copy, :cols_to_copy] = array[:rows_to_copy, :cols_to_copy]
        
        return padded
    
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
        """
        Split data temporally (no random shuffle).
        
        Args:
            X: Features array
            y: Labels array
            metadata: List of metadata dicts
            
        Returns:
            Dictionary with train/val/test splits
        """
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
        """
        Build combined dataset from multiple symbols.
        
        Args:
            data_dict: Dictionary {symbol: (df_primary, df_secondary)}
            
        Returns:
            Combined train/val/test splits
        """
        all_X = []
        all_y = []
        all_metadata = []
        
        # First pass: determine the expected input size
        expected_input_size = None
        
        for symbol, (df_primary, df_secondary) in data_dict.items():
            X, y, metadata = self.build_dataset(df_primary, df_secondary, symbol)
            
            if len(X) > 0:
                if expected_input_size is None:
                    expected_input_size = X.shape[1]
                    logger.info(f"Expected input size set to: {expected_input_size}")
                
                # Verify all samples have the same input size
                if X.shape[1] != expected_input_size:
                    logger.warning(f"Skipping {symbol}: input size mismatch {X.shape[1]} != {expected_input_size}")
                    continue
                
                all_X.append(X)
                all_y.append(y)
                all_metadata.extend(metadata)
        
        if not all_X:
            logger.error("No data to build dataset")
            return {}
        
        # Concatenate all symbols
        X_combined = np.vstack(all_X)
        y_combined = np.vstack(all_y)
        
        logger.info(f"Combined dataset shape: X={X_combined.shape}, y={y_combined.shape}")
        
        # Temporal split
        splits = self.temporal_split(X_combined, y_combined, all_metadata)
        
        logger.info(f"Built multi-symbol dataset with {len(X_combined)} total samples")
        return splits
    
    def augment_data(self, X: np.ndarray, y: np.ndarray, 
                    augmentation_factor: int = 2) -> Tuple[np.ndarray, np.ndarray]:
        """
        Augment data by varying gap lengths.
        
        Args:
            X: Features array
            y: Labels array
            augmentation_factor: Number of augmented versions per sample
            
        Returns:
            Augmented (X, y) arrays
        """
        augmented_X = [X]
        augmented_y = [y]
        
        for _ in range(augmentation_factor - 1):
            # Add small noise to features
            noise = np.random.normal(0, 0.01, X.shape)
            augmented_X.append(X + noise)
            augmented_y.append(y)
        
        X_aug = np.vstack(augmented_X)
        y_aug = np.vstack(augmented_y)
        
        logger.info(f"Augmented dataset from {len(X)} to {len(X_aug)} samples")
        return X_aug, y_aug


if __name__ == '__main__':
    builder = DatasetBuilder()
    logger.info("Dataset builder module ready")