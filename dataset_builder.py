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
        
        # Context from secondary exchange (before + during gap)
        secondary_context = df_secondary.iloc[context_start:gap_start + gap_length]
        
        # Target: primary exchange data during gap
        target = df_primary.iloc[gap_start:gap_start + gap_length]
        
        # Extract feature columns
        feature_cols = config.OHLCV_FEATURES + config.DERIVED_FEATURES + config.TEMPORAL_FEATURES
        
        sample = {
            'primary_context': primary_context[feature_cols].values,
            'secondary_context': secondary_context[feature_cols].values,
            'target': target[config.OHLCV_FEATURES].values,
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
        
        for gap_start, gap_length in gaps:
            try:
                sample = self.create_sample(
                    df_primary, 
                    df_secondary,
                    gap_start,
                    gap_length,
                    symbol
                )
                
                # Flatten and concatenate inputs
                primary_flat = sample['primary_context'].flatten()
                secondary_flat = sample['secondary_context'].flatten()
                
                # Add metadata features
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
                    'symbol': symbol
                })
                
            except Exception as e:
                logger.warning(f"Error creating sample at {gap_start}: {e}")
                continue
        
        if not samples:
            logger.warning(f"No samples created for {symbol}")
            return np.array([]), np.array([]), []
        
        # Pad sequences to max gap length
        X_padded = self._pad_sequences(samples)
        y_padded = self._pad_labels(labels)
        
        logger.info(f"Built {len(samples)} samples for {symbol}")
        return X_padded, y_padded, metadata
    
    def _pad_sequences(self, sequences: List[np.ndarray]) -> np.ndarray:
        """Pad sequences to same length."""
        max_len = max(len(seq) for seq in sequences)
        padded = np.zeros((len(sequences), max_len))
        
        for i, seq in enumerate(sequences):
            padded[i, :len(seq)] = seq
        
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
        
        for symbol, (df_primary, df_secondary) in data_dict.items():
            X, y, metadata = self.build_dataset(df_primary, df_secondary, symbol)
            
            if len(X) > 0:
                all_X.append(X)
                all_y.append(y)
                all_metadata.extend(metadata)
        
        if not all_X:
            logger.error("No data to build dataset")
            return {}
        
        # Concatenate all symbols
        X_combined = np.vstack(all_X)
        y_combined = np.vstack(all_y)
        
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
