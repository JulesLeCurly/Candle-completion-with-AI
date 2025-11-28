"""
Feature engineering module for cryptocurrency data.
Computes derived features, ratios, temporal encodings, and normalization.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Tuple
import pickle
import config

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT
)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Handles feature computation and normalization for cryptocurrency data."""
    
    def __init__(self):
        """Initialize feature engineer with normalization parameters."""
        self.normalization_params = {}
    
    def compute_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute derived OHLCV features.
        
        Args:
            df: DataFrame with OHLCV columns
            
        Returns:
            DataFrame with additional derived features
        """
        df = df.copy()
        
        # Body (difference between close and open)
        df['body'] = df['close'] - df['open']
        
        # Upper wick (distance from high to max(open, close))
        df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
        
        # Lower wick (distance from min(open, close) to low)
        df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
        
        # Range (high - low)
        df['range'] = df['high'] - df['low']
        
        # Avoid division by zero
        df['range'] = df['range'].replace(0, np.nan)
        
        logger.debug("Computed derived features: body, wicks, range")
        return df
    
    def compute_ratio_features(self, df_primary: pd.DataFrame, 
                               df_secondary: pd.DataFrame) -> pd.DataFrame:
        """
        Compute ratio features between two exchange datasets.
        
        Args:
            df_primary: DataFrame from primary exchange
            df_secondary: DataFrame from secondary exchange
            
        Returns:
            DataFrame with ratio features
        """
        df_merged = pd.merge(
            df_primary,
            df_secondary,
            on='open_time',
            suffixes=('_primary', '_secondary')
        )
        
        # Price ratio (primary close / secondary close)
        df_merged['price_ratio'] = df_merged['close_primary'] / df_merged['close_secondary']
        
        # Volume ratio
        df_merged['volume_ratio'] = df_merged['volume_primary'] / df_merged['volume_secondary']
        
        # Spread (percentage difference)
        df_merged['spread'] = (
            (df_merged['close_primary'] - df_merged['close_secondary']) / 
            df_merged['close_secondary']
        )
        
        logger.debug("Computed ratio features between exchanges")
        return df_merged
    
    def compute_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute cyclical temporal features.
        
        Args:
            df: DataFrame with 'open_time' column
            
        Returns:
            DataFrame with temporal encodings
        """
        df = df.copy()
        
        # Extract hour and day of week
        df['hour'] = df['open_time'].dt.hour
        df['day_of_week'] = df['open_time'].dt.dayofweek
        
        # Cyclical encoding for hour (0-23)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # Cyclical encoding for day of week (0-6)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        logger.debug("Computed temporal features with cyclical encoding")
        return df
    
    def fit_normalization(self, df: pd.DataFrame, symbol: str):
        """
        Fit normalization parameters for a symbol.
        
        Args:
            df: DataFrame with features to normalize
            symbol: Symbol identifier
        """
        features_to_normalize = (
            config.OHLCV_FEATURES + 
            config.DERIVED_FEATURES + 
            config.RATIO_FEATURES
        )
        
        params = {}
        
        for feature in features_to_normalize:
            if feature in df.columns:
                if config.NORMALIZATION_METHOD == 'minmax':
                    params[feature] = {
                        'min': df[feature].min(),
                        'max': df[feature].max()
                    }
                elif config.NORMALIZATION_METHOD == 'standard':
                    params[feature] = {
                        'mean': df[feature].mean(),
                        'std': df[feature].std()
                    }
        
        self.normalization_params[symbol] = params
        logger.info(f"Fitted normalization parameters for {symbol}")
    
    def normalize(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Normalize features using pre-fitted parameters.
        
        Args:
            df: DataFrame to normalize
            symbol: Symbol identifier
            
        Returns:
            Normalized DataFrame
        """
        df = df.copy()
        
        if symbol not in self.normalization_params:
            logger.warning(f"No normalization parameters for {symbol}")
            return df
        
        params = self.normalization_params[symbol]
        
        for feature, param in params.items():
            if feature in df.columns:
                if config.NORMALIZATION_METHOD == 'minmax':
                    min_val = param['min']
                    max_val = param['max']
                    if max_val > min_val:
                        df[feature] = (df[feature] - min_val) / (max_val - min_val)
                elif config.NORMALIZATION_METHOD == 'standard':
                    mean_val = param['mean']
                    std_val = param['std']
                    if std_val > 0:
                        df[feature] = (df[feature] - mean_val) / std_val
        
        logger.debug(f"Normalized features for {symbol}")
        return df
    
    def denormalize(self, df: pd.DataFrame, symbol: str, 
                    features: list = None) -> pd.DataFrame:
        """
        Denormalize features back to original scale.
        
        Args:
            df: Normalized DataFrame
            symbol: Symbol identifier
            features: List of features to denormalize (None = all)
            
        Returns:
            Denormalized DataFrame
        """
        df = df.copy()
        
        if symbol not in self.normalization_params:
            logger.warning(f"No normalization parameters for {symbol}")
            return df
        
        params = self.normalization_params[symbol]
        features_to_denorm = features or params.keys()
        
        for feature in features_to_denorm:
            if feature in df.columns and feature in params:
                param = params[feature]
                if config.NORMALIZATION_METHOD == 'minmax':
                    min_val = param['min']
                    max_val = param['max']
                    df[feature] = df[feature] * (max_val - min_val) + min_val
                elif config.NORMALIZATION_METHOD == 'standard':
                    mean_val = param['mean']
                    std_val = param['std']
                    df[feature] = df[feature] * std_val + mean_val
        
        logger.debug(f"Denormalized features for {symbol}")
        return df
    
    def prepare_sequences(self, df: pd.DataFrame, lookback: int = None) -> np.ndarray:
        """
        Prepare sequences for LSTM input.
        
        Args:
            df: DataFrame with features
            lookback: Number of timesteps to look back
            
        Returns:
            3D numpy array (samples, timesteps, features)
        """
        lookback = lookback or config.LOOKBACK_WINDOW
        
        feature_cols = (
            config.OHLCV_FEATURES + 
            config.DERIVED_FEATURES + 
            config.TEMPORAL_FEATURES
        )
        
        available_cols = [col for col in feature_cols if col in df.columns]
        
        if len(df) < lookback:
            logger.warning(f"Not enough data for sequences: {len(df)} < {lookback}")
            return np.array([])
        
        data = df[available_cols].values
        sequences = []
        
        for i in range(len(data) - lookback + 1):
            sequences.append(data[i:i + lookback])
        
        return np.array(sequences)
    
    def save_normalization_params(self, filepath: str):
        """Save normalization parameters to file."""
        with open(filepath, 'wb') as f:
            pickle.dump(self.normalization_params, f)
        logger.info(f"Saved normalization parameters to {filepath}")
    
    def load_normalization_params(self, filepath: str):
        """Load normalization parameters from file."""
        with open(filepath, 'rb') as f:
            self.normalization_params = pickle.load(f)
        logger.info(f"Loaded normalization parameters from {filepath}")
    
    def process_symbol(self, df_primary: pd.DataFrame, 
                      df_secondary: pd.DataFrame, 
                      symbol: str, 
                      fit: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Complete feature engineering pipeline for a symbol.
        
        Args:
            df_primary: Primary exchange data
            df_secondary: Secondary exchange data
            symbol: Symbol identifier
            fit: Whether to fit normalization parameters
            
        Returns:
            Tuple of processed DataFrames
        """
        logger.info(f"Processing features for {symbol}")
        
        # Compute derived features
        df_primary = self.compute_derived_features(df_primary)
        df_secondary = self.compute_derived_features(df_secondary)
        
        # Compute temporal features
        df_primary = self.compute_temporal_features(df_primary)
        df_secondary = self.compute_temporal_features(df_secondary)
        
        # Fit normalization if needed
        if fit:
            self.fit_normalization(df_primary, symbol)
        
        # Normalize
        df_primary = self.normalize(df_primary, symbol)
        df_secondary = self.normalize(df_secondary, symbol)
        
        logger.info(f"Feature engineering completed for {symbol}")
        return df_primary, df_secondary


if __name__ == '__main__':
    engineer = FeatureEngineer()
    logger.info("Feature engineering module ready")
