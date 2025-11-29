"""
Model training module - FIXED VERSION
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from typing import Dict, Tuple
import os
import json
import config
from model import create_and_compile_model, get_callbacks

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT
)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Handles model training and evaluation."""
    
    def __init__(self, model=None):
        self.model = model or create_and_compile_model()
        self.history = None
        self.evaluation_results = {}
    
    def prepare_inputs(self, X: np.ndarray, metadata: list) -> Dict[str, np.ndarray]:
        """
        Prepare input dictionary for model training - FIXED VERSION.
        
        Args:
            X: Feature array (flattened)
            metadata: List of metadata dictionaries
            
        Returns:
            Dictionary of model inputs
        """
        batch_size = len(X)
        
        # Calculate feature dimensions
        num_ohlcv = len(config.OHLCV_FEATURES)
        num_derived = len(config.DERIVED_FEATURES)
        num_temporal = len(config.TEMPORAL_FEATURES)
        num_features = num_ohlcv + num_derived + num_temporal
        
        # Calculate sizes
        primary_size = config.LOOKBACK_WINDOW * num_features
        secondary_size = (config.LOOKBACK_WINDOW + config.MAX_GAP_LENGTH) * num_features
        
        # Initialize arrays
        primary_context = np.zeros((batch_size, config.LOOKBACK_WINDOW, num_features))
        secondary_context = np.zeros((batch_size, config.LOOKBACK_WINDOW + config.MAX_GAP_LENGTH, num_features))
        
        # FIXED: Extract data from flattened X
        for i in range(batch_size):
            # Extract primary context
            primary_flat = X[i, :primary_size]
            primary_context[i] = primary_flat.reshape(config.LOOKBACK_WINDOW, num_features)
            
            # Extract secondary context
            secondary_flat = X[i, primary_size:primary_size + secondary_size]
            secondary_context[i] = secondary_flat.reshape(config.LOOKBACK_WINDOW + config.MAX_GAP_LENGTH, num_features)
        
        # Extract metadata
        symbol_ids = np.array([meta.get('symbol_id', 0) for meta in metadata])
        gap_lengths = np.array([meta.get('gap_length', 1) for meta in metadata])
        
        return {
            'primary_context': primary_context,
            'secondary_context': secondary_context,
            'symbol_id': symbol_ids.reshape(-1, 1),
            'gap_length': gap_lengths.reshape(-1, 1)
        }
    
    def train(self, train_data: Dict, val_data: Dict, 
             epochs: int = None, batch_size: int = None) -> Dict:
        """Train the model."""
        
        epochs = epochs or config.EPOCHS
        batch_size = batch_size or config.BATCH_SIZE
        
        logger.info(f"Starting training: epochs={epochs}, batch_size={batch_size}")
        
        # Prepare inputs
        X_train = self.prepare_inputs(train_data['X'], train_data['metadata'])
        y_train = train_data['y']
        
        X_val = self.prepare_inputs(val_data['X'], val_data['metadata'])
        y_val = val_data['y']
        
        logger.info(f"Training data shapes:")
        logger.info(f"  Primary context: {X_train['primary_context'].shape}")
        logger.info(f"  Secondary context: {X_train['secondary_context'].shape}")
        logger.info(f"  Targets: {y_train.shape}")
        
        # Get callbacks
        callbacks = get_callbacks()
        
        # Train model
        history = self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        self.history = history
        logger.info("Training completed")
        
        return history.history
    
    def evaluate(self, test_data: Dict) -> Dict:
        """Evaluate model on test set."""
        
        logger.info("Evaluating model on test set")
        
        X_test = self.prepare_inputs(test_data['X'], test_data['metadata'])
        y_test = test_data['y']
        
        # Evaluate
        results = self.model.evaluate(X_test, y_test, verbose=1)
        
        # Store results
        self.evaluation_results = {
            'loss': results[0],
            'mae': results[1],
            'mape': results[2]
        }
        
        logger.info(f"Test results: {self.evaluation_results}")
        return self.evaluation_results
    
    def predict(self, X: np.ndarray, metadata: list) -> np.ndarray:
        """Make predictions."""
        X_input = self.prepare_inputs(X, metadata)
        predictions = self.model.predict(X_input)
        return predictions
    
    def plot_training_history(self, save_path: str = None):
        """Plot training history."""
        
        if self.history is None:
            logger.warning("No training history available")
            return
        
        history = self.history.history
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training History', fontsize=16)
        
        # Loss
        axes[0, 0].plot(history['loss'], label='Train Loss')
        axes[0, 0].plot(history['val_loss'], label='Val Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # MAE
        axes[0, 1].plot(history['mae_metric'], label='Train MAE')
        axes[0, 1].plot(history['val_mae_metric'], label='Val MAE')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].set_title('Mean Absolute Error')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # MAPE
        axes[1, 0].plot(history['mape_metric'], label='Train MAPE')
        axes[1, 0].plot(history['val_mape_metric'], label='Val MAPE')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('MAPE (%)')
        axes[1, 0].set_title('Mean Absolute Percentage Error')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Learning rate
        if 'lr' in history:
            axes[1, 1].plot(history['lr'])
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_title('Learning Rate Schedule')
            axes[1, 1].set_yscale('log')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved training history plot to {save_path}")
        
        plt.show()
    
    def save_model(self, filepath: str = None):
        """Save trained model."""
        filepath = filepath or config.MODEL_PATH
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str = None):
        """Load trained model."""
        from tensorflow import keras
        from model import AttentionLayer, ohlc_constraint_loss, mae_metric, mape_metric
        
        filepath = filepath or config.MODEL_PATH
        self.model = keras.models.load_model(
            filepath,
            custom_objects={
                'AttentionLayer': AttentionLayer,
                'ohlc_constraint_loss': ohlc_constraint_loss,
                'mae_metric': mae_metric,
                'mape_metric': mape_metric
            }
        )
        logger.info(f"Model loaded from {filepath}")
    
    def generate_training_report(self, save_path: str = None) -> Dict:
        """Generate comprehensive training report."""
        
        report = {
            'model_config': {
                'lstm_units': config.LSTM_UNITS,
                'attention_units': config.ATTENTION_UNITS,
                'dropout_rate': config.DROPOUT_RATE,
                'learning_rate': config.LEARNING_RATE,
                'batch_size': config.BATCH_SIZE,
                'epochs_trained': len(self.history.history['loss']) if self.history else 0
            },
            'training_results': {
                'final_train_loss': self.history.history['loss'][-1] if self.history else None,
                'final_val_loss': self.history.history['val_loss'][-1] if self.history else None,
                'best_val_loss': min(self.history.history['val_loss']) if self.history else None
            },
            'evaluation_results': self.evaluation_results
        }
        
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Training report saved to {save_path}")
        
        return report