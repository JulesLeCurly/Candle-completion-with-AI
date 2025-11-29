"""
Model training module for cryptocurrency data completion.
Handles training pipeline, hyperparameter tuning, and visualization.
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
        """
        Initialize trainer.
        
        Args:
            model: Pre-initialized model (None = create new)
        """
        self.model = model or create_and_compile_model()
        self.history = None
        self.evaluation_results = {}
    
    def prepare_inputs(self, X: np.ndarray, metadata: list) -> Dict[str, np.ndarray]:
        """
        Prepare input dictionary for model training.
        
        Args:
            X: Feature array
            metadata: List of metadata dictionaries
            
        Returns:
            Dictionary of model inputs
        """
        # Extract components from flattened X
        # This is a simplified version - actual implementation needs proper reshaping
        batch_size = len(X)
        
        primary_context = np.zeros((batch_size, config.LOOKBACK_WINDOW, 
                                   len(config.OHLCV_FEATURES) + len(config.DERIVED_FEATURES)))
        secondary_context = np.zeros((batch_size, config.LOOKBACK_WINDOW + config.MAX_GAP_LENGTH,
                                     len(config.OHLCV_FEATURES) + len(config.DERIVED_FEATURES)))
        
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
        """
        Train the model.
        
        Args:
            train_data: Dictionary with 'X', 'y', 'metadata' keys
            val_data: Validation data dictionary
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Training history
        """
        epochs = epochs or config.EPOCHS
        batch_size = batch_size or config.BATCH_SIZE
        
        logger.info(f"Starting training: epochs={epochs}, batch_size={batch_size}")
        
        # Prepare inputs
        X_train = self.prepare_inputs(train_data['X'], train_data['metadata'])
        y_train = train_data['y']
        
        X_val = self.prepare_inputs(val_data['X'], val_data['metadata'])
        y_val = val_data['y']
        
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
        """
        Evaluate model on test set.
        
        Args:
            test_data: Dictionary with 'X', 'y', 'metadata' keys
            
        Returns:
            Dictionary of evaluation metrics
        """
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
        """
        Make predictions.
        
        Args:
            X: Feature array
            metadata: List of metadata dictionaries
            
        Returns:
            Predicted OHLCV values
        """
        X_input = self.prepare_inputs(X, metadata)
        predictions = self.model.predict(X_input)
        return predictions
    
    def plot_training_history(self, save_path: str = None):
        """
        Plot training history.
        
        Args:
            save_path: Path to save plot (None = display only)
        """
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
        
        # Learning rate (if available)
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
    
    def plot_predictions(self, predictions: np.ndarray, 
                        ground_truth: np.ndarray,
                        num_samples: int = 5,
                        save_path: str = None):
        """
        Plot sample predictions vs ground truth.
        
        Args:
            predictions: Predicted OHLCV values
            ground_truth: True OHLCV values
            num_samples: Number of samples to plot
            save_path: Path to save plot
        """
        num_samples = min(num_samples, len(predictions))
        
        fig, axes = plt.subplots(num_samples, 1, figsize=(15, 4 * num_samples))
        if num_samples == 1:
            axes = [axes]
        
        for i in range(num_samples):
            pred = predictions[i]
            true = ground_truth[i]
            
            # Plot OHLC
            timesteps = range(len(pred))
            
            axes[i].plot(timesteps, true[:, 3], 'b-', label='True Close', linewidth=2)
            axes[i].plot(timesteps, pred[:, 3], 'r--', label='Pred Close', linewidth=2)
            
            axes[i].fill_between(timesteps, true[:, 2], true[:, 1], 
                                alpha=0.3, color='blue', label='True Range')
            axes[i].fill_between(timesteps, pred[:, 2], pred[:, 1], 
                                alpha=0.3, color='red', label='Pred Range')
            
            axes[i].set_xlabel('Timestep')
            axes[i].set_ylabel('Price')
            axes[i].set_title(f'Sample {i+1}: Prediction vs Ground Truth')
            axes[i].legend()
            axes[i].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved prediction plot to {save_path}")
        
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
        """
        Generate comprehensive training report.
        
        Args:
            save_path: Path to save report JSON
            
        Returns:
            Report dictionary
        """
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


if __name__ == '__main__':
    trainer = ModelTrainer()
    logger.info("Trainer module ready")