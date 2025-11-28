"""
Deep learning model architecture for cryptocurrency data completion.
Implements LSTM with attention mechanism and custom loss function.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import numpy as np
import logging
import config

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT
)
logger = logging.getLogger(__name__)


class AttentionLayer(layers.Layer):
    """Custom attention mechanism layer."""
    
    def __init__(self, units):
        super(AttentionLayer, self).__init__()
        self.units = units
        self.W = layers.Dense(units)
        self.V = layers.Dense(1)
    
    def call(self, inputs):
        """
        Apply attention mechanism.
        
        Args:
            inputs: LSTM outputs (batch_size, timesteps, features)
            
        Returns:
            Context vector with attention weights applied
        """
        score = tf.nn.tanh(self.W(inputs))
        attention_weights = tf.nn.softmax(self.V(score), axis=1)
        context_vector = attention_weights * inputs
        context_vector = tf.reduce_sum(context_vector, axis=1)
        
        return context_vector


def create_model(input_shape: tuple, output_shape: tuple, 
                num_symbols: int) -> Model:
    """
    Create LSTM model with attention mechanism.
    
    Args:
        input_shape: Shape of input features
        output_shape: Shape of output (max_gap_length, num_ohlcv_features)
        num_symbols: Number of unique cryptocurrency symbols
        
    Returns:
        Compiled Keras model
    """
    # Input layers
    primary_context = layers.Input(
        shape=(config.LOOKBACK_WINDOW, len(config.OHLCV_FEATURES) + len(config.DERIVED_FEATURES)),
        name='primary_context'
    )
    secondary_context = layers.Input(
        shape=(config.LOOKBACK_WINDOW + config.MAX_GAP_LENGTH, 
               len(config.OHLCV_FEATURES) + len(config.DERIVED_FEATURES)),
        name='secondary_context'
    )
    symbol_input = layers.Input(shape=(1,), name='symbol_id')
    gap_length_input = layers.Input(shape=(1,), name='gap_length')
    
    # Symbol embedding
    symbol_embedding = layers.Embedding(
        input_dim=num_symbols,
        output_dim=config.EMBEDDING_DIM,
        name='symbol_embedding'
    )(symbol_input)
    symbol_embedding = layers.Flatten()(symbol_embedding)
    
    # LSTM for primary context
    primary_lstm = layers.LSTM(
        config.LSTM_UNITS,
        return_sequences=True,
        dropout=config.DROPOUT_RATE,
        name='primary_lstm'
    )(primary_context)
    
    # LSTM for secondary context
    secondary_lstm = layers.LSTM(
        config.LSTM_UNITS,
        return_sequences=True,
        dropout=config.DROPOUT_RATE,
        name='secondary_lstm'
    )(secondary_context)
    
    # Attention mechanism
    primary_attention = AttentionLayer(config.ATTENTION_UNITS)(primary_lstm)
    secondary_attention = AttentionLayer(config.ATTENTION_UNITS)(secondary_lstm)
    
    # Concatenate all features
    combined = layers.Concatenate()([
        primary_attention,
        secondary_attention,
        symbol_embedding,
        gap_length_input
    ])
    
    # Dense layers
    dense1 = layers.Dense(256, activation='relu')(combined)
    dense1 = layers.Dropout(config.DROPOUT_RATE)(dense1)
    
    dense2 = layers.Dense(128, activation='relu')(dense1)
    dense2 = layers.Dropout(config.DROPOUT_RATE)(dense2)
    
    # Output layer: predict all OHLCV values for max gap length
    output = layers.Dense(
        output_shape[0] * output_shape[1],
        activation='linear',
        name='ohlcv_output'
    )(dense2)
    
    # Reshape to (max_gap_length, num_features)
    output = layers.Reshape(output_shape)(output)
    
    # Create model
    model = Model(
        inputs=[primary_context, secondary_context, symbol_input, gap_length_input],
        outputs=output
    )
    
    logger.info("Model architecture created")
    return model


def ohlc_constraint_loss(y_true, y_pred):
    """
    Custom loss function with OHLC constraints.
    Penalizes predictions where high < close, low > close, etc.
    
    Args:
        y_true: True OHLCV values (batch_size, max_gap_length, 5)
        y_pred: Predicted OHLCV values
        
    Returns:
        Loss value
    """
    # Extract OHLCV components
    open_true = y_true[:, :, 0]
    high_true = y_true[:, :, 1]
    low_true = y_true[:, :, 2]
    close_true = y_true[:, :, 3]
    volume_true = y_true[:, :, 4]
    
    open_pred = y_pred[:, :, 0]
    high_pred = y_pred[:, :, 1]
    low_pred = y_pred[:, :, 2]
    close_pred = y_pred[:, :, 3]
    volume_pred = y_pred[:, :, 4]
    
    # Weighted MAE for each component
    mae_open = tf.reduce_mean(tf.abs(open_true - open_pred)) * config.LOSS_WEIGHTS['open']
    mae_high = tf.reduce_mean(tf.abs(high_true - high_pred)) * config.LOSS_WEIGHTS['high']
    mae_low = tf.reduce_mean(tf.abs(low_true - low_pred)) * config.LOSS_WEIGHTS['low']
    mae_close = tf.reduce_mean(tf.abs(close_true - close_pred)) * config.LOSS_WEIGHTS['close']
    mae_volume = tf.reduce_mean(tf.abs(volume_true - volume_pred)) * config.LOSS_WEIGHTS['volume']
    
    base_loss = mae_open + mae_high + mae_low + mae_close + mae_volume
    
    # OHLC constraint violations
    # high should be >= max(open, close)
    high_violation = tf.reduce_mean(
        tf.maximum(0.0, tf.maximum(open_pred, close_pred) - high_pred)
    )
    
    # low should be <= min(open, close)
    low_violation = tf.reduce_mean(
        tf.maximum(0.0, low_pred - tf.minimum(open_pred, close_pred))
    )
    
    # Volume should be positive
    volume_violation = tf.reduce_mean(
        tf.maximum(0.0, -volume_pred)
    )
    
    constraint_penalty = (high_violation + low_violation + volume_violation) * config.VIOLATION_PENALTY
    
    total_loss = base_loss + constraint_penalty
    
    return total_loss


def mae_metric(y_true, y_pred):
    """Mean Absolute Error metric."""
    return tf.reduce_mean(tf.abs(y_true - y_pred))


def mape_metric(y_true, y_pred):
    """Mean Absolute Percentage Error metric."""
    epsilon = 1e-7
    return tf.reduce_mean(tf.abs((y_true - y_pred) / (y_true + epsilon))) * 100


def compile_model(model: Model, learning_rate: float = None) -> Model:
    """
    Compile model with optimizer, loss, and metrics.
    
    Args:
        model: Keras model to compile
        learning_rate: Learning rate for optimizer
        
    Returns:
        Compiled model
    """
    learning_rate = learning_rate or config.LEARNING_RATE
    
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss=ohlc_constraint_loss,
        metrics=[mae_metric, mape_metric]
    )
    
    logger.info(f"Model compiled with learning_rate={learning_rate}")
    return model


def get_callbacks(model_path: str = None) -> list:
    """
    Create training callbacks.
    
    Args:
        model_path: Path to save best model
        
    Returns:
        List of Keras callbacks
    """
    model_path = model_path or config.MODEL_PATH
    
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=config.PATIENCE,
            min_delta=config.MIN_DELTA,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor='val_loss',
            save_best_only=config.SAVE_BEST_ONLY,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        keras.callbacks.TensorBoard(
            log_dir=config.LOG_DIR,
            histogram_freq=1
        )
    ]
    
    logger.info("Training callbacks created")
    return callbacks


def create_and_compile_model(num_symbols: int = None) -> Model:
    """
    Create and compile complete model.
    
    Args:
        num_symbols: Number of unique symbols
        
    Returns:
        Compiled Keras model
    """
    num_symbols = num_symbols or len(config.SYMBOLS)
    
    input_shape = (config.LOOKBACK_WINDOW, 
                  len(config.OHLCV_FEATURES) + len(config.DERIVED_FEATURES))
    output_shape = (config.MAX_GAP_LENGTH, len(config.OHLCV_FEATURES))
    
    model = create_model(input_shape, output_shape, num_symbols)
    model = compile_model(model)
    
    logger.info("Model creation and compilation complete")
    model.summary(print_fn=logger.info)
    
    return model


if __name__ == '__main__':
    model = create_and_compile_model()
    logger.info("Model module ready")
