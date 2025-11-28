"""
Configuration file for the cryptocurrency data completion system.
Centralizes all project settings, hyperparameters, and constants.
"""

import os
from datetime import datetime

# Project paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'DataBase')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'RAW_Data_1h')
COMPLETED_DATA_DIR = os.path.join(DATA_DIR, 'Completed_Data_1h')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
LOG_DIR = os.path.join(BASE_DIR, 'logs')

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, COMPLETED_DATA_DIR, MODEL_DIR, LOG_DIR]:
    os.makedirs(directory, exist_ok=True)

# Cryptocurrency symbols
SYMBOLS = [
    'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'XRP/USDT', 'ADA/USDT',
    'DOGE/USDT', 'SOL/USDT', 'DOT/USDT', 'MATIC/USDT', 'LTC/USDT',
    'AVAX/USDT', 'LINK/USDT', 'UNI/USDT', 'ATOM/USDT', 'XLM/USDT',
    'ALGO/USDT', 'VET/USDT', 'FIL/USDT', 'TRX/USDT', 'ETC/USDT'
]

# Data collection parameters
START_DATE = '2017-01-01'
END_DATE = datetime.now().strftime('%Y-%m-%d')
TIMEFRAME = '1h'

# Exchange configuration
PRIMARY_EXCHANGE = 'binance'
SECONDARY_EXCHANGE = 'kraken'

# API rate limits (milliseconds between requests)
RATE_LIMIT_DELAY = 1000
MAX_RETRIES = 3
RETRY_DELAY = 5000

# Model architecture parameters
LOOKBACK_WINDOW = 72  # hours of context before gap
MAX_GAP_LENGTH = 24  # maximum consecutive candles to predict
LSTM_UNITS = 128
ATTENTION_UNITS = 64
DROPOUT_RATE = 0.3
EMBEDDING_DIM = 16

# Feature engineering
OHLCV_FEATURES = ['open', 'high', 'low', 'close', 'volume']
DERIVED_FEATURES = ['body', 'upper_wick', 'lower_wick', 'range']
RATIO_FEATURES = ['price_ratio', 'volume_ratio', 'spread']
TEMPORAL_FEATURES = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos']

# Training parameters
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.15
TRAIN_SPLIT = 0.70

# Early stopping and checkpoints
PATIENCE = 15
MIN_DELTA = 0.0001
SAVE_BEST_ONLY = True

# Loss function weights
LOSS_WEIGHTS = {
    'open': 1.0,
    'high': 1.2,
    'low': 1.2,
    'close': 1.5,
    'volume': 0.8
}
VIOLATION_PENALTY = 10.0  # penalty for OHLC constraint violations

# Post-processing thresholds
MAX_PRICE_DEVIATION = 0.05  # 5% max deviation from context
MIN_VOLUME_RATIO = 0.1
MAX_VOLUME_RATIO = 10.0
CONFIDENCE_THRESHOLD = 0.7

# Logging configuration
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_FILE = os.path.join(LOG_DIR, 'crypto_completion.log')

# CSV output columns
OUTPUT_COLUMNS = [
    'open_time', 'open', 'high', 'low', 'close', 'volume',
    'close_time', 'number_of_trades', 'is_predicted',
    'prediction_confidence', 'source_exchange', 'gap_length'
]

# Normalization parameters (will be computed per symbol)
NORMALIZATION_METHOD = 'minmax'  # 'minmax' or 'standard'

# Synthetic dataset parameters
SYNTHETIC_GAP_RATIO = 0.15  # 15% of data will be masked for training
MIN_SYNTHETIC_GAP = 1  # minimum gap length in hours
MAX_SYNTHETIC_GAP = 24  # maximum gap length in hours

# Model checkpoint
MODEL_NAME = 'crypto_completion_model.keras'
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)

# Evaluation metrics
METRICS = ['mae', 'mape']
