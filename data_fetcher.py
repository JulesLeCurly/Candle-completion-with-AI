"""
Data fetcher module for retrieving cryptocurrency OHLCV data from exchanges.
Handles API calls, rate limiting, gap detection, and data alignment.
"""

import ccxt
import pandas as pd
import numpy as np
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import config

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT,
    handlers=[
        logging.FileHandler(config.LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DataFetcher:
    """Fetches and aligns cryptocurrency data from multiple exchanges."""
    
    def __init__(self):
        """Initialize exchange connections."""
        self.primary_exchange = getattr(ccxt, config.PRIMARY_EXCHANGE)({
            'enableRateLimit': True,
            'rateLimit': config.RATE_LIMIT_DELAY
        })
        self.secondary_exchange = getattr(ccxt, config.SECONDARY_EXCHANGE)({
            'enableRateLimit': True,
            'rateLimit': config.RATE_LIMIT_DELAY
        })
        logger.info(f"Initialized exchanges: {config.PRIMARY_EXCHANGE} and {config.SECONDARY_EXCHANGE}")
    
    def fetch_ohlcv(self, exchange: ccxt.Exchange, symbol: str, 
                    start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch OHLCV data from exchange with retry logic.
        
        Args:
            exchange: CCXT exchange instance
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            
        Returns:
            DataFrame with OHLCV data
        """
        logger.info(f"Fetching {symbol} from {exchange.id} ({start_date} to {end_date})")
        
        since = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
        end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)
        
        all_candles = []
        retries = 0
        
        while since < end_ts:
            try:
                candles = exchange.fetch_ohlcv(
                    symbol,
                    timeframe=config.TIMEFRAME,
                    since=since,
                    limit=1000
                )
                
                if not candles:
                    break
                
                all_candles.extend(candles)
                since = candles[-1][0] + 3600000  # Move to next hour
                time.sleep(config.RATE_LIMIT_DELAY / 1000)
                retries = 0
                
            except Exception as e:
                retries += 1
                if retries > config.MAX_RETRIES:
                    logger.error(f"Max retries exceeded for {symbol} on {exchange.id}: {e}")
                    break
                logger.warning(f"Retry {retries}/{config.MAX_RETRIES} after error: {e}")
                time.sleep(config.RETRY_DELAY / 1000)
        
        if not all_candles:
            logger.warning(f"No data retrieved for {symbol} from {exchange.id}")
            return pd.DataFrame()
        
        df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['open_time'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['close_time'] = df['open_time'] + timedelta(hours=1)
        df['source_exchange'] = exchange.id
        df['symbol'] = symbol
        
        logger.info(f"Retrieved {len(df)} candles for {symbol} from {exchange.id}")
        return df
    
    def detect_gaps(self, df: pd.DataFrame) -> List[Tuple[datetime, datetime, int]]:
        """
        Detect gaps in time series data.
        
        Args:
            df: DataFrame with 'open_time' column
            
        Returns:
            List of tuples (gap_start, gap_end, gap_length)
        """
        if df.empty:
            return []
        
        df = df.sort_values('open_time').reset_index(drop=True)
        gaps = []
        
        for i in range(len(df) - 1):
            current_time = df.loc[i, 'open_time']
            next_time = df.loc[i + 1, 'open_time']
            expected_next = current_time + timedelta(hours=1)
            
            if next_time > expected_next:
                gap_length = int((next_time - expected_next).total_seconds() / 3600)
                if gap_length <= config.MAX_GAP_LENGTH:
                    gaps.append((expected_next, next_time, gap_length))
        
        logger.info(f"Detected {len(gaps)} gaps in data")
        return gaps
    
    def align_timestamps(self, df_primary: pd.DataFrame, 
                        df_secondary: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Align timestamps between two DataFrames from different exchanges.
        
        Args:
            df_primary: DataFrame from primary exchange
            df_secondary: DataFrame from secondary exchange
            
        Returns:
            Tuple of aligned DataFrames
        """
        if df_primary.empty or df_secondary.empty:
            return df_primary, df_secondary
        
        common_times = set(df_primary['open_time']).intersection(set(df_secondary['open_time']))
        
        df_primary_aligned = df_primary[df_primary['open_time'].isin(common_times)].copy()
        df_secondary_aligned = df_secondary[df_secondary['open_time'].isin(common_times)].copy()
        
        df_primary_aligned = df_primary_aligned.sort_values('open_time').reset_index(drop=True)
        df_secondary_aligned = df_secondary_aligned.sort_values('open_time').reset_index(drop=True)
        
        logger.info(f"Aligned to {len(common_times)} common timestamps")
        return df_primary_aligned, df_secondary_aligned
    
    def fetch_all_symbols(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Fetch data for all configured symbols from both exchanges.
        
        Returns:
            Dictionary with structure: {symbol: {'primary': df, 'secondary': df, 'gaps': list}}
        """
        all_data = {}
        
        for symbol in config.SYMBOLS:
            logger.info(f"Processing {symbol}")
            
            try:
                df_primary = self.fetch_ohlcv(
                    self.primary_exchange,
                    symbol,
                    config.START_DATE,
                    config.END_DATE
                )
                
                df_secondary = self.fetch_ohlcv(
                    self.secondary_exchange,
                    symbol,
                    config.START_DATE,
                    config.END_DATE
                )
                
                gaps = self.detect_gaps(df_primary)
                
                all_data[symbol] = {
                    'primary': df_primary,
                    'secondary': df_secondary,
                    'gaps': gaps
                }
                
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                continue
        
        return all_data
    
    def save_raw_data(self, all_data: Dict[str, Dict[str, pd.DataFrame]]):
        """
        Save raw data to CSV files.
        
        Args:
            all_data: Dictionary containing data for all symbols
        """
        for symbol, data in all_data.items():
            safe_symbol = symbol.replace('/', '_')
            
            primary_path = f"{config.RAW_DATA_DIR}/{safe_symbol}_{config.PRIMARY_EXCHANGE}.csv"
            secondary_path = f"{config.RAW_DATA_DIR}/{safe_symbol}_{config.SECONDARY_EXCHANGE}.csv"
            
            if not data['primary'].empty:
                data['primary'].to_csv(primary_path, index=False)
                logger.info(f"Saved primary data to {primary_path}")
            
            if not data['secondary'].empty:
                data['secondary'].to_csv(secondary_path, index=False)
                logger.info(f"Saved secondary data to {secondary_path}")
            
            if data['gaps']:
                gaps_path = f"{config.RAW_DATA_DIR}/{safe_symbol}_gaps.csv"
                gaps_df = pd.DataFrame(data['gaps'], columns=['gap_start', 'gap_end', 'gap_length'])
                gaps_df.to_csv(gaps_path, index=False)
                logger.info(f"Saved gaps info to {gaps_path}")


if __name__ == '__main__':
    fetcher = DataFetcher()
    all_data = fetcher.fetch_all_symbols()
    fetcher.save_raw_data(all_data)
    logger.info("Data fetching completed")
