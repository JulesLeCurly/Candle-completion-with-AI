"""
Main orchestration script for cryptocurrency data completion system.
Handles both training and inference modes.
"""

import argparse
import logging
import os
import pandas as pd
import json
from datetime import datetime
from typing import Dict
import config
from data_fetcher import DataFetcher
from feature_engineering import FeatureEngineer
from dataset_builder import DatasetBuilder
from trainer import ModelTrainer
from predictor import GapPredictor
from tensorflow import keras

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT,
    handlers=[
        logging.FileHandler(config.LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class CryptoCompletionPipeline:
    """Main pipeline for cryptocurrency data completion."""
    
    def __init__(self):
        """Initialize pipeline components."""
        self.fetcher = DataFetcher()
        self.feature_engineer = FeatureEngineer()
        self.dataset_builder = DatasetBuilder()
        self.trainer = None
        self.predictor = None
        
        logger.info("Pipeline initialized")
    
    def run_training_mode(self, symbols: list = None, epochs: int = None, 
                          force_download: bool = False):
        """
        Run complete training pipeline.
        
        Args:
            symbols: List of symbols to train on (None = use all)
            epochs: Number of training epochs
            force_download: Force re-download of data even if exists locally
        """
        logger.info("=" * 80)
        logger.info("STARTING TRAINING MODE")
        logger.info("=" * 80)
        
        symbols = symbols or config.SYMBOLS
        epochs = epochs or config.EPOCHS
        
        # Step 1: Fetch data
        logger.info("Step 1/6: Fetching data from exchanges")
        if force_download:
            logger.info("Force download enabled - will re-download all data")
        else:
            logger.info("Checking for existing data files first")
        
        all_data = self.fetcher.fetch_all_symbols(force_download=force_download)
        
        # Save raw data (only if newly downloaded or force_download)
        if force_download:
            self.fetcher.save_raw_data(all_data)
        
        # Step 2: Feature engineering
        logger.info("Step 2/6: Engineering features")
        processed_data = {}
        
        for symbol in symbols:
            if symbol not in all_data:
                logger.warning(f"No data for {symbol}, skipping")
                continue
            
            df_primary = all_data[symbol]['primary']
            df_secondary = all_data[symbol]['secondary']
            
            if df_primary.empty or df_secondary.empty:
                logger.warning(f"Empty data for {symbol}, skipping")
                continue
            
            # Process features
            df_primary_proc, df_secondary_proc = self.feature_engineer.process_symbol(
                df_primary, df_secondary, symbol, fit=True
            )
            
            processed_data[symbol] = (df_primary_proc, df_secondary_proc)
        
        # Save normalization parameters
        norm_params_path = os.path.join(config.MODEL_DIR, 'normalization_params.pkl')
        self.feature_engineer.save_normalization_params(norm_params_path)
        
        # Step 3: Build dataset
        logger.info("Step 3/6: Building training dataset")
        splits = self.dataset_builder.build_multi_symbol_dataset(processed_data)
        
        if not splits:
            logger.error("Failed to build dataset")
            return
        
        # Step 4: Train model
        logger.info("Step 4/6: Training model")
        self.trainer = ModelTrainer()
        
        history = self.trainer.train(
            splits['train'],
            splits['val'],
            epochs=epochs
        )
        
        # Step 5: Evaluate model
        logger.info("Step 5/6: Evaluating model")
        test_results = self.trainer.evaluate(splits['test'])
        
        # Step 6: Generate reports
        logger.info("Step 6/6: Generating reports")
        
        # Save model
        self.trainer.save_model()
        
        # Plot training history
        plot_path = os.path.join(config.LOG_DIR, 'training_history.png')
        self.trainer.plot_training_history(save_path=plot_path)
        
        # Generate report
        report_path = os.path.join(config.LOG_DIR, 'training_report.json')
        report = self.trainer.generate_training_report(save_path=report_path)
        
        logger.info("=" * 80)
        logger.info("TRAINING COMPLETED SUCCESSFULLY")
        logger.info(f"Model saved to: {config.MODEL_PATH}")
        logger.info(f"Test MAE: {test_results['mae']:.4f}")
        logger.info(f"Test MAPE: {test_results['mape']:.2f}%")
        logger.info("=" * 80)
    
    def run_completion_mode(self, input_dir: str, output_dir: str, 
                           model_path: str = None):
        """
        Run completion mode to fill gaps in existing data.
        
        Args:
            input_dir: Directory with raw data CSVs
            output_dir: Directory to save completed data
            model_path: Path to trained model
        """
        logger.info("=" * 80)
        logger.info("STARTING COMPLETION MODE")
        logger.info("=" * 80)
        
        model_path = model_path or config.MODEL_PATH
        
        # Check if model exists
        if not os.path.exists(model_path):
            logger.error(f"Model not found at {model_path}")
            logger.error("Please train a model first using --mode train")
            return
        
        # Load model
        logger.info(f"Loading model from {model_path}")
        try:
            from model import AttentionLayer, ohlc_constraint_loss, mae_metric, mape_metric
            
            model = keras.models.load_model(
                model_path,
                custom_objects={
                    'AttentionLayer': AttentionLayer,
                    'ohlc_constraint_loss': ohlc_constraint_loss,
                    'mae_metric': mae_metric,
                    'mape_metric': mape_metric
                }
            )
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return
        
        # Load normalization parameters
        norm_params_path = os.path.join(config.MODEL_DIR, 'normalization_params.pkl')
        if os.path.exists(norm_params_path):
            self.feature_engineer.load_normalization_params(norm_params_path)
        else:
            logger.warning("Normalization parameters not found")
        
        # Initialize predictor
        self.predictor = GapPredictor(model, self.feature_engineer)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Process each symbol
        completion_stats = {}
        
        for symbol in config.SYMBOLS:
            logger.info(f"Processing {symbol}")
            
            safe_symbol = symbol.replace('/', '_')
            primary_file = os.path.join(input_dir, f"{safe_symbol}_{config.PRIMARY_EXCHANGE}.csv")
            secondary_file = os.path.join(input_dir, f"{safe_symbol}_{config.SECONDARY_EXCHANGE}.csv")
            
            if not os.path.exists(primary_file):
                logger.warning(f"Primary file not found: {primary_file}")
                continue
            
            if not os.path.exists(secondary_file):
                logger.warning(f"Secondary file not found: {secondary_file}")
                continue
            
            # Load data
            df_primary = pd.read_csv(primary_file)
            df_secondary = pd.read_csv(secondary_file)
            
            # Convert timestamps
            df_primary['open_time'] = pd.to_datetime(df_primary['open_time'])
            df_secondary['open_time'] = pd.to_datetime(df_secondary['open_time'])
            
            # Process features
            df_primary_proc, df_secondary_proc = self.feature_engineer.process_symbol(
                df_primary, df_secondary, symbol, fit=False
            )
            
            # Complete data
            completed = self.predictor.complete_symbol_data(
                df_primary_proc,
                df_secondary_proc,
                symbol
            )
            
            # Denormalize for output
            completed = self.feature_engineer.denormalize(
                completed, symbol, config.OHLCV_FEATURES
            )
            
            # Prepare output
            output_cols = [col for col in config.OUTPUT_COLUMNS if col in completed.columns]
            completed_output = completed[output_cols]
            
            # Save completed data
            output_file = os.path.join(output_dir, f"{safe_symbol}_completed.csv")
            completed_output.to_csv(output_file, index=False)
            
            # Calculate statistics
            total_rows = len(completed)
            predicted_rows = completed['is_predicted'].sum()
            avg_confidence = completed[completed['is_predicted']]['prediction_confidence'].mean()
            
            completion_stats[symbol] = {
                'total_rows': int(total_rows),
                'predicted_rows': int(predicted_rows),
                'completion_rate': float(predicted_rows / total_rows * 100),
                'avg_confidence': float(avg_confidence) if not pd.isna(avg_confidence) else 0.0
            }
            
            logger.info(f"Completed {symbol}: {predicted_rows}/{total_rows} rows predicted "
                       f"({completion_stats[symbol]['completion_rate']:.2f}%)")
        
        # Generate quality report
        self._generate_quality_report(completion_stats, output_dir)
        
        logger.info("=" * 80)
        logger.info("COMPLETION MODE FINISHED")
        logger.info(f"Completed data saved to: {output_dir}")
        logger.info("=" * 80)
    
    def _generate_quality_report(self, stats: Dict, output_dir: str):
        """
        Generate quality report for completed data.
        
        Args:
            stats: Dictionary of completion statistics per symbol
            output_dir: Directory to save report
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_symbols': len(stats),
            'per_symbol_stats': stats,
            'summary': {
                'total_predicted_rows': sum(s['predicted_rows'] for s in stats.values()),
                'avg_completion_rate': sum(s['completion_rate'] for s in stats.values()) / len(stats),
                'avg_confidence': sum(s['avg_confidence'] for s in stats.values()) / len(stats)
            }
        }
        
        report_path = os.path.join(output_dir, 'quality_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Quality report saved to {report_path}")
        logger.info(f"Average completion rate: {report['summary']['avg_completion_rate']:.2f}%")
        logger.info(f"Average confidence: {report['summary']['avg_confidence']:.3f}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Cryptocurrency Data Completion System'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'complete'],
        required=True,
        help='Operation mode: train or complete'
    )
    
    parser.add_argument(
        '--symbols',
        type=str,
        help='Comma-separated list of symbols (e.g., BTC/USDT,ETH/USDT)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=config.EPOCHS,
        help=f'Number of training epochs (default: {config.EPOCHS})'
    )
    
    parser.add_argument(
        '--input',
        type=str,
        default=config.RAW_DATA_DIR,
        help='Input directory for completion mode'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=config.COMPLETED_DATA_DIR,
        help='Output directory for completion mode'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        help='Path to model file'
    )
    
    parser.add_argument(
        '--force-download',
        action='store_true',
        help='Force re-download of data even if exists locally'
    )
    
    args = parser.parse_args()
    
    # Parse symbols
    symbols = None
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(',')]
    
    # Initialize pipeline
    pipeline = CryptoCompletionPipeline()
    
    # Run appropriate mode
    if args.mode == 'train':
        pipeline.run_training_mode(
            symbols=symbols, 
            epochs=args.epochs,
            force_download=args.force_download
        )
    elif args.mode == 'complete':
        pipeline.run_completion_mode(
            input_dir=args.input,
            output_dir=args.output,
            model_path=args.model
        )


if __name__ == '__main__':
    main()