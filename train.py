import logging
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

from models.safety_model import SafetyModel
from models.feature_engineering import FeatureEngineering
from data.data_processor import DataProcessor
from config import MODEL_PATH, SCALER_PATH, FEATURE_ENGINEERING_PATH

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Train safety prediction model')
    parser.add_argument('--data', type=str, required=True, help='Path to training data CSV')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set size ratio')
    parser.add_argument('--random-state', type=int, default=42, help='Random seed')
    return parser.parse_args()

def load_training_data(data_path: str) -> tuple:
    """Load and prepare training data."""
    try:
        # Load data
        df = pd.read_csv(data_path)
        
        # Separate features and target
        X = df.drop('safety_score', axis=1)
        y = df['safety_score']
        
        return X, y
        
    except Exception as e:
        logger.error(f"Error loading training data: {str(e)}")
        raise

def train_model(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
    """Train the safety prediction model."""
    try:
        # Initialize model
        model = SafetyModel()
        
        # Train model
        metrics = model.train(X.values, y.values)
        
        # Log metrics
        logger.info("Training completed with metrics:")
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                logger.info(f"{metric}: {value:.4f}")
            elif isinstance(value, dict):
                logger.info(f"{metric}:")
                for k, v in value.items():
                    logger.info(f"  {k}: {v:.4f}")
        
        return model, metrics
        
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        raise

def main():
    # Parse arguments
    args = parse_args()
    
    try:
        # Load data
        logger.info(f"Loading data from {args.data}")
        X, y = load_training_data(args.data)
        
        # Train model
        logger.info("Training model...")
        model, metrics = train_model(X, y, args.test_size, args.random_state)
        
        # Save model
        logger.info(f"Saving model to {MODEL_PATH}")
        model.save_model()
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 
