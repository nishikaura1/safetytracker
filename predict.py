import logging
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

from models.safety_model import SafetyModel
from data.data_processor import DataProcessor
from config import MODEL_PATH

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Make safety predictions')
    parser.add_argument('--lat', type=float, required=True, help='Latitude')
    parser.add_argument('--lon', type=float, required=True, help='Longitude')
    parser.add_argument('--output', type=str, help='Output file path for predictions')
    return parser.parse_args()

def collect_features(lat: float, lon: float) -> dict:
    """Collect and process features for prediction."""
    try:
        # Initialize data processor
        data_processor = DataProcessor()
        
        # Collect data
        crime_data = data_processor.collect_lapd_data(lat, lon)
        weather_data = data_processor.collect_weather_data(lat, lon)
        
        # Process data
        crime_features = data_processor.process_crime_data(crime_data)
        weather_features = data_processor.process_weather_data(weather_data)
        
        # Combine features
        features = {
            'lat': lat,
            'lon': lon,
            'timestamp': datetime.now().isoformat(),
            **crime_features,
            **weather_features
        }
        
        return features
        
    except Exception as e:
        logger.error(f"Error collecting features: {str(e)}")
        raise

def make_prediction(features: dict) -> dict:
    """Make safety prediction using the model."""
    try:
        # Initialize model
        model = SafetyModel()
        
        # Make prediction
        prediction = model.predict(
            list(features.values()),
            lat=features['lat'],
            lon=features['lon']
        )
        
        return prediction
        
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        raise

def save_prediction(prediction: dict, output_path: str):
    """Save prediction results to file."""
    try:
        # Convert prediction to DataFrame
        df = pd.DataFrame([prediction])
        
        # Save to file
        df.to_csv(output_path, index=False)
        logger.info(f"Saved prediction to {output_path}")
        
    except Exception as e:
        logger.error(f"Error saving prediction: {str(e)}")
        raise

def main():
    # Parse arguments
    args = parse_args()
    
    try:
        # Collect features
        logger.info(f"Collecting features for location ({args.lat}, {args.lon})")
        features = collect_features(args.lat, args.lon)
        
        # Make prediction
        logger.info("Making prediction...")
        prediction = make_prediction(features)
        
        # Log prediction
        logger.info("Prediction results:")
        for key, value in prediction.items():
            if isinstance(value, (int, float)):
                logger.info(f"{key}: {value:.4f}")
            elif isinstance(value, dict):
                logger.info(f"{key}:")
                for k, v in value.items():
                    logger.info(f"  {k}: {v:.4f}")
        
        # Save prediction if output path is provided
        if args.output:
            save_prediction(prediction, args.output)
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 
