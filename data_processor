import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path

from config import (
    LAPD_API_URL, WEATHER_API_URL, WEATHER_API_KEY,
    FEATURE_NAMES, LAPD_PARAMS, TIME_WINDOWS
)

logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self):
        self.data_dir = Path("data")
        self.data_dir.mkdir(exist_ok=True)
        
    def collect_lapd_data(self, lat: float, lon: float, radius: int = None, days: int = None) -> List[Dict]:
        """Collect LAPD crime data for a specific location."""
        try:
            radius = radius or LAPD_PARAMS['search_radius']
            days = days or LAPD_PARAMS['time_window']
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            params = {
                "$where": f"within_circle(location, {lat}, {lon}, {radius}) AND date_occ >= '{start_date.strftime('%Y-%m-%d')}'",
                "$limit": 1000,
                "$order": "date_occ DESC"
            }
            
            response = requests.get(LAPD_API_URL, params=params)
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            logger.error(f"Error collecting LAPD data: {str(e)}")
            return []

    def collect_weather_data(self, lat: float, lon: float) -> Dict:
        """Collect current weather data for a location."""
        try:
            params = {
                'lat': lat,
                'lon': lon,
                'appid': WEATHER_API_KEY,
                'units': 'metric'
            }
            
            response = requests.get(WEATHER_API_URL, params=params)
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            logger.error(f"Error collecting weather data: {str(e)}")
            return {}

    def process_crime_data(self, crime_data: List[Dict]) -> Dict[str, float]:
        """Process crime data into feature format."""
        if not crime_data:
            return {name: 0.0 for name in FEATURE_NAMES}
            
        df = pd.DataFrame(crime_data)
        df['date_occ'] = pd.to_datetime(df['date_occ'])
        
        # Calculate basic metrics
        total_crimes = len(df)
        recent_crimes = len(df[df['date_occ'] >= (datetime.now() - timedelta(days=7))])
        
        # Calculate time-based metrics
        df['hour'] = df['date_occ'].dt.hour
        night_crimes = len(df[df['hour'].apply(lambda x: not (TIME_WINDOWS['day'][0] <= x < TIME_WINDOWS['day'][1]))])
        
        # Calculate crime type distribution
        crime_types = {category: 0 for category in LAPD_PARAMS['crime_types'].keys()}
        for crime in df['crm_cd_desc'].str.upper():
            for category, types in LAPD_PARAMS['crime_types'].items():
                if any(t in crime for t in types):
                    crime_types[category] += 1
        
        return {
            'crime_count': total_crimes,
            'recent_crimes': recent_crimes,
            'night_crimes': night_crimes,
            **crime_types
        }

    def process_weather_data(self, weather_data: Dict) -> Dict[str, float]:
        """Process weather data into feature format."""
        if not weather_data:
            return {'bad_weather': 0.0}
            
        try:
            weather_main = weather_data['weather'][0]['main'].lower()
            weather_desc = weather_data['weather'][0]['description'].lower()
            
            # Determine if weather is bad
            bad_weather = any(term in weather_desc for term in ['rain', 'storm', 'snow', 'thunder'])
            
            return {
                'bad_weather': float(bad_weather),
                'temperature': weather_data['main']['temp'],
                'humidity': weather_data['main']['humidity'],
                'wind_speed': weather_data['wind']['speed']
            }
            
        except Exception as e:
            logger.error(f"Error processing weather data: {str(e)}")
            return {'bad_weather': 0.0}

    def create_training_data(self, locations: List[Dict[str, float]], 
                           safety_scores: List[float]) -> Tuple[pd.DataFrame, pd.Series]:
        """Create training data from collected information."""
        try:
            data = []
            for location, score in zip(locations, safety_scores):
                # Collect data
                crime_data = self.collect_lapd_data(location['lat'], location['lon'])
                weather_data = self.collect_weather_data(location['lat'], location['lon'])
                
                # Process data
                crime_features = self.process_crime_data(crime_data)
                weather_features = self.process_weather_data(weather_data)
                
                # Combine features
                features = {
                    **location,
                    **crime_features,
                    **weather_features,
                    'timestamp': datetime.now().isoformat()
                }
                
                data.append(features)
            
            # Create DataFrame
            df = pd.DataFrame(data)
            
            # Ensure all required features are present
            for feature in FEATURE_NAMES:
                if feature not in df.columns:
                    df[feature] = 0.0
            
            return df[FEATURE_NAMES], pd.Series(safety_scores)
            
        except Exception as e:
            logger.error(f"Error creating training data: {str(e)}")
            return pd.DataFrame(), pd.Series()

    def save_data(self, df: pd.DataFrame, filename: str):
        """Save processed data to file."""
        try:
            filepath = self.data_dir / filename
            df.to_csv(filepath, index=False)
            logger.info(f"Saved data to {filepath}")
        except Exception as e:
            logger.error(f"Error saving data: {str(e)}")

    def load_data(self, filename: str) -> pd.DataFrame:
        """Load processed data from file."""
        try:
            filepath = self.data_dir / filename
            return pd.read_csv(filepath)
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return pd.DataFrame() 
