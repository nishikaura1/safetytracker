import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

# Model paths
MODEL_PATH = MODEL_DIR / "safety_model.joblib"
SCALER_PATH = MODEL_DIR / "scaler.joblib"
FEATURE_ENGINEERING_PATH = MODEL_DIR / "feature_engineering.joblib"

# Default path for logged training data
DATA_LOG = BASE_DIR / "segment_logs.csv"

# API Configuration
LAPD_API_URL = "https://data.lacity.org/resource/2nrs-mtv8.json"
WEATHER_API_URL = "https://api.openweathermap.org/data/2.5/weather"
WEATHER_API_KEY = "2f8f46849820b1e349ceff6d33269056"
ORS_API_URL = "https://api.openrouteservice.org/v2/directions/driving-car"
ORS_API_KEY = "5b3ce3597851110001cf62483583f4943d614cdb84fd147b8ddf4a71"

# Feature Configuration
FEATURE_NAMES = [
    'crime_count',
    'lighting_poor',
    'past_reports',
    'bad_weather',
    'terrain_dirt',
    'mud_risk',
    'lat',
    'lon',
    'timestamp'
]

# Model Parameters
MODEL_PARAMS = {
    'n_estimators': 200,
    'max_depth': 10,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'random_state': 42
}

# Feature Engineering Parameters
FEATURE_ENGINEERING_PARAMS = {
    'pca_components': 5,
    'kmeans_clusters': 5,
    'polynomial_degree': 2,
    'rolling_window_size': 7,
    'binning_n_bins': 5
}

# Safety Score Parameters
SAFETY_SCORE_PARAMS = {
    'crime_weight': 0.3,
    'lighting_weight': 0.2,
    'weather_weight': 0.2,
    'terrain_weight': 0.15,
    'history_weight': 0.15
}

# LAPD Data Parameters
LAPD_PARAMS = {
    'search_radius': 200,  # meters
    'time_window': 30,     # days
    'crime_types': {
        'violent': ['ASSAULT', 'ROBBERY', 'HOMICIDE', 'RAPE'],
        'property': ['BURGLARY', 'THEFT', 'VANDALISM'],
        'quality_of_life': ['TRESPASSING', 'LOITERING', 'DISTURBANCE']
    }
}

# Time Windows
TIME_WINDOWS = {
    'day': (6, 18),    # 6 AM to 6 PM
    'night': (18, 6)   # 6 PM to 6 AM
}

# Weather Risk Levels
WEATHER_RISK_LEVELS = {
    'clear': 0.1,
    'cloudy': 0.2,
    'rainy': 0.4,
    'stormy': 0.8
}

# City Center Coordinates (Los Angeles)
CITY_CENTER = {
    'lat': 34.0522,
    'lon': -118.2437
}

# Create necessary directories
for directory in [MODEL_DIR, DATA_DIR]:
    directory.mkdir(exist_ok=True) 