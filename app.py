from fastapi import FastAPI, Query, HTTPException
from typing import List, Dict, Any
import random
import requests
import polyline
import os
from datetime import datetime, timedelta
import csv
from pathlib import Path
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import joblib
import json
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


from config import DATA_LOG
from safety_model import SafetyModel
from external_apis import (
    get_lapd_crime_count,
    get_forecasted_weather,
    get_openroute_path
)
from helpers import (
    simulate_safety_features,
    encode_features,
    log_segment,
    retrain_model
)
from data_processor import DataProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('safety_tracker.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Safety Tracker API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
ORS_API_KEY = "5b3ce3597851110001cf62483583f4943d614cdb84fd147b8ddf4a71"
WEATHER_API_KEY = "2f8f46849820b1e349ceff6d33269056"
MODEL_DIR = "models"
MODEL_VERSION = "v2.0-rf"
MODEL_PATH = os.path.join(MODEL_DIR, f"safety_model_{MODEL_VERSION}.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, f"scaler_{MODEL_VERSION}.pkl")
LAPD_CRIME_ENDPOINT = "https://data.lacity.org/resource/2nrs-mtv8.json"

# Create model directory if it doesn't exist
Path(MODEL_DIR).mkdir(exist_ok=True)

# Initialize safety model
safety_model = SafetyModel()
data_processor = DataProcessor()

class LocationRequest(BaseModel):
    lat: float
    lon: float

class PredictionResponse(BaseModel):
    base_score: float
    confidence: float
    feature_contributions: Dict[str, float]
    prediction_interval: Dict[str, float]
    lapd_analysis: Optional[Dict]
    adjusted_score: Optional[float]
    timestamp: str

def get_safe_route(
    home_lat: float,
    home_lon: float,
    school_lat: float,
    school_lon: float
) -> Dict[str, Any]:
    try:
        coords = get_openroute_path(home_lat, home_lon, school_lat, school_lon)
        segments = []
        scores = []

        for i, (lat, lon) in enumerate(coords):
            crime_count = get_lapd_crime_count(lat, lon)
            lighting, reports = simulate_safety_features()
            weather = get_forecasted_weather(lat, lon)
            terrain = 'dirt' if (lat + lon) % 2 > 1 else 'paved'
            mud_risk = weather in ["rain", "thunderstorm"] and terrain == "dirt"

            features = encode_features(
                crime_count, lighting, reports, weather, terrain, mud_risk
            )
            safety_score = safety_model.predict(features)
            
            scores.append(safety_score)
            log_segment(features, safety_score)

            segments.append({
                "segment_id": f"S{i+1}",
                "location": {
                    "latitude": round(lat, 6),
                    "longitude": round(lon, 6)
                },
                "crime_count": crime_count,
                "environment": {
                    "lighting": lighting,
                    "terrain": terrain,
                    "mud_risk": mud_risk
                },
                "weather": {
                    "forecasted": weather
                },
                "safety_score": safety_score,
                "icon": "⚠️" if mud_risk or lighting == "poor" or crime_count > 10 else "✅"
            })

        # Train model with new data
        training_metrics = retrain_model(safety_model)

        summary = {
            "total_segments": len(segments),
            "highest_risk_score": min(scores),
            "lowest_risk_score": max(scores),
            "overall_risk": "high" if min(scores) < 0.4 else "medium" if min(scores) < 0.7 else "low",
            "model_metrics": training_metrics
        }

        return {
            "summary": summary,
            "route": segments,
            "model_version": MODEL_VERSION
        }

    except Exception as e:
        logger.error(f"Error generating safe route: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Error generating safe route"
        )

@app.get("/route-safety")
def get_route_safety(
    home_lat: float = Query(...),
    home_lon: float = Query(...),
    school_lat: float = Query(...),
    school_lon: float = Query(...)
) -> Dict[str, Any]:
    """
    Get safety analysis for a route between two points.
    
    Args:
        home_lat: Starting point latitude
        home_lon: Starting point longitude
        school_lat: Destination latitude
        school_lon: Destination longitude
    
    Returns:
        Dict containing route safety analysis
    """
    return get_safe_route(home_lat, home_lon, school_lat, school_lon)

@app.get("/model-metrics")
def get_model_metrics() -> Dict[str, Any]:
    """
    Get current model performance metrics.
    
    Returns:
        Dict containing model metrics if available
    """
    if not Path(DATA_LOG).exists():
        raise HTTPException(
            status_code=404,
            detail="No training data available"
        )
    
    try:
        X, y = [], []
        with open(DATA_LOG, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                X.append([float(row[col]) for col in reader.fieldnames[:-1]])
                y.append(float(row["safety_score"]))
        
        if len(X) < 10:
            raise HTTPException(
                status_code=400,
                detail="Insufficient data for metrics"
            )
        
        X = np.array(X)
        y = np.array(y)
        
        # Perform cross-validation
        cv_scores = cross_val_score(
            safety_model.model,
            safety_model.scaler.transform(X),
            y,
            cv=5,
            scoring='r2'
        )
        
        return {
            "cross_validation_scores": cv_scores.tolist(),
            "mean_cv_score": cv_scores.mean(),
            "std_cv_score": cv_scores.std(),
            "feature_importance": dict(zip(
                safety_model.feature_names,
                safety_model.model.feature_importances_
            ))
        }
    except Exception as e:
        logger.error(f"Error getting model metrics: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Error calculating model metrics"
        )

@app.get("/")
async def root():
    return {"message": "Safety Tracker API"}

@app.post("/predict", response_model=PredictionResponse)
async def predict_safety(location: LocationRequest):
    try:
        # Collect features
        crime_data = data_processor.collect_lapd_data(location.lat, location.lon)
        weather_data = data_processor.collect_weather_data(location.lat, location.lon)
        
        # Process data
        crime_features = data_processor.process_crime_data(crime_data)
        weather_features = data_processor.process_weather_data(weather_data)
        
        # Combine features
        features = {
            'lat': location.lat,
            'lon': location.lon,
            'timestamp': datetime.now().isoformat(),
            **crime_features,
            **weather_features
        }
        
        # Make prediction
        prediction = safety_model.predict(
            list(features.values()),
            lat=location.lat,
            lon=location.lon
        )
        
        # Add timestamp
        prediction['timestamp'] = datetime.now().isoformat()
        
        return prediction
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model-info")
async def get_model_info():
    try:
        return {
            "model_path": str(MODEL_PATH),
            "feature_names": safety_model.feature_names,
            "last_updated": datetime.fromtimestamp(MODEL_PATH.stat().st_mtime).isoformat() if MODEL_PATH.exists() else None
        }
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 