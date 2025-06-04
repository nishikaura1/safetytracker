import random
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

from config import DATA_LOG
from models.safety_model import SafetyModel

def simulate_safety_features() -> tuple:
    lighting = random.choice(["good", "poor"])
    reports = random.choice(["yes", "no"])
    return lighting, reports

def encode_features(
    crime_count: int,
    lighting: str,
    reports: str,
    weather: str,
    terrain: str,
    mud_risk: bool
) -> List[float]:
    return [
        float(crime_count),
        1.0 if lighting == "poor" else 0.0,
        1.0 if reports == "yes" else 0.0,
        1.0 if weather in ["rain", "thunderstorm"] else 0.0,
        1.0 if terrain == "dirt" else 0.0,
        1.0 if mud_risk else 0.0
    ]

def log_segment(features: List[float], predicted_score: float) -> None:
    try:
        file_exists = Path(DATA_LOG).exists()
        with open(DATA_LOG, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    "crime_count", "lighting_poor", "past_reports",
                    "bad_weather", "terrain_dirt", "mud_risk", "safety_score"
                ])
            writer.writerow(features + [predicted_score])
    except Exception as e:
        logger.error(f"Error logging segment: {str(e)}")

def retrain_model(safety_model: SafetyModel) -> Optional[Dict[str, float]]:
    if not Path(DATA_LOG).exists():
        return None
    
    try:
        X, y = [], []
        with open(DATA_LOG, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                X.append([float(row[col]) for col in reader.fieldnames[:-1]])
                y.append(float(row["safety_score"]))
        
        if len(X) > 10:
            X = np.array(X)
            y = np.array(y)
            return safety_model.train(X, y)
        return None
    except Exception as e:
        logger.error(f"Error retraining model: {str(e)}")
        return None 
