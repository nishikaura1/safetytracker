import logging
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Any

from config import (
    LAPD_CRIME_ENDPOINT,
    WEATHER_API_URL,
    ORS_API_URL,
    WEATHER_API_KEY,
    ORS_API_KEY
)

logger = logging.getLogger(__name__)

def get_lapd_crime_count(lat: float, lon: float, radius: int = 200) -> int:
    try:
        params = {
            "$where": f"within_circle(location, {lat}, {lon}, {radius})",
            "$limit": 100
        }
        response = requests.get(LAPD_CRIME_ENDPOINT, params=params)
        response.raise_for_status()
        data = response.json()
        return len(data)
    except Exception as e:
        logger.error(f"Error fetching LAPD crime data: {str(e)}")
        return 0

def get_forecasted_weather(lat: float, lon: float) -> str:
    params = {
        "lat": lat,
        "lon": lon,
        "appid": WEATHER_API_KEY,
        "units": "metric"
    }
    try:
        response = requests.get(WEATHER_API_URL, params=params)
        response.raise_for_status()
        data = response.json()
        future_time = datetime.utcnow() + timedelta(hours=1)
        
        for entry in data.get("list", []):
            forecast_time = datetime.strptime(entry["dt_txt"], "%Y-%m-%d %H:%M:%S")
            if forecast_time >= future_time:
                return entry["weather"][0]["main"].lower()
        return "unknown"
    except Exception as e:
        logger.error(f"Error fetching weather data: {str(e)}")
        return "unknown"

def get_openroute_path(
    home_lat: float,
    home_lon: float,
    school_lat: float,
    school_lon: float
) -> List[List[float]]:
    headers = {
        "Authorization": ORS_API_KEY,
        "Content-Type": "application/json"
    }
    body = {
        "coordinates": [[home_lon, home_lat], [school_lon, school_lat]]
    }
    try:
        response = requests.post(ORS_API_URL, json=body, headers=headers)
        response.raise_for_status()
        data = response.json()
        coords = data["features"][0]["geometry"]["coordinates"]
        return [[lat, lon] for lon, lat in coords]
    except Exception as e:
        logger.error(f"Error fetching route: {str(e)}")
        raise 
