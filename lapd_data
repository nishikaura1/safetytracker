import logging
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Any
import pandas as pd
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)

class LAPDDataProcessor:
    def __init__(self):
        self.crime_types = {
            'violent': ['ASSAULT', 'ROBBERY', 'HOMICIDE', 'RAPE'],
            'property': ['BURGLARY', 'THEFT', 'VANDALISM'],
            'quality_of_life': ['TRESPASSING', 'LOITERING', 'DISTURBANCE']
        }
        self.time_windows = {
            'day': (6, 18),  # 6 AM to 6 PM
            'night': (18, 6)  # 6 PM to 6 AM
        }

    def get_crime_data(self, lat: float, lon: float, radius: int = 200, days: int = 30) -> List[Dict]:
        """Fetch crime data for a specific location and time period."""
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            params = {
                "$where": f"within_circle(location, {lat}, {lon}, {radius}) AND date_occ >= '{start_date.strftime('%Y-%m-%d')}'",
                "$limit": 1000,
                "$order": "date_occ DESC"
            }
            
            response = requests.get("https://data.lacity.org/resource/2nrs-mtv8.json", params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching LAPD crime data: {str(e)}")
            return []

    def analyze_crime_patterns(self, crime_data: List[Dict]) -> Dict[str, Any]:
        """Analyze crime patterns and return risk metrics."""
        if not crime_data:
            return self._get_default_metrics()

        df = pd.DataFrame(crime_data)
        df['date_occ'] = pd.to_datetime(df['date_occ'])
        
        # Calculate time-based metrics
        df['hour'] = df['date_occ'].dt.hour
        df['is_night'] = df['hour'].apply(lambda x: not (self.time_windows['day'][0] <= x < self.time_windows['day'][1]))
        
        # Calculate crime type frequencies
        crime_counts = defaultdict(int)
        for crime in df['crm_cd_desc'].str.upper():
            for category, types in self.crime_types.items():
                if any(t in crime for t in types):
                    crime_counts[category] += 1

        # Calculate temporal patterns
        recent_crimes = df[df['date_occ'] >= (datetime.now() - timedelta(days=7))]
        recent_count = len(recent_crimes)
        
        # Calculate spatial density
        total_crimes = len(df)
        density_score = min(total_crimes / 10, 1.0)  # Normalize to 0-1
        
        # Calculate time-based risk
        night_crimes = len(df[df['is_night']])
        night_risk = min(night_crimes / total_crimes if total_crimes > 0 else 0, 1.0)
        
        return {
            'total_crimes': total_crimes,
            'recent_crimes': recent_count,
            'crime_density': density_score,
            'night_risk': night_risk,
            'crime_type_distribution': dict(crime_counts),
            'temporal_risk': self._calculate_temporal_risk(df),
            'spatial_risk': self._calculate_spatial_risk(df)
        }

    def _calculate_temporal_risk(self, df: pd.DataFrame) -> float:
        """Calculate risk based on temporal patterns."""
        if df.empty:
            return 0.0
            
        # Weight recent crimes more heavily
        now = datetime.now()
        df['hours_ago'] = (now - df['date_occ']).dt.total_seconds() / 3600
        df['time_weight'] = np.exp(-df['hours_ago'] / 168)  # 168 hours = 1 week
        
        return min(df['time_weight'].sum() / 10, 1.0)

    def _calculate_spatial_risk(self, df: pd.DataFrame) -> float:
        """Calculate risk based on spatial clustering."""
        if df.empty:
            return 0.0
            
        # Simple clustering detection
        if len(df) > 5:
            # Check if crimes are clustered in time
            time_diffs = df['date_occ'].diff().dt.total_seconds() / 3600
            clustering_score = (time_diffs < 24).mean()  # Crimes within 24 hours
            return min(clustering_score, 1.0)
        return 0.0

    def _get_default_metrics(self) -> Dict[str, Any]:
        """Return default metrics when no data is available."""
        return {
            'total_crimes': 0,
            'recent_crimes': 0,
            'crime_density': 0.0,
            'night_risk': 0.0,
            'crime_type_distribution': {k: 0 for k in self.crime_types.keys()},
            'temporal_risk': 0.0,
            'spatial_risk': 0.0
        }

    def get_location_risk_score(self, lat: float, lon: float) -> Dict[str, Any]:
        """Get comprehensive risk analysis for a location."""
        crime_data = self.get_crime_data(lat, lon)
        risk_metrics = self.analyze_crime_patterns(crime_data)
        
        # Calculate overall risk score
        weights = {
            'crime_density': 0.3,
            'night_risk': 0.2,
            'temporal_risk': 0.3,
            'spatial_risk': 0.2
        }
        
        overall_risk = sum(
            risk_metrics[metric] * weight 
            for metric, weight in weights.items()
        )
        
        risk_metrics['overall_risk'] = round(overall_risk, 2)
        if not crime_data:
            return {"base_score": 0.5, "message": "Limited data for this route, score is estimated."}
        return risk_metrics 
