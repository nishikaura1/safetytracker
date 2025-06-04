import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, KBinsDiscretizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class FeatureEngineering:
    def __init__(self):
        self.poly_features = None
        self.pca = None
        self.kmeans = None
        self.feature_names = None
        self.bin_encoders = {}
        
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between important variables."""
        # Create polynomial features for key numerical columns
        if self.poly_features is None:
            self.poly_features = PolynomialFeatures(degree=2, include_bias=False)
            poly_features = self.poly_features.fit_transform(df.select_dtypes(include=[np.number]))
            self.feature_names = self.poly_features.get_feature_names_out(df.select_dtypes(include=[np.number]).columns)
        else:
            poly_features = self.poly_features.transform(df.select_dtypes(include=[np.number]))
            
        # Create interaction DataFrame
        poly_df = pd.DataFrame(poly_features, columns=self.feature_names)
        
        # Add to original DataFrame
        return pd.concat([df, poly_df], axis=1)

    def create_temporal_features(self, df: pd.DataFrame, time_column: str) -> pd.DataFrame:
        """Create temporal features from datetime columns."""
        df = df.copy()
        df[time_column] = pd.to_datetime(df[time_column])
        
        # Basic temporal features
        df['hour'] = df[time_column].dt.hour
        df['day_of_week'] = df[time_column].dt.dayofweek
        df['month'] = df[time_column].dt.month
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Cyclical encoding for periodic features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week']/7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week']/7)
        
        return df

    def create_spatial_features(self, df: pd.DataFrame, lat_col: str, lon_col: str) -> pd.DataFrame:
        """Create spatial features from coordinates."""
        df = df.copy()
        
        # Calculate distance from city center (example coordinates)
        city_center_lat, city_center_lon = 34.0522, -118.2437  # Los Angeles
        df['distance_from_center'] = np.sqrt(
            (df[lat_col] - city_center_lat)**2 + 
            (df[lon_col] - city_center_lon)**2
        )
        
        # Create spatial clusters if not already done
        if self.kmeans is None:
            self.kmeans = KMeans(n_clusters=5, random_state=42)
            spatial_features = self.kmeans.fit_predict(df[[lat_col, lon_col]])
        else:
            spatial_features = self.kmeans.predict(df[[lat_col, lon_col]])
            
        df['spatial_cluster'] = spatial_features
        
        return df

    def create_risk_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create risk-related features."""
        df = df.copy()
        
        # Create risk score based on multiple factors
        df['crime_density_risk'] = df['crime_count'] / df['area'] if 'area' in df.columns else df['crime_count']
        
        # Create time-based risk features
        if 'hour' in df.columns:
            df['night_risk'] = ((df['hour'] >= 18) | (df['hour'] <= 6)).astype(int)
            df['peak_hour_risk'] = ((df['hour'] >= 17) & (df['hour'] <= 19)).astype(int)
        
        # Create weather risk features
        if 'weather_condition' in df.columns:
            df['weather_risk'] = df['weather_condition'].map({
                'clear': 0.1,
                'cloudy': 0.2,
                'rainy': 0.4,
                'stormy': 0.8
            }).fillna(0.3)
        
        return df

    def create_statistical_features(self, df: pd.DataFrame, window_size: int = 7) -> pd.DataFrame:
        """Create statistical features using rolling windows."""
        df = df.copy()
        
        # Calculate rolling statistics for numerical columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in ['lat', 'lon', 'hour', 'day_of_week', 'month']:  # Skip coordinate and temporal columns
                df[f'{col}_rolling_mean'] = df[col].rolling(window=window_size, min_periods=1).mean()
                df[f'{col}_rolling_std'] = df[col].rolling(window=window_size, min_periods=1).std()
                df[f'{col}_rolling_max'] = df[col].rolling(window=window_size, min_periods=1).max()
                df[f'{col}_rolling_min'] = df[col].rolling(window=window_size, min_periods=1).min()
        
        return df

    def create_pca_features(self, df: pd.DataFrame, n_components: int = 5) -> pd.DataFrame:
        """Create PCA features from numerical columns."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if self.pca is None:
            self.pca = PCA(n_components=n_components)
            pca_features = self.pca.fit_transform(df[numeric_cols])
        else:
            pca_features = self.pca.transform(df[numeric_cols])
            
        pca_df = pd.DataFrame(
            pca_features,
            columns=[f'pca_{i+1}' for i in range(n_components)]
        )
        
        return pd.concat([df, pca_df], axis=1)

    def create_binned_features(self, df: pd.DataFrame, columns: List[str], n_bins: int = 5) -> pd.DataFrame:
        """Create binned features for numerical columns."""
        df = df.copy()
        
        for col in columns:
            if col in df.columns:
                if col not in self.bin_encoders:
                    self.bin_encoders[col] = KBinsDiscretizer(
                        n_bins=n_bins,
                        encode='ordinal',
                        strategy='quantile'
                    )
                    df[f'{col}_binned'] = self.bin_encoders[col].fit_transform(df[[col]])
                else:
                    df[f'{col}_binned'] = self.bin_encoders[col].transform(df[[col]])
        
        return df

    def create_all_features(self, df: pd.DataFrame, lat_col: str = 'lat', lon_col: str = 'lon', 
                          time_col: Optional[str] = None) -> pd.DataFrame:
        """Create all feature engineering steps in sequence."""
        try:
            # Create interaction features
            df = self.create_interaction_features(df)
            
            # Create temporal features if time column is provided
            if time_col:
                df = self.create_temporal_features(df, time_col)
            
            # Create spatial features
            df = self.create_spatial_features(df, lat_col, lon_col)
            
            # Create risk features
            df = self.create_risk_features(df)
            
            # Create statistical features
            df = self.create_statistical_features(df)
            
            # Create PCA features
            df = self.create_pca_features(df)
            
            # Create binned features for important numerical columns
            important_cols = ['crime_count', 'distance_from_center', 'crime_density_risk']
            df = self.create_binned_features(df, important_cols)
            
            return df
            
        except Exception as e:
            logger.error(f"Error in feature engineering: {str(e)}")
            return df

    def get_feature_importance(self, df: pd.DataFrame, target_col: str) -> Dict[str, float]:
        """Calculate feature importance using correlation and variance."""
        importance_scores = {}
        
        # Calculate correlation with target
        correlations = df.corr()[target_col].abs()
        
        # Calculate variance for each feature
        variances = df.var()
        
        # Combine scores (normalized)
        for col in df.columns:
            if col != target_col:
                corr_score = correlations.get(col, 0)
                var_score = variances.get(col, 0)
                importance_scores[col] = (corr_score + var_score) / 2
        
        return dict(sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)) 
