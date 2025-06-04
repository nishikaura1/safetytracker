import logging
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score, make_scorer, mean_absolute_error
from sklearn.feature_selection import SelectFromModel
import joblib
import pandas as pd
import shap
import optuna
from scipy import stats

from config import MODEL_PATH, SCALER_PATH, MODEL_PARAMS, FEATURE_NAMES
from services.lapd_data import LAPDDataProcessor
from models.feature_engineering import FeatureEngineering

logger = logging.getLogger(__name__)

class SafetyModel:
    def __init__(self):
        self.model = None
        self.scaler = RobustScaler()  # Changed to RobustScaler for better handling of outliers
        self.feature_names = FEATURE_NAMES
        self.lapd_processor = LAPDDataProcessor()
        self.feature_engineering = FeatureEngineering()
        self.feature_selector = None
        self.explainer = None
        self.load_model()

    def load_model(self):
        try:
            if Path(MODEL_PATH).exists():
                self.model = joblib.load(MODEL_PATH)
                self.scaler = joblib.load(SCALER_PATH)
                logger.info(f"Loaded model version {MODEL_PATH}")
            else:
                logger.info("No existing model found. Will train new model when data is available.")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            self.model = None

    def save_model(self):
        try:
            joblib.dump(self.model, MODEL_PATH)
            joblib.dump(self.scaler, SCALER_PATH)
            logger.info(f"Saved model to {MODEL_PATH}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")

    def _prepare_features(self, X: np.ndarray) -> pd.DataFrame:
        """Convert raw features to a DataFrame with named columns."""
        df = pd.DataFrame(X, columns=self.feature_names)
        
        # Apply feature engineering
        df = self.feature_engineering.create_all_features(
            df,
            lat_col='lat' if 'lat' in df.columns else None,
            lon_col='lon' if 'lon' in df.columns else None,
            time_col='timestamp' if 'timestamp' in df.columns else None
        )
        
        return df

    def _optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Optimize model hyperparameters using Optuna."""
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
            }
            
            model = RandomForestRegressor(**params, random_state=42)
            scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
            return -np.mean(scores)

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=50)
        return study.best_params

    def _create_ensemble(self, X: np.ndarray, y: np.ndarray) -> VotingRegressor:
        """Create an ensemble of models."""
        rf = RandomForestRegressor(**self._optimize_hyperparameters(X, y), random_state=42)
        gb = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        
        return VotingRegressor([
            ('rf', rf),
            ('gb', gb)
        ])

    def _perform_feature_selection(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """Perform feature selection using SHAP values."""
        # Train a model for feature selection
        selector = SelectFromModel(
            RandomForestRegressor(n_estimators=100, random_state=42),
            prefit=False
        )
        selector.fit(X, y)
        
        # Get selected feature indices
        selected_indices = selector.get_support()
        selected_features = [f for f, s in zip(self.feature_names, selected_indices) if s]
        
        return X[:, selected_indices], selected_features

    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        try:
            # Prepare features with engineering
            df = self._prepare_features(X)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                df, y, test_size=0.2, random_state=42
            )

            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            # Feature selection
            X_train_selected, selected_features = self._perform_feature_selection(X_train_scaled, y_train)
            X_test_selected = X_test_scaled[:, [df.columns.get_loc(f) for f in selected_features]]
            
            # Create and train ensemble
            self.model = self._create_ensemble(X_train_selected, y_train)
            self.model.fit(X_train_selected, y_train)

            # Create SHAP explainer
            self.explainer = shap.TreeExplainer(self.model.named_estimators_['rf'])

            # Cross-validation metrics
            cv_scores = cross_val_score(
                self.model, X_train_selected, y_train,
                cv=5, scoring='neg_mean_squared_error'
            )
            
            # Evaluate model
            y_pred = self.model.predict(X_test_selected)
            metrics = {
                "mse": mean_squared_error(y_test, y_pred),
                "mae": mean_absolute_error(y_test, y_pred),
                "r2": r2_score(y_test, y_pred),
                "cv_mse_mean": -np.mean(cv_scores),
                "cv_mse_std": np.std(cv_scores),
                "selected_features": selected_features,
                "feature_importance": self._get_feature_importance(X_train_selected, y_train),
                "training_size": len(X_train),
                "test_size": len(X_test),
                "engineered_features": list(df.columns)
            }

            # Save model
            self.save_model()
            
            return metrics

        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise

    def _get_feature_importance(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Get feature importance using SHAP values."""
        if self.explainer is None:
            return {}
            
        shap_values = self.explainer.shap_values(X)
        importance = np.abs(shap_values).mean(axis=0)
        return dict(zip(self.feature_names, importance))

    def predict(self, features: List[float], lat: float = None, lon: float = None) -> Dict[str, Any]:
        """Make prediction with enhanced ML analysis."""
        if self.model is None:
            return self._fallback_prediction(features)
        
        try:
            # Prepare features with engineering
            df = self._prepare_features(np.array([features]))
            features_scaled = self.scaler.transform(df)
            
            # Get base prediction
            base_score = round(self.model.predict(features_scaled)[0], 2)
            
            # Calculate SHAP values for explanation
            if self.explainer is not None:
                shap_values = self.explainer.shap_values(features_scaled)
                feature_contributions = dict(zip(
                    df.columns,
                    shap_values[0]
                ))
            else:
                feature_contributions = self._get_feature_contributions(features)
            
            prediction = {
                "base_score": base_score,
                "confidence": self._calculate_prediction_confidence(features_scaled),
                "feature_contributions": feature_contributions,
                "prediction_interval": self._calculate_prediction_interval(features_scaled),
                "engineered_features": list(df.columns)
            }
            
            # Add LAPD data analysis if coordinates are provided
            if lat is not None and lon is not None:
                lapd_metrics = self.lapd_processor.get_location_risk_score(lat, lon)
                prediction["lapd_analysis"] = lapd_metrics
                
                # Adjust score based on LAPD data
                adjusted_score = self._adjust_score_with_lapd(base_score, lapd_metrics)
                prediction["adjusted_score"] = adjusted_score
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            return self._fallback_prediction(features)

    def _calculate_prediction_interval(self, features_scaled: np.ndarray) -> Dict[str, float]:
        """Calculate prediction interval using ensemble predictions."""
        if self.model is None:
            return {"lower": 0.0, "upper": 1.0}
            
        # Get predictions from each model in the ensemble
        predictions = []
        for name, model in self.model.named_estimators_.items():
            pred = model.predict(features_scaled)
            predictions.append(pred)
        
        # Calculate 95% prediction interval
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        return {
            "lower": round(max(0, mean_pred - 1.96 * std_pred), 2),
            "upper": round(min(1, mean_pred + 1.96 * std_pred), 2)
        }

    def _calculate_prediction_confidence(self, features_scaled: np.ndarray) -> float:
        """Calculate confidence in the prediction based on feature values."""
        if self.model is None:
            return 0.0
            
        # Use the standard deviation of predictions from all trees
        predictions = np.array([tree.predict(features_scaled) 
                              for tree in self.model.estimators_])
        std = np.std(predictions)
        confidence = 1.0 / (1.0 + std)  # Convert to 0-1 scale
        return round(confidence, 2)

    def _get_feature_contributions(self, features: List[float]) -> Dict[str, float]:
        """Calculate how much each feature contributes to the prediction."""
        if self.model is None:
            return {}
            
        contributions = {}
        for feature, value in zip(self.feature_names, features):
            # Calculate feature importance * normalized feature value
            importance = self.model.feature_importances_[self.feature_names.index(feature)]
            contributions[feature] = round(importance * value, 3)
            
        return contributions

    def _adjust_score_with_lapd(self, base_score: float, lapd_metrics: Dict[str, Any]) -> float:
        """Adjust the safety score based on LAPD data analysis."""
        # Weight factors for different LAPD metrics
        weights = {
            'crime_density': 0.3,
            'night_risk': 0.2,
            'temporal_risk': 0.3,
            'spatial_risk': 0.2
        }
        
        # Calculate weighted risk adjustment
        risk_adjustment = sum(
            lapd_metrics[metric] * weight 
            for metric, weight in weights.items()
        )
        
        # Adjust the base score
        adjusted_score = base_score * (1 - risk_adjustment)
        return round(max(0, min(1, adjusted_score)), 2)

    def _fallback_prediction(self, features: List[float]) -> Dict[str, Any]:
        """Fallback prediction when model is not available."""
        score = 1.0
        crime_count, lighting_poor, past_reports, bad_weather, terrain_dirt, mud_risk = features
        
        if crime_count > 10: score -= 0.4
        elif crime_count > 5: score -= 0.2
        if lighting_poor: score -= 0.2
        if past_reports: score -= 0.2
        if bad_weather: score -= 0.2
        if terrain_dirt: score -= 0.2
        if mud_risk: score -= 0.2
        
        return {
            "base_score": round(max(score, 0), 2),
            "confidence": 0.5,
            "feature_contributions": dict(zip(self.feature_names, features)),
            "is_fallback": True
        } 
