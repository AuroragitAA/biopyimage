"""
ml_enhancement.py

Machine Learning Enhancement Module for Wolffia Analysis System.
Provides predictive analytics, automated classification, and intelligent insights.

Features:
- Automated cell classification using ensemble methods
- Growth prediction modeling
- Health assessment using ML algorithms
- Anomaly detection and quality control
- Feature importance analysis
- Population dynamics modeling
- Automated parameter optimization
"""

import json
import logging
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# Machine Learning imports
import xgboost as xgb
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    GradientBoostingRegressor,
    IsolationForest,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.feature_selection import RFE, SelectKBest, f_classif
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MLConfig:
    """Configuration for machine learning models."""
    
    # Model parameters
    random_state: int = 42
    test_size: float = 0.2
    cv_folds: int = 5
    
    # Feature selection
    max_features: int = 20
    feature_selection_method: str = 'auto'  # will be resolved intelligently
    
    # Model types to use
    classification_models: List[str] = None
    regression_models: List[str] = None
    
    # Training parameters
    enable_hyperparameter_tuning: bool = True
    max_training_time: int = 900  # seconds
    
    # Output settings
    model_save_path: str = "models"
    confidence_threshold: float = 0.7
    
    def __post_init__(self):
        if self.classification_models is None:
            self.classification_models = ['random_forest', 'svm', 'neural_network', 'xgboost']
        if self.regression_models is None:
            self.regression_models = ['random_forest', 'gradient_boosting', 'xgboost']
        
        # Smart feature selection decision (optional, moved to training time)
        if self.feature_selection_method == 'auto':
            # will be resolved dynamically later in training
            pass


class CellClassificationModel:
    """Advanced cell classification using ensemble methods."""
    
    def __init__(self, config: MLConfig):
        self.config = config
        self.models = {}
        self.feature_scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_selector = None
        self.feature_importance = {}
        self.is_trained = False
        
        # Create models directory
        Path(config.model_save_path).mkdir(exist_ok=True)
        
        logger.info("🤖 Cell Classification Model initialized")
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features for machine learning."""
        try:
            # Select numeric features (exclude IDs and categorical)
            numeric_features = df.select_dtypes(include=[np.number]).columns
            exclude_cols = ['cell_id', 'centroid_x', 'centroid_y', 'batch_index']
            feature_cols = [col for col in numeric_features if col not in exclude_cols]
            
            # Handle missing values
            X = df[feature_cols].fillna(df[feature_cols].mean())
            
            # Create target variable based on multiple criteria
            y = self._create_classification_target(df)
            
            logger.info(f"📊 Features prepared: {X.shape[1]} features, {len(y)} samples")
            return X.values, y
            
        except Exception as e:
            logger.error(f"❌ Feature preparation error: {str(e)}")
            return np.array([]), np.array([])
    
    def _create_classification_target(self, df: pd.DataFrame) -> np.ndarray:
        """Create classification target based on multiple biological criteria."""
        try:
            # Multi-criteria classification
            classifications = []
            
            for _, row in df.iterrows():
                # Size-based classification
                area = row.get('area', 0)
                
                # Health-based features
                health_score = row.get('health_score', 0.5)
                chlorophyll = row.get('chlorophyll_content', 0.5)
                integrity = row.get('cell_integrity', 0.5)
                
                # Classify based on multiple criteria
                if health_score > 0.8 and chlorophyll > 0.7 and area > 200:
                    classification = 'excellent'
                elif health_score > 0.6 and chlorophyll > 0.5 and area > 100:
                    classification = 'good'
                elif health_score > 0.4 or area > 50:
                    classification = 'fair'
                else:
                    classification = 'poor'
                
                classifications.append(classification)
            
            return np.array(classifications)
            
        except Exception as e:
            logger.error(f"❌ Target creation error: {str(e)}")
            return np.array(['unknown'] * len(df))
    
    def train_models(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Train multiple classification models."""
        try:
            logger.info("🎯 Starting model training...")
            
            # Encode labels
            y_encoded = self.label_encoder.fit_transform(y)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=self.config.test_size, 
                random_state=self.config.random_state, stratify=y_encoded
            )
            
            # Scale features
            X_train_scaled = self.feature_scaler.fit_transform(X_train)
            X_test_scaled = self.feature_scaler.transform(X_test)
            
            # Just before feature selection
            if self.config.feature_selection_method == 'auto':
                num_features = X_train.shape[1]
                if num_features > 100:
                    self.config.feature_selection_method = 'pca'
                elif num_features > 30:
                    self.config.feature_selection_method = 'univariate'
                else:
                    self.config.feature_selection_method = 'rfe'

            # Feature selection
            X_train_selected, X_test_selected = self._select_features(X_train_scaled, X_test_scaled, y_train)
            
            # Train models
            model_results = {}
            
            for model_name in self.config.classification_models:
                logger.info(f"📚 Training {model_name}...")
                
                model = self._create_classification_model(model_name)
                
                if self.config.enable_hyperparameter_tuning:
                    model = self._tune_hyperparameters(model, X_train_selected, y_train, model_name)
                
                # Train model
                model.fit(X_train_selected, y_train)
                
                # Evaluate model
                train_score = model.score(X_train_selected, y_train)
                test_score = model.score(X_test_selected, y_test)
                cv_score = cross_val_score(model, X_train_selected, y_train, cv=self.config.cv_folds).mean()
                
                # Predictions for detailed evaluation
                y_pred = model.predict(X_test_selected)
                
                model_results[model_name] = {
                    'model': model,
                    'train_accuracy': train_score,
                    'test_accuracy': test_score,
                    'cv_accuracy': cv_score,
                    'classification_report': classification_report(y_test, y_pred, output_dict=True),
                    'feature_importance': self._get_feature_importance(model, model_name)
                }
                
                self.models[model_name] = model
                
                logger.info(f"✅ {model_name}: CV Score = {cv_score:.3f}")
            
            # Select best model
            best_model_name = max(model_results.keys(), key=lambda k: model_results[k]['cv_accuracy'])
            self.best_model = self.models[best_model_name]
            self.best_model_name = best_model_name
            
            self.is_trained = True
            
            # Save models
            self._save_models()
            
            logger.info(f"🏆 Best model: {best_model_name} (CV Score: {model_results[best_model_name]['cv_accuracy']:.3f})")
            
            return {
                'best_model': best_model_name,
                'model_results': model_results,
                'feature_importance': self._aggregate_feature_importance(),
                'training_summary': {
                    'total_samples': len(X),
                    'features_selected': X_train_selected.shape[1],
                    'classes': list(self.label_encoder.classes_)
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Model training error: {str(e)}")
            return {'error': str(e)}
    
    def _create_classification_model(self, model_name: str):
        """Create classification model by name."""
        models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100, 
                random_state=self.config.random_state
            ),
            'svm': SVC(
                kernel='rbf', 
                probability=True, 
                random_state=self.config.random_state
            ),
            'neural_network': MLPClassifier(
                hidden_layer_sizes=(100, 50), 
                max_iter=500, 
                random_state=self.config.random_state
            ),
            'xgboost': xgb.XGBClassifier(
                random_state=self.config.random_state,
                eval_metric='logloss'
            ),
            'logistic_regression': LogisticRegression(
                random_state=self.config.random_state,
                max_iter=1000
            )
        }
        
        return models.get(model_name, models['random_forest'])
    
    def _select_features(self, X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Select most important features."""
        try:
            if self.config.feature_selection_method == 'rfe':
                estimator = RandomForestClassifier(n_estimators=50, random_state=self.config.random_state)
                selector = RFE(estimator, n_features_to_select=min(self.config.max_features, X_train.shape[1]))
            elif self.config.feature_selection_method == 'univariate':
                selector = SelectKBest(score_func=f_classif, k=min(self.config.max_features, X_train.shape[1]))
            else:  # PCA
                selector = PCA(n_components=min(self.config.max_features, X_train.shape[1]))
            
            self.feature_selector = selector
            X_train_selected = selector.fit_transform(X_train, y_train)
            X_test_selected = selector.transform(X_test)
            
            logger.info(f"🎯 Feature selection: {X_train.shape[1]} → {X_train_selected.shape[1]} features")
            
            return X_train_selected, X_test_selected
            
        except Exception as e:
            logger.error(f"❌ Feature selection error: {str(e)}")
            return X_train, X_test
    
    def _tune_hyperparameters(self, model, X_train: np.ndarray, y_train: np.ndarray, model_name: str):
        """Tune hyperparameters using GridSearchCV."""
        try:
            param_grids = {
                'random_forest': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5]
                },
                'svm': {
                    'C': [0.1, 1, 10],
                    'gamma': ['scale', 'auto']
                },
                'neural_network': {
                    'hidden_layer_sizes': [(50,), (100,), (100, 50)],
                    'alpha': [0.0001, 0.001, 0.01]
                },
                'xgboost': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.1, 0.2]
                }
            }
            
            if model_name in param_grids:
                grid_search = GridSearchCV(
                    model, 
                    param_grids[model_name], 
                    cv=3,  # Reduced for speed
                    scoring='accuracy',
                    n_jobs=-1
                )
                grid_search.fit(X_train, y_train)
                return grid_search.best_estimator_
            
            return model
            
        except Exception as e:
            logger.error(f"❌ Hyperparameter tuning error: {str(e)}")
            return model
    
    def _get_feature_importance(self, model, model_name: str) -> Dict:
        """Extract feature importance from model."""
        try:
            if hasattr(model, 'feature_importances_'):
                return {'importance_scores': model.feature_importances_.tolist()}
            elif hasattr(model, 'coef_'):
                return {'coefficients': np.abs(model.coef_[0]).tolist()}
            else:
                return {'message': 'Feature importance not available for this model'}
                
        except Exception as e:
            logger.error(f"❌ Feature importance extraction error: {str(e)}")
            return {}
    
    def _aggregate_feature_importance(self) -> Dict:
        """Aggregate feature importance across all models."""
        try:
            importance_scores = []
            
            for model_name, model in self.models.items():
                if hasattr(model, 'feature_importances_'):
                    importance_scores.append(model.feature_importances_)
            
            if importance_scores:
                avg_importance = np.mean(importance_scores, axis=0)
                return {
                    'average_importance': avg_importance.tolist(),
                    'top_features': np.argsort(avg_importance)[::-1][:10].tolist()
                }
            
            return {'message': 'No feature importance available'}
            
        except Exception as e:
            logger.error(f"❌ Feature importance aggregation error: {str(e)}")
            return {}
    
    def predict_classification(self, X: np.ndarray) -> Dict:
        """Predict classifications with confidence scores."""
        try:
            if not self.is_trained:
                return {'error': 'Models not trained yet'}
            
            # Preprocess features
            X_scaled = self.feature_scaler.transform(X)
            X_selected = self.feature_selector.transform(X_scaled)
            
            # Get predictions from best model
            predictions = self.best_model.predict(X_selected)
            prediction_proba = self.best_model.predict_proba(X_selected)
            
            # Convert back to original labels
            predicted_labels = self.label_encoder.inverse_transform(predictions)
            
            # Calculate confidence scores
            confidence_scores = np.max(prediction_proba, axis=1)
            
            results = {
                'predictions': predicted_labels.tolist(),
                'confidence_scores': confidence_scores.tolist(),
                'high_confidence_predictions': (confidence_scores > self.config.confidence_threshold).sum(),
                'model_used': self.best_model_name,
                'class_probabilities': prediction_proba.tolist()
            }
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Prediction error: {str(e)}")
            return {'error': str(e)}
    
    def _save_models(self):
        """Save trained models to disk."""
        try:
            model_dir = Path(self.config.model_save_path)
            
            # Save best model
            joblib.dump(self.best_model, model_dir / 'best_classification_model.joblib')
            joblib.dump(self.feature_scaler, model_dir / 'feature_scaler.joblib')
            joblib.dump(self.label_encoder, model_dir / 'label_encoder.joblib')
            joblib.dump(self.feature_selector, model_dir / 'feature_selector.joblib')
            
            # Save metadata
            metadata = {
                'best_model_name': self.best_model_name,
                'classes': self.label_encoder.classes_.tolist(),
                'training_date': datetime.now().isoformat(),
                'config': self.config.__dict__
            }
            
            with open(model_dir / 'classification_metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info("💾 Classification models saved successfully")
            
        except Exception as e:
            logger.error(f"❌ Model saving error: {str(e)}")


class GrowthPredictionModel:
    """Predict growth patterns and population dynamics."""
    
    def __init__(self, config: MLConfig):
        self.config = config
        self.models = {}
        self.feature_scaler = StandardScaler()
        self.is_trained = False
        
        logger.info("📈 Growth Prediction Model initialized")
    
    def prepare_temporal_features(self, historical_data: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare temporal features for growth prediction."""
        try:
            # Sort data by timestamp
            sorted_data = sorted(historical_data, key=lambda x: x.get('timestamp', ''))
            
            # Extract time-series features
            features = []
            targets = []
            
            for i in range(len(sorted_data) - 1):
                current = sorted_data[i]
                next_point = sorted_data[i + 1]
                
                # Current state features
                current_features = [
                    current.get('total_cells', 0),
                    current.get('avg_area', 0),
                    current.get('chlorophyll_ratio', 0),
                    current.get('health_score', 0.5)
                ]
                
                # Environmental features (if available)
                if 'environmental_data' in current:
                    env_data = current['environmental_data']
                    current_features.extend([
                        env_data.get('temperature', 25),
                        env_data.get('light_intensity', 50),
                        env_data.get('nutrients', 50)
                    ])
                
                # Time-based features
                current_time = datetime.fromisoformat(current.get('timestamp', datetime.now().isoformat()))
                hour_of_day = current_time.hour
                day_of_week = current_time.weekday()
                
                current_features.extend([hour_of_day, day_of_week])
                
                features.append(current_features)
                
                # Target: growth rate (change in cell count)
                growth_rate = (next_point.get('total_cells', 0) - current.get('total_cells', 0)) / max(current.get('total_cells', 1), 1)
                targets.append(growth_rate)
            
            return np.array(features), np.array(targets)
            
        except Exception as e:
            logger.error(f"❌ Temporal feature preparation error: {str(e)}")
            return np.array([]), np.array([])
    
    def train_growth_models(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Train growth prediction models."""
        try:
            logger.info("📊 Training growth prediction models...")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.config.test_size, random_state=self.config.random_state
            )
            
            # Scale features
            X_train_scaled = self.feature_scaler.fit_transform(X_train)
            X_test_scaled = self.feature_scaler.transform(X_test)
            
            # Train multiple models
            model_results = {}
            
            for model_name in self.config.regression_models:
                logger.info(f"🎯 Training {model_name} for growth prediction...")
                
                model = self._create_regression_model(model_name)
                model.fit(X_train_scaled, y_train)
                
                # Evaluate model
                train_score = model.score(X_train_scaled, y_train)
                test_score = model.score(X_test_scaled, y_test)
                
                y_pred = model.predict(X_test_scaled)
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                model_results[model_name] = {
                    'model': model,
                    'train_r2': train_score,
                    'test_r2': test_score,
                    'mse': mse,
                    'r2_score': r2
                }
                
                self.models[model_name] = model
                
                logger.info(f"✅ {model_name}: R² = {r2:.3f}, MSE = {mse:.4f}")
            
            # Select best model
            best_model_name = max(model_results.keys(), key=lambda k: model_results[k]['r2_score'])
            self.best_model = self.models[best_model_name]
            
            self.is_trained = True
            
            return {
                'best_model': best_model_name,
                'model_results': model_results,
                'training_summary': {
                    'total_samples': len(X),
                    'features': X.shape[1],
                    'best_r2_score': model_results[best_model_name]['r2_score']
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Growth model training error: {str(e)}")
            return {'error': str(e)}
    
    def _create_regression_model(self, model_name: str):
        """Create regression model by name."""
        models = {
            'random_forest': RandomForestRegressor(
                n_estimators=100, 
                random_state=self.config.random_state
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100, 
                random_state=self.config.random_state
            ),
            'xgboost': xgb.XGBRegressor(
                random_state=self.config.random_state
            ),
            'linear_regression': LinearRegression()
        }
        
        return models.get(model_name, models['random_forest'])
    
    def predict_growth(self, current_features: np.ndarray, time_horizon: int = 24) -> Dict:
        """Predict growth over specified time horizon."""
        try:
            if not self.is_trained:
                return {'error': 'Growth models not trained yet'}
            
            # Scale features
            current_scaled = self.feature_scaler.transform(current_features.reshape(1, -1))
            
            # Predict growth rate
            growth_rate = self.best_model.predict(current_scaled)[0]
            
            # Calculate predictions for time horizon
            predictions = []
            current_cell_count = current_features[0]  # Assuming first feature is cell count
            
            for hour in range(1, time_horizon + 1):
                predicted_count = current_cell_count * (1 + growth_rate) ** hour
                predictions.append({
                    'hour': hour,
                    'predicted_cell_count': max(0, predicted_count),
                    'growth_rate': growth_rate
                })
            
            return {
                'growth_rate_per_hour': growth_rate,
                'predictions': predictions,
                'confidence': 'medium',  # Would need prediction intervals for proper confidence
                'model_used': type(self.best_model).__name__
            }
            
        except Exception as e:
            logger.error(f"❌ Growth prediction error: {str(e)}")
            return {'error': str(e)}


class AnomalyDetector:
    """Detect anomalies in cell populations and analysis results."""
    
    def __init__(self, config: MLConfig):
        self.config = config
        self.isolation_forest = IsolationForest(
            contamination=0.1, 
            random_state=config.random_state
        )
        self.clustering_model = None
        self.is_fitted = False
        
        logger.info("🔍 Anomaly Detector initialized")
    
    def fit_anomaly_detection(self, X: np.ndarray) -> Dict:
        """Fit anomaly detection models."""
        try:
            logger.info("🎯 Training anomaly detection models...")
            
            # Fit Isolation Forest
            self.isolation_forest.fit(X)
            
            # Fit clustering for anomaly detection
            optimal_clusters = self._find_optimal_clusters(X)
            self.clustering_model = KMeans(n_clusters=optimal_clusters, random_state=self.config.random_state)
            self.clustering_model.fit(X)
            
            self.is_fitted = True
            
            return {
                'isolation_forest_fitted': True,
                'optimal_clusters': optimal_clusters,
                'training_samples': X.shape[0]
            }
            
        except Exception as e:
            logger.error(f"❌ Anomaly detection training error: {str(e)}")
            return {'error': str(e)}
    
    def detect_anomalies(self, X: np.ndarray) -> Dict:
        """Detect anomalies in new data."""
        try:
            if not self.is_fitted:
                return {'error': 'Anomaly detector not fitted yet'}
            
            # Isolation Forest anomalies
            isolation_anomalies = self.isolation_forest.predict(X) == -1
            isolation_scores = self.isolation_forest.decision_function(X)
            
            # Clustering-based anomalies (distance from cluster centers)
            cluster_labels = self.clustering_model.predict(X)
            cluster_distances = []
            
            for i, (point, label) in enumerate(zip(X, cluster_labels)):
                center = self.clustering_model.cluster_centers_[label]
                distance = np.linalg.norm(point - center)
                cluster_distances.append(distance)
            
            cluster_distances = np.array(cluster_distances)
            distance_threshold = np.percentile(cluster_distances, 95)  # Top 5% as anomalies
            clustering_anomalies = cluster_distances > distance_threshold
            
            # Combined anomaly detection
            combined_anomalies = isolation_anomalies | clustering_anomalies
            
            return {
                'total_samples': len(X),
                'isolation_anomalies': int(isolation_anomalies.sum()),
                'clustering_anomalies': int(clustering_anomalies.sum()),
                'combined_anomalies': int(combined_anomalies.sum()),
                'anomaly_indices': np.where(combined_anomalies)[0].tolist(),
                'anomaly_scores': isolation_scores.tolist(),
                'cluster_distances': cluster_distances.tolist()
            }
            
        except Exception as e:
            logger.error(f"❌ Anomaly detection error: {str(e)}")
            return {'error': str(e)}
    
    def _find_optimal_clusters(self, X: np.ndarray) -> int:
        """Find optimal number of clusters using elbow method."""
        try:
            if len(X) < 10:
                return 2
            
            max_clusters = min(10, len(X) // 2)
            inertias = []
            
            for k in range(2, max_clusters + 1):
                kmeans = KMeans(n_clusters=k, random_state=self.config.random_state)
                kmeans.fit(X)
                inertias.append(kmeans.inertia_)
            
            # Simple elbow detection (could be improved)
            if len(inertias) >= 3:
                # Find the point with maximum second derivative
                second_derivatives = []
                for i in range(1, len(inertias) - 1):
                    second_deriv = inertias[i-1] - 2*inertias[i] + inertias[i+1]
                    second_derivatives.append(second_deriv)
                
                optimal_k = np.argmax(second_derivatives) + 3  # +3 because we start from k=2 and skip first/last
                return min(optimal_k, max_clusters)
            
            return 3  # Default
            
        except Exception as e:
            logger.error(f"❌ Optimal clusters detection error: {str(e)}")
            return 3


class MLEnhancedAnalyzer:
    """Main ML-enhanced analyzer that integrates all ML components."""
    
    def __init__(self, base_analyzer, config: MLConfig = None):
        """Initialize ML-enhanced analyzer."""
        self.base_analyzer = base_analyzer
        self.config = config or MLConfig()
        
        # Initialize ML components
        self.classification_model = CellClassificationModel(self.config)
        self.growth_model = GrowthPredictionModel(self.config)
        self.anomaly_detector = AnomalyDetector(self.config)
        
        # Training data storage
        self.training_data = []
        self.models_trained = False
        
        logger.info("🤖 ML-Enhanced Analyzer initialized")
    
    def analyze_with_ml_enhancement(self, image_path: str) -> Dict:
        """Analyze image with ML enhancements."""
        try:
            # Get base analysis
            base_result = self.base_analyzer.analyze_image_professional(image_path)
            
            if not base_result.get('success'):
                return base_result
            
            # Add ML enhancements if models are trained
            if self.models_trained and 'cell_data' in base_result:
                ml_enhancements = self._apply_ml_enhancements(base_result['cell_data'])
                base_result['ml_enhancements'] = ml_enhancements
            
            # Store data for future training
            self._store_training_data(base_result)
            
            return base_result
            
        except Exception as e:
            logger.error(f"❌ ML-enhanced analysis error: {str(e)}")
            return {'error': str(e), 'success': False}
    
    def train_ml_models(self) -> Dict:
        """Train all ML models using accumulated data."""
        try:
            if len(self.training_data) < 50:
                return {'error': 'Insufficient training data (minimum 50 samples required)'}
            
            logger.info(f"🎓 Training ML models with {len(self.training_data)} samples...")
            
            # Prepare consolidated dataset
            all_cell_data = []
            for result in self.training_data:
                if 'cell_data' in result:
                    all_cell_data.extend(result['cell_data'])
            
            if len(all_cell_data) < 100:
                return {'error': 'Insufficient cell-level training data'}
            
            df = pd.DataFrame(all_cell_data)
            
            # Train classification model
            X_class, y_class = self.classification_model.prepare_features(df)
            if len(X_class) > 0:
                classification_results = self.classification_model.train_models(X_class, y_class)
            else:
                classification_results = {'error': 'Classification feature preparation failed'}
            
            # Train growth model (if temporal data available)
            growth_results = {'message': 'Temporal data insufficient for growth modeling'}
            if len(self.training_data) > 10:
                X_growth, y_growth = self.growth_model.prepare_temporal_features(self.training_data)
                if len(X_growth) > 10:
                    growth_results = self.growth_model.train_growth_models(X_growth, y_growth)
            
            # Train anomaly detector
            anomaly_results = {'error': 'Anomaly detection training failed'}
            if len(X_class) > 0:
                anomaly_results = self.anomaly_detector.fit_anomaly_detection(X_class)
            
            self.models_trained = True
            
            training_summary = {
                'training_timestamp': datetime.now().isoformat(),
                'total_training_samples': len(all_cell_data),
                'classification_results': classification_results,
                'growth_results': growth_results,
                'anomaly_results': anomaly_results,
                'models_trained': self.models_trained
            }
            
            logger.info("🎉 ML model training completed!")
            
            return training_summary
            
        except Exception as e:
            logger.error(f"❌ ML model training error: {str(e)}")
            return {'error': str(e)}
    
    def _apply_ml_enhancements(self, cell_data: List[Dict]) -> Dict:
        """Apply ML enhancements to cell data."""
        try:
            enhancements = {}
            
            # Convert to DataFrame for ML processing
            df = pd.DataFrame(cell_data)
            
            # Classification enhancements
            if self.classification_model.is_trained:
                X_class, _ = self.classification_model.prepare_features(df)
                if len(X_class) > 0:
                    classification_results = self.classification_model.predict_classification(X_class)
                    enhancements['ml_classifications'] = classification_results
            
            # Anomaly detection
            if self.anomaly_detector.is_fitted:
                X_class, _ = self.classification_model.prepare_features(df)
                if len(X_class) > 0:
                    anomaly_results = self.anomaly_detector.detect_anomalies(X_class)
                    enhancements['anomaly_detection'] = anomaly_results
            
            # Feature importance insights
            if hasattr(self.classification_model, 'feature_importance'):
                enhancements['feature_insights'] = self.classification_model.feature_importance
            
            return enhancements
            
        except Exception as e:
            logger.error(f"❌ ML enhancement application error: {str(e)}")
            return {'error': str(e)}
    
    def _store_training_data(self, result: Dict):
        """Store analysis results for future model training."""
        try:
            # Add timestamp if not present
            if 'timestamp' not in result:
                result['timestamp'] = datetime.now().isoformat()
            
            # Store with size limit
            self.training_data.append(result)
            
            # Keep only recent data (last 1000 samples)
            if len(self.training_data) > 1000:
                self.training_data = self.training_data[-1000:]
            
        except Exception as e:
            logger.error(f"❌ Training data storage error: {str(e)}")
    
    def get_ml_insights(self) -> Dict:
        """Get insights from trained ML models."""
        try:
            if not self.models_trained:
                return {'message': 'ML models not trained yet'}
            
            insights = {
                'model_status': {
                    'classification_trained': self.classification_model.is_trained,
                    'growth_trained': self.growth_model.is_trained,
                    'anomaly_trained': self.anomaly_detector.is_fitted
                },
                'training_data_size': len(self.training_data),
                'last_training': 'Recently trained'
            }
            
            # Add feature importance if available
            if hasattr(self.classification_model, 'feature_importance'):
                insights['feature_importance'] = self.classification_model.feature_importance
            
            return insights
            
        except Exception as e:
            logger.error(f"❌ ML insights error: {str(e)}")
            return {'error': str(e)}
    
    def predict_population_health(self, current_data: Dict) -> Dict:
        """Predict population health trends."""
        try:
            if not self.models_trained:
                return {'error': 'Models not trained yet'}
            
            # Extract current features
            summary = current_data.get('summary', {})
            current_features = np.array([
                summary.get('total_cells', 0),
                summary.get('avg_area', 0),
                summary.get('chlorophyll_ratio', 0),
                summary.get('health_score', 0.5)
            ])
            
            # Predict growth
            growth_prediction = self.growth_model.predict_growth(current_features)
            
            # Health trend analysis
            health_trend = 'stable'
            if growth_prediction.get('growth_rate_per_hour', 0) > 0.05:
                health_trend = 'improving'
            elif growth_prediction.get('growth_rate_per_hour', 0) < -0.02:
                health_trend = 'declining'
            
            return {
                'health_trend': health_trend,
                'growth_prediction': growth_prediction,
                'confidence': 'medium',
                'recommendations': self._generate_ml_recommendations(current_data, growth_prediction)
            }
            
        except Exception as e:
            logger.error(f"❌ Population health prediction error: {str(e)}")
            return {'error': str(e)}
    
    def _generate_ml_recommendations(self, current_data: Dict, growth_prediction: Dict) -> List[str]:
        """Generate ML-based recommendations."""
        try:
            recommendations = []
            
            growth_rate = growth_prediction.get('growth_rate_per_hour', 0)
            
            if growth_rate > 0.1:
                recommendations.append("🚀 Excellent growth rate detected. Current conditions are optimal.")
            elif growth_rate > 0.02:
                recommendations.append("📈 Moderate growth observed. Consider maintaining current conditions.")
            elif growth_rate < -0.05:
                recommendations.append("⚠️ Population decline detected. Review environmental conditions immediately.")
            else:
                recommendations.append("📊 Stable population. Monitor for trend changes.")
            
            # Anomaly-based recommendations
            if 'ml_enhancements' in current_data and 'anomaly_detection' in current_data['ml_enhancements']:
                anomaly_info = current_data['ml_enhancements']['anomaly_detection']
                anomaly_rate = anomaly_info.get('combined_anomalies', 0) / anomaly_info.get('total_samples', 1)
                
                if anomaly_rate > 0.2:
                    recommendations.append("🔍 High anomaly rate detected. Check for contamination or processing errors.")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"❌ ML recommendation generation error: {str(e)}")
            return ["ML analysis complete. Monitor population trends."]


            # Add these methods to the MLEnhancedAnalyzer class in ml_enhancement.py

    def analyze_single_image(self, image_path: str, **kwargs) -> Dict:
        """
        Analyze a single image using the base analyzer with ML enhancements.
        Passes analysis parameters such as pixel_ratio, min_cell_area, etc. via kwargs.
        Compatible with both standard and professional analyzers.
        """
        try:
            # Step 1: Delegate to base analyzer using flexible method signature
            if hasattr(self.base_analyzer, 'analyze_single_image'):
                base_result = self.base_analyzer.analyze_single_image(image_path, **kwargs)
            elif hasattr(self.base_analyzer, 'analyze_image_professional'):
                base_result = self.base_analyzer.analyze_image_professional(image_path, **kwargs)
            elif hasattr(self.base_analyzer, 'analyze_image'):
                base_result = self.base_analyzer.analyze_image(image_path)  # May not support kwargs
            else:
                return {
                    'success': False,
                    'error': 'No valid analysis method found in base analyzer.',
                    'timestamp': datetime.now().isoformat()
                }

            # Step 2: Check for success before continuing
            if not base_result or not base_result.get('success', False):
                return base_result

            # Step 3: Inject ML enhancements if model is trained and cell data exists
            if self.models_trained and 'cell_data' in base_result:
                try:
                    ml_enhancements = self._apply_ml_enhancements(base_result['cell_data'])
                    base_result['ml_enhancements'] = ml_enhancements
                except Exception as ml_error:
                    logger.warning(f"ML enhancement failed: {str(ml_error)}")
                    base_result['ml_enhancements'] = {'error': str(ml_error)}

            # Step 4: Store data for incremental training
            self._store_training_data(base_result)

            return base_result

        except Exception as e:
            logger.error(f"❌ ML-enhanced single image analysis error: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def analyze_with_ml_enhancement(self, image_path: str, **kwargs) -> Dict:
        """
        Analyze image with ML enhancements - alias for analyze_single_image
        """
        return self.analyze_single_image(image_path, **kwargs)

    def classify_cells(self, result):
        if not result.get('success'):
            return result

        # Add classification if available
        if 'cell_data' in result and hasattr(self, 'classification_model'):
            df = pd.DataFrame(result['cell_data'])
            X, _ = self.classification_model.prepare_features(df)
            predictions = self.classification_model.predict_classification(X)
            df['classification'] = predictions
            result['cell_data'] = df.to_dict(orient='records')

        return result

    
    def analyze_image_professional(self, image_path: str, **kwargs) -> Dict:
        """
        Professional analysis with ML enhancements.
        Delegates to analyze_with_ml_enhancement for consistency.
        """
        return self.analyze_with_ml_enhancement(image_path, **kwargs)
    
    def batch_analyze_images(self, image_paths: List[str], progress_callback=None, **kwargs) -> List[Dict]:
        """
        Batch analyze multiple images with ML enhancements.
        """
        try:
            results = []
            total = len(image_paths)
            
            for i, image_path in enumerate(image_paths):
                # Analyze with ML enhancements
                result = self.analyze_single_image(image_path, **kwargs)
                results.append(result)
                
                # Progress callback
                if progress_callback:
                    progress_callback(i + 1, total, result)
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Batch ML analysis error: {str(e)}")
            return []


# Example usage and testing
if __name__ == "__main__":
    print("🤖 Testing ML Enhancement Module...")
    
    try:
        # Create test configuration
        config = MLConfig()
        
        # Test individual components
        print("✅ ML Configuration created")
        print("🔬 ML Enhancement Module ready for integration")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()