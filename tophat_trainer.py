#!/usr/bin/env python3
"""
TOPHAT ML TRAINER - Professional Implementation
Streamlined ML training based on proven Random Forest patterns from python_for_microscopists
Author: BIOIMAGIN Professional Team
"""

import json
import pickle
import warnings
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage
from skimage import filters, measure
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')


class TophatMLTrainer:
    """
    Professional ML trainer for Wolffia detection
    Based on proven Random Forest patterns from python_for_microscopists (examples 060, 062-066)
    """
    
    def __init__(self):
        self.model = None
        self.feature_names = []
        self.setup_directories()
        print("✅ Tophat ML Trainer initialized")
    
    def setup_directories(self):
        """Setup required directories"""
        self.dirs = {
            'models': Path('models'),
            'annotations': Path('annotations'),
            'tophat_training': Path('tophat_training')
        }
        
        for path in self.dirs.values():
            path.mkdir(exist_ok=True)
    
    def extract_features(self, image):
        """
        Extract comprehensive features using proven microscopist approach
        Based on example 062-066 feature extraction methodology
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Initialize feature list
        features = []
        feature_names = []
        
        # Original image intensity
        features.append(image.flatten())
        feature_names.append('Original')
        
        # Gaussian filters with different sigma values (proven approach)
        for sigma in [1, 3, 5]:
            gaussian_img = ndimage.gaussian_filter(image, sigma=sigma)
            features.append(gaussian_img.flatten())
            feature_names.append(f'Gaussian_s{sigma}')
        
        # Edge detection filters (comprehensive set)
        # Canny edge
        edges_canny = cv2.Canny(image, 50, 150)
        features.append(edges_canny.flatten())
        feature_names.append('Canny')
        
        # Sobel edge
        edges_sobel = filters.sobel(image)
        features.append((edges_sobel * 255).astype(np.uint8).flatten())
        feature_names.append('Sobel')
        
        # Roberts edge
        edges_roberts = filters.roberts(image)
        features.append((edges_roberts * 255).astype(np.uint8).flatten())
        feature_names.append('Roberts')
        
        # Prewitt edge
        edges_prewitt = filters.prewitt(image)
        features.append((edges_prewitt * 255).astype(np.uint8).flatten())
        feature_names.append('Prewitt')
        
        # Scharr edge
        edges_scharr = filters.scharr(image)
        features.append((edges_scharr * 255).astype(np.uint8).flatten())
        feature_names.append('Scharr')
        
        # Median filter
        median_img = ndimage.median_filter(image, size=3)
        features.append(median_img.flatten())
        feature_names.append('Median')
        
        # Variance filter (important for texture)
        variance_img = ndimage.generic_filter(image, np.var, size=3)
        features.append(variance_img.flatten())
        feature_names.append('Variance')
        
        # Maximum filter
        maximum_img = ndimage.maximum_filter(image, size=3)
        features.append(maximum_img.flatten())
        feature_names.append('Maximum')
        
        # Minimum filter
        minimum_img = ndimage.minimum_filter(image, size=3)
        features.append(minimum_img.flatten())
        feature_names.append('Minimum')
        
        # Laplacian filter
        laplacian_img = cv2.Laplacian(image, cv2.CV_64F)
        features.append(np.abs(laplacian_img).astype(np.uint8).flatten())
        feature_names.append('Laplacian')
        
        self.feature_names = feature_names
        return np.column_stack(features)
    
    def load_annotation_data(self):
        """
        Load training data from annotation sessions
        """
        training_data = []
        
        # Load from annotation files
        annotation_files = list(self.dirs['annotations'].glob('*.json'))
        
        for annotation_file in annotation_files:
            try:
                with open(annotation_file, 'r') as f:
                    annotation_data = json.load(f)
                
                # Get corresponding image
                image_name = annotation_file.stem.replace('_drawing', '_annotated')
                image_path = annotation_file.parent / f"{image_name}.png"
                
                if image_path.exists():
                    image = cv2.imread(str(image_path))
                    
                    # Create training labels from annotations
                    labels = self.create_labels_from_annotations(image, annotation_data)
                    
                    training_data.append({
                        'image': image,
                        'labels': labels,
                        'annotation_file': str(annotation_file)
                    })
                    
            except Exception as e:
                print(f"⚠️ Failed to load annotation {annotation_file}: {e}")
        
        # Load from tophat training sessions
        session_files = list(self.dirs['tophat_training'].glob('session_*.json'))
        
        for session_file in session_files:
            try:
                with open(session_file, 'r') as f:
                    session_data = json.load(f)
                
                for image_data in session_data.get('images', []):
                    image_path = image_data.get('image_path')
                    if image_path and Path(image_path).exists():
                        image = cv2.imread(image_path)
                        
                        # Create labels from user corrections
                        labels = self.create_labels_from_corrections(image, image_data)
                        
                        training_data.append({
                            'image': image,
                            'labels': labels,
                            'session_file': str(session_file)
                        })
                        
            except Exception as e:
                print(f"⚠️ Failed to load session {session_file}: {e}")
        
        return training_data
    
    def create_labels_from_annotations(self, image, annotation_data):
        """Create training labels from annotation data"""
        labels = np.zeros(image.shape[:2], dtype=np.uint8)
        
        # Extract cell positions from annotation data
        for shape in annotation_data.get('shapes', []):
            if shape.get('shape_type') == 'circle':
                points = shape.get('points', [])
                if len(points) >= 2:
                    center = points[0]
                    edge = points[1]
                    radius = int(np.sqrt((center[0] - edge[0])**2 + (center[1] - edge[1])**2))
                    
                    # Create circular mask
                    y, x = np.ogrid[:image.shape[0], :image.shape[1]]
                    mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
                    labels[mask] = 1
        
        return labels
    
    def create_labels_from_corrections(self, image, image_data):
        """Create training labels from user corrections"""
        labels = np.zeros(image.shape[:2], dtype=np.uint8)
        
        # Positive examples (correct cells)
        for correction in image_data.get('corrections', []):
            if correction.get('type') == 'correct':
                center = correction.get('center', [0, 0])
                radius = correction.get('radius', 10)
                
                # Create circular mask
                y, x = np.ogrid[:image.shape[0], :image.shape[1]]
                mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
                labels[mask] = 1
        
        return labels
    
    def prepare_training_data(self, training_data):
        """
        Prepare training data in the format expected by scikit-learn
        Following the proven approach from microscopist examples
        """
        X_list = []
        y_list = []
        
        print(f"🔄 Processing {len(training_data)} training images...")
        
        for i, data in enumerate(training_data):
            print(f"Processing image {i+1}/{len(training_data)}")
            
            image = data['image']
            labels = data['labels']
            
            # Extract features
            features = self.extract_features(image)
            
            # Flatten labels
            labels_flat = labels.flatten()
            
            # Balance the dataset (important for good ML performance)
            positive_indices = np.where(labels_flat == 1)[0]
            negative_indices = np.where(labels_flat == 0)[0]
            
            # Sample equal number of positive and negative examples
            n_positive = len(positive_indices)
            if n_positive > 0 and len(negative_indices) > n_positive:
                # Randomly sample negative examples
                negative_sample = np.random.choice(negative_indices, n_positive, replace=False)
                selected_indices = np.concatenate([positive_indices, negative_sample])
            else:
                selected_indices = np.concatenate([positive_indices, negative_indices])
            
            if len(selected_indices) > 0:
                X_list.append(features[selected_indices])
                y_list.append(labels_flat[selected_indices])
        
        if not X_list:
            raise ValueError("No training data available")
        
        # Combine all training data
        X = np.vstack(X_list)
        y = np.hstack(y_list)
        
        print(f"✅ Training data prepared: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"Positive samples: {np.sum(y)}, Negative samples: {len(y) - np.sum(y)}")
        
        return X, y
    
    def train_model(self, n_estimators=100, test_size=0.2, random_state=42):
        """
        Train Random Forest model using proven microscopist approach
        Based on examples 060 and 062-066
        """
        try:
            # Load annotation data
            training_data = self.load_annotation_data()
            
            if not training_data:
                print("❌ No training data found. Please create annotations first.")
                return False
            
            # Prepare training data
            X, y = self.prepare_training_data(training_data)
            
            # Split data for validation (proven approach)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            
            print(f"🔄 Training Random Forest with {n_estimators} estimators...")
            
            # Train Random Forest model (proven approach)
            self.model = RandomForestClassifier(
                n_estimators=n_estimators,
                random_state=random_state,
                n_jobs=-1,  # Use all available cores
                class_weight='balanced'  # Handle class imbalance
            )
            
            self.model.fit(X_train, y_train)
            
            # Evaluate model
            train_pred = self.model.predict(X_train)
            test_pred = self.model.predict(X_test)
            
            train_accuracy = accuracy_score(y_train, train_pred)
            test_accuracy = accuracy_score(y_test, test_pred)
            
            print(f"✅ Training completed:")
            print(f"   Training accuracy: {train_accuracy:.3f}")
            print(f"   Test accuracy: {test_accuracy:.3f}")
            
            # Feature importance analysis (microscopist approach)
            if self.feature_names:
                feature_importance = pd.Series(
                    self.model.feature_importances_, 
                    index=self.feature_names
                ).sort_values(ascending=False)
                
                print("\n🔍 Top 5 Feature Importances:")
                for feature, importance in feature_importance.head().items():
                    print(f"   {feature}: {importance:.3f}")
            
            # Save model and metadata
            self.save_model(train_accuracy, test_accuracy, len(training_data))
            
            return True
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            return False
    
    def save_model(self, train_accuracy, test_accuracy, num_training_images):
        """Save trained model and metadata"""
        try:
            # Save model
            model_path = self.dirs['models'] / 'tophat_model.pkl'
            with open(model_path, 'wb') as f:
                pickle.dump(self.model, f)
            
            # Save model info
            model_info = {
                'model_type': 'RandomForestClassifier',
                'n_estimators': self.model.n_estimators,
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
                'num_training_images': num_training_images,
                'feature_names': self.feature_names,
                'timestamp': datetime.now().isoformat()
            }
            
            info_path = self.dirs['models'] / 'tophat_model_info.json'
            with open(info_path, 'w') as f:
                json.dump(model_info, f, indent=2)
            
            print(f"✅ Model saved successfully:")
            print(f"   Model: {model_path}")
            print(f"   Info: {info_path}")
            
        except Exception as e:
            print(f"❌ Failed to save model: {e}")
    
    def predict(self, image):
        """Predict using trained model"""
        if self.model is None:
            print("⚠️ No model loaded")
            return np.zeros(image.shape[:2])
        
        try:
            # Extract features
            features = self.extract_features(image)
            
            # Predict
            predictions = self.model.predict(features)
            
            # Reshape back to image dimensions
            result = predictions.reshape(image.shape[:2])
            
            return result.astype(np.uint8)
            
        except Exception as e:
            print(f"❌ Prediction failed: {e}")
            return np.zeros(image.shape[:2])
    
    def load_model(self):
        """Load saved model"""
        try:
            model_path = self.dirs['models'] / 'tophat_model.pkl'
            
            if not model_path.exists():
                print(f"⚠️ Model file not found: {model_path}")
                return False
            
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            # Load feature names if available
            info_path = self.dirs['models'] / 'tophat_model_info.json'
            if info_path.exists():
                with open(info_path, 'r') as f:
                    info = json.load(f)
                    self.feature_names = info.get('feature_names', [])
            
            print("✅ Tophat model loaded successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return False


def train_tophat_model(n_estimators=100):
    """Simple function to train tophat model"""
    trainer = TophatMLTrainer()
    return trainer.train_model(n_estimators=n_estimators)


def predict_with_tophat(image, model_path='models/tophat_model.pkl'):
    """Simple function to predict with tophat model"""
    trainer = TophatMLTrainer()
    if trainer.load_model():
        return trainer.predict(image)
    return np.zeros(image.shape[:2])


if __name__ == "__main__":
    # Test the trainer
    trainer = TophatMLTrainer()
    
    # Check for available training data
    training_data = trainer.load_annotation_data()
    print(f"📊 Found {len(training_data)} training images")
    
    if training_data:
        print("🔄 Starting training...")
        success = trainer.train_model()
        if success:
            print("✅ Training completed successfully")
        else:
            print("❌ Training failed")
    else:
        print("ℹ️ No training data found. Create annotations first.")