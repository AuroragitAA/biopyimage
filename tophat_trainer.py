#!/usr/bin/env python3
"""
ENHANCED TOPHAT ML TRAINER - Professional Implementation with Epoch-based Training
Advanced ML training with visualization and progressive learning for Wolffia detection
Author: BIOIMAGIN Professional Team
"""

import json
import pickle
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage
from skimage import filters, measure, morphology
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Optional seaborn import
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

warnings.filterwarnings('ignore')


class EnhancedTophatTrainer:
    """
    Enhanced Tophat ML Trainer with epoch-based training and comprehensive visualization
    Professional-grade training for Wolffia cell detection
    """
    
    def __init__(self):
        self.model = None
        self.feature_names = []
        self.training_history = []
        self.setup_directories()
        print("✅ Enhanced Tophat ML Trainer initialized")
    
    def setup_directories(self):
        """Setup required directories"""
        self.dirs = {
            'models': Path('models'),
            'annotations': Path('annotations'),
            'tophat_training': Path('tophat_training'),
            'training_artifacts': Path('training_artifacts')
        }
        
        for path in self.dirs.values():
            path.mkdir(exist_ok=True)
    
    def extract_comprehensive_features(self, image):
        """
        Extract comprehensive features with enhanced Wolffia-specific patterns
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Ensure consistent size for feature extraction
        h, w = image.shape
        features = []
        feature_names = []
        
        # 1. Original intensity features
        features.append(image.flatten())
        feature_names.append('Original')
        
        # 2. Multi-scale Gaussian features (important for Wolffia size variations)
        for sigma in [0.5, 1.0, 2.0, 3.0, 5.0]:
            gaussian_img = ndimage.gaussian_filter(image, sigma=sigma)
            features.append(gaussian_img.flatten())
            feature_names.append(f'Gaussian_s{sigma}')
        
        # 3. Enhanced edge detection suite
        # Canny with multiple thresholds
        for low, high in [(30, 100), (50, 150), (70, 200)]:
            edges_canny = cv2.Canny(image, low, high)
            features.append(edges_canny.flatten())
            feature_names.append(f'Canny_{low}_{high}')
        
        # Comprehensive edge filters
        edges_sobel = (filters.sobel(image) * 255).astype(np.uint8)
        features.append(edges_sobel.flatten())
        feature_names.append('Sobel')
        
        edges_roberts = (filters.roberts(image) * 255).astype(np.uint8)
        features.append(edges_roberts.flatten())
        feature_names.append('Roberts')
        
        edges_prewitt = (filters.prewitt(image) * 255).astype(np.uint8)
        features.append(edges_prewitt.flatten())
        feature_names.append('Prewitt')
        
        edges_scharr = (filters.scharr(image) * 255).astype(np.uint8)
        features.append(edges_scharr.flatten())
        feature_names.append('Scharr')
        
        # 4. Morphological features (important for cell shape)
        # Opening with different kernel sizes
        for kernel_size in [3, 5, 7]:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
            features.append(opened.flatten())
            feature_names.append(f'Opening_{kernel_size}')
        
        # Closing with different kernel sizes
        for kernel_size in [3, 5]:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
            features.append(closed.flatten())
            feature_names.append(f'Closing_{kernel_size}')
        
        # 5. Statistical filters (texture analysis)
        for size in [3, 5, 7]:
            # Variance (texture)
            variance_img = ndimage.generic_filter(image, np.var, size=size)
            features.append(variance_img.flatten())
            feature_names.append(f'Variance_{size}')
            
            # Standard deviation
            std_img = ndimage.generic_filter(image, np.std, size=size)
            features.append(std_img.flatten())
            feature_names.append(f'Std_{size}')
            
            # Median
            median_img = ndimage.median_filter(image, size=size)
            features.append(median_img.flatten())
            feature_names.append(f'Median_{size}')
        
        # 6. Min/Max filters
        for size in [3, 5]:
            maximum_img = ndimage.maximum_filter(image, size=size)
            features.append(maximum_img.flatten())
            feature_names.append(f'Maximum_{size}')
            
            minimum_img = ndimage.minimum_filter(image, size=size)
            features.append(minimum_img.flatten())
            feature_names.append(f'Minimum_{size}')
        
        # 7. Laplacian features
        laplacian_img = cv2.Laplacian(image, cv2.CV_64F)
        features.append(np.abs(laplacian_img).astype(np.uint8).flatten())
        feature_names.append('Laplacian')
        
        # 8. Hessian-based features (blob detection)
        try:
            from skimage.feature import hessian_matrix, hessian_matrix_eigvals
            hessian = hessian_matrix(image, sigma=1.0)
            eigenvals = hessian_matrix_eigvals(hessian)
            features.append((eigenvals[0] * 255).astype(np.uint8).flatten())
            features.append((eigenvals[1] * 255).astype(np.uint8).flatten())
            feature_names.extend(['Hessian_eig1', 'Hessian_eig2'])
        except:
            pass  # Skip if not available
        
        self.feature_names = feature_names
        print(f"✅ Extracted {len(feature_names)} features per pixel")
        return np.column_stack(features)
    
    def load_annotation_data_enhanced(self):
        """
        Enhanced annotation data loading with robust error handling
        """
        training_data = []
        
        print("🔍 Scanning for training data...")
        
        # 1. Load from tophat training sessions (primary source)
        session_files = list(self.dirs['tophat_training'].glob('session_*.json'))
        print(f"📁 Found {len(session_files)} session files")
        
        for session_file in session_files:
            try:
                # Check if file is not empty
                if session_file.stat().st_size == 0:
                    print(f"⚠️ Skipping empty session file: {session_file}")
                    continue
                
                with open(session_file, 'r') as f:
                    content = f.read().strip()
                    if not content:
                        print(f"⚠️ Skipping empty session: {session_file}")
                        continue
                    
                    session_data = json.loads(content)
                
                # Process session images
                session_images = 0
                for image_index, image_info in enumerate(session_data.get('images', [])):
                    image_path = image_info.get('path')
                    if image_path and Path(image_path).exists():
                        
                        # Try to find corresponding annotation
                        annotation_files = list(self.dirs['annotations'].glob(f"*_{image_index}_*_drawing.json"))
                        
                        for annotation_file in annotation_files:
                            try:
                                if annotation_file.stat().st_size > 0:
                                    with open(annotation_file, 'r') as af:
                                        annotation_data = json.load(af)
                                    
                                    # Check if annotations exist
                                    annotations = annotation_data.get('annotations', {})
                                    if any(annotations.get(key, []) for key in ['correct', 'false_positive', 'missed']):
                                        image = cv2.imread(image_path)
                                        if image is not None:
                                            training_data.append({
                                                'image': image,
                                                'image_path': image_path,
                                                'annotations': annotations,
                                                'source': f'session_{session_file.stem}'
                                            })
                                            session_images += 1
                            except Exception as e:
                                print(f"⚠️ Error loading annotation {annotation_file}: {e}")
                
                if session_images > 0:
                    print(f"✅ Loaded {session_images} images from {session_file.stem}")
                
            except json.JSONDecodeError as e:
                print(f"⚠️ JSON decode error in {session_file}: {e}")
                continue
            except Exception as e:
                print(f"⚠️ Failed to load session {session_file}: {e}")
                continue
        
        # 2. Load from standalone annotation files
        annotation_files = list(self.dirs['annotations'].glob('*_drawing.json'))
        print(f"📁 Found {len(annotation_files)} annotation files")
        
        for annotation_file in annotation_files:
            try:
                if annotation_file.stat().st_size > 0:
                    with open(annotation_file, 'r') as f:
                        annotation_data = json.load(f)
                    
                    # Try to find corresponding annotated image
                    image_name = annotation_file.stem.replace('_drawing', '_annotated')
                    image_path = annotation_file.parent / f"{image_name}.png"
                    
                    if not image_path.exists():
                        # Try original image path from annotation data
                        if 'image_filename' in annotation_data:
                            original_name = annotation_data['image_filename']
                            # Look in uploads directory
                            uploads_path = Path('uploads') / original_name
                            if uploads_path.exists():
                                image_path = uploads_path
                    
                    if image_path.exists():
                        image = cv2.imread(str(image_path))
                        if image is not None:
                            annotations = annotation_data.get('annotations', {})
                            if any(annotations.get(key, []) for key in ['correct', 'false_positive', 'missed']):
                                training_data.append({
                                    'image': image,
                                    'image_path': str(image_path),
                                    'annotations': annotations,
                                    'source': f'annotation_{annotation_file.stem}'
                                })
            except Exception as e:
                print(f"⚠️ Error loading annotation {annotation_file}: {e}")
        
        print(f"📊 Total training data collected: {len(training_data)} images")
        return training_data
    
    def create_training_labels(self, image, annotations):
        """
        Create training labels from user annotations with enhanced processing
        """
        labels = np.zeros(image.shape[:2], dtype=np.uint8)
        positive_pixels = 0
        
        # Process correct cell annotations (positive labels)
        for annotation in annotations.get('correct', []):
            if isinstance(annotation, dict) and 'x' in annotation and 'y' in annotation:
                x, y = int(annotation['x']), int(annotation['y'])
                radius = int(annotation.get('radius', 8))  # Default radius for Wolffia cells
                
                # Create circular region
                rr, cc = np.ogrid[:image.shape[0], :image.shape[1]]
                mask = (cc - x)**2 + (rr - y)**2 <= radius**2
                
                # Apply mask within image bounds
                valid_mask = (cc >= 0) & (cc < image.shape[1]) & (rr >= 0) & (rr < image.shape[0])
                final_mask = mask & valid_mask
                
                labels[final_mask] = 1
                positive_pixels += np.sum(final_mask)
        
        # Process polygon annotations if available
        for annotation in annotations.get('correct', []):
            if isinstance(annotation, list) and len(annotation) > 2:
                # Polygon annotation
                try:
                    points = np.array(annotation, dtype=np.int32)
                    cv2.fillPoly(labels, [points], 1)
                    positive_pixels += np.sum(labels == 1) - positive_pixels
                except:
                    continue
        
        print(f"  Created {positive_pixels} positive pixels from annotations")
        return labels, positive_pixels
    
    def prepare_balanced_training_data(self, training_data, max_samples_per_image=5000):
        """
        Prepare balanced training data with enhanced sampling strategy
        """
        X_list = []
        y_list = []
        total_positive = 0
        total_negative = 0
        
        print(f"🔄 Processing {len(training_data)} training images...")
        
        for i, data in enumerate(training_data):
            print(f"Processing image {i+1}/{len(training_data)} from {data['source']}")
            
            image = data['image']
            annotations = data['annotations']
            
            # Create labels
            labels, positive_pixels = self.create_training_labels(image, annotations)
            
            if positive_pixels == 0:
                print(f"  ⚠️ No positive samples in image {i+1}, skipping")
                continue
            
            # Extract features
            try:
                features = self.extract_comprehensive_features(image)
                labels_flat = labels.flatten()
                
                # Get positive and negative indices
                positive_indices = np.where(labels_flat == 1)[0]
                negative_indices = np.where(labels_flat == 0)[0]
                
                n_positive = len(positive_indices)
                
                if n_positive > 0:
                    # Sample balanced data
                    n_negative = min(n_positive * 2, len(negative_indices), max_samples_per_image - n_positive)
                    
                    if n_negative > 0:
                        negative_sample = np.random.choice(negative_indices, n_negative, replace=False)
                        selected_indices = np.concatenate([positive_indices, negative_sample])
                        
                        X_list.append(features[selected_indices])
                        y_list.append(labels_flat[selected_indices])
                        
                        total_positive += n_positive
                        total_negative += n_negative
                        
                        print(f"  ✅ Added {n_positive} positive, {n_negative} negative samples")
                    else:
                        print(f"  ⚠️ No negative samples available in image {i+1}")
                else:
                    print(f"  ⚠️ No positive samples in image {i+1}")
                    
            except Exception as e:
                print(f"  ❌ Error processing image {i+1}: {e}")
                continue
        
        if not X_list:
            raise ValueError("No valid training data found. Please check your annotations.")
        
        # Combine all training data
        X = np.vstack(X_list)
        y = np.hstack(y_list)
        
        print(f"\n✅ Training data prepared:")
        print(f"   Total samples: {X.shape[0]}")
        print(f"   Features per sample: {X.shape[1]}")
        print(f"   Positive samples: {total_positive}")
        print(f"   Negative samples: {total_negative}")
        print(f"   Class ratio: {total_positive / (total_positive + total_negative):.3f}")
        
        return X, y
    
    def train_model_with_epochs(self, n_estimators=100, epochs=10, validation_split=0.2):
        """
        Train model with epoch-based progressive learning and visualization
        """
        try:
            print("🚀 ENHANCED TOPHAT TRAINING WITH EPOCHS")
            print("=" * 50)
            
            # Load and prepare data
            training_data = self.load_annotation_data_enhanced()
            
            if not training_data:
                print("❌ No training data found. Please create annotations first.")
                print("💡 Use the web interface to create tophat training sessions")
                return False
            
            # Create visualization of training samples
            print("\n📷 Creating training data visualizations...")
            self.visualize_training_samples(training_data, max_samples=6)
            
            X, y = self.prepare_balanced_training_data(training_data)
            
            if len(np.unique(y)) < 2:
                print("❌ Need both positive and negative samples for training")
                return False
            
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=validation_split, random_state=42, stratify=y
            )
            
            print(f"\n🔄 Training with {epochs} epochs...")
            print(f"   Training samples: {len(X_train)}")
            print(f"   Validation samples: {len(X_val)}")
            
            # Initialize tracking
            self.training_history = []
            best_val_accuracy = 0
            best_model = None
            
            # Epoch-based training
            for epoch in range(epochs):
                print(f"\n📈 Epoch {epoch + 1}/{epochs}")
                
                # Increase complexity over epochs
                current_estimators = min(n_estimators, max(10, int(n_estimators * (epoch + 1) / epochs)))
                
                # Create and train model
                model = RandomForestClassifier(
                    n_estimators=current_estimators,
                    max_depth=None if epoch >= epochs // 2 else 10 + epoch,
                    random_state=42 + epoch,
                    n_jobs=-1,
                    class_weight='balanced'
                )
                
                # Train on progressively more data
                train_fraction = 0.6 + 0.4 * (epoch / epochs)
                n_train_samples = int(len(X_train) * train_fraction)
                
                indices = np.random.choice(len(X_train), n_train_samples, replace=False)
                X_epoch = X_train[indices]
                y_epoch = y_train[indices]
                
                print(f"   Training with {current_estimators} estimators on {n_train_samples} samples")
                
                model.fit(X_epoch, y_epoch)
                
                # Evaluate
                train_pred = model.predict(X_train)
                val_pred = model.predict(X_val)
                
                train_accuracy = accuracy_score(y_train, train_pred)
                val_accuracy = accuracy_score(y_val, val_pred)
                
                # Track history
                epoch_data = {
                    'epoch': epoch + 1,
                    'train_accuracy': train_accuracy,
                    'val_accuracy': val_accuracy,
                    'n_estimators': current_estimators,
                    'train_samples': n_train_samples
                }
                self.training_history.append(epoch_data)
                
                print(f"   Train accuracy: {train_accuracy:.3f}")
                print(f"   Val accuracy: {val_accuracy:.3f}")
                
                # Save best model
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    best_model = model
                    print(f"   🌟 New best model! Validation accuracy: {val_accuracy:.3f}")
            
            # Use best model
            self.model = best_model
            
            # Final evaluation
            print(f"\n✅ TRAINING COMPLETED")
            print(f"   Best validation accuracy: {best_val_accuracy:.3f}")
            
            # Feature importance analysis
            if self.feature_names:
                feature_importance = pd.Series(
                    self.model.feature_importances_, 
                    index=self.feature_names
                ).sort_values(ascending=False)
                
                print(f"\n🔍 Top 10 Feature Importances:")
                for feature, importance in feature_importance.head(10).items():
                    print(f"   {feature}: {importance:.3f}")
            
            # Save model and create visualizations
            self.save_enhanced_model(best_val_accuracy, len(training_data))
            self.create_training_visualizations()
            
            return True
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def save_enhanced_model(self, best_accuracy, num_training_images):
        """Save model with enhanced metadata"""
        try:
            # Save model
            model_path = self.dirs['models'] / 'tophat_model.pkl'
            with open(model_path, 'wb') as f:
                pickle.dump(self.model, f)
            
            # Save comprehensive model info
            model_info = {
                'model_type': 'EnhancedRandomForestClassifier',
                'training_method': 'epoch_based_progressive',
                'n_estimators': self.model.n_estimators,
                'best_val_accuracy': best_accuracy,
                'num_training_images': num_training_images,
                'feature_names': self.feature_names,
                'training_history': self.training_history,
                'timestamp': datetime.now().isoformat(),
                'num_features': len(self.feature_names),
                'epochs_trained': len(self.training_history)
            }
            
            info_path = self.dirs['models'] / 'tophat_model_info.json'
            with open(info_path, 'w') as f:
                json.dump(model_info, f, indent=2)
            
            # Save training history separately
            history_path = self.dirs['training_artifacts'] / 'tophat_training_history.json'
            with open(history_path, 'w') as f:
                json.dump(self.training_history, f, indent=2)
            
            print(f"✅ Enhanced model saved:")
            print(f"   Model: {model_path}")
            print(f"   Info: {info_path}")
            print(f"   History: {history_path}")
            
        except Exception as e:
            print(f"❌ Failed to save model: {e}")
    
    def visualize_training_samples(self, training_data, max_samples=6):
        """Create visualization of training samples and annotations"""
        try:
            print("📷 Creating training samples visualization...")
            
            if not training_data:
                print("⚠️ No training data available for visualization")
                return
            
            # Select samples to visualize
            n_samples = min(max_samples, len(training_data))
            sample_indices = np.linspace(0, len(training_data)-1, n_samples, dtype=int)
            
            # Create figure
            fig, axes = plt.subplots(n_samples, 3, figsize=(15, 5*n_samples))
            if n_samples == 1:
                axes = axes.reshape(1, -1)
            
            for i, idx in enumerate(sample_indices):
                data = training_data[idx]
                image = data['image']
                annotations = data['annotations']
                
                # Convert to grayscale if needed
                if len(image.shape) == 3:
                    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    display_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    gray_image = image
                    display_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                
                # Create labels from annotations
                labels, _ = self.create_training_labels(image, annotations)
                
                # Column 1: Original image
                axes[i, 0].imshow(display_image)
                axes[i, 0].set_title(f'Sample {i+1}: Original Image\\nSource: {data.get("source", "unknown")}')
                axes[i, 0].axis('off')
                
                # Column 2: Annotation overlay
                overlay = display_image.copy()
                
                # Draw missed cells (positive training examples)
                for annotation in annotations.get('missed', []):
                    if isinstance(annotation, list) and len(annotation) > 2:
                        # Polygon annotation
                        try:
                            points = np.array(annotation, dtype=np.int32)
                            cv2.polylines(overlay, [points], True, (255, 0, 0), 2)
                            # Fill with semi-transparent red
                            overlay_fill = overlay.copy()
                            cv2.fillPoly(overlay_fill, [points], (255, 0, 0))
                            overlay = cv2.addWeighted(overlay, 0.7, overlay_fill, 0.3, 0)
                        except:
                            continue
                
                # Draw correct cells (if any)
                for annotation in annotations.get('correct', []):
                    if isinstance(annotation, dict) and 'x' in annotation:
                        x, y = int(annotation['x']), int(annotation['y'])
                        radius = int(annotation.get('radius', 8))
                        cv2.circle(overlay, (x, y), radius, (0, 255, 0), 2)
                    elif isinstance(annotation, list) and len(annotation) > 2:
                        try:
                            points = np.array(annotation, dtype=np.int32)
                            cv2.polylines(overlay, [points], True, (0, 255, 0), 2)
                        except:
                            continue
                
                # Draw false positives (if any)
                for annotation in annotations.get('false_positive', []):
                    if isinstance(annotation, dict) and 'x' in annotation:
                        x, y = int(annotation['x']), int(annotation['y'])
                        radius = int(annotation.get('radius', 8))
                        cv2.circle(overlay, (x, y), radius, (0, 0, 255), 2)
                        cv2.line(overlay, (x-radius, y-radius), (x+radius, y+radius), (0, 0, 255), 2)
                        cv2.line(overlay, (x-radius, y+radius), (x+radius, y-radius), (0, 0, 255), 2)
                
                axes[i, 1].imshow(overlay)
                axes[i, 1].set_title(f'Annotations\\nMissed: {len(annotations.get("missed", []))} (red)\\nCorrect: {len(annotations.get("correct", []))} (green)\\nFalse+: {len(annotations.get("false_positive", []))} (blue X)')
                axes[i, 1].axis('off')
                
                # Column 3: Training labels
                axes[i, 2].imshow(labels, cmap='viridis')
                axes[i, 2].set_title(f'Training Labels\\nPositive pixels: {np.sum(labels > 0)}\\nNegative pixels: {np.sum(labels == 0)}')
                axes[i, 2].axis('off')
            
            plt.tight_layout()
            
            # Save visualization
            viz_path = self.dirs['training_artifacts'] / 'tophat_training_samples.png'
            plt.savefig(viz_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ Training samples visualization saved to {viz_path}")
            
            # Also create a summary statistics plot
            self.create_annotation_statistics(training_data)
            
        except Exception as e:
            print(f"⚠️ Failed to create training samples visualization: {e}")
            import traceback
            traceback.print_exc()
    
    def create_annotation_statistics(self, training_data):
        """Create statistics visualization for training annotations"""
        try:
            print("📊 Creating annotation statistics visualization...")
            
            # Collect statistics
            stats = {
                'missed_counts': [],
                'correct_counts': [],
                'false_positive_counts': [],
                'image_sizes': [],
                'sources': []
            }
            
            for data in training_data:
                annotations = data['annotations']
                stats['missed_counts'].append(len(annotations.get('missed', [])))
                stats['correct_counts'].append(len(annotations.get('correct', [])))
                stats['false_positive_counts'].append(len(annotations.get('false_positive', [])))
                
                image = data['image']
                stats['image_sizes'].append(image.shape[0] * image.shape[1])
                stats['sources'].append(data.get('source', 'unknown'))
            
            # Create statistics figure
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Plot 1: Annotation counts
            x = range(len(training_data))
            axes[0, 0].bar(x, stats['missed_counts'], alpha=0.7, label='Missed cells', color='red')
            axes[0, 0].bar(x, stats['correct_counts'], alpha=0.7, label='Correct cells', color='green')
            axes[0, 0].bar(x, stats['false_positive_counts'], alpha=0.7, label='False positives', color='blue')
            axes[0, 0].set_title('Annotation Counts per Image')
            axes[0, 0].set_xlabel('Training Sample')
            axes[0, 0].set_ylabel('Number of Annotations')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Plot 2: Distribution of missed cells
            axes[0, 1].hist(stats['missed_counts'], bins=max(1, len(set(stats['missed_counts']))), 
                           alpha=0.7, color='red', edgecolor='black')
            axes[0, 1].set_title('Distribution of Missed Cells')
            axes[0, 1].set_xlabel('Number of Missed Cells')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Plot 3: Image sizes
            axes[1, 0].scatter(range(len(training_data)), stats['image_sizes'], alpha=0.6, color='purple')
            axes[1, 0].set_title('Training Image Sizes')
            axes[1, 0].set_xlabel('Training Sample')
            axes[1, 0].set_ylabel('Image Size (pixels)')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Plot 4: Summary statistics
            total_missed = sum(stats['missed_counts'])
            total_correct = sum(stats['correct_counts'])
            total_false_pos = sum(stats['false_positive_counts'])
            
            categories = ['Missed\\n(Need Detection)', 'Correct\\n(Good Detection)', 'False Positive\\n(Over Detection)']
            values = [total_missed, total_correct, total_false_pos]
            colors = ['red', 'green', 'blue']
            
            axes[1, 1].bar(categories, values, color=colors, alpha=0.7)
            axes[1, 1].set_title('Total Annotation Summary')
            axes[1, 1].set_ylabel('Total Count')
            
            # Add value labels on bars
            for i, v in enumerate(values):
                axes[1, 1].text(i, v + max(values)*0.01, str(v), ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            
            # Save statistics
            stats_path = self.dirs['training_artifacts'] / 'tophat_annotation_statistics.png'
            plt.savefig(stats_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ Annotation statistics saved to {stats_path}")
            print(f"📊 Training data summary:")
            print(f"   Total images: {len(training_data)}")
            print(f"   Total missed cells: {total_missed}")
            print(f"   Total correct cells: {total_correct}")
            print(f"   Total false positives: {total_false_pos}")
            print(f"   Average missed per image: {total_missed/len(training_data):.1f}")
            
        except Exception as e:
            print(f"⚠️ Failed to create annotation statistics: {e}")

    def create_training_visualizations(self):
        """Create comprehensive training visualizations"""
        try:
            if not self.training_history:
                return
            
            print("📊 Creating training visualizations...")
            
            # Prepare data
            epochs = [h['epoch'] for h in self.training_history]
            train_acc = [h['train_accuracy'] for h in self.training_history]
            val_acc = [h['val_accuracy'] for h in self.training_history]
            n_estimators = [h['n_estimators'] for h in self.training_history]
            
            # Create figure with multiple subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # 1. Accuracy over epochs
            axes[0, 0].plot(epochs, train_acc, 'b-', label='Training Accuracy', marker='o')
            axes[0, 0].plot(epochs, val_acc, 'r-', label='Validation Accuracy', marker='s')
            axes[0, 0].set_title('Training Progress - Accuracy')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Accuracy')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. Model complexity over epochs
            axes[0, 1].plot(epochs, n_estimators, 'g-', marker='^')
            axes[0, 1].set_title('Model Complexity Over Epochs')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Number of Estimators')
            axes[0, 1].grid(True, alpha=0.3)
            
            # 3. Feature importance (top 15)
            if self.feature_names and hasattr(self.model, 'feature_importances_'):
                feature_importance = pd.Series(
                    self.model.feature_importances_, 
                    index=self.feature_names
                ).sort_values(ascending=True).tail(15)
                
                axes[1, 0].barh(range(len(feature_importance)), feature_importance.values)
                axes[1, 0].set_yticks(range(len(feature_importance)))
                axes[1, 0].set_yticklabels(feature_importance.index)
                axes[1, 0].set_title('Top 15 Feature Importances')
                axes[1, 0].set_xlabel('Importance')
            
            # 4. Learning curve
            axes[1, 1].fill_between(epochs, train_acc, alpha=0.3, color='blue', label='Training')
            axes[1, 1].fill_between(epochs, val_acc, alpha=0.3, color='red', label='Validation')
            axes[1, 1].plot(epochs, train_acc, 'b-', linewidth=2)
            axes[1, 1].plot(epochs, val_acc, 'r-', linewidth=2)
            axes[1, 1].set_title('Learning Curve')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Accuracy')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            plot_path = self.dirs['training_artifacts'] / 'tophat_training_visualization.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ Training visualizations saved to {plot_path}")
            
        except Exception as e:
            print(f"⚠️ Failed to create visualizations: {e}")


def train_enhanced_tophat_model(epochs=10, n_estimators=100):
    """Main function to train enhanced tophat model"""
    trainer = EnhancedTophatTrainer()
    return trainer.train_model_with_epochs(n_estimators=n_estimators, epochs=epochs)


if __name__ == "__main__":
    print("🧠 ENHANCED TOPHAT ML TRAINER")
    print("=" * 50)
    print("🎯 Advanced ML training with epoch-based progressive learning")
    print("📊 Professional visualization and comprehensive feature extraction")
    print()
    
    # Configuration
    print("📊 Training Configuration:")
    try:
        epochs = int(input("Number of epochs [10]: ") or "10")
        n_estimators = int(input("Number of estimators [100]: ") or "100")
    except:
        epochs, n_estimators = 10, 100
        print("Using default configuration: 10 epochs, 100 estimators")
    
    print(f"✅ Configuration: {epochs} epochs, {n_estimators} estimators")
    print()
    
    # Start training
    success = train_enhanced_tophat_model(epochs=epochs, n_estimators=n_estimators)
    
    if success:
        print("\n🎉 ENHANCED TOPHAT TRAINING COMPLETED!")
        print("=" * 50)
        print("✅ Model trained with epoch-based progressive learning")
        print("✅ Comprehensive visualizations created")
        print("✅ Feature importance analysis completed")
        print("✅ Ready for integration with analysis pipeline")
        print()
        print("📁 Artifacts created:")
        print("   • models/tophat_model.pkl - Trained model")
        print("   • models/tophat_model_info.json - Model metadata")
        print("   • training_artifacts/tophat_training_history.json - Training history")
        print("   • training_artifacts/tophat_training_visualization.png - Training plots")
        print()
        print("🔄 Restart your web server to use the enhanced tophat model!")
    else:
        print("\n❌ Training failed. Please check the error messages above.")
        print("💡 Make sure you have created annotations using the web interface first.")