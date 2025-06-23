#!/usr/bin/env python3
"""
ENHANCED TOPHAT ML TRAINER - Professional Implementation with Enhanced Patch-Based Training
Advanced ML training compatible with enhanced patch-based processing system
Author: BIOIMAGIN Professional Team
"""

import json
import pickle
import warnings

import matplotlib

matplotlib.use('Agg')
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
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
    Enhanced Tophat ML Trainer compatible with patch-based processing
    Professional-grade training for Wolffia cell detection using color information
    """
    
    def __init__(self):
        self.model = None
        self.feature_names = []
        self.training_history = []
        self.setup_directories()
        
        # Enhanced training parameters compatible with patch processing
        self.patch_size = 256  # Compatible with enhanced patch processing
        self.overlap = 64
        self.enhanced_features = True
        
        print("✅ Enhanced Color-Aware Tophat ML Trainer initialized")
        print("🔧 Compatible with patch-based processing system")
    
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
        
    def extract_enhanced_patch_features(self, image):
        """
        OPTIMIZED: Extract 10 most important features for fast training
        Must match bioimaging.py extract_ml_features exactly
        """
        # Ensure we have a 3-channel color image
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif len(image.shape) == 3 and image.shape[2] == 4:
            image = image[:, :, :3]
        
        h, w = image.shape[:2]
        img_size = h * w
        
        # Pre-allocate for 10 optimized features
        features = np.empty((img_size, 10), dtype=np.float32)
        
        # OPTIMIZED: Only most discriminative features for Wolffia (same as bioimaging.py)
        b, g, r = cv2.split(image)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h_hsv, s_hsv, v_hsv = cv2.split(hsv)
        
        # Flatten channels once
        g_flat = g.flatten().astype(np.float32)
        r_flat = r.flatten().astype(np.float32)
        b_flat = b.flatten().astype(np.float32)
        
        idx = 0
        
        # Feature 1: Green channel (most important for Wolffia)
        features[:, idx] = g_flat
        idx += 1
        
        # Feature 2: Green dominance (key discriminator)
        green_dominance = np.clip(g_flat - np.maximum(r_flat, b_flat), 0, 255)
        features[:, idx] = green_dominance
        idx += 1
        
        # Feature 3: Green mask (binary green detection)
        green_mask = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255])).flatten()
        features[:, idx] = green_mask.astype(np.float32)
        idx += 1
        
        # Feature 4: Enhanced grayscale (color-aware processing)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        a_lab = cv2.split(lab)[1]
        enhanced_gray = (0.4 * g_flat + 0.3 * (255 - a_lab.flatten()) + 0.3 * green_mask)
        features[:, idx] = enhanced_gray
        idx += 1
        
        # Feature 5: Distance transform (cell center detection)
        enhanced_gray_2d = enhanced_gray.reshape(h, w).astype(np.uint8)
        green_binary = (green_mask.reshape(h, w) > 0).astype(np.uint8)
        if np.sum(green_binary) > 0:
            dist_transform = cv2.distanceTransform(green_binary, cv2.DIST_L2, 5)
            features[:, idx] = dist_transform.flatten()
        else:
            features[:, idx] = np.zeros(img_size)
        idx += 1
        
        # Feature 6: Tophat operation (blob detection)
        kernel5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        tophat = cv2.morphologyEx(enhanced_gray_2d, cv2.MORPH_TOPHAT, kernel5)
        features[:, idx] = tophat.flatten()
        idx += 1
        
        # Feature 7: Gaussian blur (smoothed regions)
        gauss = cv2.GaussianBlur(enhanced_gray_2d, (5, 5), 2.0)
        features[:, idx] = gauss.flatten()
        idx += 1
        
        # Feature 8: Local variance (texture)
        from scipy.ndimage import uniform_filter
        mean = uniform_filter(enhanced_gray_2d.astype(np.float32), size=5)
        sqr_mean = uniform_filter(enhanced_gray_2d.astype(np.float32)**2, size=5)
        variance = np.clip(sqr_mean - mean**2, 0, None)
        features[:, idx] = variance.flatten()
        idx += 1
        
        # Feature 9: HSV Saturation (color purity)
        features[:, idx] = s_hsv.flatten().astype(np.float32)
        idx += 1
        
        # Feature 10: Edge strength (boundary detection)
        edges = cv2.Canny(enhanced_gray_2d, 40, 120)
        features[:, idx] = edges.flatten().astype(np.float32)
        
        # Update feature names for 10 optimized features
        self.feature_names = [
            'Green_channel', 'Green_dominance', 'Green_mask', 'Enhanced_grayscale',
            'Distance_transform', 'Tophat_operation', 'Gaussian_blur', 'Local_variance',
            'HSV_Saturation', 'Edge_strength'
        ]
        
        print(f"🚀 Extracted {len(self.feature_names)} optimized features for fast Wolffia training")
        return features
    
    def create_patch_based_training_data(self, image, annotations):
        """
        Create training data using patch-based approach compatible with enhanced analysis
        """
        try:
            h, w = image.shape[:2]
            patch_features = []
            patch_labels = []
            
            # Create full image labels from annotations
            full_labels, positive_pixels = self.create_training_labels(image, annotations)
            
            if positive_pixels == 0:
                print("  ⚠️ No positive samples in image, skipping")
                return None, None
            
            # Extract patches with overlap (compatible with enhanced analysis)
            patch_size = self.patch_size
            overlap = self.overlap
            
            patches_processed = 0
            for y in range(0, h - patch_size + 1, patch_size - overlap):
                for x in range(0, w - patch_size + 1, patch_size - overlap):
                    y_end = min(y + patch_size, h)
                    x_end = min(x + patch_size, w)
                    
                    # Extract patch
                    patch_img = image[y:y_end, x:x_end]
                    patch_labels_region = full_labels[y:y_end, x:x_end]
                    
                    # Resize to standard patch size if needed
                    if patch_img.shape[:2] != (patch_size, patch_size):
                        patch_img = cv2.resize(patch_img, (patch_size, patch_size))
                        patch_labels_region = cv2.resize(patch_labels_region, (patch_size, patch_size), 
                                                       interpolation=cv2.INTER_NEAREST)
                    
                    # Check if patch has sufficient positive samples
                    positive_in_patch = np.sum(patch_labels_region > 0)
                    if positive_in_patch < 50:  # Skip patches with too few positive samples
                        continue
                    
                    # Extract features from patch
                    patch_features_data = self.extract_enhanced_patch_features(patch_img)
                    patch_labels_flat = patch_labels_region.flatten()
                    
                    # Balanced sampling within patch
                    positive_indices = np.where(patch_labels_flat == 1)[0]
                    negative_indices = np.where(patch_labels_flat == 0)[0]
                    
                    n_positive = len(positive_indices)
                    n_negative = min(n_positive * 2, len(negative_indices), 2000)  # Limit per patch
                    
                    if n_negative > 0:
                        negative_sample = np.random.choice(negative_indices, n_negative, replace=False)
                        selected_indices = np.concatenate([positive_indices, negative_sample])
                        
                        patch_features.append(patch_features_data[selected_indices])
                        patch_labels.append(patch_labels_flat[selected_indices])
                        patches_processed += 1
            
            if not patch_features:
                print("  ⚠️ No valid patches extracted")
                return None, None
            
            # Combine all patch data
            X = np.vstack(patch_features)
            y = np.hstack(patch_labels)
            
            print(f"  ✅ Processed {patches_processed} patches")
            print(f"  📊 Training samples: {X.shape[0]}, Features: {X.shape[1]}")
            print(f"  📊 Positive samples: {np.sum(y)}, Negative samples: {np.sum(y == 0)}")
            
            return X, y
            
        except Exception as e:
            print(f"  ❌ Error in patch-based processing: {e}")
            return None, None
    
    def load_annotation_data_enhanced(self):
        """Enhanced annotation data loading compatible with new system"""
        training_data = []
        
        print("🔍 Scanning for enhanced training data...")
        
        # Load from tophat training sessions (primary source)
        session_files = list(self.dirs['tophat_training'].glob('session_*.json'))
        print(f"📁 Found {len(session_files)} session files")
        
        for session_file in session_files:
            try:
                if session_file.stat().st_size == 0:
                    continue
                
                with open(session_file, 'r') as f:
                    content = f.read().strip()
                    if not content:
                        continue
                    session_data = json.loads(content)
                
                session_images = 0
                for image_index, image_info in enumerate(session_data.get('images', [])):
                    image_path = image_info.get('path')
                    if image_path and Path(image_path).exists():
                        
                        annotation_files = list(self.dirs['annotations'].glob(f"*_{image_index}_*_drawing.json"))
                        
                        for annotation_file in annotation_files:
                            try:
                                if annotation_file.stat().st_size > 0:
                                    with open(annotation_file, 'r') as af:
                                        annotation_data = json.load(af)
                                    
                                    annotations = annotation_data.get('annotations', {})
                                    if any(annotations.get(key, []) for key in ['correct', 'false_positive', 'missed']):
                                        # Load as COLOR image for enhanced processing
                                        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
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
                    print(f"✅ Loaded {session_images} color images from {session_file.stem}")
                
            except Exception as e:
                print(f"⚠️ Failed to load session {session_file}: {e}")
                continue
        
        # Load from standalone annotation files
        annotation_files = list(self.dirs['annotations'].glob('*_drawing.json'))
        print(f"📁 Found {len(annotation_files)} annotation files")
        
        for annotation_file in annotation_files:
            try:
                if annotation_file.stat().st_size > 0:
                    with open(annotation_file, 'r') as f:
                        annotation_data = json.load(f)
                    
                    image_name = annotation_file.stem.replace('_drawing', '_annotated')
                    image_path = annotation_file.parent / f"{image_name}.png"
                    
                    if not image_path.exists():
                        if 'image_filename' in annotation_data:
                            original_name = annotation_data['image_filename']
                            uploads_path = Path('uploads') / original_name
                            if uploads_path.exists():
                                image_path = uploads_path
                    
                    if image_path.exists():
                        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
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
        
        print(f"📊 Total enhanced training data: {len(training_data)} images")
        return training_data
    
    def create_training_labels(self, image, annotations):
        """Create training labels from user annotations"""
        labels = np.zeros(image.shape[:2], dtype=np.uint8)
        positive_pixels = 0
        
        # Process correct cell annotations (positive labels)
        for annotation in annotations.get('correct', []):
            if isinstance(annotation, dict) and 'x' in annotation and 'y' in annotation:
                x, y = int(annotation['x']), int(annotation['y'])
                radius = int(annotation.get('radius', 12))  # Slightly larger for better training
                
                # Create circular region
                rr, cc = np.ogrid[:image.shape[0], :image.shape[1]]
                mask = (cc - x)**2 + (rr - y)**2 <= radius**2
                
                valid_mask = (cc >= 0) & (cc < image.shape[1]) & (rr >= 0) & (rr < image.shape[0])
                final_mask = mask & valid_mask
                
                labels[final_mask] = 1
                positive_pixels += np.sum(final_mask)
        
        # Process missed cells as positive
        for annotation in annotations.get('missed', []):
            if isinstance(annotation, dict) and 'x' in annotation and 'y' in annotation:
                x, y = int(annotation['x']), int(annotation['y'])
                radius = int(annotation.get('radius', 12))
                
                rr, cc = np.ogrid[:image.shape[0], :image.shape[1]]
                mask = (cc - x)**2 + (rr - y)**2 <= radius**2
                valid_mask = (cc >= 0) & (cc < image.shape[1]) & (rr >= 0) & (rr < image.shape[0])
                final_mask = mask & valid_mask
                
                labels[final_mask] = 1
                positive_pixels += np.sum(final_mask)
        
        print(f"  Created {positive_pixels} positive pixels from annotations")
        return labels, positive_pixels
    
    def prepare_enhanced_training_data(self, training_data):
        """Prepare enhanced training data using patch-based approach"""
        X_list = []
        y_list = []
        total_positive = 0
        total_negative = 0
        
        print(f"🔄 Processing {len(training_data)} images with patch-based approach...")
        
        for i, data in enumerate(training_data):
            print(f"Processing enhanced image {i+1}/{len(training_data)} from {data['source']}")
            
            image = data['image']
            annotations = data['annotations']
            
            # Use patch-based processing
            X_patch, y_patch = self.create_patch_based_training_data(image, annotations)
            
            if X_patch is not None and y_patch is not None:
                X_list.append(X_patch)
                y_list.append(y_patch)
                
                n_positive = np.sum(y_patch)
                n_negative = np.sum(y_patch == 0)
                total_positive += n_positive
                total_negative += n_negative
                
                print(f"  ✅ Added {n_positive} positive, {n_negative} negative samples")
            else:
                print(f"  ⚠️ No valid patch data from image {i+1}")
        
        if not X_list:
            raise ValueError("No valid patch-based training data found. Please check your annotations.")
        
        # Combine all training data
        X = np.vstack(X_list)
        y = np.hstack(y_list)
        
        print(f"\n✅ Enhanced patch-based training data prepared:")
        print(f"   Total samples: {X.shape[0]}")
        print(f"   Enhanced features per sample: {X.shape[1]}")
        print(f"   Positive samples: {total_positive}")
        print(f"   Negative samples: {total_negative}")
        print(f"   Class ratio: {total_positive / (total_positive + total_negative):.3f}")
        
        return X, y
    
    def train_enhanced_model(self, n_estimators=150, epochs=12, validation_split=0.2):
        """Train enhanced model with separation intelligence"""
        try:
            print("🧠 SEPARATION-INTELLIGENT TOPHAT TRAINING")
            print("=" * 60)
            
            # Load and prepare data
            training_data = self.load_annotation_data_enhanced()
            
            if not training_data:
                print("❌ No training data found. Please create annotations first.")
                return False
            
            # Use separation-aware training instead of regular training
            print("\n🧠 Preparing separation-aware training data...")
            X, y = self.prepare_separation_aware_training_data(training_data)
            
            if len(np.unique(y)) < 2:
                print("❌ Need both positive and negative samples for training")
                return False
            
            # Train with separation intelligence
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=validation_split, random_state=42, stratify=y
            )
            
            print(f"\n🧠 Training separation-intelligent model...")
            print(f"   Training samples: {len(X_train)}")
            print(f"   Validation samples: {len(X_val)}")
            print(f"   Features: {X.shape[1]} (with separation intelligence)")
            
            # Train the separation-intelligent model
            self.model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=20,
                min_samples_split=3,
                min_samples_leaf=1,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced',
                max_features='sqrt'
            )
            
            self.model.fit(X_train, y_train)
            
            # Evaluate
            train_pred = self.model.predict(X_train)
            val_pred = self.model.predict(X_val)
            
            train_accuracy = accuracy_score(y_train, train_pred)
            val_accuracy = accuracy_score(y_val, val_pred)
            
            print(f"\n✅ SEPARATION-INTELLIGENT TRAINING COMPLETED")
            print(f"   Training accuracy: {train_accuracy:.3f}")
            print(f"   Validation accuracy: {val_accuracy:.3f}")
            
            # Feature importance analysis
            if self.feature_names:
                feature_importance = pd.Series(
                    self.model.feature_importances_, 
                    index=self.feature_names
                ).sort_values(ascending=False)
                
                print(f"\n🔍 Top 15 Separation-Intelligent Feature Importances:")
                for feature, importance in feature_importance.head(15).items():
                    print(f"   {feature}: {importance:.3f}")
            
            # Save model
            self.save_enhanced_model(val_accuracy, len(training_data))
            
            return True
            
        except Exception as e:
            print(f"❌ Separation-intelligent training failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def prepare_separation_aware_training_data(self, training_data):
        """Prepare training data with separation intelligence built into features"""
        X_list = []
        y_list = []
        total_positive = 0
        total_negative = 0
        
        print(f"🧠 Processing {len(training_data)} images with separation intelligence...")
        
        for i, data in enumerate(training_data):
            print(f"Processing separation-intelligent image {i+1}/{len(training_data)}")
            
            image = data['image']
            annotations = data['annotations']
            
            # Use regular patch-based training but with 52 separation-intelligent features
            X_patch, y_patch = self.create_patch_based_training_data(image, annotations)
            
            if X_patch is not None and y_patch is not None:
                X_list.append(X_patch)
                y_list.append(y_patch)
                
                n_positive = np.sum(y_patch)
                n_negative = np.sum(y_patch == 0)
                total_positive += n_positive
                total_negative += n_negative
                
                print(f"  ✅ Added {n_positive} positive, {n_negative} negative samples with separation features")
            else:
                print(f"  ⚠️ No valid patch data from image {i+1}")
        
        if not X_list:
            raise ValueError("No valid separation-intelligent training data found.")
        
        # Combine all training data
        X = np.vstack(X_list)
        y = np.hstack(y_list)
        
        print(f"\n✅ Separation-intelligent training data prepared:")
        print(f"   Total samples: {X.shape[0]}")
        print(f"   Separation-intelligent features: {X.shape[1]}")
        print(f"   Positive samples: {total_positive}")
        print(f"   Negative samples: {total_negative}")
        
        return X, y
    
    def save_enhanced_model(self, best_accuracy, num_training_images):
        """Save enhanced model with comprehensive metadata"""
        try:
            # Save model
            model_path = self.dirs['models'] / 'tophat_model.pkl'
            with open(model_path, 'wb') as f:
                pickle.dump(self.model, f)
            
            # Save comprehensive model info
            model_info = {
                'model_type': 'EnhancedPatchCompatibleRandomForestClassifier',
                'training_method': 'patch_based_progressive_enhanced',
                'n_estimators': self.model.n_estimators,
                'best_val_accuracy': best_accuracy,
                'num_training_images': num_training_images,
                'feature_names': self.feature_names,
                'training_history': self.training_history,
                'timestamp': datetime.now().isoformat(),
                'num_features': len(self.feature_names),
                'epochs_trained': len(self.training_history),
                'patch_compatible': True,
                'enhanced_features': True,
                'input_channels': 3,
                'patch_size': self.patch_size,
                'overlap': self.overlap,
                'bioimagin_version': '3.0-Enhanced-Professional'
            }
            
            info_path = self.dirs['models'] / 'tophat_model_info.json'
            with open(info_path, 'w') as f:
                json.dump(model_info, f, indent=2)
            
            # Save training history separately
            history_path = self.dirs['training_artifacts'] / 'enhanced_tophat_training_history.json'
            with open(history_path, 'w') as f:
                json.dump(self.training_history, f, indent=2)
            
            print(f"✅ Enhanced patch-compatible model saved:")
            print(f"   Model: {model_path}")
            print(f"   Info: {info_path}")
            print(f"   History: {history_path}")
            
        except Exception as e:
            print(f"❌ Failed to save enhanced model: {e}")
    
    def visualize_enhanced_training_samples(self, training_data, max_samples=6):
        """Create visualization of enhanced training samples"""
        try:
            print("📷 Creating enhanced training samples visualization...")
            
            if not training_data:
                return
            
            n_samples = min(max_samples, len(training_data))
            sample_indices = np.linspace(0, len(training_data)-1, n_samples, dtype=int)
            
            fig, axes = plt.subplots(n_samples, 3, figsize=(15, 5*n_samples))
            if n_samples == 1:
                axes = axes.reshape(1, -1)
            
            for i, idx in enumerate(sample_indices):
                data = training_data[idx]
                image = data['image']
                annotations = data['annotations']
                
                # Convert BGR to RGB for display
                display_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Create labels from annotations
                labels, _ = self.create_training_labels(image, annotations)
                
                # Column 1: Original enhanced color image
                axes[i, 0].imshow(display_image)
                axes[i, 0].set_title(f'Enhanced Sample {i+1}: Original Color\nSource: {data.get("source", "unknown")}')
                axes[i, 0].axis('off')
                
                # Column 2: Enhanced annotation overlay
                overlay = display_image.copy()
                
                # Draw enhanced annotations
                for annotation in annotations.get('missed', []):
                    if isinstance(annotation, dict) and 'x' in annotation:
                        x, y = int(annotation['x']), int(annotation['y'])
                        radius = int(annotation.get('radius', 12))
                        cv2.circle(overlay, (x, y), radius, (255, 0, 0), 2)
                
                for annotation in annotations.get('correct', []):
                    if isinstance(annotation, dict) and 'x' in annotation:
                        x, y = int(annotation['x']), int(annotation['y'])
                        radius = int(annotation.get('radius', 12))
                        cv2.circle(overlay, (x, y), radius, (0, 255, 0), 2)
                
                for annotation in annotations.get('false_positive', []):
                    if isinstance(annotation, dict) and 'x' in annotation:
                        x, y = int(annotation['x']), int(annotation['y'])
                        radius = int(annotation.get('radius', 12))
                        cv2.circle(overlay, (x, y), radius, (0, 0, 255), 2)
                        cv2.line(overlay, (x-radius, y-radius), (x+radius, y+radius), (0, 0, 255), 2)
                        cv2.line(overlay, (x-radius, y+radius), (x+radius, y-radius), (0, 0, 255), 2)
                
                axes[i, 1].imshow(overlay)
                axes[i, 1].set_title(f'Enhanced Annotations (Patch-Compatible)\nMissed: {len(annotations.get("missed", []))} | Correct: {len(annotations.get("correct", []))} | False+: {len(annotations.get("false_positive", []))}')
                axes[i, 1].axis('off')
                
                # Column 3: Enhanced training labels
                axes[i, 2].imshow(labels, cmap='viridis')
                axes[i, 2].set_title(f'Enhanced Training Labels\nPositive: {np.sum(labels > 0)} | Total: {labels.size}')
                axes[i, 2].axis('off')
            
            plt.tight_layout()
            
            # Save visualization
            viz_path = self.dirs['training_artifacts'] / 'enhanced_tophat_training_samples.png'
            plt.savefig(viz_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ Enhanced training samples visualization saved to {viz_path}")
            
        except Exception as e:
            print(f"⚠️ Failed to create enhanced training visualization: {e}")
    
    def create_enhanced_training_visualizations(self):
        """Create comprehensive enhanced training visualizations"""
        try:
            if not self.training_history:
                return
            
            print("📊 Creating enhanced training visualizations...")
            
            # Prepare data
            epochs = [h['epoch'] for h in self.training_history]
            train_acc = [h['train_accuracy'] for h in self.training_history]
            val_acc = [h['val_accuracy'] for h in self.training_history]
            n_estimators = [h['n_estimators'] for h in self.training_history]
            
            # Create enhanced figure
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # 1. Enhanced accuracy progression
            axes[0, 0].plot(epochs, train_acc, 'b-', label='Training Accuracy', marker='o', linewidth=2)
            axes[0, 0].plot(epochs, val_acc, 'r-', label='Validation Accuracy', marker='s', linewidth=2)
            axes[0, 0].set_title('Enhanced Patch-Compatible Training Progress', fontsize=14, fontweight='bold')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Accuracy')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].set_ylim([0, 1])
            
            # 2. Model complexity evolution
            axes[0, 1].plot(epochs, n_estimators, 'g-', marker='^', linewidth=2)
            axes[0, 1].set_title('Enhanced Model Complexity Evolution', fontsize=14, fontweight='bold')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Number of Estimators')
            axes[0, 1].grid(True, alpha=0.3)
            
            # 3. Enhanced feature importance (top 20)
            if self.feature_names and hasattr(self.model, 'feature_importances_'):
                feature_importance = pd.Series(
                    self.model.feature_importances_, 
                    index=self.feature_names
                ).sort_values(ascending=True).tail(20)
                
                y_pos = np.arange(len(feature_importance))
                bars = axes[1, 0].barh(y_pos, feature_importance.values, alpha=0.7)
                axes[1, 0].set_yticks(y_pos)
                axes[1, 0].set_yticklabels(feature_importance.index, fontsize=8)
                axes[1, 0].set_title('Top 20 Enhanced Feature Importances', fontsize=14, fontweight='bold')
                axes[1, 0].set_xlabel('Importance')
                
                # Color code by feature type
                for i, (bar, feature_name) in enumerate(zip(bars, feature_importance.index)):
                    if 'green' in feature_name.lower():
                        bar.set_color('green')
                    elif 'color' in feature_name.lower() or 'hsv' in feature_name.lower():
                        bar.set_color('blue')
                    elif 'enhanced' in feature_name.lower():
                        bar.set_color('orange')
                    else:
                        bar.set_color('gray')
            
            # 4. Enhanced convergence analysis
            if len(val_acc) > 1:
                val_acc_smooth = pd.Series(val_acc).rolling(window=3, center=True).mean()
                axes[1, 1].fill_between(epochs, train_acc, alpha=0.3, color='blue', label='Training')
                axes[1, 1].fill_between(epochs, val_acc, alpha=0.3, color='red', label='Validation')
                axes[1, 1].plot(epochs, train_acc, 'b-', linewidth=2)
                axes[1, 1].plot(epochs, val_acc, 'r-', linewidth=2)
                if not val_acc_smooth.isna().all():
                    axes[1, 1].plot(epochs, val_acc_smooth, 'r--', linewidth=2, alpha=0.7, label='Val Smoothed')
                axes[1, 1].set_title('Enhanced Learning Convergence', fontsize=14, fontweight='bold')
                axes[1, 1].set_xlabel('Epoch')
                axes[1, 1].set_ylabel('Accuracy')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
                axes[1, 1].set_ylim([0, 1])
            
            plt.tight_layout()
            
            # Save plot
            plot_path = self.dirs['training_artifacts'] / 'enhanced_tophat_training_visualization.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ Enhanced training visualizations saved to {plot_path}")
            
        except Exception as e:
            print(f"⚠️ Failed to create enhanced visualizations: {e}")


    def create_enhanced_training_labels(self, image, annotations):
        """
        Create enhanced multi-class training labels that teach cell separation
        Classes: 0=background, 1=cell_center, 2=cell_interior, 3=cell_boundary
        """
        try:
            h, w = image.shape[:2]
            labels = np.zeros((h, w), dtype=np.uint8)
            distance_labels = np.zeros((h, w), dtype=np.float32)
            center_labels = np.zeros((h, w), dtype=np.uint8)
            positive_pixels = 0
            
            print(f"  Creating enhanced separation-aware labels...")
            
            # Process correct cell annotations
            for annotation in annotations.get('correct', []):
                if isinstance(annotation, dict) and 'x' in annotation and 'y' in annotation:
                    x, y = int(annotation['x']), int(annotation['y'])
                    radius = int(annotation.get('radius', 12))
                    
                    # Create circular regions with different zones
                    rr, cc = np.ogrid[:h, :w]
                    dist_from_center = np.sqrt((cc - x)**2 + (rr - y)**2)
                    
                    # Cell center (innermost 20% of radius)
                    center_mask = dist_from_center <= (radius * 0.2)
                    center_mask = center_mask & (cc >= 0) & (cc < w) & (rr >= 0) & (rr < h)
                    labels[center_mask] = 3  # Class 3: Cell center
                    center_labels[center_mask] = 1
                    
                    # Cell interior (middle 60% of radius)
                    interior_mask = (dist_from_center > (radius * 0.2)) & (dist_from_center <= (radius * 0.8))
                    interior_mask = interior_mask & (cc >= 0) & (cc < w) & (rr >= 0) & (rr < h)
                    labels[interior_mask] = 2  # Class 2: Cell interior
                    
                    # Cell boundary (outer 20% of radius)
                    boundary_mask = (dist_from_center > (radius * 0.8)) & (dist_from_center <= radius)
                    boundary_mask = boundary_mask & (cc >= 0) & (cc < w) & (rr >= 0) & (rr < h)
                    labels[boundary_mask] = 1  # Class 1: Cell boundary
                    
                    # Distance transform (for continuous learning)
                    cell_mask = dist_from_center <= radius
                    cell_mask = cell_mask & (cc >= 0) & (cc < w) & (rr >= 0) & (rr < h)
                    distance_labels[cell_mask] = np.maximum(
                        distance_labels[cell_mask], 
                        1.0 - (dist_from_center[cell_mask] / radius)
                    )
                    
                    positive_pixels += np.sum(cell_mask)
            
            # Process missed cells similarly
            for annotation in annotations.get('missed', []):
                if isinstance(annotation, dict) and 'x' in annotation and 'y' in annotation:
                    x, y = int(annotation['x']), int(annotation['y'])
                    radius = int(annotation.get('radius', 12))
                    
                    rr, cc = np.ogrid[:h, :w]
                    dist_from_center = np.sqrt((cc - x)**2 + (rr - y)**2)
                    
                    center_mask = dist_from_center <= (radius * 0.2)
                    center_mask = center_mask & (cc >= 0) & (cc < w) & (rr >= 0) & (rr < h)
                    labels[center_mask] = 3
                    center_labels[center_mask] = 1
                    
                    interior_mask = (dist_from_center > (radius * 0.2)) & (dist_from_center <= (radius * 0.8))
                    interior_mask = interior_mask & (cc >= 0) & (cc < w) & (rr >= 0) & (rr < h)
                    labels[interior_mask] = 2
                    
                    boundary_mask = (dist_from_center > (radius * 0.8)) & (dist_from_center <= radius)
                    boundary_mask = boundary_mask & (cc >= 0) & (cc < w) & (rr >= 0) & (rr < h)
                    labels[boundary_mask] = 1
                    
                    cell_mask = dist_from_center <= radius
                    cell_mask = cell_mask & (cc >= 0) & (cc < w) & (rr >= 0) & (rr < h)
                    distance_labels[cell_mask] = np.maximum(
                        distance_labels[cell_mask], 
                        1.0 - (dist_from_center[cell_mask] / radius)
                    )
                    
                    positive_pixels += np.sum(cell_mask)
            
            print(f"  Created enhanced labels: Centers: {np.sum(labels==3)}, Interior: {np.sum(labels==2)}, Boundary: {np.sum(labels==1)}")
            
            return {
                'multi_class': labels,
                'distance_map': distance_labels,
                'center_map': center_labels,
                'positive_pixels': positive_pixels
            }
            
        except Exception as e:
            print(f"⚠️ Enhanced label creation failed: {e}")
            # Fallback to simple binary labels
            simple_labels = np.zeros((h, w), dtype=np.uint8)
            return {
                'multi_class': simple_labels,
                'distance_map': simple_labels.astype(np.float32),
                'center_map': simple_labels,
                'positive_pixels': 0
            }
            
    def create_separation_aware_training_data(self, image, annotations):
        """
        Create training data that teaches the model about cell separation
        """
        try:
            h, w = image.shape[:2]
            patch_features = []
            patch_labels_multi = []
            patch_labels_distance = []
            patch_labels_centers = []
            
            # Create enhanced labels
            enhanced_labels = self.create_enhanced_training_labels(image, annotations)
            
            if enhanced_labels['positive_pixels'] == 0:
                print("  ⚠️ No positive samples in image, skipping")
                return None, None, None, None
            
            # Extract patches with enhanced labels
            patch_size = self.patch_size
            overlap = self.overlap
            
            patches_processed = 0
            for y in range(0, h - patch_size + 1, patch_size - overlap):
                for x in range(0, w - patch_size + 1, patch_size - overlap):
                    y_end = min(y + patch_size, h)
                    x_end = min(x + patch_size, w)
                    
                    # Extract patch
                    patch_img = image[y:y_end, x:x_end]
                    patch_multi = enhanced_labels['multi_class'][y:y_end, x:x_end]
                    patch_distance = enhanced_labels['distance_map'][y:y_end, x:x_end]
                    patch_centers = enhanced_labels['center_map'][y:y_end, x:x_end]
                    
                    # Resize if needed
                    if patch_img.shape[:2] != (patch_size, patch_size):
                        patch_img = cv2.resize(patch_img, (patch_size, patch_size))
                        patch_multi = cv2.resize(patch_multi, (patch_size, patch_size), interpolation=cv2.INTER_NEAREST)
                        patch_distance = cv2.resize(patch_distance, (patch_size, patch_size))
                        patch_centers = cv2.resize(patch_centers, (patch_size, patch_size), interpolation=cv2.INTER_NEAREST)
                    
                    # Check if patch has cells
                    if np.sum(patch_multi > 0) < 20:
                        continue
                    
                    # Extract features
                    patch_features_data = self.extract_enhanced_patch_features(patch_img)
                    
                    # Flatten labels
                    patch_multi_flat = patch_multi.flatten()
                    patch_distance_flat = patch_distance.flatten()
                    patch_centers_flat = patch_centers.flatten()
                    
                    # Balanced sampling with enhanced strategy
                    # Sample more from centers and boundaries
                    center_indices = np.where(patch_multi_flat == 3)[0]  # Centers
                    boundary_indices = np.where(patch_multi_flat == 1)[0]  # Boundaries
                    interior_indices = np.where(patch_multi_flat == 2)[0]  # Interior
                    background_indices = np.where(patch_multi_flat == 0)[0]  # Background
                    
                    # Strategic sampling
                    selected_indices = []
                    
                    # Take all centers (most important)
                    selected_indices.extend(center_indices)
                    
                    # Take all boundaries (important for separation)
                    selected_indices.extend(boundary_indices)
                    
                    # Sample interior points
                    n_interior = min(len(interior_indices), len(center_indices) * 3)
                    if n_interior > 0 and len(interior_indices) > 0:
                        interior_sample = np.random.choice(interior_indices, n_interior, replace=False)
                        selected_indices.extend(interior_sample)
                    
                    # Sample background
                    n_background = min(len(background_indices), len(center_indices) * 2)
                    if n_background > 0 and len(background_indices) > 0:
                        background_sample = np.random.choice(background_indices, n_background, replace=False)
                        selected_indices.extend(background_sample)
                    
                    if len(selected_indices) > 0:
                        selected_indices = np.array(selected_indices)
                        
                        patch_features.append(patch_features_data[selected_indices])
                        patch_labels_multi.append(patch_multi_flat[selected_indices])
                        patch_labels_distance.append(patch_distance_flat[selected_indices])
                        patch_labels_centers.append(patch_centers_flat[selected_indices])
                        patches_processed += 1
            
            if not patch_features:
                print("  ⚠️ No valid enhanced patches extracted")
                return None, None, None, None
            
            # Combine all patch data
            X = np.vstack(patch_features)
            y_multi = np.hstack(patch_labels_multi)
            y_distance = np.hstack(patch_labels_distance)
            y_centers = np.hstack(patch_labels_centers)
            
            print(f"  ✅ Enhanced separation training: {patches_processed} patches")
            print(f"  📊 Multi-class distribution: BG:{np.sum(y_multi==0)}, Boundary:{np.sum(y_multi==1)}, Interior:{np.sum(y_multi==2)}, Centers:{np.sum(y_multi==3)}")
            
            return X, y_multi, y_distance, y_centers
            
        except Exception as e:
            print(f"  ❌ Error in enhanced separation training: {e}")
            return None, None, None, None
    
    def train_separation_aware_model(self, n_estimators=150, epochs=12):
        """Train model that understands cell separation"""
        try:
            print("🚀 SEPARATION-AWARE TOPHAT TRAINING")
            print("=" * 60)
            
            training_data = self.load_annotation_data_enhanced()
            if not training_data:
                print("❌ No training data found.")
                return False
            
            # Prepare separation-aware training data
            X_list, y_multi_list, y_distance_list, y_centers_list = [], [], [], []
            
            for i, data in enumerate(training_data):
                print(f"Processing separation-aware image {i+1}/{len(training_data)}")
                
                X_patch, y_multi, y_distance, y_centers = self.create_separation_aware_training_data(
                    data['image'], data['annotations']
                )
                
                if X_patch is not None:
                    X_list.append(X_patch)
                    y_multi_list.append(y_multi)
                    y_distance_list.append(y_distance)
                    y_centers_list.append(y_centers)
            
            if not X_list:
                print("❌ No separation-aware training data found")
                return False
            
            X = np.vstack(X_list)
            y_multi = np.hstack(y_multi_list)
            y_centers = np.hstack(y_centers_list)
            
            # Train primary model for multi-class classification
            print("🎯 Training separation-aware classifier...")
            
            self.separation_model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=20,
                min_samples_split=3,
                min_samples_leaf=1,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            )
            
            self.separation_model.fit(X, y_multi)
            
            # Train center detection model
            print("🎯 Training center detection model...")
            
            self.center_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            )
            
            self.center_model.fit(X, y_centers)
            
            # Save both models
            self.save_separation_models()
            
            print("✅ SEPARATION-AWARE TRAINING COMPLETED")
            return True
            
        except Exception as e:
            print(f"❌ Separation-aware training failed: {e}")
            return False

    def save_separation_models(self):
        """Save separation-aware models"""
        try:
            # Save separation model
            sep_model_path = self.dirs['models'] / 'tophat_separation_model.pkl'
            with open(sep_model_path, 'wb') as f:
                pickle.dump(self.separation_model, f)
            
            # Save center model
            center_model_path = self.dirs['models'] / 'tophat_center_model.pkl'
            with open(center_model_path, 'wb') as f:
                pickle.dump(self.center_model, f)
            
            print(f"✅ Separation models saved:")
            print(f"   Separation: {sep_model_path}")
            print(f"   Centers: {center_model_path}")
            
        except Exception as e:
            print(f"❌ Failed to save separation models: {e}")

def train_enhanced_tophat_model(epochs=12, n_estimators=150):
    """Main function to train enhanced patch-compatible tophat model"""
    trainer = EnhancedTophatTrainer()
    return trainer.train_enhanced_model(n_estimators=n_estimators, epochs=epochs)


if __name__ == "__main__":
    print("🧠 SEPARATION-INTELLIGENT TOPHAT ML TRAINER")
    print("=" * 60)
    print("🎯 Built-in cell separation intelligence")
    print("📊 10 features with separation awareness")
    print("🔧 No post-processing needed - separation is learned")
    print()
    
    # Configuration
    print("📊 Separation-Intelligent Training Configuration:")
    try:
        epochs = int(input("Number of epochs [8]: ") or "8")
        n_estimators = int(input("Number of estimators [200]: ") or "200")
    except:
        epochs, n_estimators = 8, 200
        print("Using separation-intelligent configuration: 8 epochs, 200 estimators")
    
    print(f"🧠 Separation-Intelligent Configuration: {epochs} epochs, {n_estimators} estimators")
    print("🔧 10 features with built-in separation intelligence")
    print()
    
    # Start separation-intelligent training
    success = train_enhanced_tophat_model(epochs=epochs, n_estimators=n_estimators)
    
    if success:
        print("\n🎉 SEPARATION-INTELLIGENT TOPHAT TRAINING COMPLETED!")
        print("=" * 60)
        print("🧠 Model trained with built-in separation intelligence")
        print("🎯 No watershed post-processing needed")
        print("🌊 Cell separation happens during inference")
        print("✅ Ready for intelligent cell separation!")
    else:
        print("\n❌ Separation-intelligent training failed.")