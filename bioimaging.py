#!/usr/bin/env python3
"""
BIOIMAGIN Professional Wolffia Analysis System - DEPLOYMENT VERSION
Streamlined, professional-grade implementation based on python_for_microscopists best practices
Author: BIOIMAGIN Professional Team
"""

import base64
import json
import os
import pickle
import traceback
import uuid
import warnings
from datetime import datetime
from io import BytesIO
from pathlib import Path

import cv2
import matplotlib

matplotlib.use('Agg')
import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy import ndimage
from scipy import ndimage as ndi
from skimage import (
    color,
    data,
    exposure,
    feature,
    filters,
    graph,
    measure,
    morphology,
    restoration,
    segmentation,
)
from skimage.draw import disk
from skimage.feature import peak_local_max, shape_index
from skimage.restoration import (
    denoise_bilateral,
    denoise_tv_chambolle,
    denoise_wavelet,
    estimate_sigma,
)
from skimage.segmentation import clear_border, watershed
from skimage.util import img_as_float, random_noise
from sklearn.ensemble import RandomForestClassifier

from wolffia_cnn_model import GreenEnhancedPreprocessor

# Suppress warnings for cleaner output
# Suppress warnings for cleaner output and better performance
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Memory optimization: Configure numpy for better performance
np.seterr(divide='ignore', invalid='ignore')
matplotlib.rcParams['figure.max_open_warning'] = 0  # Disable figure limit warnings
# Optional imports with graceful fallback
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False




try:
    import celldetection as cd
    CELLDETECTION_AVAILABLE = True
    print("✅ CellDetection available")
except ImportError:
    CELLDETECTION_AVAILABLE = False
    print("⚠️ CellDetection not available - using classical methods")


class WolffiaAnalyzer:
    """
    Professional Wolffia Analysis System - Deployment Ready
    Streamlined implementation using proven microscopist patterns
    """
    
    def __init__(self):
        """Initialize with enhanced configuration for professional analysis"""
        self.setup_directories()
        
        # Enhanced parameters for Wolffia analysis
        self.min_cell_area = 30
        self.max_cell_area = 500
        self.pixel_to_micron = 0.5
        self.preprocessor = GreenEnhancedPreprocessor()
        
        # Enhanced biomass calculation parameters
        self.wolffia_density = 1.1  # g/cm³ (typical plant cell density)
        self.cell_thickness_micron = 15  # Average Wolffia cell thickness
        self.dry_weight_ratio = 0.2  # Dry/wet weight ratio for biomass
        
        # Color analysis parameters for wavelength analysis
        self.green_wavelength_range = (495, 570)  # nm - Green light spectrum
        self.chlorophyll_peak = 530  # nm - Peak chlorophyll absorption
        
        # Time-series tracking storage
        self.time_series_data = {}
        
        # Models loaded on demand
        self._tophat_model = None
        self._cnn_model = None
        self._celldetection_model = None
        self._device = None
        
        # Device property for CUDA support
        self._device = 'cuda' if torch.cuda.is_available() and TORCH_AVAILABLE else 'cpu'
        
        # Status properties for frontend compatibility
        self.wolffia_cnn_available = False
        self.celldetection_available = False
        self.tophat_model = None
        self.wolffia_cnn_model = None
        
        print("✅ Enhanced WolffiaAnalyzer initialized - Professional Analysis Ready")
    
    def setup_directories(self):
        """Setup required directories"""
        self.dirs = {
            'results': Path('results'),
            'models': Path('models'),
            'uploads': Path('uploads'),
            'annotations': Path('annotations'),
            'time_series': Path('results/time_series')  # New for time tracking
        }
        
        for path in self.dirs.values():
            path.mkdir(parents=True, exist_ok=True)
    
    @property 
    def device(self):
        """Consistent device detection with better error handling"""
        if self._device is None:
            if TORCH_AVAILABLE:
                try:
                    import torch
                    if torch.cuda.is_available():
                        self._device = 'cuda'
                        print(f"🎯 Using CUDA device: {torch.cuda.get_device_name()}")
                    else:
                        self._device = 'cpu'
                        print("🎯 Using CPU device")
                except Exception as e:
                    print(f"⚠️ Device detection error: {e}")
                    self._device = 'cpu'
            else:
                self._device = 'cpu'
        return self._device
    
    @property
    def celldetection_model(self):
        """Lazy loading for CellDetection AI model"""
        if self._celldetection_model is None:
            self.initialize_celldetection_model()
        return self._celldetection_model
    

    def initialize_celldetection_model(self):
        """Initialize CellDetection model for AI-powered detection (called lazily)"""
        try:
            if not CELLDETECTION_AVAILABLE:
                self._celldetection_model = None
                self.celldetection_available = False
                print("⚠️ CellDetection not available - using classical methods only")
                return
            
            print(f"🎯 CellDetection device: {self.device}")
            
            # Load pretrained model only when needed
            model_name = 'ginoro_CpnResNeXt101UNet-fbe875f1a3e5ce2c'
            print(f"📥 Loading CellDetection model: {model_name}")
            
            self._celldetection_model = cd.fetch_model(model_name, check_hash=True)
            self._celldetection_model = self._celldetection_model.to(self.device)
            self._celldetection_model.eval()
            
            self.celldetection_available = True
            print("✅ CellDetection model loaded successfully")
            
        except Exception as e:
            print(f"❌ Failed to initialize CellDetection model: {e}")
            self._celldetection_model = None
            self.celldetection_available = False

        
    
    def analyze_image(self, image_path, use_tophat=True, use_cnn=True, use_celldetection=False, 
                     timestamp=None, image_series_id=None):
        """
        Enhanced main analysis method with comprehensive biomass and color analysis
        """
        try:
            # Load and preprocess image
            img = cv2.imread(str(image_path))
            if img is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Enhanced color image processing
            color_img = img.copy()
            
            # Advanced green content and wavelength analysis
            color_analysis = color_img
            
            # Create enhanced grayscale
            enhanced_gray = color_img
            
            # Get results from available methods with enhanced patch processing
            results = []
            
            # Method 1: Enhanced Watershed with patch processing
            watershed_pipeline = {}  # Initialize to avoid undefined variable errors
            try:
                watershed_result = self.watershed_segmentation(color_img, return_pipeline=True)
                if isinstance(watershed_result, tuple) and len(watershed_result) == 2:
                    watershed_labels, watershed_pipeline = watershed_result
                else:
                    watershed_labels = watershed_result
                results.append(('watershed', watershed_labels))
            except Exception as e:
                print(f"⚠️ Watershed method failed: {e}")
                # Try fallback without pipeline
                try:
                    watershed_labels = self.color_aware_watershed_segmentation(color_img) 
                    results.append(('watershed', watershed_labels))
                except Exception as e2:
                    print(f"⚠️ Watershed fallback failed: {e2}")

            
            # Method 2: Enhanced Tophat ML with patches
            if use_tophat and self.load_tophat_model():
                tophat_result = self.enhanced_patch_tophat(color_img, enhanced_gray)
                results.append(('tophat', tophat_result))
            
            # Method 3: Enhanced CNN with existing patch system
            if use_cnn and TORCH_AVAILABLE and self.load_cnn_model():
                cnn_result = self.cnn_detection(enhanced_gray, color_img)
                results.append(('cnn', cnn_result))
            
            # Method 4: Enhanced CellDetection with patch processing
            if use_celldetection and CELLDETECTION_AVAILABLE:
                celldetection_result = self.enhanced_patch_celldetection(color_img)
                results.append(('celldetection', celldetection_result))
            
            # Intelligent fusion of results
            final_result = self.fuse_detection_results(results, color_img.shape[:2])
            
            # Enhanced cell property extraction with biomass calculations
            cell_data = self.extract_enhanced_cell_properties(final_result, enhanced_gray, color_img)
            
            # Calculate comprehensive biomass and statistics
            biomass_analysis = self.calculate_comprehensive_biomass(cell_data, color_img.shape[:2])
            
            # Enhanced color wavelength analysis
            enhanced_color_analysis = self.enhanced_wavelength_analysis(color_img, final_result)
            
            # Time-series analysis if applicable
            time_series_analysis = None
            if timestamp and image_series_id:
                time_series_analysis = self.update_time_series(
                    image_series_id, timestamp, cell_data, biomass_analysis, enhanced_color_analysis
                )
            
            # Create comprehensive visualization
            # Build visualizations
            vis_data = self.create_comprehensive_visualization(
                img, final_result, cell_data, enhanced_color_analysis, biomass_analysis
            )

            # ✅ Inject enhanced pipeline steps for watershed if available
            if isinstance(final_result, np.ndarray) and np.max(final_result) > 0:
                try:
                    # Collect intermediate images (example: grayscale, labeled overlay)
                    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    denoised = cv2.fastNlMeansDenoising(gray_img, h=10)
                    overlay = color.label2rgb(final_result, image=img, bg_label=0, alpha=0.4)

                    def to_b64(image):
                        if image.ndim == 2:
                            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                        elif image.shape[2] == 3:
                            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        _, buffer = cv2.imencode('.png', image)
                        return base64.b64encode(buffer).decode('utf-8')

                    if watershed_pipeline:  # Only create pipeline steps if watershed succeeded
                        vis_data['pipeline_steps'] = {
                            'step_descriptions': {
                                'Original': 'Original uploaded image',
                                'Denoised': 'Denoised image (TV-Chambolle)',
                                'Green_enhanced': 'Green dominance and contrast enhanced',
                                'Green_mask': 'Binary mask of green regions',
                                'Shape_index': 'Topological shape descriptor map',
                                'Watershed': 'Final segmented watershed result',
                                'Shape_index_3d': '3D projection of shape index over intensity surface'
                            },
                            'individual_steps': {
                                'Original': to_b64((watershed_pipeline['Original'] * 255).astype(np.uint8)),
                                'Denoised': to_b64((watershed_pipeline['Denoised'] * 255).astype(np.uint8)),
                                'Green_enhanced': to_b64((watershed_pipeline['Green_enhanced'] * 255).astype(np.uint8)),
                                'Green_mask': to_b64((watershed_pipeline['Green_mask'] * 255).astype(np.uint8)),
                                'Shape_index': to_b64((watershed_pipeline['Shape_index'] * 255).astype(np.uint8)),
                                'Watershed': to_b64((color.label2rgb(watershed_pipeline['Watershed'], bg_label=0, alpha=0.5) * 255).astype(np.uint8)),
                                'Shape_index_3d': watershed_pipeline['Shape_index_3d']  # already base64 PNG string
                            },
                            'pipeline_overview': to_b64((color.label2rgb(watershed_pipeline['Watershed'], image=color_img, bg_label=0, alpha=0.5) * 255).astype(np.uint8)),
                            'step_count': 7
                        }
                    else:
                        print("⚠️ Watershed pipeline not available - skipping pipeline visualization")


                except Exception as viz_err:
                    print(f"⚠️ Failed to generate pipeline steps: {viz_err}")

            
            # Compile enhanced results
            analysis_result = {
                # Legacy compatibility
                'total_cells': len(cell_data),
                'total_area': sum(cell.get('area_microns_sq', cell.get('area', 0)) for cell in cell_data),
                'average_area': np.mean([cell.get('area_microns_sq', cell.get('area', 0)) for cell in cell_data]) if cell_data else 0,
                'cells': cell_data,
                'labeled_image_path': str(vis_data.get('main_visualization')),
                'method_used': [method for method, _ in results],
                'processing_time': 0,
                
                # Enhanced analysis results
                'detection_results': {
                    'detection_method': f"Enhanced Multi-Method Analysis ({len(results)} methods)",
                    'cells_detected': len(cell_data),
                    'total_area': sum(cell.get('area_microns_sq', cell.get('area', 0)) for cell in cell_data),
                    'cells_data': cell_data
                },
                
                # Comprehensive quantitative analysis
                'quantitative_analysis': {
                    'biomass_analysis': biomass_analysis,
                    'color_analysis': enhanced_color_analysis,
                    'morphometric_analysis': self.calculate_morphometric_statistics(cell_data),
                    'spatial_analysis': self.calculate_spatial_distribution(cell_data, color_img.shape[:2]),
                    'health_assessment': self.assess_culture_health(cell_data, enhanced_color_analysis)
                },
                
                # Enhanced visualizations
                'visualizations': vis_data,
                
                # Time-series data if available
                'time_series_analysis': time_series_analysis
            }
            
            
            
        
            # ✅ Build method_results block for per-method tabbed visualization
            method_results = {}

            for method_name, label_img in results:
                if not isinstance(label_img, np.ndarray) or np.max(label_img) == 0:
                    continue

                try:
                    # Extract stats per method
                    method_cells = self.extract_enhanced_cell_properties(label_img, enhanced_gray, color_img)
                    # Handle both old and new cell data formats
                    total_area = sum(cell.get('area_microns_sq', cell.get('area', 0)) for cell in method_cells)
                    areas = [cell.get('area_microns_sq', cell.get('area', 0)) for cell in method_cells]
                    avg_area = np.mean(areas) if areas else 0

                    # Create base64 overlay
                    rgb_image = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB) if color_img.shape[2] == 3 else color_img
                    overlay = color.label2rgb(label_img, image=rgb_image, bg_label=0, alpha=0.5)

                    # Boost contrast artificially if mean intensity is low
                    if np.mean(rgb_image) < 60:
                        overlay = np.clip(overlay * 1.5, 0, 1)

                    # Convert to uint8 safely
                    overlay_uint8 = (overlay * 255).astype(np.uint8)
                    _, buffer = cv2.imencode('.png', overlay_uint8)
                    viz_b64 = base64.b64encode(buffer).decode('utf-8')

                    method_results[method_name] = {
                        'method_name': method_name.capitalize(),
                        'cells_detected': len(method_cells),
                        'total_area': total_area,
                        'average_area': avg_area,
                        'visualization_b64': viz_b64,
                        'cells_data': method_cells  # Add detailed cell data per method
                    }

                except Exception as m_err:
                    print(f"⚠️ Failed to build method result for {method_name}: {m_err}")

            # Determine best method by highest cell count
            if method_results:
                best_method_key = max(method_results, key=lambda k: method_results[k]['cells_detected'])
                analysis_result['detection_results']['method_results'] = method_results
                analysis_result['detection_results']['best_method'] = best_method_key



            return analysis_result
            
        except Exception as e:
            print(f"❌ Enhanced analysis failed: {str(e)}")
            return self.get_error_result(str(e))
    
    
    
    
    def watershed_segmentation(self, image, return_pipeline=True):
        """Run green-enhanced watershed and return label image + optional pipeline steps."""
        pipeline = {}

        # Step 1: Convert input to float RGB
        image = np.asarray(image)
        if image.ndim == 2:
            image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        elif image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original = img_as_float(image)
        pipeline["Original"] = original

        # Step 2: Denoising
        sigma_est = estimate_sigma(original, channel_axis=-1, average_sigmas=True)
        print(f"Estimated noise σ = {sigma_est:.3f}")
        denoised = denoise_tv_chambolle(original, weight=0.1, channel_axis=-1)
        pipeline["Denoised"] = denoised

        # Step 3: Green-Enhanced Detection
        b, g, r = denoised[:, :, 0].astype(np.float32), denoised[:, :, 1].astype(np.float32), denoised[:, :, 2].astype(np.float32)

        # Normalize RGB and compute green dominance
        brightness = r + g + b + 1e-6
        green_ratio = g / brightness
        green_dominance = g - np.maximum(r, b)
        green_confidence = np.clip(green_ratio * green_dominance * 3, 0, 1)
        green_enhanced_rgb = np.power(green_confidence, 0.2)
        pipeline["Green_enhanced"] = green_enhanced_rgb

        green_mask = green_enhanced_rgb > 0.02
        pipeline["Green_mask"] = green_mask

        # Step 4: Shape Index
        shape_idx = shape_index((green_enhanced_rgb * 255).astype(np.float32))
        pipeline["Shape_index"] = shape_idx

        # Step 5: Watershed Segmentation
        distance = ndi.distance_transform_edt(green_mask)
        coords = peak_local_max(distance, footprint=np.ones((3, 3)), labels=green_mask)
        markers = np.zeros_like(distance, dtype=bool)
        markers[tuple(coords.T)] = True
        markers, _ = ndi.label(markers)
        labels = watershed(-distance, markers, mask=green_mask)
        pipeline["Watershed"] = labels

        # Optional visualization outputs
        if return_pipeline:
            # Add shape index 3D visualization
            pipeline["Shape_index_3d"] = self.render_shape_index_3d(original[:, :, 1], shape_idx)

            # Save composite pipeline plot for archive/debug
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            images = [
                (original, "Original"),
                (denoised, "Denoised"),
                (green_enhanced_rgb, "Green Enhanced"),
                (green_mask, "Green Mask"),
                (shape_idx, "Shape Index"),
                (labels, "Watershed")
            ]
            for ax, (img, title) in zip(axes.ravel(), images):
                cmap = 'nipy_spectral' if "Watershed" in title else 'gray' if img.ndim == 2 else None
                ax.imshow(img, cmap=cmap)
                ax.set_title(title)
                ax.axis('off')
            plt.tight_layout()
            fig_path = self.dirs['results'] / f"pipeline_watershed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(fig_path)
            plt.close()
            print(f"📦 Saved pipeline figure: {fig_path}")

            return labels, pipeline

        return labels



    def render_shape_index_3d(self, image, shape_idx_map, delta=0.05, smooth_sigma=0.5):
        """
        Render a 3D visualization of the shape index and return it as a base64 PNG string.
        """
        import base64
        import io

        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib import cm
        from scipy.ndimage import gaussian_filter

        # Smooth the shape index for less noise
        s_smooth = gaussian_filter(shape_idx_map, sigma=smooth_sigma)

        # Extract relevant points
        point_y, point_x = np.where(np.abs(shape_idx_map - 1) < delta)
        point_z = image[point_y, point_x]

        point_y_s, point_x_s = np.where(np.abs(s_smooth - 1) < delta)
        point_z_s = image[point_y_s, point_x_s]

        x, y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))

        # Create the plot
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(x, y, image, cmap=cm.gray, alpha=0.6, linewidth=0)

        ax.scatter(point_x, point_y, point_z, color='blue', label='|s - 1|<0.05', alpha=0.75, s=10)
        ax.scatter(point_x_s, point_y_s, point_z_s, color='green', label='|s\' - 1|<0.05', alpha=0.75, s=10)

        ax.set_title("3D Shape Index Detection")
        ax.legend()
        ax.axis('off')
        plt.tight_layout()

        # Convert to base64
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        b64_img = base64.b64encode(buf.getvalue()).decode("utf-8")
        return b64_img


    def enhanced_patch_tophat(self, color_img, enhanced_gray):
        """Enhanced tophat with optimized patch processing"""
        try:
            if self._tophat_model is None:
                return np.zeros(color_img.shape[:2], dtype=np.int32)
            
            print("🎯 Enhanced Patch-based Tophat Analysis...")
            
            # Optimized parameters for speed vs accuracy
            patch_size = 256
            overlap = 32  # Reduced overlap for speed (was 64)
            h, w = color_img.shape[:2]
            
            # Pre-allocate for better memory performance
            prediction_map = np.zeros((h, w), dtype=np.float32)
            count_map = np.zeros((h, w), dtype=np.float32)
            
            # Quick green pre-filtering to skip empty patches
            hsv = cv2.cvtColor(color_img, cv2.COLOR_BGR2HSV)
            green_mask = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
            
            patches_processed = 0
            patches_skipped = 0
            
            # Optimized patch processing loop
            for y in range(0, h - patch_size + 1, patch_size - overlap):
                for x in range(0, w - patch_size + 1, patch_size - overlap):
                    y_end = min(y + patch_size, h)
                    x_end = min(x + patch_size, w)
                    
                    # Quick green content check - skip if no green
                    patch_green = green_mask[y:y_end, x:x_end]
                    green_ratio = np.sum(patch_green > 0) / patch_green.size
                    
                    if green_ratio < 0.01:  # Skip patches with <1% green content
                        patches_skipped += 1
                        continue
                    
                    # Extract patch (avoid unnecessary resize when possible)
                    patch = color_img[y:y_end, x:x_end]
                    original_patch_shape = patch.shape[:2]
                    
                    # Only resize if absolutely necessary
                    if original_patch_shape != (patch_size, patch_size):
                        patch = cv2.resize(patch, (patch_size, patch_size))
                    
                    # Fast feature extraction and prediction
                    try:
                        features = self.extract_ml_features(patch)
                        predictions = self._tophat_model.predict_proba(
                            features.reshape(-1, features.shape[-1])
                        )
                        
                        # Get positive class probabilities
                        if predictions.shape[1] > 1:
                            patch_pred = predictions[:, 1].reshape(patch_size, patch_size)
                        else:
                            patch_pred = predictions[:, 0].reshape(patch_size, patch_size)
                        
                        # Resize back only if needed
                        if patch_pred.shape != original_patch_shape:
                            patch_pred = cv2.resize(patch_pred, (x_end - x, y_end - y))
                        
                        # Accumulate predictions efficiently
                        prediction_map[y:y_end, x:x_end] += patch_pred
                        count_map[y:y_end, x:x_end] += 1
                        patches_processed += 1
                        
                    except Exception as patch_error:
                        print(f"⚠️ Patch failed at ({x},{y}): {patch_error}")
                        continue
            
            print(f"📊 Processed {patches_processed} patches, skipped {patches_skipped} empty patches")
            
            # Avoid division by zero
            count_map[count_map == 0] = 1
            averaged_pred = prediction_map / count_map
            
            # Optimized thresholding and labeling
            binary_mask = (averaged_pred > 0.5).astype(np.uint8)
            
            # Quick morphological cleanup
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
            
            # Connected components
            num_labels, labeled_img = cv2.connectedComponents(binary_mask)
            
            # Fast green filtering
            filtered_labels = self.apply_green_filtering_fast(labeled_img, color_img)
            
            print(f"✅ Enhanced Tophat: {np.max(filtered_labels)} cells detected")
            return filtered_labels
            
        except Exception as e:
            print(f"⚠️ Enhanced patch tophat failed: {e}")
            return self.tophat_ml_detection(enhanced_gray)

    def apply_green_filtering_fast(self, labeled_img, color_img):
        """Fast green filtering without complex calculations"""
        try:
            if np.max(labeled_img) == 0:
                return labeled_img
            
            # Quick HSV conversion
            hsv = cv2.cvtColor(color_img, cv2.COLOR_BGR2HSV)
            green_mask = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
            
            # Fast region filtering
            filtered_img = np.zeros_like(labeled_img)
            new_label = 1
            
            for region_label in range(1, np.max(labeled_img) + 1):
                region_mask = labeled_img == region_label
                
                # Quick green content check
                green_pixels = np.sum(green_mask[region_mask] > 0)
                total_pixels = np.sum(region_mask)
                
                if total_pixels > 0 and green_pixels / total_pixels > 0.1:  # >10% green
                    filtered_img[region_mask] = new_label
                    new_label += 1
            
            return filtered_img
            
        except Exception as e:
            print(f"⚠️ Fast green filtering failed: {e}")
            return labeled_img
    
    def enhanced_patch_celldetection(self, color_img):
        """Enhanced CellDetection with patch processing (like CNN)"""
        try:
            if not CELLDETECTION_AVAILABLE or self.celldetection_model is None:
                return np.zeros(color_img.shape[:2], dtype=np.int32)
            
            print("🧠 Enhanced Patch-based CellDetection Analysis...")
            
            # Patch configuration similar to CNN
            patch_size = 256
            overlap = 64
            h, w = color_img.shape[:2]
            
            final_labels = np.zeros((h, w), dtype=np.int32)
            label_counter = 1
            
            # Process overlapping patches
            for y in range(0, h - patch_size + 1, patch_size - overlap):
                for x in range(0, w - patch_size + 1, patch_size - overlap):
                    y_end = min(y + patch_size, h)
                    x_end = min(x + patch_size, w)
                    
                    # Extract patch
                    patch = color_img[y:y_end, x:x_end]
                    
                    if patch.shape[:2] != (patch_size, patch_size):
                        patch = cv2.resize(patch, (patch_size, patch_size))
                    
                    # Convert to tensor for CellDetection
                    x_tensor = cd.to_tensor(patch, transpose=True, device=self.device, dtype=torch.float32)
                    x_tensor = x_tensor / 255.0
                    x_tensor = x_tensor.unsqueeze(0)
                    
                    # Run CellDetection inference
                    with torch.no_grad():
                        outputs = self.celldetection_model(x_tensor)
                        contours = outputs.get('contours', [])
                        scores = outputs.get('scores', [])
                    
                    # Convert to labels for this patch
                    if len(contours) > 0 and len(contours[0]) > 0:
                        patch_labels = self._convert_celldetection_to_labels(
                            contours[0], 
                            scores[0] if len(scores) > 0 else None, 
                            (patch_size, patch_size)
                        )
                        
                        # Resize back if needed
                        if patch_labels.shape != (y_end - y, x_end - x):
                            patch_labels = cv2.resize(patch_labels.astype(np.uint8), 
                                                    (x_end - x, y_end - y), 
                                                    interpolation=cv2.INTER_NEAREST).astype(np.int32)
                        
                        # Update labels and merge (avoid edge artifacts)
                        if np.max(patch_labels) > 0:
                            patch_labels[patch_labels > 0] += label_counter - 1
                            label_counter += np.max(patch_labels)
                            
                            # Merge center region to avoid edge artifacts
                            center_y_start = y + overlap//2 if y > 0 else y
                            center_y_end = y_end - overlap//2 if y_end < h else y_end
                            center_x_start = x + overlap//2 if x > 0 else x
                            center_x_end = x_end - overlap//2 if x_end < w else x_end
                            
                            patch_center_y_start = overlap//2 if y > 0 else 0
                            patch_center_y_end = patch_labels.shape[0] - overlap//2 if y_end < h else patch_labels.shape[0]
                            patch_center_x_start = overlap//2 if x > 0 else 0
                            patch_center_x_end = patch_labels.shape[1] - overlap//2 if x_end < w else patch_labels.shape[1]
                            
                            patch_region = patch_labels[patch_center_y_start:patch_center_y_end, 
                                                      patch_center_x_start:patch_center_x_end]
                            final_region = final_labels[center_y_start:center_y_end, 
                                                       center_x_start:center_x_end]
                            
                            # Add patch labels where final is zero
                            mask = (final_region == 0) & (patch_region > 0)
                            final_region[mask] = patch_region[mask]
            
            print(f"✅ Enhanced CellDetection: {np.max(final_labels)} cells detected")
            return final_labels
            
        except Exception as e:
            print(f"⚠️ Enhanced patch CellDetection failed: {e}")
            return self.celldetection_detection(color_img)
    
    def analyze_enhanced_color_content(self, color_img):
        """Enhanced color analysis with wavelength-specific measurements"""
        try:
            # Convert to different color spaces for comprehensive analysis
            hsv = cv2.cvtColor(color_img, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(color_img, cv2.COLOR_BGR2LAB)
            rgb = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
            
            # Enhanced green analysis in multiple color spaces
            b, g, r = cv2.split(color_img)
            h, s, v = cv2.split(hsv)
            l_lab, a_lab, b_lab = cv2.split(lab)
            
            # Green wavelength analysis (495-570 nm range)
            green_lower = np.array([35, 40, 40])
            green_upper = np.array([85, 255, 255])
            green_mask = cv2.inRange(hsv, green_lower, green_upper)
            
            # Chlorophyll-specific analysis (peak at 530nm)
            chlorophyll_lower = np.array([45, 50, 50])
            chlorophyll_upper = np.array([75, 255, 255])
            chlorophyll_mask = cv2.inRange(hsv, chlorophyll_lower, chlorophyll_upper)
            
            # Calculate comprehensive color metrics
            total_pixels = color_img.shape[0] * color_img.shape[1]
            green_pixels = np.sum(green_mask > 0)
            chlorophyll_pixels = np.sum(chlorophyll_mask > 0)
            
            # Color intensity analysis
            green_intensity_mean = np.mean(g[green_mask > 0]) if green_pixels > 0 else 0
            green_intensity_std = np.std(g[green_mask > 0]) if green_pixels > 0 else 0
            
            # Advanced color metrics
            green_saturation_mean = np.mean(s[green_mask > 0]) if green_pixels > 0 else 0
            green_value_mean = np.mean(v[green_mask > 0]) if green_pixels > 0 else 0
            
            # Color uniformity (coefficient of variation)
            color_uniformity = (green_intensity_std / green_intensity_mean) if green_intensity_mean > 0 else 0
            
            return {
                'green_percentage': (green_pixels / total_pixels) * 100,
                'chlorophyll_percentage': (chlorophyll_pixels / total_pixels) * 100,
                'green_intensity': {
                    'mean': float(green_intensity_mean),
                    'std': float(green_intensity_std),
                    'range': [float(np.min(g)), float(np.max(g))]
                },
                'green_saturation_mean': float(green_saturation_mean),
                'green_value_mean': float(green_value_mean),
                'color_uniformity': float(color_uniformity),
                'wavelength_analysis': {
                    'green_range_nm': self.green_wavelength_range,
                    'chlorophyll_peak_nm': self.chlorophyll_peak,
                    'estimated_chlorophyll_content': float(chlorophyll_pixels / total_pixels * 100)
                }
            }
            
        except Exception as e:
            print(f"⚠️ Enhanced color analysis failed: {e}")
            return {'green_percentage': 0, 'error': str(e)}
    
    def calculate_comprehensive_biomass(self, cell_data, image_shape):
        """Calculate comprehensive biomass using multiple approaches"""
        try:
            if not cell_data:
                return {'total_biomass_mg': 0, 'error': 'No cells detected'}
            
            # Basic calculations
            total_area_pixels = sum(cell['area'] for cell in cell_data)
            total_area_um2 = total_area_pixels * (self.pixel_to_micron ** 2)
            
            # Volume estimation assuming average cell thickness
            total_volume_um3 = total_area_um2 * self.cell_thickness_micron
            total_volume_cm3 = total_volume_um3 * 1e-12  # Convert to cm³
            
            # Biomass calculations
            fresh_weight_mg = total_volume_cm3 * self.wolffia_density * 1000  # Convert to mg
            dry_weight_mg = fresh_weight_mg * self.dry_weight_ratio
            
            # Statistical biomass analysis
            individual_areas = [cell['area'] * (self.pixel_to_micron ** 2) for cell in cell_data]
            individual_volumes = [area * self.cell_thickness_micron for area in individual_areas]
            individual_biomass = [vol * 1e-12 * self.wolffia_density * 1000 for vol in individual_volumes]
            
            # Biomass density (cells per unit area)
            image_area_um2 = image_shape[0] * image_shape[1] * (self.pixel_to_micron ** 2)
            cell_density = len(cell_data) / (image_area_um2 * 1e-6)  # cells/mm²
            
            return {
                'total_biomass_mg': float(fresh_weight_mg),
                'dry_biomass_mg': float(dry_weight_mg),
                'total_area_um2': float(total_area_um2),
                'total_volume_um3': float(total_volume_um3),
                'cell_count': len(cell_data),
                'cell_density_per_mm2': float(cell_density),
                'average_cell_biomass_mg': float(np.mean(individual_biomass)) if individual_biomass else 0,
                'biomass_std_mg': float(np.std(individual_biomass)) if individual_biomass else 0,
                'biomass_range_mg': [float(np.min(individual_biomass)), float(np.max(individual_biomass))] if individual_biomass else [0, 0],
                'calculation_parameters': {
                    'cell_density_g_cm3': self.wolffia_density,
                    'cell_thickness_um': self.cell_thickness_micron,
                    'dry_weight_ratio': self.dry_weight_ratio,
                    'pixel_to_micron': self.pixel_to_micron
                }
            }
            
        except Exception as e:
            print(f"⚠️ Biomass calculation failed: {e}")
            return {'total_biomass_mg': 0, 'error': str(e)}
    
    def enhanced_wavelength_analysis(self, color_img, labeled_img):
        """Enhanced wavelength-specific analysis for each detected cell"""
        try:
            if np.max(labeled_img) == 0:
                return {'error': 'No cells for wavelength analysis'}
            
            # Convert to different color spaces
            hsv = cv2.cvtColor(color_img, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(color_img, cv2.COLOR_BGR2LAB)
            
            regions = measure.regionprops(labeled_img, intensity_image=cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY))
            
            cell_wavelength_data = []
            
            for region in regions:
                mask = labeled_img == region.label
                
                if np.sum(mask) == 0:
                    continue
                
                # Extract color values for this cell
                h_values = hsv[:, :, 0][mask]
                s_values = hsv[:, :, 1][mask]
                v_values = hsv[:, :, 2][mask]
                
                # Green content analysis
                green_hue_mask = (h_values >= 35) & (h_values <= 85)
                green_percentage = np.sum(green_hue_mask) / len(h_values) * 100
                
                # Wavelength estimation from hue (approximation)
                # Hue 60 ≈ 530nm (peak chlorophyll), scaling accordingly
                estimated_wavelengths = 480 + (h_values / 179) * 200  # Rough mapping
                dominant_wavelength = np.mean(estimated_wavelengths[green_hue_mask]) if np.any(green_hue_mask) else 0
                
                cell_data = {
                    'cell_id': region.label,
                    'green_percentage': float(green_percentage),
                    'dominant_wavelength_nm': float(dominant_wavelength),
                    'mean_hue': float(np.mean(h_values)),
                    'mean_saturation': float(np.mean(s_values)),
                    'mean_value': float(np.mean(v_values)),
                    'color_intensity': float(region.mean_intensity),
                    'chlorophyll_indicator': float(green_percentage * np.mean(s_values) / 255)
                }
                cell_wavelength_data.append(cell_data)
            
            # Summary statistics
            if cell_wavelength_data:
                green_percentages = [cell['green_percentage'] for cell in cell_wavelength_data]
                wavelengths = [cell['dominant_wavelength_nm'] for cell in cell_wavelength_data if cell['dominant_wavelength_nm'] > 0]
                chlorophyll_indicators = [cell['chlorophyll_indicator'] for cell in cell_wavelength_data]
                
                summary = {
                    'total_cells_analyzed': len(cell_wavelength_data),
                    'mean_green_percentage': float(np.mean(green_percentages)),
                    'std_green_percentage': float(np.std(green_percentages)),
                    'mean_wavelength_nm': float(np.mean(wavelengths)) if wavelengths else 0,
                    'std_wavelength_nm': float(np.std(wavelengths)) if wavelengths else 0,
                    'mean_chlorophyll_indicator': float(np.mean(chlorophyll_indicators)),
                    'cell_data': cell_wavelength_data
                }
            else:
                summary = {'error': 'No valid cell data for analysis'}
            
            return summary
            
        except Exception as e:
            print(f"⚠️ Wavelength analysis failed: {e}")
            return {'error': str(e)}
    
    def calculate_morphometric_statistics(self, cell_data):
        """Calculate comprehensive morphometric statistics"""
        try:
            if not cell_data:
                return {'error': 'No cell data for morphometric analysis'}
            
            areas = [cell['area'] for cell in cell_data]
            perimeters = [cell['perimeter'] for cell in cell_data]
            
            # Calculate additional morphometric parameters
            circularities = []
            aspect_ratios = []
            
            for cell in cell_data:
                # Circularity
                if cell['perimeter'] > 0:
                    circularity = 4 * np.pi * cell['area'] / (cell['perimeter'] ** 2)
                    circularities.append(circularity)
                
                # Aspect ratio
                if 'minor_axis_length' in cell and cell['minor_axis_length'] > 0:
                    aspect_ratio = cell['major_axis_length'] / cell['minor_axis_length']
                    aspect_ratios.append(aspect_ratio)
            
            return {
                'area_statistics': {
                    'mean': float(np.mean(areas)),
                    'std': float(np.std(areas)),
                    'min': float(np.min(areas)),
                    'max': float(np.max(areas)),
                    'median': float(np.median(areas)),
                    'q25': float(np.percentile(areas, 25)),
                    'q75': float(np.percentile(areas, 75))
                },
                'perimeter_statistics': {
                    'mean': float(np.mean(perimeters)),
                    'std': float(np.std(perimeters)),
                    'min': float(np.min(perimeters)),
                    'max': float(np.max(perimeters))
                },
                'circularity_statistics': {
                    'mean': float(np.mean(circularities)) if circularities else 0,
                    'std': float(np.std(circularities)) if circularities else 0
                },
                'aspect_ratio_statistics': {
                    'mean': float(np.mean(aspect_ratios)) if aspect_ratios else 0,
                    'std': float(np.std(aspect_ratios)) if aspect_ratios else 0
                },
                'size_distribution': {
                    'small_cells': len([a for a in areas if a < np.percentile(areas, 33)]),
                    'medium_cells': len([a for a in areas if np.percentile(areas, 33) <= a <= np.percentile(areas, 67)]),
                    'large_cells': len([a for a in areas if a > np.percentile(areas, 67)])
                }
            }
            
        except Exception as e:
            print(f"⚠️ Morphometric analysis failed: {e}")
            return {'error': str(e)}
    
    def calculate_spatial_distribution(self, cell_data, image_shape):
        """Calculate spatial distribution patterns"""
        try:
            if len(cell_data) < 2:
                return {'error': 'Need at least 2 cells for spatial analysis'}
            
            centroids = np.array([cell['centroid'] for cell in cell_data])
            
            # Calculate distances between all pairs
            from scipy.spatial.distance import pdist, squareform
            distances = pdist(centroids)
            distance_matrix = squareform(distances)
            
            # Remove diagonal (distance to self)
            np.fill_diagonal(distance_matrix, np.inf)
            nearest_neighbor_distances = np.min(distance_matrix, axis=1) * self.pixel_to_micron
            
            # Spatial statistics
            mean_nn_distance = np.mean(nearest_neighbor_distances)
            std_nn_distance = np.std(nearest_neighbor_distances)
            
            # Clustering analysis
            image_area = image_shape[0] * image_shape[1] * (self.pixel_to_micron ** 2)
            expected_distance = 0.5 * np.sqrt(image_area / len(cell_data))  # Expected for random distribution
            clustering_index = expected_distance / mean_nn_distance  # >1 = clustered, <1 = dispersed
            
            # Spatial distribution in image quadrants
            h, w = image_shape
            quadrant_counts = {
                'top_left': len([c for c in centroids if c[1] < w/2 and c[0] < h/2]),
                'top_right': len([c for c in centroids if c[1] >= w/2 and c[0] < h/2]),
                'bottom_left': len([c for c in centroids if c[1] < w/2 and c[0] >= h/2]),
                'bottom_right': len([c for c in centroids if c[1] >= w/2 and c[0] >= h/2])
            }
            
            return {
                'nearest_neighbor_distance_um': {
                    'mean': float(mean_nn_distance),
                    'std': float(std_nn_distance),
                    'min': float(np.min(nearest_neighbor_distances)),
                    'max': float(np.max(nearest_neighbor_distances))
                },
                'clustering_index': float(clustering_index),
                'clustering_interpretation': 'clustered' if clustering_index > 1.2 else 'dispersed' if clustering_index < 0.8 else 'random',
                'quadrant_distribution': quadrant_counts,
                'cell_density_per_area': len(cell_data) / (image_area * 1e-6)  # cells per mm²
            }
            
        except Exception as e:
            print(f"⚠️ Spatial analysis failed: {e}")
            return {'error': str(e)}
    
    def assess_culture_health(self, cell_data, color_analysis):
        """Assess overall culture health based on multiple parameters"""
        try:
            if not cell_data:
                return {'overall_health': 'unknown', 'health_score': 0}
            
            health_indicators = []
            
            # 1. Cell count indicator (more cells = healthier, up to a point)
            cell_count = len(cell_data)
            if cell_count > 50:
                count_score = 1.0
            elif cell_count > 20:
                count_score = 0.8
            elif cell_count > 10:
                count_score = 0.6
            else:
                count_score = 0.4
            health_indicators.append(('cell_count', count_score))
            
            # 2. Green content indicator (higher green = healthier)
            green_percentage = color_analysis.get('green_percentage', 0)
            if green_percentage > 20:
                green_score = 1.0
            elif green_percentage > 10:
                green_score = 0.8
            elif green_percentage > 5:
                green_score = 0.6
            else:
                green_score = 0.3
            health_indicators.append(('green_content', green_score))
            
            # 3. Size uniformity indicator (consistent sizes = healthier)
            areas = [cell['area'] for cell in cell_data]
            cv_area = np.std(areas) / np.mean(areas) if np.mean(areas) > 0 else 1
            if cv_area < 0.3:
                uniformity_score = 1.0
            elif cv_area < 0.5:
                uniformity_score = 0.8
            elif cv_area < 0.7:
                uniformity_score = 0.6
            else:
                uniformity_score = 0.4
            health_indicators.append(('size_uniformity', uniformity_score))
            
            # 4. Color intensity indicator
            color_intensity = color_analysis.get('green_intensity', {}).get('mean', 0)
            if color_intensity > 120:
                intensity_score = 1.0
            elif color_intensity > 80:
                intensity_score = 0.8
            elif color_intensity > 50:
                intensity_score = 0.6
            else:
                intensity_score = 0.4
            health_indicators.append(('color_intensity', intensity_score))
            
            # Calculate overall health score
            overall_score = np.mean([score for _, score in health_indicators])
            
            # Determine health category
            if overall_score > 0.8:
                health_category = 'excellent'
            elif overall_score > 0.7:
                health_category = 'good'
            elif overall_score > 0.5:
                health_category = 'fair'
            else:
                health_category = 'poor'
            
            return {
                'overall_health': health_category,
                'health_score': float(overall_score),
                'health_indicators': dict(health_indicators),
                'recommendations': self.generate_health_recommendations(health_indicators, color_analysis)
            }
            
        except Exception as e:
            print(f"⚠️ Health assessment failed: {e}")
            return {'overall_health': 'unknown', 'health_score': 0, 'error': str(e)}
    
    def generate_health_recommendations(self, health_indicators, color_analysis):
        """Generate specific recommendations based on health assessment"""
        recommendations = []
        
        indicator_dict = dict(health_indicators)
        
        if indicator_dict.get('cell_count', 0) < 0.6:
            recommendations.append("Low cell count detected. Consider optimizing growth conditions or nutrient availability.")
        
        if indicator_dict.get('green_content', 0) < 0.6:
            recommendations.append("Low chlorophyll content. Check lighting conditions and ensure adequate light intensity.")
        
        if indicator_dict.get('size_uniformity', 0) < 0.6:
            recommendations.append("High size variation detected. This may indicate stress or suboptimal growth conditions.")
        
        if indicator_dict.get('color_intensity', 0) < 0.6:
            recommendations.append("Low color intensity suggests possible nutrient deficiency or light stress.")
        
        if color_analysis.get('color_uniformity', 0) > 0.5:
            recommendations.append("High color variation detected. Consider more uniform lighting or culture conditions.")
        
        if not recommendations:
            recommendations.append("Culture appears healthy. Continue current maintenance practices.")
        
        return recommendations
    
    def update_time_series(self, series_id, timestamp, cell_data, biomass_analysis, color_analysis):
        """Update time-series tracking for multiple images"""
        try:
            if series_id not in self.time_series_data:
                self.time_series_data[series_id] = {
                    'created': datetime.now().isoformat(),
                    'data_points': []
                }
            
            # Create time point data
            time_point = {
                'timestamp': timestamp,
                'cell_count': len(cell_data),
                'total_biomass_mg': biomass_analysis.get('total_biomass_mg', 0),
                'green_percentage': color_analysis.get('green_percentage', 0),
                'mean_cell_area': biomass_analysis.get('total_area_um2', 0) / len(cell_data) if cell_data else 0,
                'chlorophyll_percentage': color_analysis.get('chlorophyll_percentage', 0)
            }
            
            self.time_series_data[series_id]['data_points'].append(time_point)
            
            # Calculate trends if we have multiple points
            data_points = self.time_series_data[series_id]['data_points']
            if len(data_points) >= 2:
                # Sort by timestamp
                sorted_points = sorted(data_points, key=lambda x: x['timestamp'])
                
                # Calculate trends
                cell_counts = [p['cell_count'] for p in sorted_points]
                biomass_values = [p['total_biomass_mg'] for p in sorted_points]
                green_percentages = [p['green_percentage'] for p in sorted_points]
                
                trends = {
                    'cell_count_trend': self.calculate_trend(cell_counts),
                    'biomass_trend': self.calculate_trend(biomass_values),
                    'green_content_trend': self.calculate_trend(green_percentages),
                    'growth_rate': self.calculate_growth_rate(sorted_points)
                }
                
                # Save time series data
                series_file = self.dirs['time_series'] / f"series_{series_id}.json"
                with open(series_file, 'w') as f:
                    json.dump(self.time_series_data[series_id], f, indent=2)
                
                return {
                    'series_id': series_id,
                    'time_points': len(data_points),
                    'current_point': time_point,
                    'trends': trends,
                    'visualization_data': self.create_time_series_visualization(sorted_points, series_id)
                }
            
            return {
                'series_id': series_id,
                'time_points': len(data_points),
                'current_point': time_point,
                'message': 'Need at least 2 time points for trend analysis'
            }
            
        except Exception as e:
            print(f"⚠️ Time series update failed: {e}")
            return {'error': str(e)}
    
    def calculate_trend(self, values):
        """Calculate trend direction and magnitude"""
        if len(values) < 2:
            return 'insufficient_data'
        
        # Simple linear trend
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if abs(slope) < 0.01:
            return 'stable'
        elif slope > 0:
            return 'increasing'
        else:
            return 'decreasing'
    
    def calculate_growth_rate(self, sorted_points):
        """Calculate growth rate based on cell count changes"""
        if len(sorted_points) < 2:
            return 0
        
        initial_count = sorted_points[0]['cell_count']
        final_count = sorted_points[-1]['cell_count']
        
        if initial_count == 0:
            return 0
        
        growth_rate = (final_count - initial_count) / initial_count * 100
        return float(growth_rate)
    
    def create_time_series_visualization(self, sorted_points, series_id):
        """Create time series visualization"""
        try:
            timestamps = [p['timestamp'] for p in sorted_points]
            cell_counts = [p['cell_count'] for p in sorted_points]
            biomass_values = [p['total_biomass_mg'] for p in sorted_points]
            green_percentages = [p['green_percentage'] for p in sorted_points]
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Cell count over time
            axes[0, 0].plot(range(len(timestamps)), cell_counts, 'bo-', linewidth=2, markersize=8)
            axes[0, 0].set_title('Cell Count Over Time', fontsize=14, fontweight='bold')
            axes[0, 0].set_xlabel('Time Point')
            axes[0, 0].set_ylabel('Cell Count')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Biomass over time
            axes[0, 1].plot(range(len(timestamps)), biomass_values, 'go-', linewidth=2, markersize=8)
            axes[0, 1].set_title('Biomass Over Time', fontsize=14, fontweight='bold')
            axes[0, 1].set_xlabel('Time Point')
            axes[0, 1].set_ylabel('Biomass (mg)')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Green content over time
            axes[1, 0].plot(range(len(timestamps)), green_percentages, 'ro-', linewidth=2, markersize=8)
            axes[1, 0].set_title('Green Content Over Time', fontsize=14, fontweight='bold')
            axes[1, 0].set_xlabel('Time Point')
            axes[1, 0].set_ylabel('Green Content (%)')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Combined normalized trends
            if len(cell_counts) > 1:
                norm_cells = np.array(cell_counts) / np.max(cell_counts)
                norm_biomass = np.array(biomass_values) / np.max(biomass_values) if np.max(biomass_values) > 0 else np.zeros_like(biomass_values)
                norm_green = np.array(green_percentages) / 100
                
                axes[1, 1].plot(range(len(timestamps)), norm_cells, 'b-', label='Cell Count', linewidth=2)
                axes[1, 1].plot(range(len(timestamps)), norm_biomass, 'g-', label='Biomass', linewidth=2)
                axes[1, 1].plot(range(len(timestamps)), norm_green, 'r-', label='Green Content', linewidth=2)
                axes[1, 1].set_title('Normalized Trends', fontsize=14, fontweight='bold')
                axes[1, 1].set_xlabel('Time Point')
                axes[1, 1].set_ylabel('Normalized Value')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save visualization
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            viz_path = self.dirs['time_series'] / f"time_series_{series_id}_{timestamp}.png"
            plt.savefig(viz_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Convert to base64 for web display
            with open(viz_path, 'rb') as f:
                viz_b64 = base64.b64encode(f.read()).decode('utf-8')
            
            return {
                'visualization_path': str(viz_path),
                'visualization_b64': viz_b64
            }
            
        except Exception as e:
            print(f"⚠️ Time series visualization failed: {e}")
            return {'error': str(e)}
    
    def create_comprehensive_visualization(self, original_img, labeled_img, cell_data, color_analysis, biomass_analysis):
        """Create comprehensive visualization with histograms and analysis charts"""
        try:
            # Create figure with multiple subplots
            fig = plt.figure(figsize=(20, 16))
            
            # Main detection visualization
            ax1 = plt.subplot(3, 4, (1, 6))  # Spans 2x3 grid
            if labeled_img is not None and np.max(labeled_img) > 0:
                overlay = color.label2rgb(labeled_img, image=original_img, bg_label=0, alpha=0.4)
                ax1.imshow(overlay)
                
                # Add cell numbers
                for cell in cell_data:
                    center = cell['centroid']
                    ax1.plot(center[0], center[1], 'yo', markersize=8)
                    ax1.text(center[0], center[1], str(cell['id']), 
                            color='yellow', fontsize=8, ha='center', va='center', weight='bold')
            else:
                ax1.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
            
            ax1.set_title(f'Cell Detection Results\n{len(cell_data)} Cells Detected', 
                         fontsize=16, fontweight='bold')
            ax1.axis('off')
            
            # Cell size histogram
            if cell_data:
                ax2 = plt.subplot(3, 4, 3)
                areas = [cell['area'] for cell in cell_data]
                ax2.hist(areas, bins=min(15, len(areas)), alpha=0.7, color='skyblue', edgecolor='black')
                ax2.set_title('Cell Size Distribution', fontweight='bold')
                ax2.set_xlabel('Area (μm²)')
                ax2.set_ylabel('Count')
                ax2.grid(True, alpha=0.3)
            
            # Green content analysis
            ax3 = plt.subplot(3, 4, 4)
            green_data = [color_analysis.get('green_percentage', 0), 
                         100 - color_analysis.get('green_percentage', 0)]
            colors = ['green', 'lightgray']
            labels = ['Green Content', 'Other']
            wedges, texts, autotexts = ax3.pie(green_data, labels=labels, colors=colors, 
                                              autopct='%1.1f%%', startangle=90)
            ax3.set_title('Color Composition', fontweight='bold')
            
            # Biomass analysis bar chart
            ax4 = plt.subplot(3, 4, 7)
            biomass_categories = ['Fresh Weight', 'Dry Weight']
            biomass_values = [biomass_analysis.get('total_biomass_mg', 0), 
                            biomass_analysis.get('dry_biomass_mg', 0)]
            bars = ax4.bar(biomass_categories, biomass_values, color=['lightgreen', 'brown'], alpha=0.7)
            ax4.set_title('Biomass Analysis', fontweight='bold')
            ax4.set_ylabel('Weight (mg)')
            ax4.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, biomass_values):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
            
            # Cell density visualization
            ax5 = plt.subplot(3, 4, 8)
            density = biomass_analysis.get('cell_density_per_mm2', 0)
            ax5.bar(['Cell Density'], [density], color='orange', alpha=0.7)
            ax5.set_title('Cell Density', fontweight='bold')
            ax5.set_ylabel('Cells/mm²')
            ax5.grid(True, alpha=0.3)
            
            # Statistics summary
            ax6 = plt.subplot(3, 4, (9, 10))
            ax6.axis('off')
            
            stats_text = f"""
QUANTITATIVE ANALYSIS SUMMARY

Cell Count: {len(cell_data)}
Total Biomass: {biomass_analysis.get('total_biomass_mg', 0):.3f} mg
Dry Biomass: {biomass_analysis.get('dry_biomass_mg', 0):.3f} mg
Green Content: {color_analysis.get('green_percentage', 0):.1f}%
Chlorophyll: {color_analysis.get('chlorophyll_percentage', 0):.1f}%

Average Cell Area: {np.mean([cell['area'] for cell in cell_data]) if cell_data else 0:.1f} μm²
Cell Density: {density:.1f} cells/mm²
Total Coverage: {biomass_analysis.get('total_area_um2', 0):.0f} μm²

Color Intensity: {color_analysis.get('green_intensity', {}).get('mean', 0):.1f}
Color Uniformity: {color_analysis.get('color_uniformity', 0):.2f}
Estimated Wavelength: {color_analysis.get('wavelength_analysis', {}).get('chlorophyll_peak_nm', 530)} nm
            """
            
            ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes, fontsize=11,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            
            # Morphometric analysis
            if cell_data and len(cell_data) > 1:
                ax7 = plt.subplot(3, 4, 11)
                # Scatter plot of major vs minor axis
                major_axes = [cell.get('major_axis_length', 0) for cell in cell_data]
                minor_axes = [cell.get('minor_axis_length', 0) for cell in cell_data]
                
                if any(major_axes) and any(minor_axes):
                    ax7.scatter(minor_axes, major_axes, alpha=0.6, s=50, c='purple')
                    ax7.set_xlabel('Minor Axis (μm)')
                    ax7.set_ylabel('Major Axis (μm)')
                    ax7.set_title('Cell Shape Analysis', fontweight='bold')
                    ax7.grid(True, alpha=0.3)
                else:
                    ax7.text(0.5, 0.5, 'Shape data\nnot available', 
                            ha='center', va='center', transform=ax7.transAxes)
                    ax7.set_title('Cell Shape Analysis', fontweight='bold')
            
            # Method comparison (if multiple methods used)
            ax8 = plt.subplot(3, 4, 12)
            method_info = "Detection Methods Used:\n"
            if hasattr(self, '_last_methods_used'):
                for method in self._last_methods_used:
                    method_info += f"• {method.capitalize()}\n"
            else:
                method_info += "• Multi-method analysis"
            
            ax8.text(0.05, 0.95, method_info, transform=ax8.transAxes, fontsize=12,
                    verticalalignment='top', fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
            ax8.set_title('Analysis Methods', fontweight='bold')
            ax8.axis('off')
            
            plt.tight_layout()
            
            # Save comprehensive visualization
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            main_viz_path = self.dirs['results'] / f"comprehensive_analysis_{timestamp}.png"
            plt.savefig(main_viz_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Create additional histograms
            histogram_paths = self.create_detailed_histograms(cell_data, color_analysis, biomass_analysis, timestamp)
            
            # Convert main visualization to base64
            with open(main_viz_path, 'rb') as f:
                main_viz_b64 = base64.b64encode(f.read()).decode('utf-8')
            
            return {
                'main_visualization': main_viz_path,
                'main_visualization_b64': main_viz_b64,
                'histogram_paths': histogram_paths,
                'detection_overview': main_viz_b64  # For compatibility
            }
            
        except Exception as e:
            print(f"⚠️ Comprehensive visualization failed: {e}")
            return {'error': str(e)}
    
    def create_detailed_histograms(self, cell_data, color_analysis, biomass_analysis, timestamp):
        """Create detailed histogram visualizations"""
        try:
            histogram_paths = {}
            
            if not cell_data:
                return histogram_paths
            
            # Area distribution histogram
            fig, ax = plt.subplots(figsize=(10, 6))
            areas = [cell['area'] for cell in cell_data]
            n, bins, patches = ax.hist(areas, bins=min(20, len(areas)), alpha=0.7, 
                                     color='skyblue', edgecolor='black', linewidth=1.2)
            
            # Color code bins by size
            for i, patch in enumerate(patches):
                if bins[i] < np.percentile(areas, 33):
                    patch.set_facecolor('lightgreen')  # Small cells
                elif bins[i] < np.percentile(areas, 67):
                    patch.set_facecolor('yellow')  # Medium cells
                else:
                    patch.set_facecolor('orange')  # Large cells
            
            ax.axvline(np.mean(areas), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(areas):.1f} μm²')
            ax.axvline(np.median(areas), color='blue', linestyle='--', linewidth=2, label=f'Median: {np.median(areas):.1f} μm²')
            
            ax.set_title('Cell Area Distribution', fontsize=16, fontweight='bold')
            ax.set_xlabel('Cell Area (μm²)', fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            area_hist_path = self.dirs['results'] / f"area_histogram_{timestamp}.png"
            plt.savefig(area_hist_path, dpi=300, bbox_inches='tight')
            plt.close()
            histogram_paths['area_distribution'] = str(area_hist_path)
            
            # Convert to base64
            with open(area_hist_path, 'rb') as f:
                histogram_paths['area_distribution_b64'] = base64.b64encode(f.read()).decode('utf-8')
            
            # Intensity distribution histogram
            if any('mean_intensity' in cell for cell in cell_data):
                fig, ax = plt.subplots(figsize=(10, 6))
                intensities = [cell.get('mean_intensity', 0) for cell in cell_data]
                ax.hist(intensities, bins=min(15, len(intensities)), alpha=0.7, 
                       color='lightcoral', edgecolor='black', linewidth=1.2)
                
                ax.axvline(np.mean(intensities), color='darkred', linestyle='--', 
                          linewidth=2, label=f'Mean: {np.mean(intensities):.1f}')
                
                ax.set_title('Cell Intensity Distribution', fontsize=16, fontweight='bold')
                ax.set_xlabel('Mean Intensity', fontsize=12)
                ax.set_ylabel('Frequency', fontsize=12)
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                intensity_hist_path = self.dirs['results'] / f"intensity_histogram_{timestamp}.png"
                plt.savefig(intensity_hist_path, dpi=300, bbox_inches='tight')
                plt.close()
                histogram_paths['intensity_distribution'] = str(intensity_hist_path)
                
                with open(intensity_hist_path, 'rb') as f:
                    histogram_paths['intensity_distribution_b64'] = base64.b64encode(f.read()).decode('utf-8')
            
            return histogram_paths
            
        except Exception as e:
            print(f"⚠️ Detailed histogram creation failed: {e}")
            return {}
        
        
    def analyze_green_content(self, color_img):
        """
        Analyze green content in the color image before grayscale conversion
        Returns percentage of green pixels relative to total image area
        """
        try:
            # Convert BGR to HSV for better color analysis
            hsv = cv2.cvtColor(color_img, cv2.COLOR_BGR2HSV)
            
            # Define green color range in HSV
            # Lower and upper bounds for green hue
            lower_green = np.array([35, 40, 40])    # Lower bound for green
            upper_green = np.array([85, 255, 255])  # Upper bound for green
            
            # Create mask for green pixels
            green_mask = cv2.inRange(hsv, lower_green, upper_green)
            
            # Calculate percentage of green pixels
            total_pixels = color_img.shape[0] * color_img.shape[1]
            green_pixels = np.sum(green_mask > 0)
            green_percentage = (green_pixels / total_pixels) * 100
            
            print(f"🟢 Green content analysis: {green_percentage:.1f}% of image")
            
            return round(green_percentage, 1)
            
        except Exception as e:
            print(f"⚠️ Green analysis failed: {e}")
            return 0.0

    def create_green_enhanced_grayscale(self, color_img, for_cnn=False):
        """
        Create enhanced grayscale image that emphasizes green regions.
        Optionally return 3-channel version for CNN compatibility.

        Args:
            color_img: Input BGR image.
            for_cnn: If True, returns a 3-channel grayscale image [H, W, 3] for CNN.

        Returns:
            enhanced_gray: 1-channel or 3-channel enhanced image depending on `for_cnn`.
        """
        try:
            hsv = cv2.cvtColor(color_img, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(color_img, cv2.COLOR_BGR2LAB)

            b, g, r = cv2.split(color_img)
            _, _, v = cv2.split(hsv)
            _, a, _ = cv2.split(lab)

            lower_green = np.array([35, 40, 40])
            upper_green = np.array([85, 255, 255])
            green_mask = cv2.inRange(hsv, lower_green, upper_green)

            green_enhanced = g.astype(np.float32)
            green_lab = 255 - a.astype(np.float32)
            green_score = np.zeros_like(green_enhanced)
            green_score[green_mask > 0] = 255

            enhanced_gray = (
                0.4 * green_enhanced +
                0.3 * green_lab +
                0.3 * green_score
            )

            enhanced_gray = np.clip(enhanced_gray, 0, 255).astype(np.uint8)

            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced_gray = clahe.apply(enhanced_gray)

            print(f"🟢 Created green-enhanced grayscale (green pixels: {np.sum(green_mask > 0)})")

            if for_cnn:
                # Return 3-channel grayscale for CNN use
                return cv2.merge([enhanced_gray] * 3)

            return enhanced_gray

        except Exception as e:
            print(f"⚠️ Green enhancement failed, using fallback grayscale: {e}")
            gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
            return cv2.merge([gray] * 3) if for_cnn else gray


        
        
    def apply_green_filtering(self, labeled_img, color_img):
        """Apply green content filtering to detected regions"""
        try:
            hsv = cv2.cvtColor(color_img, cv2.COLOR_BGR2HSV)
            green_lower = np.array([35, 40, 40])
            green_upper = np.array([85, 255, 255])
            green_mask = cv2.inRange(hsv, green_lower, green_upper)
            
            filtered_img = np.zeros_like(labeled_img)
            regions = measure.regionprops(labeled_img)
            new_label = 1
            
            for region in regions:
                mask = labeled_img == region.label
                green_pixels = np.sum(green_mask[mask] > 0)
                total_pixels = np.sum(mask)
                
                if total_pixels > 0:
                    green_ratio = green_pixels / total_pixels
                    if green_ratio > 0.1 and self.min_cell_area <= region.area <= self.max_cell_area:
                        filtered_img[mask] = new_label
                        new_label += 1
            
            return filtered_img
            
        except Exception as e:
            print(f"⚠️ Green filtering failed: {e}")
            return labeled_img
    
    def extract_enhanced_cell_properties(self, labeled_img, gray_img, color_img):
        """Enhanced cell property extraction with color and morphometric analysis"""
        if labeled_img.dtype != np.int32:
            labeled_img = labeled_img.astype(np.int32)
        
        regions = measure.regionprops(labeled_img, intensity_image=gray_img)
        
        # Get color information
        hsv = cv2.cvtColor(color_img, cv2.COLOR_BGR2HSV)
        b, g, r = cv2.split(color_img)
        
        cells = []
        for i, region in enumerate(regions, 1):
            if region.area >= self.min_cell_area:
                # Basic properties
                cell_data = {
                    'id': i,
                    'area': region.area * (self.pixel_to_micron ** 2),
                    'perimeter': region.perimeter * self.pixel_to_micron,
                    'centroid': [int(region.centroid[1]), int(region.centroid[0])],
                    'major_axis_length': region.major_axis_length * self.pixel_to_micron,
                    'minor_axis_length': region.minor_axis_length * self.pixel_to_micron,
                    'eccentricity': region.eccentricity,
                    'mean_intensity': region.mean_intensity,
                    'max_intensity': region.max_intensity,
                    'min_intensity': region.min_intensity
                }
                
                # Enhanced color properties
                mask = labeled_img == region.label
                if np.any(mask):
                    # Green content for this cell
                    cell_green = np.mean(g[mask])
                    cell_hsv_h = np.mean(hsv[:, :, 0][mask])
                    cell_hsv_s = np.mean(hsv[:, :, 1][mask])
                    cell_hsv_v = np.mean(hsv[:, :, 2][mask])
                    
                    cell_data.update({
                        'green_intensity': float(cell_green),
                        'hue_mean': float(cell_hsv_h),
                        'saturation_mean': float(cell_hsv_s),
                        'value_mean': float(cell_hsv_v),
                        'color_ratio_green': float(cell_green / (np.mean(r[mask]) + np.mean(b[mask]) + 1e-6))
                    })
                
                # Morphometric properties
                if region.perimeter > 0:
                    cell_data['circularity'] = 4 * np.pi * region.area / (region.perimeter ** 2)
                else:
                    cell_data['circularity'] = 0
                
                cell_data['solidity'] = region.solidity
                cell_data['extent'] = region.extent
                
                # Biomass estimation for individual cell
                cell_volume_um3 = cell_data['area'] * self.cell_thickness_micron
                cell_biomass_mg = cell_volume_um3 * 1e-12 * self.wolffia_density * 1000
                cell_data['estimated_biomass_mg'] = float(cell_biomass_mg)
                
                cells.append(cell_data)
        
        return cells

        
        
        
        
    def color_aware_tophat_detection(self, color_img, enhanced_gray):
        """
        Enhanced tophat detection that uses both color and enhanced grayscale
        """
        try:
            # Extract green information from color image
            hsv = cv2.cvtColor(color_img, cv2.COLOR_BGR2HSV)
            lower_green = np.array([35, 40, 40])
            upper_green = np.array([85, 255, 255])
            green_mask = cv2.inRange(hsv, lower_green, upper_green)
            
            # Run tophat detection on enhanced grayscale
            tophat_result = self.tophat_ml_detection(enhanced_gray)
            
            # Filter results to prioritize green regions
            if np.max(tophat_result) > 0:
                # Apply green filtering to tophat results
                filtered_result = tophat_result.copy()
                
                # Check each detected region for green content
                from skimage import measure
                regions = measure.regionprops(tophat_result)
                
                for region in regions:
                    # Get region mask
                    region_mask = tophat_result == region.label
                    
                    # Calculate green content in this region
                    green_pixels_in_region = np.sum(green_mask[region_mask] > 0)
                    total_pixels_in_region = np.sum(region_mask)
                    
                    if total_pixels_in_region > 0:
                        green_percentage = green_pixels_in_region / total_pixels_in_region
                        
                        # If region has low green content, reduce its confidence
                        if green_percentage < 0.1:  # Less than 10% green
                            filtered_result[region_mask] = 0
                
                print(f"🟢 Color-aware tophat: filtered based on green content")
                return filtered_result
            
            return tophat_result
            
        except Exception as e:
            print(f"⚠️ Color-aware tophat failed, using standard: {e}")
            return self.tophat_ml_detection(enhanced_gray)


    

        
    def load_tophat_model(self):
        """Load tophat model if available"""
        if self._tophat_model is not None:
            return True
            
        model_path = self.dirs['models'] / 'tophat_model.pkl'
        if model_path.exists():
            try:
                with open(model_path, 'rb') as f:
                    self._tophat_model = pickle.load(f)
                    self.tophat_model = self._tophat_model  # Set public property
                print("✅ Tophat model loaded successfully")
                return True
            except Exception as e:
                print(f"⚠️ Failed to load tophat model: {e}")
        return False
    
    def tophat_ml_detection(self, color_img):
        """
         Tophat ML detection - Everything in one place
        Green-focused, GPU-accelerated, self-contained cell detection
        """
        try:
            if self._tophat_model is None:
                print("⚠️ Tophat model not loaded")
                return np.zeros(color_img.shape[:2], dtype=np.int32)

            # Quick channel validation and conversion
            if len(color_img.shape) == 2:
                color_img = cv2.cvtColor(color_img, cv2.COLOR_GRAY2BGR)
            elif len(color_img.shape) == 3:
                if color_img.shape[2] == 1:
                    color_img = cv2.cvtColor(color_img.squeeze(), cv2.COLOR_GRAY2BGR)
                elif color_img.shape[2] == 4:
                    color_img = color_img[:, :, :3]
                elif color_img.shape[2] != 3:
                    color_img = color_img[:, :, :3]

            print("⚡ Running  green-focused tophat detection...")

            # Extract fast features
            features = self.extract_ml_features(color_img)

            # Quick prediction
            predictions = self._tophat_model.predict(features.reshape(-1, features.shape[-1]))
            binary_mask = predictions.reshape(color_img.shape[:2]).astype(np.uint8)

            # Fast post-processing with green validation
            if np.any(binary_mask > 0):
                # GPU-accelerated morphology
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                cleaned = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
                cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

                # Connected components
                num_labels, labeled_img = cv2.connectedComponents(cleaned)
                
                if num_labels > 1:
                    # INLINE GREEN FILTERING - autonomous cell selection
                    filtered_img = np.zeros_like(labeled_img, dtype=np.int32)
                    new_label = 1
                    
                    # Extract green channel for validation
                    b, g, r = cv2.split(color_img)
                    # Determine if color_img is BGR or RGB (assuming RGB now)
                    hsv = cv2.cvtColor(color_img, cv2.COLOR_RGB2HSV)
                    green_mask = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
                    
                    # Process each region autonomously
                    unique_labels, counts = np.unique(labeled_img, return_counts=True)
                    for label, area in zip(unique_labels, counts):
                        if label == 0 or not (self.min_cell_area <= area <= self.max_cell_area):
                            continue
                        
                        # Quick green validation
                        region_mask = labeled_img == label
                        mean_green = np.mean(g[region_mask])
                        mean_other = np.mean(np.maximum(r[region_mask], b[region_mask]))
                        green_pixels = np.sum(green_mask[region_mask] > 0)
                        green_ratio = green_pixels / area if area > 0 else 0
                        
                        # Autonomous decision: is this a green cell?
                        if mean_green > mean_other and green_ratio > 0.1 and mean_green > 70:
                            filtered_img[region_mask] = new_label
                            new_label += 1
                    
                    cell_count = new_label - 1
                    print(f"⚡  Green Tophat: {cell_count} green cells detected")
                    return filtered_img

            print("⚡  Green Tophat: No green cells found")
            return np.zeros(color_img.shape[:2], dtype=np.int32)

        except Exception as e:
            print(f"❌ Fast Tophat failed: {e}")
            return np.zeros(color_img.shape[:2], dtype=np.int32)

    def extract_ml_features(self, img):
            """
            10-feature extraction for speed and accuracy
            Focus on most discriminative features for Wolffia cells
            """
            try:
                # Fast preprocessing
                if len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                elif len(img.shape) == 3 and img.shape[2] != 3:
                    img = img[:, :, :3]
                
                h, w = img.shape[:2]
                img_size = h * w
                
                # Pre-allocate for 10 optimized features
                features = np.empty((img_size, 10), dtype=np.float32)
                
                # OPTIMIZED: Only most discriminative features for Wolffia
                b, g, r = cv2.split(img)
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
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
                lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                a_lab = cv2.split(lab)[1]
                enhanced_gray = (0.4 * g_flat + 0.3 * (255 - a_lab.flatten()) + 0.3 * green_mask)
                features[:, idx] = enhanced_gray
                idx += 1
                
                # Feature 5: Distance transform (cell center detection)
                green_binary = (green_mask.reshape(h, w) > 0).astype(np.uint8)
                if np.sum(green_binary) > 0:
                    dist_transform = cv2.distanceTransform(green_binary, cv2.DIST_L2, 5)
                    features[:, idx] = dist_transform.flatten()
                else:
                    features[:, idx] = np.zeros(img.shape[0] * img.shape[1])
                idx += 1
                
                # Feature 6: Tophat operation (blob detection)
                enhanced_gray_2d = enhanced_gray.reshape(h, w).astype(np.uint8)
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
                
                print(f"🚀 Extracted 10 optimized features for fast Wolffia detection")
                return features
                
            except Exception as e:
                print(f"❌ Enhanced feature extraction failed: {e}")
                import traceback
                traceback.print_exc()
                return np.zeros((img.shape[0] * img.shape[1], 10), dtype=np.float32)

    def extract_basic_fallback_features(self, img):
        """Fallback feature extraction if comprehensive fails"""
        try:
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            
            b, g, r = cv2.split(img)
            
            # Create 46 basic features to match expected input size
            features = []
            
            # Add channels multiple times to reach 46 features
            for _ in range(5):
                features.extend([b.flatten(), g.flatten(), r.flatten()])
            
            # Add more basic features
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            features.extend([gray.flatten()])
            
            # Pad to exactly 46 features if needed
            while len(features) < 46:
                features.append(gray.flatten())
            
            return np.column_stack(features[:46])  # Ensure exactly 46 features
            
        except Exception as e:
            print(f"❌ Even fallback features failed: {e}")
            # Return zeros with correct shape
            return np.zeros((img.shape[0] * img.shape[1], 46))
            
        
        
    def color_aware_watershed_segmentation(self, color_img):
        """
        Enhanced watershed segmentation that uses color information
        Specifically optimized for green Wolffia cells
        """
        try:
            # Extract color channels for analysis
            b, g, r = cv2.split(color_img)
            hsv = cv2.cvtColor(color_img, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            
            # Create green-enhanced mask for better segmentation
            lower_green = np.array([35, 40, 40])
            upper_green = np.array([85, 255, 255])
            green_mask = cv2.inRange(hsv, lower_green, upper_green)
            
            # Use the green channel as primary source for Wolffia cells
            # But enhance it with color information
            enhanced_channel = g.copy().astype(np.float32)
            
            # Boost green regions in the enhanced channel
            enhanced_channel[green_mask > 0] = enhanced_channel[green_mask > 0] * 1.3
            enhanced_channel = np.clip(enhanced_channel, 0, 255).astype(np.uint8)
            
            # Apply CLAHE for local contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced_channel = clahe.apply(enhanced_channel)
            
            # Now use the existing watershed segmentation on the enhanced channel
            return self.watershed_segmentation(enhanced_channel, color_img)
            
        except Exception as e:
            print(f"⚠️ Color-aware watershed failed, using fallback: {e}")
            # Fallback to grayscale watershed
            gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
            return self.watershed_segmentation(gray, color_img)
    
    def load_cnn_model(self):
        """
        SIMPLIFIED: Load only wolffia_cnn_best.pth with smart configuration detection
        Enhanced to automatically detect model configuration from checkpoint
        """
        if self._cnn_model is not None:
            self.wolffia_cnn_available = True
            return True
        
        if not TORCH_AVAILABLE:
            print("⚠️ PyTorch not available - CNN models disabled")
            self.wolffia_cnn_available = False
            return False
        
        print("🤖 Loading Wolffia CNN model...")
        
        # Only look for the main model file
        model_path = self.dirs['models'] / 'wolffia_cnn_best.pth'
        
        if not model_path.exists():
            print("❌ wolffia_cnn_best.pth not found")
            print("💡 Train a model with: python train_wolffia_cnn.py")
            self.wolffia_cnn_available = False
            return False
        
        try:
            print(f"🔍 Loading {model_path}")
            
            # Load checkpoint to inspect configuration
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # Smart configuration detection from checkpoint
            if isinstance(checkpoint, dict) and 'model_config' in checkpoint:
                # New format with embedded configuration
                config = checkpoint['model_config']
                input_channels = config.get('input_channels', 1)
                use_attention = config.get('use_attention', True)
                multi_task = config.get('multi_task', True)
                base_filters = config.get('base_filters', 32)
                print(f"✅ Found embedded config: {input_channels}ch, attention={use_attention}, multi_task={multi_task}")
            else:
                # Legacy format - try to detect from state dict
                state_dict = checkpoint.get('model_state_dict', checkpoint)
                
                # Detect input channels from first layer
                first_conv_key = None
                for key in state_dict.keys():
                    if 'enc1' in key and 'weight' in key:
                        first_conv_key = key
                        break
                
                if first_conv_key and first_conv_key in state_dict:
                    input_channels = state_dict[first_conv_key].shape[1]
                    print(f"✅ Detected {input_channels} input channels from model weights ({first_conv_key})")
                else:
                    # Fallback to default
                    input_channels = 1
                    print("⚠️ Could not detect input channels, defaulting to 1")
                
                # Default settings for legacy models
                use_attention = True
                multi_task = True
                base_filters = 32
            
            # Import and create model with detected configuration
            from wolffia_cnn_model import WolffiaCNN
            self._cnn_model = WolffiaCNN(
                input_channels=input_channels,
                output_channels=1,
                base_filters=base_filters,
                use_attention=use_attention,
                multi_task=multi_task
            )
            self._cnn_input_channels = input_channels
            print(f"🧠 Created WolffiaCNN: {input_channels}-channel input, multi_task={multi_task}")
                    
            # Load the model weights with proper error handling
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            
            # Try to load state dict with error handling for incompatible architectures
            try:
                self._cnn_model.load_state_dict(state_dict, strict=True)
                print("✅ Model state dictionary loaded successfully (strict)")
            except RuntimeError as strict_error:
                print(f"⚠️ Strict loading failed: {strict_error}")
                print("🔄 Attempting non-strict loading...")
                try:
                    self._cnn_model.load_state_dict(state_dict, strict=False)
                    print("✅ Model state dictionary loaded successfully (non-strict)")
                except Exception as non_strict_error:
                    print(f"❌ Non-strict loading also failed: {non_strict_error}")
                    print("🔄 Trying to filter compatible keys...")
                    
                    # Filter state dict to only include compatible keys
                    model_keys = set(self._cnn_model.state_dict().keys())
                    checkpoint_keys = set(state_dict.keys())
                    compatible_keys = model_keys.intersection(checkpoint_keys)
                    
                    if compatible_keys:
                        filtered_state_dict = {k: state_dict[k] for k in compatible_keys}
                        self._cnn_model.load_state_dict(filtered_state_dict, strict=False)
                        print(f"✅ Partial model loaded with {len(compatible_keys)} compatible parameters")
                    else:
                        raise Exception("No compatible parameters found")
            
            # Move to device and set to eval mode with verification
            target_device = self.device
            self._cnn_model.to(target_device)
            self._cnn_model.eval()
            
            # Verify model is on correct device
            model_device = next(self._cnn_model.parameters()).device
            print(f"🎯 Model moved to device: {model_device}")
            
            # Set public properties for frontend compatibility
            self.wolffia_cnn_model = self._cnn_model
            self.wolffia_cnn_available = True
            
            print(f"✅ Wolffia CNN loaded successfully!")
            print(f"🎯 Input channels: {input_channels}")
            print(f"🧠 Model parameters: {sum(p.numel() for p in self._cnn_model.parameters())}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to load wolffia_cnn_best.pth: {e}")
            import traceback
            traceback.print_exc()
            self.wolffia_cnn_available = False
            self.wolffia_cnn_model = None
            return False
        
    def cnn_detection(self, gray_img, color_img=None):
        """
        SIMPLIFIED: CNN Detection always uses BGR with green-enhanced channels
        
        Args:
            gray_img: Grayscale fallback (not used)
            color_img: BGR color image (required)
        """
        try:
            if self._cnn_model is None:
                print("⚠️ CNN model not available")
                return np.zeros_like(gray_img, dtype=np.int32)
            
            # SIMPLIFIED: Always use BGR with green-enhanced channels
            if color_img is None:
                print("⚠️ Converting grayscale to BGR for CNN detection")
                color_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
            
            # Always use BGR directly with green-enhanced processing
            return self._bgr_cnn_detection(color_img)
                
        except Exception as e:
            print(f"⚠️ CNN detection failed: {e}")
            import traceback
            print(f"Detailed error: {traceback.format_exc()}")
            return np.zeros_like(gray_img)
    
    def _bgr_cnn_detection(self, color_img):
        """
        SIMPLIFIED: CNN detection using BGR directly with green-enhanced channels
        """
        try:
            from wolffia_cnn_model import GreenEnhancedPreprocessor
            preprocessor = GreenEnhancedPreprocessor()
            
            print("🧠 CNN Detection: Using BGR with green-enhanced channels...")
            
            # Create green-enhanced 3-channel input from BGR
            enhanced_rgb = preprocessor.create_green_enhanced_channels(color_img)
            
            # SIMPLIFIED: Always process with green-enhanced channels
            green_percentage = preprocessor.analyze_green_content(color_img)
            print(f"🟢 Green content: {green_percentage:.1f}% - Processing anyway")
            
            enhanced_img = enhanced_rgb
            original_shape = enhanced_img.shape[:2]

            patch_size = 128
            overlap = 32
            full_prediction = np.zeros(original_shape, dtype=np.float32)
            count_map = np.zeros(original_shape, dtype=np.float32)

            print(f"🔍 Processing {original_shape} image with {patch_size}x{patch_size} patches...")

            for y in range(0, original_shape[0] - patch_size + 1, patch_size - overlap):
                for x in range(0, original_shape[1] - patch_size + 1, patch_size - overlap):
                    y_end = min(y + patch_size, original_shape[0])
                    x_end = min(x + patch_size, original_shape[1])
                    patch = enhanced_img[y:y_end, x:x_end, :]

                    if patch.shape[:2] != (patch_size, patch_size):
                        patch = cv2.resize(patch, (patch_size, patch_size))

                    patch_tensor = torch.from_numpy(patch.transpose(2, 0, 1)).unsqueeze(0).float().to(next(self._cnn_model.parameters()).device)

                    with torch.no_grad():
                        output = self._cnn_model(patch_tensor)
                        patch_pred = output[0] if isinstance(output, (tuple, list)) else output
                        patch_pred = patch_pred.squeeze().cpu().numpy()

                    if patch_pred.shape != (y_end - y, x_end - x):
                        patch_pred = cv2.resize(patch_pred, (x_end - x, y_end - y))

                    full_prediction[y:y_end, x:x_end] += patch_pred
                    count_map[y:y_end, x:x_end] += 1

            count_map[count_map == 0] = 1
            averaged = full_prediction / count_map
            sigmoid_map = 1 / (1 + np.exp(-averaged))

            if np.isinf(averaged).any() or sigmoid_map.max() - sigmoid_map.min() < 1e-4:
                print("🚫 CNN output too flat or invalid — skipping detection")
                return np.zeros_like(sigmoid_map, dtype=np.int32)

            otsu_val, _ = cv2.threshold((sigmoid_map * 255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            otsu_val = float(otsu_val)
            # More conservative thresholding to prevent large regions
            high_thresh = max(min(otsu_val / 255.0, 0.7), 0.5)  # Cap at 0.7, min 0.5
            low_thresh = max(high_thresh * 0.8, 0.3)  # Raise low threshold

            high_mask = (sigmoid_map > high_thresh).astype(np.uint8)
            potential_mask = (sigmoid_map > low_thresh).astype(np.uint8)
            print(f"🔍 Thresholds: high={high_thresh:.3f}, low={low_thresh:.3f}")
            print(f"🔍 High mask pixels: {high_mask.sum()}, Potential mask pixels: {potential_mask.sum()}")
            
            binary_mask = cv2.morphologyEx(high_mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=2)
            print(f"🔍 Binary mask pixels after morphology: {binary_mask.sum()}")

            if binary_mask.sum() > 0:
                # More conservative smart expansion to prevent connecting distant regions
                dilated = cv2.dilate(binary_mask, np.ones((2, 2), np.uint8), iterations=1)  # Smaller kernel
                smart = (dilated & potential_mask).astype(np.uint8)
                # Only add smart expansion if it doesn't create regions that are too large
                smart_regions = measure.label(smart)
                filtered_smart = np.zeros_like(smart)
                for region in measure.regionprops(smart_regions):
                    if region.area <= self.max_cell_area * 2:  # Prevent huge regions
                        filtered_smart[smart_regions == region.label] = 1
                binary_mask |= filtered_smart
                print(f"🧠 Conservative smart expansion added regions")

            dist_transform = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 3)
            if dist_transform.max() > 0:
                # More conservative distance transform parameters
                min_dist = max(12, int(np.sqrt(original_shape[0]**2 + original_shape[1]**2) * 0.012))  # Increased
                local_maxima = feature.peak_local_max(dist_transform, min_distance=min_dist, 
                                                      threshold_abs=0.3 * dist_transform.max(),  # Increased threshold
                                                      exclude_border=True)
                print(f"🎯 Found {len(local_maxima)} potential cell centers with conservative parameters")
            else:
                local_maxima = np.array([])

            markers = np.zeros_like(binary_mask, dtype=np.int32)
            for i, (y, x) in enumerate(local_maxima, 1):
                markers[y, x] = i

            if len(local_maxima) > 0:
                markers = cv2.dilate(markers.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1).astype(np.int32)
                # Use original color image for watershed
                labels = cv2.watershed(color_img, markers)
                labels[labels == -1] = 0
            else:
                print("⚠️ No local maxima found — using erosion-based fallback for markers")
                fallback_kernel = np.ones((3, 3), np.uint8)
                eroded = cv2.erode(binary_mask, fallback_kernel, iterations=2)
                _, fallback_markers = cv2.connectedComponents(eroded)
                fallback_markers = cv2.dilate(fallback_markers.astype(np.uint8), fallback_kernel, iterations=1).astype(np.int32)

                # Use original color image for watershed
                labels = cv2.watershed(color_img, fallback_markers)
                labels[labels == -1] = 0
                cv2.imwrite("debug_fallback_markers.png", (fallback_markers > 0).astype(np.uint8) * 255)

            regions = measure.regionprops(labels, intensity_image=sigmoid_map)
            filtered = np.zeros_like(labels)
            label_id = 1

            for region in regions:
                area = region.area
                ecc = region.eccentricity
                sol = region.solidity
                ext = region.extent
                mean_int = region.mean_intensity
                bbox_area = region.bbox_area
                compact = area / bbox_area if bbox_area > 0 else 0

                # Safety check: reject extremely large regions that might be the entire image
                if area > original_shape[0] * original_shape[1] * 0.1:  # More than 10% of image
                    print(f"⚠️ Rejecting region with area {area} (too large)")
                    continue

                criteria = sum([
                    self.min_cell_area <= area <= self.max_cell_area,
                    ecc < 0.85,
                    sol > 0.6,  # Increased solidity requirement
                    ext > 0.5,  # Increased extent requirement
                    mean_int > 0.4,  # Increased intensity requirement
                    compact > 0.5  # Increased compactness requirement
                ])

                # More strict filtering - require at least 5 criteria or very high intensity
                if criteria >= 5 or (criteria >= 4 and mean_int > 0.7):
                    filtered[labels == region.label] = label_id
                    label_id += 1

            print(f"✅ Enhanced CNN: {label_id - 1} valid Wolffia cells detected")
            return filtered.astype(np.int32)

        except Exception as e:
            print(f"⚠️ Enhanced CNN detection failed: {e}")
            import traceback
            print(traceback.format_exc())
            return np.zeros(color_img.shape[:2], dtype=np.int32)
    



    
    def celldetection_detection(self, input_img):
        """
        CellDetection AI detection - ENHANCED AND FIXED
        Based on proven implementation with robust error handling and proper format conversion
        """
        try:
            if not CELLDETECTION_AVAILABLE:
                print("⚠️ CellDetection not available")
                # Return proper 2D labeled image
                if len(input_img.shape) == 3:
                    return np.zeros((input_img.shape[0], input_img.shape[1]), dtype=np.int32)
                else:
                    return np.zeros_like(input_img, dtype=np.int32)
            
            # Handle input image format properly
            if len(input_img.shape) == 2:
                # Grayscale input - convert to RGB for CellDetection
                img_rgb = cv2.cvtColor(input_img, cv2.COLOR_GRAY2RGB)
                target_shape = input_img.shape
            elif len(input_img.shape) == 3:
                # RGB input
                img_rgb = input_img.copy()
                target_shape = input_img.shape[:2]
            else:
                raise ValueError(f"Unsupported image shape: {input_img.shape}")
            
            # Ensure uint8 format
            if img_rgb.dtype != np.uint8:
                img_rgb = np.clip(img_rgb, 0, 255).astype(np.uint8)
            
            # Use lazy-loaded model
            if self.celldetection_model is None:
                print("⚠️ CellDetection model not initialized")
                return np.zeros(target_shape, dtype=np.int32)
            
            with torch.no_grad():
                # Convert to tensor using CellDetection utilities
                x = cd.to_tensor(img_rgb, transpose=True, device=self.device, dtype=torch.float32)
                x = x / 255.0
                x = x.unsqueeze(0)
                
                # Run inference
                outputs = self.celldetection_model(x)
                
                # Extract contours and scores
                contours = outputs.get('contours', [])
                scores = outputs.get('scores', [])
                
                if len(contours) > 0 and len(contours[0]) > 0:
                    # Convert CellDetection results to labeled image
                    labeled_img = self._convert_celldetection_to_labels(
                        contours[0], 
                        scores[0] if len(scores) > 0 else None, 
                        target_shape
                    )
                    
                    # Ensure proper format (2D, int32)
                    if labeled_img.ndim > 2:
                        labeled_img = labeled_img.squeeze()
                    if labeled_img.dtype != np.int32:
                        labeled_img = labeled_img.astype(np.int32)
                    
                    print(f"✅ CellDetection: {np.max(labeled_img)} cells detected")
                    return labeled_img
                
                print("⚠️ CellDetection: No contours detected")
                return np.zeros(target_shape, dtype=np.int32)
                
        except Exception as e:
            print(f"⚠️ CellDetection failed: {e}")
            import traceback
            print(f"Detailed error: {traceback.format_exc()}")
            # Return proper fallback
            if len(input_img.shape) == 3:
                return np.zeros((input_img.shape[0], input_img.shape[1]), dtype=np.int32)
            else:
                return np.zeros_like(input_img, dtype=np.int32)
    
    def _convert_celldetection_to_labels(self, contours, scores, image_shape):
        """
        Convert CellDetection contours to labeled image
        """
        try:
            labeled_img = np.zeros(image_shape, dtype=np.int32)
            
            if scores is None:
                scores = [1.0] * len(contours)
            
            label_id = 1
            for contour, score in zip(contours, scores):
                try:
                    # Convert contour to numpy
                    if isinstance(contour, torch.Tensor):
                        contour_np = contour.cpu().numpy()
                    else:
                        contour_np = np.array(contour)
                    
                    if len(contour_np.shape) == 2 and contour_np.shape[1] == 2:
                        contour_cv = contour_np.reshape((-1, 1, 2)).astype(np.int32)
                    else:
                        continue
                    
                    # Calculate area
                    area = cv2.contourArea(contour_cv)
                    
                    # Size and confidence filters (Wolffia-specific)
                    if (self.min_cell_area <= area <= self.max_cell_area and 
                        score >= 0.3):  # Confidence threshold
                        
                        # Create mask and fill with label
                        mask = np.zeros(image_shape, dtype=np.uint8)
                        cv2.fillPoly(mask, [contour_cv], 255)
                        labeled_img[mask > 0] = label_id
                        label_id += 1
                        
                except Exception as e:
                    continue
            
            return labeled_img
            
        except Exception as e:
            print(f"❌ Failed to convert CellDetection contours: {e}")
            return np.zeros(image_shape, dtype=np.int32)
    
    def fuse_detection_results(self, results, img_shape):
        """
        Intelligent fusion of detection results
        Priority: CNN > Tophat > Watershed
        """
        if not results:
            return np.zeros(img_shape, dtype=np.int32)
        
        # Use highest priority method with valid results
        method_priority = ['cnn', 'celldetection', 'tophat', 'watershed']
        
        for priority_method in method_priority:
            for method, result in results:
                if method == priority_method and np.any(result > 0):
                    return self.filter_by_size(result)
        
        # Fallback to first available result
        return self.filter_by_size(results[0][1])
    
    def filter_by_size(self, labeled_img):
        """Filter objects by size using Wolffia-specific parameters"""
        # Ensure input is integer type
        if labeled_img.dtype != np.int32:
            labeled_img = labeled_img.astype(np.int32)
        
        # Get region properties
        regions = measure.regionprops(labeled_img)
        
        # Create filtered image
        filtered_img = np.zeros_like(labeled_img, dtype=np.int32)
        new_label = 1
        
        for region in regions:
            if self.min_cell_area <= region.area <= self.max_cell_area:
                mask = labeled_img == region.label
                filtered_img[mask] = new_label
                new_label += 1
        
        return filtered_img
    
    def extract_cell_properties(self, labeled_img, gray_img, mask=None, method_name=None):
        """
        Extract cell properties using proven regionprops approach
        Based on microscopist examples 033 and 035
        """
        # Ensure labeled_img is integer type for regionprops
        if labeled_img.dtype != np.int32:
            labeled_img = labeled_img.astype(np.int32)
        
        regions = measure.regionprops(labeled_img, intensity_image=gray_img)
        
        cells = []
        for i, region in enumerate(regions, 1):
            if region.area >= self.min_cell_area:
                cell_data = {
                    'id': i,
                    'area': region.area * (self.pixel_to_micron ** 2),
                    'perimeter': region.perimeter * self.pixel_to_micron,
                    'centroid': [int(region.centroid[1]), int(region.centroid[0])],
                    'major_axis_length': region.major_axis_length * self.pixel_to_micron,
                    'minor_axis_length': region.minor_axis_length * self.pixel_to_micron,
                    'eccentricity': region.eccentricity,
                    'mean_intensity': region.mean_intensity,
                    'max_intensity': region.max_intensity,
                    'min_intensity': region.min_intensity,
                    'method': method_name or 'unknown'
                }
                cells.append(cell_data)
        
        return cells
        
    def analyze_green_content(self, color_img):
        """
        Analyze green content in image for health assessment
        Based on HSV color space analysis for plant health indicators
        
        Args:
            color_img: RGB/BGR color image
            
        Returns:
            float: Percentage of green pixels in the image (0-100)
        """
        try:
            if len(color_img.shape) != 3:
                print("⚠️ Input image must be color (3 channels)")
                return 0.0
            
            # Convert BGR to HSV for better color analysis
            hsv = cv2.cvtColor(color_img, cv2.COLOR_BGR2HSV)
            
            # Define green color range in HSV
            # Green hue is typically around 60-120 in OpenCV HSV (0-179)
            # Use broader range to capture various shades of green
            lower_green = np.array([40, 40, 40])   # Lower bound for green (H,S,V)
            upper_green = np.array([80, 255, 255]) # Upper bound for green (H,S,V)
            
            # Create mask for green pixels
            green_mask = cv2.inRange(hsv, lower_green, upper_green)
            
            # Calculate green percentage
            total_pixels = color_img.shape[0] * color_img.shape[1]
            green_pixels = np.sum(green_mask > 0)
            green_percentage = (green_pixels / total_pixels) * 100.0
            
            print(f"🌱 Green content analysis: {green_percentage:.1f}% green pixels")
            
            return green_percentage
            
        except Exception as e:
            print(f"⚠️ Green content analysis failed: {e}")
            return 0.0
        
    def create_professional_visualization(self, original_img, labeled_img, cell_data):
        """Create professional visualization with cell numbering"""
        # Create overlay
        overlay = original_img.copy()
        
        # Color the segmented regions
        colored_labels = color.label2rgb(labeled_img, bg_label=0, alpha=0.3)
        colored_labels = (colored_labels * 255).astype(np.uint8)
        
        # Blend with original
        result = cv2.addWeighted(overlay, 0.7, colored_labels, 0.3, 0)
        
        # Add cell numbers
        for cell in cell_data:
            center = tuple(cell['centroid'])
            cv2.putText(result, str(cell['id']), center, 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.circle(result, center, 3, (255, 255, 0), -1)
        
        # Save visualization
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        vis_path = self.dirs['results'] / f"analysis_{timestamp}.png"
        cv2.imwrite(str(vis_path), result)
        
        return vis_path
    
    def get_error_result(self, error_msg):
        """Return error result structure"""
        return {
            'total_cells': 0,
            'total_area': 0,
            'average_area': 0,
            'cells': [],
            'labeled_image_path': '',
            'error': error_msg,
            'method_used': [],
            'processing_time': 0,
            'quantitative_analysis': {
                'biomass_analysis': {'total_biomass_mg': 0},
                'color_analysis': {'green_percentage': 0},
                'morphometric_analysis': {'error': error_msg},
                'spatial_analysis': {'error': error_msg},
                'health_assessment': {'overall_health': 'unknown', 'health_score': 0}
            }
        }
    # Enhanced Training methods for tophat ML
    def start_tophat_training(self, file_infos):
        """
        Start tophat training session with image analysis
        FIXED: Now handles file info structure with upload paths
        """
        try:
            session_id = str(uuid.uuid4())[:8]
            session = {
                'id': session_id,
                'created': datetime.now().isoformat(),
                'file_infos': file_infos,
                'images': [],
                'annotations': {}
            }
            
            # Analyze each image to get initial detection results
            for i, file_info in enumerate(file_infos):
                try:
                    # Handle both old format (string paths) and new format (dict with file info)
                    if isinstance(file_info, str):
                        file_path = file_info
                        upload_filename = Path(file_path).name
                        original_filename = Path(file_path).name
                    else:
                        file_path = file_info['server_path']
                        upload_filename = file_info['upload_filename']
                        original_filename = file_info['original_filename']
                    
                    img = cv2.imread(str(file_path))
                    if img is None:
                        continue
                    
                    # Run initial analysis - use tophat if available, otherwise fallback to watershed
                    # This provides the best base detection for users to annotate
                    if self.load_tophat_model():
                        print(f"🎯 Using tophat model for training base detection")
                        result = self.analyze_image(file_path, use_tophat=True, use_cnn=False, use_celldetection=False)
                    else:
                        print(f"🎯 Using watershed method for training base detection (tophat not available)")
                        result = self.analyze_image(file_path, use_tophat=False, use_cnn=False, use_celldetection=False)
                    
                    # Store image info for training session with upload path info
                    # Use the actual detection results as base for annotation
                    cells_data = result.get('cells', [])
                    detection_method = result.get('method_used', ['watershed'])
                    
                    # Get the detection visualization for annotation overlay
                    detection_visualization = None
                    if 'detection_results' in result and 'visualizations' in result:
                        detection_visualization = result['visualizations'].get('detection_overview')
                    elif 'visualizations' in result:
                        detection_visualization = result['visualizations'].get('detection_overview')
                    
                    image_info = {
                        'index': i,
                        'filename': original_filename,
                        'path': str(file_path),
                        'upload_filename': upload_filename,  # For frontend /uploads/ access
                        'initial_cells': cells_data,
                        'total_cells': len(cells_data),
                        'cells_count': len(cells_data),  # Frontend compatibility
                        'detection_method': detection_method,  # Track which method was used for base detection
                        'labeled_image_path': result.get('labeled_image_path', ''),  # Include visualization
                        'detection_visualization': detection_visualization,  # Base64 encoded detection image
                        'image_data': {
                            'visualizations': {
                                'detection_overview': detection_visualization
                            }
                        }  # Frontend compatibility structure
                    }
                    session['images'].append(image_info)
                    
                except Exception as e:
                    print(f"⚠️ Error processing {file_path} for training: {e}")
                    continue
            
            print(f"✅ Training session {session_id} created with {len(session['images'])} images")
            return session
            
        except Exception as e:
            print(f"❌ Failed to start training session: {e}")
            raise
    
    def save_drawing_annotations(self, session_id, image_filename, image_index, annotations, annotated_image):
        """
        Save user drawing annotations for training
        Restored from backup implementation  
        """
        try:
            # Create annotation entry
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            annotation = {
                'session_id': session_id,
                'image_filename': image_filename,
                'image_index': image_index,
                'timestamp': timestamp,
                'annotations': annotations,
                'annotation_counts': {
                    'correct': len(annotations.get('correct', [])),
                    'false_positive': len(annotations.get('false_positive', [])),
                    'missed': len(annotations.get('missed', []))
                }
            }
            
            # Save annotation file
            annotation_filename = f"{timestamp}_{image_index}_{session_id}_{image_filename}_drawing.json"
            annotation_path = self.dirs['annotations'] / annotation_filename
            
            with open(annotation_path, 'w') as f:
                json.dump(annotation, f, indent=2)
            
            # Save annotated image if provided
            if annotated_image:
                try:
                    # Decode base64 image
                    image_data = base64.b64decode(annotated_image.split(',')[1])
                    image_filename_png = f"{timestamp}_{image_index}_{session_id}_{image_filename}_annotated.png"
                    image_path = self.dirs['annotations'] / image_filename_png
                    
                    with open(image_path, 'wb') as f:
                        f.write(image_data)
                    
                    annotation['annotated_image_path'] = str(image_path)
                    
                except Exception as e:
                    print(f"⚠️ Failed to save annotated image: {e}")
            
            print(f"✅ Saved annotations for {image_filename}: {annotation['annotation_counts']}")
            return annotation
            
        except Exception as e:
            print(f"❌ Failed to save annotations: {e}")
            raise

    def train_tophat_model(self, session_id):
        """
        Train tophat model using saved annotations
        Enhanced implementation with Random Forest based on microscopist examples 062-066
        """
        try:
            # Find all annotation files for this session
            annotation_files = list(self.dirs['annotations'].glob(f"*_{session_id}_*_drawing.json"))
            
            if not annotation_files:
                print(f"❌ No annotation files found for session {session_id}")
                return False
            
            print(f"📚 Found {len(annotation_files)} annotation files for training")
            
            # Collect training data
            X_train = []
            y_train = []
            
            for annotation_file in annotation_files:
                try:
                    with open(annotation_file, 'r') as f:
                        annotation = json.load(f)
                    
                    # Find corresponding image
                    image_filename = annotation['image_filename']
                    image_path = None
                    
                    # Enhanced image file resolution with UUID prefix matching
                    image_path = None
                    
                    # Strategy 1: Direct path matching
                    possible_paths = [
                        self.dirs['uploads'] / image_filename,  # Direct uploads path
                        Path(image_filename),  # If it's already a full path
                        Path('uploads') / image_filename,  # Relative uploads path
                        Path(f"uploads/{image_filename}"),  # Alternative uploads format
                    ]
                    
                    # Also check if annotation contains upload_filename or original path info
                    if 'upload_filename' in annotation:
                        possible_paths.extend([
                            self.dirs['uploads'] / annotation['upload_filename'],
                            Path('uploads') / annotation['upload_filename']
                        ])
                    
                    # Check each possible path
                    for possible_path in possible_paths:
                        if possible_path.exists():
                            image_path = possible_path
                            print(f"📁 Found image at: {image_path}")
                            break
                    
                    # Strategy 2: Search for UUID-prefixed files if direct search failed
                    if not image_path:
                        print(f"🔍 Searching for UUID-prefixed file containing: {image_filename}")
                        uploads_dir = self.dirs['uploads']
                        if uploads_dir.exists():
                            # Look for files ending with the target filename
                            for file_path in uploads_dir.glob(f"*_{image_filename}"):
                                if file_path.exists():
                                    image_path = file_path
                                    print(f"📁 Found UUID-prefixed image: {image_path}")
                                    break
                            
                            # Also try direct pattern matching in case the filename is embedded
                            if not image_path:
                                for file_path in uploads_dir.glob(f"*{image_filename}*"):
                                    if file_path.exists():
                                        image_path = file_path
                                        print(f"📁 Found pattern-matched image: {image_path}")
                                        break
                    
                    # Strategy 3: Time-based matching for orphaned annotations
                    if not image_path:
                        print(f"🕐 Attempting time-based matching for annotation: {annotation.get('timestamp', 'unknown')}")
                        timestamp_str = annotation.get('timestamp', '')
                        if timestamp_str:
                            # Extract date components for fuzzy time matching
                            try:
                                annotation_time = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                                uploads_dir = self.dirs['uploads']
                                if uploads_dir.exists():
                                    base_name = image_filename.replace('.png', '').replace('.jpg', '').replace('.jpeg', '').replace('.tiff', '')
                                    
                                    # Find files with similar base name and close timestamp
                                    for file_path in uploads_dir.glob(f"*{base_name}*"):
                                        try:
                                            file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                                            time_diff = abs((annotation_time - file_time).total_seconds())
                                            
                                            # If file was created/modified within 10 minutes of annotation
                                            if time_diff < 600:  # 10 minutes
                                                image_path = file_path
                                                print(f"📁 Found time-matched image: {image_path} (diff: {time_diff:.0f}s)")
                                                break
                                        except:
                                            continue
                            except Exception as e:
                                print(f"⚠️ Could not parse timestamp for time-based matching: {e}")
                    
                    # Check final result - no need for extra print since we already print above
                    
                    if not image_path:
                        print(f"⚠️ Could not find image file: {image_filename}")
                        print(f"   Searched paths: {[str(p) for p in possible_paths]}")
                        # Check if uploads directory exists and list its contents
                        uploads_dir = self.dirs['uploads']
                        if uploads_dir.exists():
                            available_files = list(uploads_dir.glob('*'))
                            print(f"   Available in uploads: {[f.name for f in available_files[:5]]}...")
                        continue
                    
                    # Load image
                    img = cv2.imread(str(image_path))
                    if img is None:
                        continue
                    
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    
                    # Extract ML features
                    features = self.extract_ml_features(gray)
                    
                    # Create training labels from annotations
                    labels = self._create_training_labels_from_annotations(annotation, gray.shape)
                    
                    # Add to training data
                    X_train.append(features.reshape(-1, features.shape[-1]))
                    y_train.append(labels.flatten())
                    
                    print(f"✅ Processed {image_filename} for training")
                    
                except Exception as e:
                    print(f"⚠️ Error processing annotation {annotation_file}: {e}")
                    continue
            
            if not X_train:
                print("❌ No valid training data collected")
                return False
            
            # Combine all training data
            X = np.vstack(X_train)
            y = np.hstack(y_train)
            
            print(f"📊 Training data shape: X={X.shape}, y={y.shape}")
            print(f"📊 Label distribution: {np.bincount(y.astype(int))}")
            
            # Train Random Forest model (Citation: python_for_microscopists examples 062-066)
            model = RandomForestClassifier(
                n_estimators=100, 
                random_state=42,
                max_depth=10,
                min_samples_split=5,
                n_jobs=-1
            )
            
            print("🔧 Training Random Forest model...")
            model.fit(X, y)
            
            # Save model
            model_path = self.dirs['models'] / 'tophat_model.pkl'
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            # Save training info
            training_info = {
                'session_id': session_id,
                'training_date': datetime.now().isoformat(),
                'num_images': len(annotation_files),
                'num_samples': len(y),
                'feature_shape': features.shape,
                'label_distribution': np.bincount(y.astype(int)).tolist(),
                'model_params': model.get_params()
            }
            
            info_path = self.dirs['models'] / 'tophat_model_info.json'
            with open(info_path, 'w') as f:
                json.dump(training_info, f, indent=2)
            
            # Update model reference
            self._tophat_model = model
            self.tophat_model = model
            
            print("✅ Tophat model trained and saved successfully")
            print(f"📊 Model accuracy on training data: {model.score(X, y):.3f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Tophat model training failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _create_training_labels_from_annotations(self, annotation, image_shape):
        """Create training labels from user annotations"""
        try:
            labels = np.zeros(image_shape, dtype=np.uint8)
            annotations_data = annotation.get('annotations', {})
            
            # Mark correct detections as positive (1)
            for correct_cell in annotations_data.get('correct', []):
                if 'center' in correct_cell:
                    cx, cy = correct_cell['center']
                    # Create small positive region around center
                    y_min = max(0, cy - 10)
                    y_max = min(image_shape[0], cy + 10)
                    x_min = max(0, cx - 10)
                    x_max = min(image_shape[1], cx + 10)
                    labels[y_min:y_max, x_min:x_max] = 1
            
            # Mark false positives as negative (0) - they're already 0
            # Mark missed cells as positive (1)
            for missed_cell in annotations_data.get('missed', []):
                if 'center' in missed_cell:
                    cx, cy = missed_cell['center']
                    # Create positive region around missed cell
                    y_min = max(0, cy - 10)
                    y_max = min(image_shape[0], cy + 10)
                    x_min = max(0, cx - 10)
                    x_max = min(image_shape[1], cx + 10)
                    labels[y_min:y_max, x_min:x_max] = 1
            
            return labels
            
        except Exception as e:
            print(f"⚠️ Error creating training labels: {e}")
            return np.zeros(image_shape, dtype=np.uint8)
    
    def get_tophat_status(self):
        """Get comprehensive tophat model status"""
        try:
            model_path = self.dirs['models'] / 'tophat_model.pkl'
            info_path = self.dirs['models'] / 'tophat_model_info.json'
            
            # Check if model file exists and is valid
            model_file_valid = False
            if model_path.exists():
                try:
                    # Try to validate the model file
                    import pickle
                    with open(model_path, 'rb') as f:
                        test_model = pickle.load(f)
                    # If we can load it and it has the right methods, it's valid
                    model_file_valid = hasattr(test_model, 'predict') and hasattr(test_model, 'predict_proba')
                except Exception as e:
                    print(f"⚠️ Model file exists but is invalid: {e}")
                    model_file_valid = False
            
            status = {
                'model_available': model_path.exists(),
                'model_trained': model_file_valid,  # File exists and is a valid trained model
                'model_loaded_in_memory': self._tophat_model is not None,
                'model_path': str(model_path),
                'training_info_available': info_path.exists(),
                'model_file_size': model_path.stat().st_size if model_path.exists() else 0
            }
            
            # Load training info if available
            if info_path.exists():
                try:
                    with open(info_path, 'r') as f:
                        training_info = json.load(f)
                    status['training_info'] = training_info
                    status['training_sessions_count'] = training_info.get('training_sessions', 0)
                    status['last_trained'] = training_info.get('last_trained', 'Unknown')
                except Exception as e:
                    print(f"⚠️ Error reading training info: {e}")
            
            # Additional validation - check if we have annotation data
            annotations_dir = self.dirs.get('annotations', Path('annotations'))
            if annotations_dir.exists():
                annotation_files = list(annotations_dir.glob('*.json'))
                status['annotation_files_count'] = len(annotation_files)
                status['has_training_data'] = len(annotation_files) > 0
            else:
                status['annotation_files_count'] = 0
                status['has_training_data'] = False
            
            print(f"🔍 Tophat status: model_available={status['model_available']}, model_trained={status['model_trained']}, has_training_data={status['has_training_data']}")
            
            return status
            
        except Exception as e:
            print(f"⚠️ Error getting tophat status: {e}")
            return {
                'model_available': False,
                'model_trained': False,
                'error': str(e)
            }
    
    def get_celldetection_status(self):
        """Get CellDetection model status for web interface"""
        return {
            'available': CELLDETECTION_AVAILABLE,
            'model_loaded': self._celldetection_model is not None,
            'device': getattr(self, 'device', 'cpu'),
            'model_name': 'ginoro_CpnResNeXt101UNet-fbe875f1a3e5ce2c' if CELLDETECTION_AVAILABLE else None
        }


# Simple function interface for backward compatibility
def analyze_wolffia_image(image_path, **kwargs):
    """Simple function interface for image analysis"""
    analyzer = WolffiaAnalyzer()
    return analyzer.analyze_image(image_path, **kwargs)


if __name__ == "__main__":
    # Test the system
    analyzer = WolffiaAnalyzer()
    print("✅ BIOIMAGIN Professional System - Deployment Ready")