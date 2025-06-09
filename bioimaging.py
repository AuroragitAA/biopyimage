#!/usr/bin/env python3
"""
BIOIMAGIN Professional Wolffia Analysis System - OPTIMIZED FINAL VERSION
Specifically tuned for Wolffia arrhiza cell detection and analysis
Author: BIOIMAGIN Senior Bioimage Analysis Engineer
"""

import base64
import json
import os
import pickle
import traceback
import warnings
from datetime import datetime
from io import BytesIO
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use('Agg')  # Use non-interactive backend to prevent threading issues
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import pandas as pd
from scipy import ndimage
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
import celldetection as cd

# GPU and advanced processing imports
try:
    import torch
    TORCH_AVAILABLE = True
    print("✅ PyTorch available")
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch not available")

try:
    from cellpose import models
    CELLPOSE_AVAILABLE = True
    print("✅ CellPose available")
except ImportError:
    CELLPOSE_AVAILABLE = False
    print("⚠️ CellPose not available - using optimized watershed")

try:
    from skimage import feature, filters, measure, morphology, segmentation
    from skimage.color import rgb2gray, rgb2hsv
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("❌ Skimage required for advanced processing")



try:
    import celldetection as cd
    import torch
    CELLDETECTION_AVAILABLE = True
    print("✅ CellDetection library loaded successfully")
except ImportError as e:
    CELLDETECTION_AVAILABLE = False
    print(f"⚠️ CellDetection not available: {e}")

class WolffiaAnalyzer:
    """
    Professional Wolffia Arrhiza Analysis System
    Optimized for accurate small cell detection and quantification with tophat AI training
    """
    
    def __init__(self, pixel_to_micron_ratio=0.5, chlorophyll_threshold=0.6):
        """Initialize analyzer with Wolffia-specific parameters"""
        self.setup_directories()
        self.initialize_parameters()
        self.load_tophat_model()
        self.initialize_biomass_models()
        
        # NEW: Initialize CellDetection model
        self.initialize_celldetection_model()
        
        # User parameters
        self.pixel_to_micron_ratio = pixel_to_micron_ratio
        self.chlorophyll_threshold = chlorophyll_threshold
        
        print("🔬 BIOIMAGIN Wolffia Analyzer Initialized with CellDetection")
        
        # ============================================================================
    def initialize_celldetection_model(self):
        """Initialize CellDetection model for professional cell detection"""
        try:
            if not CELLDETECTION_AVAILABLE:
                self.celldetection_model = None
                self.device = 'cpu'
                print("⚠️ CellDetection not available - using classical methods only")
                return
            
            # Set device (GPU if available, otherwise CPU)
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"🎯 CellDetection device: {self.device}")
            
            # Load pretrained model (trained on multiple cell datasets)
            model_name = 'ginoro_CpnResNeXt101UNet-fbe875f1a3e5ce2c'
            print(f"📥 Loading CellDetection model: {model_name}")
            
            self.celldetection_model = cd.fetch_model(model_name, check_hash=True)
            self.celldetection_model = self.celldetection_model.to(self.device)
            self.celldetection_model.eval()
            
            print("✅ CellDetection model loaded successfully")
            
            # Model parameters for Wolffia
            self.celldetection_params = {
                'confidence_threshold': 0.3,    # Lower for small cells
                'nms_threshold': 0.5,           # Non-maximum suppression
                'min_cell_size': 50,            # Minimum cell size in pixels
                'max_cell_size': 3000,          # Maximum cell size in pixels
            }
            
        except Exception as e:
            print(f"❌ Failed to initialize CellDetection model: {e}")
            self.celldetection_model = None
            self.device = 'cpu'

    def celldetection_inference(self, image_rgb):
        """Run CellDetection inference on image"""
        try:
            if self.celldetection_model is None:
                print("⚠️ CellDetection model not available")
                return []
            
            print("🧠 Running CellDetection inference...")
            
            # Preprocess image for CellDetection
            if len(image_rgb.shape) == 2:
                # Convert grayscale to RGB
                img = cv2.cvtColor(image_rgb, cv2.COLOR_GRAY2RGB)
            else:
                img = image_rgb.copy()
            
            print(f"📸 Input image shape: {img.shape}, dtype: {img.dtype}, range: [{img.min()}, {img.max()}]")
            
            # Ensure image is uint8
            if img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)
            
            # Convert to tensor
            with torch.no_grad():
                # Convert to tensor: numpy[H,W,C] -> tensor[C,H,W]
                x = cd.to_tensor(img, transpose=True, device=self.device, dtype=torch.float32)
                
                # Normalize to 0-1 range
                x = x / 255.0
                
                # Add batch dimension: [C,H,W] -> [1,C,H,W]
                x = x.unsqueeze(0)
                
                print(f"🎯 Tensor shape: {x.shape}, device: {x.device}")
                
                # Run inference
                start_time = datetime.now()
                outputs = self.celldetection_model(x)
                inference_time = (datetime.now() - start_time).total_seconds()
                
                print(f"⚡ CellDetection inference completed in {inference_time:.2f}s")
                
                # Extract results
                contours = outputs.get('contours', [])
                scores = outputs.get('scores', [])
                
                if len(contours) > 0 and len(contours[0]) > 0:
                    print(f"🎯 CellDetection found {len(contours[0])} potential cells")
                    
                    # Convert to our cell format
                    cells = self._convert_celldetection_results(contours[0], scores[0] if len(scores) > 0 else None, img)
                    
                    print(f"✅ CellDetection processed {len(cells)} valid cells")
                    return cells
                else:
                    print("⚠️ CellDetection found no cells")
                    return []
                    
        except Exception as e:
            print(f"❌ CellDetection inference failed: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _convert_celldetection_results(self, contours, scores, original_image):
        """Convert CellDetection results to our cell format"""
        try:
            cells = []
            
            if scores is None:
                scores = [1.0] * len(contours)  # Default scores if not available
            
            for i, (contour, score) in enumerate(zip(contours, scores)):
                try:
                    # Convert contour to numpy array
                    if isinstance(contour, torch.Tensor):
                        contour_np = contour.cpu().numpy()
                    else:
                        contour_np = np.array(contour)
                    
                    # Ensure contour has correct shape for OpenCV
                    if len(contour_np.shape) == 2 and contour_np.shape[1] == 2:
                        # Reshape to OpenCV format: (n_points, 1, 2)
                        contour_cv = contour_np.reshape((-1, 1, 2)).astype(np.int32)
                    else:
                        print(f"⚠️ Unexpected contour shape: {contour_np.shape}")
                        continue
                    
                    # Calculate basic properties
                    area = cv2.contourArea(contour_cv)
                    
                    # Filter by size
                    if not (self.celldetection_params['min_cell_size'] <= area <= self.celldetection_params['max_cell_size']):
                        continue
                    
                    # Filter by confidence
                    if score < self.celldetection_params['confidence_threshold']:
                        continue
                    
                    # Calculate center
                    M = cv2.moments(contour_cv)
                    if M["m00"] == 0:
                        continue
                    
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # Calculate additional properties
                    perimeter = cv2.arcLength(contour_cv, True)
                    circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                    
                    # Calculate intensity
                    mask = np.zeros(original_image.shape[:2], dtype=np.uint8)
                    cv2.fillPoly(mask, [contour_cv], 255)
                    
                    if len(original_image.shape) == 3:
                        cell_region = original_image[mask > 0]
                        avg_intensity = np.mean(cell_region) if len(cell_region) > 0 else 0
                        green_intensity = np.mean(cell_region[:, 1]) if len(cell_region) > 0 and cell_region.ndim > 1 else avg_intensity
                    else:
                        cell_pixels = original_image[mask > 0]
                        avg_intensity = np.mean(cell_pixels) if len(cell_pixels) > 0 else 0
                        green_intensity = avg_intensity
                    
                    # Create cell object
                    cell = {
                        'id': len(cells) + 1,
                        'center': (cx, cy),
                        'area': area,
                        'contour': contour_cv.tolist(),
                        'intensity': float(avg_intensity),
                        'green_intensity': float(green_intensity),
                        'method': 'celldetection_cpn',
                        'confidence': float(score),
                        'circularity': circularity,
                        'perimeter': perimeter,
                        'validation_score': float(score),  # Use CPN confidence as validation score
                    }
                    
                    cells.append(cell)
                    
                except Exception as e:
                    print(f"⚠️ Error processing cell {i}: {e}")
                    continue
            
            return cells
            
        except Exception as e:
            print(f"❌ Failed to convert CellDetection results: {e}")
            return []
        
    def initialize_biomass_models(self):
        """Initialize Wolffia-specific biomass calculation models"""
        self.biomass_params = {
            # Wolffia arrhiza specific parameters (research-based)
            'cell_thickness_microns': 15.0,  # Average cell thickness
            'cell_density_mg_mm3': 1.05,     # Cell density (similar to plant tissue)
            'volume_correction_factor': 0.85,  # Accounts for cell shape irregularity
            'chlorophyll_density_mg_g': 2.5,  # mg chlorophyll per g fresh weight
            'max_growth_rate_per_day': 0.3,   # Maximum growth rate
            'optimal_size_range_microns': (80, 150),  # Optimal cell size range
            'green_intensity_biomass_factor': 0.0012,  # Biomass per green intensity unit
        }
    
    def setup_directories(self):
        """Create necessary directories"""
        self.dirs = {
            'results': Path('results'),
            'uploads': Path('uploads'), 
            'models': Path('models'),
            'annotations': Path('annotations'),
            'tophat_training': Path('tophat_training'),
            'wolffia_results': Path('wolffia_results'),
            'learning_system': Path('learning_system'),
            'training_data': Path('training_data')
        }
        for path in self.dirs.values():
            path.mkdir(exist_ok=True)
        
    def initialize_parameters(self):
        """Initialize Wolffia-specific analysis parameters - FIXED for small cells"""
        self.wolffia_params = {
            # FIXED: Much smaller size parameters for actual small Wolffia in images
            'min_cell_area_pixels': 5,       # Reduced from 400 - much smaller!
            'max_cell_area_pixels': 4000,     # Reduced from 8000
            'min_cell_diameter': 10,          # Reduced from 20 
            'max_cell_diameter': 60,          # Reduced from 100
            
            # Detection sensitivity
            'detection_sensitivity': 0.2,     
            'edge_threshold': 0.2,           
            'area_threshold': 0.2,           
            
            # RELAXED: Morphology parameters for small irregular cells
            'circularity_min': 0.25,         # Much more relaxed from 0.4
            'circularity_max': 1.0,          
            'aspect_ratio_max': 3.0,         # More relaxed from 2.5
            
            # Color analysis (keep relaxed)
            'green_threshold': 0.3,          
            'min_green_intensity': 60,       
            'hsv_hue_min': 25,               
            'hsv_hue_max': 85,               
            'hsv_saturation_min': 0.15,     
            'hsv_value_min': 40,             
            
            # CellPose parameters
            'cellpose_diameter': 25,         # Reduced from 40
            'cellpose_flow_threshold': 0.6, 
            'cellpose_model': 'cyto2',       
            
            # RELAXED: Intensity-based filtering
            'min_cell_intensity': 10,        # Reduced from 50
            'max_background_intensity': 180, 
            
            # RELAXED: Advanced filtering
            'solidity_min': 0.3,            # Much more relaxed from 0.5
            'extent_min': 0.25,             # Much more relaxed from 0.4
            
            # NEW: Seed size parameters for watershed
            'min_seed_area_pixels': 10,     # Very small seeds allowed
            'min_final_cell_area': 50,      # Minimum for final validation
        }

    # ============================================================================
    def prefilter_green_regions(self, image_rgb):
        """RELAXED green region pre-filtering for real-world images"""
        try:
            if len(image_rgb.shape) == 3:
                # Multiple approaches to detect green regions
                green_masks = []
                
                # Approach 1: Simple green channel dominance
                r, g, b = image_rgb[:, :, 0], image_rgb[:, :, 1], image_rgb[:, :, 2]
                green_dominant = (g > r) & (g > b) & (g > self.wolffia_params['min_green_intensity'])
                green_masks.append(green_dominant)
                
                # Approach 2: Relaxed HSV filtering
                hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
                hue, saturation, value = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
                
                # Much more relaxed HSV criteria
                hsv_green = (
                    (hue >= self.wolffia_params['hsv_hue_min']) & 
                    (hue <= self.wolffia_params['hsv_hue_max']) &
                    (saturation >= self.wolffia_params['hsv_saturation_min'] * 255) &
                    (value >= self.wolffia_params.get('hsv_value_min', 40))
                )
                green_masks.append(hsv_green)
                
                # Approach 3: Green intensity above median
                green_channel = image_rgb[:, :, 1]
                green_median = np.median(green_channel)
                intensity_green = green_channel > (green_median + 20)  # Above median + small offset
                green_masks.append(intensity_green)
                
                # Approach 4: Color difference approach (green - red, green - blue)
                green_minus_red = (g.astype(np.int16) - r.astype(np.int16)) > 10
                green_minus_blue = (g.astype(np.int16) - b.astype(np.int16)) > 5
                color_diff_green = green_minus_red & green_minus_blue & (g > 50)
                green_masks.append(color_diff_green)
                
                # Combine all approaches (OR operation - if any method finds green, include it)
                combined_mask = np.zeros_like(green_dominant, dtype=bool)
                for mask in green_masks:
                    combined_mask |= mask
                
                # Apply morphological operations to clean mask
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                combined_mask = cv2.morphologyEx(combined_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
                combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
                
                green_pixel_count = np.sum(combined_mask > 0)
                print(f"🌿 RELAXED green pre-filtering: {green_pixel_count} green pixels found")
                print(f"🌿 Green detection breakdown:")
                print(f"   - Green dominant: {np.sum(green_masks[0])}")
                print(f"   - HSV green: {np.sum(green_masks[1])}")
                print(f"   - Intensity green: {np.sum(green_masks[2])}")
                print(f"   - Color difference: {np.sum(green_masks[3])}")
                print(f"   - Combined total: {green_pixel_count}")
                
                return combined_mask
            else:
                # Grayscale - use intensity thresholding
                intensity_mask = image_rgb > self.wolffia_params['min_cell_intensity']
                print(f"🌿 Grayscale intensity filtering: {np.sum(intensity_mask)} pixels above threshold")
                return intensity_mask.astype(np.uint8)
                
        except Exception as e:
            print(f"⚠️ Green pre-filtering failed: {e}")
            # Return all pixels as potential green (fallback)
            return np.ones(image_rgb.shape[:2], dtype=np.uint8)

    
    def load_tophat_model(self):
        """Load or initialize tophat AI model"""
        model_path = self.dirs['models'] / 'tophat_model.pkl'
        if model_path.exists():
            try:
                with open(model_path, 'rb') as f:
                    self.tophat_model = pickle.load(f)
                print("✅ Tophat model loaded")
            except Exception as e:
                print(f"⚠️ Failed to load tophat model: {e}")
                self.tophat_model = None
        else:
            self.tophat_model = None
            print("📝 No tophat model found - training available")

    # ============================================================================

    def analyze_image(self, image_path, use_tophat=False, use_celldetection=True, **kwargs):
        """
        ENHANCED main analysis method with CellDetection integration
        
        Args:
            image_path: Path to image file
            use_tophat: Whether to use tophat AI model (if available)
            use_celldetection: Whether to use CellDetection AI model (default: True)
            **kwargs: Additional parameters
        """
        print(f"🔬 Starting ENHANCED analysis of: {Path(image_path).name}")
        print(f"🎯 Analysis options: tophat={use_tophat}, celldetection={use_celldetection}")
        start_time = datetime.now()
        
        try:
            # Load and validate image
            if not Path(image_path).exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Cannot load image: {image_path}")
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            print(f"📸 Image loaded successfully: {image.shape[1]}x{image.shape[0]} pixels")
            
            # Smart preprocessing
            print("🔧 Smart preprocessing...")
            processed = self.smart_preprocess(image_rgb)
            
            # Enhanced cell detection with CellDetection option
            print("🧬 Enhanced cell detection...")
            if use_celldetection and CELLDETECTION_AVAILABLE and hasattr(self, 'celldetection_model'):
                # Store the use_celldetection flag for the detection method
                self._use_celldetection = use_celldetection
            else:
                self._use_celldetection = False
                if use_celldetection:
                    print("⚠️ CellDetection requested but not available - using classical methods")
            
            cells = self.smart_detect_cells(processed, use_tophat)
            
            # Calculate enhanced metrics with biomass
            print("📊 Calculating enhanced metrics with biomass...")
            metrics = self.calculate_metrics(cells)
            
            # Create essential visualization
            print("📊 Creating visualization...")
            visualization = self.create_essential_visualization(image_rgb, cells, metrics)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Determine primary detection method used
            primary_method = self._determine_primary_method(cells)
            
            # Prepare ENHANCED results
            result = {
                'success': True,
                'timestamp': start_time.isoformat(),
                'processing_time': processing_time,
                'cells_detected': len(cells),
                'total_cell_area': metrics['total_area'],
                'average_cell_area': metrics['avg_area'],
                'cells_data': cells,
                'visualization': visualization,
                
                # NEW: Detection method information
                'detection_info': {
                    'primary_method': primary_method,
                    'celldetection_used': self._use_celldetection,
                    'tophat_used': use_tophat,
                    'methods_breakdown': self._get_methods_breakdown(cells)
                },
                
                # Enhanced metrics
                'enhanced_metrics': {
                    'biomass_analysis': {
                        'total_biomass_mg': metrics['total_biomass_mg'],
                        'average_biomass_mg': metrics['avg_biomass_mg'],
                        'total_chlorophyll_mg': metrics['total_chlorophyll_mg'],
                        'biomass_density': metrics['total_biomass_mg'] / (metrics['total_area'] * (self.pixel_to_micron_ratio ** 2)) if metrics['total_area'] > 0 else 0
                    },
                    'color_analysis': {
                        'average_green_ratio': metrics['avg_green_ratio'],
                        'green_cell_percentage': metrics['green_cell_percentage']
                    },
                    'population_analysis': {
                        'health_distribution': metrics['health_distribution'],
                        'size_distribution': metrics['size_distribution'],
                        'detection_quality': metrics['detection_confidence']
                    }
                },
                
                'image_info': {
                    'filename': Path(image_path).name,
                    'dimensions': f"{image.shape[1]}x{image.shape[0]}",
                    'channels': image.shape[2] if len(image.shape) == 3 else 1,
                    'file_size_mb': Path(image_path).stat().st_size / (1024*1024) if Path(image_path).exists() else 0,
                    'pixel_to_micron_ratio': self.pixel_to_micron_ratio
                },
                
                # Legacy format compatibility
                'cells': cells,
                'summary': {
                    'total_cells': len(cells),
                    'total_area': metrics['total_area'],
                    'average_area': metrics['avg_area'],
                    'processing_time': processing_time
                }
            }
            
            # Log results with method information
            celldetection_status = "CellDetection" if self._use_celldetection else "Classical"
            print(f"✅ ENHANCED analysis complete: {len(cells)} cells detected using {primary_method}")
            print(f"📊 Biomass: {metrics['total_biomass_mg']:.3f}mg, Green cells: {metrics['green_cell_percentage']:.1f}%")
            print(f"⚡ Processing time: {processing_time:.2f}s ({celldetection_status} methods)")
            
            return result
            
        except Exception as e:
            print(f"❌ Analysis failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return self.create_error_result(str(e))

    def _determine_primary_method(self, cells):
        """Determine the primary detection method used"""
        try:
            if not cells:
                return "none"
            
            # Count cells by method
            method_counts = {}
            for cell in cells:
                method = cell.get('method', 'unknown')
                method_counts[method] = method_counts.get(method, 0) + 1
            
            # Find method with most cells
            primary_method = max(method_counts.keys(), key=lambda k: method_counts[k])
            
            # Simplify method names for display
            method_map = {
                'celldetection_cpn': 'CellDetection AI',
                'watershed_strict': 'Watershed',
                'watershed_relaxed': 'Watershed',
                'green_contours': 'Green Region Detection',
                'fallback_intensity': 'Intensity Detection',
                'fallback_relaxed': 'Fallback Detection'
            }
            
            return method_map.get(primary_method, primary_method)
            
        except Exception as e:
            print(f"⚠️ Could not determine primary method: {e}")
            return "mixed"

    def _get_methods_breakdown(self, cells):
        """Get breakdown of detection methods used"""
        try:
            method_counts = {}
            for cell in cells:
                method = cell.get('method', 'unknown')
                method_counts[method] = method_counts.get(method, 0) + 1
            
            return method_counts
            
        except Exception as e:
            print(f"⚠️ Could not get methods breakdown: {e}")
            return {}
        
    # Legacy method alias for compatibility
    def analyze_single_image(self, image_path, **kwargs):
        """Legacy method alias for compatibility"""
        return self.analyze_image(image_path, **kwargs)
    
    def smart_preprocess(self, image):
        """Comprehensive preprocessing pipeline for Wolffia cell detection"""
        print("🔧 Advanced preprocessing pipeline...")
        
        # Convert to grayscale for processing
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        
        # Assess image quality
        contrast = np.std(gray)
        brightness = np.mean(gray)
        noise_level = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        preprocessing_steps = []
        
        # Step 1: Background removal and normalization
        print("🌄 Step 1: Background removal...")
        background_removed = self.remove_background(gray)
        preprocessing_steps.append(('background_removal', background_removed))
        
        # Step 2: Illumination correction and shadow removal
        print("☀️ Step 2: Illumination correction...")
        illumination_corrected = self.correct_illumination(background_removed)
        preprocessing_steps.append(('illumination_correction', illumination_corrected))
        
        # Step 3: Noise reduction
        print("🧹 Step 3: Noise reduction...")
        denoised = self.reduce_noise(illumination_corrected, noise_level)
        preprocessing_steps.append(('noise_reduction', denoised))
        
        # Step 4: Contrast enhancement
        print("📈 Step 4: Contrast enhancement...")
        contrast_enhanced = self.enhance_contrast(denoised, contrast)
        preprocessing_steps.append(('contrast_enhancement', contrast_enhanced))
        
        # Step 5: Edge preservation and sharpening
        print("🔍 Step 5: Edge enhancement...")
        edge_enhanced = self.enhance_edges(contrast_enhanced)
        preprocessing_steps.append(('edge_enhancement', edge_enhanced))
        
        # Step 6: Final quality assessment
        final_quality = self.assess_preprocessing_quality(edge_enhanced, gray)
        
        return {
            'original': image,
            'processed': edge_enhanced,
            'gray': gray,
            'preprocessing_steps': preprocessing_steps,
            'quality_metrics': {
                'original_contrast': contrast,
                'original_brightness': brightness,
                'original_noise': noise_level,
                'final_quality': final_quality,
                'improvement_score': final_quality.get('overall_score', 0.5)
            }
        }
    
    def remove_background(self, image):
        """Remove background using multiple techniques"""
        try:
            # Method 1: Top-hat transform for background removal
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            tophat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
            
            # Method 2: Rolling ball background subtraction simulation
            blurred_bg = cv2.GaussianBlur(image, (51, 51), 0)
            background_sub = cv2.subtract(image, blurred_bg)
            background_sub = cv2.add(background_sub, 50)  # Add offset
            
            # Method 3: Adaptive background estimation
            adaptive_bg = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                              cv2.THRESH_BINARY, 101, 2)
            
            # Combine methods
            combined = cv2.addWeighted(tophat, 0.4, background_sub, 0.4, 0)
            combined = cv2.addWeighted(combined, 0.8, cv2.bitwise_not(adaptive_bg), 0.2, 0)
            
            # Normalize result
            result = cv2.normalize(combined, None, 0, 255, cv2.NORM_MINMAX)
            
            print("✅ Background removed using hybrid approach")
            return result.astype(np.uint8)
            
        except Exception as e:
            print(f"⚠️ Background removal failed: {e}")
            return image
    
    def correct_illumination(self, image):
        """Correct uneven illumination and remove shadows"""
        try:
            # Method 1: Homomorphic filtering for illumination correction
            image_log = np.log1p(np.array(image, dtype=np.float32))
            
            # Apply Gaussian filter in log domain
            blur = cv2.GaussianBlur(image_log, (21, 21), 0)
            homomorphic = image_log - blur
            
            # Convert back and normalize
            result = np.expm1(homomorphic)
            result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)
            
            # Method 2: CLAHE with multiple tile sizes
            clahe_small = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            clahe_large = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(16, 16))
            
            result_small = clahe_small.apply(result.astype(np.uint8))
            result_large = clahe_large.apply(result.astype(np.uint8))
            
            # Combine different CLAHE results
            final_result = cv2.addWeighted(result_small, 0.7, result_large, 0.3, 0)
            
            print("✅ Illumination corrected using homomorphic filtering + CLAHE")
            return final_result
            
        except Exception as e:
            print(f"⚠️ Illumination correction failed: {e}")
            return image
    
    def reduce_noise(self, image, noise_level):
        """Multi-stage noise reduction"""
        try:
            result = image.copy()
            
            # Stage 1: Bilateral filtering for edge-preserving denoising
            if noise_level < 50:  # Very noisy
                result = cv2.bilateralFilter(result, 9, 80, 80)
                print("📱 Applied strong bilateral filtering")
            elif noise_level < 200:  # Moderately noisy
                result = cv2.bilateralFilter(result, 7, 50, 50)
                print("📱 Applied moderate bilateral filtering")
            
            # Stage 2: Non-local means denoising for texture preservation
            if noise_level < 100:
                result = cv2.fastNlMeansDenoising(result, None, 10, 7, 21)
                print("🎯 Applied non-local means denoising")
            
            # Stage 3: Morphological noise removal
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel, iterations=1)
            result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel, iterations=1)
            
            print("✅ Multi-stage noise reduction completed")
            return result
            
        except Exception as e:
            print(f"⚠️ Noise reduction failed: {e}")
            return image
    
    def enhance_contrast(self, image, original_contrast):
        """Adaptive contrast enhancement"""
        try:
            # Method 1: Adaptive histogram equalization
            if original_contrast < 20:  # Very low contrast
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                result = clahe.apply(image)
                print("📊 Applied strong CLAHE for low contrast")
            elif original_contrast < 40:  # Moderate contrast
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(12, 12))
                result = clahe.apply(image)
                print("📊 Applied moderate CLAHE")
            else:
                result = image
            
            # Method 2: Gamma correction for fine-tuning
            gamma = 1.0
            if np.mean(result) < 100:  # Dark image
                gamma = 1.2
            elif np.mean(result) > 180:  # Bright image
                gamma = 0.8
            
            if gamma != 1.0:
                inv_gamma = 1.0 / gamma
                table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)])
                result = cv2.LUT(result.astype(np.uint8), table.astype(np.uint8))
                print(f"🌟 Applied gamma correction: {gamma}")
            
            return result
            
        except Exception as e:
            print(f"⚠️ Contrast enhancement failed: {e}")
            return image
    
    def enhance_edges(self, image):
        """FIXED edge enhancement with proper type handling"""
        try:
            # Ensure input is uint8
            image = image.astype(np.uint8)
            
            # Method 1: Unsharp masking with safe type conversion
            blurred = cv2.GaussianBlur(image, (5, 5), 1.0)
            
            # Safe arithmetic operations
            image_float = image.astype(np.float32)
            blurred_float = blurred.astype(np.float32)
            
            unsharp_mask = image_float * 1.5 - blurred_float * 0.5
            unsharp_mask = np.clip(unsharp_mask, 0, 255).astype(np.uint8)
            
            # Method 2: Laplacian sharpening with safe operations
            laplacian = cv2.Laplacian(image, cv2.CV_32F)  # Use float32 for Laplacian
            laplacian_abs = np.abs(laplacian)
            laplacian_scaled = (laplacian_abs * 0.3).astype(np.uint8)
            
            # Safe addition
            sharpened = cv2.add(image, laplacian_scaled)
            
            # Combine methods safely
            result = cv2.addWeighted(unsharp_mask, 0.7, sharpened, 0.3, 0)
            
            print("✅ Edge enhancement applied")
            return result
            
        except Exception as e:
            print(f"⚠️ Edge enhancement failed: {e}")
            return image
        
    def assess_preprocessing_quality(self, processed, original):
        """Assess the quality of preprocessing"""
        try:
            # Calculate quality metrics
            contrast_improvement = np.std(processed) / (np.std(original) + 1e-6)
            sharpness = cv2.Laplacian(processed, cv2.CV_64F).var()
            histogram_spread = np.std(cv2.calcHist([processed], [0], None, [256], [0, 256]))
            
            # Overall quality score
            quality_score = min(1.0, (contrast_improvement * 0.4 + 
                                    min(sharpness / 1000, 1) * 0.3 + 
                                    min(histogram_spread / 100, 1) * 0.3))
            
            return {
                'contrast_improvement': contrast_improvement,
                'sharpness': sharpness,
                'histogram_spread': histogram_spread,
                'overall_score': quality_score
            }
            
        except Exception as e:
            print(f"⚠️ Quality assessment failed: {e}")
            return {'overall_score': 0.5}
    


    # ============================================================================
    def assess_image_quality(self, image):
        """Assess image quality to adapt detection parameters"""
        try:
            # Convert to grayscale for analysis
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            
            # Quality metrics
            height, width = gray.shape
            total_pixels = height * width
            
            # 1. Resolution score
            resolution_score = min(1.0, (total_pixels / (1000 * 1000)))  # Normalize to 1MP
            
            # 2. Contrast score
            contrast_score = np.std(gray) / 255.0  # Higher std = better contrast
            
            # 3. Sharpness score (Laplacian variance)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness_score = min(1.0, laplacian_var / 1000.0)  # Normalize
            
            # 4. Noise level (lower is better)
            noise_level = cv2.Laplacian(gray, cv2.CV_64F).var()
            noise_score = max(0.0, 1.0 - (noise_level / 2000.0))
            
            # 5. Dynamic range
            pixel_range = gray.max() - gray.min()
            range_score = pixel_range / 255.0
            
            # Overall quality score (weighted average)
            quality_score = (
                resolution_score * 0.25 +
                contrast_score * 0.25 +
                sharpness_score * 0.2 +
                noise_score * 0.15 +
                range_score * 0.15
            )
            
            # Determine quality category
            if quality_score >= 0.7:
                quality_category = "high"
            elif quality_score >= 0.4:
                quality_category = "medium"
            else:
                quality_category = "low"
            
            quality_info = {
                'overall_score': quality_score,
                'category': quality_category,
                'resolution': (width, height),
                'total_pixels': total_pixels,
                'resolution_score': resolution_score,
                'contrast_score': contrast_score,
                'sharpness_score': sharpness_score,
                'noise_score': noise_score,
                'range_score': range_score
            }
            
            print(f"📊 Image quality assessment:")
            print(f"   - Overall score: {quality_score:.3f} ({quality_category})")
            print(f"   - Resolution: {width}x{height} (score: {resolution_score:.3f})")
            print(f"   - Contrast: {contrast_score:.3f}, Sharpness: {sharpness_score:.3f}")
            
            return quality_info
            
        except Exception as e:
            print(f"⚠️ Quality assessment failed: {e}")
            return {
                'overall_score': 0.5,
                'category': 'medium',
                'resolution': image.shape[:2] if len(image.shape) >= 2 else (100, 100),
                'total_pixels': image.size if hasattr(image, 'size') else 10000
            }


    # ============================================================================
    def _validate_wolffia_cell_properties(self, cell, image_quality=None):
        """ADAPTIVE validation based on image quality"""
        try:
            # Get adaptive thresholds based on image quality
            if image_quality is None:
                # Default medium quality thresholds
                size_min, size_max = 30, 3000
                confidence_min = 0.15
                circularity_min = 0.05
            else:
                quality_score = image_quality['overall_score']
                quality_category = image_quality['category']
                
                print(f"🔍 Adaptive validation for {quality_category} quality image (score: {quality_score:.3f})")
                
                # Adaptive thresholds based on quality
                if quality_category == "high":
                    # High quality - can be more strict
                    size_min, size_max = 80, 3000
                    confidence_min = 0.3
                    circularity_min = 0.15
                    print(f"   - Using STRICT validation: size=[{size_min}-{size_max}], confidence>={confidence_min}")
                elif quality_category == "medium":
                    # Medium quality - moderate thresholds
                    size_min, size_max = 50, 4000
                    confidence_min = 0.2
                    circularity_min = 0.1
                    print(f"   - Using MODERATE validation: size=[{size_min}-{size_max}], confidence>={confidence_min}")
                else:
                    # Low quality - very relaxed thresholds
                    size_min, size_max = 30, 5000
                    confidence_min = 0.1
                    circularity_min = 0.05
                    print(f"   - Using RELAXED validation: size=[{size_min}-{size_max}], confidence>={confidence_min}")
            
            # Size validation
            area = cell.get('area', 0)
            if not (size_min <= area <= size_max):
                print(f"   ❌ Size validation failed: {area} not in [{size_min}-{size_max}]")
                return False
            
            # Confidence validation  
            confidence = cell.get('confidence', 0)
            if confidence < confidence_min:
                print(f"   ❌ Confidence validation failed: {confidence:.3f} < {confidence_min}")
                return False
            
            # Shape validation (if available)
            circularity = cell.get('circularity', 0.5)
            if circularity < circularity_min:
                print(f"   ❌ Circularity validation failed: {circularity:.3f} < {circularity_min}")
                return False
            
            print(f"   ✅ Cell validated: area={area}, confidence={confidence:.3f}, circularity={circularity:.3f}")
            return True
            
        except Exception as e:
            print(f"⚠️ Adaptive validation failed: {e}")
            return False


    # ============================================================================
    def smart_detect_cells(self, processed, use_tophat=False):
        """Quality-adaptive cell detection with CellDetection"""
        print("🧬 QUALITY-ADAPTIVE cell detection with CellDetection...")
        
        enhanced = processed['processed']
        original = processed['original']
        
        # Assess image quality first
        image_quality = self.assess_image_quality(original)
        
        # Method 1: Try CellDetection first (most accurate)
        cells_celldetection = []
        if CELLDETECTION_AVAILABLE and self.celldetection_model is not None:
            print("🧠 Method 1: CellDetection (Primary)...")
            cells_celldetection = self.celldetection_inference(original)
            print(f"   → CellDetection detected: {len(cells_celldetection)} cells")
            
            # ADAPTIVE validation based on image quality
            if len(cells_celldetection) > 0:
                validated_cells = []
                for cell in cells_celldetection:
                    if self._validate_wolffia_cell_properties(cell, image_quality):
                        validated_cells.append(cell)
                
                print(f"   → Validated CellDetection cells: {len(validated_cells)}")
                
                # ADAPTIVE threshold for using CellDetection results
                quality_category = image_quality['category']
                if quality_category == "high":
                    min_cells_threshold = 3  # Need at least 3 cells for high quality
                elif quality_category == "medium":
                    min_cells_threshold = 2  # Need at least 2 cells for medium quality
                else:
                    min_cells_threshold = 1  # Accept even 1 cell for low quality
                
                if len(validated_cells) >= min_cells_threshold:
                    print(f"✅ Using CellDetection as primary method ({len(validated_cells)} >= {min_cells_threshold} threshold)")
                    return validated_cells
                else:
                    print(f"⚠️ CellDetection found {len(validated_cells)} cells, below {min_cells_threshold} threshold for {quality_category} quality")
        
        # Method 2: Quality-adaptive fallback
        print("🔄 Method 2: Quality-adaptive fallback...")
        
        # Choose fallback strategy based on image quality
        quality_category = image_quality['category']
        
        if quality_category == "high":
            # High quality: try classical methods first
            fallback_cells = self._high_quality_fallback(enhanced, original, image_quality)
        elif quality_category == "medium":
            # Medium quality: balanced approach
            fallback_cells = self._medium_quality_fallback(enhanced, original, image_quality)
        else:
            # Low quality: very conservative approach
            fallback_cells = self._low_quality_fallback(enhanced, original, image_quality)
        
        # Combine CellDetection with fallback if both found cells
        if len(cells_celldetection) > 0 and len(fallback_cells) > 0:
            print("🔀 Combining CellDetection with fallback results...")
            combined_results = {
                'celldetection': cells_celldetection,
                'fallback': fallback_cells
            }
            final_cells = self._quality_adaptive_fusion(combined_results, image_quality)
        elif len(fallback_cells) > 0:
            final_cells = fallback_cells
        else:
            final_cells = cells_celldetection
        
        print(f"   → Final result: {len(final_cells)} cells")
        return final_cells


    # ============================================================================
    def _high_quality_fallback(self, enhanced, original, image_quality):
        """Fallback for high quality images - can be more aggressive"""
        try:
            print("🔸 High quality fallback: using classical watershed...")
            
            # For high quality images, try watershed first
            cells_watershed = self.watershed_detection(enhanced, original)
            
            # If watershed finds reasonable number, use it
            if 1 <= len(cells_watershed) <= 50:
                return cells_watershed
            
            # Otherwise try green region detection
            green_mask = self.prefilter_green_regions(original)
            cells_green = self._green_region_detection(enhanced, original, green_mask)
            
            return cells_green[:20] if len(cells_green) > 20 else cells_green  # Limit to 20
            
        except Exception as e:
            print(f"⚠️ High quality fallback failed: {e}")
            return []

    def _medium_quality_fallback(self, enhanced, original, image_quality):
        """Fallback for medium quality images - balanced approach"""
        try:
            print("🔹 Medium quality fallback: balanced detection...")
            
            # Try multiple methods and pick best result
            candidates = []
            
            # Method 1: Green regions (if enough green pixels)
            green_mask = self.prefilter_green_regions(original)
            green_ratio = np.sum(green_mask) / green_mask.size
            
            if green_ratio > 0.001:  # If we have some green pixels
                cells_green = self._green_region_detection(enhanced, original, green_mask)
                if 1 <= len(cells_green) <= 30:
                    candidates.append(('green_regions', cells_green))
            
            # Method 2: Conservative intensity detection
            cells_intensity = self._conservative_intensity_detection(enhanced, original)
            if 1 <= len(cells_intensity) <= 25:
                candidates.append(('conservative_intensity', cells_intensity))
            
            # Pick the method with most reasonable count (not too few, not too many)
            if candidates:
                # Prefer green regions if available, otherwise intensity
                for method_name, cells in candidates:
                    if method_name == 'green_regions':
                        return cells
                return candidates[0][1]  # Return first candidate if no green regions
            
            return []
            
        except Exception as e:
            print(f"⚠️ Medium quality fallback failed: {e}")
            return []

    def _low_quality_fallback(self, enhanced, original, image_quality):
        """Fallback for low quality images - very conservative"""
        try:
            print("🔻 Low quality fallback: conservative detection...")
            
            # For low quality, be very conservative
            # Only use intensity-based detection with high thresholds
            
            working_image = enhanced.copy()
            
            # Use much higher percentile for low quality images
            nonzero_pixels = working_image[working_image > 0]
            if len(nonzero_pixels) == 0:
                return []
            
            # Use top 5% of pixels only (very conservative)
            intensity_threshold = np.percentile(nonzero_pixels, 95)
            print(f"🔻 Conservative threshold: {intensity_threshold}")
            
            binary = working_image > intensity_threshold
            
            # More aggressive morphological operations for low quality
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            binary = cv2.morphologyEx(binary.astype(np.uint8), cv2.MORPH_OPEN, kernel, iterations=2)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
            
            # Find connected components
            num_labels, labels = cv2.connectedComponents(binary)
            print(f"🔻 Found {num_labels - 1} high intensity regions")
            
            cells = []
            for label in range(1, num_labels):
                mask = (labels == label).astype(np.uint8)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    area = cv2.contourArea(contour)
                    
                    # Very conservative size constraints for low quality
                    if 100 <= area <= 2000:  # Stricter than regular fallback
                        perimeter = cv2.arcLength(contour, True)
                        if perimeter > 0:
                            circularity = 4 * np.pi * area / (perimeter * perimeter)
                            if circularity > 0.2:  # Reasonable shape
                                
                                M = cv2.moments(contour)
                                if M["m00"] != 0:
                                    cx = int(M["m10"] / M["m00"])
                                    cy = int(M["m01"] / M["m00"])
                                    
                                    # Calculate intensity
                                    if len(original.shape) == 3:
                                        cell_region = original[mask > 0]
                                        avg_intensity = np.mean(cell_region) if len(cell_region) > 0 else 0
                                    else:
                                        cell_pixels = original[mask > 0]
                                        avg_intensity = np.mean(cell_pixels) if len(cell_pixels) > 0 else 0
                                    
                                    cells.append({
                                        'id': len(cells) + 1,
                                        'center': (cx, cy),
                                        'area': area,
                                        'contour': contour.tolist(),
                                        'intensity': float(avg_intensity),
                                        'method': 'low_quality_conservative',
                                        'circularity': circularity,
                                        'confidence': 0.3  # Lower confidence for low quality
                                    })
                                    
                                    # Limit to reasonable number for low quality
                                    if len(cells) >= 15:
                                        break
                
                if len(cells) >= 15:
                    break
            
            print(f"🔻 Conservative detection found {len(cells)} cells")
            return cells
            
        except Exception as e:
            print(f"⚠️ Low quality fallback failed: {e}")
            return []

    def _conservative_intensity_detection(self, enhanced, original):
        """More conservative intensity detection for medium quality images"""
        try:
            working_image = enhanced.copy()
            nonzero_pixels = working_image[working_image > 0]
            
            if len(nonzero_pixels) == 0:
                return []
            
            # Use top 10% instead of 25% (more conservative)
            intensity_threshold = np.percentile(nonzero_pixels, 90)
            binary = working_image > intensity_threshold
            
            # Moderate morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
            binary = cv2.morphologyEx(binary.astype(np.uint8), cv2.MORPH_OPEN, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
            # Find and validate regions
            num_labels, labels = cv2.connectedComponents(binary)
            cells = []
            
            for label in range(1, num_labels):
                mask = (labels == label).astype(np.uint8)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if 80 <= area <= 3000:  # Conservative size range
                        perimeter = cv2.arcLength(contour, True)
                        if perimeter > 0:
                            circularity = 4 * np.pi * area / (perimeter * perimeter)
                            if circularity > 0.25:  # Reasonable shape
                                
                                M = cv2.moments(contour)
                                if M["m00"] != 0:
                                    cx = int(M["m10"] / M["m00"])
                                    cy = int(M["m01"] / M["m00"])
                                    
                                    cells.append({
                                        'id': len(cells) + 1,
                                        'center': (cx, cy),
                                        'area': area,
                                        'contour': contour.tolist(),
                                        'intensity': float(np.mean(original[mask > 0])) if np.sum(mask) > 0 else 0,
                                        'method': 'conservative_intensity',
                                        'circularity': circularity,
                                        'confidence': 0.4
                                    })
                                    
                                    if len(cells) >= 20:  # Limit to 20 cells
                                        break
                
                if len(cells) >= 20:
                    break
            
            return cells
            
        except Exception as e:
            print(f"⚠️ Conservative intensity detection failed: {e}")
            return []

    # ============================================================================
    # 5. ADD quality-adaptive fusion
    # ============================================================================
    def _quality_adaptive_fusion(self, detection_results, image_quality):
        """Fusion that adapts to image quality"""
        try:
            quality_category = image_quality['category']
            
            # For high quality images, prefer CellDetection strongly
            # For low quality images, give more weight to classical methods
            
            if quality_category == "high":
                method_weights = {'celldetection': 0.9, 'fallback': 0.3}
            elif quality_category == "medium":
                method_weights = {'celldetection': 0.7, 'fallback': 0.5}
            else:  # low quality
                method_weights = {'celldetection': 0.6, 'fallback': 0.6}
            
            return self._intelligent_fusion_with_celldetection(detection_results, None, None)
            
        except Exception as e:
            print(f"⚠️ Quality adaptive fusion failed: {e}")
            return detection_results.get('celldetection', [])
    
    def _intelligent_fusion_with_celldetection(self, detection_results, enhanced, original):
        """Enhanced fusion that prioritizes CellDetection results"""
        try:
            all_cells = []
            
            # Prioritize methods by reliability
            method_priorities = {
                'celldetection': 1.0,      # Highest priority - deep learning
                'watershed': 0.7,          # Medium priority
                'green_contours': 0.6,     # Lower priority
                'fallback': 0.4           # Lowest priority
            }
            
            # Collect all cells with priorities
            for method, cells in detection_results.items():
                priority = method_priorities.get(method, 0.5)
                for cell in cells:
                    cell_copy = cell.copy()
                    cell_copy['method'] = method
                    cell_copy['method_priority'] = priority
                    cell_copy['total_confidence'] = cell.get('confidence', 0.5) * priority
                    all_cells.append(cell_copy)
            
            if not all_cells:
                return []
            
            # Sort by total confidence
            all_cells.sort(key=lambda x: x['total_confidence'], reverse=True)
            
            # Spatial filtering with priority consideration
            final_cells = []
            min_distance = 30  # Minimum distance between cells
            
            for cell in all_cells:
                cx, cy = cell['center']
                
                # Check if too close to existing cells
                too_close = False
                for existing in final_cells:
                    ex, ey = existing['center']
                    distance = np.sqrt((cx - ex)**2 + (cy - ey)**2)
                    
                    if distance < min_distance:
                        # If close, keep the one with higher priority/confidence
                        if cell['total_confidence'] > existing['total_confidence']:
                            final_cells.remove(existing)
                            break
                        else:
                            too_close = True
                            break
                
                if not too_close:
                    final_cells.append(cell)
                    
                    # Limit maximum number of cells
                    if len(final_cells) >= 100:
                        break
            
            # Re-number cells
            for i, cell in enumerate(final_cells):
                cell['id'] = i + 1
            
            return final_cells
            
        except Exception as e:
            print(f"⚠️ Enhanced fusion failed: {e}")
            return []

    # ============================================================================
    # 5. ADD celldetection status check method
    # ============================================================================
    def get_celldetection_status(self):
        """Get CellDetection model status for web interface"""
        return {
            'available': CELLDETECTION_AVAILABLE,
            'model_loaded': self.celldetection_model is not None if hasattr(self, 'celldetection_model') else False,
            'device': getattr(self, 'device', 'cpu'),
            'model_name': 'ginoro_CpnResNeXt101UNet-fbe875f1a3e5ce2c' if CELLDETECTION_AVAILABLE else None
        }


    # ============================================================================
    def _fallback_intensity_detection(self, enhanced, original):
        """RELAXED fallback detection for small cells"""
        try:
            print("🔄 Starting RELAXED fallback intensity-based detection...")
            cells = []
            
            # Use the enhanced (preprocessed) image
            working_image = enhanced.copy()
            
            # Find bright regions using LOWER percentile thresholding
            nonzero_pixels = working_image[working_image > 0]
            if len(nonzero_pixels) == 0:
                return []
            
            # Use top 25% of bright pixels instead of 15%
            intensity_threshold = np.percentile(nonzero_pixels, 75)  # More inclusive
            print(f"💡 Using fallback intensity threshold: {intensity_threshold}")
            
            # Create binary mask
            binary = working_image > intensity_threshold
            
            # Gentler morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            binary = cv2.morphologyEx(binary.astype(np.uint8), cv2.MORPH_OPEN, kernel, iterations=1)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
            
            # Find connected components
            num_labels, labels = cv2.connectedComponents(binary)
            print(f"💡 Found {num_labels - 1} bright regions in fallback")
            
            # Much more relaxed validation for fallback mode
            for label in range(1, num_labels):
                mask = (labels == label).astype(np.uint8)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    area = cv2.contourArea(contour)
                    
                    # MUCH more relaxed size constraints for fallback
                    if 30 <= area <= 5000:  # Very relaxed range
                        perimeter = cv2.arcLength(contour, True)
                        if perimeter > 0:
                            circularity = 4 * np.pi * area / (perimeter * perimeter)
                            if circularity > 0.15:  # Very relaxed circularity
                                
                                # Calculate center and intensity
                                M = cv2.moments(contour)
                                if M["m00"] != 0:
                                    cx = int(M["m10"] / M["m00"])
                                    cy = int(M["m01"] / M["m00"])
                                    
                                    # Calculate average intensity
                                    if len(original.shape) == 3:
                                        cell_region = original[mask > 0]
                                        avg_intensity = np.mean(cell_region) if len(cell_region) > 0 else 0
                                    else:
                                        cell_pixels = original[mask > 0]
                                        avg_intensity = np.mean(cell_pixels) if len(cell_pixels) > 0 else 0
                                    
                                    cells.append({
                                        'id': len(cells) + 1,
                                        'center': (cx, cy),
                                        'area': area,
                                        'contour': contour.tolist(),
                                        'intensity': float(avg_intensity),
                                        'method': 'fallback_relaxed',
                                        'circularity': circularity,
                                        'validation_score': 0.4  # Lower confidence for fallback
                                    })
            
            print(f"🔄 RELAXED fallback detection found {len(cells)} potential cells")
            if len(cells) > 0:
                sizes = [cell['area'] for cell in cells]
                print(f"🔄 Fallback cell sizes: {[f'{s:.0f}' for s in sizes[:10]]}")  # Show first 10
            
            return cells
            
        except Exception as e:
            print(f"⚠️ Relaxed fallback detection failed: {e}")
            return []

    # ============================================================================
    def _green_region_detection(self, enhanced, original, green_mask):
        """Detect cells specifically in green regions"""
        try:
            cells = []
            
            # Apply green mask to enhanced image
            masked = enhanced.copy()
            masked[green_mask == 0] = 0
            
            # Use adaptive threshold on green regions only
            if np.sum(green_mask) > 0:
                # Threshold only the green regions
                green_regions = masked[green_mask > 0]
                if len(green_regions) > 100:  # Enough pixels to work with
                    threshold_val = np.percentile(green_regions, 75)  # Top 25% intensity
                    binary = (masked > threshold_val) & (green_mask > 0)
                    
                    # Clean up binary image
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                    binary = cv2.morphologyEx(binary.astype(np.uint8), cv2.MORPH_OPEN, kernel)
                    
                    # Find contours
                    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    for contour in contours:
                        # Create mask for this contour
                        mask = np.zeros(enhanced.shape, dtype=np.uint8)
                        cv2.fillPoly(mask, [contour], 255)
                        
                        cell = self._strict_validate_wolffia_cell(contour, original, mask, "green_contours")
                        if cell:
                            cells.append(cell)
            
            return cells
            
        except Exception as e:
            print(f"⚠️ Green region detection failed: {e}")
            return []

    def _intensity_blob_detection(self, enhanced, original, green_mask):
        """Simple blob detection on high intensity green regions"""
        try:
            cells = []
            
            # Create intensity-based mask
            if len(original.shape) == 3:
                intensity_image = original[:, :, 1]  # Green channel
            else:
                intensity_image = original
            
            # Combine with green mask
            masked_intensity = intensity_image.copy()
            masked_intensity[green_mask == 0] = 0
            
            # Use very high threshold - only brightest regions
            if np.max(masked_intensity) > 0:
                high_intensity_threshold = np.percentile(masked_intensity[masked_intensity > 0], 90)
                binary = masked_intensity > high_intensity_threshold
                
                # Morphological operations with larger kernel
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
                binary = cv2.morphologyEx(binary.astype(np.uint8), cv2.MORPH_OPEN, kernel)
                
                # Find connected components
                num_labels, labels = cv2.connectedComponents(binary)
                
                for label in range(1, num_labels):
                    mask = (labels == label).astype(np.uint8)
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    for contour in contours:
                        cell = self._strict_validate_wolffia_cell(contour, original, mask, "intensity_blobs")
                        if cell:
                            cells.append(cell)
            
            return cells
            
        except Exception as e:
            print(f"⚠️ Intensity blob detection failed: {e}")
            return []

    # ============================================================================
    # ============================================================================
    def _restrictive_fusion(self, detection_results, enhanced, original):
        """Very restrictive fusion for Wolffia - prefer quality over quantity"""
        try:
            all_cells = []
            
            # Collect all cells with confidence scores
            for method, cells in detection_results.items():
                for cell in cells:
                    cell_copy = cell.copy()
                    cell_copy['method'] = method
                    # Higher confidence for methods that found fewer cells
                    cell_copy['method_confidence'] = 1.0 / (len(cells) + 1)  # Fewer detections = higher confidence
                    cell_copy['total_confidence'] = cell.get('validation_score', 0.5) * cell_copy['method_confidence']
                    all_cells.append(cell_copy)
            
            if not all_cells:
                return []
            
            # Sort by total confidence
            all_cells.sort(key=lambda x: x['total_confidence'], reverse=True)
            
            # Very restrictive spatial filtering - large minimum distance
            final_cells = []
            min_distance = 50  # Minimum 50 pixels between cell centers
            
            for cell in all_cells:
                cx, cy = cell['center']
                
                # Check if too close to existing cells
                too_close = False
                for existing in final_cells:
                    ex, ey = existing['center']
                    distance = np.sqrt((cx - ex)**2 + (cy - ey)**2)
                    if distance < min_distance:
                        too_close = True
                        break
                
                # Only add if not too close and has good confidence
                if not too_close and cell['total_confidence'] > 0.3:
                    final_cells.append(cell)
                    
                    # Limit maximum number of cells (safety check)
                    if len(final_cells) >= 50:  # Max 50 cells per image
                        break
            
            # Re-number cells
            for i, cell in enumerate(final_cells):
                cell['id'] = i + 1
            
            return final_cells
            
        except Exception as e:
            print(f"⚠️ Restrictive fusion failed: {e}")
            return []
    
    def advanced_watershed_detection(self, enhanced, original, preprocessing_steps):
        """Advanced watershed with multiple preprocessing approaches"""
        try:
            all_cells = []
            
            # Try watershed on different preprocessing stages
            for step_name, step_image in preprocessing_steps[-3:]:  # Use last 3 steps
                try:
                    # Multiple thresholding approaches
                    thresholds = [
                        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
                        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
                        cv2.THRESH_TOZERO + cv2.THRESH_OTSU
                    ]
                    
                    for thresh_type in thresholds:
                        _, binary = cv2.threshold(step_image, 0, 255, thresh_type)
                        
                        # Morphological operations with different kernels
                        kernels = [
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                            cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
                        ]
                        
                        for kernel in kernels:
                            # Clean binary image
                            cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
                            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
                            
                            # Distance transform and watershed
                            dist = cv2.distanceTransform(cleaned, cv2.DIST_L2, 5)
                            if dist.max() > 0:
                                _, markers = cv2.threshold(dist, 0.3 * dist.max(), 255, 0)
                                markers = np.uint8(markers)
                                _, markers = cv2.connectedComponents(markers)
                                
                                # Apply watershed
                                if len(original.shape) == 2:
                                    original_bgr = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
                                else:
                                    original_bgr = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)
                                
                                markers_ws = cv2.watershed(original_bgr, markers)
                                
                                # Extract cells from this variant
                                variant_cells = self.extract_cells_from_markers(markers_ws, original, f"watershed_{step_name}_{thresh_type}")
                                all_cells.extend(variant_cells)
                                
                except Exception as e:
                    continue
            
            # Remove duplicates and return best cells
            return self.remove_duplicate_cells(all_cells)
            
        except Exception as e:
            print(f"⚠️ Advanced watershed failed: {e}")
            return []
    
    def edge_based_detection(self, enhanced, original):
        """Edge-based cell detection using multiple edge detectors"""
        try:
            cells = []
            
            # Multiple edge detection methods
            edge_methods = [
                ('canny', cv2.Canny(enhanced, 50, 150)),
                ('sobel', cv2.Sobel(enhanced, cv2.CV_64F, 1, 1, ksize=3)),
                ('laplacian', cv2.Laplacian(enhanced, cv2.CV_64F))
            ]
            
            for method_name, edges in edge_methods:
                try:
                    # Convert to uint8 if needed
                    if edges.dtype != np.uint8:
                        edges = np.uint8(np.absolute(edges))
                    
                    # Find contours in edges
                    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    for contour in contours:
                        area = cv2.contourArea(contour)
                        if 30 <= area <= 2000:  # Wolffia size range
                            # Additional shape validation
                            perimeter = cv2.arcLength(contour, True)
                            if perimeter > 0:
                                circularity = 4 * np.pi * area / (perimeter * perimeter)
                                if 0.2 <= circularity <= 1.0:
                                    M = cv2.moments(contour)
                                    if M["m00"] != 0:
                                        cx = int(M["m10"] / M["m00"])
                                        cy = int(M["m01"] / M["m00"])
                                        
                                        # Calculate intensity
                                        mask = np.zeros(enhanced.shape, dtype=np.uint8)
                                        cv2.fillPoly(mask, [contour], 255)
                                        cell_intensity = np.mean(original[mask > 0])
                                        
                                        cells.append({
                                            'id': len(cells) + 1,
                                            'center': (cx, cy),
                                            'area': area,
                                            'contour': contour.tolist(),
                                            'intensity': float(cell_intensity),
                                            'method': f'edge_{method_name}',
                                            'circularity': circularity,
                                            'perimeter': perimeter
                                        })
                except Exception as e:
                    continue
            
            return self.remove_duplicate_cells(cells)
            
        except Exception as e:
            print(f"⚠️ Edge-based detection failed: {e}")
            return []
    
    def region_growing_detection(self, enhanced, original):
        """Region growing and blob detection"""
        try:
            cells = []
            
            # Method 1: Blob detection with multiple parameters
            params = cv2.SimpleBlobDetector_Params()
            
            # Multiple parameter sets for different cell sizes
            param_sets = [
                {'minArea': 30, 'maxArea': 800, 'minCircularity': 0.3},
                {'minArea': 50, 'maxArea': 1200, 'minCircularity': 0.2},
                {'minArea': 80, 'maxArea': 1500, 'minCircularity': 0.15}
            ]
            
            for param_set in param_sets:
                try:
                    params.minArea = param_set['minArea']
                    params.maxArea = param_set['maxArea']
                    params.minCircularity = param_set['minCircularity']
                    params.filterByArea = True
                    params.filterByCircularity = True
                    params.filterByConvexity = False
                    params.filterByInertia = False
                    
                    detector = cv2.SimpleBlobDetector_create(params)
                    keypoints = detector.detect(enhanced)
                    
                    for kp in keypoints:
                        x, y = int(kp.pt[0]), int(kp.pt[1])
                        size = int(kp.size)
                        
                        # Create approximate contour from keypoint
                        radius = size // 2
                        center = (x, y)
                        
                        # Generate circular contour
                        angles = np.linspace(0, 2*np.pi, 20)
                        contour_points = []
                        for angle in angles:
                            px = int(x + radius * np.cos(angle))
                            py = int(y + radius * np.sin(angle))
                            contour_points.append([px, py])
                        
                        contour = np.array(contour_points).reshape((-1, 1, 2))
                        area = cv2.contourArea(contour)
                        
                        if 30 <= area <= 2000:
                            # Calculate intensity
                            mask = np.zeros(enhanced.shape, dtype=np.uint8)
                            cv2.circle(mask, center, radius, 255, -1)
                            cell_intensity = np.mean(original[mask > 0])
                            
                            cells.append({
                                'id': len(cells) + 1,
                                'center': center,
                                'area': area,
                                'contour': contour.tolist(),
                                'intensity': float(cell_intensity),
                                'method': 'blob_detection'
                            })
                
                except Exception as e:
                    continue
            
            return self.remove_duplicate_cells(cells)
            
        except Exception as e:
            print(f"⚠️ Region growing detection failed: {e}")
            return []
    
    def contour_based_detection(self, enhanced, original):
        """Contour-based detection with advanced shape analysis"""
        try:
            cells = []
            
            # Multiple threshold levels for contour detection
            threshold_levels = [0.4, 0.5, 0.6, 0.7]
            
            for level in threshold_levels:
                try:
                    # Adaptive threshold
                    binary = cv2.adaptiveThreshold(enhanced, 255, 
                                                 cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                 cv2.THRESH_BINARY, 
                                                 int(21 * level) | 1,  # Ensure odd number
                                                 5)
                    
                    # Find contours
                    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    for contour in contours:
                        area = cv2.contourArea(contour)
                        if 30 <= area <= 2000:
                            # Advanced shape analysis
                            perimeter = cv2.arcLength(contour, True)
                            if perimeter > 0:
                                # Calculate multiple shape descriptors
                                circularity = 4 * np.pi * area / (perimeter * perimeter)
                                
                                # Fit ellipse for aspect ratio
                                if len(contour) >= 5:
                                    ellipse = cv2.fitEllipse(contour)
                                    aspect_ratio = ellipse[1][0] / ellipse[1][1] if ellipse[1][1] > 0 else 1
                                else:
                                    aspect_ratio = 1
                                
                                # Convexity and solidity
                                hull = cv2.convexHull(contour)
                                hull_area = cv2.contourArea(hull)
                                solidity = area / hull_area if hull_area > 0 else 0
                                
                                # Filter based on multiple criteria
                                if (0.15 <= circularity <= 1.0 and 
                                    aspect_ratio <= 3.0 and 
                                    solidity >= 0.3):
                                    
                                    M = cv2.moments(contour)
                                    if M["m00"] != 0:
                                        cx = int(M["m10"] / M["m00"])
                                        cy = int(M["m01"] / M["m00"])
                                        
                                        # Calculate intensity
                                        mask = np.zeros(enhanced.shape, dtype=np.uint8)
                                        cv2.fillPoly(mask, [contour], 255)
                                        cell_intensity = np.mean(original[mask > 0])
                                        
                                        cells.append({
                                            'id': len(cells) + 1,
                                            'center': (cx, cy),
                                            'area': area,
                                            'contour': contour.tolist(),
                                            'intensity': float(cell_intensity),
                                            'method': f'contour_adaptive_{level}',
                                            'circularity': circularity,
                                            'aspect_ratio': aspect_ratio,
                                            'solidity': solidity,
                                            'perimeter': perimeter
                                        })
                
                except Exception as e:
                    continue
            
            return self.remove_duplicate_cells(cells)
            
        except Exception as e:
            print(f"⚠️ Contour-based detection failed: {e}")
            return []
    
    def extract_cells_from_markers(self, markers, original, method_name):
        """Extract cells from watershed markers"""
        try:
            cells = []
            
            for label in np.unique(markers):
                if label <= 1:  # Skip background and boundaries
                    continue
                
                mask = (markers == label).astype(np.uint8)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if 30 <= area <= 2000:  # Wolffia size range
                        M = cv2.moments(contour)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            
                            # Calculate intensity
                            cell_intensity = np.mean(original[mask > 0])
                            
                            cells.append({
                                'id': len(cells) + 1,
                                'center': (cx, cy),
                                'area': area,
                                'contour': contour.tolist(),
                                'intensity': float(cell_intensity),
                                'method': method_name
                            })
            
            return cells
            
        except Exception as e:
            print(f"⚠️ Cell extraction failed: {e}")
            return []
    
    def remove_duplicate_cells(self, cells):
        """Remove duplicate cells based on proximity and overlap"""
        if not cells:
            return []
        
        try:
            # Sort by area (larger cells first)
            cells.sort(key=lambda x: x['area'], reverse=True)
            
            filtered_cells = []
            
            for cell in cells:
                is_duplicate = False
                cx, cy = cell['center']
                
                for existing in filtered_cells:
                    ex, ey = existing['center']
                    distance = np.sqrt((cx - ex)**2 + (cy - ey)**2)
                    
                    # Consider cells as duplicates if they're very close
                    min_distance = min(np.sqrt(cell['area']/np.pi), np.sqrt(existing['area']/np.pi)) * 1.5
                    
                    if distance < min_distance:
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    filtered_cells.append(cell)
            
            # Re-number cells
            for i, cell in enumerate(filtered_cells):
                cell['id'] = i + 1
            
            return filtered_cells
            
        except Exception as e:
            print(f"⚠️ Duplicate removal failed: {e}")
            return cells
    
    def intelligent_fusion(self, detection_results, enhanced, original):
        """
        OPTIMIZED fusion with confidence scoring instead of strict consensus
        """
        try:
            all_cells = []
            
            # Collect all cells with method weights
            method_weights = {
                'watershed': 1.0,
                'edge': 0.8,
                'contour': 0.9,
                'adaptive': 0.7,
                'cellpose': 1.0
            }
            
            for method, cells in detection_results.items():
                weight = method_weights.get(method, 0.5)
                for cell in cells:
                    cell_copy = cell.copy()
                    cell_copy['method'] = method
                    cell_copy['method_weight'] = weight
                    # Calculate detection confidence
                    validation_score = self._calculate_validation_score(
                        cell['area'], 
                        cell.get('circularity', 0.5),
                        cell.get('aspect_ratio', 1.0),
                        cell['intensity']
                    )
                    cell_copy['detection_confidence'] = validation_score * weight
                    all_cells.append(cell_copy)

            if not all_cells:
                return []

            # Spatial clustering with confidence weighting
            final_cells = []
            used_cells = set()

            # Sort by detection confidence (highest first)
            all_cells.sort(key=lambda x: x['detection_confidence'], reverse=True)

            for i, cell in enumerate(all_cells):
                if i in used_cells:
                    continue

                # Find nearby cells
                cluster = [cell]
                cluster_indices = [i]

                cx, cy = cell['center']
                search_radius = max(20, np.sqrt(cell['area'] / np.pi) * 2)

                for j, other_cell in enumerate(all_cells[i+1:], i+1):
                    if j in used_cells:
                        continue

                    ox, oy = other_cell['center']
                    distance = np.sqrt((cx - ox)**2 + (cy - oy)**2)

                    if distance < search_radius:
                        cluster.append(other_cell)
                        cluster_indices.append(j)

                # Mark cells as used
                used_cells.update(cluster_indices)

                # Create representative cell from cluster
                if len(cluster) == 1:
                    # Single detection - require high confidence
                    if cluster[0]['detection_confidence'] >= 0.3:  # Lowered threshold
                        representative = self._create_representative_cell(cluster)
                        representative['confidence'] = cluster[0]['detection_confidence']
                        representative['detection_count'] = 1
                        final_cells.append(representative)
                else:
                    # Multiple detections - always include
                    representative = self._create_representative_cell(cluster)
                    representative['confidence'] = np.mean([c['detection_confidence'] for c in cluster])
                    representative['detection_count'] = len(cluster)
                    final_cells.append(representative)

            # Sort by confidence and re-number
            final_cells.sort(key=lambda x: x['confidence'], reverse=True)
            for i, cell in enumerate(final_cells):
                cell['id'] = i + 1

            print(f"🎯 Optimized fusion: {len(final_cells)} high-quality cells")
            return final_cells

        except Exception as e:
            print(f"❌ Fusion failed: {e}")
            return []
        
        
    def _calculate_validation_score(self, area, circularity, aspect_ratio, intensity):
        """Calculate validation score for cell quality"""
        try:
            score = 0.0
            
            # Area score (optimal range gets higher score)
            optimal_area = (self.wolffia_params['min_cell_area_pixels'] + 
                        self.wolffia_params['max_cell_area_pixels']) / 2
            area_score = 1.0 - abs(area - optimal_area) / optimal_area
            score += max(0, area_score) * 0.3
            
            # Circularity score
            circularity_score = circularity  # Higher circularity = higher score
            score += circularity_score * 0.3
            
            # Aspect ratio score (closer to 1 = higher score)
            aspect_score = 1.0 / (1.0 + abs(aspect_ratio - 1.0))
            score += aspect_score * 0.2
            
            # Intensity score (moderate intensity preferred)
            intensity_score = 1.0 - abs(intensity - 128) / 128
            score += intensity_score * 0.2
            
            return max(0.0, min(1.0, score))
        except:
            return 0.5

    def _create_representative_cell(self, cluster):
        """Create representative cell from cluster of detections"""
        try:
            # Weight-averaged properties
            total_weight = sum(c.get('method_weight', 1.0) for c in cluster)
            
            if total_weight == 0:
                total_weight = len(cluster)
            
            # Weighted averages
            avg_cx = sum(c['center'][0] * c.get('method_weight', 1.0) for c in cluster) / total_weight
            avg_cy = sum(c['center'][1] * c.get('method_weight', 1.0) for c in cluster) / total_weight
            avg_area = sum(c['area'] * c.get('method_weight', 1.0) for c in cluster) / total_weight
            avg_intensity = sum(c['intensity'] * c.get('method_weight', 1.0) for c in cluster) / total_weight
            
            # Use contour from highest confidence detection
            best_cell = max(cluster, key=lambda x: x.get('detection_confidence', 0))
            
            return {
                'id': 0,  # Will be set later
                'center': (int(avg_cx), int(avg_cy)),
                'area': int(avg_area),
                'intensity': float(avg_intensity),
                'contour': best_cell.get('contour'),
                'methods': list(set(c['method'] for c in cluster)),
                'circularity': best_cell.get('circularity', 0.5),
                'aspect_ratio': best_cell.get('aspect_ratio', 1.0),
                'perimeter': best_cell.get('perimeter', 0)
            }
        except Exception as e:
            print(f"⚠️ Representative cell creation failed: {e}")
            return cluster[0] if cluster else {}
        

    def watershed_detection(self, enhanced, original):
        """FIXED watershed with relaxed size filtering and detailed debugging"""
        try:
            print("🌊 Starting RELAXED watershed for small Wolffia...")
            
            # Step 1: Pre-filter for green regions
            green_mask = self.prefilter_green_regions(original)
            green_pixels = np.sum(green_mask)
            
            if green_pixels < 500:  # Very low threshold
                print(f"⚠️ Only {green_pixels} green pixels found, skipping watershed")
                return []
            
            # Step 2: Apply green mask
            masked_enhanced = enhanced.copy()
            masked_enhanced[green_mask == 0] = 0
            
            # Step 3: More permissive thresholding
            # Use lower percentile to be more inclusive
            _, binary = cv2.threshold(masked_enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Use lower intensity threshold - top 50% instead of 70%
            if np.sum(masked_enhanced > 0) > 0:
                intensity_thresh = np.percentile(masked_enhanced[masked_enhanced > 0], 50)  # More inclusive
                binary = binary & (masked_enhanced > intensity_thresh)
                print(f"💡 Using intensity threshold: {intensity_thresh}")
            
            # Step 4: Gentler morphological cleaning
            # Use smaller kernels to preserve small objects
            kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            kernel_tiny = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            
            # More gentle cleaning
            cleaned = cv2.morphologyEx(binary.astype(np.uint8), cv2.MORPH_OPEN, kernel_tiny, iterations=1)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_small, iterations=1)
            
            print(f"🧹 After gentle morphological cleaning: {np.sum(cleaned > 0)} pixels remain")
            
            # Step 5: More permissive distance transform
            dist_transform = cv2.distanceTransform(cleaned, cv2.DIST_L2, 5)
            
            if dist_transform.max() == 0:
                print("⚠️ No distance transform peaks found")
                return []
            
            # Use much lower threshold for seeds
            dist_threshold = 0.3  # Reduced from 0.7 - much more permissive
            ret, sure_fg = cv2.threshold(dist_transform, dist_threshold * dist_transform.max(), 255, 0)
            sure_fg = np.uint8(sure_fg)
            
            print(f"🎯 Distance threshold: {dist_threshold * dist_transform.max():.1f}")
            print(f"🎯 Foreground pixels after distance transform: {np.sum(sure_fg > 0)}")
            
            if np.sum(sure_fg) == 0:
                print("⚠️ No foreground regions found after distance transform")
                return []
            
            # Step 6: Create watershed markers with RELAXED size filter
            ret, markers = cv2.connectedComponents(sure_fg)
            print(f"🎯 Initial markers found: {ret - 1}")
            
            # Get marker statistics for debugging
            marker_stats = cv2.connectedComponentsWithStats(sure_fg)[2]
            valid_markers = []
            rejected_sizes = []
            
            # Use much smaller minimum seed size
            min_seed_size = self.wolffia_params.get('min_seed_area_pixels', 20)
            
            for i in range(1, ret):  # Skip background (0)
                area = marker_stats[i, cv2.CC_STAT_AREA]
                if area >= min_seed_size:  # Much smaller threshold
                    valid_markers.append(i)
                else:
                    rejected_sizes.append(area)
            
            print(f"🎯 Marker size analysis:")
            print(f"   - Minimum seed size required: {min_seed_size}")
            print(f"   - Valid markers: {len(valid_markers)}")
            print(f"   - Rejected markers: {len(rejected_sizes)}")
            if rejected_sizes:
                print(f"   - Rejected sizes: {rejected_sizes}")
            if len(valid_markers) > 0:
                valid_areas = [marker_stats[i, cv2.CC_STAT_AREA] for i in valid_markers]
                print(f"   - Valid marker sizes: {valid_areas}")
            
            if len(valid_markers) == 0:
                print("⚠️ No markers passed size filter - try reducing min_seed_area_pixels")
                return []
            
            # Create new marker image with only valid markers
            new_markers = np.zeros_like(markers)
            for i, valid_marker in enumerate(valid_markers):
                new_markers[markers == valid_marker] = i + 1
            
            markers = new_markers + 1  # +1 for background
            markers[cleaned == 0] = 0
            
            # Step 7: Watershed
            if len(original.shape) == 3:
                watershed_input = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)
            else:
                watershed_input = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
                
            markers_ws = cv2.watershed(watershed_input, markers)
            
            # Step 8: Extract and validate cells with RELAXED criteria
            cells = []
            final_rejected_sizes = []
            
            for label in np.unique(markers_ws):
                if label <= 1:  # Skip background and boundaries
                    continue
                    
                mask = (markers_ws == label).astype(np.uint8)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    area = cv2.contourArea(contour)
                    
                    # RELAXED final validation with debugging
                    if area >= self.wolffia_params.get('min_final_cell_area', 50):
                        cell = self._relaxed_validate_wolffia_cell(contour, original, mask, "watershed_relaxed")
                        if cell:
                            cells.append(cell)
                        else:
                            print(f"🔍 Cell rejected by shape validation: area={area:.0f}")
                    else:
                        final_rejected_sizes.append(area)
            
            print(f"🎯 Final validation:")
            print(f"   - Cells accepted: {len(cells)}")
            print(f"   - Cells rejected by final size filter: {len(final_rejected_sizes)}")
            if final_rejected_sizes:
                print(f"   - Final rejected sizes: {[f'{s:.0f}' for s in final_rejected_sizes[:10]]}")  # Show first 10
            
            print(f"✅ Relaxed watershed detected {len(cells)} valid Wolffia cells")
            return cells

        except Exception as e:
            print(f"❌ Relaxed watershed detection failed: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _relaxed_validate_wolffia_cell(self, contour, original, mask, method):
        """RELAXED validation for small Wolffia cells with detailed debugging"""
        try:
            area = cv2.contourArea(contour)
            
            # RELAXED size validation
            min_area = self.wolffia_params.get('min_final_cell_area', 50)
            max_area = self.wolffia_params['max_cell_area_pixels']
            
            if not (min_area <= area <= max_area):
                return None
            
            # RELAXED shape validation
            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 0:
                return None
            
            # Much more relaxed circularity
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            min_circularity = self.wolffia_params['circularity_min']
            
            if circularity < min_circularity:
                return None
            
            # RELAXED aspect ratio check
            aspect_ratio = 1.0
            if len(contour) >= 5:
                try:
                    ellipse = cv2.fitEllipse(contour)
                    if ellipse[1][1] > 0:
                        aspect_ratio = ellipse[1][0] / ellipse[1][1]
                        if aspect_ratio > self.wolffia_params['aspect_ratio_max']:
                            return None
                except:
                    pass
            
            # RELAXED solidity check
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = 1.0
            if hull_area > 0:
                solidity = area / hull_area
                if solidity < self.wolffia_params.get('solidity_min', 0.3):
                    return None
            
            # Calculate center and intensity
            M = cv2.moments(contour)
            if M["m00"] == 0:
                return None
            
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            # RELAXED intensity validation
            if len(original.shape) == 3:
                cell_region = original[mask > 0]
                if len(cell_region) == 0:
                    return None
                
                avg_intensity = np.mean(cell_region.mean(axis=1)) if cell_region.ndim > 1 else np.mean(cell_region)
                green_intensity = np.mean(cell_region[:, 1]) if cell_region.ndim > 1 else avg_intensity
            else:
                cell_pixels = original[mask > 0]
                avg_intensity = np.mean(cell_pixels)
                green_intensity = avg_intensity
            
            # Much more relaxed intensity requirement
            min_intensity = self.wolffia_params.get('min_cell_intensity', 40)
            if avg_intensity < min_intensity:
                return None
            
            # Calculate validation score
            validation_score = self._calculate_validation_score(area, circularity, aspect_ratio, avg_intensity)
            
            return {
                'id': 0,  # Will be set later
                'center': (cx, cy),
                'area': area,
                'contour': contour.tolist(),
                'intensity': float(avg_intensity),
                'green_intensity': float(green_intensity),
                'method': method,
                'circularity': circularity,
                'solidity': solidity,
                'aspect_ratio': aspect_ratio,
                'perimeter': perimeter,
                'validation_score': validation_score
            }
            
        except Exception as e:
            print(f"⚠️ Relaxed cell validation failed: {e}")
            return None

    def _strict_validate_wolffia_cell(self, contour, original, mask, method):
        """Strict validation specifically for Wolffia cells"""
        try:
            area = cv2.contourArea(contour)
            
            # Size validation - much stricter
            if not (self.wolffia_params['min_cell_area_pixels'] <= area <= self.wolffia_params['max_cell_area_pixels']):
                return None
            
            # Shape validation
            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 0:
                return None
            
            # Circularity - must be reasonably circular for Wolffia
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < self.wolffia_params['circularity_min']:
                return None
            
            # Aspect ratio check
            if len(contour) >= 5:
                try:
                    ellipse = cv2.fitEllipse(contour)
                    if ellipse[1][1] > 0:
                        aspect_ratio = ellipse[1][0] / ellipse[1][1]
                        if aspect_ratio > self.wolffia_params['aspect_ratio_max']:
                            return None
                except:
                    pass
            
            # Solidity check (how "filled" the shape is)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                solidity = area / hull_area
                if solidity < self.wolffia_params.get('solidity_min', 0.7):
                    return None
            
            # Calculate center and intensity
            M = cv2.moments(contour)
            if M["m00"] == 0:
                return None
            
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            # Intensity validation - must be bright enough
            if len(original.shape) == 3:
                cell_region = original[mask > 0]
                if len(cell_region) == 0:
                    return None
                
                # Check green intensity specifically
                green_intensity = np.mean(cell_region[:, 1]) if cell_region.ndim > 1 else np.mean(cell_region)
                avg_intensity = np.mean(cell_region.mean(axis=1)) if cell_region.ndim > 1 else np.mean(cell_region)
                
                # Must have sufficient green color
                if green_intensity < self.wolffia_params['min_green_intensity']:
                    return None
                    
            else:
                cell_pixels = original[mask > 0]
                avg_intensity = np.mean(cell_pixels)
                green_intensity = avg_intensity
            
            # Must be bright enough to be a cell
            if avg_intensity < self.wolffia_params['min_cell_intensity']:
                return None
            
            return {
                'id': 0,  # Will be set later
                'center': (cx, cy),
                'area': area,
                'contour': contour.tolist(),
                'intensity': float(avg_intensity),
                'green_intensity': float(green_intensity),
                'method': method,
                'circularity': circularity,
                'solidity': solidity,
                'perimeter': perimeter,
                'validation_score': self._calculate_validation_score(area, circularity, 1.0, avg_intensity)
            }
            
        except Exception as e:
            print(f"⚠️ Cell validation failed: {e}")
            return None
        
    def cellpose_detection(self, enhanced, original):
        """CellPose detection for comparison"""
        try:

            model = models.Cellpose(model_type=self.wolffia_params['cellpose_model'])
            
            # Run CellPose
            masks, flows, styles, diams = model.eval(
                enhanced, 
                diameter=self.wolffia_params['cellpose_diameter'],
                flow_threshold=self.wolffia_params['cellpose_flow_threshold'],
                channels=[0,0]
            )
            
            # Extract cells
            cells = []
            for cell_id in np.unique(masks):
                if cell_id == 0:  # Skip background
                    continue
                    
                mask = (masks == cell_id).astype(np.uint8)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    contour = max(contours, key=cv2.contourArea)
                    area = cv2.contourArea(contour)
                    
                    # Size filter
                    if self.wolffia_params['min_cell_area_pixels'] < area < self.wolffia_params['max_cell_area_pixels']:
                        M = cv2.moments(contour)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            
                            # Calculate cell intensity for color coding
                            cell_mask = (masks == cell_id).astype(np.uint8)
                            if len(original.shape) == 3:
                                gray_original = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
                            else:
                                gray_original = original
                            
                            cell_intensity = np.mean(gray_original[cell_mask > 0])
                            
                            cells.append({
                                'id': len(cells) + 1,
                                'center': (cx, cy),
                                'area': area,
                                'contour': contour.tolist() if contour is not None else [],
                                'intensity': float(cell_intensity),
                                'method': 'cellpose'
                            })
            
            return cells
            
        except Exception as e:
            print(f"⚠️ CellPose detection failed: {e}")
            return []
    
    def tophat_detection(self, enhanced, original):
        """AI-powered tophat detection using trained model"""
        try:
            if self.tophat_model is None:
                return []
            
            # Extract features for AI model
            features = self.extract_detection_features(enhanced)
            
            if len(features) == 0:
                return []
            
            # Predict cell locations
            predictions = self.tophat_model.predict(features)
            
            # Convert predictions to cell objects
            cells = []
            # This is a simplified implementation - would need proper feature extraction
            # and prediction-to-cell conversion based on training approach
            
            return cells
            
        except Exception as e:
            print(f"⚠️ Tophat AI detection failed: {e}")
            return []
    
    def fuse_detections(self, watershed_cells, cellpose_cells, tophat_cells, enhanced):
        """Intelligently fuse multiple detection results"""
        all_cells = watershed_cells + cellpose_cells + tophat_cells
        
        if not all_cells:
            return []
        
        # Remove duplicates based on proximity
        final_cells = []
        for cell in all_cells:
            is_duplicate = False
            cx, cy = cell['center']
            
            for existing in final_cells:
                ex, ey = existing['center']
                distance = np.sqrt((cx-ex)**2 + (cy-ey)**2)
                
                if distance < 15:  # Cells too close, likely duplicate
                    # Keep the one with larger area (more confident detection)
                    if cell['area'] > existing['area']:
                        final_cells.remove(existing)
                        final_cells.append(cell)
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                final_cells.append(cell)
        
        # Re-number cells
        for i, cell in enumerate(final_cells):
            cell['id'] = i + 1
        
        return final_cells
    
    def calculate_metrics(self, cells):
        """FIXED metrics calculation with proper error handling"""
        if not cells:
            print("📊 No cells detected - returning zero metrics")
            return {
                'cell_count': 0,
                'total_area': 0,
                'avg_area': 0,
                'min_area': 0,
                'max_area': 0,
                'std_area': 0,
                'avg_intensity': 0,
                'std_intensity': 0,
                'total_biomass_mg': 0,
                'avg_biomass_mg': 0,
                'std_biomass_mg': 0,
                'total_chlorophyll_mg': 0,
                'avg_chlorophyll_mg': 0,
                'avg_green_ratio': 0,
                'green_cell_percentage': 0,  # FIXED: Always include this field
                'health_distribution': {},
                'size_distribution': {'small': 0, 'medium': 0, 'large': 0},
                'detection_confidence': 0,
            }
        
        # Basic metrics
        areas = [cell['area'] for cell in cells]
        intensities = [cell['intensity'] for cell in cells]
        
        # Calculate biomass for all cells
        try:
            cells_with_biomass = self._add_biomass_to_cells(cells)
        except Exception as e:
            print(f"⚠️ Biomass calculation failed: {e}")
            cells_with_biomass = cells  # Use original cells if biomass fails
        
        # Biomass metrics
        biomass_data = []
        chlorophyll_data = []
        health_statuses = []
        color_data = []
        
        for cell in cells_with_biomass:
            if 'biomass' in cell:
                biomass_data.append(cell['biomass']['fresh_weight_mg'])
                chlorophyll_data.append(cell['biomass']['chlorophyll_content_mg'])
            
            if 'health_status' in cell:
                health_statuses.append(cell['health_status']['status'])
            
            if 'color_analysis' in cell:
                color_data.append(cell['color_analysis']['green_ratio'])
        
        # Health distribution
        health_distribution = {}
        for status in health_statuses:
            health_distribution[status] = health_distribution.get(status, 0) + 1
        
        # Size distribution
        size_bins = {'small': 0, 'medium': 0, 'large': 0}
        if areas:
            min_area = min(areas)
            max_area = max(areas)
            range_size = (max_area - min_area) / 3 if max_area > min_area else 1
            
            for area in areas:
                if area < min_area + range_size:
                    size_bins['small'] += 1
                elif area < min_area + 2 * range_size:
                    size_bins['medium'] += 1
                else:
                    size_bins['large'] += 1
        
        # FIXED: Safe calculation of green cell percentage
        try:
            if color_data:
                green_cell_percentage = (len([g for g in color_data if g > 0.35]) / len(color_data) * 100)
            else:
                green_cell_percentage = 0.0  # Default when no color data
        except Exception as e:
            print(f"⚠️ Green cell percentage calculation failed: {e}")
            green_cell_percentage = 0.0
        
        return {
            # Basic metrics (maintain compatibility)
            'cell_count': len(cells),
            'total_area': sum(areas),
            'avg_area': np.mean(areas),
            'min_area': min(areas) if areas else 0,
            'max_area': max(areas) if areas else 0,
            'std_area': np.std(areas) if areas else 0,
            
            # Intensity metrics
            'avg_intensity': np.mean(intensities),
            'std_intensity': np.std(intensities),
            
            # Biomass metrics
            'total_biomass_mg': sum(biomass_data) if biomass_data else 0,
            'avg_biomass_mg': np.mean(biomass_data) if biomass_data else 0,
            'std_biomass_mg': np.std(biomass_data) if biomass_data else 0,
            
            # Chlorophyll metrics
            'total_chlorophyll_mg': sum(chlorophyll_data) if chlorophyll_data else 0,
            'avg_chlorophyll_mg': np.mean(chlorophyll_data) if chlorophyll_data else 0,
            
            # FIXED: Color metrics with safe calculation
            'avg_green_ratio': np.mean(color_data) if color_data else 0,
            'green_cell_percentage': green_cell_percentage,  # Now always present
            
            # Distributions
            'health_distribution': health_distribution,
            'size_distribution': size_bins,
            
            # Quality metrics
            'detection_confidence': np.mean([cell.get('confidence', 0.5) for cell in cells]),
        }
        
    
    def _add_biomass_to_cells(self, cells):
        """Add biomass calculations to cells"""
        cells_with_biomass = []
        
        for cell in cells:
            try:
                # Basic measurements
                area_pixels = cell['area']
                intensity = cell['intensity']
                
                # Convert to real-world measurements
                area_microns_sq = area_pixels * (self.pixel_to_micron_ratio ** 2)
                
                # Biomass calculations
                biomass_data = self._calculate_cell_biomass(area_microns_sq, intensity)
                
                # Color analysis
                color_data = self._analyze_cell_color(cell, intensity)
                
                # Health assessment
                health_data = self._assess_cell_health(biomass_data, color_data)
                
                # Enhanced cell data
                enhanced_cell = cell.copy()
                enhanced_cell.update({
                    'area_microns_squared': area_microns_sq,
                    'biomass': biomass_data,
                    'color_analysis': color_data,
                    'health_status': health_data
                })
                
                cells_with_biomass.append(enhanced_cell)
                
            except Exception as e:
                print(f"⚠️ Biomass calculation failed for cell {cell.get('id', 'unknown')}: {e}")
                cells_with_biomass.append(cell)  # Add without biomass data
        
        return cells_with_biomass

    def _calculate_cell_biomass(self, area_microns_sq, intensity):
        """Calculate biomass metrics for a single cell"""
        try:
            # Volume estimation (assuming ellipsoid shape)
            thickness = self.biomass_params['cell_thickness_microns']
            volume_microns_cubed = area_microns_sq * thickness * self.biomass_params['volume_correction_factor']
            
            # Convert to mm³ for standard units
            volume_mm_cubed = volume_microns_cubed / (1000 ** 3)
            
            # Fresh weight estimation
            density = self.biomass_params['cell_density_mg_mm3']
            fresh_weight_mg = volume_mm_cubed * density
            
            # Dry weight estimation (typically 10-15% of fresh weight)
            dry_weight_mg = fresh_weight_mg * 0.12
            
            # Chlorophyll content estimation based on green intensity
            normalized_intensity = intensity / 255.0
            chlorophyll_mg = fresh_weight_mg * self.biomass_params['chlorophyll_density_mg_g'] * normalized_intensity / 1000
            
            # Biomass density
            biomass_density = fresh_weight_mg / area_microns_sq if area_microns_sq > 0 else 0
            
            return {
                'volume_microns_cubed': volume_microns_cubed,
                'volume_mm_cubed': volume_mm_cubed,
                'fresh_weight_mg': fresh_weight_mg,
                'dry_weight_mg': dry_weight_mg,
                'chlorophyll_content_mg': chlorophyll_mg,
                'biomass_density_mg_per_micron_sq': biomass_density
            }
            
        except Exception as e:
            print(f"⚠️ Biomass calculation error: {e}")
            return {
                'volume_microns_cubed': 0,
                'volume_mm_cubed': 0,
                'fresh_weight_mg': 0,
                'dry_weight_mg': 0,
                'chlorophyll_content_mg': 0,
                'biomass_density_mg_per_micron_sq': 0
            }

    def _analyze_cell_color(self, cell, intensity):
        """Analyze color properties of a cell (simplified for grayscale)"""
        try:
            # For grayscale images, estimate green ratio from intensity
            # Higher intensity often correlates with chlorophyll content
            normalized_intensity = intensity / 255.0
            
            # Estimate green ratio based on intensity
            # This is a simplified model - in real RGB analysis this would be more accurate
            green_ratio = min(0.8, max(0.1, normalized_intensity * 0.6 + 0.2))
            
            return {
                'average_rgb': [intensity, intensity, intensity],
                'green_ratio': float(green_ratio),
                'green_intensity': float(intensity),
                'estimated_chlorophyll_level': 'high' if green_ratio > 0.5 else 'medium' if green_ratio > 0.3 else 'low'
            }
            
        except Exception as e:
            print(f"⚠️ Color analysis failed: {e}")
            return {
                'average_rgb': [0, 0, 0],
                'green_ratio': 0,
                'green_intensity': 0,
                'estimated_chlorophyll_level': 'unknown'
            }

    def _assess_cell_health(self, biomass_data, color_data):
        """Assess cell health based on biomass and color data"""
        try:
            health_score = 0.0
            health_factors = []
            
            # Biomass health indicators
            fresh_weight = biomass_data['fresh_weight_mg']
            if 0.01 <= fresh_weight <= 0.5:  # Optimal range for Wolffia
                health_score += 0.4
                health_factors.append("optimal_biomass")
            elif fresh_weight > 0:
                health_score += 0.2
                health_factors.append("detectable_biomass")
            
            # Chlorophyll content
            chlorophyll = biomass_data['chlorophyll_content_mg']
            if chlorophyll > 0.001:
                health_score += 0.3
                health_factors.append("good_chlorophyll")
            
            # Color indicators (green ratio)
            green_ratio = color_data['green_ratio']
            if green_ratio > 0.5:  # High green content
                health_score += 0.3
                health_factors.append("healthy_green_color")
            elif green_ratio > 0.3:
                health_score += 0.15
                health_factors.append("moderate_green_color")
            
            # Determine health status
            if health_score >= 0.8:
                status = "excellent"
            elif health_score >= 0.6:
                status = "healthy"
            elif health_score >= 0.4:
                status = "moderate"
            elif health_score >= 0.2:
                status = "poor"
            else:
                status = "very_poor"
            
            return {
                'status': status,
                'score': health_score,
                'factors': health_factors
            }
            
        except Exception as e:
            print(f"⚠️ Health assessment failed: {e}")
            return {
                'status': 'unknown',
                'score': 0.0,
                'factors': []
            }
        
    def create_essential_visualization(self, original, cells, metrics):
        """Create single essential visualization: intensity-based colored cell borders with dynamic line width"""
        try:
            # Ensure we have a valid image
            if original is None:
                print("❌ No original image provided for visualization")
                return self._create_fallback_visualization(cells, metrics)

            # Handle different image formats
            if len(original.shape) == 2:
                # Grayscale - convert to RGB
                original_display = np.stack([original, original, original], axis=2)
            else:
                original_display = original.copy()

            fig, ax = plt.subplots(1, 1, figsize=(12, 8))

            # Show original image
            try:
                ax.imshow(original_display)
            except Exception as img_error:
                print(f"⚠️ Image display error: {img_error}")
                ax.imshow(np.zeros((100, 100, 3)))

            ax.set_title(f'Cell Detection Results: {len(cells)} cells detected', fontsize=14, fontweight='bold')

            if len(cells) > 0:
                # Extract intensity values for normalization
                intensities = [
                    float(cell['intensity'])
                    for cell in cells
                    if cell.get('intensity') is not None and not np.isnan(cell['intensity'])
                ]
                if intensities:
                    min_intensity = min(intensities)
                    max_intensity = max(intensities)
                    intensity_range = max_intensity - min_intensity
                else:
                    min_intensity = 0
                    max_intensity = 255
                    intensity_range = 255

                valid_cells_drawn = 0
                for i, cell in enumerate(cells):
                    try:
                        contour = cell.get('contour')
                        if contour is None or len(contour) == 0:
                            print(f"⚠️ Cell {i} has empty contour")
                            continue

                        # Convert to numpy array if needed
                        if isinstance(contour, list):
                            contour = np.array(contour, dtype=np.float32)

                        # Handle typical OpenCV format (n_points, 1, 2)
                        if len(contour.shape) == 3 and contour.shape[1] == 1 and contour.shape[2] == 2:
                            points = contour[:, 0, :]
                        elif len(contour.shape) == 2 and contour.shape[1] == 2:
                            points = contour
                        else:
                            print(f"⚠️ Unexpected contour shape for cell {i}: {contour.shape}")
                            continue

                        # Skip invalid shapes
                        if len(points) < 3:
                            print(f"⚠️ Cell {i} contour has too few points: {len(points)}")
                            continue

                        # Check if points contain invalid entries
                        if np.any(np.isnan(points)) or np.any(np.isinf(points)):
                            print(f"⚠️ Cell {i} contour has NaN or Inf values")
                            continue

                        # Compute color and line width
                        cell_intensity = cell.get('intensity', 128)
                        normalized_intensity = (cell_intensity - min_intensity) / (intensity_range or 1)
                        normalized_intensity = max(0, min(1, normalized_intensity))
                        line_width = 1 + (5 - 1) * normalized_intensity

                        if normalized_intensity < 0.33:
                            color = (0, normalized_intensity * 3, 1)
                        elif normalized_intensity < 0.66:
                            t = (normalized_intensity - 0.33) * 3
                            color = (t, 1, 1 - t)
                        else:
                            t = (normalized_intensity - 0.66) * 3
                            color = (1, 1 - t, 0)

                        # Draw polygon
                        polygon = plt.Polygon(points, fill=False, edgecolor=color,
                                            linewidth=line_width, alpha=0.9)
                        ax.add_patch(polygon)
                        valid_cells_drawn += 1

                    except Exception as e:
                        print(f"⚠️ Polygon drawing error for cell {i}: {e}")
                        continue


                print(f"📊 Drew {valid_cells_drawn}/{len(cells)} cell boundaries")

            # Add statistics text
            stats_text = f"""
    ANALYSIS RESULTS:
    • Cells Detected: {metrics['cell_count']}
    • Total Cell Area: {metrics['total_area']:.0f} pixels²
    • Average Cell Area: {metrics['avg_area']:.1f} pixels²
    • Size Range: {metrics.get('min_area', 0):.0f} - {metrics.get('max_area', 0):.0f} pixels²
            """.strip()
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8))

            ax.axis('off')
            plt.tight_layout()

            # Convert to base64 for web display
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight', facecolor='white')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()

            return image_base64

        except Exception as e:
            print(f"❌ Visualization failed: {e}")
            import traceback
            traceback.print_exc()
            return self._create_fallback_visualization(cells, metrics)

    
    def _create_fallback_visualization(self, cells, metrics):
        """Create a simple fallback visualization when main visualization fails"""
        try:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            
            # Create a simple text-based visualization
            ax.text(0.5, 0.7, f'Analysis Complete', transform=ax.transAxes, 
                   fontsize=20, ha='center', fontweight='bold')
            
            ax.text(0.5, 0.5, f'Cells Detected: {len(cells)}', transform=ax.transAxes, 
                   fontsize=16, ha='center')
            
            if metrics:
                ax.text(0.5, 0.4, f'Total Area: {metrics.get("total_area", 0):.0f} pixels²', 
                       transform=ax.transAxes, fontsize=14, ha='center')
                ax.text(0.5, 0.3, f'Average Area: {metrics.get("avg_area", 0):.1f} pixels²', 
                       transform=ax.transAxes, fontsize=14, ha='center')
            
            ax.text(0.5, 0.1, 'Note: Image visualization failed, but analysis completed successfully', 
                   transform=ax.transAxes, fontsize=10, ha='center', style='italic', color='gray')
            
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            
            # Convert to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight', facecolor='white')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return image_base64
            
        except Exception as e:
            print(f"❌ Fallback visualization also failed: {e}")
            return None
    
    def create_error_result(self, error_message):
        """Create error result structure"""
        return {
            'success': False,
            'error': error_message,
            'timestamp': datetime.now().isoformat(),
            'cells_detected': 0,
            'total_cell_area': 0,
            'visualization': None
        }
    
    # TOPHAT TRAINING METHODS
    
    def start_tophat_training(self, image_paths):
        """Start tophat training session with multiple images"""
        print("🎯 Starting Tophat Training Session...")
        
        training_session = {
            'id': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'images': [],
            'annotations': [],
            'created': datetime.now().isoformat()
        }
        
        # Process each image for training
        for image_path in image_paths:
            print(f"📸 Processing training image: {Path(image_path).name}")
            
            # Load and analyze image
            result = self.analyze_image(image_path, use_tophat=False)
            
            if result['success']:
                training_image = {
                    'path': str(image_path),
                    'filename': Path(image_path).name,
                    'auto_detected_cells': result['cells_data'],
                    'cells_count': result['cells_detected'],
                    'image_data': result
                }
                training_session['images'].append(training_image)
        
        # Save training session
        session_path = self.dirs['tophat_training'] / f"session_{training_session['id']}.json"
        with open(session_path, 'w') as f:
            json.dump(training_session, f, indent=2, default=str)
        
        print(f"✅ Training session created: {training_session['id']}")
        return training_session
    
    def save_user_annotations(self, session_id, image_filename, correct_cells, incorrect_cells):
        """Save user annotations for training (legacy method)"""
        print(f"💾 Saving annotations for {image_filename}...")
        
        annotation = {
            'session_id': session_id,
            'image_filename': image_filename,
            'correct_cells': correct_cells,
            'incorrect_cells': incorrect_cells,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save annotation
        annotation_path = self.dirs['annotations'] / f"{session_id}_{image_filename}_annotation.json"
        with open(annotation_path, 'w') as f:
            json.dump(annotation, f, indent=2)
        
        print("✅ Annotations saved")
        return annotation
    
    def save_drawing_annotations(self, session_id, image_filename, image_index, annotations, annotated_image):
        """Save user drawing annotations for training"""
        print(f"🎨 Saving drawing annotations for {image_filename}...")
        
        # Create annotation data structure
        annotation = {
            'session_id': session_id,
            'image_filename': image_filename,
            'image_index': image_index,
            'annotations': annotations,  # Contains correct, false_positive, missed lists
            'timestamp': datetime.now().isoformat(),
            'annotation_type': 'drawing'
        }
        
        # Save annotation JSON
        annotation_path = self.dirs['annotations'] / f"{session_id}_{image_index}_{image_filename}_drawing.json"
        with open(annotation_path, 'w') as f:
            json.dump(annotation, f, indent=2)
        
        # Save annotated image if provided
        if annotated_image:
            try:
                # Remove data URL prefix if present
                if annotated_image.startswith('data:image/png;base64,'):
                    annotated_image = annotated_image[22:]  # Remove prefix
                
                # Decode and save image
                import base64
                image_data = base64.b64decode(annotated_image)
                image_path = self.dirs['annotations'] / f"{session_id}_{image_index}_{image_filename}_annotated.png"
                with open(image_path, 'wb') as f:
                    f.write(image_data)
                
                annotation['annotated_image_path'] = str(image_path)
                print("✅ Annotated image saved")
                
            except Exception as e:
                print(f"⚠️ Failed to save annotated image: {e}")
        
        print("✅ Drawing annotations saved")
        return annotation
    
    def train_tophat_model(self, session_id):
        """Train the tophat AI model using user annotations"""
        print(f"🧠 Training Tophat AI model from session {session_id}...")
        
        try:
            # Load training data
            training_data = self.collect_training_data(session_id)
            
            if len(training_data['features']) < 10:
                print("❌ Not enough training data (minimum 10 samples required)")
                return False
            
            # Prepare features and labels
            X = np.array(training_data['features'])
            y = np.array(training_data['labels'])
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train model
            self.tophat_model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.tophat_model.fit(X_train, y_train)
            
            # Evaluate
            accuracy = self.tophat_model.score(X_test, y_test)
            print(f"✅ Model trained with accuracy: {accuracy:.2f}")
            
            # Save model
            model_path = self.dirs['models'] / 'tophat_model.pkl'
            with open(model_path, 'wb') as f:
                pickle.dump(self.tophat_model, f)
            
            print("💾 Tophat model saved")
            return True
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            return False
    
    def collect_training_data(self, session_id):
        """Collect and prepare training data from annotations"""
        features = []
        labels = []
        
        # Load all annotations for this session (both legacy and drawing)
        annotation_files = list(self.dirs['annotations'].glob(f"{session_id}_*_annotation.json"))
        drawing_files = list(self.dirs['annotations'].glob(f"{session_id}_*_drawing.json"))
        
        # Handle legacy annotation format
        for annotation_file in annotation_files:
            with open(annotation_file, 'r') as f:
                annotation = json.load(f)
            
            # Extract features from correct and incorrect cells
            for cell in annotation.get('correct_cells', []):
                cell_features = self.extract_cell_features_for_training(cell)
                features.append(cell_features)
                labels.append(1)  # Correct cell
            
            for cell in annotation.get('incorrect_cells', []):
                cell_features = self.extract_cell_features_for_training(cell)
                features.append(cell_features)
                labels.append(0)  # Incorrect cell
        
        # Handle drawing annotation format
        for drawing_file in drawing_files:
            with open(drawing_file, 'r') as f:
                annotation = json.load(f)
            
            # Process drawing annotations
            drawing_annotations = annotation.get('annotations', {})
            
            # For drawing annotations, we'll create synthetic features
            # based on the drawing regions
            for annotation_type in ['correct', 'false_positive', 'missed']:
                regions = drawing_annotations.get(annotation_type, [])
                for region in regions:
                    # Create features for the drawn region
                    region_features = self.extract_drawing_features(annotation, region, annotation_type)
                    features.append(region_features)
                    
                    # Label: 1 for correct/missed, 0 for false_positive
                    if annotation_type in ['correct', 'missed']:
                        labels.append(1)
                    else:
                        labels.append(0)
        
        print(f"📊 Collected {len(features)} training samples from {len(annotation_files) + len(drawing_files)} annotation files")
        return {'features': features, 'labels': labels}
    
    def extract_drawing_features(self, annotation, region, annotation_type):
        """Extract features from drawing annotation regions"""
        # For now, create basic features based on the annotation
        # In a real implementation, you'd analyze the drawn region
        base_features = [
            len(region) if isinstance(region, list) else 100,  # Region complexity
            1 if annotation_type == 'correct' else 0,          # Is correct
            1 if annotation_type == 'false_positive' else 0,   # Is false positive
            1 if annotation_type == 'missed' else 0,           # Is missed
            annotation.get('image_index', 0),                  # Image index
        ]
        
        # Pad to match standard feature length
        while len(base_features) < 10:
            base_features.append(0)
        
        return base_features[:10]  # Limit to 10 features
    
    def extract_cell_features_for_training(self, cell):
        """Extract features from a cell for training"""
        return [
            cell.get('area', 0),
            cell.get('center', [0, 0])[0],  # x coordinate
            cell.get('center', [0, 0])[1],  # y coordinate
        ]
    
    def extract_detection_features(self, image):
        """Extract features for detection prediction"""
        # Placeholder for feature extraction
        return []

# Legacy function for backward compatibility
def analyze_uploaded_image(image_path, pixel_to_micron_ratio=0.5, **kwargs):
    """Legacy function for backward compatibility"""
    analyzer = WolffiaAnalyzer(pixel_to_micron_ratio=pixel_to_micron_ratio)
    return analyzer.analyze_image(image_path, **kwargs)

def analyze_multiple_images(image_paths, **kwargs):
    """Analyze multiple images"""
    analyzer = WolffiaAnalyzer()
    results = []
    for image_path in image_paths:
        result = analyzer.analyze_image(image_path, **kwargs)
        results.append(result)
    return results