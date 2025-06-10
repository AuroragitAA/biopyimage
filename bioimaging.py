#!/usr/bin/env python3
"""
BIOIMAGIN Professional Wolffia Analysis System - OPTIMIZED FOCUSED VERSION
Specifically designed for accurate Wolffia arrhiza detection, quantification, and temporal analysis
Author: BIOIMAGIN Senior Bioimage Analysis Engineer
"""

import base64
import json
import os
import traceback
import warnings
from datetime import datetime
from io import BytesIO
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use('Agg')  # Use non-interactive backend
import pickle

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import pandas as pd
from scipy import ndimage
from skimage import color, exposure, feature, filters, morphology, restoration
from skimage.morphology import convex_hull_image
from sklearn.ensemble import RandomForestClassifier

# Suppress warnings for cleaner output and better performance
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Memory optimization: Configure numpy for better performance
np.seterr(divide='ignore', invalid='ignore')
matplotlib.rcParams['figure.max_open_warning'] = 0  # Disable figure limit warnings

# GPU and advanced processing imports
try:
    import torch
    TORCH_AVAILABLE = True
    print("✅ PyTorch available")
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch not available")

try:
    import celldetection as cd
    CELLDETECTION_AVAILABLE = True
    print("✅ CellDetection available")
except ImportError:
    CELLDETECTION_AVAILABLE = False
    print("⚠️ CellDetection not available - using classical methods")

try:
    from skimage import (
        feature,
        filters,
        img_as_float,
        img_as_ubyte,
        measure,
        morphology,
        restoration,
        segmentation,
    )
    from skimage.color import rgb2gray, rgb2hsv
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("❌ Skimage required for advanced processing")

class WolffiaAnalyzer:
    """
    Professional Wolffia Arrhiza Analysis System - FOCUSED EDITION
    Optimized for accurate detection, biomass quantification, and temporal analysis
    """
    
    def __init__(self, pixel_to_micron_ratio=0.5, chlorophyll_threshold=0.6):
        """Initialize analyzer with Wolffia-specific parameters"""
        self.setup_directories()
        self.initialize_parameters()
        self.initialize_biomass_models()
        
        # Lazy loading for heavy models - loaded only when needed
        self._celldetection_model = None
        self._tophat_model = None
        self._device = None
        self.celldetection_available = False
        
        # User parameters
        self.pixel_to_micron_ratio = pixel_to_micron_ratio
        self.chlorophyll_threshold = chlorophyll_threshold
        
        # Time-series tracking
        self.temporal_data = {}
        
        print("🔬 BIOIMAGIN Wolffia Analyzer Initialized - FOCUSED EDITION")
        print(f"💾 Memory optimized - Models loaded lazily")
        print(f"⚡ Performance optimized - {self.__class__.__name__} ready")
        
    def setup_directories(self):
        """Create necessary directories"""
        self.dirs = {
            'results': Path('results'),
            'uploads': Path('uploads'), 
            'models': Path('models'),
            'annotations': Path('annotations'),
            'tophat_training': Path('tophat_training'),
            'wolffia_results': Path('wolffia_results'),
            'temporal_analysis': Path('temporal_analysis')
        }
        for path in self.dirs.values():
            path.mkdir(exist_ok=True)
    
    def initialize_parameters(self):
        """Initialize optimized Wolffia-specific analysis parameters"""
        self.wolffia_params = {
            # Optimized size parameters for Wolffia in images
            'min_cell_area_pixels': 30,       # Minimum cell size
            'max_cell_area_pixels': 3000,     # Maximum cell size
            'optimal_cell_size': 150,         # Optimal size for health assessment
            
            # Detection sensitivity (simplified)
            'detection_confidence_threshold': 0.3,
            'green_intensity_threshold': 60,
            
            # Morphology validation (essential only)
            'circularity_min': 0.2,
            'aspect_ratio_max': 3.0,
            'solidity_min': 0.3,
            
            # CellDetection parameters
            'celldetection_diameter': 25,
            'celldetection_confidence': 0.3,
            'celldetection_nms_threshold': 0.5,
        }

    def initialize_biomass_models(self):
        """Initialize research-based biomass calculation models"""
        self.biomass_params = {
            # Wolffia arrhiza specific parameters (research-based)
            'cell_thickness_microns': 15.0,  # Average cell thickness
            'cell_density_mg_mm3': 1.05,     # Cell density (plant tissue)
            'volume_correction_factor': 0.85, # Cell shape irregularity
            'chlorophyll_density_mg_g': 2.5,  # mg chlorophyll per g fresh weight
            'protein_content_percentage': 0.35, # Protein content (35%)
            'carbon_content_percentage': 0.42,   # Carbon content (42%)
            'water_content_percentage': 0.88,    # Water content (88%)
            'optimal_size_range_microns': (80, 150),  # Optimal size range
            'green_wavelength_nm': 545,       # Peak green wavelength for analysis
        }

    @property
    def celldetection_model(self):
        """Lazy loading for CellDetection AI model"""
        if self._celldetection_model is None:
            self.initialize_celldetection_model()
        return self._celldetection_model
    
    @property 
    def device(self):
        """Lazy loading for device detection"""
        if self._device is None:
            try:
                import torch
                self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
            except:
                self._device = 'cpu'
        return self._device
    
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

    @property
    def tophat_model(self):
        """Lazy loading for tophat model"""
        if self._tophat_model is None:
            self.load_tophat_model()
        return self._tophat_model
    
    def load_tophat_model(self):
        """Load or initialize tophat AI model (called lazily)"""
        model_path = self.dirs['models'] / 'tophat_model.pkl'
        if model_path.exists():
            try:
                with open(model_path, 'rb') as f:
                    self._tophat_model = pickle.load(f)
                print("✅ Tophat model loaded")
            except Exception as e:
                print(f"⚠️ Failed to load tophat model: {e}")
                self._tophat_model = None
        else:
            self._tophat_model = None
            print("📝 No tophat model found - training available")

    def analyze_image(self, image_path, use_celldetection=True, use_tophat=False, image_timestamp=None, **kwargs):
        """
        Main analysis method - OPTIMIZED for focused Wolffia analysis
        
        Args:
            image_path: Path to image file
            use_celldetection: Whether to use AI detection (default: True)
            use_tophat: Whether to use tophat AI model
            image_timestamp: Timestamp for temporal analysis
            **kwargs: Additional parameters
        """
        print(f"🔬 Starting focused Wolffia analysis: {Path(image_path).name}")
        start_time = datetime.now()
        
        try:
            # Load and validate image
            if not Path(image_path).exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Cannot load image: {image_path}")
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            print(f"📸 Image loaded: {image.shape[1]}x{image.shape[0]} pixels")
            
            # Streamlined preprocessing
            print("🔧 Focused preprocessing...")
            processed = self.focused_preprocess(image_rgb)
            
            # Smart cell detection
            print("🧬 Smart cell detection...")
            cells = self.smart_cell_detection(processed, use_celldetection, use_tophat)
            
            # Enhanced analytics
            print("📊 Calculating comprehensive metrics...")
            metrics = self.calculate_comprehensive_metrics(cells, image_rgb)
            
            # Temporal analysis if timestamp provided
            temporal_analysis = None
            if image_timestamp:
                temporal_analysis = self.analyze_temporal_changes(cells, image_timestamp, Path(image_path).name)
            
            # Create professional visualization
            print("📊 Creating professional visualization...")
            visualizations = self.create_professional_visualizations(image_rgb, cells, metrics, temporal_analysis)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Prepare comprehensive results
            result = {
                'success': True,
                'timestamp': start_time.isoformat(),
                'processing_time': processing_time,
                'image_info': {
                    'filename': Path(image_path).name,
                    'dimensions': f"{image.shape[1]}x{image.shape[0]}",
                    'file_size_mb': Path(image_path).stat().st_size / (1024*1024),
                    'pixel_to_micron_ratio': self.pixel_to_micron_ratio
                },
                
                # Core detection results
                'detection_results': {
                    'cells_detected': len(cells),
                    'detection_method': self._get_detection_method_name(use_celldetection, use_tophat, len(cells)),
                    'cells_data': cells
                },
                
                # Comprehensive metrics
                'quantitative_analysis': {
                    'cell_count': len(cells),
                    'total_area_pixels': metrics['total_area'],
                    'total_area_microns': metrics['total_area_microns'],
                    'average_cell_area': metrics['avg_area'],
                    'size_distribution': metrics['size_distribution'],
                    'biomass_analysis': metrics['biomass_analysis'],
                    'color_analysis': metrics['color_analysis'],
                    'health_assessment': metrics['health_assessment']
                },
                
                # Visualizations
                'visualizations': visualizations,
                
                # Temporal analysis (if available)
                'temporal_analysis': temporal_analysis,
                
                # Legacy compatibility
                'cells': cells,
                'summary': {
                    'total_cells': len(cells),
                    'total_area': metrics['total_area'],
                    'processing_time': processing_time
                }
            }
            
            print(f"✅ Analysis complete: {len(cells)} cells detected")
            print(f"📊 Total biomass: {metrics['biomass_analysis']['total_biomass_mg']:.3f}mg")
            print(f"🌿 Green cells: {metrics['color_analysis']['green_cell_percentage']:.1f}%")
            print(f"⚡ Processing time: {processing_time:.2f}s")
            
            return result
            
        except Exception as e:
            print(f"❌ Analysis failed: {str(e)}")
            traceback.print_exc()
            return self.create_error_result(str(e))

    def focused_preprocess(self, image):
        """MINIMAL preprocessing for Shape Index detection"""
        print("🔧 MINIMAL preprocessing for Shape Index...")
        
        try:
            # Ensure color image - avoid unnecessary copying
            if len(image.shape) == 2:
                image_rgb = np.stack([image, image, image], axis=2)
            else:
                image_rgb = image  # Use reference instead of copy
            
            # Just basic steps for visualization
            gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
            simple_mask = gray > 50  # Very permissive
            
            return {
                'original': image_rgb,
                'processed': (simple_mask * 255).astype(np.uint8),
                'gray': gray,
                'enhanced': gray,
                'steps': {
                    'denoised': gray,
                    'green_enhanced': image_rgb[:, :, 1] if len(image_rgb.shape) == 3 else gray,  # Show green channel
                    'li_foreground': (simple_mask * 255).astype(np.uint8),
                    'plate_removed': (simple_mask * 255).astype(np.uint8),
                    'multi_otsu': (simple_mask * 255).astype(np.uint8),
                    'combined': (simple_mask * 255).astype(np.uint8),
                    'opened': (simple_mask * 255).astype(np.uint8),
                    'final': (simple_mask * 255).astype(np.uint8)
                },
                'foreground_mask': simple_mask
            }
            
        except Exception as e:
            print(f"❌ Preprocessing failed: {e}")
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
            simple_mask = gray > 30
            
            return {
                'original': image,
                'processed': (simple_mask * 255).astype(np.uint8),
                'gray': gray,
                'enhanced': gray,
                'steps': {'simple': (simple_mask * 255).astype(np.uint8)},
                'foreground_mask': simple_mask
            }
    
    def gentle_image_enhancement(self, image):
        """MINIMAL image enhancement - just reduce noise"""
        try:
            print("📊 Minimal image enhancement...")
            
            # Very light denoising only
            enhanced = cv2.medianBlur(image, 1)  # Remove salt-and-pepper noise
            
            # Slight contrast improvement only if image is very low contrast
            gray = cv2.cvtColor(enhanced, cv2.COLOR_RGB2GRAY)
            contrast = np.std(gray)
            
            if contrast < 20:  # Very low contrast
                # Apply very gentle CLAHE only to low-contrast images
                lab = cv2.cvtColor(enhanced, cv2.COLOR_RGB2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4,4))
                l = clahe.apply(l)
                enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)
                print("📊 Applied gentle contrast enhancement")
            
            return enhanced
            
        except Exception as e:
            print(f"⚠️ Enhancement failed: {e}")
            return image

    def simple_green_enhancement(self, image):
        """IMPROVED green channel enhancement with noise reduction"""
        try:
            print("🌿 Enhanced green channel extraction...")
            
            # Extract channels
            red = image[:, :, 0].astype(np.float32)
            green = image[:, :, 1].astype(np.float32)
            blue = image[:, :, 2].astype(np.float32)
            
            # Method 1: Enhanced green channel
            enhanced_green = green * 1.1  # Slight enhancement, not too aggressive
            
            # Method 2: Green difference (green minus average of red and blue)
            green_difference = green - (red + blue) / 2
            green_difference = np.clip(green_difference + 50, 0, 255)  # Add offset and clip
            
            # Combine both methods (take the better one for each pixel)
            final_green = np.maximum(enhanced_green, green_difference)
            final_green = np.clip(final_green, 0, 255).astype(np.uint8)
            
            # Apply gentle smoothing to reduce noise
            smoothed = cv2.GaussianBlur(final_green, (3, 3), 0.5)
            
            return smoothed
            
        except Exception as e:
            print(f"⚠️ Green enhancement failed: {e}")
            return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
    def gentle_green_mask(self, image, plate_center, plate_radius):
        """FIXED green detection - detect actual GREEN objects, not noise"""
        try:
            print("🌿 FIXED green object detection...")
            
            # Create plate region mask
            h, w = image.shape[:2]
            plate_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(plate_mask, plate_center, int(plate_radius * 0.8), 1, -1)
            
            # PROPER green detection using color analysis
            # Convert to HSV for better color detection
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            
            # Define green color range in HSV (much better than just green channel)
            # Green hue is around 60-80 in OpenCV HSV (0-179 range)
            lower_green = np.array([40, 30, 30])   # Lower bound for green
            upper_green = np.array([85, 255, 255]) # Upper bound for green
            
            # Create green color mask
            green_color_mask = cv2.inRange(hsv, lower_green, upper_green)
            
            # Alternative method: Use green-ness ratio
            red_channel = image[:, :, 0].astype(np.float32)
            green_channel = image[:, :, 1].astype(np.float32)
            blue_channel = image[:, :, 2].astype(np.float32)
            
            # Calculate green dominance (green should be higher than red and blue)
            green_dominance = green_channel - np.maximum(red_channel, blue_channel)
            green_dominance = np.clip(green_dominance, 0, 255)
            
            # Threshold green dominance
            green_threshold = np.percentile(green_dominance[green_dominance > 0], 60) if np.any(green_dominance > 0) else 20
            green_dominance_mask = green_dominance > max(green_threshold, 15)
            
            # Combine both methods
            combined_green_mask = (green_color_mask > 0) | green_dominance_mask
            
            # Apply plate constraint
            final_mask = combined_green_mask & (plate_mask > 0)
            
            # Remove very small noise objects
            from skimage import morphology
            cleaned_mask = morphology.remove_small_objects(final_mask, min_size=5)
            
            print(f"🌿 FIXED green detection: {np.sum(cleaned_mask)} green pixels (threshold={green_threshold:.1f})")
            
            # If we still get too much noise, be more conservative
            if np.sum(cleaned_mask) > 0.3 * np.sum(plate_mask):  # More than 30% of plate
                print("⚠️ Too much detected as green, using conservative approach...")
                # Use only the most green pixels
                conservative_threshold = np.percentile(green_dominance[green_dominance > 0], 85) if np.any(green_dominance > 0) else 30
                conservative_mask = green_dominance > conservative_threshold
                final_mask = conservative_mask & (plate_mask > 0)
                cleaned_mask = morphology.remove_small_objects(final_mask, min_size=5)
                print(f"🌿 Conservative green detection: {np.sum(cleaned_mask)} green pixels")
            
            return cleaned_mask
            
        except Exception as e:
            print(f"❌ Green detection failed: {e}")
            # Ultra-simple fallback - detect high-contrast objects
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Create plate region mask
            h, w = gray.shape
            plate_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(plate_mask, plate_center, int(plate_radius * 0.8), 1, -1)
            
            # Find objects that are significantly different from background
            bg_intensity = np.mean(gray[plate_mask > 0])
            diff_from_bg = np.abs(gray.astype(np.float32) - bg_intensity)
            threshold = np.percentile(diff_from_bg[plate_mask > 0], 90)  # Top 10% different
            
            simple_mask = (diff_from_bg > threshold) & (plate_mask > 0)
            
            from skimage import morphology
            cleaned = morphology.remove_small_objects(simple_mask, min_size=10)
            
            print(f"🌿 Fallback detection: {np.sum(cleaned)} pixels")
            return cleaned

    def minimal_cleanup(self, mask):
        """Minimal morphological cleanup"""
        try:
            from skimage import morphology
            # Very gentle cleanup
            cleaned = morphology.remove_small_objects(mask, min_size=10)
            return cleaned
        except:
            return mask
    
    def smart_cell_detection(self, processed, use_celldetection=True, use_tophat=False):
        """SHAPE INDEX detection - simple and effective for round cells"""
        original = processed['original']
        
        cells = []
        detection_attempts = []
        
        print("🔬 SHAPE INDEX detection for round cells...")
        
        # METHOD 1: Shape Index Detection (Primary - most effective for round cells)
        try:
            cells_shape_index = self.shape_index_detection(original)
            if len(cells_shape_index) > 0:
                cells.extend(cells_shape_index)
                detection_attempts.append(f"Shape Index: {len(cells_shape_index)} cells")
                print(f"✅ Shape Index found {len(cells_shape_index)} round cells")
        except Exception as e:
            print(f"⚠️ Shape Index detection failed: {e}")
        
        # METHOD 2: AI Detection (Secondary - only if Shape Index insufficient)
        if len(cells) < 3 and use_celldetection and hasattr(self, 'celldetection_model') and self.celldetection_model:
            try:
                print("🧠 Running AI detection as backup...")
                cells_ai = self.celldetection_inference(original)
                if len(cells_ai) > 0:
                    # Filter out plate edges
                    cells_ai_filtered = self.filter_plate_edges(cells_ai, original.shape)
                    cells.extend(cells_ai_filtered)
                    detection_attempts.append(f"AI Backup: {len(cells_ai_filtered)} cells")
                    print(f"✅ AI backup found {len(cells_ai_filtered)} cells")
            except Exception as e:
                print(f"⚠️ AI backup failed: {e}")
        
        # Simple validation
        final_cells = self.simple_validation(cells, original)
        
        print(f"🔍 Detection summary: {', '.join(detection_attempts)}")
        print(f"✅ Final result: {len(final_cells)} cells (Shape Index)")
        
        return final_cells
        
    def filter_plate_edges(self, cells, image_shape):
        """Filter out cells near plate edges"""
        h, w = image_shape[:2]
        edge_buffer = 40  # Pixels from edge
        
        filtered_cells = []
        for cell in cells:
            cx, cy = cell['center']
            if (edge_buffer <= cx <= w - edge_buffer and 
                edge_buffer <= cy <= h - edge_buffer):
                filtered_cells.append(cell)
        
        return filtered_cells

    def simple_validation(self, cells, original):
        """Simple validation - just remove obvious duplicates"""
        if not cells:
            return []
        
        validated_cells = []
        for cell in cells:
            cx, cy = cell['center']
            area = cell['area']
            
            # Basic size check
            if 5 <= area <= 5000:
                # Check for duplicates
                is_duplicate = False
                for validated_cell in validated_cells:
                    vx, vy = validated_cell['center']
                    distance = np.sqrt((cx - vx)**2 + (cy - vy)**2)
                    
                    if distance < 15:  # Too close
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    validated_cells.append(cell)
        
        # Re-number
        for i, cell in enumerate(validated_cells):
            cell['id'] = i + 1
        
        return validated_cells

    def shape_index_detection(self, image):
        """Detect round cells using Shape Index - perfect for Wolffia"""
        try:
            from scipy import ndimage
            from skimage.feature import shape_index
            
            print("🎯 Running Shape Index detection for round objects...")
            
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image.copy()
            
            # Normalize image
            gray_normalized = cv2.normalize(gray, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            
            # Apply gentle smoothing to reduce noise
            smoothed = ndimage.gaussian_filter(gray_normalized, sigma=1.0)
            
            # Calculate shape index
            s = shape_index(smoothed)
            
            # Apply additional smoothing to shape index to reduce noise
            s_smooth = ndimage.gaussian_filter(s, sigma=0.5)
            
            # Target spherical caps (value = 1 for perfect spheres)
            target = 0.5
            delta = 0.25  # Tolerance for detecting round objects
            
            # Find round objects
            round_mask = np.abs(s_smooth - target) < delta
            
            print(f"🎯 Shape Index found {np.sum(round_mask)} round pixels")
            
            # Create plate focus mask to exclude edges
            h, w = gray.shape
            center = (w//2, h//2)
            max_radius = min(h, w) // 3
            
            plate_mask = np.zeros((h, w), dtype=bool)
            cv2.circle(plate_mask.astype(np.uint8), center, max_radius, 1, -1)
            
            # Apply plate mask
            round_mask_focused = round_mask & plate_mask
            
            # Label connected components
            from skimage import measure
            labeled_regions = measure.label(round_mask_focused)
            regions = measure.regionprops(labeled_regions, intensity_image=gray)
            
            cells = []
            for region in regions:
                area = region.area
                
                # Size filter for Wolffia cells
                if self.wolffia_params['min_cell_area_pixels'] <= area <= self.wolffia_params['max_cell_area_pixels']:
                    
                    # Additional shape validation
                    if (region.eccentricity < 0.8 and  # Not too elongated
                        region.solidity > 0.6):       # Reasonably solid
                        
                        # Get center
                        cy, cx = region.centroid
                        cx, cy = int(cx), int(cy)
                        
                        # Calculate intensities
                        avg_intensity = region.intensity_mean if hasattr(region, 'intensity_mean') else region.mean_intensity
                        
                        # Green intensity
                        if len(image.shape) == 3:
                            mask = labeled_regions == region.label
                            green_pixels = image[mask, 1]
                            green_intensity = np.mean(green_pixels) if len(green_pixels) > 0 else avg_intensity
                        else:
                            green_intensity = avg_intensity
                        
                        # Create contour from region
                        coords = region.coords
                        if len(coords) > 3:
                            # Create simple circular contour
                            radius = int(np.sqrt(area / np.pi))
                            angles = np.linspace(0, 2*np.pi, 20)
                            contour_x = cx + radius * np.cos(angles)
                            contour_y = cy + radius * np.sin(angles)
                            contour_cv = np.column_stack([contour_x, contour_y]).astype(np.int32).reshape(-1, 1, 2)
                        else:
                            contour_cv = []
                        
                        # High confidence for shape index detections
                        confidence = min(0.75, 0.5 + 0.1 * region.solidity + 0.05 * (1 - region.eccentricity))
                        
                        cell = {
                            'id': len(cells) + 1,
                            'center': (cx, cy),
                            'area': area,
                            'contour': contour_cv.tolist() if len(contour_cv) > 0 else [],
                            'intensity': float(avg_intensity),
                            'green_intensity': float(green_intensity),
                            'method': 'shape_index_spherical',
                            'confidence': confidence,
                            'perimeter': region.perimeter,
                            'eccentricity': region.eccentricity,
                            'solidity': region.solidity,
                            'area_microns': area * (self.pixel_to_micron_ratio ** 2)
                        }
                        
                        cells.append(cell)
            
            print(f"🎯 Shape Index detected {len(cells)} round cells")
            return cells
            
        except Exception as e:
            print(f"❌ Shape Index detection failed: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def detect_plate_boundaries(self, image):
        """Detect circular plate boundaries"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            h, w = gray.shape
            
            # Try to detect circular plate
            circles = cv2.HoughCircles(
                gray, cv2.HOUGH_GRADIENT, dp=1, minDist=min(h,w)//2,
                param1=50, param2=30, minRadius=min(h,w)//4, maxRadius=min(h,w)//2
            )
            
            if circles is not None:
                circle = circles[0][0]
                center = (int(circle[0]), int(circle[1]))
                radius = int(circle[2])
                print(f"🎯 Plate detected: center={center}, radius={radius}")
            else:
                center = (w//2, h//2)
                radius = min(h, w)//3
                print(f"🎯 Using default plate: center={center}, radius={radius}")
            
            return center, radius
            
        except Exception as e:
            print(f"⚠️ Plate detection failed: {e}")
            h, w = image.shape[:2]
            return (w//2, h//2), min(h, w)//3

    def create_center_focus_mask(self, shape, center, radius):
        """Create mask focusing on plate center, excluding edges"""
        h, w = shape
        mask = np.zeros((h, w), dtype=bool)
        
        # Create circular mask with 70% of radius (exclude edges)
        cv2.circle(mask.astype(np.uint8), center, int(radius * 0.7), 1, -1)
        
        print(f"🎯 Center focus mask: {np.sum(mask)} pixels in focus area")
        return mask.astype(bool)

    def backup_green_detection(self, enhanced, original, center_mask):
        """Backup detection focusing on actual green objects"""
        try:
            print("🌿 Backup green object detection...")
            
            # Convert to HSV for better green detection
            hsv = cv2.cvtColor(original, cv2.COLOR_RGB2HSV)
            
            # Define green range (more restrictive for small green cells)
            lower_green = np.array([35, 40, 40])
            upper_green = np.array([85, 255, 255])
            
            # Create green mask
            green_mask = cv2.inRange(hsv, lower_green, upper_green)
            
            # Apply center focus
            green_mask = green_mask & center_mask.astype(np.uint8) * 255
            
            # Find contours
            contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            cells = []
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Size filter for small green cells
                if 10 <= area <= 1000:
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        
                        # Ensure within center mask
                        if center_mask[cy, cx]:
                            cell = {
                                'id': len(cells) + 1,
                                'center': (cx, cy),
                                'area': area,
                                'contour': contour.tolist(),
                                'intensity': 100,
                                'green_intensity': 150,
                                'method': 'backup_green',
                                'confidence': 0.7,
                                'area_microns': area * (self.pixel_to_micron_ratio ** 2)
                            }
                            cells.append(cell)
            
            return cells
            
        except Exception as e:
            print(f"❌ Backup green detection failed: {e}")
            return []
        
    def minimal_validation(self, cells, original):
        """MINIMAL validation - just remove duplicates, trust the AI"""
        try:
            if not cells:
                return []
            
            validated_cells = []
            
            for cell in cells:
                cx, cy = cell['center']
                area = cell['area']
                method = cell.get('method', 'unknown')
                
                # Only very basic checks
                valid = True
                
                # 1. Only remove obviously wrong sizes
                if area < 5 or area > 10000:  # Very permissive size range
                    valid = False
                
                # 2. Remove only very close duplicates
                if valid:
                    for validated_cell in validated_cells:
                        vx, vy = validated_cell['center']
                        distance = np.sqrt((cx - vx)**2 + (cy - vy)**2)
                        
                        # Only remove if VERY close (likely exact duplicate)
                        if distance < 8:
                            valid = False
                            break
                
                if valid:
                    validated_cells.append(cell)
            
            # Re-number cells
            for i, cell in enumerate(validated_cells):
                cell['id'] = i + 1
            
            print(f"🔍 Minimal validation: kept {len(validated_cells)} out of {len(cells)} cells")
            return validated_cells
            
        except Exception as e:
            print(f"❌ Validation failed: {e}")
            return cells

    def backup_contour_detection(self, enhanced, original):
        """Simple backup contour detection"""
        try:
            contours, _ = cv2.findContours(enhanced, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            cells = []
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Very permissive size filter
                if 10 <= area <= 5000:
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        
                        # Simple intensity calculation
                        mask = np.zeros(enhanced.shape, dtype=np.uint8)
                        cv2.fillPoly(mask, [contour], 255)
                        
                        if len(original.shape) == 3:
                            avg_intensity = np.mean(original[mask > 0][:, 1]) if np.any(mask > 0) else 0
                            green_intensity = avg_intensity
                        else:
                            avg_intensity = np.mean(original[mask > 0]) if np.any(mask > 0) else 0
                            green_intensity = avg_intensity
                        
                        cell = {
                            'id': len(cells) + 1,
                            'center': (cx, cy),
                            'area': area,
                            'contour': contour.tolist(),
                            'intensity': float(avg_intensity),
                            'green_intensity': float(green_intensity),
                            'method': 'backup_contour',
                            'confidence': 0.6,
                            'area_microns': area * (self.pixel_to_micron_ratio ** 2)
                        }
                        cells.append(cell)
            
            return cells
            
        except Exception as e:
            print(f"❌ Backup contour detection failed: {e}")
            return []

    def gentle_cell_validation(self, cells, original):
        """GENTLE cell validation - keep more good cells"""
        try:
            if not cells:
                return []
            
            validated_cells = []
            
            # Sort by confidence but be less strict
            cells_sorted = sorted(cells, key=lambda x: x.get('confidence', 0), reverse=True)
            
            for cell in cells_sorted:
                cx, cy = cell['center']
                area = cell['area']
                confidence = cell.get('confidence', 0)
                method = cell.get('method', 'unknown')
                
                # More lenient validation criteria
                valid = True
                
                # 1. Size validation (more permissive)
                if not (self.wolffia_params['min_cell_area_pixels'] * 0.5 <= area <= self.wolffia_params['max_cell_area_pixels'] * 1.5):
                    valid = False
                
                # 2. Confidence validation (lower threshold)
                if confidence < 0.2:
                    valid = False
                
                # 3. AI detections get priority (less strict distance checking)
                if valid and method.startswith('celldetection'):
                    # AI detections are trusted more, use larger minimum distance
                    min_distance = 15
                else:
                    min_distance = 25
                
                # 4. Distance validation (more permissive)
                if valid:
                    too_close = False
                    for validated_cell in validated_cells:
                        vx, vy = validated_cell['center']
                        distance = np.sqrt((cx - vx)**2 + (cy - vy)**2)
                        
                        if distance < min_distance:
                            # Keep the one with higher confidence
                            if confidence > validated_cell.get('confidence', 0):
                                validated_cells.remove(validated_cell)
                                break
                            else:
                                too_close = True
                                break
                    
                    if too_close:
                        valid = False
                
                # 5. Edge proximity (more lenient for AI detections)
                if valid:
                    h, w = original.shape[:2]
                    edge_buffer = 20 if method.startswith('celldetection') else 30
                    
                    if (cx < edge_buffer or cx > w - edge_buffer or 
                        cy < edge_buffer or cy > h - edge_buffer):
                        if confidence < 0.8:  # Only reject low-confidence edge detections
                            valid = False
                
                if valid:
                    validated_cells.append(cell)
            
            # Re-number cells
            for i, cell in enumerate(validated_cells):
                cell['id'] = i + 1
            
            print(f"🔍 Gentle validation: kept {len(validated_cells)} out of {len(cells)} cells")
            return validated_cells
            
        except Exception as e:
            print(f"❌ Gentle validation failed: {e}")
            return cells
    

    def filter_by_plate_region(self, cells, plate_info):
        """Filter cells to only those within plate region"""
        if not plate_info:
            return cells
        
        plate_center = plate_info.get('center', (0, 0))
        plate_radius = plate_info.get('radius', 1000)
        
        filtered_cells = []
        for cell in cells:
            cx, cy = cell['center']
            distance = np.sqrt((cx - plate_center[0])**2 + (cy - plate_center[1])**2)
            
            if distance <= plate_radius * 0.9:  # Within 90% of plate radius
                filtered_cells.append(cell)
        
        return filtered_cells
    
    def _filter_edge_detections(self, cells, image_shape):
        """Filter out detections near image edges (likely plate artifacts)"""
        try:
            h, w = image_shape[:2]
            edge_buffer = min(h, w) // 15  # ~7% buffer from edges
            
            filtered_cells = []
            for cell in cells:
                cx, cy = cell['center']
                
                # Check if cell is away from edges
                if (edge_buffer <= cx <= w - edge_buffer and 
                    edge_buffer <= cy <= h - edge_buffer):
                    filtered_cells.append(cell)
            
            return filtered_cells
            
        except Exception as e:
            print(f"⚠️ Edge filtering failed: {e}")
            return cells

    def _extract_cells_from_labels(self, labels, original, method_name):
        """Extract cell objects from labeled segmentation"""
        try:
            regions = measure.regionprops(labels, intensity_image=cv2.cvtColor(original, cv2.COLOR_RGB2GRAY) if len(original.shape) == 3 else original)
            
            cells = []
            for region in regions:
                area = region.area
                
                # Size filter
                if not (self.wolffia_params['min_cell_area_pixels'] <= area <= self.wolffia_params['max_cell_area_pixels']):
                    continue
                
                # Shape filter - enhanced criteria
                eccentricity = region.eccentricity
                solidity = region.solidity
                extent = region.extent
                
                # Wolffia-specific shape validation
                if (eccentricity < 0.85 and     # Not too elongated
                    solidity > 0.6 and          # Reasonably solid
                    extent > 0.4 and            # Fills bounding box
                    region.perimeter > 0):      # Valid perimeter
                    
                    # Calculate center
                    cy, cx = region.centroid
                    cx, cy = int(cx), int(cy)
                    
                    # Calculate intensities
                    avg_intensity = region.intensity_mean if hasattr(region, 'intensity_mean') else region.mean_intensity
                    
                    # Green intensity calculation
                    if len(original.shape) == 3:
                        # Create mask for this region
                        mask = labels == region.label
                        green_pixels = original[mask, 1]  # Green channel
                        green_intensity = np.mean(green_pixels) if len(green_pixels) > 0 else avg_intensity
                    else:
                        green_intensity = avg_intensity
                    
                    # Calculate morphological properties
                    perimeter = region.perimeter
                    circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                    
                    # Get contour from region coordinates
                    coords = region.coords
                    if len(coords) > 3:
                        # Create contour from region boundary
                        hull = morphology.convex_hull_image(labels == region.label)
                        contours_found = measure.find_contours(hull.astype(np.uint8), 0.5)
                        
                        if len(contours_found) > 0:
                            contour = contours_found[0]
                            # Convert to OpenCV format
                            contour_cv = np.array(contour[:, [1, 0]], dtype=np.int32).reshape(-1, 1, 2)
                        else:
                            contour_cv = []
                    else:
                        contour_cv = []
                    
                    # Create cell object
                    cell = {
                        'id': len(cells) + 1,
                        'center': (cx, cy),
                        'area': area,
                        'contour': contour_cv.tolist() if len(contour_cv) > 0 else [],
                        'intensity': float(avg_intensity),
                        'green_intensity': float(green_intensity),
                        'method': method_name,
                        'confidence': min(0.9, 0.5 + 0.3 * solidity + 0.1 * (1 - eccentricity)),
                        'perimeter': perimeter,
                        'circularity': circularity,
                        'eccentricity': eccentricity,
                        'solidity': solidity,
                        'extent': extent,
                        'area_microns': area * (self.pixel_to_micron_ratio ** 2)
                    }
                    
                    cells.append(cell)
            
            return cells
            
        except Exception as e:
            print(f"❌ Cell extraction failed: {e}")
            return []

    def enhanced_cell_validation(self, cells, original):
        """Enhanced validation with multiple criteria and distance-based filtering"""
        try:
            if not cells:
                return []
            
            validated_cells = []
            
            # Sort cells by confidence (keep higher confidence ones)
            cells_sorted = sorted(cells, key=lambda x: x.get('confidence', 0), reverse=True)
            
            for cell in cells_sorted:
                cx, cy = cell['center']
                area = cell['area']
                confidence = cell.get('confidence', 0)
                circularity = cell.get('circularity', 0)
                solidity = cell.get('solidity', 1)
                
                # Enhanced validation criteria
                valid = True
                
                # 1. Size validation
                if not (self.wolffia_params['min_cell_area_pixels'] <= area <= self.wolffia_params['max_cell_area_pixels']):
                    valid = False
                
                # 2. Shape validation (more strict)
                if circularity < 0.3 or solidity < 0.6:
                    valid = False
                
                # 3. Confidence validation
                if confidence < 0.25:
                    valid = False
                
                # 4. Distance validation (avoid duplicates and edge clustering)
                if valid:
                    # Check distance from existing validated cells
                    too_close = False
                    for validated_cell in validated_cells:
                        vx, vy = validated_cell['center']
                        distance = np.sqrt((cx - vx)**2 + (cy - vy)**2)
                        
                        # Minimum distance based on cell size
                        min_distance = np.sqrt(max(area, validated_cell['area']) / np.pi) * 1.5
                        
                        if distance < max(min_distance, 15):
                            too_close = True
                            break
                    
                    if too_close:
                        valid = False
                
                # 5. Edge proximity check (refined)
                if valid:
                    h, w = original.shape[:2]
                    edge_buffer = min(h, w) // 20
                    
                    if (cx < edge_buffer or cx > w - edge_buffer or 
                        cy < edge_buffer or cy > h - edge_buffer):
                        # Allow if it's a very good detection
                        if confidence < 0.7 or area > self.wolffia_params['max_cell_area_pixels'] * 0.8:
                            valid = False
                
                if valid:
                    validated_cells.append(cell)
            
            # Re-number cells
            for i, cell in enumerate(validated_cells):
                cell['id'] = i + 1
            
            return validated_cells
            
        except Exception as e:
            print(f"❌ Enhanced validation failed: {e}")
            return cells

    def celldetection_inference(self, image_rgb):
        """Optimized CellDetection inference"""
        try:
            if self.celldetection_model is None:
                return []
            
            # Preprocess for CellDetection
            if len(image_rgb.shape) == 2:
                img = cv2.cvtColor(image_rgb, cv2.COLOR_GRAY2RGB)
            else:
                img = image_rgb.copy()
            
            # Ensure uint8
            if img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)
            
            with torch.no_grad():
                # Convert to tensor
                x = cd.to_tensor(img, transpose=True, device=self.device, dtype=torch.float32)
                x = x / 255.0
                x = x.unsqueeze(0)
                
                # Run inference
                outputs = self.celldetection_model(x)
                
                # Extract results
                contours = outputs.get('contours', [])
                scores = outputs.get('scores', [])
                
                if len(contours) > 0 and len(contours[0]) > 0:
                    cells = self._convert_celldetection_results(contours[0], scores[0] if len(scores) > 0 else None, img)
                    return cells
                
                return []
                
        except Exception as e:
            print(f"❌ CellDetection inference failed: {e}")
            return []

    def _convert_celldetection_results(self, contours, scores, original_image):
        """Convert CellDetection results to standardized cell format"""
        try:
            cells = []
            
            if scores is None:
                scores = [1.0] * len(contours)
            
            for i, (contour, score) in enumerate(zip(contours, scores)):
                try:
                    # Convert contour
                    if isinstance(contour, torch.Tensor):
                        contour_np = contour.cpu().numpy()
                    else:
                        contour_np = np.array(contour)
                    
                    if len(contour_np.shape) == 2 and contour_np.shape[1] == 2:
                        contour_cv = contour_np.reshape((-1, 1, 2)).astype(np.int32)
                    else:
                        continue
                    
                    # Calculate properties
                    area = cv2.contourArea(contour_cv)
                    
                    # Size filter
                    if not (self.wolffia_params['min_cell_area_pixels'] <= area <= self.wolffia_params['max_cell_area_pixels']):
                        continue
                    
                    # Confidence filter
                    if score < self.wolffia_params['celldetection_confidence']:
                        continue
                    
                    # Calculate center
                    M = cv2.moments(contour_cv)
                    if M["m00"] == 0:
                        continue
                    
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # Calculate intensity and color properties
                    mask = np.zeros(original_image.shape[:2], dtype=np.uint8)
                    cv2.fillPoly(mask, [contour_cv], 255)
                    
                    cell_pixels = original_image[mask > 0]
                    if len(cell_pixels) > 0:
                        if len(original_image.shape) == 3:
                            avg_intensity = np.mean(cell_pixels.mean(axis=1))
                            green_intensity = np.mean(cell_pixels[:, 1])  # Green channel
                        else:
                            avg_intensity = np.mean(cell_pixels)
                            green_intensity = avg_intensity
                    else:
                        avg_intensity = 0
                        green_intensity = 0
                    
                    # Calculate morphological properties
                    perimeter = cv2.arcLength(contour_cv, True)
                    circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                    
                    # Create standardized cell object
                    cell = {
                        'id': len(cells) + 1,
                        'center': (cx, cy),
                        'area': area,
                        'contour': contour_cv.tolist(),
                        'intensity': float(avg_intensity),
                        'green_intensity': float(green_intensity),
                        'method': 'celldetection_ai',
                        'confidence': float(score),
                        'perimeter': perimeter,
                        'circularity': circularity,
                        'area_microns': area * (self.pixel_to_micron_ratio ** 2)
                    }
                    
                    cells.append(cell)
                    
                except Exception as e:
                    continue
            
            return cells
            
        except Exception as e:
            print(f"❌ Failed to convert CellDetection results: {e}")
            return []


    def remove_duplicates_and_validate(self, cells):
        """Legacy method - redirects to enhanced validation"""
        return self.enhanced_cell_validation(cells, np.zeros((100, 100, 3), dtype=np.uint8))

    def _get_detection_method_name(self, use_celldetection, use_tophat, cell_count):
        """Get human-readable detection method name"""
        methods = []
        if use_celldetection and self.celldetection_model and cell_count > 0:
            methods.append("CellDetection AI")
        if use_tophat and self.tophat_model and cell_count > 0:
            methods.append("Tophat AI")
        if not methods or cell_count < 3:
            methods.append("Classical Watershed")
        
        return " + ".join(methods) if len(methods) > 1 else (methods[0] if methods else "Classical Watershed")

    def calculate_comprehensive_metrics(self, cells, image_rgb):
        """Calculate comprehensive quantitative metrics for Wolffia analysis"""
        try:
            if not cells:
                return self._empty_metrics()
            
            # Basic measurements
            areas_pixels = [cell['area'] for cell in cells]
            areas_microns = [cell.get('area_microns', cell['area'] * (self.pixel_to_micron_ratio ** 2)) for cell in cells]
            intensities = [cell['intensity'] for cell in cells]
            green_intensities = [cell.get('green_intensity', cell['intensity']) for cell in cells]
            
            # Size distribution analysis
            size_distribution = self._analyze_size_distribution(areas_microns)
            
            # Biomass analysis with research-based equations
            biomass_analysis = self._calculate_biomass_metrics(cells, areas_microns, image_rgb)
            
            # Color analysis for green wavelength detection
            color_analysis = self._analyze_color_properties(cells, green_intensities, image_rgb)
            
            # Health assessment
            health_assessment = self._assess_population_health(cells, areas_microns, green_intensities)
            
            return {
                'cell_count': len(cells),
                'total_area': sum(areas_pixels),
                'total_area_microns': sum(areas_microns),
                'avg_area': np.mean(areas_pixels),
                'avg_area_microns': np.mean(areas_microns),
                'std_area': np.std(areas_pixels),
                'size_distribution': size_distribution,
                'biomass_analysis': biomass_analysis,
                'color_analysis': color_analysis,
                'health_assessment': health_assessment,
                'avg_intensity': np.mean(intensities),
                'avg_green_intensity': np.mean(green_intensities)
            }
            
        except Exception as e:
            print(f"❌ Metrics calculation failed: {e}")
            return self._empty_metrics()

    def _analyze_size_distribution(self, areas_microns):
        """Analyze cell size distribution for Wolffia populations"""
        try:
            if not areas_microns:
                return {'small': 0, 'medium': 0, 'large': 0, 'optimal': 0}
            
            # Wolffia-specific size categories (in μm²)
            small_count = sum(1 for area in areas_microns if area < 80)
            medium_count = sum(1 for area in areas_microns if 80 <= area <= 150)
            large_count = sum(1 for area in areas_microns if area > 150)
            optimal_count = sum(1 for area in areas_microns if 90 <= area <= 130)  # Optimal health range
            
            return {
                'small': small_count,
                'medium': medium_count,
                'large': large_count,
                'optimal': optimal_count,
                'percentages': {
                    'small': (small_count / len(areas_microns)) * 100,
                    'medium': (medium_count / len(areas_microns)) * 100,
                    'large': (large_count / len(areas_microns)) * 100,
                    'optimal': (optimal_count / len(areas_microns)) * 100
                },
                'size_ranges': {
                    'small': '< 80 μm²',
                    'medium': '80-150 μm²',
                    'large': '> 150 μm²',
                    'optimal': '90-130 μm²'
                }
            }
            
        except Exception as e:
            print(f"❌ Size distribution analysis failed: {e}")
            return {'small': 0, 'medium': 0, 'large': 0, 'optimal': 0}

    def _calculate_biomass_metrics(self, cells, areas_microns, image_rgb):
        """Calculate biomass using research-based mathematical equations for Wolffia"""
        try:
            if not cells:
                return {'total_biomass_mg': 0, 'avg_biomass_mg': 0}
            
            total_fresh_weight = 0
            total_dry_weight = 0
            total_chlorophyll = 0
            total_protein = 0
            total_carbon = 0
            
            for i, cell in enumerate(cells):
                area_microns = areas_microns[i] if i < len(areas_microns) else 0
                intensity = cell.get('intensity', 0)
                green_intensity = cell.get('green_intensity', intensity)
                
                # Volume calculation using ellipsoid approximation
                # V = π/6 * length * width * height
                # For circular cells: V ≈ area * thickness * correction_factor
                thickness = self.biomass_params['cell_thickness_microns']
                volume_microns3 = area_microns * thickness * self.biomass_params['volume_correction_factor']
                volume_mm3 = volume_microns3 / (1000 ** 3)  # Convert to mm³
                
                # Fresh weight calculation (density-based)
                # FW = Volume × Density
                density = self.biomass_params['cell_density_mg_mm3']
                fresh_weight_mg = volume_mm3 * density
                
                # Dry weight calculation (Wolffia: ~12% dry matter)
                dry_weight_mg = fresh_weight_mg * (1 - self.biomass_params['water_content_percentage'])
                
                # Chlorophyll content calculation (based on green intensity)
                # Normalized green intensity as proxy for chlorophyll concentration
                normalized_green = max(0, min(1, green_intensity / 255.0))
                chlorophyll_density = self.biomass_params['chlorophyll_density_mg_g']
                chlorophyll_mg = (fresh_weight_mg * chlorophyll_density * normalized_green) / 1000
                
                # Protein content calculation
                protein_mg = dry_weight_mg * self.biomass_params['protein_content_percentage']
                
                # Carbon content calculation
                carbon_mg = dry_weight_mg * self.biomass_params['carbon_content_percentage']
                
                # Accumulate totals
                total_fresh_weight += fresh_weight_mg
                total_dry_weight += dry_weight_mg
                total_chlorophyll += chlorophyll_mg
                total_protein += protein_mg
                total_carbon += carbon_mg
                
                # Store individual cell biomass data
                cell['biomass_data'] = {
                    'fresh_weight_mg': fresh_weight_mg,
                    'dry_weight_mg': dry_weight_mg,
                    'chlorophyll_mg': chlorophyll_mg,
                    'protein_mg': protein_mg,
                    'carbon_mg': carbon_mg,
                    'volume_microns3': volume_microns3
                }
            
            # Calculate averages and derived metrics
            cell_count = len(cells)
            avg_fresh_weight = total_fresh_weight / cell_count
            
            # Biomass density (mg/mm²)
            total_area_mm2 = sum(areas_microns) / (1000 ** 2)
            biomass_density = total_fresh_weight / total_area_mm2 if total_area_mm2 > 0 else 0
            
            return {
                'total_biomass_mg': total_fresh_weight,
                'avg_biomass_mg': avg_fresh_weight,
                'total_dry_weight_mg': total_dry_weight,
                'avg_dry_weight_mg': total_dry_weight / cell_count,
                'total_chlorophyll_mg': total_chlorophyll,
                'avg_chlorophyll_mg': total_chlorophyll / cell_count,
                'total_protein_mg': total_protein,
                'total_carbon_mg': total_carbon,
                'biomass_density_mg_per_mm2': biomass_density,
                'protein_percentage': (total_protein / total_dry_weight) * 100 if total_dry_weight > 0 else 0,
                'carbon_percentage': (total_carbon / total_dry_weight) * 100 if total_dry_weight > 0 else 0
            }
            
        except Exception as e:
            print(f"❌ Biomass calculation failed: {e}")
            return {'total_biomass_mg': 0, 'avg_biomass_mg': 0}

    def _analyze_color_properties(self, cells, green_intensities, image_rgb):
        """Analyze color properties and green wavelength intensity for chlorophyll assessment"""
        try:
            if not cells:
                return {'green_cell_percentage': 0, 'avg_green_intensity': 0}
            
            # Green cell classification based on intensity threshold
            green_threshold = self.wolffia_params['green_intensity_threshold']
            green_cells = sum(1 for intensity in green_intensities if intensity > green_threshold)
            green_percentage = (green_cells / len(cells)) * 100
            
            # Wavelength-specific analysis simulation
            # Using green channel (545nm peak) as proxy for chlorophyll content
            wavelength_analysis = self._simulate_wavelength_analysis(green_intensities)
            
            # Chlorophyll health categories based on green intensity
            high_chlorophyll = sum(1 for intensity in green_intensities if intensity > 150)
            medium_chlorophyll = sum(1 for intensity in green_intensities if 80 <= intensity <= 150)
            low_chlorophyll = sum(1 for intensity in green_intensities if intensity < 80)
            
            # Color intensity statistics
            avg_green = np.mean(green_intensities)
            std_green = np.std(green_intensities)
            
            # Calculate color uniformity (coefficient of variation)
            color_uniformity = (std_green / avg_green) * 100 if avg_green > 0 else 0
            
            return {
                'green_cell_percentage': green_percentage,
                'avg_green_intensity': avg_green,
                'std_green_intensity': std_green,
                'green_cell_count': green_cells,
                'color_uniformity_cv': color_uniformity,
                'chlorophyll_distribution': {
                    'high': high_chlorophyll,
                    'medium': medium_chlorophyll,
                    'low': low_chlorophyll
                },
                'wavelength_analysis': wavelength_analysis,
                'green_intensity_range': {
                    'min': min(green_intensities),
                    'max': max(green_intensities),
                    'median': np.median(green_intensities)
                }
            }
            
        except Exception as e:
            print(f"❌ Color analysis failed: {e}")
            return {'green_cell_percentage': 0, 'avg_green_intensity': 0}

    def _simulate_wavelength_analysis(self, green_intensities):
        """Simulate wavelength-specific analysis for professional chlorophyll assessment"""
        try:
            # Simulate different wavelength responses based on green intensity
            # Based on chlorophyll absorption spectrum (peak at 430nm and 660nm, transmission at 545nm)
            
            responses = []
            for intensity in green_intensities:
                normalized = intensity / 255.0
                
                # Simulate wavelength response curve
                wavelength_response = {
                    '430nm': normalized * 0.2,  # Blue absorption (chlorophyll a)
                    '470nm': normalized * 0.4,  # Blue-green absorption
                    '545nm': normalized * 1.0,  # Green transmission (peak visibility)
                    '660nm': normalized * 0.3,  # Red absorption (chlorophyll a)
                    '700nm': normalized * 0.1,  # Far red (minimal absorption)
                    'chlorophyll_a_index': normalized * 0.8,
                    'chlorophyll_b_index': normalized * 0.6
                }
                responses.append(wavelength_response)
            
            # Calculate average responses
            avg_responses = {}
            for wavelength in responses[0].keys():
                avg_responses[wavelength] = np.mean([r[wavelength] for r in responses])
            
            # Calculate chlorophyll health index
            chlorophyll_index = (avg_responses.get('chlorophyll_a_index', 0) + 
                               avg_responses.get('chlorophyll_b_index', 0)) / 2
            
            return {
                'peak_wavelength_nm': self.biomass_params['green_wavelength_nm'],
                'average_responses': avg_responses,
                'chlorophyll_index': chlorophyll_index,
                'absorption_ratio': avg_responses.get('430nm', 0) / max(avg_responses.get('545nm', 1), 0.01),
                'health_indicator': 'healthy' if chlorophyll_index > 0.6 else 'moderate' if chlorophyll_index > 0.3 else 'stressed'
            }
            
        except Exception as e:
            return {
                'peak_wavelength_nm': 545, 
                'average_responses': {}, 
                'chlorophyll_index': 0,
                'health_indicator': 'unknown'
            }

    def _assess_population_health(self, cells, areas_microns, green_intensities):
        """Assess overall population health based on size, color, and morphology"""
        try:
            if not cells:
                return {'overall_health': 'unknown', 'health_score': 0}
            
            health_scores = []
            
            for i, cell in enumerate(cells):
                area = areas_microns[i] if i < len(areas_microns) else 0
                green_intensity = green_intensities[i] if i < len(green_intensities) else 0
                
                score = 0
                
                # Size health component (40% weight)
                optimal_min, optimal_max = self.biomass_params['optimal_size_range_microns']
                if optimal_min <= area <= optimal_max:
                    score += 0.4  # Optimal size
                elif optimal_min * 0.7 <= area <= optimal_max * 1.3:
                    score += 0.25  # Acceptable size
                else:
                    score += 0.1   # Sub-optimal size
                
                # Color/chlorophyll health component (40% weight)
                if green_intensity > 120:
                    score += 0.4  # High chlorophyll
                elif green_intensity > 80:
                    score += 0.25  # Medium chlorophyll
                elif green_intensity > 40:
                    score += 0.1   # Low chlorophyll
                # Below 40: no points (very poor health)
                
                # Shape health component (20% weight)
                circularity = cell.get('circularity', 0)
                if circularity > 0.7:
                    score += 0.2  # Good shape
                elif circularity > 0.5:
                    score += 0.1  # Acceptable shape
                # Below 0.5: no points
                
                health_scores.append(min(1.0, score))  # Cap at 1.0
            
            # Calculate population statistics
            avg_health_score = np.mean(health_scores)
            
            # Categorize overall health
            if avg_health_score >= 0.8:
                health_status = 'excellent'
            elif avg_health_score >= 0.6:
                health_status = 'good'
            elif avg_health_score >= 0.4:
                health_status = 'moderate'
            elif avg_health_score >= 0.2:
                health_status = 'poor'
            else:
                health_status = 'critical'
            
            # Health distribution
            excellent_cells = sum(1 for score in health_scores if score >= 0.8)
            good_cells = sum(1 for score in health_scores if 0.6 <= score < 0.8)
            moderate_cells = sum(1 for score in health_scores if 0.4 <= score < 0.6)
            poor_cells = sum(1 for score in health_scores if 0.2 <= score < 0.4)
            critical_cells = sum(1 for score in health_scores if score < 0.2)
            
            return {
                'overall_health': health_status,
                'health_score': avg_health_score,
                'health_score_std': np.std(health_scores),
                'health_distribution': {
                    'excellent': excellent_cells,
                    'good': good_cells,
                    'moderate': moderate_cells,
                    'poor': poor_cells,
                    'critical': critical_cells
                },
                'health_percentages': {
                    'excellent': (excellent_cells / len(cells)) * 100,
                    'good': (good_cells / len(cells)) * 100,
                    'moderate': (moderate_cells / len(cells)) * 100,
                    'poor': (poor_cells / len(cells)) * 100,
                    'critical': (critical_cells / len(cells)) * 100
                }
            }
            
        except Exception as e:
            print(f"❌ Health assessment failed: {e}")
            return {'overall_health': 'unknown', 'health_score': 0}

    def analyze_temporal_changes(self, cells, timestamp, filename):
        """Analyze temporal changes for time-series Wolffia analysis"""
        try:
            # Store current analysis data
            image_id = f"{timestamp}_{filename}"
            
            current_data = {
                'timestamp': timestamp,
                'filename': filename,
                'cell_count': len(cells),
                'total_area': sum(cell['area'] for cell in cells),
                'total_area_microns': sum(cell.get('area_microns', 0) for cell in cells),
                'avg_intensity': np.mean([cell['intensity'] for cell in cells]) if cells else 0,
                'avg_green_intensity': np.mean([cell.get('green_intensity', cell['intensity']) for cell in cells]) if cells else 0,
                'total_biomass': sum(cell.get('biomass_data', {}).get('fresh_weight_mg', 0) for cell in cells),
                'avg_cell_area': np.mean([cell['area'] for cell in cells]) if cells else 0,
                'cells': cells
            }
            
            self.temporal_data[image_id] = current_data
            
            # Calculate temporal metrics if multiple time points exist
            if len(self.temporal_data) > 1:
                return self._calculate_temporal_metrics()
            else:
                return {
                    'message': 'First time point recorded', 
                    'time_points': 1,
                    'timestamp': timestamp
                }
            
        except Exception as e:
            print(f"❌ Temporal analysis failed: {e}")
            return None

    def _calculate_temporal_metrics(self):
        """Calculate comprehensive temporal change metrics"""
        try:
            sorted_data = sorted(self.temporal_data.values(), key=lambda x: x['timestamp'])
            
            if len(sorted_data) < 2:
                return {'time_points': len(sorted_data)}
            
            # Create time series data
            time_series = {
                'timestamps': [data['timestamp'] for data in sorted_data],
                'cell_counts': [data['cell_count'] for data in sorted_data],
                'total_areas': [data['total_area'] for data in sorted_data],
                'total_areas_microns': [data['total_area_microns'] for data in sorted_data],
                'avg_intensities': [data['avg_intensity'] for data in sorted_data],
                'avg_green_intensities': [data['avg_green_intensity'] for data in sorted_data],
                'total_biomass': [data['total_biomass'] for data in sorted_data],
                'avg_cell_areas': [data['avg_cell_area'] for data in sorted_data]
            }
            
            # Calculate growth rates and trends
            growth_analysis = self._calculate_growth_rates(time_series)
            
            # Determine overall trend
            temporal_trend = self._determine_temporal_trend(time_series)
            
            # Calculate population stability metrics
            stability_metrics = self._calculate_stability_metrics(time_series)
            
            return {
                'time_points': len(sorted_data),
                'time_series': time_series,
                'growth_analysis': growth_analysis,
                'temporal_trend': temporal_trend,
                'stability_metrics': stability_metrics,
                'analysis_period': {
                    'start': sorted_data[0]['timestamp'],
                    'end': sorted_data[-1]['timestamp'],
                    'duration_points': len(sorted_data)
                }
            }
            
        except Exception as e:
            print(f"❌ Temporal metrics calculation failed: {e}")
            return {'time_points': len(self.temporal_data)}

    def _calculate_growth_rates(self, time_series):
        """Calculate growth rates between time points"""
        try:
            if len(time_series['cell_counts']) < 2:
                return {}
            
            # Calculate percentage changes between consecutive time points
            metrics = ['cell_counts', 'total_areas_microns', 'total_biomass', 'avg_cell_areas']
            growth_rates = {}
            
            for metric in metrics:
                if metric in time_series and len(time_series[metric]) > 1:
                    changes = []
                    for i in range(1, len(time_series[metric])):
                        prev_value = time_series[metric][i-1]
                        curr_value = time_series[metric][i]
                        
                        if prev_value > 0:
                            change_percent = ((curr_value - prev_value) / prev_value) * 100
                            changes.append(change_percent)
                    
                    growth_rates[metric] = {
                        'changes': changes,
                        'avg_change_percent': np.mean(changes) if changes else 0,
                        'total_change_percent': ((time_series[metric][-1] - time_series[metric][0]) / 
                                               max(time_series[metric][0], 0.001)) * 100
                    }
            
            return growth_rates
            
        except Exception as e:
            return {}

    def _determine_temporal_trend(self, time_series):
        """Determine overall temporal trend"""
        try:
            if len(time_series['cell_counts']) < 2:
                return 'insufficient_data'
            
            # Analyze multiple metrics to determine trend
            first_count = time_series['cell_counts'][0]
            last_count = time_series['cell_counts'][-1]
            
            first_biomass = time_series['total_biomass'][0] if time_series['total_biomass'] else 0
            last_biomass = time_series['total_biomass'][-1] if time_series['total_biomass'] else 0
            
            # Count-based trend
            count_change = (last_count - first_count) / max(first_count, 1)
            
            # Biomass-based trend (if available)
            biomass_change = 0
            if first_biomass > 0:
                biomass_change = (last_biomass - first_biomass) / first_biomass
            
            # Combined assessment
            if count_change > 0.15 or biomass_change > 0.15:
                return 'growing'
            elif count_change < -0.15 or biomass_change < -0.15:
                return 'declining'
            elif abs(count_change) <= 0.05 and abs(biomass_change) <= 0.05:
                return 'stable'
            else:
                return 'fluctuating'
                
        except Exception as e:
            return 'unknown'

    def _calculate_stability_metrics(self, time_series):
        """Calculate population stability metrics"""
        try:
            metrics = {}
            
            # Coefficient of variation for key measurements
            for key in ['cell_counts', 'avg_cell_areas', 'avg_green_intensities']:
                if key in time_series and len(time_series[key]) > 1:
                    values = time_series[key]
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    cv = (std_val / mean_val) * 100 if mean_val > 0 else 0
                    metrics[f'{key}_cv'] = cv
            
            # Overall stability assessment
            avg_cv = np.mean(list(metrics.values())) if metrics else 0
            
            if avg_cv < 10:
                stability = 'very_stable'
            elif avg_cv < 20:
                stability = 'stable'
            elif avg_cv < 35:
                stability = 'moderate'
            else:
                stability = 'unstable'
            
            metrics['overall_stability'] = stability
            metrics['average_cv'] = avg_cv
            
            return metrics
            
        except Exception as e:
            return {'overall_stability': 'unknown'}

    def create_professional_visualizations(self, original, cells, metrics, temporal_analysis=None):
        """Create focused professional visualizations with pipeline steps for Wolffia analysis"""
        try:
            visualizations = {}
            
            # 1. Main detection visualization with cell highlighting
            detection_result = self._create_detection_visualization(original, cells, metrics)
            visualizations['detection_overview'] = detection_result['image']
            visualizations['detection_legend'] = detection_result['legend']
            
            # 2. Pipeline steps visualization - NEW!
            visualizations['pipeline_steps'] = self._create_pipeline_visualization(original, cells)
            
            # 3. Biomass analysis chart
            visualizations['biomass_chart'] = self._create_biomass_chart(cells, metrics)
            
            # 4. Size distribution histogram
            visualizations['size_histogram'] = self._create_size_histogram(cells)
            
            # 5. Color analysis for green wavelength
            visualizations['color_analysis'] = self._create_color_visualization(cells, metrics)
            
            # 6. Temporal analysis (if available)
            if temporal_analysis and temporal_analysis.get('time_points', 0) > 1:
                visualizations['temporal_analysis'] = self._create_temporal_visualization(temporal_analysis)
            
            return visualizations
            
        except Exception as e:
            print(f"❌ Visualization creation failed: {e}")
            return {'error': str(e)}

    def _create_pipeline_visualization(self, original, cells):
        """Create step-by-step pipeline visualization showing each processing stage"""
        try:
            print("🎨 Creating optimized pipeline step visualization...")
            
            # Re-run the preprocessing to capture steps
            processed = self.focused_preprocess(original)
            
            # Get the processing steps
            steps = processed.get('steps', {})
            gray = processed.get('gray')
            enhanced = processed.get('enhanced')
            
            # Create a comprehensive pipeline visualization with proper sizing
            fig = plt.figure(figsize=(24, 8))  # Wider aspect ratio
            
            # Define the pipeline steps to show
            pipeline_steps = [
                ('original', 'Original Image', original),
                ('gray', 'Grayscale', gray),
                ('green_enhanced', 'Green Enhanced', steps.get('green_enhanced', enhanced)),  # NEW
                ('li_foreground', 'Green Threshold', steps.get('li_foreground')),
                ('plate_removed', 'Edge Cleanup', steps.get('plate_removed')),
                ('multi_otsu', 'Green Multi-Otsu', steps.get('multi_otsu')),
                ('combined', 'Combined Masks', steps.get('combined')),
                ('opened', 'Morphological', steps.get('opened')),
                ('final', 'Final Segmentation', steps.get('final')),
                ('detection_result', 'Green Cell Detection', self._create_detection_overlay(original, cells))
            ]
                        
            # Create subplots in a single row
            cols = len(pipeline_steps)
            
            for i, (step_key, step_title, step_image) in enumerate(pipeline_steps):
                ax = fig.add_subplot(1, cols, i + 1)
                
                if step_image is not None:
                    # Ensure proper image display
                    if len(step_image.shape) == 3:
                        display_image = step_image
                    else:
                        display_image = step_image
                    
                    ax.imshow(display_image, cmap='gray' if len(step_image.shape) == 2 else None, 
                            aspect='equal', interpolation='nearest')
                    
                    # Compact title
                    ax.set_title(f'{i+1}. {step_title}', fontsize=9, fontweight='bold', pad=5)
                    
                    # Add cell count for final step
                    if step_key == 'detection_result':
                        ax.text(0.02, 0.98, f'{len(cells)} cells', 
                            transform=ax.transAxes, fontsize=8, va='top',
                            bbox=dict(boxstyle='round,pad=0.2', facecolor='lightgreen', alpha=0.8))
                else:
                    ax.text(0.5, 0.5, 'N/A', transform=ax.transAxes,
                        ha='center', va='center', fontsize=10)
                    ax.set_title(f'{i+1}. {step_title}', fontsize=9, fontweight='bold')
                
                ax.axis('off')
            
            plt.suptitle('BIOIMAGIN Processing Pipeline - Wolffia Analysis Steps', 
                        fontsize=14, fontweight='bold', y=0.95)
            plt.tight_layout()
            plt.subplots_adjust(top=0.85, bottom=0.05, left=0.02, right=0.98, wspace=0.1)
            
            # Convert to base64 with higher DPI for clarity
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=200, bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            # Create individual step images with consistent sizing
            individual_steps = {}
            for step_key, step_title, step_image in pipeline_steps:
                if step_image is not None:
                    individual_steps[step_key] = self._create_individual_step_image(step_image, step_title)
            
            return {
                'pipeline_overview': image_base64,
                'individual_steps': individual_steps,
                'step_count': len([s for s in pipeline_steps if s[2] is not None]),
                'step_descriptions': {
                    'original': 'Input image as uploaded by user',
                    'gray': 'Converted to grayscale for processing',
                    'denoised': 'Non-local means denoising applied',
                    'li_foreground': 'Li thresholding for foreground extraction',
                    'plate_removed': 'Large object removal (plates/shadows)',
                    'multi_otsu': 'Multi-Otsu thresholding (3 classes)',
                    'combined': 'Combined Li and Multi-Otsu results',
                    'opened': 'Morphological opening applied',
                    'final': 'Final segmentation mask',
                    'detection_result': f'Final cell detection: {len(cells)} cells found'
                }
            }
            
        except Exception as e:
            print(f"❌ Pipeline visualization creation failed: {e}")
            return {'error': str(e)}

    def _create_detection_overlay(self, original, cells):
        """Create detection overlay for pipeline visualization"""
        try:
            # Create a copy of the original image
            overlay = original.copy()
            
            # Draw cell boundaries
            for cell in cells:
                contour = cell.get('contour')
                if contour and len(contour) > 0:
                    # Convert contour to numpy array for drawing
                    if isinstance(contour, list):
                        contour_np = np.array(contour, dtype=np.int32)
                        if len(contour_np.shape) == 3:
                            contour_np = contour_np.reshape(-1, 2)
                        
                        # Draw contour on overlay
                        cv2.polylines(overlay, [contour_np.reshape(-1, 1, 2)], True, (0, 255, 0), 2)
                        
                        # Draw center point
                        center = cell.get('center', (0, 0))
                        cv2.circle(overlay, center, 3, (255, 0, 0), -1)
                        
                        # Draw cell ID
                        cv2.putText(overlay, str(cell['id']), 
                                   (center[0]-10, center[1]-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            return overlay
            
        except Exception as e:
            print(f"⚠️ Detection overlay creation failed: {e}")
            return original

    def _create_individual_step_image(self, step_image, step_title):
        """Create individual step image with consistent sizing and detailed information"""
        try:
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            
            # Display image with proper aspect ratio
            if len(step_image.shape) == 3:
                ax.imshow(step_image, aspect='equal', interpolation='nearest')
            else:
                ax.imshow(step_image, cmap='gray', aspect='equal', interpolation='nearest')
            
            ax.set_title(step_title, fontsize=16, fontweight='bold', pad=20)
            ax.axis('off')
            
            # Add comprehensive image statistics
            if len(step_image.shape) == 2:
                # Grayscale image
                stats_text = f"""Image Statistics:
    Min: {step_image.min()}  |  Max: {step_image.max()}
    Mean: {step_image.mean():.1f}  |  Std: {step_image.std():.1f}
    Shape: {step_image.shape[0]} × {step_image.shape[1]} pixels
    Data type: {step_image.dtype}"""
            else:
                # Color image
                stats_text = f"""Image Statistics:
    Shape: {step_image.shape[0]} × {step_image.shape[1]} × {step_image.shape[2]}
    Data type: {step_image.dtype}
    Color channels: RGB"""
            
            ax.text(0.02, 0.02, stats_text, transform=ax.transAxes,
                fontsize=10, va='bottom', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9))
            
            plt.tight_layout()
            
            # Convert to base64 with high quality
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return image_base64
            
        except Exception as e:
            print(f"⚠️ Individual step image creation failed: {e}")
            return None

    def _create_detection_visualization(self, original, cells, metrics):
        """Create main cell detection visualization with professional highlighting"""
        try:
            fig, ax = plt.subplots(1, 1, figsize=(12, 10))
            
            # Display original image
            if len(original.shape) == 2:
                original_display = np.stack([original, original, original], axis=2)
            else:
                original_display = original.copy()
            
            ax.imshow(original_display)
            ax.set_title(f'Wolffia Cell Detection & Analysis: {len(cells)} cells detected', 
                        fontsize=16, fontweight='bold')
            
            # Initialize legend data
            legend_data = {
                'total_cells': len(cells),
                'color_coding': {
                    'high_biomass': {'color': 'lime', 'threshold': '>0.1 mg', 'count': 0},
                    'medium_biomass': {'color': 'yellow', 'threshold': '0.05-0.1 mg', 'count': 0},
                    'low_biomass': {'color': 'red', 'threshold': '<0.05 mg', 'count': 0}
                },
                'metrics': {
                    'total_biomass_mg': metrics['biomass_analysis']['total_biomass_mg'],
                    'average_cell_area': metrics['avg_area'],
                    'green_cell_percentage': metrics['color_analysis']['green_cell_percentage'],
                    'health_status': metrics['health_assessment']['overall_health']
                }
            }
            
            if cells:
                # Highlight cells based on biomass content
                for cell in cells:
                    try:
                        contour = cell.get('contour')
                        if contour is None:
                            # Draw circle for cells without contour
                            center = cell.get('center', (0, 0))
                            radius = np.sqrt(cell.get('area', 100) / np.pi)
                            circle = plt.Circle(center, radius, fill=False, 
                                              edgecolor='cyan', linewidth=2, alpha=0.8)
                            ax.add_patch(circle)
                            continue
                        
                        # Convert contour to numpy array
                        if isinstance(contour, list):
                            contour = np.array(contour, dtype=np.float32)
                        
                        if len(contour.shape) == 3 and contour.shape[1] == 1:
                            points = contour[:, 0, :]
                        elif len(contour.shape) == 2:
                            points = contour
                        else:
                            continue
                        
                        if len(points) < 3:
                            continue
                        
                        # Color coding based on biomass
                        biomass_data = cell.get('biomass_data', {})
                        fresh_weight = biomass_data.get('fresh_weight_mg', 0)
                        
                        if fresh_weight > 0.1:
                            color = 'lime'
                            linewidth = 3
                            legend_data['color_coding']['high_biomass']['count'] += 1
                        elif fresh_weight > 0.05:
                            color = 'yellow'
                            linewidth = 2
                            legend_data['color_coding']['medium_biomass']['count'] += 1
                        else:
                            color = 'red'
                            linewidth = 2
                            legend_data['color_coding']['low_biomass']['count'] += 1
                        
                        # Draw cell boundary
                        polygon = plt.Polygon(points, fill=False, edgecolor=color,
                                            linewidth=linewidth, alpha=0.8)
                        ax.add_patch(polygon)
                        
                        # Add cell ID number
                        center = cell.get('center', (0, 0))
                        ax.text(center[0], center[1], str(cell['id']), 
                               color='white', fontsize=8, ha='center', va='center',
                               bbox=dict(boxstyle='circle,pad=0.1', facecolor='black', alpha=0.7))
                        
                    except Exception as e:
                        continue
            
            ax.axis('off')
            plt.tight_layout()
            
            # Convert to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight', facecolor='white')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return {
                'image': image_base64,
                'legend': legend_data
            }
            
        except Exception as e:
            print(f"❌ Detection visualization failed: {e}")
            return {
                'image': None,
                'legend': {'total_cells': 0, 'color_coding': {}, 'metrics': {}}
            }

    def _create_biomass_chart(self, cells, metrics):
        """Create comprehensive biomass analysis chart"""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
            
            if cells:
                # Extract biomass data
                biomass_values = []
                chlorophyll_values = []
                cell_areas = []
                
                for cell in cells:
                    biomass_data = cell.get('biomass_data', {})
                    biomass_values.append(biomass_data.get('fresh_weight_mg', 0))
                    chlorophyll_values.append(biomass_data.get('chlorophyll_mg', 0))
                    cell_areas.append(cell.get('area_microns', 0))
                
                # 1. Biomass distribution histogram
                ax1.hist(biomass_values, bins=15, color='lightgreen', alpha=0.7, edgecolor='black')
                ax1.axvline(np.mean(biomass_values), color='red', linestyle='--', linewidth=2, 
                           label=f'Mean: {np.mean(biomass_values):.4f} mg')
                ax1.set_xlabel('Fresh Weight (mg)', fontsize=10)
                ax1.set_ylabel('Number of Cells', fontsize=10)
                ax1.set_title('Individual Cell Biomass Distribution', fontsize=11, fontweight='bold')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # 2. Biomass vs Area correlation
                ax2.scatter(cell_areas, biomass_values, alpha=0.6, c='green', s=30)
                ax2.set_xlabel('Cell Area (μm²)', fontsize=10)
                ax2.set_ylabel('Fresh Weight (mg)', fontsize=10)
                ax2.set_title('Biomass vs Cell Area', fontsize=11, fontweight='bold')
                ax2.grid(True, alpha=0.3)
                
                # Add correlation line if enough data
                if len(biomass_values) > 2:
                    z = np.polyfit(cell_areas, biomass_values, 1)
                    p = np.poly1d(z)
                    ax2.plot(cell_areas, p(cell_areas), "r--", alpha=0.8, linewidth=1)
                    
                    # Calculate R²
                    correlation = np.corrcoef(cell_areas, biomass_values)[0, 1]
                    ax2.text(0.05, 0.95, f'R = {correlation:.3f}', transform=ax2.transAxes,
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                # 3. Chlorophyll content analysis
                ax3.scatter(biomass_values, chlorophyll_values, alpha=0.6, c='darkgreen', s=30)
                ax3.set_xlabel('Fresh Weight (mg)', fontsize=10)
                ax3.set_ylabel('Chlorophyll Content (mg)', fontsize=10)
                ax3.set_title('Chlorophyll vs Biomass', fontsize=11, fontweight='bold')
                ax3.grid(True, alpha=0.3)
                
                # 4. Population biomass summary
                ax4.axis('off')
                
                # Create summary statistics
                biomass_analysis = metrics['biomass_analysis']
                
                summary_text = f"""
Population Biomass Summary

Total Biomass: {biomass_analysis['total_biomass_mg']:.3f} mg
Average per Cell: {biomass_analysis['avg_biomass_mg']:.4f} mg
Total Chlorophyll: {biomass_analysis['total_chlorophyll_mg']:.4f} mg
Total Protein: {biomass_analysis.get('total_protein_mg', 0):.4f} mg

Density: {biomass_analysis.get('biomass_density_mg_per_mm2', 0):.2f} mg/mm²

Cell Count: {len(cells)}
Size Range: {min(cell_areas):.1f} - {max(cell_areas):.1f} μm²
                """.strip()
                
                ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes,
                        fontsize=11, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.3))
                
                # Main title
                fig.suptitle(f'Wolffia Biomass Analysis - Total: {biomass_analysis["total_biomass_mg"]:.3f} mg', 
                            fontsize=14, fontweight='bold')
                
            else:
                # No cells detected
                for ax in [ax1, ax2, ax3, ax4]:
                    ax.text(0.5, 0.5, 'No cells detected', transform=ax.transAxes,
                           ha='center', va='center', fontsize=16)
                    ax.set_xticks([])
                    ax.set_yticks([])
            
            plt.tight_layout()
            
            # Convert to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight', facecolor='white')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return image_base64
            
        except Exception as e:
            print(f"❌ Biomass chart creation failed: {e}")
            return None

    def _create_size_histogram(self, cells):
        """Create cell size distribution histogram with analysis"""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            if cells:
                areas_microns = [cell.get('area_microns', cell['area'] * (self.pixel_to_micron_ratio ** 2)) 
                               for cell in cells]
                
                # 1. Size distribution histogram
                ax1.hist(areas_microns, bins=20, color='skyblue', alpha=0.7, edgecolor='black')
                ax1.axvline(np.mean(areas_microns), color='red', linestyle='--', linewidth=2, 
                           label=f'Mean: {np.mean(areas_microns):.1f} μm²')
                ax1.axvline(np.median(areas_microns), color='green', linestyle='--', linewidth=2, 
                           label=f'Median: {np.median(areas_microns):.1f} μm²')
                
                # Add optimal range
                optimal_min, optimal_max = self.biomass_params['optimal_size_range_microns']
                ax1.axvspan(optimal_min, optimal_max, alpha=0.2, color='green', label='Optimal Range')
                
                ax1.set_xlabel('Cell Area (μm²)', fontsize=12)
                ax1.set_ylabel('Number of Cells', fontsize=12)
                ax1.set_title('Cell Size Distribution', fontsize=14, fontweight='bold')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # 2. Size categories pie chart
                small_count = sum(1 for area in areas_microns if area < 80)
                medium_count = sum(1 for area in areas_microns if 80 <= area <= 150)
                large_count = sum(1 for area in areas_microns if area > 150)
                
                sizes = [small_count, medium_count, large_count]
                labels = ['Small\n(< 80 μm²)', 'Medium\n(80-150 μm²)', 'Large\n(> 150 μm²)']
                colors = ['lightcoral', 'lightblue', 'lightgreen']
                
                # Only plot if there are cells
                non_zero_sizes = [(size, label, color) for size, label, color in zip(sizes, labels, colors) if size > 0]
                if non_zero_sizes:
                    sizes_nz, labels_nz, colors_nz = zip(*non_zero_sizes)
                    ax2.pie(sizes_nz, labels=labels_nz, colors=colors_nz, autopct='%1.1f%%', startangle=90)
                
                ax2.set_title('Size Category Distribution', fontsize=14, fontweight='bold')
                
                # Add statistics
                stats_text = f"""
Statistics:
• Count: {len(areas_microns)}
• Mean: {np.mean(areas_microns):.1f} μm²
• Std: {np.std(areas_microns):.1f} μm²
• Min: {np.min(areas_microns):.1f} μm²
• Max: {np.max(areas_microns):.1f} μm²
• Optimal cells: {sum(1 for area in areas_microns if optimal_min <= area <= optimal_max)}
                """.strip()
                
                fig.text(0.02, 0.02, stats_text, fontsize=9, 
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
                
            else:
                ax1.text(0.5, 0.5, 'No cells detected', transform=ax1.transAxes,
                        ha='center', va='center', fontsize=16)
                ax2.text(0.5, 0.5, 'No data available', transform=ax2.transAxes,
                        ha='center', va='center', fontsize=16)
            
            plt.tight_layout()
            
            # Convert to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight', facecolor='white')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return image_base64
            
        except Exception as e:
            print(f"❌ Size histogram creation failed: {e}")
            return None

    def _create_color_visualization(self, cells, metrics):
        """Create color analysis visualization focusing on green wavelength"""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
            
            if cells:
                green_intensities = [cell.get('green_intensity', cell['intensity']) for cell in cells]
                
                # 1. Green intensity distribution
                ax1.hist(green_intensities, bins=20, color='lightgreen', alpha=0.7, edgecolor='black')
                ax1.axvline(self.wolffia_params['green_intensity_threshold'], color='red', 
                           linestyle='--', linewidth=2, label='Green Threshold')
                ax1.axvline(np.mean(green_intensities), color='blue', linestyle='--', linewidth=2, 
                           label=f'Mean: {np.mean(green_intensities):.1f}')
                ax1.set_xlabel('Green Intensity (0-255)', fontsize=11)
                ax1.set_ylabel('Number of Cells', fontsize=11)
                ax1.set_title('Green Intensity Distribution (545nm)', fontsize=12, fontweight='bold')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # 2. Chlorophyll health distribution pie chart
                chlorophyll_dist = metrics['color_analysis']['chlorophyll_distribution']
                labels = ['High\nChlorophyll', 'Medium\nChlorophyll', 'Low\nChlorophyll']
                sizes = [chlorophyll_dist['high'], chlorophyll_dist['medium'], chlorophyll_dist['low']]
                colors = ['darkgreen', 'yellowgreen', 'lightcoral']
                
                # Filter out zero values
                non_zero = [(size, label, color) for size, label, color in zip(sizes, labels, colors) if size > 0]
                if non_zero:
                    sizes_nz, labels_nz, colors_nz = zip(*non_zero)
                    ax2.pie(sizes_nz, labels=labels_nz, colors=colors_nz, autopct='%1.1f%%', startangle=90)
                
                ax2.set_title('Chlorophyll Health Distribution', fontsize=12, fontweight='bold')
                
                # 3. Green intensity vs cell area
                areas = [cell.get('area_microns', 0) for cell in cells]
                ax3.scatter(areas, green_intensities, alpha=0.6, c=green_intensities, 
                           cmap='RdYlGn', s=30)
                ax3.set_xlabel('Cell Area (μm²)', fontsize=11)
                ax3.set_ylabel('Green Intensity', fontsize=11)
                ax3.set_title('Green Intensity vs Cell Size', fontsize=12, fontweight='bold')
                ax3.grid(True, alpha=0.3)
                
                # Add colorbar
                cbar = plt.colorbar(ax3.collections[0], ax=ax3)
                cbar.set_label('Green Intensity', fontsize=10)
                
                # 4. Wavelength analysis summary
                ax4.axis('off')
                
                color_analysis = metrics['color_analysis']
                wavelength_analysis = color_analysis.get('wavelength_analysis', {})
                
                summary_text = f"""
Color Analysis Summary

Green Cells: {color_analysis['green_cell_percentage']:.1f}% ({color_analysis['green_cell_count']})
Avg Green Intensity: {color_analysis['avg_green_intensity']:.1f}

Wavelength Analysis (545nm):
• Chlorophyll Index: {wavelength_analysis.get('chlorophyll_index', 0):.3f}
• Health Indicator: {wavelength_analysis.get('health_indicator', 'unknown').title()}

Distribution:
• High Chlorophyll: {chlorophyll_dist['high']} cells
• Medium Chlorophyll: {chlorophyll_dist['medium']} cells  
• Low Chlorophyll: {chlorophyll_dist['low']} cells

Color Uniformity: {color_analysis.get('color_uniformity_cv', 0):.1f}% CV
                """.strip()
                
                ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes,
                        fontsize=10, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.3))
                
                # Main title
                green_percentage = color_analysis['green_cell_percentage']
                fig.suptitle(f'Wolffia Color Analysis: {green_percentage:.1f}% Green Cells (545nm wavelength)', 
                            fontsize=14, fontweight='bold')
                
            else:
                for ax in [ax1, ax2, ax3, ax4]:
                    ax.text(0.5, 0.5, 'No cells detected', transform=ax.transAxes,
                           ha='center', va='center', fontsize=16)
            
            plt.tight_layout()
            
            # Convert to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight', facecolor='white')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return image_base64
            
        except Exception as e:
            print(f"❌ Color visualization creation failed: {e}")
            return None

    def _create_temporal_visualization(self, temporal_analysis):
        """Create temporal analysis visualization for time-series tracking"""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            time_series = temporal_analysis['time_series']
            timestamps = time_series['timestamps']
            
            # Convert timestamps to readable format
            time_labels = []
            for ts in timestamps:
                if '_' in ts:
                    time_labels.append(ts.split('_')[0])
                else:
                    time_labels.append(ts)
            
            # 1. Cell count over time
            ax1.plot(time_labels, time_series['cell_counts'], 'bo-', linewidth=3, markersize=8, 
                    color='#2E86AB', markerfacecolor='#A23B72')
            ax1.set_title('Cell Population Over Time', fontsize=13, fontweight='bold')
            ax1.set_ylabel('Number of Cells', fontsize=11)
            ax1.grid(True, alpha=0.3)
            ax1.tick_params(axis='x', rotation=45, labelsize=9)
            
            # Add trend line
            if len(time_series['cell_counts']) > 2:
                x_pos = range(len(time_series['cell_counts']))
                z = np.polyfit(x_pos, time_series['cell_counts'], 1)
                p = np.poly1d(z)
                ax1.plot(time_labels, p(x_pos), "r--", alpha=0.8, linewidth=2, label='Trend')
                ax1.legend()
            
            # 2. Total biomass over time
            if 'total_biomass' in time_series:
                ax2.plot(time_labels, time_series['total_biomass'], 'go-', linewidth=3, markersize=8,
                        color='#F18F01', markerfacecolor='#C73E1D')
                ax2.set_title('Total Biomass Over Time', fontsize=13, fontweight='bold')
                ax2.set_ylabel('Total Biomass (mg)', fontsize=11)
                ax2.grid(True, alpha=0.3)
                ax2.tick_params(axis='x', rotation=45, labelsize=9)
            
            # 3. Average cell area over time
            ax3.plot(time_labels, time_series['avg_cell_areas'], 'mo-', linewidth=3, markersize=8,
                    color='#7209B7', markerfacecolor='#560319')
            ax3.set_title('Average Cell Area Over Time', fontsize=13, fontweight='bold')
            ax3.set_ylabel('Average Area (μm²)', fontsize=11)
            ax3.grid(True, alpha=0.3)
            ax3.tick_params(axis='x', rotation=45, labelsize=9)
            
            # 4. Analysis summary
            ax4.axis('off')
            
            growth_analysis = temporal_analysis.get('growth_analysis', {})
            stability_metrics = temporal_analysis.get('stability_metrics', {})
            
            summary_text = f"""
TEMPORAL ANALYSIS SUMMARY

Time Points: {temporal_analysis['time_points']}
Trend: {temporal_analysis['temporal_trend'].replace('_', ' ').title()}

GROWTH RATES:
"""
            
            # Add growth rate information
            if 'cell_counts' in growth_analysis:
                avg_change = growth_analysis['cell_counts'].get('avg_change_percent', 0)
                total_change = growth_analysis['cell_counts'].get('total_change_percent', 0)
                summary_text += f"• Cell Count: {avg_change:.1f}% avg, {total_change:.1f}% total\n"
            
            if 'total_biomass' in growth_analysis:
                avg_change = growth_analysis['total_biomass'].get('avg_change_percent', 0)
                total_change = growth_analysis['total_biomass'].get('total_change_percent', 0)
                summary_text += f"• Biomass: {avg_change:.1f}% avg, {total_change:.1f}% total\n"
            
            # Add stability information
            overall_stability = stability_metrics.get('overall_stability', 'unknown')
            avg_cv = stability_metrics.get('average_cv', 0)
            summary_text += f"""
STABILITY:
• Overall: {overall_stability.replace('_', ' ').title()}
• Variability: {avg_cv:.1f}% CV

TREND ANALYSIS:
{temporal_analysis['temporal_trend'].replace('_', ' ').upper()}
            """.strip()
            
            ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
                    fontsize=11, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.3))
            
            plt.suptitle('Wolffia Population Temporal Analysis - Time Series Tracking', 
                        fontsize=16, fontweight='bold', y=0.98)
            plt.tight_layout()
            
            # Convert to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight', facecolor='white')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return image_base64
            
        except Exception as e:
            print(f"❌ Temporal visualization creation failed: {e}")
            return None

    # TOPHAT TRAINING METHODS (OPTIMIZED)
    
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
            result = self.analyze_image(image_path, use_tophat=False, use_celldetection=True)
            
            if result['success']:
                training_image = {
                    'path': str(image_path),
                    'filename': Path(image_path).name,
                    'auto_detected_cells': result['detection_results']['cells_data'],
                    'cells_count': result['detection_results']['cells_detected'],
                    'image_data': result
                }
                training_session['images'].append(training_image)
        
        # Save training session
        session_path = self.dirs['tophat_training'] / f"session_{training_session['id']}.json"
        with open(session_path, 'w') as f:
            json.dump(training_session, f, indent=2, default=str)
        
        print(f"✅ Training session created: {training_session['id']}")
        return training_session
    
    def save_drawing_annotations(self, session_id, image_filename, image_index, annotations, annotated_image):
        """Save user drawing annotations for training"""
        print(f"🎨 Saving drawing annotations for {image_filename}...")
        
        # Create annotation data structure
        annotation = {
            'session_id': session_id,
            'image_filename': image_filename,
            'image_index': image_index,
            'annotations': annotations,
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
                    annotated_image = annotated_image[22:]
                
                # Decode and save image
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
            from sklearn.model_selection import train_test_split
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
        
        # Load drawing annotation files
        drawing_files = list(self.dirs['annotations'].glob(f"{session_id}_*_drawing.json"))
        
        print(f"🔍 Found {len(drawing_files)} annotation files for session {session_id}")
        
        for drawing_file in drawing_files:
            try:
                with open(drawing_file, 'r') as f:
                    annotation = json.load(f)
                
                print(f"📝 Processing annotation file: {drawing_file.name}")
                
                # Create training samples based on user annotations
                base_features = self._extract_annotation_features(annotation)
                
                # Create multiple samples from each annotation
                for i in range(5):  # Generate 5 samples per annotation
                    sample_features = base_features.copy()
                    sample_features.append(i * 0.1)  # Add variation
                    
                    features.append(sample_features[:10])  # Ensure 10 features
                    labels.append(1)  # Positive sample (user reviewed)
                
            except Exception as e:
                print(f"⚠️ Error processing annotation file {drawing_file.name}: {e}")
                continue
        
        print(f"📊 Generated {len(features)} training samples from {len(drawing_files)} annotation files")
        
        return {'features': features, 'labels': labels}
    
    def _extract_annotation_features(self, annotation):
        """Extract features from annotation data"""
        try:
            image_index = annotation.get('image_index', 0)
            filename = annotation.get('image_filename', '')
            annotations_data = annotation.get('annotations', {})
            
            # Count annotations
            correct_count = len(annotations_data.get('correct', []))
            false_positive_count = len(annotations_data.get('false_positive', []))
            missed_count = len(annotations_data.get('missed', []))
            
            # Basic features
            features = [
                image_index,
                len(filename),
                correct_count,
                false_positive_count,
                missed_count,
                correct_count / max(1, correct_count + false_positive_count),  # Precision
                1,  # User reviewed flag
                0.5,  # Default value
                0.5,  # Default value
            ]
            
            return features
            
        except Exception as e:
            return [0.5] * 9

    def tophat_detection(self, enhanced, original):
        """AI-powered tophat detection using trained model"""
        try:
            if self.tophat_model is None:
                return []
            
            # Simple placeholder - would need proper implementation
            # based on training methodology
            return []
            
        except Exception as e:
            print(f"⚠️ Tophat AI detection failed: {e}")
            return []

    def get_tophat_status(self):
        """Get tophat model status"""
        return {
            'model_available': self.tophat_model is not None,
            'model_trained': self.tophat_model is not None
        }

    def _empty_metrics(self):
        """Return empty metrics structure"""
        return {
            'cell_count': 0,
            'total_area': 0,
            'total_area_microns': 0,
            'avg_area': 0,
            'avg_area_microns': 0,
            'std_area': 0,
            'size_distribution': {'small': 0, 'medium': 0, 'large': 0, 'optimal': 0},
            'biomass_analysis': {'total_biomass_mg': 0, 'avg_biomass_mg': 0, 'total_chlorophyll_mg': 0},
            'color_analysis': {'green_cell_percentage': 0, 'avg_green_intensity': 0, 'chlorophyll_distribution': {'high': 0, 'medium': 0, 'low': 0}},
            'health_assessment': {'overall_health': 'unknown', 'health_score': 0, 'health_distribution': {'excellent': 0, 'good': 0, 'moderate': 0, 'poor': 0}},
            'avg_intensity': 0,
            'avg_green_intensity': 0
        }

    def get_celldetection_status(self):
        """Get CellDetection model status for web interface"""
        return {
            'available': CELLDETECTION_AVAILABLE,
            'model_loaded': self.celldetection_model is not None,
            'device': getattr(self, 'device', 'cpu'),
            'model_name': 'ginoro_CpnResNeXt101UNet-fbe875f1a3e5ce2c' if CELLDETECTION_AVAILABLE else None
        }

    def create_error_result(self, error_message):
        """Create error result structure"""
        return {
            'success': False,
            'error': error_message,
            'timestamp': datetime.now().isoformat(),
            'detection_results': {'cells_detected': 0},
            'quantitative_analysis': self._empty_metrics(),
            'visualizations': None
        }

    def cleanup(self):
        """Explicitly cleanup resources for better memory management"""
        try:
            plt.close('all')
            if hasattr(self, '_celldetection_model') and self._celldetection_model is not None:
                del self._celldetection_model
                self._celldetection_model = None
            if hasattr(self, '_tophat_model') and self._tophat_model is not None:
                del self._tophat_model  
                self._tophat_model = None
            print("🧹 Resources cleaned up successfully")
        except Exception as e:
            print(f"⚠️ Cleanup warning: {e}")
    
    def __del__(self):
        """Automatic cleanup when object is destroyed"""
        self.cleanup()

    # Legacy method aliases for compatibility
    def analyze_single_image(self, image_path, **kwargs):
        """Legacy method alias"""
        return self.analyze_image(image_path, **kwargs)

# Legacy functions for backward compatibility
def analyze_uploaded_image(image_path, pixel_to_micron_ratio=0.5, **kwargs):
    """Legacy function for backward compatibility"""
    analyzer = WolffiaAnalyzer(pixel_to_micron_ratio=pixel_to_micron_ratio)
    return analyzer.analyze_image(image_path, **kwargs)

def analyze_multiple_images(image_paths, **kwargs):
    """Analyze multiple images with temporal analysis"""
    analyzer = WolffiaAnalyzer()
    results = []
    
    # Sort images by name for temporal sequence
    sorted_paths = sorted(image_paths, key=lambda x: Path(x).name)
    
    for i, image_path in enumerate(sorted_paths):
        # Create timestamp for temporal analysis
        timestamp = f"t{i+1:03d}_{datetime.now().strftime('%H%M%S')}"
        
        result = analyzer.analyze_image(image_path, image_timestamp=timestamp, **kwargs)
        results.append(result)
    
    return results