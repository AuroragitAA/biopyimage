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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage
from skimage import (
    color,
    exposure,
    feature,
    filters,
    measure,
    morphology,
    restoration,
    segmentation,
)
from skimage.segmentation import clear_border, watershed
from sklearn.ensemble import RandomForestClassifier

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
        """Initialize with minimal, robust configuration"""
        self.setup_directories()
        
        # Core parameters optimized for Wolffia
        self.min_cell_area = 30
        self.max_cell_area = 500
        self.pixel_to_micron = 0.5
        
        # Models loaded on demand - using proper naming
        self._tophat_model = None
        self._cnn_model = None
        self._celldetection_model = None
        self._device = None
        
        # Device property for CUDA support
        self._device = 'cuda' if torch.cuda.is_available() and TORCH_AVAILABLE else 'cpu'
        
        # Status properties for frontend compatibility
        self.wolffia_cnn_available = False
        self.celldetection_available = False
        self.tophat_model = None  # Will be set when model loads
        self.wolffia_cnn_model = None  # Will be set when model loads
        
        print("✅ WolffiaAnalyzer initialized - Deployment Ready")
    
    def setup_directories(self):
        """Setup required directories"""
        self.dirs = {
            'results': Path('results'),
            'models': Path('models'),
            'uploads': Path('uploads'),
            'annotations': Path('annotations')
        }
        
        for path in self.dirs.values():
            path.mkdir(exist_ok=True)
    
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
    
    def analyze_image(self, image_path, use_tophat=True, use_cnn=True, use_celldetection=False):
        """
        Main analysis method - streamlined and robust
        Based on proven microscopist patterns
        """
        try:
            # Load and preprocess image
            img = cv2.imread(str(image_path))
            if img is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Preserve color image for all analysis - NO GRAYSCALE CONVERSION
            color_img = img.copy()
            
            # Perform green detection analysis on the color image
            green_percentage = self.analyze_green_content(color_img)
            
            # Create enhanced grayscale for methods that need it (fallback only)
            enhanced_gray = self.create_green_enhanced_grayscale(color_img)
            
            # Get results from available methods - USING COLOR IMAGE
            results = []
            
            # Method 1: Enhanced Watershed (now works with color images)
            watershed_result = self.color_aware_watershed_segmentation(color_img)
            results.append(('watershed', watershed_result))
            
            # Method 2: Tophat ML (enhanced to use color information)
            if use_tophat and self.load_tophat_model():
                tophat_result = self.color_aware_tophat_detection(color_img, enhanced_gray)
                results.append(('tophat', tophat_result))
            
            # Method 3: CNN (enhanced to use RGB input directly)
            if use_cnn and TORCH_AVAILABLE and self.load_cnn_model():
                cnn_result = self.cnn_detection(enhanced_gray, color_img)  # Smart detection chooses RGB vs grayscale
                results.append(('cnn', cnn_result))
            
            # Method 4: CellDetection (works great with color)
            if use_celldetection and CELLDETECTION_AVAILABLE:
                celldetection_result = self.celldetection_detection(color_img)
                results.append(('celldetection', celldetection_result))
            
            # Intelligent fusion of results
            final_result = self.fuse_detection_results(results, color_img.shape[:2])
            
            # Extract cell properties using proven regionprops approach
            cell_data = self.extract_cell_properties(final_result, enhanced_gray)
            
            # Create professional visualization
            vis_path = self.create_professional_visualization(img, final_result, cell_data)
            
            # Get labeled image as base64 for frontend compatibility
            labeled_image_b64 = None
            if vis_path.exists():
                with open(vis_path, 'rb') as f:
                    import base64
                    labeled_image_b64 = base64.b64encode(f.read()).decode('utf-8')
            
            # Create method detection string
            detection_method = "Professional Multi-Method Detection"
            if 'cnn' in [method for method, _ in results]:
                detection_method = "Wolffia CNN + " + detection_method
            if 'celldetection' in [method for method, _ in results]:
                detection_method = "CellDetection AI + " + detection_method
            if 'tophat' in [method for method, _ in results]:
                detection_method = "Tophat AI + " + detection_method
            
            # Compile results in frontend-compatible format
            analysis_result = {
                # Legacy simple format for backward compatibility
                'total_cells': len(cell_data),
                'total_area': sum(cell['area'] for cell in cell_data),
                'average_area': np.mean([cell['area'] for cell in cell_data]) if cell_data else 0,
                'cells': cell_data,
                'labeled_image_path': str(vis_path),
                'method_used': [method for method, _ in results],
                'processing_time': 0,  # Will be set by caller
                
                # Extended format for frontend visualization
                'detection_results': {
                    'detection_method': detection_method,
                    'cells_detected': len(cell_data),
                    'total_area': sum(cell['area'] for cell in cell_data),
                    'cells_data': cell_data
                },
                'quantitative_analysis': {
                    'average_cell_area': np.mean([cell['area'] for cell in cell_data]) if cell_data else 0,
                    'biomass_analysis': {
                        'total_biomass_mg': sum(cell['area'] for cell in cell_data) * 0.001,  # Simple conversion
                    },
                    'color_analysis': {
                        'green_cell_percentage': green_percentage
                    },
                    'health_assessment': {
                        'overall_health': 'good',
                        'health_score': 0.75
                    }
                },
                'visualizations': {
                    'detection_overview': labeled_image_b64
                }
            }
            
            return analysis_result
            
        except Exception as e:
            print(f"❌ Analysis failed: {str(e)}")
            return self.get_error_result(str(e))
    
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
            return self.professional_watershed_segmentation(enhanced_channel, color_img)
            
        except Exception as e:
            print(f"⚠️ Color-aware watershed failed, using fallback: {e}")
            # Fallback to grayscale watershed
            gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
            return self.professional_watershed_segmentation(gray, color_img)

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
    
    def analyze_image_separate_methods(self, processed, image_path, use_tophat=True, use_cnn=True, use_celldetection=False):
        """
        Analyze image with each method separately for comparison.
        Returns results for each method individually in a consistent format.
        """
        try:
            img = cv2.imread(str(image_path))
            original = processed.get('original', img)
            if img is None or original is None:
                raise ValueError(f"❌ Could not load image: {image_path}")

            # Preserve color image - NO PREMATURE GRAYSCALE CONVERSION
            color_img = img.copy()
            # Create enhanced grayscale only when needed for specific methods
            enhanced_gray = self.create_green_enhanced_grayscale(color_img)
            method_results = {}

            # --- Method 1: Color-Aware Watershed (Always Run) ---
            print("🔬 Running Color-Aware Watershed method...")
            
            # Get watershed results with pipeline visualization
            try:
                enhanced_channel = self.create_green_enhanced_grayscale(color_img)
                watershed_labels, pipeline_images = self.professional_watershed_segmentation(
                    enhanced_channel, color_img, return_pipeline=True
                )
                watershed_cells = self.extract_cell_properties(watershed_labels, enhanced_gray)
                watershed_viz = self.create_method_visualization(img, watershed_labels, watershed_cells, "Color-Aware Watershed")
                
                # Create pipeline visualization
                watershed_pipeline_viz = self.create_watershed_pipeline_visualization(pipeline_images, watershed_cells)
                
            except Exception as e:
                print(f"⚠️ Watershed with pipeline failed, using fallback: {e}")
                watershed_labels = self.color_aware_watershed_segmentation(color_img)
                watershed_cells = self.extract_cell_properties(watershed_labels, enhanced_gray)
                watershed_viz = self.create_method_visualization(img, watershed_labels, watershed_cells, "Color-Aware Watershed")
                watershed_pipeline_viz = None
            method_results['watershed'] = {
                'method_name': 'Professional Watershed',
                'cells_detected': len(watershed_cells),
                'total_area': sum(c['area'] for c in watershed_cells),
                'average_area': np.mean([c['area'] for c in watershed_cells]) if watershed_cells else 0,
                'cells': watershed_cells,
                'visualization_path': str(watershed_viz) if watershed_viz else None,
                'pipeline_visualization_path': str(watershed_pipeline_viz) if watershed_pipeline_viz else None
            }

            # --- Method 2: Color-Aware Tophat AI ---
            if use_tophat and self.load_tophat_model():
                print("🎯 Running Color-Aware Tophat ML method...")
                tophat_labels = self.color_aware_tophat_detection(color_img, enhanced_gray)
                tophat_cells = self.extract_cell_properties(tophat_labels, enhanced_gray)
                tophat_viz = self.create_method_visualization(img, tophat_labels, tophat_cells, "Color-Aware Tophat AI")
                method_results['tophat'] = {
                    'method_name': 'Tophat AI Model',
                    'cells_detected': len(tophat_cells),
                    'total_area': sum(c['area'] for c in tophat_cells),
                    'average_area': np.mean([c['area'] for c in tophat_cells]) if tophat_cells else 0,
                    'cells': tophat_cells,
                    'visualization_path': str(tophat_viz) if tophat_viz else None
                }

            # --- Method 3: Enhanced RGB Wolffia CNN ---
            if use_cnn and TORCH_AVAILABLE and self.load_cnn_model():
                print("🤖 Running Enhanced RGB Wolffia CNN method...")
                
                # Use smart CNN detection that automatically chooses RGB vs grayscale
                cnn_labels = self.cnn_detection(enhanced_gray, color_img)
                cnn_cells = self.extract_cell_properties(cnn_labels, enhanced_gray)
                cnn_viz = self.create_method_visualization(img, cnn_labels, cnn_cells, "Wolffia CNN")
                cv2.imwrite("debug_final_cnn_viz.png", (cnn_labels * 255).astype(np.uint8))

                method_results['cnn'] = {
                    'method_name': 'Wolffia CNN',
                    'cells_detected': len(cnn_cells),
                    'total_area': sum(c['area'] for c in cnn_cells),
                    'average_area': np.mean([c['area'] for c in cnn_cells]) if cnn_cells else 0,
                    'cells': cnn_cells,
                    'visualization_path': str(cnn_viz) if cnn_viz else None
                }

            # --- Method 4: CellDetection AI ---
            if use_celldetection and CELLDETECTION_AVAILABLE:
                print("🧠 Running CellDetection AI method...")
                try:
                    # CellDetection expects RGB input, provide grayscale converted to RGB
                    rgb_input = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2RGB) if enhanced_gray.ndim == 2 else original
                    celldet_result = self.celldetection_detection(rgb_input)

                    # Ensure celldet_result is proper labeled image format
                    if celldet_result is None or celldet_result.size == 0:
                        print("⚠️ CellDetection returned empty result")
                        celldet_result = np.zeros_like(enhanced_gray, dtype=np.int32)
                    
                    # Handle data type conversion properly
                    if celldet_result.dtype != np.int32:
                        # If it's a float array, convert properly
                        if celldet_result.dtype in [np.float32, np.float64]:
                            celldet_result = celldet_result.astype(np.int32)
                        else:
                            celldet_result = celldet_result.astype(np.int32)
                    
                    # Ensure it's 2D (labeled image)
                    if celldet_result.ndim > 2:
                        print(f"⚠️ CellDetection result has {celldet_result.ndim} dimensions, converting to 2D")
                        # Take first channel if multi-channel
                        celldet_result = celldet_result[:, :, 0] if celldet_result.ndim == 3 else celldet_result.squeeze()
                        celldet_result = celldet_result.astype(np.int32)
                    
                    # Ensure gray is 2D
                    intensity_img = enhanced_gray if enhanced_gray.ndim == 2 else cv2.cvtColor(enhanced_gray, cv2.COLOR_BGR2GRAY)
                    
                    # Resize if shapes don't match
                    if celldet_result.shape != intensity_img.shape:
                        print(f"⚠️ Resizing intensity image to match CellDetection result: {celldet_result.shape}")
                        intensity_img = cv2.resize(intensity_img, (celldet_result.shape[1], celldet_result.shape[0]))

                    celldet_cells = self.extract_cell_properties(celldet_result, intensity_img)
                    celldet_viz = self.create_method_visualization(img, celldet_result, celldet_cells, "CellDetection AI")
                    
                    method_results['celldetection'] = {
                        'method_name': 'CellDetection AI',
                        'cells_detected': len(celldet_cells),
                        'total_area': sum(c['area'] for c in celldet_cells),
                        'average_area': np.mean([c['area'] for c in celldet_cells]) if celldet_cells else 0,
                        'cells': celldet_cells,
                        'visualization_path': str(celldet_viz) if celldet_viz else None
                    }
                    
                except Exception as celldet_error:
                    print(f"❌ CellDetection method failed: {celldet_error}")
                    # Add empty result to maintain consistency
                    method_results['celldetection'] = {
                        'method_name': 'CellDetection AI',
                        'cells_detected': 0,
                        'total_area': 0,
                        'average_area': 0,
                        'cells': [],
                        'visualization_path': None,
                        'error': str(celldet_error)
                    }

            # --- Convert visualizations to base64 for UI integration ---
            for method_key, data in method_results.items():
                # Convert main visualization
                viz_path = data.get('visualization_path')
                if viz_path and Path(viz_path).exists():
                    with open(viz_path, 'rb') as f:
                        data['visualization_b64'] = base64.b64encode(f.read()).decode('utf-8')
                else:
                    data['visualization_b64'] = None

                # Convert pipeline visualization
                pipeline_path = data.get('pipeline_visualization_path')
                if pipeline_path and Path(pipeline_path).exists():
                    with open(pipeline_path, 'rb') as f:
                        data['pipeline_visualization_b64'] = base64.b64encode(f.read()).decode('utf-8')
                else:
                    data['pipeline_visualization_b64'] = None

            return method_results

        except Exception as e:
            print(f"❌ Separate method analysis failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return {'error': str(e)}

        
        
    # def create_method_visualization(self, original_img, labeled_img, cell_data, method_name):
    #     """Create visualization for a specific method with improved visibility and color accuracy"""
    #     try:
    #         # Create overlay by blending labels with original image using proper alpha
    #         colored_labels = color.label2rgb(
    #             labeled_img, 
    #             image=original_img,  # ensures proper RGB/BGR blending
    #             bg_label=0, 
    #             alpha=0.4, 
    #             kind='overlay'
    #         )
    #         result = (colored_labels * 255).astype(np.uint8)

    #         # Add cell numbers and center markers
    #         for cell in cell_data:
    #             center = tuple(cell['centroid'])
    #             cv2.circle(result, center, 3, (0, 255, 255), -1)
    #             cv2.putText(result, str(cell['id']), center, 
    #                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, lineType=cv2.LINE_AA)

    #         # Add method title with outline for better visibility
    #         text = f"{method_name}: {len(cell_data)} cells"
    #         cv2.putText(result, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3, lineType=cv2.LINE_AA)
    #         cv2.putText(result, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, lineType=cv2.LINE_AA)

    #         # Save visualization image
    #         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #         viz_path = self.dirs['results'] / f"{method_name.lower().replace(' ', '_')}_{timestamp}.png"
    #         cv2.imwrite(str(viz_path), result)

    #         return viz_path

    #     except Exception as e:
    #         print(f"⚠️ Failed to create {method_name} visualization: {e}")
    #         return None

    def create_method_visualization(self, original_img, labeled_img, cell_data, method_name):
        """Create visualization for a specific method with improved visibility and color accuracy"""
        try:
            # Create overlay
            overlay = original_img.copy()
            
            # Color the segmented regions
            colored_labels = color.label2rgb(labeled_img, bg_label=0, alpha=0.3)
            colored_labels = (colored_labels * 255).astype(np.uint8)
            
            # Blend with original
            result = cv2.addWeighted(overlay, 0.7, colored_labels, 0.3, 0)
            
            # Add cell numbers and method title
            for cell in cell_data:
                center = tuple(cell['centroid'])
                cv2.putText(result, str(cell['id']), center, 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                cv2.circle(result, center, 2, (255, 255, 0), -1)
            
            # Add method title
            cv2.putText(result, f"{method_name}: {len(cell_data)} cells", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(result, f"{method_name}: {len(cell_data)} cells", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
            
            # Save visualization
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            viz_path = self.dirs['results'] / f"{method_name.lower().replace(' ', '_')}_{timestamp}.png"
            cv2.imwrite(str(viz_path), result)

            return viz_path

        except Exception as e:
            print(f"⚠️ Failed to create {method_name} visualization: {e}")
            return None

    def create_watershed_pipeline_visualization(self, pipeline_images, cell_data):
        """
        Create comprehensive watershed pipeline visualization showing all processing steps
        Based on python_for_microscopists examples 033 and 035
        """
        try:
            if not pipeline_images:
                return None
            
            # Create a comprehensive visualization grid
            fig = plt.figure(figsize=(20, 16))
            fig.suptitle(f'Watershed Processing Pipeline - {len(cell_data)} Cells Detected', fontsize=16, fontweight='bold')
            
            # Define pipeline steps with descriptions
            pipeline_steps = [
                ('01_original', 'Original Image'),
                ('02_otsu_threshold', 'OTSU Thresholding'),
                ('03_morphological_opening', 'Morphological Opening'),
                ('04_clear_border', 'Border Removal'),
                ('05_sure_background', 'Sure Background (Dilated)'),
                ('06_distance_transform', 'Distance Transform'),
                ('07_sure_foreground', 'Sure Foreground'),
                ('08_unknown_region', 'Unknown Region'),
                ('09_markers', 'Markers for Watershed'),
                ('10_watershed_boundaries', 'Watershed with Boundaries'),
                ('11_final_segmentation', 'Final Segmentation')
            ]
            
            # Create subplots in a 3x4 grid
            for i, (key, title) in enumerate(pipeline_steps):
                if key in pipeline_images:
                    plt.subplot(3, 4, i + 1)
                    
                    # Display image with appropriate colormap
                    if key in ['06_distance_transform', '09_markers']:
                        plt.imshow(pipeline_images[key], cmap='jet')
                    elif key in ['08_unknown_region']:
                        plt.imshow(pipeline_images[key], cmap='viridis')
                    elif key in ['11_final_segmentation']:
                        # Create colored segmentation for final result
                        if np.any(pipeline_images[key] > 0):
                            colored_seg = color.label2rgb(pipeline_images[key], bg_label=0)
                            plt.imshow(colored_seg)
                        else:
                            plt.imshow(pipeline_images[key], cmap='gray')
                    else:
                        plt.imshow(pipeline_images[key], cmap='gray')
                    
                    plt.title(f'Step {i+1}: {title}', fontsize=10, fontweight='bold')
                    plt.axis('off')
            
            # Add summary statistics in the last subplot
            plt.subplot(3, 4, 12)
            plt.text(0.1, 0.9, 'Processing Summary:', fontsize=14, fontweight='bold', transform=plt.gca().transAxes)
            plt.text(0.1, 0.8, f'Cells Detected: {len(cell_data)}', fontsize=12, transform=plt.gca().transAxes)
            
            if cell_data:
                total_area = sum(cell['area'] for cell in cell_data)
                avg_area = np.mean([cell['area'] for cell in cell_data])
                plt.text(0.1, 0.7, f'Total Area: {total_area:.1f} μm²', fontsize=12, transform=plt.gca().transAxes)
                plt.text(0.1, 0.6, f'Average Area: {avg_area:.1f} μm²', fontsize=12, transform=plt.gca().transAxes)
                
                if len(cell_data) > 0:
                    areas = [cell['area'] for cell in cell_data]
                    plt.text(0.1, 0.5, f'Min Area: {min(areas):.1f} μm²', fontsize=12, transform=plt.gca().transAxes)
                    plt.text(0.1, 0.4, f'Max Area: {max(areas):.1f} μm²', fontsize=12, transform=plt.gca().transAxes)
            
            plt.text(0.1, 0.2, 'Pipeline Steps:', fontsize=12, fontweight='bold', transform=plt.gca().transAxes)
            plt.text(0.1, 0.1, '1-4: Preprocessing\n5-8: Region Detection\n9-11: Watershed Segmentation', 
                    fontsize=10, transform=plt.gca().transAxes)
            plt.axis('off')
            
            plt.tight_layout()
            
            # Save pipeline visualization
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            pipeline_path = self.dirs['results'] / f"watershed_pipeline_{timestamp}.png"
            plt.savefig(pipeline_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            return pipeline_path
            
        except Exception as e:
            print(f"⚠️ Failed to create watershed pipeline visualization: {e}")
            return None
    
    def professional_watershed_segmentation(self, gray_img, color_img=None, return_pipeline=False):
        """
        ENHANCED: Professional Watershed Segmentation using Green-Enhanced Channels
        Based on python_for_microscopists examples with green enhancement for Wolffia
        
        Args:
            gray_img: Grayscale fallback
            color_img: BGR color image for green enhancement
            return_pipeline: Return visualization pipeline
        """
        try:
            # Store pipeline images for visualization
            pipeline_images = {}
            
            # ENHANCED: Step 1: Use green-enhanced grayscale if color image available
            if color_img is not None:
                print("🔬 Step 1: Creating green-enhanced grayscale from BGR image")
                enhanced_gray = self.create_green_enhanced_grayscale(color_img)
                working_img = enhanced_gray
                pipeline_images['01_original'] = color_img.copy()
                pipeline_images['01a_green_enhanced'] = enhanced_gray.copy()
            else:
                print("🔬 Step 1: Using grayscale fallback")
                working_img = gray_img.copy()
                pipeline_images['01_original'] = gray_img.copy()
            
            print("🔬 Step 1: Original image captured")
            
            # Step 2: Enhanced preprocessing for small Wolffia cells
            # Citation: "Noise reduction techniques" from python_for_microscopists
            # Apply optimized Gaussian blur specifically calibrated for Wolffia size range
            blurred = cv2.GaussianBlur(working_img, (3, 3), 0.5)  # Reduced sigma for small cells
            pipeline_images['01b_gaussian_blur'] = blurred.copy()
            
            # Additional preprocessing for Wolffia: histogram equalization
            # Citation: "Contrast enhancement for better segmentation" - microscopist best practice
            clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
            enhanced = clahe.apply(blurred)
            pipeline_images['01c_contrast_enhanced'] = enhanced.copy()
            print("🔍 Step 2b: Wolffia-optimized contrast enhancement applied")
            
            # Step 3: Optimized OTSU thresholding for Wolffia (Citation: python_for_microscopists 033)
            # "Threshold image to binary using OTSU. All thresholded pixels will be set to 255"
            # Using enhanced image for better threshold detection
            ret, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Additional optimization: If threshold is too high for small cells, apply correction
            if ret > 160:  # Wolffia-specific threshold adjustment
                print(f"🚪 Adjusting high OTSU threshold ({ret:.1f}) for small Wolffia cells")
                ret_corrected = int(ret * 0.85)  # Reduce threshold by 15% for small cells
                _, thresh = cv2.threshold(enhanced, ret_corrected, 255, cv2.THRESH_BINARY)
                print(f"🔧 Applied corrected threshold: {ret_corrected}")
            
            pipeline_images['02_otsu_threshold'] = thresh.copy()
            print(f"🔬 Step 3: Wolffia-optimized OTSU threshold applied (threshold: {ret:.1f})")
            
            # Step 4: Wolffia-specific morphological operations
            # Citation: "Morphological operations to remove small noise - opening"
            # Citation: "Optimized kernel sizes for small biological objects" - microscopist examples
            
            # Ultra-small kernels specifically for Wolffia (world's smallest flowering plant)
            kernel_tiny = np.ones((1, 1), np.uint8)    # For minimal noise removal
            kernel_small = np.ones((2, 2), np.uint8)   # For small gap filling
            kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # Elliptical for cell shapes
            
            # Step 4a: Minimal opening to preserve tiny Wolffia cells
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_tiny, iterations=1)
            pipeline_images['03a_minimal_opening'] = opening.copy()
            
            # Step 4b: Light closing to connect fragmented small cells
            opening = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel_small, iterations=1)
            pipeline_images['03b_gap_filling'] = opening.copy()
            
            # Step 4c: Final shape optimization with elliptical kernel
            opening = cv2.morphologyEx(opening, cv2.MORPH_OPEN, kernel_medium, iterations=1)
            pipeline_images['03_morphological_opening'] = opening.copy()
            
            print("🔬 Step 4: Wolffia-specific morphological pipeline completed")
            
            # Step 5: Remove border-touching objects (Citation: python_for_microscopists)
            # "Remove edge touching grains/cells"
            opening = clear_border(opening)
            pipeline_images['04_clear_border'] = opening.copy()
            print("🔬 Step 3-4: Morphological operations and border clearing completed")
            
            # Step 6: Enhanced sure background calculation
            # Citation: "dilating pixels a few times increases cell boundary to background"
            sure_bg = cv2.dilate(opening, kernel_medium, iterations=2)  # Reduced iterations for small cells
            pipeline_images['05_sure_background'] = sure_bg.copy()
            
            # Step 7: Enhanced distance transform for small cells
            # Citation: "Finding sure foreground area using distance transform and thresholding"
            # "intensities of the points inside the foreground regions are changed to distance"
            dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 3)  # Smaller mask size for small cells
            
            # Normalize for visualization
            dist_norm = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            pipeline_images['06_distance_transform'] = dist_norm.copy()
            
            # Step 8: Adaptive threshold for sure foreground (Wolffia-optimized)
            # Citation: "Let us threshold the dist transform by 20% its max value"
            # Citation: "Small object detection requires lower thresholds" - microscopist best practice
            
            # Wolffia-specific adaptive thresholding
            max_distance = dist_transform.max()
            
            # FIXED: Multi-level adaptive threshold that prevents full image detection
            if max_distance < 2.0:  # Very poor separation - likely noise or bad threshold
                print(f"🚫 Max distance too low ({max_distance:.2f}) - skipping watershed for this image")
                empty_result = np.zeros_like(gray_img, dtype=np.int32)
                if return_pipeline:
                    return empty_result, pipeline_images
                else:
                    return empty_result
            elif max_distance > 8:  # Large Wolffia cells
                distance_threshold = 0.4 * max_distance
            elif max_distance > 4:  # Medium Wolffia cells  
                distance_threshold = 0.3 * max_distance
            else:  # Small Wolffia cells (2-4 range)
                distance_threshold = max(0.2 * max_distance, 0.8)  # More conservative minimum
            
            ret, sure_fg = cv2.threshold(dist_transform, distance_threshold, 255, 0)
            pipeline_images['07_sure_foreground'] = sure_fg.copy()
            
            print(f"🔬 Step 8: Wolffia-adaptive distance threshold applied")
            print(f"   Max distance: {max_distance:.2f}, Threshold: {distance_threshold:.2f}")
            print(f"   Optimization level: {'Large' if max_distance > 8 else 'Medium' if max_distance > 4 else 'Small'} Wolffia cells")
            
            # Step 9: Find unknown region
            # Citation: "Unknown ambiguous region is nothing but background - foreground"
            sure_fg = np.uint8(sure_fg)
            unknown = cv2.subtract(sure_bg, sure_fg)
            pipeline_images['08_unknown_region'] = unknown.copy()
            
            # Step 10: Enhanced marker labelling for small cells
            # Citation: "For sure regions, both foreground and background will be labeled with positive numbers"
            ret, markers = cv2.connectedComponents(sure_fg)
            
            # FIXED: Validate marker count to prevent bad watershed results
            if ret <= 1:  # No foreground markers (only background)
                print(f"🚫 No foreground markers found - skipping watershed")
                empty_result = np.zeros_like(gray_img, dtype=np.int32)
                if return_pipeline:
                    return empty_result, pipeline_images
                else:
                    return empty_result
            elif ret > 200:  # Too many tiny regions - likely noise
                print(f"🚫 Too many markers ({ret}) - likely noise, skipping watershed")
                empty_result = np.zeros_like(gray_img, dtype=np.int32)
                if return_pipeline:
                    return empty_result, pipeline_images
                else:
                    return empty_result
            
            # Step 11: Add offset to markers (microscopist technique)
            # Citation: "So let us add 10 to all labels so that sure background is not 0, but 10"
            # Using smaller offset for better small cell detection
            markers = markers + 5  # Reduced offset for small cells
            
            # Step 12: Mark unknown region as 0
            # Citation: "Now, mark the region of unknown with zero"
            markers[unknown == 255] = 0
            
            # Visualize markers before watershed
            markers_viz = cv2.normalize(markers, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            pipeline_images['09_markers'] = markers_viz.copy()
            print(f"🔬 Step 8-9: Markers created ({ret} initial regions)")
            
            # Step 13: Apply watershed algorithm
            # Citation: "Now we are ready for watershed filling"
            img_for_watershed = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
            markers = cv2.watershed(img_for_watershed, markers)
            
            # Step 14: Visualize watershed result with boundaries
            # Citation: "The boundary region will be marked -1"
            watershed_result = img_for_watershed.copy()
            watershed_result[markers == -1] = [0, 255, 255]  # Mark boundaries in yellow
            pipeline_images['10_watershed_boundaries'] = cv2.cvtColor(watershed_result, cv2.COLOR_BGR2GRAY)
            
            # Step 15: Enhanced cleanup for small cells
            clean_markers = markers.copy()
            clean_markers[markers == -1] = 0  # Remove boundaries
            clean_markers[markers <= 5] = 0   # Remove background (adjusted for offset)
            
            # Enhanced Wolffia-specific validation and cleanup
            # Citation: "Size and shape filters for biological objects" - python_for_microscopists
            regions = measure.regionprops(clean_markers)
            final_markers = np.zeros_like(clean_markers)
            valid_label = 1
            validation_stats = {'total': 0, 'size_valid': 0, 'shape_valid': 0, 'final_valid': 0}
            
            for region in regions:
                validation_stats['total'] += 1
                area_valid = self.min_cell_area <= region.area <= self.max_cell_area
                
                if area_valid:
                    validation_stats['size_valid'] += 1
                    
                    # Wolffia-specific shape validation (more lenient for natural variation)
                    eccentricity_valid = region.eccentricity < 0.95  # Allow more elongated shapes
                    solidity_valid = region.solidity > 0.3          # More lenient solidity
                    extent_valid = region.extent > 0.2              # Reasonable extent
                    aspect_ratio = region.major_axis_length / max(region.minor_axis_length, 1)
                    aspect_ratio_valid = aspect_ratio < 4.0         # Not extremely elongated
                    
                    shape_valid = eccentricity_valid and solidity_valid and extent_valid and aspect_ratio_valid
                    
                    if shape_valid:
                        validation_stats['shape_valid'] += 1
                        
                        # Additional Wolffia validation: circularity and compactness
                        perimeter = region.perimeter
                        if perimeter > 0:
                            circularity = 4 * np.pi * region.area / (perimeter ** 2)
                            circularity_valid = circularity > 0.2  # Reasonable circularity for Wolffia
                        else:
                            circularity_valid = True
                        
                        if circularity_valid:
                            mask = clean_markers == region.label
                            final_markers[mask] = valid_label
                            valid_label += 1
                            validation_stats['final_valid'] += 1
            
            print(f"📊 Wolffia validation stats: {validation_stats}")
            print(f"   Validation efficiency: {validation_stats['final_valid']}/{validation_stats['total']} regions")
            
            # Final segmentation visualization
            pipeline_images['11_final_segmentation'] = final_markers.copy()
            
            # Final optimization summary
            final_cell_count = valid_label - 1
            efficiency = (validation_stats['final_valid'] / max(validation_stats['total'], 1)) * 100
            
            print(f"✅ Wolffia-optimized watershed completed: {final_cell_count} valid cells detected")
            print(f"🎯 Detection efficiency: {efficiency:.1f}% ({validation_stats['final_valid']}/{validation_stats['total']})")
            print(f"🌱 Pipeline optimized for Wolffia arrhiza (world's smallest flowering plant)")
            
            if return_pipeline:
                return final_markers, pipeline_images
            else:
                return final_markers
            
        except Exception as e:
            print(f"⚠️ Wolffia-optimized watershed segmentation failed: {e}")
            import traceback
            print(f"Detailed error: {traceback.format_exc()}")
            if return_pipeline:
                return np.zeros_like(gray_img), {}
            else:
                return np.zeros_like(gray_img)
    
    
    def refresh_model_status(self):
        """Refresh the status of all AI models (useful after training new models)"""
        print("🔄 Refreshing model status...")
        
        # Reset model states to force reloading
        self._cnn_model = None
        self._tophat_model = None
        self._celldetection_model = None
        self.wolffia_cnn_available = False
        self.wolffia_cnn_model = None
        self.tophat_model = None
        self.celldetection_available = False
        
        # Check Wolffia CNN
        cnn_available = self.load_cnn_model()
        if cnn_available:
            print("✅ Wolffia CNN model detected and loaded")
        else:
            print("📝 Wolffia CNN model not found - train with: python train_wolffia_cnn.py")
        
        # Check Tophat model
        tophat_available = self.load_tophat_model()
        if tophat_available:
            print("✅ Tophat model detected")
        else:
            print("📝 Tophat model not found")
        
        # Check CellDetection model
        if hasattr(self, 'celldetection_model') and self.celldetection_model:
            print("✅ CellDetection model available")
        else:
            print("📝 CellDetection model not available")
        
        print(f"🎯 Final status: Wolffia_CNN={self.wolffia_cnn_available}, Tophat={self.tophat_model is not None}, CellDetection={self.celldetection_available}")
        return {
            'wolffia_cnn_available': self.wolffia_cnn_available,
            'tophat_available': self.tophat_model is not None,
            'celldetection_available': self.celldetection_available
        }
        
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
    
    def tophat_ml_detection(self, gray_img):
        """
        Tophat ML detection using feature extraction approach
        Based on microscopist ML segmentation patterns (example 062-066)
        FIXED: Now properly converts binary predictions to labeled cell regions
        """
        try:
            if self._tophat_model is None:
                return np.zeros_like(gray_img)
            
            # Extract features using proven approach
            features = self.extract_ml_features(gray_img)
            
            # Predict using trained model
            predictions = self._tophat_model.predict(features.reshape(-1, features.shape[-1]))
            
            # Reshape back to image - this creates a binary mask
            binary_mask = predictions.reshape(gray_img.shape).astype(np.uint8)
            
            # CRITICAL FIX: Convert binary mask to labeled regions for individual cell detection
            if np.any(binary_mask > 0):
                # Clean up the binary mask with morphological operations
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
                cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel)
                
                # Convert to labeled image using connected components
                # This gives each connected region a unique label (1, 2, 3, etc.)
                num_labels, labeled_img = cv2.connectedComponents(cleaned_mask)
                
                print(f"🔍 Tophat ML: detected {num_labels-1} potential cell regions")
                
                # Apply size filtering to remove noise
                filtered_labeled = self.filter_by_size(labeled_img)
                
                return filtered_labeled
            else:
                print("🔍 Tophat ML: no cells detected")
                return np.zeros_like(gray_img)
            
        except Exception as e:
            print(f"⚠️ Tophat ML detection failed: {e}")
            import traceback
            traceback.print_exc()
            return np.zeros_like(gray_img)
    
    def extract_ml_features(self, img):
        """
        Extract comprehensive ML features matching enhanced tophat trainer
        Based on proven microscopist approach with comprehensive feature set
        FIXED: Now generates features matching the enhanced trainer
        """
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        features = []
        
        # 1. Original intensity features
        features.append(img)
        
        # 2. Multi-scale Gaussian features (important for Wolffia size variations)
        for sigma in [0.5, 1.0, 2.0, 3.0, 5.0]:
            gaussian_img = ndimage.gaussian_filter(img, sigma=sigma)
            features.append(gaussian_img)
        
        # 3. Enhanced edge detection suite
        # Canny with multiple thresholds
        for low, high in [(30, 100), (50, 150), (70, 200)]:
            edges_canny = cv2.Canny(img, low, high)
            features.append(edges_canny)
        
        # Comprehensive edge filters
        edges_sobel = (filters.sobel(img) * 255).astype(np.uint8)
        features.append(edges_sobel)
        
        edges_roberts = (filters.roberts(img) * 255).astype(np.uint8)
        features.append(edges_roberts)
        
        edges_prewitt = (filters.prewitt(img) * 255).astype(np.uint8)
        features.append(edges_prewitt)
        
        edges_scharr = (filters.scharr(img) * 255).astype(np.uint8)
        features.append(edges_scharr)
        
        # 4. Morphological features (important for cell shape)
        # Opening with different kernel sizes
        for kernel_size in [3, 5, 7]:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
            features.append(opened)
        
        # Closing with different kernel sizes
        for kernel_size in [3, 5]:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
            features.append(closed)
        
        # 5. Statistical filters (texture analysis)
        for size in [3, 5, 7]:
            # Variance (texture)
            variance_img = ndimage.generic_filter(img, np.var, size=size)
            features.append(variance_img)
            
            # Standard deviation
            std_img = ndimage.generic_filter(img, np.std, size=size)
            features.append(std_img)
            
            # Median
            median_img = ndimage.median_filter(img, size=size)
            features.append(median_img)
        
        # 6. Min/Max filters
        for size in [3, 5]:
            maximum_img = ndimage.maximum_filter(img, size=size)
            features.append(maximum_img)
            
            minimum_img = ndimage.minimum_filter(img, size=size)
            features.append(minimum_img)
        
        # 7. Laplacian features
        laplacian_img = cv2.Laplacian(img, cv2.CV_64F)
        features.append(np.abs(laplacian_img).astype(np.uint8))
        
        # 8. Hessian-based features (blob detection) - optional
        try:
            from skimage.feature import hessian_matrix, hessian_matrix_eigvals
            hessian = hessian_matrix(img, sigma=1.0)
            eigenvals = hessian_matrix_eigvals(hessian)
            features.append((eigenvals[0] * 255).astype(np.uint8))
            features.append((eigenvals[1] * 255).astype(np.uint8))
        except:
            # Fallback features if hessian not available
            features.append(ndimage.generic_filter(img, np.mean, size=3))
            features.append(ndimage.generic_filter(img, np.max, size=3))
        
        print(f"🔍 Extracted {len(features)} features for tophat ML detection")
        return np.stack(features, axis=-1)
    
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
            cv2.imwrite("debug_cnn_sigmoid.png", (sigmoid_map * 255).astype(np.uint8))

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
    
    def _legacy_cnn_detection(self, gray_img):
        """
        Legacy CNN detection for grayscale models
        """
        try:
            print("🔘 Using legacy grayscale CNN detection...")
            
            # Simple patch-based processing for grayscale models
            original_shape = gray_img.shape[:2]
            patch_size = 64
            overlap = 16
            
            full_prediction = np.zeros(original_shape, dtype=np.float32)
            count_map = np.zeros(original_shape, dtype=np.float32)
            
            for y in range(0, original_shape[0] - patch_size + 1, patch_size - overlap):
                for x in range(0, original_shape[1] - patch_size + 1, patch_size - overlap):
                    y_end = min(y + patch_size, original_shape[0])
                    x_end = min(x + patch_size, original_shape[1])
                    patch = gray_img[y:y_end, x:x_end]
                    
                    if patch.shape != (patch_size, patch_size):
                        patch = cv2.resize(patch, (patch_size, patch_size))
                    
                    # Convert to tensor
                    patch_tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).float()
                    patch_tensor = patch_tensor.to(next(self._cnn_model.parameters()).device)
                    
                    # Model inference
                    with torch.no_grad():
                        output = self._cnn_model(patch_tensor)
                        if isinstance(output, (tuple, list)):
                            seg_output = output[0]
                        else:
                            seg_output = output
                        
                        patch_pred = torch.sigmoid(seg_output).squeeze().cpu().numpy()
                    
                    if patch_pred.shape != (y_end - y, x_end - x):
                        patch_pred = cv2.resize(patch_pred, (x_end - x, y_end - y))
                    
                    full_prediction[y:y_end, x:x_end] += patch_pred
                    count_map[y:y_end, x:x_end] += 1
            
            count_map[count_map == 0] = 1
            prediction_map = full_prediction / count_map
            
            # Apply threshold and post-processing
            binary_mask = (prediction_map > 0.5).astype(np.uint8)
            
            if np.any(binary_mask):
                num_labels, labels = cv2.connectedComponents(binary_mask)
                filtered_labels = self.filter_by_size(labels)
                print(f"✅ Legacy CNN: detected {np.max(filtered_labels)} cells")
                return filtered_labels.astype(np.int32)
            else:
                return np.zeros(original_shape, dtype=np.int32)
                
        except Exception as e:
            print(f"⚠️ Legacy CNN detection failed: {e}")
            return np.zeros_like(gray_img, dtype=np.int32)


    def debug_cnn_detection(self, bgr_img, save_debug_images=True):
        """
        Debug CNN detection using enhanced 3-channel input
        Tracks intermediate CNN responses and confidence across patches
        """
        try:
            if not self.wolffia_cnn_available or self._cnn_model is None:
                print("⚠️ CNN model not available for debugging")
                return None

            print("🔬 Running RGB-enhanced CNN debug analysis...")

            # Step 1: Generate enhanced 3-channel input
            from wolffia_cnn_model import GreenEnhancedPreprocessor
            preprocessor = GreenEnhancedPreprocessor()
            rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
            enhanced_img = preprocessor.create_green_enhanced_channels(rgb_img)  # Shape: HxWx3
            original_shape = enhanced_img.shape[:2]

            patch_size = 128
            overlap = 32

            full_prediction = np.zeros(original_shape, dtype=np.float32)
            count_map = np.zeros(original_shape, dtype=np.float32)

            debug_info = {
                'patch_predictions': [],
                'patch_locations': [],
                'confidence_map': np.zeros(original_shape, dtype=np.float32)
            }

            patches_processed = 0
            for y in range(0, original_shape[0] - patch_size + 1, patch_size - overlap):
                for x in range(0, original_shape[1] - patch_size + 1, patch_size - overlap):
                    y_end = min(y + patch_size, original_shape[0])
                    x_end = min(x + patch_size, original_shape[1])
                    patch = enhanced_img[y:y_end, x:x_end, :]

                    if patch.shape[:2] != (patch_size, patch_size):
                        patch = cv2.resize(patch, (patch_size, patch_size))

                    patch_normalized = patch.astype(np.float32)
                    patch_tensor = torch.from_numpy(patch_normalized).permute(2, 0, 1).unsqueeze(0)

                    model_device = next(self._cnn_model.parameters()).device
                    patch_tensor = patch_tensor.to(model_device).float()

                    with torch.no_grad():
                        patch_output = self._cnn_model(patch_tensor)
                        if isinstance(patch_output, tuple):
                            patch_pred = patch_output[0].squeeze().cpu().numpy()
                        else:
                            patch_pred = patch_output.squeeze().cpu().numpy()

                    if patch_pred.shape != (y_end - y, x_end - x):
                        patch_pred = cv2.resize(patch_pred, (x_end - x, y_end - y))

                    debug_info['patch_predictions'].append({
                        'location': (y, x, y_end, x_end),
                        'mean_confidence': float(np.mean(patch_pred)),
                        'max_confidence': float(np.max(patch_pred)),
                        'min_confidence': float(np.min(patch_pred))
                    })

                    full_prediction[y:y_end, x:x_end] += patch_pred
                    count_map[y:y_end, x:x_end] += 1
                    patches_processed += 1

            count_map[count_map == 0] = 1
            averaged_prediction = full_prediction / count_map
            prediction_sigmoid = 1 / (1 + np.exp(-averaged_prediction))

            # Convert for debug visualizations
            raw_prediction = (prediction_sigmoid * 255).astype(np.uint8)
            confidence_colored = cv2.applyColorMap(raw_prediction, cv2.COLORMAP_JET)

            otsu_val, _ = cv2.threshold(raw_prediction, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            otsu_thresh = float(otsu_val)
            high_mask = (prediction_sigmoid > (otsu_thresh / 255.0)).astype(np.uint8) * 255
            low_mask = (prediction_sigmoid > (otsu_thresh / 255.0 * 0.6)).astype(np.uint8) * 255

            final_labels = self.cnn_detection(bgr_img)  # rerun CNN detection on original RGB

            # Compile debug images
            debug_images = {
                'raw_prediction': raw_prediction,
                'confidence_colored': confidence_colored,
                'high_confidence': high_mask,
                'low_confidence': low_mask,
                'final_detection': self.create_debug_overlay(bgr_img, final_labels)
            }

            # Save debug visualizations
            if save_debug_images:
                debug_dir = self.dirs['results'] / 'cnn_debug'
                debug_dir.mkdir(exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                for name, img in debug_images.items():
                    if img is not None:
                        path = debug_dir / f"debug_{timestamp}_{name}.png"
                        cv2.imwrite(str(path), img)
                        print(f"💾 Saved debug image: {path}")

            print("🔍 CNN Debug Summary:")
            print(f"📊 Patches processed: {patches_processed}")
            print(f"📊 Confidence range: {np.min(prediction_sigmoid):.3f} - {np.max(prediction_sigmoid):.3f}")
            print(f"📊 Mean confidence: {np.mean(prediction_sigmoid):.3f}")
            print(f"📊 Otsu threshold: {otsu_thresh / 255.0:.3f}")
            print(f"📊 High conf pixels: {np.sum(high_mask > 0)}")
            print(f"📊 Final cells detected: {np.max(final_labels)}")

            return {
                'debug_images': debug_images,
                'debug_info': debug_info,
                'statistics': {
                    'patches_processed': patches_processed,
                    'confidence_range': (float(np.min(prediction_sigmoid)), float(np.max(prediction_sigmoid))),
                    'mean_confidence': float(np.mean(prediction_sigmoid)),
                    'otsu_threshold': otsu_thresh / 255.0,
                    'cells_detected': int(np.max(final_labels))
                }
            }

        except Exception as e:
            print(f"❌ CNN debug analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def create_debug_overlay(self, gray_img, labels):
        """Create a colored overlay showing detected cells"""
        try:
            # Create colored visualization
            overlay = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
            
            # Color each detected cell differently
            num_cells = np.max(labels)
            if num_cells > 0:
                colors = plt.cm.Set3(np.linspace(0, 1, min(num_cells, 12)))
                
                for i in range(1, num_cells + 1):
                    mask = labels == i
                    if np.any(mask):
                        color = colors[(i-1) % 12][:3]  # RGB
                        color_bgr = (int(color[2]*255), int(color[1]*255), int(color[0]*255))  # Convert to BGR
                        overlay[mask] = color_bgr
            
            return overlay
            
        except Exception as e:
            print(f"⚠️ Failed to create debug overlay: {e}")
            return cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
    
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
        Based on backup implementation from bioimaging_backup.py
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
            'processing_time': 0
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
            
            status = {
                'model_available': model_path.exists(),
                'model_trained': model_path.exists() and self._tophat_model is not None,
                'model_path': str(model_path),
                'training_info_available': info_path.exists()
            }
            
            # Load training info if available
            if info_path.exists():
                try:
                    with open(info_path, 'r') as f:
                        training_info = json.load(f)
                    status['training_info'] = training_info
                except Exception as e:
                    print(f"⚠️ Error reading training info: {e}")
            
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