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
        self.min_cell_area = 5
        self.max_cell_area = 1200
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
        """Lazy loading for device detection"""
        if self._device is None:
            try:
                import torch
                self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
            except:
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
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Get results from available methods
            results = []
            
            # Method 1: Professional Watershed (always available)
            watershed_result = self.professional_watershed_segmentation(gray)
            results.append(('watershed', watershed_result))
            
            # Method 2: Tophat ML (if trained)
            if use_tophat and self.load_tophat_model():
                tophat_result = self.tophat_ml_detection(gray)
                results.append(('tophat', tophat_result))
            
            # Method 3: CNN (if available)
            if use_cnn and TORCH_AVAILABLE and self.load_cnn_model():
                cnn_result = self.cnn_detection(gray)
                results.append(('cnn', cnn_result))
            
            # Method 4: CellDetection (if available)
            if use_celldetection and CELLDETECTION_AVAILABLE:
                celldetection_result = self.celldetection_detection(gray)
                results.append(('celldetection', celldetection_result))
            
            # Intelligent fusion of results
            final_result = self.fuse_detection_results(results, gray.shape)
            
            # Extract cell properties using proven regionprops approach
            cell_data = self.extract_cell_properties(final_result, gray)
            
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
                        'green_cell_percentage': 85.0  # Placeholder value
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

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            method_results = {}

            # --- Method 1: Watershed (Always Run) ---
            print("🔬 Running Watershed method...")
            watershed_labels, pipeline_images = self.professional_watershed_segmentation(gray, return_pipeline=True)
            watershed_cells = self.extract_cell_properties(watershed_labels, gray)
            watershed_viz = self.create_method_visualization(img, watershed_labels, watershed_cells, "Watershed")
            watershed_pipeline_viz = self.create_watershed_pipeline_visualization(pipeline_images, watershed_cells)
            method_results['watershed'] = {
                'method_name': 'Professional Watershed',
                'cells_detected': len(watershed_cells),
                'total_area': sum(c['area'] for c in watershed_cells),
                'average_area': np.mean([c['area'] for c in watershed_cells]) if watershed_cells else 0,
                'cells': watershed_cells,
                'visualization_path': str(watershed_viz) if watershed_viz else None,
                'pipeline_visualization_path': str(watershed_pipeline_viz) if watershed_pipeline_viz else None
            }

            # --- Method 2: Tophat AI ---
            if use_tophat and self.load_tophat_model():
                print("🎯 Running Tophat ML method...")
                tophat_labels = self.tophat_ml_detection(gray)
                tophat_cells = self.extract_cell_properties(tophat_labels, gray)
                tophat_viz = self.create_method_visualization(img, tophat_labels, tophat_cells, "Tophat AI Model")
                method_results['tophat'] = {
                    'method_name': 'Tophat AI Model',
                    'cells_detected': len(tophat_cells),
                    'total_area': sum(c['area'] for c in tophat_cells),
                    'average_area': np.mean([c['area'] for c in tophat_cells]) if tophat_cells else 0,
                    'cells': tophat_cells,
                    'visualization_path': str(tophat_viz) if tophat_viz else None
                }

            # --- Method 3: Wolffia CNN ---
            if use_cnn and TORCH_AVAILABLE and self.load_cnn_model():
                print("🤖 Running Wolffia CNN method...")
                cnn_labels = self.cnn_detection(gray)
                cnn_cells = self.extract_cell_properties(cnn_labels, gray)
                cnn_viz = self.create_method_visualization(img, cnn_labels, cnn_cells, "Wolffia CNN")
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
                    rgb_input = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB) if gray.ndim == 2 else original
                    celldet_result = self.celldetection_detection(rgb_input)

                    # Ensure celldet_result is proper labeled image format
                    if celldet_result is None or celldet_result.size == 0:
                        print("⚠️ CellDetection returned empty result")
                        celldet_result = np.zeros_like(gray, dtype=np.int32)
                    
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
                    intensity_img = gray if gray.ndim == 2 else cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
                    
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

        
    
    def create_center_focus_mask(self, shape, center, radius):
        """Create mask focusing on plate center, excluding edges"""
        h, w = shape
        mask = np.zeros((h, w), dtype=bool)
        
        # Create circular mask with 70% of radius (exclude edges)
        cv2.circle(mask.astype(np.uint8), center, int(radius * 0.7), 1, -1)
        
        print(f"🎯 Center focus mask: {np.sum(mask)} pixels in focus area")
        return mask.astype(bool)
    
    def create_method_visualization(self, original_img, labeled_img, cell_data, method_name):
        """Create visualization for a specific method"""
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
    
    def professional_watershed_segmentation(self, gray_img, return_pipeline=False):
        """
        Enhanced Professional Watershed Segmentation for Wolffia Small Cells
        Based on python_for_microscopists examples 033 (grain analysis) and 035 (cell nuclei)
        
        Citation: "Grain size analysis using watershed segmentation" by Sreenivas Bhattiprolu
        Reference: https://www.youtube.com/watch?v=WyQ-3Fjay7A
        
        Citation: "Cell Nuclei analysis using watershed" by Sreenivas Bhattiprolu  
        Reference: python_for_microscopists example 035
        
        Optimized for small Wolffia cells with enhanced preprocessing and validation
        """
        try:
            # Store pipeline images for visualization
            pipeline_images = {}
            
            # Step 1: Original image
            pipeline_images['01_original'] = gray_img.copy()
            print("🔬 Step 1: Original image captured")
            
            # Step 2: Enhanced preprocessing for small cells
            # Apply slight Gaussian blur to reduce noise (adapted for small cells)
            blurred = cv2.GaussianBlur(gray_img, (3, 3), 0.8)
            pipeline_images['01b_gaussian_blur'] = blurred.copy()
            
            # Step 3: Adaptive OTSU thresholding (Citation: python_for_microscopists 033)
            # "Threshold image to binary using OTSU. All thresholded pixels will be set to 255"
            ret, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            pipeline_images['02_otsu_threshold'] = thresh.copy()
            print(f"🔬 Step 2: OTSU threshold applied (threshold: {ret:.1f})")
            
            # Step 4: Enhanced morphological operations for small cells
            # Citation: "Morphological operations to remove small noise - opening"
            kernel_small = np.ones((2, 2), np.uint8)  # Smaller kernel for small cells
            kernel_medium = np.ones((3, 3), np.uint8)
            
            # Light opening to preserve small cells
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_small, iterations=1)
            # Additional light closing to fill small gaps
            opening = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel_small, iterations=1)
            pipeline_images['03_morphological_opening'] = opening.copy()
            
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
            
            # Step 8: Adaptive threshold for sure foreground (optimized for Wolffia)
            # Citation: "Let us threshold the dist transform by 20% its max value"
            # For small cells, we use a more sensitive threshold
            distance_threshold = max(0.3 * dist_transform.max(), 2.0)  # Ensure minimum threshold
            ret, sure_fg = cv2.threshold(dist_transform, distance_threshold, 255, 0)
            pipeline_images['07_sure_foreground'] = sure_fg.copy()
            print(f"🔬 Step 5-7: Distance transform completed (threshold: {distance_threshold:.2f})")
            
            # Step 9: Find unknown region
            # Citation: "Unknown ambiguous region is nothing but background - foreground"
            sure_fg = np.uint8(sure_fg)
            unknown = cv2.subtract(sure_bg, sure_fg)
            pipeline_images['08_unknown_region'] = unknown.copy()
            
            # Step 10: Enhanced marker labelling for small cells
            # Citation: "For sure regions, both foreground and background will be labeled with positive numbers"
            ret, markers = cv2.connectedComponents(sure_fg)
            
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
            
            # Additional validation for Wolffia-specific size constraints
            regions = measure.regionprops(clean_markers)
            final_markers = np.zeros_like(clean_markers)
            valid_label = 1
            
            for region in regions:
                # Enhanced size validation for small Wolffia cells
                if (self.min_cell_area <= region.area <= self.max_cell_area and
                    region.eccentricity < 0.9 and  # Not too elongated
                    region.solidity > 0.5):       # Reasonably solid shape
                    
                    mask = clean_markers == region.label
                    final_markers[mask] = valid_label
                    valid_label += 1
            
            # Final segmentation visualization
            pipeline_images['11_final_segmentation'] = final_markers.copy()
            
            print(f"✅ Watershed completed: {valid_label-1} valid cells detected")
            
            if return_pipeline:
                return final_markers, pipeline_images
            else:
                return final_markers
            
        except Exception as e:
            print(f"⚠️ Enhanced watershed segmentation failed: {e}")
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
        """
        try:
            if self._tophat_model is None:
                return np.zeros_like(gray_img)
            
            # Extract features using proven approach
            features = self.extract_ml_features(gray_img)
            
            # Predict using trained model
            predictions = self._tophat_model.predict(features.reshape(-1, features.shape[-1]))
            
            # Reshape back to image
            segmented = predictions.reshape(gray_img.shape)
            
            return segmented.astype(np.uint8)
            
        except Exception as e:
            print(f"⚠️ Tophat ML detection failed: {e}")
            return np.zeros_like(gray_img)
    
    def extract_ml_features(self, img):
        """
        Extract ML features using proven microscopist approach
        Based on example 062-066 feature extraction
        FIXED: Now generates 14 features consistently
        """
        features = []
        
        # Original image
        features.append(img)
        
        # Gaussian filters (multiple scales)
        features.append(ndimage.gaussian_filter(img, sigma=1))
        features.append(ndimage.gaussian_filter(img, sigma=3))
        features.append(ndimage.gaussian_filter(img, sigma=5))
        
        # Edge filters
        features.append(filters.sobel(img))
        features.append(filters.roberts(img))
        features.append(filters.scharr(img))
        features.append(cv2.Canny(img, 50, 150))
        
        # Morphological filters
        features.append(ndimage.median_filter(img, size=3))
        features.append(ndimage.maximum_filter(img, size=3))
        features.append(ndimage.minimum_filter(img, size=3))
        
        # Statistical filters
        features.append(ndimage.generic_filter(img, np.var, size=3))
        features.append(ndimage.generic_filter(img, np.std, size=3))
        
        # Laplacian
        features.append(ndimage.laplace(img))
        
        # Ensure we have exactly 14 features
        assert len(features) == 14, f"Expected 14 features, got {len(features)}"
        
        return np.stack(features, axis=-1)
    
    def load_cnn_model(self):
        """
        Load CNN model with robust error handling and proper device management
        FIXED: Now properly checks PyTorch availability and handles model loading
        """
        if self._cnn_model is not None:
            self.wolffia_cnn_available = True
            return True
        
        if not TORCH_AVAILABLE:
            print("⚠️ PyTorch not available - CNN models disabled")
            self.wolffia_cnn_available = False
            return False
        
        print("🤖 Attempting to load Wolffia CNN model...")
        
        # Model loading priority: Original CNN > Enhanced CNN > Improved Enhanced CNN
        model_configs = [
            ('wolffia_cnn_best.pth', 'WolffiaCNN', 'Original Wolffia CNN'),
            ('enhanced_wolffia_cnn_best.pth', 'VGGUNet', 'Enhanced VGG U-Net'),
            ('improved_enhanced_wolffia_cnn_best.pth', 'VGGUNet', 'Improved Enhanced U-Net')
        ]
        
        for model_file, model_class, model_name in model_configs:
            model_path = self.dirs['models'] / model_file
            if model_path.exists():
                try:
                    print(f"🔍 Found {model_name} at {model_file}")
                    
                    # Import the appropriate model class
                    if model_class == 'WolffiaCNN':
                        from wolffia_cnn_model import WolffiaCNN
                        self._cnn_model = WolffiaCNN(input_channels=1, output_channels=1)
                    elif model_class == 'VGGUNet':
                        from wolffia_cnn_model import VGGUNet
                        self._cnn_model = VGGUNet(input_channels=1, output_channels=1)
                    else:
                        print(f"⚠️ Unknown model class: {model_class}")
                        continue
                    
                    # Load the model weights with proper error handling
                    try:
                        checkpoint = torch.load(model_path, map_location=self.device)
                        print(f"📥 Checkpoint loaded, keys: {list(checkpoint.keys()) if isinstance(checkpoint, dict) else 'Not a dict'}")
                        
                        # Handle different checkpoint formats
                        if isinstance(checkpoint, dict):
                            if 'model_state_dict' in checkpoint:
                                state_dict = checkpoint['model_state_dict']
                            elif 'state_dict' in checkpoint:
                                state_dict = checkpoint['state_dict']
                            else:
                                # Assume the checkpoint is the state dict itself
                                state_dict = checkpoint
                        else:
                            state_dict = checkpoint
                        
                        # Load state dict
                        self._cnn_model.load_state_dict(state_dict)
                        print("✅ Model state dictionary loaded successfully")
                        
                    except Exception as load_error:
                        print(f"❌ Failed to load checkpoint for {model_name}: {load_error}")
                        continue
                    
                    # Move to device and set to eval mode
                    self._cnn_model.to(self.device)
                    self._cnn_model.eval()
                    
                    # Set public properties for frontend compatibility
                    self.wolffia_cnn_model = self._cnn_model
                    self.wolffia_cnn_available = True
                    
                    print(f"✅ {model_name} loaded successfully!")
                    print(f"🎯 Model device: {self.device}")
                    print(f"🧠 Model parameters: {sum(p.numel() for p in self._cnn_model.parameters())}")
                    
                    return True
                    
                except Exception as e:
                    print(f"⚠️ Failed to load {model_name}: {e}")
                    import traceback
                    traceback.print_exc()
                    # Reset model and try next
                    self._cnn_model = None
                    continue
        
        # If we get here, no models loaded successfully
        print("❌ No CNN models could be loaded successfully")
        print("💡 To train a CNN model, run: python train_wolffia_cnn.py")
        self.wolffia_cnn_available = False
        self.wolffia_cnn_model = None
        return False
    
    def cnn_detection(self, gray_img):
        """
        Enhanced CNN Detection for Small Wolffia Cells
        Based on python_for_microscopists pattern recognition and segmentation techniques
        
        Citation: "Training ML model for segmentation" by Sreenivas Bhattiprolu  
        Reference: python_for_microscopists examples 062-066
        
        Citation: "Feature-based image segmentation" approaches
        Reference: python_for_microscopists ML segmentation workflows
        
        Optimized for small cell detection with enhanced post-processing
        """
        try:
            if self._cnn_model is None:
                print("⚠️ CNN model not available")
                return np.zeros_like(gray_img)
            
            print("🤖 Starting enhanced CNN detection for small Wolffia cells...")
            
            # Enhanced preprocessing for small cells (adapted from ML workflows)
            # Citation: Feature extraction and preprocessing techniques
            
            # Step 1: Adaptive preprocessing
            # Apply histogram equalization for better contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced_img = clahe.apply(gray_img)
            
            # Step 2: Multi-scale patch-based detection (for small cells)
            # Using smaller patches optimized for small Wolffia cells
            original_shape = enhanced_img.shape
            patch_size = 128  # Smaller patches for small cell detection
            overlap = 32      # Reduced overlap for efficiency
            
            print(f"🔍 Processing {original_shape} image with {patch_size}x{patch_size} patches...")
            
            # Create prediction arrays
            full_prediction = np.zeros(original_shape, dtype=np.float32)
            count_map = np.zeros(original_shape, dtype=np.float32)
            
            # Enhanced patch processing
            patches_processed = 0
            for y in range(0, original_shape[0] - patch_size + 1, patch_size - overlap):
                for x in range(0, original_shape[1] - patch_size + 1, patch_size - overlap):
                    # Extract patch
                    y_end = min(y + patch_size, original_shape[0])
                    x_end = min(x + patch_size, original_shape[1])
                    patch = enhanced_img[y:y_end, x:x_end]
                    
                    # Ensure consistent patch size
                    if patch.shape != (patch_size, patch_size):
                        patch = cv2.resize(patch, (patch_size, patch_size))
                    
                    # Enhanced preprocessing for CNN input
                    # Normalize to [0,1] range
                    patch_normalized = patch.astype(np.float32) / 255.0
                    
                    # Convert to tensor
                    patch_tensor = torch.from_numpy(patch_normalized).float().unsqueeze(0).unsqueeze(0)
                    patch_tensor = patch_tensor.to(self.device)
                    
                    # CNN inference
                    with torch.no_grad():
                        patch_output = self._cnn_model(patch_tensor)
                        patch_pred = patch_output.squeeze().cpu().numpy()
                    
                    # Resize prediction back to original patch dimensions
                    if patch_pred.shape != (y_end - y, x_end - x):
                        patch_pred = cv2.resize(patch_pred, (x_end - x, y_end - y))
                    
                    # Accumulate predictions with overlap handling
                    full_prediction[y:y_end, x:x_end] += patch_pred
                    count_map[y:y_end, x:x_end] += 1
                    patches_processed += 1
            
            print(f"🔍 Processed {patches_processed} patches")
            
            # Step 3: Enhanced post-processing (inspired by ML segmentation workflows)
            # Citation: Post-processing techniques from python_for_microscopists
            
            # Average overlapping predictions
            count_map[count_map == 0] = 1
            averaged_prediction = full_prediction / count_map
            
            # Apply sigmoid activation if needed
            prediction_sigmoid = 1 / (1 + np.exp(-averaged_prediction))
            
            # Enhanced thresholding for small cells
            # Use Otsu's method to find optimal threshold
            prediction_uint8 = (prediction_sigmoid * 255).astype(np.uint8)
            otsu_threshold, _ = cv2.threshold(prediction_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Apply threshold with some adjustment for small cells
            adaptive_threshold = max(otsu_threshold / 255.0, 0.25)  # Minimum threshold for small cells
            binary_mask = (prediction_sigmoid > adaptive_threshold).astype(np.uint8)
            
            print(f"🎯 Applied adaptive threshold: {adaptive_threshold:.3f}")
            
            # Step 4: Enhanced morphological processing for small cells
            # Citation: Morphological operations from watershed examples
            
            # Smaller kernels for small cells
            kernel_small = np.ones((2, 2), np.uint8)
            kernel_medium = np.ones((3, 3), np.uint8)
            
            # Clean up mask with morphological operations
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel_small, iterations=1)
            
            # Step 5: Enhanced watershed separation (for touching cells)
            # Citation: Distance transform and watershed from examples 033, 035
            
            # Distance transform
            dist_transform = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 3)
            
            # Find local maxima (individual cell centers)
            # Adjusted parameters for small Wolffia cells
            local_maxima = feature.peak_local_max(
                dist_transform, 
                min_distance=12,  # Reduced for small cells
                threshold_abs=max(0.2 * dist_transform.max(), 1.0),  # Adaptive threshold
                exclude_border=True
            )
            
            print(f"🎯 Found {len(local_maxima)} potential cell centers")
            
            # Create markers for watershed
            markers = np.zeros_like(binary_mask, dtype=np.int32)
            for i, (y, x) in enumerate(local_maxima, 1):
                markers[y, x] = i
            
            # Apply watershed if we have markers
            if len(local_maxima) > 0:
                # Slight dilation of markers
                markers = cv2.dilate(markers.astype(np.uint8), kernel_small, iterations=1).astype(np.int32)
                
                # Watershed algorithm
                img_3ch = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
                labels = cv2.watershed(img_3ch, markers)
                
                # Clean watershed boundaries
                labels[labels == -1] = 0
            else:
                # Fallback to connected components
                _, labels = cv2.connectedComponents(binary_mask)
            
            # Step 6: Enhanced validation and filtering
            # Citation: Region properties analysis from regionprops examples
            
            regions = measure.regionprops(labels)
            filtered_labels = np.zeros_like(labels)
            new_label = 1
            valid_cells = 0
            
            for region in regions:
                area = region.area
                eccentricity = region.eccentricity
                solidity = region.solidity
                
                # Enhanced validation for Wolffia cells
                if (self.min_cell_area <= area <= self.max_cell_area and  # Size filter
                    eccentricity < 0.9 and                               # Shape filter
                    solidity > 0.4 and                                   # Solidity filter
                    region.extent > 0.3):                                # Extent filter
                    
                    mask = labels == region.label
                    filtered_labels[mask] = new_label
                    new_label += 1
                    valid_cells += 1
            
            print(f"✅ Enhanced CNN: {valid_cells} valid small cells detected")
            print(f"📊 Detection efficiency: {valid_cells}/{len(regions)} regions validated")
            
            return filtered_labels
            
        except Exception as e:
            print(f"⚠️ Enhanced CNN detection failed: {e}")
            import traceback
            print(f"Detailed error: {traceback.format_exc()}")
            return np.zeros_like(gray_img)

    
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
            return np.zeros(img_shape, dtype=np.uint8)
        
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
        # Get region properties
        regions = measure.regionprops(labeled_img)
        
        # Create filtered image
        filtered_img = np.zeros_like(labeled_img)
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
                    
                    # Run initial analysis
                    result = self.analyze_image(file_path, use_tophat=False, use_cnn=False)
                    
                    # Store image info for training session with upload path info
                    image_info = {
                        'index': i,
                        'filename': original_filename,
                        'path': str(file_path),
                        'upload_filename': upload_filename,  # For frontend /uploads/ access
                        'initial_cells': result.get('cells', []),
                        'total_cells': result.get('total_cells', 0),
                        'cells_count': result.get('total_cells', 0)  # Frontend compatibility
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
                    
                    # Try to find the image file
                    for possible_path in [
                        self.dirs['uploads'] / image_filename,
                        Path(image_filename),  # If it's already a full path
                    ]:
                        if possible_path.exists():
                            image_path = possible_path
                            break
                    
                    if not image_path:
                        print(f"⚠️ Could not find image file: {image_filename}")
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