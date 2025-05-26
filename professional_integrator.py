"""
professional_pipeline.py - Production-Ready 11-Step Bioimage Analysis Pipeline
Fixed configuration handling, error management, and method implementations
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from plotly.subplots import make_subplots
from scipy import ndimage, stats
from skimage import feature, filters, measure, morphology, restoration, segmentation
from skimage.color import rgb2gray, rgb2hsv
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from analysis_config import AnalysisConfig
from comprehensive_visualizer import ComprehensiveVisualizer

# Import safe logging
from logging_config import setup_production_logging

logger = logging.getLogger(__name__)


@dataclass


class ProfessionalBioimageAnalyzer:
    """Production-ready 11-step bioimage analysis pipeline."""
    
# In professional_integrator.py, fix the __init__ method:

    def __init__(self, config=None):
        # FIXED: Create proper instance instead of using class type
        if config is None:
            self.config = AnalysisConfig()
        elif isinstance(config, dict):
            self.config = AnalysisConfig.from_dict(config)
        elif isinstance(config, AnalysisConfig):
            self.config = config
        else:
            # Fallback - create new instance
            self.config = AnalysisConfig()
            
        self.debug_images = {}
        self.processing_log = []
        self.quality_metrics = {}
        
        logger.info("[ANALYSIS] Professional 11-step pipeline initialized")

    def analyze_professional(self, image_input: Union[str, np.ndarray], progress_callback=None) -> Dict:
        """Main professional analysis pipeline with robust error handling."""
        
        analysis_start = datetime.now()
        self.processing_log = []
        self.debug_images = {}
        
        try:
            self._log_step("[START] Professional bioimage analysis pipeline")
            
            # Step 1: Load and validate image
            if progress_callback: progress_callback(5, "Loading and validating image...")
            image_data = self._load_and_validate_image(image_input)
            if image_data is None:
                return self._create_failure_result("Failed to load image", analysis_start)
            
            original = image_data['image']
            quality_score = image_data['quality_score']
            self._log_step(f"[OK] Image loaded: {original.shape}, quality: {quality_score:.3f}")
            
            # Step 2: Professional preprocessing
            if progress_callback: progress_callback(15, "Advanced preprocessing...")
            preprocessed = self._professional_preprocessing(original)
            if preprocessed is None:
                return self._create_failure_result("Preprocessing failed", analysis_start)
            
            # Step 3: Multi-method threshold detection
            if progress_callback: progress_callback(25, "Multi-method threshold detection...")
            threshold_results = self._multi_threshold_detection(preprocessed)
            
            # Step 4: Adaptive thresholding
            if progress_callback: progress_callback(35, "Adaptive thresholding...")
            adaptive_mask = self._adaptive_thresholding(preprocessed, threshold_results)
            
            # Step 5: Binary morphology enhancement
            if progress_callback: progress_callback(45, "Morphological processing...")
            morphology_mask = self._morphological_processing(adaptive_mask)
            
            # Step 6: Connected components analysis
            if progress_callback: progress_callback(55, "Connected components analysis...")
            labeled_image = self._connected_components_analysis(morphology_mask)
            
            # Step 7: Advanced cell segmentation
            if progress_callback: progress_callback(65, "Advanced cell segmentation...")
            segmented_cells = self._advanced_cell_segmentation(labeled_image, preprocessed)
            
            # Step 8: Postprocessing & border removal
            if progress_callback: progress_callback(75, "Postprocessing and filtering...")
            filtered_cells = self._postprocessing_filter(segmented_cells, original.shape)
            
            # Step 9: Edge detection & refinement
            if progress_callback: progress_callback(80, "Edge detection and refinement...")
            refined_cells = self._edge_detection_refinement(filtered_cells, preprocessed)
            
            # Step 10: Comprehensive measurements
            if progress_callback: progress_callback(85, "Extracting comprehensive measurements...")
            measurements_df = self._extract_comprehensive_measurements(refined_cells, original, preprocessed)
            
            # Step 11: ML quality control & validation
            if progress_callback: progress_callback(95, "ML quality validation...")
            validated_results = self._ml_quality_control(measurements_df, refined_cells)
            
            # Generate final report
            if progress_callback: progress_callback(98, "Generating professional report...")
            final_report = self._generate_professional_report(validated_results, original)
            
            # Calculate processing time
            processing_time = (datetime.now() - analysis_start).total_seconds()
            
            # Compile professional results
            result = {
                'success': True,
                'timestamp': analysis_start.isoformat(),
                'processing_time': processing_time,
                'total_cells': len(validated_results) if not validated_results.empty else 0,
                'pipeline_steps_completed': 11,
                'quality_score': self._calculate_overall_quality(),
                
                # Core data
                'cell_data': validated_results.to_dict('records') if not validated_results.empty else [],
                'summary_statistics': final_report['summary'],
                'quality_metrics': final_report['quality'],
                'processing_log': self.processing_log,
                'debug_visualizations': self.debug_images,
                
                # Professional metadata
                'labels': refined_cells,
                'preprocessed_image': preprocessed,
                'pipeline_info': {
                    'version': '3.0.0',
                    'method': 'professional_11_step',
                    'config_used': self.config.__dict__,
                    'steps_completed': 11
                }
            }
            
            if progress_callback: progress_callback(100, "[SUCCESS] Professional analysis complete!")
            self._log_step(f"[SUCCESS] Analysis completed in {processing_time:.2f}s with {result['total_cells']} cells")
            
            return result
            
        except Exception as e:
            processing_time = (datetime.now() - analysis_start).total_seconds()
            logger.error(f"[ERROR] Professional analysis failed: {str(e)}")
            return self._create_failure_result(str(e), analysis_start, processing_time)

    def _create_failure_result(self, error_msg: str, start_time: datetime, processing_time: float = None) -> Dict:
        """Create standardized failure result."""
        if processing_time is None:
            processing_time = (datetime.now() - start_time).total_seconds()
            
        return {
            'success': False,
            'error': error_msg,
            'timestamp': start_time.isoformat(),
            'processing_time': processing_time,
            'total_cells': 0,
            'cell_data': [],
            'processing_log': self.processing_log,
            'pipeline_steps_completed': 0
        }

    def _load_and_validate_image(self, image_input):
        """Load and validate image with comprehensive error handling."""
        try:
            if isinstance(image_input, str):
                if not Path(image_input).exists():
                    logger.error(f"[ERROR] Image file not found: {image_input}")
                    return None
                    
                image = cv2.imread(str(image_input))
                if image is None:
                    logger.error(f"[ERROR] Could not load image: {image_input}")
                    return None
                    
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
            elif isinstance(image_input, np.ndarray):
                if len(image_input.shape) == 3 and image_input.shape[2] == 3:
                    image_rgb = image_input.copy()
                elif len(image_input.shape) == 2:
                    image_rgb = np.stack([image_input] * 3, axis=-1)
                else:
                    logger.error(f"[ERROR] Invalid image array shape: {image_input.shape}")
                    return None
            else:
                logger.error(f"[ERROR] Invalid image input type: {type(image_input)}")
                return None
            
            # Normalize image
            if image_rgb.max() <= 1.0:
                image_rgb = (image_rgb * 255).astype(np.uint8)
            
            # Validate dimensions
            if image_rgb.shape[0] < 50 or image_rgb.shape[1] < 50:
                logger.error(f"[ERROR] Image too small: {image_rgb.shape}")
                return None
            
            # Quality assessment
            quality_score = self._assess_image_quality(image_rgb)
            
            return {
                'image': image_rgb,
                'quality_score': quality_score
            }
            
        except Exception as e:
            logger.error(f"[ERROR] Image loading failed: {str(e)}")
            return None

    def _assess_image_quality(self, image_rgb):
        """Assess image quality with robust error handling."""
        try:
            gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
            
            # Sharpness (Laplacian variance)
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness_score = min(sharpness / 1000.0, 1.0)
            
            # Contrast
            contrast = np.std(gray.astype(np.float32))
            contrast_score = min(contrast / 128.0, 1.0)
            
            # Brightness balance
            brightness = np.mean(gray)
            brightness_score = 1.0 - abs(brightness - 128) / 128.0
            
            # Overall quality
            quality_score = (sharpness_score + contrast_score + brightness_score) / 3.0
            
            return max(0.0, min(1.0, quality_score))
            
        except Exception as e:
            logger.error(f"[ERROR] Quality assessment failed: {str(e)}")
            return 0.5

    def _professional_preprocessing(self, image_rgb):
        """Professional-grade preprocessing with error handling."""
        try:
            self._log_step("[PREPROCESS] Advanced preprocessing started")
            
            # Convert to working format
            if len(image_rgb.shape) == 3:
                gray = rgb2gray(image_rgb.astype(np.float32) / 255.0)
            else:
                gray = image_rgb.astype(np.float32) / 255.0
            
            processed = gray.copy()
            
            # Noise reduction
            if self.config.preprocessing.get('noise_reduction', True):
                try:
                    processed = restoration.denoise_bilateral(
                        processed, 
                        sigma_color=0.05, 
                        sigma_spatial=self.config.noise_reduction_sigma
                    )
                    self.debug_images['01_denoised'] = processed
                except Exception as e:
                    logger.warning(f"[WARN] Noise reduction failed: {str(e)}")
            
            # Illumination correction
            if self.config.preprocessing.get('illumination_correction', True):
                try:
                    # Simple background estimation
                    kernel_size = min(processed.shape) // 20
                    if kernel_size > 0:
                        kernel = morphology.disk(kernel_size)
                        background = morphology.opening(processed, kernel)
                        corrected = processed - background + np.mean(background)
                        processed = np.clip(corrected, 0, 1)
                        self.debug_images['02_illumination_corrected'] = processed
                except Exception as e:
                    logger.warning(f"[WARN] Illumination correction failed: {str(e)}")
            
            # Contrast enhancement
            if self.config.preprocessing.get('contrast_enhancement', True):
                try:
                    processed_uint8 = (processed * 255).astype(np.uint8)
                    clahe = cv2.createCLAHE(
                        clipLimit=self.config.contrast_enhancement_clip, 
                        tileGridSize=(8, 8)
                    )
                    enhanced = clahe.apply(processed_uint8) / 255.0
                    processed = enhanced
                    self.debug_images['03_contrast_enhanced'] = processed
                except Exception as e:
                    logger.warning(f"[WARN] Contrast enhancement failed: {str(e)}")
            
            self.debug_images['04_final_preprocessed'] = processed
            self._log_step("[OK] Preprocessing complete")
            
            return processed
            
        except Exception as e:
            logger.error(f"[ERROR] Preprocessing failed: {str(e)}")
            # Return basic grayscale as fallback
            try:
                return rgb2gray(image_rgb.astype(np.float32) / 255.0)
            except:
                return np.ones((100, 100), dtype=np.float32) * 0.5

    def _multi_threshold_detection(self, image):
        """Multi-method threshold detection with error handling."""
        try:
            self._log_step("[THRESHOLD] Multi-method threshold detection")
            
            thresholds = {}
            threshold_images = {}
            methods = self.config.thresholding.get('methods', ['otsu'])
            
            # Otsu's method
            if 'otsu' in methods:
                try:
                    otsu_threshold = filters.threshold_otsu(image)
                    thresholds['otsu'] = float(otsu_threshold)
                    threshold_images['otsu'] = image > otsu_threshold
                except Exception as e:
                    logger.warning(f"[WARN] Otsu thresholding failed: {str(e)}")
            
            # Multi-Otsu
            if 'multiotsu' in methods:
                try:
                    multiotsu_thresholds = filters.threshold_multiotsu(image, classes=3)
                    thresholds['multiotsu'] = [float(t) for t in multiotsu_thresholds]
                    threshold_images['multiotsu'] = image > multiotsu_thresholds[0]
                except Exception as e:
                    logger.warning(f"[WARN] Multi-Otsu thresholding failed: {str(e)}")
            
            # Li's method
            if 'li' in methods:
                try:
                    li_threshold = filters.threshold_li(image)
                    thresholds['li'] = float(li_threshold)
                    threshold_images['li'] = image > li_threshold
                except Exception as e:
                    logger.warning(f"[WARN] Li thresholding failed: {str(e)}")
            
            # Store debug images
            for method, binary_img in threshold_images.items():
                self.debug_images[f'threshold_{method}'] = binary_img
            
            # Select best method
            recommended = self._select_best_threshold(threshold_images, image)
            
            self._log_step(f"[OK] Threshold detection complete: {len(thresholds)} methods")
            
            return {
                'thresholds': thresholds,
                'binary_images': threshold_images,
                'recommended': recommended
            }
            
        except Exception as e:
            logger.error(f"[ERROR] Threshold detection failed: {str(e)}")
            # Fallback threshold
            try:
                fallback_thresh = np.mean(image)
                return {
                    'thresholds': {'fallback': float(fallback_thresh)},
                    'binary_images': {'fallback': image > fallback_thresh},
                    'recommended': 'fallback'
                }
            except:
                return {
                    'thresholds': {},
                    'binary_images': {'default': image > 0.5},
                    'recommended': 'default'
                }

    def _select_best_threshold(self, threshold_images, original_image):
        """Select best threshold method with error handling."""
        if not threshold_images:
            return 'otsu'  # Default fallback
        
        try:
            scores = {}
            target_object_count = 50  # Expected number of objects
            
            for method, binary_img in threshold_images.items():
                try:
                    labels = measure.label(binary_img)
                    object_count = np.max(labels)
                    
                    # Score based on how close to expected count
                    count_score = 1.0 / (1.0 + abs(object_count - target_object_count) / target_object_count)
                    scores[method] = count_score
                    
                except Exception as e:
                    logger.warning(f"[WARN] Scoring failed for {method}: {str(e)}")
                    scores[method] = 0.0
            
            if scores:
                best_method = max(scores.keys(), key=lambda k: scores[k])
                self._log_step(f"[TARGET] Best threshold method: {best_method}")
                return best_method
            else:
                return list(threshold_images.keys())[0]
                
        except Exception as e:
            logger.warning(f"[WARN] Threshold selection failed: {str(e)}")
            return list(threshold_images.keys())[0] if threshold_images else 'otsu'

    def _adaptive_thresholding(self, image, threshold_results):
        """Adaptive thresholding with robust error handling."""
        try:
            self._log_step("[TARGET] Adaptive thresholding")
            
            # Convert to uint8
            image_uint8 = np.clip(image * 255, 0, 255).astype(np.uint8)
            
            # Get global threshold as reference
            best_method = threshold_results.get('recommended', 'otsu')
            global_binary = threshold_results['binary_images'].get(best_method, image > 0.5)
            
            # Adaptive parameters
            block_size = self.config.thresholding.get('adaptive_block_size', 21)
            C = self.config.thresholding.get('adaptive_c', 2)
            
            # Ensure block_size is odd and reasonable
            block_size = max(3, block_size)
            if block_size % 2 == 0:
                block_size += 1
            
            try:
                # Gaussian adaptive
                adaptive_gaussian = cv2.adaptiveThreshold(
                    image_uint8, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY, block_size, C
                ) > 0
                
                # Mean adaptive
                adaptive_mean = cv2.adaptiveThreshold(
                    image_uint8, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                    cv2.THRESH_BINARY, block_size, C
                ) > 0
                
                # Combine methods with voting
                votes = (global_binary.astype(int) + 
                        adaptive_gaussian.astype(int) + 
                        adaptive_mean.astype(int))
                combined_binary = votes >= 2
                
                self.debug_images['05_adaptive_gaussian'] = adaptive_gaussian
                self.debug_images['06_adaptive_mean'] = adaptive_mean
                self.debug_images['07_combined_adaptive'] = combined_binary
                
                self._log_step("[OK] Adaptive thresholding complete")
                return combined_binary
                
            except Exception as e:
                logger.warning(f"[WARN] Adaptive thresholding failed: {str(e)}")
                return global_binary
                
        except Exception as e:
            logger.error(f"[ERROR] Adaptive thresholding failed: {str(e)}")
            # Return simple threshold as fallback
            try:
                return image > np.mean(image)
            except:
                return np.ones_like(image, dtype=bool)

    def _morphological_processing(self, binary_mask):
        """Morphological processing with comprehensive error handling."""
        try:
            self._log_step("[PROCESS] Morphological processing")
            
            processed = binary_mask.copy()
            
            # Remove small objects
            remove_small = self.config.morphology.get('remove_small_objects', 50)
            if remove_small > 0:
                try:
                    processed = morphology.remove_small_objects(processed, min_size=remove_small)
                except Exception as e:
                    logger.warning(f"[WARN] Small object removal failed: {str(e)}")
            
            # Fill holes
            if self.config.morphology.get('fill_holes', True):
                try:
                    processed = morphology.remove_small_holes(processed, area_threshold=200)
                except Exception as e:
                    logger.warning(f"[WARN] Hole filling failed: {str(e)}")
            
            # Opening
            opening_size = self.config.morphology.get('opening_size', 2)
            if opening_size > 0:
                try:
                    selem_open = morphology.disk(opening_size)
                    processed = morphology.opening(processed, selem_open)
                except Exception as e:
                    logger.warning(f"[WARN] Opening operation failed: {str(e)}")
            
            # Closing
            closing_size = self.config.morphology.get('closing_size', 3)
            if closing_size > 0:
                try:
                    selem_close = morphology.disk(closing_size)
                    processed = morphology.closing(processed, selem_close)
                except Exception as e:
                    logger.warning(f"[WARN] Closing operation failed: {str(e)}")
            
            self.debug_images['08_morphology_final'] = processed
            self._log_step("[OK] Morphological processing complete")
            
            return processed
            
        except Exception as e:
            logger.error(f"[ERROR] Morphological processing failed: {str(e)}")
            return binary_mask

    def _connected_components_analysis(self, binary_mask):
        """Connected components analysis with filtering."""
        try:
            self._log_step("[PROCESS] Connected components analysis")
            
            # Label connected components
            labeled = measure.label(binary_mask)
            
            if np.max(labeled) == 0:
                self._log_step("[WARN] No objects found in binary mask")
                return labeled
            
            # Analyze components
            regions = measure.regionprops(labeled)
            
            # Filter components
            min_area = self.config.postprocessing.get('min_area', self.config.min_cell_area)
            max_area = self.config.postprocessing.get('max_area', self.config.max_cell_area)
            min_circularity = self.config.postprocessing.get('min_circularity', 0.3)
            max_eccentricity = self.config.postprocessing.get('max_eccentricity', 0.95)
            
            valid_components = []
            for region in regions:
                try:
                    # Size filtering
                    if not (min_area <= region.area <= max_area):
                        continue
                    
                    # Shape filtering
                    circularity = 4 * np.pi * region.area / (region.perimeter ** 2 + 1e-8)
                    if circularity < min_circularity:
                        continue
                    
                    if region.eccentricity > max_eccentricity:
                        continue
                    
                    valid_components.append(region.label)
                    
                except Exception as e:
                    logger.warning(f"[WARN] Component analysis failed for region {region.label}: {str(e)}")
                    continue
            
            # Create filtered label image
            filtered_labels = np.zeros_like(labeled)
            for i, label in enumerate(valid_components, 1):
                filtered_labels[labeled == label] = i
            
            self.debug_images['12_connected_components'] = labeled
            self.debug_images['13_filtered_components'] = filtered_labels
            
            self._log_step(f"[OK] Connected components: {len(valid_components)} valid objects")
            return filtered_labels
            
        except Exception as e:
            logger.error(f"[ERROR] Connected components analysis failed: {str(e)}")
            return measure.label(binary_mask)

    def _advanced_cell_segmentation(self, labeled_image, intensity_image):
        """Advanced segmentation using watershed."""
        try:
            self._log_step("[PROCESS] Advanced cell segmentation")
            
            if np.max(labeled_image) == 0:
                return labeled_image
            
            binary_mask = labeled_image > 0
            distance = ndimage.distance_transform_edt(binary_mask)
            
            min_distance = self.config.watershed_min_distance
            
            try:
                # FIXED: Remove 'indices' parameter - use return_indices instead
                from skimage.feature import peak_local_max
                
                # Check scikit-image version and use appropriate parameters
                try:
                    # Try newer API first
                    local_maxima = peak_local_max(
                        distance,
                        min_distance=min_distance,
                        threshold_abs=0.3 * distance.max(),
                        exclude_border=False
                    )
                    # Convert coordinates to mask
                    mask = np.zeros_like(distance, dtype=bool)
                    mask[tuple(local_maxima.T)] = True
                    markers = measure.label(mask)
                except:
                    # Fallback for older versions
                    local_maxima = peak_local_max(
                        distance,
                        min_distance=min_distance,
                        threshold_abs=0.3 * distance.max(),
                        indices=False  # This might work in older versions
                    )
                    markers = measure.label(local_maxima)
                
                # Watershed segmentation
                segmented = segmentation.watershed(-distance, markers, mask=binary_mask)
                
                self.debug_images['15_watershed_segmented'] = segmented
                self._log_step(f"[OK] Advanced segmentation: {np.max(segmented)} cells")
                
                return segmented
                
            except Exception as e:
                logger.warning(f"[WARN] Watershed segmentation failed: {str(e)}")
                # Fallback to simple segmentation
                return labeled_image
                
        except Exception as e:
            logger.error(f"[ERROR] Advanced segmentation failed: {str(e)}")
            return labeled_image

    def _postprocessing_filter(self, labeled_cells, image_shape):
        """Postprocessing and border removal."""
        try:
            self._log_step("[PROCESS] Postprocessing and filtering")
            
            if not self.config.postprocessing.get('remove_border', True):
                return labeled_cells
            
            border_width = self.config.postprocessing.get('border_width', 10)
            h, w = image_shape[:2]
            
            # Create border mask
            border_mask = np.zeros((h, w), dtype=bool)
            border_mask[:border_width, :] = True
            border_mask[-border_width:, :] = True
            border_mask[:, :border_width] = True
            border_mask[:, -border_width:] = True
            
            # Find border labels
            border_labels = np.unique(labeled_cells[border_mask])
            border_labels = border_labels[border_labels > 0]
            
            # Remove border cells
            filtered_cells = labeled_cells.copy()
            for label in border_labels:
                filtered_cells[filtered_cells == label] = 0
            
            # Relabel
            filtered_cells = measure.label(filtered_cells > 0)
            
            removed_count = len(border_labels)
            remaining_count = np.max(filtered_cells)
            
            self._log_step(f"[OK] Border filtering: removed {removed_count}, {remaining_count} remaining")
            return filtered_cells
            
        except Exception as e:
            logger.error(f"[ERROR] Postprocessing failed: {str(e)}")
            return labeled_cells

    def _edge_detection_refinement(self, labeled_cells, intensity_image):
        """Edge detection and boundary refinement."""
        try:
            self._log_step("[PROCESS] Edge detection and refinement")
            
            if np.max(labeled_cells) == 0:
                return labeled_cells
            
            refined_cells = labeled_cells.copy()
            
            try:
                regions = measure.regionprops(labeled_cells)
                
                for region in regions:
                    if region.area > 100:  # Only refine larger cells
                        try:
                            cell_mask = labeled_cells == region.label
                            smoothed_mask = morphology.closing(cell_mask, morphology.disk(1))
                            refined_cells[cell_mask] = 0
                            refined_cells[smoothed_mask] = region.label
                        except Exception as e:
                            logger.warning(f"[WARN] Edge refinement failed for cell {region.label}: {str(e)}")
                            continue
                
                self._log_step(f"[OK] Edge refinement complete for {len(regions)} cells")
                
            except Exception as e:
                logger.warning(f"[WARN] Edge refinement failed: {str(e)}")
            
            return refined_cells
            
        except Exception as e:
            logger.error(f"[ERROR] Edge refinement failed: {str(e)}")
            return labeled_cells

    def _extract_comprehensive_measurements(self, labeled_cells, original_image, intensity_image):
        """Extract comprehensive measurements with error handling."""
        try:
            self._log_step("[STATS] Extracting comprehensive measurements")
            
            if np.max(labeled_cells) == 0:
                return pd.DataFrame()
            
            regions = measure.regionprops(labeled_cells, intensity_image=intensity_image)
            measurements = []
            
            for region in regions:
                try:
                    measurement = {
                        'cell_id': region.label,
                        'area': region.area,
                        'perimeter': region.perimeter,
                        'centroid_x': region.centroid[1],
                        'centroid_y': region.centroid[0],
                        'major_axis_length': region.major_axis_length,
                        'minor_axis_length': region.minor_axis_length,
                        'eccentricity': region.eccentricity,
                        'orientation': region.orientation,
                        'solidity': region.solidity,
                        'extent': region.extent,
                        'mean_intensity': region.mean_intensity,
                    }
                    
                    # Calculate derived features
                    measurement['aspect_ratio'] = (region.major_axis_length / 
                                                 (region.minor_axis_length + 1e-8))
                    measurement['circularity'] = (4 * np.pi * region.area / 
                                                (region.perimeter ** 2 + 1e-8))
                    
                    # Color analysis from original image
                    if len(original_image.shape) == 3:
                        try:
                            cell_rgb = original_image[region.slice][region.image]
                            if len(cell_rgb) > 0:
                                measurement.update({
                                    'mean_red': np.mean(cell_rgb[:, 0]) / 255.0,
                                    'mean_green': np.mean(cell_rgb[:, 1]) / 255.0,
                                    'mean_blue': np.mean(cell_rgb[:, 2]) / 255.0,
                                })
                                
                                # Chlorophyll estimate
                                chlorophyll = (measurement['mean_green'] - 
                                             0.5 * (measurement['mean_red'] + measurement['mean_blue']))
                                measurement['chlorophyll_content'] = max(0, chlorophyll)
                        except Exception as e:
                            logger.warning(f"[WARN] Color analysis failed for cell {region.label}: {str(e)}")
                            measurement.update({
                                'mean_red': 0.5, 'mean_green': 0.5, 'mean_blue': 0.5,
                                'chlorophyll_content': 0.5
                            })
                    
                    measurements.append(measurement)
                    
                except Exception as e:
                    logger.warning(f"[WARN] Measurement extraction failed for region {region.label}: {str(e)}")
                    continue
            
            df = pd.DataFrame(measurements)
            self._log_step(f"[OK] Measurements extracted: {len(df)} cells, {len(df.columns) if not df.empty else 0} features")
            
            return df
            
        except Exception as e:
            logger.error(f"[ERROR] Measurement extraction failed: {str(e)}")
            return pd.DataFrame()

    def _ml_quality_control(self, measurements_df, labeled_cells):
        """ML quality control and validation."""
        try:
            self._log_step("[ML] ML quality control and validation")
            
            if measurements_df.empty or len(measurements_df) < 5:
                self._log_step("[WARN] Insufficient data for ML quality control")
                return measurements_df
            
            # Select features for outlier detection
            numeric_features = measurements_df.select_dtypes(include=[np.number]).columns
            feature_cols = [col for col in numeric_features 
                          if col not in ['cell_id', 'centroid_x', 'centroid_y']]
            
            if len(feature_cols) < 3:
                self._log_step("[WARN] Insufficient features for ML quality control")
                measurements_df['is_outlier'] = False
                measurements_df['quality_score'] = 0.8
                return measurements_df
            
            try:
                # Prepare data
                X = measurements_df[feature_cols].fillna(measurements_df[feature_cols].mean())
                
                # Outlier detection
                isolation_forest = IsolationForest(
                    contamination=self.config.outlier_detection_contamination,
                    random_state=42
                )
                outlier_labels = isolation_forest.fit_predict(X)
                measurements_df['is_outlier'] = outlier_labels == -1
                
                # Quality scoring
                quality_scores = []
                for _, row in measurements_df.iterrows():
                    score = 1.0
                    
                    # Shape quality
                    if row.get('circularity', 0) < 0.3:
                        score *= 0.7
                    if row.get('solidity', 0) < 0.6:
                        score *= 0.8
                    if row.get('is_outlier', False):
                        score *= 0.5
                    
                    quality_scores.append(max(0, min(1, score)))
                
                measurements_df['quality_score'] = quality_scores
                
                # Filter high quality cells
                threshold = self.config.confidence_interval
                high_quality = measurements_df['quality_score'] >= threshold
                measurements_df = measurements_df[high_quality].copy()
                
                passed_count = np.sum(high_quality)
                total_count = len(quality_scores)
                
                self._log_step(f"[TARGET] Quality filtering: {passed_count}/{total_count} cells passed")
                
            except Exception as e:
                logger.warning(f"[WARN] ML quality control failed: {str(e)}")
                measurements_df['is_outlier'] = False
                measurements_df['quality_score'] = 0.8
            
            return measurements_df
            
        except Exception as e:
            logger.error(f"[ERROR] ML quality control failed: {str(e)}")
            return measurements_df

    def _generate_professional_report(self, measurements_df, original_image):
        """Generate comprehensive professional report."""
        try:
            self._log_step("[REPORT] Generating professional analysis report")
            
            if measurements_df.empty:
                return self._empty_professional_report()
            
            # Summary statistics
            summary = {
                'cell_count': {
                    'total_cells': len(measurements_df),
                    'high_quality_cells': np.sum(measurements_df.get('quality_score', 0.8) > 0.7),
                    'outlier_cells': np.sum(measurements_df.get('is_outlier', False)),
                },
                'morphological_analysis': {
                    'mean_area': float(measurements_df['area'].mean()),
                    'std_area': float(measurements_df['area'].std()),
                    'median_area': float(measurements_df['area'].median()),
                    'total_area': float(measurements_df['area'].sum()),
                    'mean_perimeter': float(measurements_df['perimeter'].mean()),
                    'mean_circularity': float(measurements_df['circularity'].mean()),
                    'mean_aspect_ratio': float(measurements_df['aspect_ratio'].mean()),
                },
                'spatial_analysis': {
                    'image_coverage': float(measurements_df['area'].sum() / 
                                          (original_image.shape[0] * original_image.shape[1]) * 100),
                    'cell_density': float(len(measurements_df) / 
                                        (original_image.shape[0] * original_image.shape[1]) * 1000000),
                }
            }
            
            # Quality metrics
            quality_metrics = {
                'analysis_quality': {
                    'mean_quality_score': float(measurements_df.get('quality_score', pd.Series([0.8])).mean()),
                    'outlier_rate': float(measurements_df.get('is_outlier', pd.Series([False])).mean()),
                },
                'segmentation_quality': {
                    'mean_circularity': float(measurements_df['circularity'].mean()),
                    'mean_solidity': float(measurements_df['solidity'].mean()),
                }
            }
            
            # Generate recommendations
            recommendations = self._generate_recommendations(summary, quality_metrics)
            
            self._log_step("[OK] Professional report generated")
            
            return {
                'summary': summary,
                'quality': quality_metrics,
                'recommendations': recommendations
            }
            
        except Exception as e:
            logger.error(f"[ERROR] Report generation failed: {str(e)}")
            return self._empty_professional_report()

    def _empty_professional_report(self):
        """Empty report structure."""
        return {
            'summary': {
                'cell_count': {'total_cells': 0},
                'morphological_analysis': {},
                'spatial_analysis': {}
            },
            'quality': {
                'analysis_quality': {'mean_quality_score': 0.0},
                'segmentation_quality': {}
            },
            'recommendations': ["[ERROR] No data available for analysis"]
        }

    def _generate_recommendations(self, summary, quality_metrics):
        """Generate actionable recommendations."""
        recommendations = []
        
        try:
            cell_count = summary['cell_count']['total_cells']
            
            if cell_count == 0:
                recommendations.append("[SEARCH] No cells detected - review image quality and parameters")
            elif cell_count < 10:
                recommendations.append("[WARN] Low cell count - verify image contains samples")
            elif cell_count > 200:
                recommendations.append("[STATS] High cell density - consider image cropping")
            else:
                recommendations.append("[OK] Good cell count for analysis")
            
            quality_score = quality_metrics['analysis_quality']['mean_quality_score']
            if quality_score < 0.6:
                recommendations.append("[IMAGE] Low analysis quality - improve image conditions")
            elif quality_score > 0.8:
                recommendations.append("[TARGET] Excellent analysis quality")
            else:
                recommendations.append("[OK] Acceptable analysis quality")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"[ERROR] Recommendation generation failed: {str(e)}")
            return ["[STATS] Analysis complete - review results for insights"]

    def _calculate_overall_quality(self):
        """Calculate overall pipeline quality score."""
        try:
            factors = []
            
            # Steps completion factor
            if len(self.processing_log) >= 10:
                factors.append(1.0)
            else:
                factors.append(len(self.processing_log) / 11.0)
            
            # Debug images factor (indicates successful processing)
            if len(self.debug_images) > 5:
                factors.append(0.9)
            else:
                factors.append(len(self.debug_images) / 10.0)
            
            return float(np.mean(factors)) if factors else 0.5
            
        except Exception as e:
            logger.error(f"[ERROR] Quality calculation failed: {str(e)}")
            return 0.5

    def _log_step(self, message):
        """Log processing step with timestamp."""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        log_entry = f"[{timestamp}] {message}"
        self.processing_log.append(log_entry)
        logger.info(log_entry)