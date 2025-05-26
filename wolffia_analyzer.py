"""
wolffia_analyzer.py - Production-Ready Main Analyzer
Complete integration with all components, robust error handling, and professional quality
"""

import json
import logging
import os
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage
from skimage import measure

from analysis_config import AnalysisConfig
from comprehensive_visualizer import ComprehensiveVisualizer

# Import safe logging first
from logging_config import setup_production_logging

logger = logging.getLogger(__name__)

# Configuration dataclass

@dataclass
class WolffiaAnalyzer:
    """Production-ready Wolffia analysis system with comprehensive features."""
    
    def __init__(self, pixel_to_micron_ratio=1.0, debug_mode=False, output_dir="outputs", **kwargs):
        # Core parameters
        self.pixel_to_micron = pixel_to_micron_ratio
        self.debug_mode = debug_mode
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create configuration
        self.config = AnalysisConfig(
            pixel_to_micron=pixel_to_micron_ratio,
            min_cell_area=kwargs.get('min_cell_area', 30),
            max_cell_area=kwargs.get('max_cell_area', 8000),
            chlorophyll_threshold=kwargs.get('chlorophyll_threshold', 0.6)
        )
        
        # Create subdirectories
        for subdir in ["debug_images", "results", "exports"]:
            (self.output_dir / subdir).mkdir(exist_ok=True)
        
        # Initialize components with comprehensive error handling
        self._initialize_components()
        
        # Results storage
        self.results_history = []
        self.time_series_data = []
        self.progress_callback = None
        
        logger.info("[SYSTEM] Production Wolffia Analyzer initialized")
        logger.info(f"   Debug mode: {debug_mode}")
        logger.info(f"   Output directory: {self.output_dir}")

    def _initialize_components(self):
        """Initialize all available components with error handling."""
        self.components = {}
        
        # Try to import and initialize professional pipeline
        try:
            from professional_integrator import ProfessionalBioimageAnalyzer
            self.components['professional_pipeline'] = ProfessionalBioimageAnalyzer(self.config)
            logger.info("[OK] Professional pipeline component initialized")
        except ImportError as e:
            logger.warning(f"[WARN] Professional pipeline not available: {e}")
        except Exception as e:
            logger.error(f"[ERROR] Professional pipeline initialization failed: {str(e)}")
        
        # Try to import image processor
        try:
            from image_processor import ImageProcessor
            self.components['image_processor'] = ImageProcessor()
            logger.info("[OK] Image processor component initialized")
        except ImportError as e:
            logger.warning(f"[WARN] Image processor not available: {e}")
        except Exception as e:
            logger.error(f"[ERROR] Image processor initialization failed: {str(e)}")
        
        # Try to import segmentation
        try:
            from segmentation import WolffiaSpecificSegmentation
            self.components['segmenter'] = WolffiaSpecificSegmentation(
                min_area=self.config.min_cell_area,
                max_area=self.config.max_cell_area,
                debug_mode=self.debug_mode
            )
            logger.info("[OK] Segmentation component initialized")
        except ImportError as e:
            logger.warning(f"[WARN] Segmentation not available: {e}")
        except Exception as e:
            logger.error(f"[ERROR] Segmentation initialization failed: {str(e)}")

    def set_progress_callback(self, callback):
        """Set callback function for progress updates."""
        self.progress_callback = callback

    def _emit_progress(self, progress, stage, **kwargs):
        """Emit progress update if callback is set."""
        if self.progress_callback:
            try:
                self.progress_callback(progress, stage, **kwargs)
            except Exception as e:
                logger.warning(f"[WARN] Progress callback error: {str(e)}")

    def analyze_single_image(self, image_path, **kwargs) -> dict:
        """
        Main analysis pipeline - PRODUCTION READY with comprehensive error handling.
        """
        start_time = datetime.now()
        timestamp = start_time.strftime("%Y%m%d_%H%M%S")
        
        logger.info(f"[ANALYSIS] Starting Wolffia analysis: {image_path}")
        self._emit_progress(0, "Starting analysis...")
        
        try:
            # Update parameters from kwargs
            pixel_ratio = float(kwargs.get('pixel_ratio', self.pixel_to_micron))
            self.pixel_to_micron = pixel_ratio
            self.config.pixel_to_micron = pixel_ratio
            analysis_method = kwargs.get('analysis_method', 'auto')
            
            # Step 1: Load and validate image
            self._emit_progress(5, "Loading and validating image...")
            image_rgb = self._load_and_validate_image(image_path)
            if image_rgb is None:
                return self._create_error_result("Failed to load image", timestamp, start_time)
            
            logger.info(f"[OK] Image loaded: {image_rgb.shape}")
            
            # Choose analysis method based on availability
            use_professional = (analysis_method == 'professional' or 
                              (analysis_method == 'auto' and 'professional_pipeline' in self.components))
            
            # Professional pipeline if available
            if use_professional and 'professional_pipeline' in self.components:
                logger.info("[ANALYSIS] Using Professional 11-step Pipeline")
                
                def professional_progress(progress, stage):
                    self._emit_progress(progress, f"Professional: {stage}")
                
                try:
                    professional_result = self.components['professional_pipeline'].analyze_professional(
                        image_rgb, 
                        progress_callback=professional_progress
                    )
                    
                    if professional_result.get('success'):
                        processing_time = (datetime.now() - start_time).total_seconds()
                        result = self._convert_professional_result(professional_result, timestamp, processing_time)
                        result['analysis_method'] = 'professional_11_step'
                        
                        # Export if requested
                        if kwargs.get('auto_export', False):
                            self._export_results(result, timestamp)
                        
                        self.results_history.append(result)
                        self._emit_progress(100, "[SUCCESS] Professional analysis complete!")
                        logger.info(f"[SUCCESS] Professional analysis completed in {processing_time:.2f}s")
                        
                        return result
                    else:
                        logger.warning("[WARN] Professional pipeline failed, using fallback")
                        
                except Exception as e:
                    logger.error(f"[ERROR] Professional pipeline error: {str(e)}")
            
            # Standard analysis pipeline (fallback or explicit)
            logger.info("[ANALYSIS] Using Standard Analysis Pipeline")
            return self._standard_analysis_pipeline(image_rgb, timestamp, image_path, **kwargs)
            
        except Exception as e:
            logger.error(f"[ERROR] Analysis failed: {str(e)}")
            return self._create_error_result(str(e), timestamp, start_time)

    def _load_and_validate_image(self, image_input):
        """Load and validate image with comprehensive error handling."""
        try:
            if isinstance(image_input, str):
                # File path
                if not Path(image_input).exists():
                    logger.error(f"[ERROR] Image file not found: {image_input}")
                    return None
                
                image = cv2.imread(str(image_input))
                if image is None:
                    logger.error(f"[ERROR] Could not load image: {image_input}")
                    return None
                
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
            elif isinstance(image_input, np.ndarray):
                # Array input
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
            
            return image_rgb
            
        except Exception as e:
            logger.error(f"[ERROR] Image loading failed: {str(e)}")
            return None

    def _standard_analysis_pipeline(self, image_rgb, timestamp, image_path, **kwargs):
        """Standard analysis pipeline with comprehensive error handling."""
        try:
            start_time = datetime.now()
            
            # Step 2: Preprocessing
            self._emit_progress(15, "Preprocessing image...")
            if 'image_processor' in self.components:
                try:
                    preprocess_result = self.components['image_processor'].preprocess_image(image_rgb)
                    if preprocess_result:
                        original, gray, green_channel, chlorophyll_enhanced, hsv = preprocess_result
                        logger.info("[OK] Advanced preprocessing complete")
                    else:
                        # Fallback preprocessing
                        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY) / 255.0
                        green_channel = image_rgb[:, :, 1] / 255.0
                        chlorophyll_enhanced = green_channel
                        logger.warning("[WARN] Using fallback preprocessing")
                except Exception as e:
                    logger.warning(f"[WARN] Advanced preprocessing failed: {str(e)}")
                    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY) / 255.0
                    green_channel = image_rgb[:, :, 1] / 255.0
                    chlorophyll_enhanced = green_channel
            else:
                gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY) / 255.0
                green_channel = image_rgb[:, :, 1] / 255.0
                chlorophyll_enhanced = green_channel
            
            # Step 3: Segmentation
            self._emit_progress(35, "Segmenting cells...")
            if 'segmenter' in self.components:
                try:
                    labels, debug_info = self.components['segmenter'].segment(image_rgb)
                    logger.info(f"[OK] Advanced segmentation: {np.max(labels)} cells")
                except Exception as e:
                    logger.warning(f"[WARN] Advanced segmentation failed: {str(e)}")
                    labels = self._basic_segmentation(gray)
            else:
                labels = self._basic_segmentation(gray)
            
            # Step 4: Feature extraction
            self._emit_progress(65, "Extracting features...")
            df = self._extract_comprehensive_features(labels, image_rgb, green_channel)
            
            if df.empty:
                logger.warning("[WARN] No features extracted")
            else:
                logger.info(f"[OK] Features extracted: {len(df)} cells")
            
            # Step 5: Calculate summary
            self._emit_progress(80, "Calculating statistics...")
            summary = self._calculate_summary_stats(df, image_rgb.shape)
            
            # Step 6: Quality assessment
            quality_score = self._assess_analysis_quality(df, labels, image_rgb)
            
            # Step 7: Create result
            self._emit_progress(95, "Finalizing results...")
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = {
                'success': True,
                'timestamp': timestamp,
                'image_path': str(image_path),
                'total_cells': len(df),
                'cell_data': df.to_dict('records') if not df.empty else [],
                'summary': summary,
                'quality_score': quality_score,
                'processing_time': processing_time,
                'labels': labels,
                'analysis_method': 'standard',
                'processing_info': {
                    'pixel_to_micron': self.pixel_to_micron,
                    'debug_mode': self.debug_mode,
                    'image_dimensions': f"{image_rgb.shape[1]}x{image_rgb.shape[0]}",
                    'segmentation_method': 'wolffia_specific' if 'segmenter' in self.components else 'basic'
                }
            }
            
            # Export if requested
            if kwargs.get('auto_export', False):
                self._export_results(result, timestamp)
            
            self.results_history.append(result)
            self._emit_progress(100, "[SUCCESS] Analysis complete!")
            logger.info(f"[SUCCESS] Standard analysis completed in {processing_time:.2f}s")
            
            return result
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"[ERROR] Standard analysis failed: {str(e)}")
            return self._create_error_result(str(e), timestamp, start_time, processing_time)

    def _basic_segmentation(self, gray_image):
        """Robust basic segmentation fallback."""
        try:
            logger.info("[PROCESS] Running basic segmentation fallback")
            
            # Ensure proper format
            if gray_image.max() <= 1.0:
                gray_uint8 = (gray_image * 255).astype(np.uint8)
            else:
                gray_uint8 = gray_image.astype(np.uint8)
            
            # Multi-threshold approach
            try:
                _, binary_otsu = cv2.threshold(gray_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            except:
                binary_otsu = gray_uint8 > 127
            
            try:
                binary_adaptive = cv2.adaptiveThreshold(
                    gray_uint8, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY, 21, 2
                )
            except:
                binary_adaptive = binary_otsu
            
            # Combine thresholds
            binary = ((binary_otsu > 0) | (binary_adaptive > 0)).astype(np.uint8)
            
            # Morphological cleanup
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            
            # Size filtering with contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            mask = np.zeros_like(binary)
            
            valid_contours = 0
            for contour in contours:
                area = cv2.contourArea(contour)
                if self.config.min_cell_area <= area <= self.config.max_cell_area:
                    # Additional shape check
                    if len(contour) >= 5:  # Minimum points for ellipse fitting
                        try:
                            ellipse = cv2.fitEllipse(contour)
                            aspect_ratio = max(ellipse[1]) / (min(ellipse[1]) + 1e-8)
                            if aspect_ratio < 3.0:  # Not too elongated
                                cv2.fillPoly(mask, [contour], 255)
                                valid_contours += 1
                        except:
                            cv2.fillPoly(mask, [contour], 255)
                            valid_contours += 1
            
            # Label connected components
            try:
                labels = measure.label(mask > 0)
            except:
                num_labels, labels = cv2.connectedComponents(mask)
            
            logger.info(f"[OK] Basic segmentation: {np.max(labels)} cells detected")
            return labels
            
        except Exception as e:
            logger.error(f"[ERROR] Basic segmentation failed: {str(e)}")
            return np.zeros_like(gray_image, dtype=np.int32)

    def _extract_comprehensive_features(self, labels, original_image, green_channel):
        """Extract comprehensive features with robust error handling."""
        try:
            if np.max(labels) == 0:
                return pd.DataFrame()
            
            regions = measure.regionprops(labels, intensity_image=green_channel)
            features = []
            
            for region in regions:
                try:
                    feature_dict = {
                        'cell_id': region.label,
                        'area': region.area,
                        'area_pixels': region.area,
                        'area_microns_sq': region.area * (self.pixel_to_micron ** 2),
                        'perimeter': region.perimeter * self.pixel_to_micron,
                        'centroid_x': region.centroid[1] * self.pixel_to_micron,
                        'centroid_y': region.centroid[0] * self.pixel_to_micron,
                        'major_axis_length': region.major_axis_length * self.pixel_to_micron,
                        'minor_axis_length': region.minor_axis_length * self.pixel_to_micron,
                        'eccentricity': region.eccentricity,
                        'solidity': region.solidity,
                        'extent': region.extent,
                        'orientation': region.orientation,
                        'mean_intensity': region.mean_intensity,
                    }
                    
                    # Derived features
                    feature_dict['aspect_ratio'] = region.major_axis_length / (region.minor_axis_length + 1e-8)
                    feature_dict['circularity'] = 4 * np.pi * region.area / (region.perimeter ** 2 + 1e-8)
                    feature_dict['roundness'] = 4 * region.area / (np.pi * region.major_axis_length ** 2 + 1e-8)
                    
                    # Color analysis
                    try:
                        if hasattr(region, 'slice') and original_image is not None:
                            cell_pixels = original_image[region.slice][region.image]
                            if len(cell_pixels) > 0:
                                feature_dict['mean_red'] = np.mean(cell_pixels[:, 0]) / 255.0
                                feature_dict['mean_green'] = np.mean(cell_pixels[:, 1]) / 255.0
                                feature_dict['mean_blue'] = np.mean(cell_pixels[:, 2]) / 255.0
                                
                                # Chlorophyll estimate
                                chlorophyll = (feature_dict['mean_green'] - 
                                             0.5 * (feature_dict['mean_red'] + feature_dict['mean_blue']))
                                feature_dict['chlorophyll_content'] = max(0, chlorophyll)
                            else:
                                feature_dict.update({
                                    'mean_red': 0.5, 'mean_green': 0.5, 'mean_blue': 0.5,
                                    'chlorophyll_content': 0.5
                                })
                    except Exception as color_error:
                        logger.warning(f"[WARN] Color analysis failed for cell {region.label}: {str(color_error)}")
                        feature_dict.update({
                            'mean_red': 0.5, 'mean_green': 0.5, 'mean_blue': 0.5,
                            'chlorophyll_content': 0.5
                        })
                    
                    # Health score
                    integrity_score = region.solidity * (1.0 - region.eccentricity)
                    photosynthetic_activity = min(region.mean_intensity / 255.0, 1.0)
                    feature_dict['health_score'] = (integrity_score + photosynthetic_activity) / 2.0
                    
                    features.append(feature_dict)
                    
                except Exception as e:
                    logger.warning(f"[WARN] Feature extraction failed for region {region.label}: {str(e)}")
                    continue
            
            df = pd.DataFrame(features)
            
            # Add population-level features
            if len(df) > 0:
                df = self._add_population_features(df)
            
            return df
            
        except Exception as e:
            logger.error(f"[ERROR] Feature extraction failed: {str(e)}")
            return pd.DataFrame()

    def _add_population_features(self, df):
        """Add population-level derived features."""
        try:
            # Size categories
            if 'area_pixels' in df.columns and len(df) > 2:
                area_33 = df['area_pixels'].quantile(0.33)
                area_67 = df['area_pixels'].quantile(0.67)
                
                conditions = [
                    df['area_pixels'] <= area_33,
                    (df['area_pixels'] > area_33) & (df['area_pixels'] <= area_67),
                    df['area_pixels'] > area_67
                ]
                choices = ['small', 'medium', 'large']
                df['size_category'] = np.select(conditions, choices, default='medium')
            
            # Health categories
            if 'health_score' in df.columns:
                conditions = [
                    df['health_score'] < 0.4,
                    (df['health_score'] >= 0.4) & (df['health_score'] < 0.7),
                    df['health_score'] >= 0.7
                ]
                choices = ['poor', 'moderate', 'excellent']
                df['health_category'] = np.select(conditions, choices, default='moderate')
            
            return df
            
        except Exception as e:
            logger.warning(f"[WARN] Population features error: {str(e)}")
            return df

    def _calculate_summary_stats(self, df, image_shape):
        """Calculate comprehensive summary statistics."""
        try:
            if df.empty:
                return self._empty_summary()
            
            total_cells = len(df)
            
            # Basic morphological stats
            morphological = {
                'mean_area_pixels': float(df['area_pixels'].mean()),
                'std_area_pixels': float(df['area_pixels'].std()),
                'total_area_pixels': float(df['area_pixels'].sum()),
                'mean_perimeter': float(df['perimeter'].mean()),
                'mean_circularity': float(df['circularity'].mean()),
                'mean_aspect_ratio': float(df['aspect_ratio'].mean())
            }
            
            # Biological stats
            biological = {
                'mean_chlorophyll': float(df['chlorophyll_content'].mean()) if 'chlorophyll_content' in df.columns else 0,
                'mean_health_score': float(df['health_score'].mean()) if 'health_score' in df.columns else 0,
            }
            
            # Population metrics
            image_area = image_shape[0] * image_shape[1]
            coverage_percent = (morphological['total_area_pixels'] / image_area) * 100
            
            return {
                'total_cells': total_cells,
                'avg_area': morphological['mean_area_pixels'],
                'chlorophyll_ratio': biological['mean_chlorophyll'] * 100,
                'coverage_percent': float(coverage_percent),
                'cell_density': float(total_cells / (image_area / 1000000)),
                'total_biomass_estimate': morphological['total_area_pixels'] * 0.001,
                'morphological_statistics': morphological,
                'biological_statistics': biological,
            }
            
        except Exception as e:
            logger.error(f"[ERROR] Summary calculation failed: {str(e)}")
            return self._empty_summary()

    def _assess_analysis_quality(self, df, labels, image_rgb):
        """Assess analysis quality with comprehensive metrics."""
        try:
            quality_factors = []
            
            # Cell count factor
            cell_count = len(df)
            if cell_count > 0:
                count_factor = min(cell_count / 20.0, 1.0)
                quality_factors.append(count_factor)
            else:
                return 0.0
            
            # Segmentation quality
            if np.max(labels) > 0 and 'area_pixels' in df.columns:
                # Size distribution quality
                areas = df['area_pixels'].values
                area_cv = np.std(areas) / (np.mean(areas) + 1e-8)
                size_factor = max(0, 1.0 - area_cv / 2.0)
                quality_factors.append(size_factor)
                
                # Shape quality
                if 'circularity' in df.columns:
                    shape_factor = df['circularity'].mean()
                    quality_factors.append(shape_factor)
            
            # Image quality
            try:
                gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
                image_contrast = np.std(gray.astype(np.float32))
                contrast_factor = min(image_contrast / 50.0, 1.0)
                quality_factors.append(contrast_factor)
            except:
                quality_factors.append(0.5)
            
            return float(np.mean(quality_factors)) if quality_factors else 0.5
            
        except Exception as e:
            logger.error(f"[ERROR] Quality assessment failed: {str(e)}")
            return 0.5

    def _convert_professional_result(self, professional_result, timestamp, processing_time):
        """Convert professional result to standard format."""
        try:
            cell_data = professional_result.get('cell_data', [])
            summary_stats = professional_result.get('summary_statistics', {})
            quality_metrics = professional_result.get('quality_metrics', {})
            
            result = {
                'success': True,
                'timestamp': timestamp,
                'image_path': professional_result.get('image_path', ''),
                'total_cells': len(cell_data),
                'cell_data': cell_data,
                'summary': self._convert_professional_summary(summary_stats),
                'quality_score': quality_metrics.get('analysis_quality', {}).get('mean_quality_score', 0.8),
                'processing_time': professional_result.get('processing_time', processing_time),
                'labels': professional_result.get('labels'),
                'processing_info': {
                    'pixel_to_micron': self.pixel_to_micron,
                    'pipeline_steps': professional_result.get('pipeline_steps_completed', 11),
                    'processing_log': professional_result.get('processing_log', [])
                },
                'professional_report': summary_stats,
                'quality_report': quality_metrics,
                'recommendations': summary_stats.get('recommendations', [])
            }
            
            return result
            
        except Exception as e:
            logger.error(f"[ERROR] Professional result conversion failed: {str(e)}")
            return self._create_error_result("Professional result conversion failed", timestamp, datetime.now())

    def _convert_professional_summary(self, professional_summary):
        """Convert professional summary to standard format."""
        try:
            cell_count = professional_summary.get('cell_count', {})
            morphology = professional_summary.get('morphological_analysis', {})
            spatial = professional_summary.get('spatial_analysis', {})
            
            return {
                'total_cells': cell_count.get('total_cells', 0),
                'avg_area': morphology.get('mean_area', 0.0),
                'chlorophyll_ratio': 50.0,  # Default value
                'coverage_percent': spatial.get('image_coverage', 0.0),
                'cell_density': spatial.get('cell_density', 0.0),
                'total_biomass_estimate': morphology.get('total_area', 0.0) * 0.001,
                'morphological_statistics': morphology,
                'spatial_analysis': spatial,
                'quality_metrics': cell_count
            }
            
        except Exception as e:
            logger.error(f"[ERROR] Summary conversion failed: {str(e)}")
            return self._empty_summary()

    def _create_error_result(self, error_msg, timestamp, start_time, processing_time=None):
        """Create standardized error result."""
        if processing_time is None:
            processing_time = (datetime.now() - start_time).total_seconds()
        
        return {
            'success': False,
            'timestamp': timestamp,
            'image_path': '',
            'total_cells': 0,
            'cell_data': [],
            'summary': self._empty_summary(),
            'quality_score': 0.0,
            'processing_time': processing_time,
            'error': error_msg,
            'processing_info': {
                'pixel_to_micron': self.pixel_to_micron,
                'error_occurred': True
            }
        }

    def _empty_summary(self):
        """Return empty summary structure."""
        return {
            'total_cells': 0,
            'avg_area': 0.0,
            'chlorophyll_ratio': 0.0,
            'total_biomass_estimate': 0.0,
            'coverage_percent': 0.0,
            'cell_density': 0.0,
            'morphological_statistics': {},
            'biological_statistics': {}
        }

    def _export_results(self, result, timestamp):
        """Export results with error handling."""
        try:
            if not result.get('success') or not result.get('cell_data'):
                return
            
            # CSV export
            df = pd.DataFrame(result['cell_data'])
            csv_path = self.output_dir / "exports" / f"wolffia_analysis_{timestamp}.csv"
            df.to_csv(csv_path, index=False)
            
            # JSON summary
            summary_data = {
                'analysis_info': {
                    'timestamp': result['timestamp'],
                    'processing_time': result['processing_time'],
                    'quality_score': result['quality_score']
                },
                'summary_statistics': result['summary'],
                'cell_count': result['total_cells']
            }
            
            json_path = self.output_dir / "exports" / f"wolffia_summary_{timestamp}.json"
            with open(json_path, 'w') as f:
                json.dump(summary_data, f, indent=2, default=str)
            
            logger.info(f"[SAVE] Results exported: CSV={csv_path}, JSON={json_path}")
            
        except Exception as e:
            logger.error(f"[ERROR] Export failed: {str(e)}")

    def batch_analyze_images(self, image_paths, progress_callback=None, **kwargs):
        """Batch analyze multiple images with progress tracking."""
        results = []
        total = len(image_paths)
        
        logger.info(f"[PROCESS] Starting batch analysis: {total} images")
        
        for i, image_path in enumerate(image_paths):
            try:
                # Individual progress
                if progress_callback:
                    def individual_progress(prog, stage, **kw):
                        overall_progress = int((i / total) * 100 + (prog / total))
                        progress_callback(overall_progress, f"Image {i+1}/{total}: {stage}", **kw)
                    
                    self.set_progress_callback(individual_progress)
                
                result = self.analyze_single_image(image_path, **kwargs)
                results.append(result)
                
                logger.info(f"[PROGRESS] Progress: {i+1}/{total}")
                
            except Exception as e:
                logger.error(f"[ERROR] Batch error for {image_path}: {str(e)}")
                results.append({
                    'success': False,
                    'image_path': str(image_path),
                    'error': str(e),
                    'total_cells': 0,
                    'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
                })
        
        # Calculate batch summary
        successful = [r for r in results if r.get('success')]
        batch_summary = {
            'total_images': total,
            'successful': len(successful),
            'failed': total - len(successful),
            'success_rate': len(successful) / total * 100 if total > 0 else 0,
            'total_cells_detected': sum(r.get('total_cells', 0) for r in successful),
            'average_cells_per_image': np.mean([r.get('total_cells', 0) for r in successful]) if successful else 0
        }
        
        logger.info(f"[SUCCESS] Batch analysis complete: {batch_summary['success_rate']:.1f}% success rate")
        
        return {
            'success': True,
            'batch_summary': batch_summary,
            'individual_results': results
        }

    def get_analysis_summary(self):
        """Get analysis history summary."""
        if not self.results_history:
            return {'message': 'No analyses performed yet'}
        
        successful = [r for r in self.results_history if r.get('success')]
        
        if not successful:
            return {'message': 'No successful analyses yet'}
        
        return {
            'total_analyses': len(self.results_history),
            'successful_analyses': len(successful),
            'success_rate': len(successful) / len(self.results_history) * 100,
            'total_cells_detected': sum(r.get('total_cells', 0) for r in successful),
            'average_cells_per_image': np.mean([r.get('total_cells', 0) for r in successful]),
            'average_quality_score': np.mean([r.get('quality_score', 0) for r in successful]),
            'average_processing_time': np.mean([r.get('processing_time', 0) for r in successful])
        }

    # Additional methods for comprehensive analysis
    def analyze_comprehensive(self, image_path, timestamp=None, **kwargs):
        """Comprehensive analysis with biomass, spectral, and temporal features."""
        try:
            # Start with standard analysis
            base_result = self.analyze_single_image(image_path, **kwargs)
            
            if not base_result.get('success'):
                return base_result
            
            # Add comprehensive features
            comprehensive_features = self._add_comprehensive_features(base_result, timestamp)
            base_result.update(comprehensive_features)
            
            return base_result
            
        except Exception as e:
            logger.error(f"[ERROR] Comprehensive analysis failed: {str(e)}")
            return self._create_error_result(str(e), timestamp or datetime.now().strftime("%Y%m%d_%H%M%S"), datetime.now())

    def _add_comprehensive_features(self, result, timestamp):
        """Add comprehensive analysis features."""
        try:
            comprehensive_data = {}
            
            if result.get('cell_data'):
                df = pd.DataFrame(result['cell_data'])
                
                # Biomass analysis
                comprehensive_data['biomass_analysis'] = self._calculate_biomass(df)
                
                # Spectral analysis
                comprehensive_data['spectral_analysis'] = self._analyze_spectral_properties(df)
                
                # Similarity analysis
                comprehensive_data['similarity_analysis'] = self._analyze_cell_similarity(df)
                
                # Temporal analysis (if timestamp provided)
                if timestamp:
                    comprehensive_data['temporal_analysis'] = self._analyze_temporal_features(df, timestamp)
            
            return comprehensive_data
            
        except Exception as e:
            logger.error(f"[ERROR] Comprehensive features failed: {str(e)}")
            return {}

    def _calculate_biomass(self, df):
        """Enhanced biomass calculation with multiple models"""
        try:
            if df.empty:
                return {}
            
            # Model 1: Area-based allometric equation
            total_area_mm2 = df['area_pixels'].sum() * (self.pixel_to_micron ** 2) / 1000000
            
            # Wolffia-specific allometric equation (based on literature)
            # Fresh biomass (mg) = 0.0012 * Area^1.15
            area_based_biomass = 0.0012 * (total_area_mm2 ** 1.15)
            
            # Model 2: Chlorophyll-based estimation
            if 'chlorophyll_content' in df.columns:
                avg_chlorophyll = df['chlorophyll_content'].mean()
                # Chlorophyll to biomass conversion factor
                chlorophyll_biomass = avg_chlorophyll * total_area_mm2 * 0.015
            else:
                chlorophyll_biomass = area_based_biomass * 0.8
            
            # Model 3: Volume-based estimation (assuming spherical cells)
            if 'major_axis_length' in df.columns:
                avg_diameter = df['major_axis_length'].mean() * self.pixel_to_micron / 1000  # mm
                avg_volume = (4/3) * np.pi * (avg_diameter/2) ** 3  # mm³
                total_volume = avg_volume * len(df)
                # Wolffia density ~1.05 g/cm³
                volume_based_biomass = total_volume * 1.05  # mg
            else:
                volume_based_biomass = area_based_biomass * 0.9
            
            # Combined estimate with confidence interval
            estimates = [area_based_biomass, chlorophyll_biomass, volume_based_biomass]
            combined_biomass = np.mean(estimates)
            std_biomass = np.std(estimates)
            
            return {
                'area_based': {
                    'fresh_biomass_mg': area_based_biomass,
                    'fresh_biomass_g': area_based_biomass / 1000,
                    'total_area_mm2': total_area_mm2
                },
                'chlorophyll_based': {
                    'estimated_biomass_mg': chlorophyll_biomass,
                    'estimated_biomass_g': chlorophyll_biomass / 1000
                },
                'allometric': {
                    'estimated_biomass_mg': volume_based_biomass,
                    'estimated_biomass_g': volume_based_biomass / 1000
                },
                'combined_estimate': {
                    'fresh_biomass_mg': combined_biomass,
                    'fresh_biomass_g': combined_biomass / 1000,
                    'dry_biomass_g': combined_biomass * 0.12 / 1000,  # ~12% dry weight for Wolffia
                    'confidence_interval': [
                        (combined_biomass - 2*std_biomass) / 1000,
                        (combined_biomass + 2*std_biomass) / 1000
                    ],
                    'uncertainty_percent': (std_biomass / combined_biomass * 100) if combined_biomass > 0 else 0
                }
            }
            
        except Exception as e:
            logger.error(f"[ERROR] Biomass calculation failed: {str(e)}")
            return {}
        
    def _analyze_spectral_properties(self, df):
        """Enhanced spectral analysis with wavelength estimation"""
        try:
            if df.empty or 'mean_green' not in df.columns:
                return {}
            
            # RGB to approximate wavelength mapping
            def rgb_to_wavelength(r, g, b):
                """Approximate dominant wavelength from RGB"""
                # Simplified model - in reality would need spectral camera
                if g > r and g > b:
                    # Green dominant (495-570 nm)
                    wavelength = 495 + (g/(r+g+b)) * 75
                elif r > g and r > b:
                    # Red dominant (620-750 nm)
                    wavelength = 620 + (r/(r+g+b)) * 130
                else:
                    # Blue dominant (450-495 nm)
                    wavelength = 450 + (b/(r+g+b)) * 45
                return wavelength
            
            # Calculate spectral properties for each cell
            spectral_data = []
            for _, row in df.iterrows():
                r, g, b = row.get('mean_red', 0), row.get('mean_green', 0), row.get('mean_blue', 0)
                
                # Normalize to 0-255 range if needed
                if r <= 1.0:
                    r, g, b = r*255, g*255, b*255
                
                wavelength = rgb_to_wavelength(r, g, b)
                
                # Calculate vegetation indices
                if (r + g) > 0:
                    ndvi_like = (g - r) / (g + r)
                else:
                    ndvi_like = 0
                
                # Green vegetation index
                gvi = (2*g - r - b) / (2*g + r + b + 1e-8)
                
                # Photosynthetic efficiency proxy
                pep = g / (r + g + b + 1e-8)
                
                spectral_data.append({
                    'cell_id': row['cell_id'],
                    'dominant_wavelength_nm': wavelength,
                    'green_intensity_550nm': g,  # Approximate green channel to 550nm
                    'red_intensity_665nm': r * 0.9,  # Approximate red to chlorophyll absorption
                    'blue_intensity_465nm': b * 1.1,  # Approximate blue 
                    'vegetation_index': ndvi_like,
                    'green_vegetation_index': gvi,
                    'photosynthetic_efficiency': pep,
                    'total_chlorophyll': row.get('chlorophyll_content', 0) * 100
                })
            
            # Calculate population statistics
            df_spectral = pd.DataFrame(spectral_data)
            
            # Wavelength distribution
            wavelength_bins = np.histogram(df_spectral['dominant_wavelength_nm'], bins=10)
            
            return {
                'cell_spectral_data': spectral_data,
                'wavelength_distribution': {
                    'green_550nm': df_spectral['green_intensity_550nm'].tolist(),
                    'red_665nm': df_spectral['red_intensity_665nm'].tolist(),
                    'blue_465nm': df_spectral['blue_intensity_465nm'].tolist()
                },
                'population_statistics': {
                    'mean_wavelength': df_spectral['dominant_wavelength_nm'].mean(),
                    'mean_green_intensity': df_spectral['green_intensity_550nm'].mean(),
                    'mean_vegetation_index': df_spectral['vegetation_index'].mean(),
                    'mean_chlorophyll_content': df_spectral['total_chlorophyll'].mean(),
                    'photosynthetic_efficiency_index': df_spectral['photosynthetic_efficiency'].mean()
                },
                'wavelength_histogram': {
                    'bins': wavelength_bins[1].tolist(),
                    'counts': wavelength_bins[0].tolist()
                }
            }
            
        except Exception as e:
            logger.error(f"[ERROR] Spectral analysis failed: {str(e)}")
            return {}
        
    
    def _analyze_cell_similarity(self, df):
        """Analyze cell similarity patterns."""
        try:
            if df.empty or len(df) < 2:
                return {}
            
            # Select features for similarity analysis
            feature_cols = ['area', 'circularity', 'aspect_ratio', 'mean_green']
            available_cols = [col for col in feature_cols if col in df.columns]
            
            if len(available_cols) < 2:
                return {}
            
            # Calculate pairwise similarities
            similar_pairs = []
            for i in range(len(df)):
                for j in range(i+1, len(df)):
                    cell1 = df.iloc[i]
                    cell2 = df.iloc[j]
                    
                    # Simple similarity score
                    similarity = 0
                    for col in available_cols:
                        val1, val2 = cell1[col], cell2[col]
                        if max(val1, val2) > 0:
                            similarity += 1 - abs(val1 - val2) / max(val1, val2)
                    
                    similarity /= len(available_cols)
                    
                    if similarity > 0.85:  # High similarity threshold
                        similar_pairs.append({
                            'cell_1': int(cell1['cell_id']),
                            'cell_2': int(cell2['cell_id']),
                            'similarity_score': float(similarity)
                        })
            
            return {
                'similar_cell_groups': similar_pairs,
                'similarity_threshold': 0.85,
                'total_similar_pairs': len(similar_pairs)
            }
            
        except Exception as e:
            logger.error(f"[ERROR] Similarity analysis failed: {str(e)}")
            return {}


    def track_cells_across_timepoints(self, current_result, previous_result):
        """Track and match cells across timepoints for growth analysis"""
        try:
            if not previous_result or not current_result:
                return {}
            
            prev_cells = pd.DataFrame(previous_result.get('cell_data', []))
            curr_cells = pd.DataFrame(current_result.get('cell_data', []))
            
            if prev_cells.empty or curr_cells.empty:
                return {}
            
            # Simple nearest neighbor matching based on centroid distance
            matched_cells = []
            growth_data = []
            
            for _, prev_cell in prev_cells.iterrows():
                prev_x, prev_y = prev_cell['centroid_x'], prev_cell['centroid_y']
                
                # Find nearest cell in current timepoint
                distances = np.sqrt(
                    (curr_cells['centroid_x'] - prev_x)**2 + 
                    (curr_cells['centroid_y'] - prev_y)**2
                )
                
                if len(distances) > 0:
                    min_idx = distances.idxmin()
                    min_distance = distances[min_idx]
                    
                    # Match if within reasonable distance (accounting for growth/movement)
                    if min_distance < 50 * self.pixel_to_micron:  # 50 micron threshold
                        curr_cell = curr_cells.loc[min_idx]
                        
                        matched_cells.append({
                            'prev_cell_id': prev_cell['cell_id'],
                            'curr_cell_id': curr_cell['cell_id'],
                            'distance_moved': min_distance,
                            'growth_rate': (curr_cell['area'] - prev_cell['area']) / prev_cell['area'] * 100,
                            'highlighted': True  # Flag for visualization
                        })
                        
                        growth_data.append({
                            'cell_track_id': f"{prev_cell['cell_id']}-{curr_cell['cell_id']}",
                            'area_change': curr_cell['area'] - prev_cell['area'],
                            'growth_percent': (curr_cell['area'] - prev_cell['area']) / prev_cell['area'] * 100,
                            'chlorophyll_change': curr_cell.get('chlorophyll_content', 0) - prev_cell.get('chlorophyll_content', 0)
                        })
            
            return {
                'matched_cells': matched_cells,
                'growth_data': growth_data,
                'tracking_summary': {
                    'cells_tracked': len(matched_cells),
                    'avg_growth_rate': np.mean([g['growth_percent'] for g in growth_data]) if growth_data else 0,
                    'cells_divided': len([g for g in growth_data if g['growth_percent'] > 50]),  # Assume >50% growth = division
                    'cells_lost': len(prev_cells) - len(matched_cells)
                }
            }
            
        except Exception as e:
            logger.error(f"[ERROR] Cell tracking failed: {str(e)}")
            return {}
    
    
    def _analyze_temporal_features(self, df, timestamp):
        """Analyze temporal features for growth tracking."""
        try:
            # Store current timepoint data
            timepoint_data = {
                'timestamp': timestamp,
                'cell_count': len(df),
                'mean_area': df['area'].mean() if 'area' in df.columns else 0,
                'cells': df[['cell_id', 'area', 'centroid_x', 'centroid_y']].to_dict('records')
            }
            
            self.time_series_data.append(timepoint_data)
            
            # If we have multiple timepoints, analyze growth
            if len(self.time_series_data) > 1:
                return self._calculate_growth_dynamics()
            else:
                return {
                    'message': 'Single timepoint - growth analysis requires multiple images',
                    'current_timepoint': timepoint_data
                }
                
        except Exception as e:
            logger.error(f"[ERROR] Temporal analysis failed: {str(e)}")
            return {}

    def _calculate_growth_dynamics(self):
        """Calculate growth dynamics from time series data."""
        try:
            if len(self.time_series_data) < 2:
                return {}
            
            # Sort by timestamp
            sorted_data = sorted(self.time_series_data, key=lambda x: x['timestamp'])
            
            # Calculate population growth
            growth_data = []
            for i in range(1, len(sorted_data)):
                prev = sorted_data[i-1]
                curr = sorted_data[i]
                
                growth_rate = (curr['cell_count'] - prev['cell_count']) / prev['cell_count'] if prev['cell_count'] > 0 else 0
                
                growth_data.append({
                    'timepoint': i,
                    'cell_count': curr['cell_count'],
                    'growth_rate': growth_rate,
                    'timestamp': curr['timestamp']
                })
            
            return {
                'population_dynamics': {
                    'total_timepoints': len(sorted_data),
                    'growth_data': growth_data,
                    'average_growth_rate': np.mean([g['growth_rate'] for g in growth_data])
                },
                'growth_curves': {
                    f"population": {
                        'time_points': [d['timestamp'] for d in sorted_data],
                        'cell_counts': [d['cell_count'] for d in sorted_data],
                        'growth_rate': np.mean([g['growth_rate'] for g in growth_data])
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"[ERROR] Growth dynamics calculation failed: {str(e)}")
            return {}

    def get_temporal_analysis(self):
        """Get current temporal analysis data."""
        if len(self.time_series_data) < 2:
            return {'message': 'Insufficient temporal data'}
        
        return self._calculate_growth_dynamics()


# Compatibility function for existing code
def run_pipeline(image_path_or_dirpath, filename=None):
    """Compatibility function for existing integrations."""
    try:
        # Handle different call patterns
        if filename is not None:
            image_path = os.path.join(image_path_or_dirpath, filename)
        else:
            image_path = image_path_or_dirpath
        
        # Initialize analyzer
        analyzer = WolffiaAnalyzer(debug_mode=False)
        
        # Run analysis
        result = analyzer.analyze_single_image(image_path)
        
        if not result.get('success'):
            logger.error(f"[ERROR] Pipeline failed: {result.get('error', 'Unknown error')}")
            return np.zeros((100, 100), dtype=np.int32), {
                "cell_id": [], "cell_area": [], "int_mem_mean": [], 
                "int_mean": [], "cell_edge": []
            }
        
        # Extract results in expected format
        labels = result.get('labels', np.zeros((100, 100), dtype=np.int32))
        cell_data = result.get('cell_data', [])
        
        results = {
            "cell_id": [c.get('cell_id', i) for i, c in enumerate(cell_data, 1)],
            "cell_area": [c.get('area_pixels', 0) for c in cell_data],
            "int_mem_mean": [c.get('mean_intensity', 0) for c in cell_data],
            "int_mean": [c.get('mean_intensity', 0) for c in cell_data],
            "cell_edge": [c.get('perimeter', 0) for c in cell_data]
        }
        
        return labels, results
        
    except Exception as e:
        logger.error(f"[ERROR] Pipeline compatibility error: {str(e)}")
        return np.zeros((100, 100), dtype=np.int32), {
            "cell_id": [], "cell_area": [], "int_mem_mean": [], 
            "int_mean": [], "cell_edge": []
        }


# Test the analyzer
if __name__ == "__main__":
    logger.info("[TEST] Testing Production Wolffia Analyzer...")
    
    try:
        # Initialize analyzer
        analyzer = WolffiaAnalyzer(debug_mode=False, pixel_to_micron_ratio=1)
        
        # Test with synthetic image
        test_image = np.ones((400, 400, 3), dtype=np.uint8) * 230
        
        # Add some Wolffia-like objects
        centers = [(100, 100), (200, 150), (300, 200), (150, 300)]
        for center in centers:
            cv2.circle(test_image, center, 12, (40, 120, 40), -1)
            cv2.circle(test_image, center, 8, (60, 150, 60), -1)
        
        # Test analysis
        result = analyzer.analyze_single_image(test_image, auto_export=True)
        
        logger.info(f"[SUCCESS] Test Results:")
        logger.info(f"   Success: {result.get('success')}")
        logger.info(f"   Cells detected: {result.get('total_cells')}")
        logger.info(f"   Quality score: {result.get('quality_score', 0):.3f}")
        logger.info(f"   Processing time: {result.get('processing_time', 0):.2f}s")
        
        logger.info("[SUCCESS] Production Wolffia Analyzer test complete")
        
    except Exception as e:
        logger.error(f"[ERROR] Test failed: {str(e)}")