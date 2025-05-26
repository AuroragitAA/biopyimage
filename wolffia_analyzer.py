"""
wolffia_analyzer.py - Production-Ready Wolffia Analysis System
Enhanced for real-world Wolffia arrhiza analysis with robust error handling and standardized interfaces
"""

import json
import logging
import os
import warnings
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# Core scientific computing
import matplotlib.pyplot as plt
from skimage import measure

# Import enhanced components with proper fallbacks
IMAGE_PROCESSOR_AVAILABLE = False
SEGMENTATION_AVAILABLE = False

try:
    from image_processor import ImageProcessor
    IMAGE_PROCESSOR_AVAILABLE = True
    print("✅ Professional image processor loaded")
except ImportError as e:
    print(f"⚠️ Professional image processor not available: {e}")

try:
    from segmentation import WolffiaSpecificSegmentation
    # Test if we can actually instantiate it
    test_seg = WolffiaSpecificSegmentation()
    SEGMENTATION_AVAILABLE = True
    print("✅ Professional segmentation loaded")
except ImportError as e:
    print(f"⚠️ Professional segmentation not available: {e}")
except Exception as e:
    print(f"⚠️ Segmentation test failed: {e}")
    SEGMENTATION_AVAILABLE = False

# Also try to import the run_pipeline function separately
try:
    from segmentation import run_pipeline as segmentation_pipeline
    PIPELINE_AVAILABLE = True
except ImportError:
    PIPELINE_AVAILABLE = False
    segmentation_pipeline = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WolffiaAnalyzer:
    """Production-ready Wolffia analysis system with standardized interface"""
    
    def __init__(self, pixel_to_micron_ratio=1.0, debug_mode=False, output_dir="outputs", **kwargs):
        self.pixel_to_micron = pixel_to_micron_ratio
        self.debug_mode = debug_mode
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Additional parameters
        self.chlorophyll_threshold = kwargs.get('chlorophyll_threshold', 0.6)
        self.min_cell_area = kwargs.get('min_cell_area', 30)
        self.max_cell_area = kwargs.get('max_cell_area', 8000)
        
        # Create subdirectories
        (self.output_dir / "debug_images").mkdir(exist_ok=True)
        (self.output_dir / "results").mkdir(exist_ok=True)
        (self.output_dir / "exports").mkdir(exist_ok=True)
        
        # Initialize components with better error handling
        self.image_processor = None
        self.segmenter = None
        
        if IMAGE_PROCESSOR_AVAILABLE:
            try:
                self.image_processor = ImageProcessor()
                logger.info("✅ Image processor initialized")
            except Exception as e:
                logger.warning(f"⚠️ Image processor initialization failed: {e}")
                self.image_processor = None
        
        if SEGMENTATION_AVAILABLE:
            try:
                self.segmenter = WolffiaSpecificSegmentation(
                    min_area=self.min_cell_area, 
                    max_area=self.max_cell_area, 
                    debug_mode=debug_mode
                )
                logger.info("✅ Advanced segmentation initialized")
            except Exception as e:
                logger.warning(f"⚠️ Advanced segmentation initialization failed: {e}")
                self.segmenter = None
        
        # Results storage
        self.results_history = []
        
        # Progress callback for WebSocket updates
        self.progress_callback = None
        
        logger.info("🔬 Production Wolffia Analyzer initialized")
        logger.info(f"   Debug mode: {debug_mode}")
        logger.info(f"   Output directory: {self.output_dir}")
    
    def set_progress_callback(self, callback):
        """Set callback function for progress updates (for WebSocket integration)"""
        self.progress_callback = callback
    
    def _emit_progress(self, progress, stage, **kwargs):
        """Emit progress update if callback is set"""
        if self.progress_callback:
            try:
                self.progress_callback(progress, stage, **kwargs)
            except Exception as e:
                logger.warning(f"Progress callback error: {e}")
    
    def analyze_single_image(self, image_path, **kwargs) -> dict:
        """
        Main analysis pipeline for single Wolffia image - STANDARDIZED INTERFACE
        
        Args:
            image_path: Path to image file or numpy array
            **kwargs: Additional parameters (pixel_ratio, etc.)
        
        Returns:
            dict: Standardized analysis results
        """
        start_time = datetime.now()
        timestamp = start_time.strftime("%Y%m%d_%H%M%S")
        
        logger.info(f"🔍 Starting Wolffia analysis: {image_path}")
        self._emit_progress(0, "Starting analysis...")
        
        try:
            # Update parameters from kwargs
            pixel_ratio = float(kwargs.get('pixel_ratio', self.pixel_to_micron))
            self.pixel_to_micron = pixel_ratio
            
            # Step 1: Load and validate image
            self._emit_progress(10, "Loading and validating image...")
            image_rgb = self._load_and_validate_image(image_path)
            if image_rgb is None:
                raise ValueError("Failed to load image")
            
            logger.info(f"✅ Image loaded: {image_rgb.shape}")
            
            # Step 2: Image preprocessing
            self._emit_progress(20, "Preprocessing image...")
            if self.image_processor:
                processed_result = self.image_processor.preprocess_image(image_rgb)
                if processed_result:
                    original, gray, green_channel, chlorophyll_enhanced, hsv = processed_result
                else:
                    raise RuntimeError("Image preprocessing failed")
            else:
                # Fallback preprocessing
                original = image_rgb
                gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY) / 255.0
                green_channel = image_rgb[:,:,1] / 255.0
                chlorophyll_enhanced = green_channel - 0.5 * (image_rgb[:,:,0]/255.0 + image_rgb[:,:,2]/255.0)
                chlorophyll_enhanced = np.clip(chlorophyll_enhanced, 0, 1)
                hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
            
            # Step 3: Segmentation with robust fallback
            self._emit_progress(40, "Segmenting cells...")
            debug_path = None
            if self.debug_mode:
                debug_path = self.output_dir / "debug_images" / f"debug_{timestamp}.png"
            
            labels = None
            seg_info = {}
            
            # Try advanced segmentation first
            if self.segmenter:
                try:
                    labels, seg_info = self.segmenter.segment_wolffia_cells(image_rgb, debug_path)
                    if seg_info.get('error'):
                        logger.warning(f"⚠️ Advanced segmentation failed: {seg_info['error']}")
                        labels = None
                except Exception as e:
                    logger.warning(f"⚠️ Advanced segmentation exception: {e}")
                    labels = None
            
            # Try pipeline function if advanced failed
            if labels is None and PIPELINE_AVAILABLE and segmentation_pipeline:
                try:
                    logger.info("🔄 Trying pipeline segmentation...")
                    labels, pipeline_results = segmentation_pipeline(image_rgb, debug_mode=self.debug_mode)
                    logger.info(f"✅ Pipeline segmentation: {np.max(labels)} cells")
                except Exception as e:
                    logger.warning(f"⚠️ Pipeline segmentation failed: {e}")
                    labels = None
            
            # Final fallback to basic segmentation
            if labels is None:
                logger.info("🔄 Using basic fallback segmentation...")
                labels = self._basic_segmentation(gray)
            
            if labels is None:
                raise RuntimeError("All segmentation methods failed")
            
            self._emit_progress(60, "Extracting features...")
            cell_count = np.max(labels)
            logger.info(f"✅ Segmentation complete: {cell_count} cells detected")
            
            # Step 4: Feature extraction
            df = self._extract_comprehensive_features(labels, image_rgb, green_channel)
            
            if df.empty:
                logger.warning("⚠️ No features extracted")
            else:
                logger.info(f"✅ Features extracted: {len(df)} cells")
            
            # Step 5: Calculate summary statistics
            self._emit_progress(80, "Calculating statistics...")
            summary = self._calculate_summary_stats(df, image_rgb.shape)
            
            # Step 6: Quality assessment
            quality_score = self._assess_analysis_quality(df, labels, image_rgb)
            
            # Step 7: Create result structure
            self._emit_progress(90, "Finalizing results...")
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
                'processing_info': {
                    'pixel_to_micron': pixel_ratio,
                    'debug_mode': self.debug_mode,
                    'image_dimensions': f"{image_rgb.shape[1]}x{image_rgb.shape[0]}",
                    'segmentation_method': 'wolffia_specific'
                }
            }
            
            # Step 8: Export results if requested
            if kwargs.get('auto_export', False):
                self._export_results(result, timestamp)
            
            # Store in history
            self.results_history.append(result)
            
            self._emit_progress(100, "Analysis complete!")
            logger.info(f"✅ Analysis complete in {processing_time:.2f}s")
            logger.info(f"   Quality score: {quality_score:.3f}")
            logger.info(f"   Cells detected: {len(df)}")
            
            return result
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"❌ Analysis failed: {str(e)}")
            
            error_result = {
                'success': False,
                'timestamp': timestamp,
                'image_path': str(image_path),
                'total_cells': 0,
                'cell_data': [],
                'summary': self._empty_summary(),
                'quality_score': 0.0,
                'processing_time': processing_time,
                'error': str(e),
                'processing_info': {
                    'pixel_to_micron': self.pixel_to_micron,
                    'error_occurred': True
                }
            }
            
            self._emit_progress(0, f"Analysis failed: {str(e)}")
            return error_result
    
    def batch_analyze_images(self, image_paths, progress_callback=None, **kwargs):
        """Analyze multiple images with progress updates"""
        results = []
        total = len(image_paths)
        
        logger.info(f"🔄 Starting batch analysis: {total} images")
        
        for i, image_path in enumerate(image_paths):
            try:
                # Set up individual progress callback
                if progress_callback:
                    def individual_progress(prog, stage, **kw):
                        overall_progress = int((i / total) * 100 + (prog / total))
                        progress_callback(overall_progress, f"Image {i+1}/{total}: {stage}", **kw)
                    
                    self.set_progress_callback(individual_progress)
                
                result = self.analyze_single_image(image_path, **kwargs)
                results.append(result)
                
                logger.info(f"📊 Progress: {i+1}/{total}")
                
            except Exception as e:
                logger.error(f"❌ Batch error for {image_path}: {str(e)}")
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
        
        logger.info(f"✅ Batch analysis complete: {batch_summary['success_rate']:.1f}% success rate")
        
        return {
            'success': True,
            'batch_summary': batch_summary,
            'individual_results': results
        }
    
    def _load_and_validate_image(self, image_input):
        """Load and validate image input"""
        try:
            if isinstance(image_input, str):
                # Load from file path
                if not Path(image_input).exists():
                    logger.error(f"❌ Image file not found: {image_input}")
                    return None
                
                image = cv2.imread(str(image_input))
                if image is None:
                    logger.error(f"❌ Could not load image: {image_input}")
                    return None
                
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
            elif isinstance(image_input, np.ndarray):
                # Handle numpy array input
                if len(image_input.shape) == 3 and image_input.shape[2] == 3:
                    image_rgb = image_input
                else:
                    logger.error(f"❌ Invalid image array shape: {image_input.shape}")
                    return None
            else:
                logger.error(f"❌ Invalid image input type: {type(image_input)}")
                return None
            
            # Validate image properties
            if image_rgb.shape[0] < 100 or image_rgb.shape[1] < 100:
                logger.error(f"❌ Image too small: {image_rgb.shape}")
                return None
            
            if image_rgb.shape[0] > 5000 or image_rgb.shape[1] > 5000:
                logger.warning(f"⚠️ Large image, may be slow: {image_rgb.shape}")
            
            return image_rgb
            
        except Exception as e:
            logger.error(f"❌ Image loading error: {str(e)}")
            return None
    
    def _basic_segmentation(self, gray_image):
        """Basic fallback segmentation when advanced segmentation is not available"""
        try:
            logger.info("🔧 Running basic segmentation fallback...")
            
            # Ensure proper format
            if gray_image.max() <= 1.0:
                gray_uint8 = (gray_image * 255).astype(np.uint8)
            else:
                gray_uint8 = gray_image.astype(np.uint8)
            
            # Multiple thresholding approaches
            # 1. Otsu thresholding
            try:
                _, binary_otsu = cv2.threshold(gray_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            except:
                binary_otsu = gray_uint8 > 127
            
            # 2. Adaptive thresholding
            try:
                binary_adaptive = cv2.adaptiveThreshold(
                    gray_uint8, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                    cv2.THRESH_BINARY, 21, 2
                )
            except:
                binary_adaptive = binary_otsu
            
            # Combine thresholds
            binary = ((binary_otsu > 0) | (binary_adaptive > 0)).astype(np.uint8)
            
            # Morphological operations to clean up
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            
            # Remove small objects
            try:
                # Try scikit-image approach
                from skimage.morphology import remove_small_objects
                binary_bool = binary > 0
                cleaned = remove_small_objects(binary_bool, min_size=self.min_cell_area)
                binary = cleaned.astype(np.uint8) * 255
            except:
                # Fallback: simple contour filtering
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                mask = np.zeros_like(binary)
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if self.min_cell_area <= area <= self.max_cell_area:
                        cv2.fillPoly(mask, [contour], 255)
                binary = mask
            
            # Connected components labeling
            try:
                labels = measure.label(binary > 0)
            except:
                # Fallback using OpenCV
                num_labels, labels = cv2.connectedComponents(binary)
            
            # Final size filtering
            if np.max(labels) > 0:
                regions = measure.regionprops(labels)
                valid_labels = []
                
                for region in regions:
                    if self.min_cell_area <= region.area <= self.max_cell_area:
                        # Additional shape filtering
                        if region.eccentricity < 0.9 and region.solidity > 0.5:
                            valid_labels.append(region.label)
                
                # Create filtered label image
                if valid_labels:
                    filtered_labels = np.zeros_like(labels)
                    for i, old_label in enumerate(valid_labels, 1):
                        filtered_labels[labels == old_label] = i
                    labels = filtered_labels
                else:
                    labels = np.zeros_like(labels)
            
            logger.info(f"✅ Basic segmentation complete: {np.max(labels)} cells")
            return labels
            
        except Exception as e:
            logger.error(f"❌ Basic segmentation error: {str(e)}")
            # Return empty labels as last resort
            return np.zeros_like(gray_image, dtype=np.int32)
    
    def _extract_comprehensive_features(self, labels, original_image, green_channel):
        """Extract comprehensive features for each detected cell"""
        try:
            if np.max(labels) == 0:
                return pd.DataFrame()
            
            regions = measure.regionprops(labels, intensity_image=green_channel)
            
            features = []
            for region in regions:
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
                            chlorophyll = feature_dict['mean_green'] - 0.5 * (
                                feature_dict['mean_red'] + feature_dict['mean_blue']
                            )
                            feature_dict['chlorophyll_content'] = max(0, chlorophyll)
                        else:
                            feature_dict.update({
                                'mean_red': 0.5, 'mean_green': 0.5, 'mean_blue': 0.5,
                                'chlorophyll_content': 0.5
                            })
                except:
                    feature_dict.update({
                        'mean_red': 0.5, 'mean_green': 0.5, 'mean_blue': 0.5,
                        'chlorophyll_content': 0.5
                    })
                
                # Health score calculation
                integrity_score = region.solidity * (1.0 - region.eccentricity)
                photosynthetic_activity = min(region.mean_intensity, 1.0)
                feature_dict['health_score'] = (integrity_score + photosynthetic_activity) / 2.0
                
                features.append(feature_dict)
            
            df = pd.DataFrame(features)
            
            # Add population-level features
            if len(df) > 0:
                df = self._add_population_features(df)
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Feature extraction error: {str(e)}")
            return pd.DataFrame()
    
    def _add_population_features(self, df):
        """Add population-level derived features"""
        try:
            # Size categories
            if 'area_pixels' in df.columns:
                area_33 = df['area_pixels'].quantile(0.33)
                area_67 = df['area_pixels'].quantile(0.67)
                
                conditions = [
                    df['area_pixels'] <= area_33,
                    (df['area_pixels'] > area_33) & (df['area_pixels'] <= area_67),
                    df['area_pixels'] > area_67
                ]
                choices = ['small', 'medium', 'large']
                df['size_category'] = np.select(conditions, choices, default='medium')
            
            # Health categories based on health score
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
            logger.error(f"❌ Population features error: {str(e)}")
            return df
    
    def _calculate_summary_stats(self, df, image_shape):
        """Calculate comprehensive summary statistics"""
        try:
            if df.empty:
                return self._empty_summary()
            
            # Basic counts
            total_cells = len(df)
            
            # Morphological stats
            morphological = {
                'mean_area_pixels': float(df['area_pixels'].mean()),
                'std_area_pixels': float(df['area_pixels'].std()),
                'median_area_pixels': float(df['area_pixels'].median()),
                'total_area_pixels': float(df['area_pixels'].sum()),
                'mean_area_microns': float(df['area_microns_sq'].mean()) if 'area_microns_sq' in df.columns else 0,
                'mean_perimeter': float(df['perimeter'].mean()),
                'mean_circularity': float(df['circularity'].mean()),
                'mean_aspect_ratio': float(df['aspect_ratio'].mean())
            }
            
            # Biological stats
            biological = {
                'mean_chlorophyll': float(df['chlorophyll_content'].mean()) if 'chlorophyll_content' in df.columns else 0,
                'std_chlorophyll': float(df['chlorophyll_content'].std()) if 'chlorophyll_content' in df.columns else 0,
                'mean_health_score': float(df['health_score'].mean()) if 'health_score' in df.columns else 0,
            }
            
            # Size distribution
            size_dist = {}
            if 'size_category' in df.columns:
                size_counts = df['size_category'].value_counts()
                size_dist = {
                    'small_cells': int(size_counts.get('small', 0)),
                    'medium_cells': int(size_counts.get('medium', 0)),
                    'large_cells': int(size_counts.get('large', 0))
                }
            
            # Health distribution
            health_dist = {}
            if 'health_category' in df.columns:
                health_counts = df['health_category'].value_counts()
                health_dist = {
                    'excellent_cells': int(health_counts.get('excellent', 0)),
                    'moderate_cells': int(health_counts.get('moderate', 0)),
                    'poor_cells': int(health_counts.get('poor', 0))
                }
            
            # Population metrics
            image_area = image_shape[0] * image_shape[1]
            coverage_percent = (morphological['total_area_pixels'] / image_area) * 100
            density = total_cells / (image_area / 1000000)  # cells per square mm (approx)
            
            return {
                'total_cells': total_cells,
                'morphological_statistics': morphological,
                'biological_statistics': biological,
                'size_distribution': size_dist,
                'health_distribution': health_dist,
                'coverage_percent': float(coverage_percent),
                'cell_density': float(density),
                'image_area_pixels': int(image_area),
                
                # Legacy compatibility for existing UI
                'avg_area': morphological['mean_area_pixels'],
                'chlorophyll_ratio': biological['mean_chlorophyll'] * 100,
                'total_biomass_estimate': morphological['total_area_pixels'] * 0.001
            }
            
        except Exception as e:
            logger.error(f"❌ Summary calculation error: {str(e)}")
            return self._empty_summary()
    
    def _assess_analysis_quality(self, df, labels, image_rgb):
        """Assess overall analysis quality"""
        try:
            quality_factors = []
            
            # Cell count factor
            cell_count = len(df)
            if cell_count > 0:
                count_factor = min(cell_count / 20.0, 1.0)  # Normalize by expected count
                quality_factors.append(count_factor)
            else:
                return 0.0
            
            # Segmentation quality
            if np.max(labels) > 0:
                # Check for reasonable size distribution
                if 'area_pixels' in df.columns:
                    areas = df['area_pixels'].values
                    area_cv = np.std(areas) / (np.mean(areas) + 1e-8)
                    size_factor = max(0, 1.0 - area_cv / 2.0)  # Lower CV is better
                    quality_factors.append(size_factor)
                
                # Check for reasonable shapes
                if 'circularity' in df.columns:
                    shape_factor = df['circularity'].mean()
                    quality_factors.append(shape_factor)
            
            # Image quality factors
            image_contrast = np.std(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY))
            contrast_factor = min(image_contrast / 50.0, 1.0)
            quality_factors.append(contrast_factor)
            
            # Overall quality
            if quality_factors:
                return float(np.mean(quality_factors))
            else:
                return 0.5
                
        except Exception as e:
            logger.error(f"❌ Quality assessment error: {str(e)}")
            return 0.5
    
    def _empty_summary(self):
        """Return empty summary structure"""
        return {
            'total_cells': 0,
            'avg_area': 0.0,
            'chlorophyll_ratio': 0.0,
            'total_biomass_estimate': 0.0,
            'coverage_percent': 0.0,
            'cell_density': 0.0,
            'morphological_statistics': {},
            'biological_statistics': {},
            'size_distribution': {},
            'health_distribution': {}
        }
    
    def _export_results(self, result, timestamp):
        """Export analysis results"""
        try:
            if not result.get('success') or not result.get('cell_data'):
                return
            
            # Export CSV
            df = pd.DataFrame(result['cell_data'])
            csv_path = self.output_dir / "exports" / f"wolffia_analysis_{timestamp}.csv"
            df.to_csv(csv_path, index=False)
            logger.info(f"📊 Results exported to: {csv_path}")
            
            # Export JSON summary
            summary_data = {
                'analysis_info': {
                    'timestamp': result['timestamp'],
                    'image_path': result['image_path'],
                    'processing_time': result['processing_time'],
                    'quality_score': result['quality_score']
                },
                'summary_statistics': result['summary'],
                'cell_count': result['total_cells']
            }
            
            json_path = self.output_dir / "exports" / f"wolffia_summary_{timestamp}.json"
            with open(json_path, 'w') as f:
                json.dump(summary_data, f, indent=2, default=str)
            
            logger.info(f"📋 Summary exported to: {json_path}")
            
        except Exception as e:
            logger.error(f"❌ Export error: {str(e)}")
    
    def get_analysis_summary(self):
        """Get comprehensive analysis history summary"""
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
            'average_processing_time': np.mean([r.get('processing_time', 0) for r in successful]),
            'analysis_period': {
                'first': successful[0]['timestamp'],
                'last': successful[-1]['timestamp']
            }
        }


# Compatibility function for existing integrations
def run_pipeline(image_path_or_dirpath, filename=None):
    """
    Compatibility function for existing integration code
    Supports both run_pipeline(image_path) and run_pipeline(dirpath, filename)
    """
    try:
        # Handle different call patterns
        if filename is not None:
            # Called as run_pipeline(dirpath, filename)
            image_path = os.path.join(image_path_or_dirpath, filename)
        else:
            # Called as run_pipeline(image_path) or with array
            image_path = image_path_or_dirpath
        
        # Initialize analyzer
        analyzer = WolffiaAnalyzer(debug_mode=False)
        
        # Run analysis
        result = analyzer.analyze_single_image(image_path)
        
        if not result.get('success'):
            logger.error(f"Pipeline failed: {result.get('error', 'Unknown error')}")
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
        logger.error(f"❌ Pipeline compatibility error: {str(e)}")
        return np.zeros((100, 100), dtype=np.int32), {
            "cell_id": [], "cell_area": [], "int_mem_mean": [], 
            "int_mean": [], "cell_edge": []
        }


# Test the analyzer
if __name__ == "__main__":
    print("🧪 Testing Production Wolffia Analyzer...")
    
    try:
        # Initialize analyzer
        analyzer = WolffiaAnalyzer(debug_mode=True, pixel_to_micron_ratio=0.5)
        
        # Test with synthetic image
        test_image = np.ones((400, 400, 3), dtype=np.uint8) * 230
        
        # Add some Wolffia-like objects
        centers = [(100, 100), (200, 150), (300, 200), (150, 300)]
        for center in centers:
            cv2.circle(test_image, center, 12, (40, 120, 40), -1)
            cv2.circle(test_image, center, 8, (60, 150, 60), -1)
        
        # Test analysis
        result = analyzer.analyze_single_image(test_image, auto_export=True)
        
        print(f"✅ Test Results:")
        print(f"   Success: {result.get('success')}")
        print(f"   Cells detected: {result.get('total_cells')}")
        print(f"   Quality score: {result.get('quality_score', 0):.3f}")
        print(f"   Processing time: {result.get('processing_time', 0):.2f}s")
        
        # Test summary
        summary = analyzer.get_analysis_summary()
        print(f"   Analysis summary available: {len(summary)} metrics")
        
        print("✅ Production Wolffia Analyzer test complete")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()