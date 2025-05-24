"""
wolffia_analyzer.py

Professional Wolffia Bioimage Analysis System - Main Analyzer
Integrates advanced image processing, segmentation, and biological feature extraction.

This enhanced version combines the best features from the comprehensive analysis modules
to provide professional-grade bioimage analysis for Wolffia specimens.
"""

import numpy as np
import pandas as pd
import cv2
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# Core scientific computing
from scipy import ndimage, stats
from scipy.spatial.distance import pdist, squareform
from skimage import measure, morphology, segmentation, filters, feature, restoration
from skimage.color import rgb2lab, rgb2hsv, rgb2gray
from skimage.exposure import equalize_adapthist

# Import your existing modules
try:
    from image_processor import ImageProcessor
    from segmentation import EnhancedCellSegmentation
    CORE_MODULES_AVAILABLE = True
    print("✅ Core modules imported successfully")
except ImportError as e:
    print(f"⚠️ Core modules not available: {e}")
    CORE_MODULES_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdvancedFeatureExtractor:
    """Enhanced feature extractor with comprehensive biological analysis."""
    
    def __init__(self, pixel_to_micron=1.0, chlorophyll_threshold=0.6):
        self.pixel_to_micron = pixel_to_micron
        self.chlorophyll_threshold = chlorophyll_threshold
        logger.info(f"🧬 AdvancedFeatureExtractor initialized")
    
    def extract_comprehensive_features(self, labels, original_image, green_channel, chlorophyll_enhanced, hsv_image=None):
        """Extract comprehensive biological features for each detected cell."""
        try:
            if np.max(labels) == 0:
                return pd.DataFrame()
            
            # Get region properties
            regions = measure.regionprops(labels, intensity_image=green_channel)
            
            features_list = []
            for region in regions:
                features = self._extract_single_cell_features(region, original_image, green_channel, chlorophyll_enhanced, hsv_image)
                if features:
                    features_list.append(features)
            
            if not features_list:
                return pd.DataFrame()
            
            df = pd.DataFrame(features_list)
            
            # Add derived features and classifications
            df = self._add_derived_features(df)
            df = self._calculate_health_scores(df)
            df = self._classify_cells(df)
            
            logger.info(f"✅ Extracted features for {len(df)} cells")
            return df
            
        except Exception as e:
            logger.error(f"❌ Feature extraction error: {str(e)}")
            return pd.DataFrame()
    
    def _extract_single_cell_features(self, region, original_image, green_channel, chlorophyll_enhanced, hsv_image):
        """Extract comprehensive features for a single cell."""
        try:
            # Basic morphological features
            area_pixels = region.area
            area_microns = area_pixels * (self.pixel_to_micron ** 2)
            perimeter = region.perimeter * self.pixel_to_micron
            
            features = {
                'cell_id': region.label,
                'area': area_pixels,
                'area_microns_sq': area_microns,
                'perimeter': perimeter,
                'centroid_x': region.centroid[1] * self.pixel_to_micron,
                'centroid_y': region.centroid[0] * self.pixel_to_micron,
                'major_axis_length': region.major_axis_length * self.pixel_to_micron,
                'minor_axis_length': region.minor_axis_length * self.pixel_to_micron,
                'eccentricity': region.eccentricity,
                'orientation': region.orientation,
                'solidity': region.solidity,
                'extent': region.extent,
                'euler_number': getattr(region, 'euler_number', 1)
            }
            
            # Shape descriptors
            features.update(self._calculate_shape_descriptors(region))
            
            # Intensity and chlorophyll features
            features.update(self._calculate_intensity_features(region, original_image, green_channel, chlorophyll_enhanced))
            
            # Color analysis if HSV available
            if hsv_image is not None:
                features.update(self._calculate_color_features(region, hsv_image))
            
            # Biological indices
            features.update(self._calculate_biological_indices(features))
            
            return features
            
        except Exception as e:
            logger.error(f"❌ Single cell feature extraction error: {str(e)}")
            return None
    
    def _calculate_shape_descriptors(self, region):
        """Calculate advanced shape descriptors."""
        try:
            # Aspect ratio
            aspect_ratio = region.major_axis_length / (region.minor_axis_length + 1e-8)
            
            # Compactness (isoperimetric quotient)
            compactness = 4 * np.pi * region.area / (region.perimeter ** 2 + 1e-8)
            
            # Circularity
            circularity = 4 * np.pi * region.area / (region.perimeter ** 2 + 1e-8)
            
            # Roundness
            roundness = 4 * region.area / (np.pi * region.major_axis_length ** 2 + 1e-8)
            
            return {
                'aspect_ratio': aspect_ratio,
                'compactness': compactness,
                'circularity': circularity,
                'roundness': roundness,
                'shape_factor': region.perimeter / (2 * np.sqrt(np.pi * region.area) + 1e-8)
            }
            
        except Exception as e:
            logger.error(f"❌ Shape descriptor error: {str(e)}")
            return {}
    
    def _calculate_intensity_features(self, region, original_image, green_channel, chlorophyll_enhanced):
        """Calculate intensity-based features."""
        try:
            mean_intensity = region.mean_intensity
            max_intensity = getattr(region, 'max_intensity', mean_intensity)
            min_intensity = getattr(region, 'min_intensity', mean_intensity)
            
            # Extract cell region for detailed analysis
            cell_mask = region.image
            
            # Chlorophyll analysis
            try:
                if hasattr(region, 'slice') and chlorophyll_enhanced is not None:
                    chlorophyll_values = chlorophyll_enhanced[region.slice][cell_mask]
                    chlorophyll_mean = np.mean(chlorophyll_values) if len(chlorophyll_values) > 0 else 0
                    chlorophyll_std = np.std(chlorophyll_values) if len(chlorophyll_values) > 0 else 0
                else:
                    chlorophyll_mean = mean_intensity
                    chlorophyll_std = 0
            except:
                chlorophyll_mean = mean_intensity
                chlorophyll_std = 0
            
            # RGB color analysis
            try:
                if hasattr(region, 'slice') and original_image is not None:
                    cell_rgb_pixels = original_image[region.slice][cell_mask]
                    if len(cell_rgb_pixels) > 0:
                        mean_rgb = np.mean(cell_rgb_pixels, axis=0)
                        std_rgb = np.std(cell_rgb_pixels, axis=0)
                    else:
                        mean_rgb = [128, 128, 128]
                        std_rgb = [0, 0, 0]
                else:
                    mean_rgb = [128, 128, 128]
                    std_rgb = [0, 0, 0]
            except:
                mean_rgb = [128, 128, 128]
                std_rgb = [0, 0, 0]
            
            return {
                'mean_intensity': mean_intensity,
                'max_intensity': max_intensity,
                'min_intensity': min_intensity,
                'chlorophyll_content': chlorophyll_mean,
                'chlorophyll_variability': chlorophyll_std,
                'intensity_uniformity': 1.0 / (1.0 + chlorophyll_std + 1e-8),
                'mean_red': mean_rgb[0] / 255.0,
                'mean_green': mean_rgb[1] / 255.0,
                'mean_blue': mean_rgb[2] / 255.0,
                'color_variation': np.mean(std_rgb) / (np.mean(mean_rgb) + 1e-8)
            }
            
        except Exception as e:
            logger.error(f"❌ Intensity feature error: {str(e)}")
            return {'mean_intensity': 0.5, 'chlorophyll_content': 0.5}
    
    def _calculate_color_features(self, region, hsv_image):
        """Calculate HSV color space features."""
        try:
            if not hasattr(region, 'slice'):
                return {}
            
            cell_mask = region.image
            hsv_values = hsv_image[region.slice][cell_mask]
            
            if len(hsv_values) == 0:
                return {}
            
            # HSV statistics
            hue_mean = np.mean(hsv_values[:, 0])
            saturation_mean = np.mean(hsv_values[:, 1])
            value_mean = np.mean(hsv_values[:, 2])
            
            return {
                'hue_mean': hue_mean,
                'saturation_mean': saturation_mean,
                'value_mean': value_mean,
                'color_intensity': saturation_mean * value_mean
            }
            
        except Exception as e:
            logger.error(f"❌ Color feature error: {str(e)}")
            return {}
    
    def _calculate_biological_indices(self, features):
        """Calculate biologically relevant indices."""
        try:
            # Cell integrity score
            integrity_score = features.get('solidity', 0.5) * (1.0 - features.get('eccentricity', 0.5))
            
            # Photosynthetic activity estimate
            chlorophyll_content = features.get('chlorophyll_content', 0.5)
            photosynthetic_activity = min(chlorophyll_content, 1.0)
            
            # Cell maturity estimate
            area = features.get('area', 100)
            size_factor = min(area / 1000.0, 1.0)
            shape_factor = 1.0 - abs(features.get('eccentricity', 0.5) - 0.3)
            maturity_estimate = (size_factor + shape_factor) / 2.0
            
            # Stress indicators
            shape_irregularity = 1.0 - features.get('solidity', 0.5)
            size_deviation = abs(area - 500) / 500.0
            stress_indicator = (shape_irregularity + min(size_deviation, 1.0)) / 2.0
            
            return {
                'cell_integrity': integrity_score,
                'photosynthetic_activity': photosynthetic_activity,
                'maturity_estimate': maturity_estimate,
                'stress_indicator': stress_indicator,
                'health_potential': (integrity_score + photosynthetic_activity) / 2.0
            }
            
        except Exception as e:
            logger.error(f"❌ Biological indices error: {str(e)}")
            return {}
    
    def _add_derived_features(self, df):
        """Add derived features based on population statistics."""
        try:
            if len(df) == 0:
                return df
            
            # Size categories
            area_33 = df['area'].quantile(0.33)
            area_67 = df['area'].quantile(0.67)
            
            df['size_category'] = 'medium'
            df.loc[df['area'] <= area_33, 'size_category'] = 'small'
            df.loc[df['area'] > area_67, 'size_category'] = 'large'
            
            # Chlorophyll categories
            if 'chlorophyll_content' in df.columns:
                chl_33 = df['chlorophyll_content'].quantile(0.33)
                chl_67 = df['chlorophyll_content'].quantile(0.67)
                
                df['chlorophyll_category'] = 'medium'
                df.loc[df['chlorophyll_content'] <= chl_33, 'chlorophyll_category'] = 'low'
                df.loc[df['chlorophyll_content'] > chl_67, 'chlorophyll_category'] = 'high'
            
            # Spatial features
            if 'centroid_x' in df.columns and 'centroid_y' in df.columns:
                center_x, center_y = df['centroid_x'].mean(), df['centroid_y'].mean()
                df['distance_from_center'] = np.sqrt(
                    (df['centroid_x'] - center_x)**2 + (df['centroid_y'] - center_y)**2
                )
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Derived features error: {str(e)}")
            return df
    
    def _calculate_health_scores(self, df):
        """Calculate comprehensive health scores."""
        try:
            if len(df) == 0:
                return df
            
            # Health score components
            components = []
            weights = []
            
            if 'chlorophyll_content' in df.columns:
                components.append(df['chlorophyll_content'])
                weights.append(0.3)
            
            if 'cell_integrity' in df.columns:
                components.append(df['cell_integrity'])
                weights.append(0.25)
            
            if 'circularity' in df.columns:
                components.append(df['circularity'])
                weights.append(0.2)
            
            if 'intensity_uniformity' in df.columns:
                components.append(df['intensity_uniformity'])
                weights.append(0.15)
            
            if 'compactness' in df.columns:
                components.append(df['compactness'])
                weights.append(0.1)
            
            # Calculate weighted health score
            if components:
                # Normalize weights
                weights = np.array(weights) / np.sum(weights)
                
                health_scores = np.zeros(len(df))
                for component, weight in zip(components, weights):
                    normalized_component = (component - component.min()) / (component.max() - component.min() + 1e-8)
                    health_scores += weight * normalized_component
                
                df['health_score'] = health_scores
            else:
                df['health_score'] = 0.5
            
            # Health categories
            df['health_category'] = pd.cut(
                df['health_score'], 
                bins=[0, 0.3, 0.7, 1.0], 
                labels=['poor', 'moderate', 'excellent'],
                include_lowest=True
            )
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Health score error: {str(e)}")
            df['health_score'] = 0.5
            df['health_category'] = 'moderate'
            return df
    
    def _classify_cells(self, df):
        """Advanced cell classification based on multiple criteria."""
        try:
            if len(df) == 0:
                return df
            
            # Multi-dimensional classification
            df['cell_type'] = 'unknown'
            
            # Size + Chlorophyll classification
            for idx, row in df.iterrows():
                size_cat = row.get('size_category', 'medium')
                chl_cat = row.get('chlorophyll_category', 'medium')
                health_score = row.get('health_score', 0.5)
                
                if health_score > 0.7:
                    cell_type = f"{size_cat}_healthy"
                elif health_score < 0.3:
                    cell_type = f"{size_cat}_stressed"
                else:
                    cell_type = f"{size_cat}_{chl_cat}"
                
                df.at[idx, 'cell_type'] = cell_type
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Cell classification error: {str(e)}")
            df['cell_type'] = 'unknown'
            return df


class WolffiaAnalyzer:
    """Professional Wolffia Bioimage Analysis System - Main Analyzer."""

    def __init__(self, pixel_to_micron_ratio=1.0, chlorophyll_threshold=0.6, 
                 min_cell_area=30, max_cell_area=8000):
        """Initialize the professional Wolffia analyzer."""
        
        # Core parameters
        self.pixel_to_micron = pixel_to_micron_ratio
        self.chlorophyll_threshold = chlorophyll_threshold
        self.min_cell_area = min_cell_area
        self.max_cell_area = max_cell_area
        
        # Results storage
        self.results_history = []
        
        # Initialize analysis components
        self._initialize_components()
        
        logger.info("🔬 Professional Wolffia Analyzer initialized")
        logger.info(f"   Pixel to micron ratio: {pixel_to_micron_ratio}")
        logger.info(f"   Chlorophyll threshold: {chlorophyll_threshold}")
        logger.info(f"   Cell area range: {min_cell_area}-{max_cell_area} pixels")

    def _initialize_components(self):
        """Initialize all analysis components."""
        try:
            # Core image processing components
            if CORE_MODULES_AVAILABLE:
                self.image_processor = ImageProcessor()
                self.segmentation = EnhancedCellSegmentation(
                    min_area=self.min_cell_area, 
                    max_area=self.max_cell_area
                )
            else:
                # Fallback components
                self.image_processor = self._create_fallback_image_processor()
                self.segmentation = self._create_fallback_segmentation()
            
            # Advanced feature extractor
            self.feature_extractor = AdvancedFeatureExtractor(
                self.pixel_to_micron, 
                self.chlorophyll_threshold
            )
            
            logger.info("✅ All analysis components initialized")
            
        except Exception as e:
            logger.error(f"❌ Component initialization failed: {str(e)}")
            raise

    def _create_fallback_image_processor(self):
        """Create fallback image processor if main one fails."""
        class FallbackImageProcessor:
            def preprocess_image(self, image_input, **kwargs):
                try:
                    if isinstance(image_input, str):
                        image = cv2.imread(image_input)
                        if image is None:
                            return None
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    else:
                        image = image_input
                    
                    original = image.copy()
                    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) / 255.0
                    green_channel = image[:, :, 1] / 255.0
                    
                    # Simple chlorophyll enhancement
                    r, g, b = image[:, :, 0]/255.0, image[:, :, 1]/255.0, image[:, :, 2]/255.0
                    chlorophyll_enhanced = g - 0.5 * (r + b)
                    chlorophyll_enhanced = np.clip(chlorophyll_enhanced, 0, 1)
                    
                    # HSV conversion
                    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV) / 255.0
                    
                    return original, gray, green_channel, chlorophyll_enhanced, hsv
                except:
                    return None
        
        return FallbackImageProcessor()

    def _create_fallback_segmentation(self):
        """Create fallback segmentation if main one fails."""
        class FallbackSegmentation:
            def segment_cells(self, gray, green_channel, chlorophyll_enhanced, method='auto'):
                try:
                    # Simple Otsu thresholding
                    if chlorophyll_enhanced.max() <= 1.0:
                        thresh_img = (chlorophyll_enhanced * 255).astype(np.uint8)
                    else:
                        thresh_img = chlorophyll_enhanced.astype(np.uint8)
                    
                    _, binary = cv2.threshold(thresh_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    
                    # Simple connected components
                    labels = measure.label(binary > 0)
                    
                    return labels
                except:
                    return np.zeros_like(gray, dtype=np.int32)
        
        return FallbackSegmentation()

    def analyze_single_image(self, image_path, timestamp=None, method='auto'):
        """Complete professional analysis pipeline for a single image."""
        
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        logger.info(f"🔍 Starting professional analysis: {image_path}")
        
        start_time = datetime.now()

        try:
            # Step 1: Image Preprocessing
            logger.info("📷 Step 1: Image preprocessing...")
            processed_images = self.image_processor.preprocess_image(
                image_path, 
                enhance_contrast=True, 
                denoise=True
            )
            
            if processed_images is None:
                logger.error("❌ Image preprocessing failed")
                return self._create_failed_result(timestamp, image_path, "Image preprocessing failed")

            original, gray, green_channel, chlorophyll_enhanced, hsv = processed_images
            logger.info("✅ Image preprocessing complete")

            # Step 2: Cell Segmentation
            logger.info("🔬 Step 2: Cell segmentation...")
            labels = self.segmentation.segment_cells(
                gray, green_channel, chlorophyll_enhanced, method=method
            )

            if np.max(labels) == 0:
                logger.warning("⚠️ No cells detected after segmentation")
                return self._create_failed_result(timestamp, image_path, "No cells detected")

            cell_count = np.max(labels)
            logger.info(f"✅ Segmentation complete: {cell_count} regions found")

            # Step 3: Advanced Feature Extraction
            logger.info("🧬 Step 3: Advanced feature extraction...")
            df = self.feature_extractor.extract_comprehensive_features(
                labels, original, green_channel, chlorophyll_enhanced, hsv
            )

            if len(df) == 0:
                logger.warning("⚠️ No features extracted")
                return self._create_failed_result(timestamp, image_path, "Feature extraction failed")

            logger.info(f"✅ Feature extraction complete: {len(df)} cells analyzed")

            # Step 4: Add metadata and timestamps
            df['timestamp'] = timestamp
            df['image_path'] = str(image_path)
            df['analysis_method'] = method

            # Step 5: Calculate comprehensive summary statistics
            summary = self._calculate_comprehensive_summary(df)

            # Step 6: Quality assessment
            quality_score = self._assess_analysis_quality(df, original)

            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()

            # Step 7: Compile comprehensive results
            result = {
                'timestamp': timestamp,
                'image_path': str(image_path),
                'cell_data': df.to_dict('records'),
                'summary': summary,
                'total_cells': len(df),
                'success': True,
                'quality_score': quality_score,
                'processing_time': processing_time,
                'processing_info': {
                    'pixel_to_micron': self.pixel_to_micron,
                    'chlorophyll_threshold': self.chlorophyll_threshold,
                    'min_cell_area': self.min_cell_area,
                    'max_cell_area': self.max_cell_area,
                    'method': method,
                    'image_dimensions': f"{original.shape[1]}x{original.shape[0]}",
                    'total_pixels': original.shape[0] * original.shape[1]
                },
                'labels': labels,
                'quality_details': {
                    'overall_score': quality_score,
                    'cell_count_consistency': self._check_cell_count_consistency(df),
                    'feature_completeness': self._check_feature_completeness(df)
                }
            }

            # Store in history
            self.results_history.append(result)
            
            logger.info(f"✅ Professional analysis complete: {len(df)} cells detected and analyzed")
            logger.info(f"⏱️ Processing time: {processing_time:.2f} seconds")
            logger.info(f"📊 Quality score: {quality_score:.3f}")

            return result

        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"❌ Analysis error: {str(e)}")
            
            return self._create_failed_result(timestamp, image_path, str(e), processing_time)

    def _calculate_comprehensive_summary(self, df):
        """Calculate comprehensive summary statistics."""
        try:
            if len(df) == 0:
                return self._empty_summary()
            
            # Basic statistics
            total_cells = len(df)
            
            # Morphological statistics
            morphological_stats = {
                'mean_area': float(df['area'].mean()),
                'std_area': float(df['area'].std()),
                'median_area': float(df['area'].median()),
                'total_area': float(df['area'].sum()),
                'mean_perimeter': float(df['perimeter'].mean()),
                'mean_circularity': float(df['circularity'].mean()) if 'circularity' in df.columns else 0.0,
                'mean_aspect_ratio': float(df['aspect_ratio'].mean()) if 'aspect_ratio' in df.columns else 1.0
            }
            
            # Biological statistics
            biological_stats = {
                'mean_chlorophyll': float(df['chlorophyll_content'].mean()) if 'chlorophyll_content' in df.columns else 0.5,
                'high_chlorophyll_count': int(len(df[df['chlorophyll_content'] > self.chlorophyll_threshold])) if 'chlorophyll_content' in df.columns else 0,
                'chlorophyll_ratio': float(len(df[df['chlorophyll_content'] > self.chlorophyll_threshold]) / total_cells * 100) if 'chlorophyll_content' in df.columns else 0.0,
                'mean_health_score': float(df['health_score'].mean()) if 'health_score' in df.columns else 0.5,
                'healthy_cells': int(len(df[df['health_score'] > 0.7])) if 'health_score' in df.columns else 0,
                'stressed_cells': int(len(df[df['health_score'] < 0.3])) if 'health_score' in df.columns else 0
            }
            
            # Size distribution
            size_distribution = {
                'small_cells': len(df[df['size_category'] == 'small']) if 'size_category' in df.columns else 0,
                'medium_cells': len(df[df['size_category'] == 'medium']) if 'size_category' in df.columns else total_cells,
                'large_cells': len(df[df['size_category'] == 'large']) if 'size_category' in df.columns else 0
            }
            
            # Health distribution
            health_distribution = {}
            if 'health_category' in df.columns:
                health_counts = df['health_category'].value_counts()
                health_distribution = {
                    'excellent': int(health_counts.get('excellent', 0)),
                    'moderate': int(health_counts.get('moderate', 0)),
                    'poor': int(health_counts.get('poor', 0))
                }
            
            # Biomass estimation
            total_biomass = morphological_stats['total_area'] * 0.001  # Simple conversion
            
            # Compile comprehensive summary
            summary = {
                'total_cells': total_cells,
                'morphological_statistics': morphological_stats,
                'biological_statistics': biological_stats,
                'size_distribution': size_distribution,
                'health_distribution': health_distribution,
                'total_biomass_estimate': total_biomass,
                'population_density': total_cells / (morphological_stats['total_area'] / 1000.0) if morphological_stats['total_area'] > 0 else 0,
                'diversity_index': self._calculate_diversity_index(df)
            }
            
            # Add legacy compatibility fields
            summary.update({
                'avg_area': morphological_stats['mean_area'],
                'chlorophyll_ratio': biological_stats['chlorophyll_ratio']
            })
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ Summary calculation error: {str(e)}")
            return self._empty_summary()

    def _calculate_diversity_index(self, df):
        """Calculate population diversity index."""
        try:
            if 'cell_type' not in df.columns or len(df) == 0:
                return 0.0
            
            # Shannon diversity index
            type_counts = df['cell_type'].value_counts()
            proportions = type_counts / len(df)
            shannon_index = -np.sum(proportions * np.log(proportions + 1e-10))
            
            return float(shannon_index)
            
        except Exception as e:
            logger.error(f"❌ Diversity index error: {str(e)}")
            return 0.0

    def _assess_analysis_quality(self, df, original_image):
        """Assess the quality of the analysis results."""
        try:
            quality_factors = []
            
            # Cell count factor
            cell_count = len(df)
            if cell_count > 5:
                quality_factors.append(min(cell_count / 50.0, 1.0))
            else:
                quality_factors.append(cell_count / 5.0)
            
            # Feature completeness
            expected_features = ['area', 'chlorophyll_content', 'health_score', 'circularity']
            present_features = [f for f in expected_features if f in df.columns]
            completeness = len(present_features) / len(expected_features)
            quality_factors.append(completeness)
            
            # Data consistency
            if len(df) > 0:
                area_cv = df['area'].std() / (df['area'].mean() + 1e-8)
                consistency = max(0, 1.0 - area_cv / 2.0)  # Lower CV = higher consistency
                quality_factors.append(consistency)
            else:
                quality_factors.append(0.0)
            
            # Image quality proxy
            image_quality = min(np.std(original_image) / 50.0, 1.0)
            quality_factors.append(image_quality)
            
            # Overall quality score
            overall_quality = np.mean(quality_factors)
            
            return float(np.clip(overall_quality, 0.0, 1.0))
            
        except Exception as e:
            logger.error(f"❌ Quality assessment error: {str(e)}")
            return 0.5

    def _check_cell_count_consistency(self, df):
        """Check consistency of cell count detection."""
        try:
            return len(df) > 0 and 'area' in df.columns
        except:
            return False

    def _check_feature_completeness(self, df):
        """Check completeness of extracted features."""
        try:
            required_features = ['area', 'chlorophyll_content', 'centroid_x', 'centroid_y']
            return all(feature in df.columns for feature in required_features)
        except:
            return False

    def _create_failed_result(self, timestamp, image_path, error_message, processing_time=0):
        """Create a standardized failed result."""
        return {
            'timestamp': timestamp,
            'image_path': str(image_path),
            'cell_data': [],
            'summary': self._empty_summary(),
            'total_cells': 0,
            'success': False,
            'error': error_message,
            'quality_score': 0.0,
            'processing_time': processing_time,
            'processing_info': {
                'pixel_to_micron': self.pixel_to_micron,
                'method': 'failed'
            }
        }

    def _empty_summary(self):
        """Return empty summary statistics structure."""
        return {
            'total_cells': 0,
            'avg_area': 0.0,
            'total_biomass_estimate': 0.0,
            'chlorophyll_ratio': 0.0,
            'morphological_statistics': {
                'mean_area': 0.0,
                'std_area': 0.0,
                'total_area': 0.0
            },
            'biological_statistics': {
                'mean_chlorophyll': 0.0,
                'healthy_cells': 0,
                'stressed_cells': 0
            },
            'size_distribution': {
                'small_cells': 0,
                'medium_cells': 0,
                'large_cells': 0
            }
        }

    # Professional analysis methods
    def analyze_image_professional(self, image_path, **kwargs):
        """Professional analysis with full feature set."""
        return self.analyze_single_image(image_path, **kwargs)

    def batch_analyze_images(self, image_paths, progress_callback=None):
        """Analyze multiple images with progress tracking."""
        results = []
        total_images = len(image_paths)
        
        logger.info(f"🔄 Starting batch analysis of {total_images} images")
        
        for i, image_path in enumerate(image_paths):
            try:
                result = self.analyze_single_image(image_path)
                results.append(result)
                
                if progress_callback:
                    progress_callback(i + 1, total_images, result)
                    
                logger.info(f"📊 Progress: {i+1}/{total_images} images processed")
                
            except Exception as e:
                logger.error(f"❌ Batch analysis error for {image_path}: {str(e)}")
                results.append(self._create_failed_result(
                    datetime.now().strftime("%Y%m%d_%H%M%S"), 
                    image_path, 
                    str(e)
                ))
        
        logger.info(f"✅ Batch analysis complete: {len(results)} images processed")
        return results

    def get_component_status(self):
        """Get detailed status of all analyzer components."""
        return {
            'image_processor': hasattr(self, 'image_processor') and self.image_processor is not None,
            'segmentation': hasattr(self, 'segmentation') and self.segmentation is not None,
            'feature_extractor': hasattr(self, 'feature_extractor') and self.feature_extractor is not None,
            'core_modules_available': CORE_MODULES_AVAILABLE,
            'results_count': len(self.results_history),
            'last_analysis': self.results_history[-1]['timestamp'] if self.results_history else None
        }

    def get_analysis_summary(self):
        """Get comprehensive summary of all analyses performed."""
        if not self.results_history:
            return {"message": "No analyses performed yet"}

        successful_results = [r for r in self.results_history if r.get('success', False)]
        
        if not successful_results:
            return {"message": "No successful analyses performed yet"}

        total_images = len(successful_results)
        total_cells = sum(r['total_cells'] for r in successful_results)
        avg_quality = np.mean([r.get('quality_score', 0.5) for r in successful_results])
        avg_processing_time = np.mean([r.get('processing_time', 0) for r in successful_results])
        
        return {
            'total_images_analyzed': total_images,
            'total_cells_detected': total_cells,
            'average_cells_per_image': total_cells / total_images if total_images > 0 else 0,
            'success_rate': len(successful_results) / len(self.results_history) * 100,
            'average_quality_score': avg_quality,
            'average_processing_time': avg_processing_time,
            'analysis_period': {
                'first_analysis': successful_results[0]['timestamp'],
                'last_analysis': successful_results[-1]['timestamp']
            }
        }

    def export_results(self, format='csv', output_path=None):
        """Export analysis results in various formats."""
        try:
            if not self.results_history:
                logger.warning("⚠️ No results to export")
                return None
            
            # Collect all cell data
            all_cell_data = []
            for result in self.results_history:
                if result.get('success') and 'cell_data' in result:
                    all_cell_data.extend(result['cell_data'])
            
            if not all_cell_data:
                logger.warning("⚠️ No cell data to export")
                return None
            
            df = pd.DataFrame(all_cell_data)
            
            # Generate output path if not provided
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f"wolffia_analysis_results_{timestamp}.{format}"
            
            # Export based on format
            if format.lower() == 'csv':
                df.to_csv(output_path, index=False)
            elif format.lower() == 'excel':
                df.to_excel(output_path, index=False)
            elif format.lower() == 'json':
                df.to_json(output_path, orient='records', indent=2)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"✅ Results exported to: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"❌ Export error: {str(e)}")
            return None

    def reset_analysis_history(self):
        """Clear all stored analysis results."""
        self.results_history = []
        logger.info("🔄 Analysis history cleared")


# Testing and validation
if __name__ == "__main__":
    print("🧪 Testing Professional Wolffia Analyzer...")
    
    try:
        # Test initialization
        analyzer = WolffiaAnalyzer(
            pixel_to_micron_ratio=0.5,
            chlorophyll_threshold=0.6
        )
        
        # Test component status
        status = analyzer.get_component_status()
        print(f"📊 Component Status: {status}")
        
        # Test summary (should be empty)
        summary = analyzer.get_analysis_summary()
        print(f"📈 Analysis Summary: {summary}")
        
        print("✅ Professional Wolffia Analyzer test complete")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()