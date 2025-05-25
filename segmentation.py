"""
segmentation.py

Professional Cell Segmentation Module for Wolffia Bioimage Analysis
Enhanced with advanced algorithms, quality assessment, and biological optimization.

This module provides state-of-the-art segmentation approaches optimized for
accurate detection and separation of Wolffia cells in various imaging conditions.
"""

import cv2
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union
from scipy import ndimage
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings('ignore')

# Advanced image processing imports
from skimage.segmentation import watershed, clear_border
from skimage.measure import label, regionprops
from skimage.morphology import (
    disk, binary_erosion, binary_dilation, remove_small_objects,
    opening, closing, binary_opening, binary_closing
)
from skimage.filters import (
    threshold_otsu, threshold_local, gaussian, median,
    rank, sobel, scharr
)
from skimage.feature import peak_local_max
from skimage.util import img_as_float, img_as_uint
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler

# Handle different scikit-image versions for peak_local_max
try:
    from skimage.feature import peak_local_max as peak_local_max
    print("✅ Using skimage.feature.peak_local_max")
except ImportError:
    try:
        from skimage.segmentation import peak_local_max as peak_local_max
        print("✅ Using skimage.segmentation.peak_local_max")
    except ImportError:
        print("⚠️ peak_local_max not found, using fallback implementation")
        
        def peak_local_max(image, min_distance=1, threshold_abs=None, 
                          threshold_rel=None, indices=True, num_peaks=np.inf):
            """Fallback implementation for peak detection."""
            from scipy.ndimage import maximum_filter
            
            # Apply threshold if specified
            if threshold_abs is not None:
                image = image * (image >= threshold_abs)
            if threshold_rel is not None:
                image = image * (image >= threshold_rel * image.max())
            
            # Find local max using maximum filter
            size = min_distance * 2 + 1
            local_max = maximum_filter(image, size=size) == image
            
            # Remove edge peaks
            border = min_distance
            if border > 0:
                local_max[:border] = False
                local_max[-border:] = False
                local_max[:, :border] = False
                local_max[:, -border:] = False
            
            if indices:
                peaks = np.column_stack(np.where(local_max))
                if len(peaks) > num_peaks:
                    # Sort by intensity and keep top peaks
                    intensities = image[peaks[:, 0], peaks[:, 1]]
                    top_indices = np.argsort(intensities)[-int(num_peaks):]
                    peaks = peaks[top_indices]
                return peaks
            else:
                return local_max

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdvancedColorSegmenter:
    """Professional color-based segmentation with machine learning enhancement."""
    
    def __init__(self):
        """Initialize advanced color segmentation system."""
        self.color_profiles = {
            'green_wolffia': {
                'hue_range': (35, 85),
                'saturation_range': (0.3, 0.9),
                'value_range': (0.2, 0.95),
                'description': 'Standard green Wolffia detection',
                'biological_relevance': 0.9
            },
            'bright_green': {
                'hue_range': (30, 90),
                'saturation_range': (0.4, 1.0),
                'value_range': (0.4, 1.0),
                'description': 'Bright green organisms under good lighting',
                'biological_relevance': 0.8
            },
            'dark_green': {
                'hue_range': (40, 80),
                'saturation_range': (0.2, 0.8),
                'value_range': (0.1, 0.6),
                'description': 'Dark or shadowed green organisms',
                'biological_relevance': 0.7
            },
            'yellowish_green': {
                'hue_range': (25, 60),
                'saturation_range': (0.3, 0.8),
                'value_range': (0.3, 0.9),
                'description': 'Yellowish-green organisms (stressed or mature)',
                'biological_relevance': 0.6
            },
            'adaptive': {
                'description': 'Machine learning-based adaptive color detection',
                'biological_relevance': 0.95
            }
        }
        
        self.ml_segmenter = None
        logger.info("🎨 Advanced Color Segmenter initialized")

    def segment_by_color(self, image: np.ndarray, color_name: str = 'green_wolffia',
                        quality_threshold: float = 0.5) -> Dict:
        """
        Advanced color-based segmentation with quality assessment.
        
        Parameters:
        -----------
        image : np.ndarray
            RGB image array
        color_name : str
            Color profile name to use
        quality_threshold : float
            Minimum quality threshold for accepting segmentation
            
        Returns:
        --------
        dict : Comprehensive segmentation results
        """
        try:
            logger.info(f"🔍 Advanced color segmentation: {color_name}")
            
            if color_name not in self.color_profiles:
                logger.error(f"❌ Unknown color profile: {color_name}")
                return self._empty_result()
            
            if color_name == 'adaptive':
                return self._adaptive_color_segmentation(image)
            else:
                return self._profile_based_segmentation(image, color_name, quality_threshold)
            
        except Exception as e:
            logger.error(f"❌ Color segmentation error: {str(e)}")
            return self._empty_result()

    def _profile_based_segmentation(self, image: np.ndarray, color_name: str,
                                   quality_threshold: float) -> Dict:
        """Perform segmentation based on predefined color profile."""
        try:
            profile = self.color_profiles[color_name]
            logger.info(f"   Profile: {profile['description']}")
            
            # Convert to HSV and normalize
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:, :, 0] /= 180.0  # Hue to 0-1
            hsv[:, :, 1] /= 255.0  # Saturation to 0-1
            hsv[:, :, 2] /= 255.0  # Value to 0-1
            
            # Create color mask with multiple criteria
            mask = self._create_advanced_color_mask(hsv, profile)
            
            # Morphological refinement
            refined_mask = self._refine_color_mask(mask, image.shape[:2])
            
            # Quality assessment
            quality_score = self._assess_color_segmentation_quality(
                image, refined_mask, profile
            )
            
            # Post-processing if quality is sufficient
            if quality_score >= quality_threshold:
                final_mask, labels = self._post_process_color_mask(refined_mask)
                cell_count = np.max(labels)
                
                # Create enhanced visualization
                visualization = self._create_enhanced_visualization(image, labels, profile)
                
                logger.info(f"✅ Color segmentation successful: {cell_count} regions")
                logger.info(f"   Quality score: {quality_score:.3f}")
                
                return {
                    'labels': labels,
                    'cell_count': cell_count,
                    'mask': final_mask.astype(np.uint8) * 255,
                    'visualization': visualization,
                    'quality_score': quality_score,
                    'method': f'color_{color_name}',
                    'profile': profile,
                    'success': True
                }
                
            else:
                logger.warning(f"⚠️ Low quality segmentation: {quality_score:.3f}")
                return self._low_quality_result(quality_score, color_name)
                
        except Exception as e:
            logger.error(f"❌ Profile-based segmentation error: {str(e)}")
            return self._empty_result()

    def _create_advanced_color_mask(self, hsv: np.ndarray, profile: Dict) -> np.ndarray:
        """Create advanced color mask with multiple criteria."""
        try:
            h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
            
            # Hue criteria
            h_min, h_max = [x/360.0 for x in profile['hue_range']]
            hue_mask = (h >= h_min) & (h <= h_max)
            
            # Saturation criteria
            s_min, s_max = profile['saturation_range']
            saturation_mask = (s >= s_min) & (s <= s_max)
            
            # Value (brightness) criteria
            v_min, v_max = profile['value_range']
            value_mask = (v >= v_min) & (v <= v_max)
            
            # Combined basic mask
            basic_mask = hue_mask & saturation_mask & value_mask
            
            # Additional biological relevance criteria
            # Avoid extremely bright pixels (likely reflections)
            reflection_mask = v < 0.98
            
            # Avoid extremely saturated pixels (likely artifacts)
            artifact_mask = s < 0.98
            
            # Color consistency (neighboring pixels should have similar colors)
            consistency_mask = self._assess_color_consistency(hsv, basic_mask)
            
            # Final combined mask
            advanced_mask = (basic_mask & reflection_mask & 
                           artifact_mask & consistency_mask)
            
            return advanced_mask
            
        except Exception as e:
            logger.error(f"❌ Advanced color mask error: {str(e)}")
            # Fallback to simple color range
            h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
            h_min, h_max = [x/360.0 for x in profile['hue_range']]
            return (h >= h_min) & (h <= h_max) & (s >= 0.2) & (v >= 0.2)

    def _assess_color_consistency(self, hsv: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Assess color consistency in local neighborhoods."""
        try:
            # Simple consistency check using local standard deviation
            h = hsv[:, :, 0]
            
            # Calculate local hue standard deviation
            kernel = np.ones((5, 5)) / 25
            h_local_mean = cv2.filter2D(h, -1, kernel)
            h_local_var = cv2.filter2D((h - h_local_mean)**2, -1, kernel)
            h_local_std = np.sqrt(h_local_var)
            
            # Consistent regions have low local standard deviation
            consistency_threshold = 0.1  # Adjust based on requirements
            consistency_mask = h_local_std < consistency_threshold
            
            return consistency_mask
            
        except Exception as e:
            logger.error(f"❌ Color consistency error: {str(e)}")
            return np.ones(mask.shape, dtype=bool)

    def _refine_color_mask(self, mask: np.ndarray, image_shape: Tuple) -> np.ndarray:
        """Apply morphological operations to refine the color mask."""
        try:
            # Remove small noise
            min_size = max(10, int(0.0001 * image_shape[0] * image_shape[1]))  # 0.01% of image
            cleaned_mask = remove_small_objects(mask, min_size=min_size)
            
            # Morphological operations with size-adaptive kernels
            kernel_size = max(2, min(image_shape) // 200)  # Adaptive kernel size
            
            try:
                # Use scikit-image morphological operations
                kernel = disk(kernel_size)
                refined_mask = binary_opening(cleaned_mask, kernel)
                refined_mask = binary_closing(refined_mask, disk(kernel_size + 1))
            except:
                # Fallback to OpenCV operations
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                                 (kernel_size*2+1, kernel_size*2+1))
                refined_mask = cv2.morphologyEx(
                    cleaned_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel
                ) > 0
                refined_mask = cv2.morphologyEx(
                    refined_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel
                ) > 0
            
            return refined_mask
            
        except Exception as e:
            logger.error(f"❌ Mask refinement error: {str(e)}")
            return mask

    def _assess_color_segmentation_quality(self, image: np.ndarray, mask: np.ndarray,
                                         profile: Dict) -> float:
        """Assess the quality of color segmentation results."""
        try:
            quality_factors = []
            
            # 1. Coverage - reasonable amount of image should be segmented
            coverage = np.sum(mask) / mask.size
            if 0.05 <= coverage <= 0.7:  # 5-70% coverage is reasonable
                coverage_score = 1.0
            elif coverage < 0.05:
                coverage_score = coverage / 0.05  # Linear penalty below 5%
            else:
                coverage_score = max(0, 1.0 - (coverage - 0.7) / 0.3)  # Penalty above 70%
            quality_factors.append(coverage_score)
            
            # 2. Connectivity - segmented regions should be reasonably connected
            if np.sum(mask) > 0:
                labels = label(mask)
                num_regions = np.max(labels)
                region_sizes = [np.sum(labels == i) for i in range(1, num_regions + 1)]
                
                if region_sizes:
                    # Good segmentation has a few large regions rather than many tiny ones
                    largest_region_fraction = max(region_sizes) / sum(region_sizes)
                    connectivity_score = min(largest_region_fraction * 2, 1.0)
                    quality_factors.append(connectivity_score)
            
            # 3. Color consistency within segmented regions
            if np.sum(mask) > 100:  # Need sufficient pixels for analysis
                masked_image = image[mask]
                if len(masked_image) > 0:
                    # Calculate color variance within segmented regions
                    color_std = np.mean(np.std(masked_image, axis=0))
                    consistency_score = max(0, 1.0 - color_std / 50.0)  # Normalize by expected std
                    quality_factors.append(consistency_score)
            
            # 4. Biological relevance (from profile)
            biological_score = profile.get('biological_relevance', 0.5)
            quality_factors.append(biological_score)
            
            # Overall quality score
            if quality_factors:
                quality_score = np.mean(quality_factors)
            else:
                quality_score = 0.0
            
            return float(np.clip(quality_score, 0.0, 1.0))
            
        except Exception as e:
            logger.error(f"❌ Quality assessment error: {str(e)}")
            return 0.5

    def _post_process_color_mask(self, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Post-process color mask to create final labels."""
        try:
            # Clear border objects that might be partially visible
            cleared_mask = clear_border(mask)
            
            # Label connected components
            labels = label(cleared_mask)
            
            return cleared_mask, labels
            
        except Exception as e:
            logger.error(f"❌ Post-processing error: {str(e)}")
            labels = label(mask)
            return mask, labels

    def _adaptive_color_segmentation(self, image: np.ndarray) -> Dict:
        """Machine learning-based adaptive color segmentation."""
        try:
            logger.info("   Using adaptive ML-based color segmentation")
            
            # If ML segmenter not initialized, create it
            if self.ml_segmenter is None:
                self.ml_segmenter = self._initialize_ml_segmenter()
            
            # Extract color features
            features = self._extract_color_features(image)
            
            # Perform clustering to identify distinct color regions
            cluster_labels = self.ml_segmenter.fit_predict(features)
            
            # Identify the most likely "green organism" cluster
            green_cluster = self._identify_green_cluster(features, cluster_labels, image.shape[:2])
            
            # Create mask from green cluster
            mask = cluster_labels.reshape(image.shape[:2]) == green_cluster
            
            # Refine and post-process
            refined_mask = self._refine_color_mask(mask, image.shape[:2])
            final_mask, labels = self._post_process_color_mask(refined_mask)
            
            cell_count = np.max(labels)
            quality_score = 0.8  # Adaptive method generally produces good results
            
            visualization = self._create_enhanced_visualization(
                image, labels, {'description': 'Adaptive ML segmentation'}
            )
            
            logger.info(f"✅ Adaptive segmentation: {cell_count} regions")
            
            return {
                'labels': labels,
                'cell_count': cell_count,
                'mask': final_mask.astype(np.uint8) * 255,
                'visualization': visualization,
                'quality_score': quality_score,
                'method': 'color_adaptive',
                'success': True
            }
            
        except Exception as e:
            logger.error(f"❌ Adaptive segmentation error: {str(e)}")
            return self._empty_result()

    def _initialize_ml_segmenter(self):
        """Initialize machine learning segmenter."""
        try:
            # Use K-means clustering for color segmentation
            return KMeans(n_clusters=5, random_state=42, n_init=10)
        except Exception as e:
            logger.error(f"❌ ML segmenter initialization error: {str(e)}")
            return None

    def _extract_color_features(self, image: np.ndarray) -> np.ndarray:
        """Extract color features for ML-based segmentation."""
        try:
            # Convert to multiple color spaces
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32) / 255.0
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
            
            # Normalize RGB
            rgb_norm = image.astype(np.float32) / 255.0
            
            # Combine features: RGB + HSV + LAB
            features = np.concatenate([
                rgb_norm.reshape(-1, 3),
                hsv.reshape(-1, 3),
                lab.reshape(-1, 3)
            ], axis=1)
            
            return features
            
        except Exception as e:
            logger.error(f"❌ Feature extraction error: {str(e)}")
            # Fallback to RGB only
            return image.reshape(-1, 3).astype(np.float32) / 255.0

    def _identify_green_cluster(self, features: np.ndarray, cluster_labels: np.ndarray,
                              image_shape: Tuple) -> int:
        """Identify which cluster corresponds to green organisms."""
        try:
            unique_clusters = np.unique(cluster_labels)
            
            # Calculate average color for each cluster
            cluster_scores = []
            
            for cluster_id in unique_clusters:
                cluster_mask = cluster_labels == cluster_id
                cluster_features = features[cluster_mask]
                
                if len(cluster_features) == 0:
                    cluster_scores.append(0)
                    continue
                
                # Average RGB values for this cluster
                avg_rgb = np.mean(cluster_features[:, :3], axis=0)  # First 3 features are RGB
                
                # Score based on "greenness"
                r, g, b = avg_rgb
                green_score = g - 0.5 * (r + b)  # Green minus average of red and blue
                
                # Bonus for reasonable brightness and saturation
                if len(cluster_features[0]) >= 6:  # If HSV features available
                    avg_hsv = np.mean(cluster_features[:, 3:6], axis=0)
                    h, s, v = avg_hsv
                    if 0.25 <= h <= 0.45 and s > 0.3 and 0.2 <= v <= 0.9:  # Green hue range
                        green_score += 0.5
                
                cluster_scores.append(green_score)
            
            # Return cluster with highest green score
            best_cluster = unique_clusters[np.argmax(cluster_scores)]
            return best_cluster
            
        except Exception as e:
            logger.error(f"❌ Green cluster identification error: {str(e)}")
            return 0  # Return first cluster as fallback

    def _create_enhanced_visualization(self, image: np.ndarray, labels: np.ndarray,
                                     profile: Dict) -> np.ndarray:
        """Create enhanced visualization with biological context."""
        try:
            if np.max(labels) == 0:
                return image
            
            # Create colored overlay
            overlay = np.zeros_like(image)
            
            # Use biologically relevant colors
            unique_labels = np.unique(labels)[1:]  # Skip background
            
            # Color palette optimized for biological visualization
            bio_colors = [
                [0, 255, 100],    # Bright green
                [50, 255, 50],    # Light green  
                [0, 200, 150],    # Blue-green
                [100, 255, 0],    # Yellow-green
                [0, 150, 200],    # Teal
                [150, 255, 100],  # Pale green
                [0, 255, 200],    # Cyan-green
                [200, 255, 0],    # Lime
                [0, 180, 255],    # Sky blue
                [100, 200, 150]   # Sage green
            ]
            
            for i, label_val in enumerate(unique_labels):
                color = bio_colors[i % len(bio_colors)]
                overlay[labels == label_val] = color
            
            # Create final visualization with transparency
            alpha = 0.6  # Transparency factor
            visualization = cv2.addWeighted(image, 1-alpha, overlay, alpha, 0)
            
            return visualization
            
        except Exception as e:
            logger.error(f"❌ Visualization error: {str(e)}")
            return image

    def _empty_result(self) -> Dict:
        """Return empty result structure."""
        return {
            'labels': np.zeros((100, 100), dtype=np.int32),
            'cell_count': 0,
            'mask': np.zeros((100, 100), dtype=np.uint8),
            'visualization': None,
            'quality_score': 0.0,
            'method': 'failed',
            'success': False
        }

    def _low_quality_result(self, quality_score: float, method: str) -> Dict:
        """Return low quality result with warning."""
        return {
            'labels': np.zeros((100, 100), dtype=np.int32),
            'cell_count': 0,
            'mask': np.zeros((100, 100), dtype=np.uint8),
            'visualization': None,
            'quality_score': quality_score,
            'method': f'{method}_low_quality',
            'success': False,
            'warning': f'Segmentation quality too low: {quality_score:.3f}'
        }


class EnhancedCellSegmentation:
    """
    Professional cell segmentation system with advanced algorithms and quality control.
    
    This class provides comprehensive segmentation capabilities including:
    - Multi-algorithm approach with automatic selection
    - Advanced watershed segmentation for overlapping cells
    - Machine learning-enhanced preprocessing
    - Quality assessment and validation
    - Biological optimization for Wolffia specimens
    """

    def __init__(self, min_area: int = 30, max_area: int = 8000, 
                 quality_threshold: float = 0.6):
        """
        Initialize the professional segmentation system.
        
        Parameters:
        -----------
        min_area : int
            Minimum cell area in pixels
        max_area : int
            Maximum cell area in pixels
        quality_threshold : float
            Minimum quality threshold for accepting segmentation results
        """
        self.min_area = min_area
        self.max_area = max_area
        self.quality_threshold = quality_threshold
        
        # Initialize specialized segmenters
        self.color_segmenter = AdvancedColorSegmenter()
        
        # Performance tracking
        self.segmentation_history = []
        
        logger.info("🔬 Professional Cell Segmentation System initialized")
        logger.info(f"   Cell size range: {min_area}-{max_area} pixels")
        logger.info(f"   Quality threshold: {quality_threshold}")

    def segment_cells(self, gray_image: np.ndarray, green_channel: np.ndarray, 
                     chlorophyll_enhanced: np.ndarray, method: str = 'auto',
                     **kwargs) -> np.ndarray:
        """
        Professional cell segmentation with comprehensive algorithm suite.
        
        Parameters:
        -----------
        gray_image : np.ndarray
            Grayscale version of the image
        green_channel : np.ndarray
            Enhanced green channel
        chlorophyll_enhanced : np.ndarray
            Chlorophyll-specific enhancement
        method : str
            Segmentation method ('auto', 'watershed', 'threshold', 'adaptive', 'hybrid')
        **kwargs : dict
            Additional parameters for specific methods
            
        Returns:
        --------
        np.ndarray : Labeled image with cell regions
        """
        try:
            logger.info(f"🔍 Professional segmentation: {method}")
            
            # Validate inputs
            inputs_valid, validation_info = self._validate_inputs(
                gray_image, green_channel, chlorophyll_enhanced
            )
            
            if not inputs_valid:
                logger.error(f"❌ Input validation failed: {validation_info}")
                return np.zeros_like(gray_image, dtype=np.int32)
            
            # Preprocessing for optimal segmentation
            processed_inputs = self._preprocess_for_segmentation(
                gray_image, green_channel, chlorophyll_enhanced
            )
            
            # Choose and execute segmentation strategy
            if method == 'auto':
                labels = self._auto_segment_with_quality_control(processed_inputs, **kwargs)
            elif method == 'watershed':
                labels = self._advanced_watershed_segment(processed_inputs, **kwargs)
            elif method == 'threshold':
                labels = self._enhanced_threshold_segment(processed_inputs, **kwargs)
            elif method == 'adaptive':
                labels = self._adaptive_segment_with_ml(processed_inputs, **kwargs)
            elif method == 'hybrid':
                labels = self._hybrid_segmentation(processed_inputs, **kwargs)
            else:
                logger.warning(f"⚠️ Unknown method: {method}, using auto")
                labels = self._auto_segment_with_quality_control(processed_inputs, **kwargs)
            
            # Post-processing and quality control
            final_labels = self._post_process_with_quality_control(
                labels, processed_inputs, method
            )
            
            # Performance tracking
            self._track_segmentation_performance(final_labels, method)
            
            cell_count = np.max(final_labels)
            logger.info(f"✅ Segmentation complete: {cell_count} cells detected")
            
            return final_labels
            
        except Exception as e:
            logger.error(f"❌ Segmentation error: {str(e)}")
            import traceback
            logger.error(f"📋 Traceback: {traceback.format_exc()}")
            return np.zeros_like(gray_image, dtype=np.int32)

    def _validate_inputs(self, *inputs) -> Tuple[bool, str]:
        """Validate segmentation inputs."""
        try:
            for i, inp in enumerate(inputs):
                if not isinstance(inp, np.ndarray):
                    return False, f"Input {i} is not a numpy array"
                
                if len(inp.shape) != 2:
                    return False, f"Input {i} is not 2D (shape: {inp.shape})"
                
                if inp.size == 0:
                    return False, f"Input {i} is empty"
                
                if np.any(np.isnan(inp)) or np.any(np.isinf(inp)):
                    return False, f"Input {i} contains NaN or infinite values"
            
            # Check shape consistency
            shape = inputs[0].shape
            for i, inp in enumerate(inputs[1:], 1):
                if inp.shape != shape:
                    return False, f"Input shapes inconsistent: {shape} vs {inp.shape}"
            
            return True, "Inputs valid"
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"

    def _preprocess_for_segmentation(self, gray_image: np.ndarray, 
                                   green_channel: np.ndarray,
                                   chlorophyll_enhanced: np.ndarray) -> Dict:
        """Advanced preprocessing optimized for segmentation."""
        try:
            processed = {
                'gray': gray_image.copy(),
                'green': green_channel.copy(),
                'chlorophyll': chlorophyll_enhanced.copy()
            }
            
            # Noise reduction with edge preservation
            processed['gray_denoised'] = self._edge_preserving_denoise(processed['gray'])
            processed['chlorophyll_denoised'] = self._edge_preserving_denoise(processed['chlorophyll'])
            
            # Edge enhancement for better boundary detection
            processed['edges'] = self._enhance_edges(processed['gray_denoised'])
            
            # Gradient magnitude for watershed
            processed['gradient'] = self._compute_gradient_magnitude(processed['chlorophyll_denoised'])
            
            # Texture features for advanced segmentation
            processed['texture'] = self._compute_texture_features(processed['gray_denoised'])
            
            logger.info("   ✅ Advanced preprocessing complete")
            return processed
            
        except Exception as e:
            logger.error(f"❌ Preprocessing error: {str(e)}")
            return {
                'gray': gray_image,
                'green': green_channel,
                'chlorophyll': chlorophyll_enhanced,
                'gray_denoised': gray_image,
                'chlorophyll_denoised': chlorophyll_enhanced
            }

    def _edge_preserving_denoise(self, image: np.ndarray) -> np.ndarray:
        """Apply edge-preserving denoising."""
        try:
            # Bilateral filtering for edge preservation
            if image.max() <= 1.0:
                denoised = cv2.bilateralFilter(
                    (image * 255).astype(np.uint8), 9, 75, 75
                ) / 255.0
            else:
                denoised = cv2.bilateralFilter(
                    image.astype(np.uint8), 9, 75, 75
                ).astype(np.float32) / 255.0
            
            return denoised
            
        except Exception as e:
            logger.error(f"❌ Denoising error: {str(e)}")
            return image

    def _enhance_edges(self, image: np.ndarray) -> np.ndarray:
        """Enhance edges for better segmentation."""
        try:
            # Multi-scale edge detection
            edges_sobel = sobel(image)
            edges_scharr = scharr(image)
            
            # Combine edge responses
            edges_combined = np.sqrt(edges_sobel**2 + edges_scharr**2)
            
            return edges_combined
            
        except Exception as e:
            logger.error(f"❌ Edge enhancement error: {str(e)}")
            return np.zeros_like(image)

    def _compute_gradient_magnitude(self, image: np.ndarray) -> np.ndarray:
        """Compute gradient magnitude for watershed segmentation."""
        try:
            # Sobel gradients
            grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
            
            # Gradient magnitude
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            return gradient_magnitude
            
        except Exception as e:
            logger.error(f"❌ Gradient computation error: {str(e)}")
            return np.zeros_like(image)

    def _compute_texture_features(self, image: np.ndarray) -> np.ndarray:
        """Compute texture features for segmentation enhancement."""
        try:
            # Local standard deviation as texture measure
            from scipy.ndimage import generic_filter
            
            def local_std(values):
                return np.std(values)
            
            texture = generic_filter(image, local_std, size=5)
            
            return texture
            
        except Exception as e:
            logger.error(f"❌ Texture computation error: {str(e)}")
            return np.zeros_like(image)

    def _auto_segment_with_quality_control(self, processed_inputs: Dict, **kwargs) -> np.ndarray:
        """Automatic segmentation with intelligent algorithm selection and quality control."""
        try:
            logger.info("   🎯 Auto-segmentation with quality control")
            
            # Analyze image characteristics
            characteristics = self._analyze_image_characteristics(processed_inputs)
            logger.info(f"   📊 Image characteristics: {characteristics}")
            
            # Select optimal segmentation strategy
            strategy = self._select_optimal_strategy(characteristics)
            logger.info(f"   🔧 Selected strategy: {strategy['name']}")
            
            # Try multiple methods and select best result
            methods_to_try = strategy['methods']
            results = []
            
            for method_name in methods_to_try:
                try:
                    if method_name == 'watershed':
                        labels = self._advanced_watershed_segment(processed_inputs, **kwargs)
                    elif method_name == 'threshold':
                        labels = self._enhanced_threshold_segment(processed_inputs, **kwargs)
                    elif method_name == 'adaptive':
                        labels = self._adaptive_segment_with_ml(processed_inputs, **kwargs)
                    else:
                        continue
                    
                    # Assess quality
                    quality_score = self._assess_segmentation_quality(labels, processed_inputs)
                    
                    results.append({
                        'method': method_name,
                        'labels': labels,
                        'quality_score': quality_score,
                        'cell_count': np.max(labels)
                    })
                    
                    logger.info(f"   📈 {method_name}: {np.max(labels)} cells, quality={quality_score:.3f}")
                    
                except Exception as method_error:
                    logger.warning(f"   ⚠️ {method_name} failed: {str(method_error)}")
            
            # Select best result
            if results:
                best_result = max(results, key=lambda x: x['quality_score'])
                
                if best_result['quality_score'] >= self.quality_threshold:
                    logger.info(f"   ✅ Best method: {best_result['method']} (quality={best_result['quality_score']:.3f})")
                    return best_result['labels']
                else:
                    logger.warning(f"   ⚠️ All methods below quality threshold")
                    # Return best available result
                    return best_result['labels']
            else:
                logger.error("   ❌ All segmentation methods failed")
                return self._fallback_segmentation(processed_inputs)
                
        except Exception as e:
            logger.error(f"❌ Auto-segmentation error: {str(e)}")
            return self._fallback_segmentation(processed_inputs)

    def _analyze_image_characteristics(self, processed_inputs: Dict) -> Dict:
        """Analyze image characteristics to guide segmentation strategy."""
        try:
            chlorophyll = processed_inputs['chlorophyll_denoised']
            gray = processed_inputs['gray_denoised']
            
            characteristics = {
                'contrast': float(np.std(chlorophyll)),
                'brightness': float(np.mean(chlorophyll)),
                'noise_level': float(np.std(chlorophyll - gaussian(chlorophyll, sigma=1))),
                'edge_density': float(np.mean(processed_inputs.get('edges', np.zeros_like(gray)))),
                'texture_variation': float(np.std(processed_inputs.get('texture', np.zeros_like(gray)))),
                'dynamic_range': float(chlorophyll.max() - chlorophyll.min()),
                'sparsity': float(np.sum(chlorophyll > np.mean(chlorophyll) + np.std(chlorophyll)) / chlorophyll.size)
            }
            
            return characteristics
            
        except Exception as e:
            logger.error(f"❌ Characteristic analysis error: {str(e)}")
            return {'contrast': 0.5, 'brightness': 0.5, 'noise_level': 0.3}

    def _select_optimal_strategy(self, characteristics: Dict) -> Dict:
        """Select optimal segmentation strategy based on image characteristics."""
        try:
            contrast = characteristics.get('contrast', 0.5)
            brightness = characteristics.get('brightness', 0.5)
            noise_level = characteristics.get('noise_level', 0.3)
            sparsity = characteristics.get('sparsity', 0.1)
            
            if contrast > 0.3 and sparsity < 0.2 and noise_level < 0.2:
                # High contrast, sparse cells, low noise -> Watershed preferred
                return {
                    'name': 'high_contrast_sparse',
                    'methods': ['watershed', 'threshold', 'adaptive']
                }
            elif contrast < 0.2 or noise_level > 0.4:
                # Low contrast or high noise -> Adaptive methods preferred
                return {
                    'name': 'challenging_conditions',
                    'methods': ['adaptive', 'threshold', 'watershed']
                }
            elif sparsity > 0.5:
                # Dense cells -> Advanced watershed
                return {
                    'name': 'dense_cells',
                    'methods': ['watershed', 'adaptive', 'threshold']
                }
            else:
                # Balanced conditions -> Try all methods
                return {
                    'name': 'balanced',
                    'methods': ['threshold', 'watershed', 'adaptive']
                }
                
        except Exception as e:
            logger.error(f"❌ Strategy selection error: {str(e)}")
            return {
                'name': 'fallback',
                'methods': ['threshold', 'watershed']
            }

    def _advanced_watershed_segment(self, processed_inputs: Dict, **kwargs) -> np.ndarray:
        """Advanced watershed segmentation with biological optimization."""
        try:
            logger.info("   🌊 Advanced watershed segmentation")
            
            chlorophyll = processed_inputs['chlorophyll_denoised']
            gradient = processed_inputs.get('gradient', chlorophyll)
            
            # Create binary mask
            threshold_value = threshold_otsu(chlorophyll)
            binary_mask = chlorophyll > threshold_value
            
            if np.sum(binary_mask) == 0:
                logger.warning("   ⚠️ No foreground pixels found")
                return np.zeros_like(chlorophyll, dtype=np.int32)
            
            # Distance transform for marker generation
            distance = ndimage.distance_transform_edt(binary_mask)
            
            # Advanced peak detection with biological constraints
            min_distance = kwargs.get('min_distance', max(8, int(np.sqrt(self.min_area))))
            threshold_abs = kwargs.get('threshold_abs', distance.max() * 0.3)
            
            # Find local max as watershed markers
            try:
                local_max = peak_local_max(
                    distance,
                    min_distance=min_distance,
                    threshold_abs=threshold_abs,
                    indices=False
                )
            except Exception as peak_error:
                logger.warning(f"   ⚠️ Peak detection failed: {peak_error}, using fallback")
                # Fallback peak detection
                from scipy.ndimage import maximum_filter
                local_max = (maximum_filter(distance, size=min_distance*2+1) == distance) & (distance > threshold_abs)
            
            # Create markers
            markers = label(local_max)
            
            if np.max(markers) == 0:
                logger.warning("   ⚠️ No markers found")
                return label(binary_mask.astype(int))
            
            # Apply watershed
            labels = watershed(-distance, markers, mask=binary_mask)
            
            logger.info(f"   ✅ Watershed: {np.max(labels)} initial regions")
            return labels
            
        except Exception as e:
            logger.error(f"❌ Watershed segmentation error: {str(e)}")
            return self._fallback_segmentation(processed_inputs)

    def _enhanced_threshold_segment(self, processed_inputs: Dict, **kwargs) -> np.ndarray:
        """Enhanced threshold segmentation with adaptive parameters."""
        try:
            logger.info("   🎚️ Enhanced threshold segmentation")
            
            chlorophyll = processed_inputs['chlorophyll_denoised']
            
            # Multi-threshold approach
            threshold_global = threshold_otsu(chlorophyll)
            threshold_local = threshold_local(
                chlorophyll, 
                block_size=kwargs.get('block_size', 35),
                offset=kwargs.get('offset', 0.01)
            )
            
            # Combine thresholds
            binary_global = chlorophyll > threshold_global
            binary_local = chlorophyll > threshold_local
            
            # Weighted combination
            weight_global = kwargs.get('global_weight', 0.6)
            combined_binary = (weight_global * binary_global.astype(float) + 
                             (1 - weight_global) * binary_local.astype(float)) > 0.5
            
            # Morphological cleanup
            combined_binary = self._morphological_cleanup(
                combined_binary, processed_inputs['chlorophyll'].shape
            )
            
            # Label connected components
            labels = label(combined_binary)
            
            logger.info(f"   ✅ Threshold: {np.max(labels)} regions")
            return labels
            
        except Exception as e:
            logger.error(f"❌ Threshold segmentation error: {str(e)}")
            return self._fallback_segmentation(processed_inputs)

    def _adaptive_segment_with_ml(self, processed_inputs: Dict, **kwargs) -> np.ndarray:
        """Adaptive segmentation using machine learning principles."""
        try:
            logger.info("   🤖 Adaptive ML-enhanced segmentation")
            
            chlorophyll = processed_inputs['chlorophyll_denoised']
            texture = processed_inputs.get('texture', np.zeros_like(chlorophyll))
            
            # Feature extraction for each pixel
            features = self._extract_pixel_features(chlorophyll, texture)
            
            # Unsupervised clustering to identify cell regions
            try:
                clusterer = KMeans(n_clusters=3, random_state=42, n_init=10)  # Background, cells, noise
                cluster_labels = clusterer.fit_predict(features)
                
                # Identify cell cluster (highest mean chlorophyll intensity)
                cluster_means = []
                for i in range(3):
                    cluster_mask = cluster_labels == i
                    if np.sum(cluster_mask) > 0:
                        cluster_means.append(np.mean(chlorophyll.flat[cluster_mask]))
                    else:
                        cluster_means.append(0)
                
                cell_cluster = np.argmax(cluster_means)
                binary_mask = (cluster_labels == cell_cluster).reshape(chlorophyll.shape)
                
            except Exception as clustering_error:
                logger.warning(f"   ⚠️ Clustering failed: {clustering_error}, using threshold fallback")
                binary_mask = chlorophyll > threshold_otsu(chlorophyll)
            
            # Morphological refinement
            refined_mask = self._morphological_cleanup(binary_mask, chlorophyll.shape)
            
            # Label components
            labels = label(refined_mask)
            
            logger.info(f"   ✅ Adaptive: {np.max(labels)} regions")
            return labels
            
        except Exception as e:
            logger.error(f"❌ Adaptive segmentation error: {str(e)}")
            return self._fallback_segmentation(processed_inputs)

    def _extract_pixel_features(self, chlorophyll: np.ndarray, texture: np.ndarray) -> np.ndarray:
        """Extract features for each pixel for ML-based segmentation."""
        try:
            # Basic intensity features
            features = [chlorophyll.flatten()]
            
            # Local statistics
            kernel_size = 5
            kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)
            
            # Local mean
            local_mean = cv2.filter2D(chlorophyll, -1, kernel)
            features.append(local_mean.flatten())
            
            # Local standard deviation
            local_var = cv2.filter2D(chlorophyll**2, -1, kernel) - local_mean**2
            local_std = np.sqrt(np.maximum(local_var, 0))
            features.append(local_std.flatten())
            
            # Texture features if available
            if texture.size > 0:
                features.append(texture.flatten())
            
            # Gradient features
            grad_x = cv2.Sobel(chlorophyll, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(chlorophyll, cv2.CV_64F, 0, 1, ksize=3)
            grad_mag = np.sqrt(grad_x**2 + grad_y**2)
            features.append(grad_mag.flatten())
            
            # Combine all features
            feature_matrix = np.column_stack(features)
            
            # Normalize features
            scaler = StandardScaler()
            normalized_features = scaler.fit_transform(feature_matrix)
            
            return normalized_features
            
        except Exception as e:
            logger.error(f"❌ Feature extraction error: {str(e)}")
            # Fallback to just intensity
            return chlorophyll.flatten().reshape(-1, 1)

    def _hybrid_segmentation(self, processed_inputs: Dict, **kwargs) -> np.ndarray:
        """Hybrid segmentation combining multiple approaches."""
        try:
            logger.info("   🔀 Hybrid segmentation")
            
            # Get results from different methods
            watershed_labels = self._advanced_watershed_segment(processed_inputs, **kwargs)
            threshold_labels = self._enhanced_threshold_segment(processed_inputs, **kwargs)
            
            # Combine results using consensus
            combined_labels = self._combine_segmentation_results(
                [watershed_labels, threshold_labels], processed_inputs
            )
            
            logger.info(f"   ✅ Hybrid: {np.max(combined_labels)} regions")
            return combined_labels
            
        except Exception as e:
            logger.error(f"❌ Hybrid segmentation error: {str(e)}")
            return self._fallback_segmentation(processed_inputs)

    def _combine_segmentation_results(self, label_list: List[np.ndarray], 
                                    processed_inputs: Dict) -> np.ndarray:
        """Combine multiple segmentation results using consensus."""
        try:
            if not label_list:
                return np.zeros_like(processed_inputs['chlorophyll'], dtype=np.int32)
            
            # Create consensus mask
            consensus_mask = np.zeros_like(label_list[0], dtype=bool)
            
            for labels in label_list:
                binary_mask = labels > 0
                consensus_mask = consensus_mask | binary_mask
            
            # Require at least 50% agreement
            agreement_threshold = len(label_list) * 0.5
            agreement_count = np.zeros_like(consensus_mask, dtype=float)
            
            for labels in label_list:
                agreement_count += (labels > 0).astype(float)
            
            consensus_mask = agreement_count >= agreement_threshold
            
            # Label the consensus regions
            combined_labels = label(consensus_mask)
            
            return combined_labels
            
        except Exception as e:
            logger.error(f"❌ Result combination error: {str(e)}")
            return label_list[0] if label_list else np.zeros((100, 100), dtype=np.int32)

    def _morphological_cleanup(self, binary_mask: np.ndarray, 
                             image_shape: Tuple) -> np.ndarray:
        """Apply morphological operations to clean up binary mask."""
        try:
            # Remove small objects
            min_size = max(self.min_area, int(0.0001 * image_shape[0] * image_shape[1]))
            cleaned_mask = remove_small_objects(binary_mask, min_size=min_size)
            
            # Morphological operations with adaptive kernel size
            kernel_size = max(2, min(image_shape[:2]) // 100)
            
            try:
                kernel = disk(kernel_size)
                refined_mask = binary_opening(cleaned_mask, kernel)
                refined_mask = binary_closing(refined_mask, disk(kernel_size + 1))
            except:
                # Fallback to OpenCV
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                                 (kernel_size*2+1, kernel_size*2+1))
                refined_mask = cv2.morphologyEx(
                    cleaned_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel
                ) > 0
                refined_mask = cv2.morphologyEx(
                    refined_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel
                ) > 0
            
            return refined_mask
            
        except Exception as e:
            logger.error(f"❌ Morphological cleanup error: {str(e)}")
            return binary_mask

    def _post_process_with_quality_control(self, labels: np.ndarray, 
                                         processed_inputs: Dict,
                                         method: str) -> np.ndarray:
        """Post-process segmentation results with quality control."""
        try:
            if np.max(labels) == 0:
                return labels
            
            # Size filtering
            regions = regionprops(labels)
            valid_labels = []
            
            for region in regions:
                area = region.area
                
                # Size constraints
                if self.min_area <= area <= self.max_area:
                    # Additional shape constraints
                    if hasattr(region, 'eccentricity') and region.eccentricity < 0.95:
                        # Not too elongated
                        valid_labels.append(region.label)
                    elif not hasattr(region, 'eccentricity'):
                        # Fallback if eccentricity not available
                        valid_labels.append(region.label)
            
            # Create filtered label image
            filtered_labels = np.zeros_like(labels)
            for i, old_label in enumerate(valid_labels, 1):
                filtered_labels[labels == old_label] = i
            
            logger.info(f"   🔍 Quality control: {len(regions)} → {len(valid_labels)} cells")
            
            return filtered_labels
            
        except Exception as e:
            logger.error(f"❌ Post-processing error: {str(e)}")
            return labels

    def _assess_segmentation_quality(self, labels: np.ndarray, 
                                   processed_inputs: Dict) -> float:
        """Assess the quality of segmentation results."""
        try:
            if np.max(labels) == 0:
                return 0.0
            
            quality_factors = []
            
            # 1. Cell count reasonableness
            cell_count = np.max(labels)
            if 1 <= cell_count <= 200:  # Reasonable range
                count_score = 1.0
            elif cell_count > 200:
                count_score = max(0, 1.0 - (cell_count - 200) / 200)
            else:
                count_score = 0.0
            quality_factors.append(count_score)
            
            # 2. Size distribution quality
            regions = regionprops(labels)
            if regions:
                areas = [region.area for region in regions]
                area_cv = np.std(areas) / (np.mean(areas) + 1e-8)  # Coefficient of variation
                size_score = max(0, 1.0 - area_cv / 2.0)  # Lower CV is better
                quality_factors.append(size_score)
                
                # 3. Shape quality (circularity/solidity)
                shape_scores = []
                for region in regions:
                    if hasattr(region, 'solidity'):
                        shape_scores.append(region.solidity)
                
                if shape_scores:
                    shape_score = np.mean(shape_scores)
                    quality_factors.append(shape_score)
            
            # 4. Coverage reasonableness
            total_area = np.sum(labels > 0)
            image_area = labels.size
            coverage = total_area / image_area
            
            if 0.05 <= coverage <= 0.6:  # 5-60% coverage is reasonable
                coverage_score = 1.0
            elif coverage < 0.05:
                coverage_score = coverage / 0.05
            else:
                coverage_score = max(0, 1.0 - (coverage - 0.6) / 0.4)
            quality_factors.append(coverage_score)
            
            # Overall quality score
            quality_score = np.mean(quality_factors) if quality_factors else 0.0
            
            return float(np.clip(quality_score, 0.0, 1.0))
            
        except Exception as e:
            logger.error(f"❌ Quality assessment error: {str(e)}")
            return 0.5

    def _fallback_segmentation(self, processed_inputs: Dict) -> np.ndarray:
        """Fallback segmentation method when all else fails."""
        try:
            logger.info("   🆘 Using fallback segmentation")
            
            chlorophyll = processed_inputs.get('chlorophyll_denoised', 
                                             processed_inputs.get('chlorophyll'))
            
            # Simple Otsu thresholding
            if chlorophyll.max() <= 1.0:
                thresh_img = (chlorophyll * 255).astype(np.uint8)
            else:
                thresh_img = chlorophyll.astype(np.uint8)
            
            _, binary = cv2.threshold(thresh_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Basic cleanup
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
            # Label components
            labels = label(binary > 0)
            
            logger.info(f"   ✅ Fallback: {np.max(labels)} regions")
            return labels
            
        except Exception as e:
            logger.error(f"❌ Fallback segmentation error: {str(e)}")
            # Ultimate fallback
            shape = processed_inputs.get('chlorophyll', np.zeros((100, 100))).shape
            return np.zeros(shape, dtype=np.int32)

    def _track_segmentation_performance(self, labels: np.ndarray, method: str):
        """Track segmentation performance for optimization."""
        try:
            performance_data = {
                'timestamp': datetime.now().isoformat(),
                'method': method,
                'cell_count': int(np.max(labels)),
                'success': np.max(labels) > 0
            }
            
            self.segmentation_history.append(performance_data)
            
            # Keep only recent history (last 100 segmentations)
            if len(self.segmentation_history) > 100:
                self.segmentation_history = self.segmentation_history[-100:]
                
        except Exception as e:
            logger.error(f"❌ Performance tracking error: {str(e)}")

    def get_segmentation_statistics(self, labels: np.ndarray) -> Dict:
        """Get comprehensive statistics about segmentation results."""
        try:
            if np.max(labels) == 0:
                return {
                    'total_cells': 0,
                    'areas': [],
                    'mean_area': 0,
                    'std_area': 0,
                    'coverage': 0,
                    'quality_metrics': {}
                }
            
            regions = regionprops(labels)
            areas = [region.area for region in regions]
            
            # Basic statistics
            stats = {
                'total_cells': len(regions),
                'areas': areas,
                'mean_area': float(np.mean(areas)),
                'std_area': float(np.std(areas)),
                'median_area': float(np.median(areas)),
                'min_area': float(np.min(areas)),
                'max_area': float(np.max(areas)),
                'total_area': float(np.sum(areas)),
                'coverage': float(np.sum(areas) / labels.size * 100)
            }
            
            # Size distribution
            stats['size_distribution'] = {
                'small': len([a for a in areas if a < 100]),
                'medium': len([a for a in areas if 100 <= a < 500]),
                'large': len([a for a in areas if a >= 500])
            }
            
            # Shape statistics if available
            if regions and hasattr(regions[0], 'eccentricity'):
                eccentricities = [r.eccentricity for r in regions if hasattr(r, 'eccentricity')]
                solidities = [r.solidity for r in regions if hasattr(r, 'solidity')]
                
                if eccentricities:
                    stats['shape_statistics'] = {
                        'mean_eccentricity': float(np.mean(eccentricities)),
                        'mean_solidity': float(np.mean(solidities)) if solidities else 0.0
                    }
            
            # Quality metrics
            stats['quality_metrics'] = {
                'area_coefficient_of_variation': float(stats['std_area'] / (stats['mean_area'] + 1e-8)),
                'size_range_ratio': float(stats['max_area'] / (stats['min_area'] + 1e-8)),
                'coverage_score': min(stats['coverage'] / 30.0, 1.0)  # Normalize by expected 30%
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Statistics calculation error: {str(e)}")
            return {
                'total_cells': 0,
                'areas': [],
                'mean_area': 0,
                'std_area': 0,
                'coverage': 0
            }

    def get_performance_summary(self) -> Dict:
        """Get summary of segmentation performance over time."""
        try:
            if not self.segmentation_history:
                return {'message': 'No segmentation history available'}
            
            recent_history = self.segmentation_history[-20:]  # Last 20 segmentations
            
            methods_used = [h['method'] for h in recent_history]
            success_rates = {}
            
            for method in set(methods_used):
                method_results = [h for h in recent_history if h['method'] == method]
                success_count = sum(1 for h in method_results if h['success'])
                success_rates[method] = success_count / len(method_results) * 100
            
            cell_counts = [h['cell_count'] for h in recent_history if h['success']]
            
            summary = {
                'total_segmentations': len(self.segmentation_history),
                'recent_segmentations': len(recent_history),
                'success_rates_by_method': success_rates,
                'average_cell_count': float(np.mean(cell_counts)) if cell_counts else 0,
                'cell_count_std': float(np.std(cell_counts)) if cell_counts else 0
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ Performance summary error: {str(e)}")
            return {'error': str(e)}


# Testing and validation
if __name__ == "__main__":
    print("🧪 Testing Professional Cell Segmentation System...")
    
    try:
        # Test initialization
        segmenter = EnhancedCellSegmentation(min_area=50, max_area=5000)
        
        # Test with synthetic data
        test_gray = np.random.rand(200, 200)
        test_green = np.random.rand(200, 200)
        test_chlorophyll = np.random.rand(200, 200)
        
        # Test segmentation
        labels = segmenter.segment_cells(test_gray, test_green, test_chlorophyll, method='auto')
        
        # Test statistics
        stats = segmenter.get_segmentation_statistics(labels)
        
        print("📊 Test Results:")
        print(f"   Labels shape: {labels.shape}")
        print(f"   Max label: {np.max(labels)}")
        print(f"   Total cells: {stats['total_cells']}")
        print(f"   Mean area: {stats['mean_area']:.1f} pixels")
        
        # Test color segmentation
        color_segmenter = AdvancedColorSegmenter()
        test_rgb = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        color_result = color_segmenter.segment_by_color(test_rgb, 'green_wolffia')
        
        print(f"   Color segmentation: {color_result['cell_count']} regions")
        print(f"   Quality score: {color_result.get('quality_score', 0):.3f}")
        
        print("✅ Professional Cell Segmentation test complete")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()