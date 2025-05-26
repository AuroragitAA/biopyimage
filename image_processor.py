"""
image_processor.py

Professional Image Processing Module for Wolffia Bioimage Analysis
Enhanced with advanced preprocessing, quality assessment, and biological optimization.

This module handles all image preprocessing operations to optimize images for
accurate cell detection and morphological analysis of Wolffia specimens.
"""

import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

# Scientific computing imports
from skimage.color import rgb2gray, rgb2hsv, rgb2lab
from skimage.exposure import equalize_adapthist, rescale_intensity
from skimage.filters import gaussian
from skimage.morphology import closing, disk, opening
from skimage.restoration import denoise_bilateral, denoise_wavelet

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageProcessor:
    """
    Professional image processor optimized for Wolffia bioimage analysis.
    
    This class provides comprehensive image preprocessing capabilities including:
    - Multi-format image loading with validation
    - Advanced noise reduction and enhancement 
    - Biological feature optimization
    - Quality assessment and metadata extraction
    - Adaptive preprocessing based on image characteristics
    """

    def __init__(self):
        """Initialize the professional image processor."""
        self.supported_formats = [
            '.jpg', '.jpeg', '.png', '.tif', '.tiff', 
            '.bmp', '.jfif', '.webp', '.jp2'
        ]
        
        # Processing parameters
        self.default_params = {
            'enhance_contrast': True,
            'denoise': True,
            'preserve_quantitative': True,
            'biological_optimization': True,
            'quality_assessment': True
        }
        
        logger.info("📷 Professional Image Processor initialized")
        logger.info(f"   Supported formats: {', '.join(self.supported_formats)}")

    def preprocess_image(self, image_input: Union[str, np.ndarray], 
                        enhance_contrast: bool = True, 
                        denoise: bool = True,
                        biological_optimization: bool = True,
                        **kwargs) -> Optional[Tuple]:
        """
        Professional image preprocessing pipeline optimized for Wolffia analysis.
        
        This method performs comprehensive preprocessing to enhance image quality
        and optimize for accurate cell detection and measurement.

        Parameters:
        -----------
        image_input : str or np.ndarray
            Path to image file or image array
        enhance_contrast : bool
            Whether to apply adaptive contrast enhancement
        denoise : bool
            Whether to apply noise reduction
        biological_optimization : bool
            Whether to apply biological feature optimization
        **kwargs : dict
            Additional processing parameters

        Returns:
        --------
        tuple : (original, gray, green_channel, chlorophyll_enhanced, hsv) or None
            Processed image components optimized for analysis
        """
        try:
            logger.info("🔍 Starting professional image preprocessing")
            
            # Step 1: Load and validate image
            load_result = self._load_and_validate_image(image_input)
            if not load_result['success']:
                logger.error(f"❌ Image loading failed: {load_result['error']}")
                return None

            original = load_result['image']
            metadata = load_result.get('metadata', {})
            
            logger.info(f"   📐 Image loaded: {original.shape[1]}x{original.shape[0]} pixels")
            logger.info(f"   📊 Bit depth: {original.dtype}, Range: {original.min()}-{original.max()}")

            # Step 2: Quality assessment
            quality_metrics = self._assess_image_quality(original)
            logger.info(f"   📈 Image quality score: {quality_metrics['overall_quality']:.3f}")

            # Step 3: Adaptive preprocessing strategy
            processing_strategy = self._determine_processing_strategy(
                original, quality_metrics, enhance_contrast, denoise, biological_optimization
            )
            logger.info(f"   🎯 Processing strategy: {processing_strategy['name']}")

            # Step 4: Color space conversions
            color_spaces = self._convert_color_spaces(original)
            gray_base = color_spaces['gray']
            hsv = color_spaces['hsv']
            lab = color_spaces['lab']

            # Step 5: Advanced denoising
            if processing_strategy['apply_denoising']:
                denoised = self._advanced_denoising(original, processing_strategy)
                logger.info("   🧹 Advanced denoising applied")
            else:
                denoised = original.copy()

            # Step 6: Illumination correction
            if processing_strategy['illumination_correction']:
                corrected = self._correct_illumination(denoised, processing_strategy)
                logger.info("   🔆 Illumination correction applied")
            else:
                corrected = denoised

            # Step 7: Biological feature enhancement
            if biological_optimization:
                bio_enhanced = self._enhance_biological_features(corrected, hsv)
                green_channel = bio_enhanced['green_enhanced']
                chlorophyll_enhanced = bio_enhanced['chlorophyll_enhanced']
                logger.info("   🧬 Biological feature enhancement applied")
            else:
                green_channel = corrected[:, :, 1] / 255.0
                chlorophyll_enhanced = green_channel - 0.5 * (
                    corrected[:, :, 0]/255.0 + corrected[:, :, 2]/255.0
                )
                chlorophyll_enhanced = np.clip(chlorophyll_enhanced, 0, 1)

            # Step 8: Contrast enhancement
            if processing_strategy['contrast_enhancement']:
                enhanced_results = self._enhance_contrast_adaptive(
                    gray_base, green_channel, chlorophyll_enhanced, processing_strategy
                )
                gray = enhanced_results['gray']
                green_channel = enhanced_results['green']
                chlorophyll_enhanced = enhanced_results['chlorophyll']
                logger.info("   ✨ Adaptive contrast enhancement applied")
            else:
                gray = gray_base

            # Step 9: Final quality check and optimization
            final_quality = self._assess_processing_quality(
                original, gray, green_channel, chlorophyll_enhanced
            )
            
            logger.info(f"   ✅ Preprocessing complete - Final quality: {final_quality:.3f}")

            return original, gray, green_channel, chlorophyll_enhanced, hsv

        except Exception as e:
            logger.error(f"   ❌ Preprocessing error: {str(e)}")
            import traceback
            logger.error(f"   📋 Traceback: {traceback.format_exc()}")
            return None

    def _load_and_validate_image(self, image_input: Union[str, np.ndarray]) -> Dict:
        """Enhanced image loading with comprehensive validation."""
        try:
            result = {'success': False, 'image': None, 'metadata': {}}
            
            if isinstance(image_input, str):
                # Validate file path and format
                if not self._validate_image_path(image_input):
                    result['error'] = f"Invalid image path or unsupported format: {image_input}"
                    return result
                
                # Load image with metadata preservation
                image = cv2.imread(image_input, cv2.IMREAD_UNCHANGED)
                if image is None:
                    result['error'] = f"Could not load image from {image_input}"
                    return result
                
                # Handle different color formats
                if len(image.shape) == 3:
                    if image.shape[2] == 4:  # RGBA
                        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
                    elif image.shape[2] == 3:  # BGR
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                elif len(image.shape) == 2:  # Grayscale
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                
                # Extract metadata
                result['metadata'] = self._extract_image_metadata(image_input)
                
            elif isinstance(image_input, np.ndarray):
                # Validate and normalize array
                validation_result = self._validate_image_array(image_input)
                if not validation_result['valid']:
                    result['error'] = validation_result['error']
                    return result
                
                image = validation_result['image']
                result['metadata'] = {'source': 'array', 'validated': True}
                
            else:
                result['error'] = "Image input must be file path (str) or numpy array"
                return result
            
            # Final validation and normalization
            image = self._normalize_image(image)
            
            result.update({
                'success': True,
                'image': image,
                'shape': image.shape,
                'dtype': str(image.dtype),
                'size_mb': image.nbytes / (1024 * 1024)
            })
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Image loading error: {str(e)}",
                'image': None
            }

    def _validate_image_path(self, image_path: str) -> bool:
        """Validate image file path and format."""
        try:
            path = Path(image_path)
            
            # Check if file exists
            if not path.exists():
                logger.error(f"   ❌ Image file not found: {image_path}")
                return False
            
            # Check file size
            file_size = path.stat().st_size
            if file_size == 0:
                logger.error(f"   ❌ Empty file: {image_path}")
                return False
            
            if file_size > 100 * 1024 * 1024:  # 100MB limit
                logger.error(f"   ❌ File too large: {file_size / (1024*1024):.1f}MB")
                return False
            
            # Check file extension
            file_ext = path.suffix.lower()
            if file_ext not in self.supported_formats:
                logger.error(f"   ❌ Unsupported format: {file_ext}")
                logger.error(f"   ℹ️  Supported formats: {', '.join(self.supported_formats)}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"   ❌ Path validation error: {str(e)}")
            return False

    def _validate_image_array(self, image_array: np.ndarray) -> Dict:
        """Enhanced validation and normalization of input image array."""
        try:
            result = {'valid': False, 'image': None, 'error': None}
            
            # Check array dimensions
            if len(image_array.shape) not in [2, 3]:
                result['error'] = f"Image must be 2D or 3D array, got shape: {image_array.shape}"
                return result
            
            # Handle different array formats
            if len(image_array.shape) == 2:
                # Convert grayscale to RGB
                image = np.stack([image_array] * 3, axis=-1)
            elif image_array.shape[2] == 1:
                # Convert single channel to RGB
                image = np.repeat(image_array, 3, axis=-1)
            elif image_array.shape[2] == 3:
                # Already RGB
                image = image_array.copy()
            elif image_array.shape[2] == 4:
                # Convert RGBA to RGB
                image = image_array[:, :, :3]
            else:
                result['error'] = f"Unsupported number of channels: {image_array.shape[2]}"
                return result
            
            # Check for valid data
            if np.any(np.isnan(image)) or np.any(np.isinf(image)):
                result['error'] = "Image contains NaN or infinite values"
                return result
            
            # Check dimensions
            if image.shape[0] < 50 or image.shape[1] < 50:
                result['error'] = f"Image too small: {image.shape[1]}x{image.shape[0]} (minimum 50x50)"
                return result
            
            if image.shape[0] > 10000 or image.shape[1] > 10000:
                result['error'] = f"Image too large: {image.shape[1]}x{image.shape[0]} (maximum 10000x10000)"
                return result
            
            result.update({
                'valid': True,
                'image': image,
                'original_shape': image_array.shape,
                'converted': len(image_array.shape) != 3 or image_array.shape[2] != 3
            })
            
            return result
            
        except Exception as e:
            return {
                'valid': False,
                'error': f"Array validation error: {str(e)}",
                'image': None
            }

    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize image to standard format."""
        try:
            # Ensure correct data type and range
            if image.dtype == np.float32 or image.dtype == np.float64:
                if image.max() <= 1.0:
                    # Float image in [0, 1] range
                    image = (image * 255).astype(np.uint8)
                else:
                    # Float image in [0, 255] range
                    image = np.clip(image, 0, 255).astype(np.uint8)
            elif image.dtype == np.uint16:
                # 16-bit image, scale to 8-bit
                image = (image / 256).astype(np.uint8)
            elif image.dtype == np.uint8:
                # Already correct format
                pass
            else:
                # Convert other types
                image = np.clip(image, 0, 255).astype(np.uint8)
            
            return image
            
        except Exception as e:
            logger.error(f"   ❌ Image normalization error: {str(e)}")
            # Fallback normalization
            return np.clip(image, 0, 255).astype(np.uint8)

    def _assess_image_quality(self, image: np.ndarray) -> Dict:
        """Comprehensive image quality assessment."""
        try:
            gray = rgb2gray(image.astype(np.float32) / 255.0)
            
            # 1. Sharpness (Laplacian variance)
            laplacian = cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F)
            sharpness = np.var(laplacian)
            sharpness_score = min(sharpness * 10, 1.0)
            
            # 2. Contrast assessment
            contrast = np.std(gray.astype(np.float32))
            contrast_score = min(contrast * 3, 1.0)
            
            # 3. Brightness assessment
            brightness = np.mean(gray)
            brightness_score = 1.0 - abs(brightness - 0.5) * 2
            
            # 4. Noise estimation
            noise_estimate = self._estimate_noise_level(gray)
            noise_score = max(0, 1.0 - noise_estimate)
            
            # 5. Dynamic range
            hist, _ = np.histogram(gray, bins=256, range=(0, 1))
            non_zero_bins = np.sum(hist > 0)
            dynamic_range = non_zero_bins / 256.0
            
            # 6. Green content (biological relevance)
            hsv = rgb2hsv(image.astype(np.float32) / 255.0)
            green_mask = self._create_green_mask(hsv)
            green_content = np.sum(green_mask) / green_mask.size
            green_score = min(green_content * 5, 1.0)
            
            # Composite quality score
            quality_weights = {
                'sharpness': 0.25,
                'contrast': 0.20,
                'brightness': 0.15,
                'noise': 0.15,
                'dynamic_range': 0.15,
                'green_content': 0.10
            }
            
            scores = {
                'sharpness': sharpness_score,
                'contrast': contrast_score,
                'brightness': brightness_score,
                'noise': noise_score,
                'dynamic_range': dynamic_range,
                'green_content': green_score
            }
            
            overall_quality = sum(scores[k] * quality_weights[k] for k in quality_weights.keys())
            
            return {
                'overall_quality': float(np.clip(overall_quality, 0, 1)),
                'scores': scores,
                'metrics': {
                    'sharpness_raw': float(sharpness),
                    'contrast_raw': float(contrast),
                    'brightness_raw': float(brightness),
                    'noise_level': float(noise_estimate),
                    'green_content_percent': float(green_content * 100)
                },
                'quality_grade': self._grade_quality(overall_quality)
            }
            
        except Exception as e:
            logger.error(f"❌ Quality assessment error: {e}")
            return {
                'overall_quality': 0.5,
                'quality_grade': 'unknown',
                'scores': {},
                'metrics': {}
            }

    def _estimate_noise_level(self, gray_image: np.ndarray) -> float:
        """Estimate noise level in the image."""
        try:
            # Use wavelet-based noise estimation
            from scipy import ndimage
            
            # Apply Laplacian filter to detect high-frequency content
            laplacian = ndimage.laplace(gray_image)
            
            # Estimate noise as standard deviation of high-frequency content
            noise_level = np.std(laplacian)
            
            # Normalize to 0-1 range
            return min(noise_level * 5, 1.0)
            
        except Exception as e:
            logger.error(f"❌ Noise estimation error: {e}")
            return 0.5

    def _grade_quality(self, quality_score: float) -> str:
        """Assign quality grade based on score."""
        if quality_score >= 0.8:
            return 'excellent'
        elif quality_score >= 0.6:
            return 'good'
        elif quality_score >= 0.4:
            return 'fair'
        else:
            return 'poor'

    def _determine_processing_strategy(self, image: np.ndarray, quality_metrics: Dict,
                                     enhance_contrast: bool, denoise: bool, 
                                     biological_optimization: bool) -> Dict:
        """Determine optimal processing strategy based on image characteristics."""
        try:
            quality_score = quality_metrics['overall_quality']
            
            # Base strategy
            strategy = {
                'name': 'adaptive',
                'apply_denoising': denoise,
                'contrast_enhancement': enhance_contrast,
                'illumination_correction': False,
                'biological_optimization': biological_optimization
            }
            
            # Adjust based on quality metrics
            if quality_score < 0.5:
                strategy.update({
                    'name': 'aggressive_enhancement',
                    'apply_denoising': True,
                    'contrast_enhancement': True,
                    'illumination_correction': True,
                    'denoising_strength': 'strong',
                    'contrast_method': 'clahe'
                })
            elif quality_score < 0.7:
                strategy.update({
                    'name': 'moderate_enhancement',
                    'illumination_correction': True,
                    'denoising_strength': 'moderate',
                    'contrast_method': 'adaptive'
                })
            else:
                strategy.update({
                    'name': 'gentle_enhancement',
                    'denoising_strength': 'light',
                    'contrast_method': 'histogram'
                })
            
            # Specific adjustments based on metrics
            scores = quality_metrics.get('scores', {})
            
            if scores.get('noise', 1.0) < 0.5:
                strategy['apply_denoising'] = True
                strategy['denoising_strength'] = 'strong'
            
            if scores.get('contrast', 1.0) < 0.4:
                strategy['contrast_enhancement'] = True
                strategy['contrast_method'] = 'clahe'
            
            if scores.get('brightness', 1.0) < 0.4:
                strategy['illumination_correction'] = True
            
            return strategy
            
        except Exception as e:
            logger.error(f"❌ Strategy determination error: {e}")
            return {
                'name': 'fallback',
                'apply_denoising': denoise,
                'contrast_enhancement': enhance_contrast,
                'illumination_correction': False
            }

    def _convert_color_spaces(self, image: np.ndarray) -> Dict:
        """Convert image to multiple color spaces."""
        try:
            # Normalize for conversion
            float_image = image.astype(np.float32) / 255.0
            
            color_spaces = {
                'gray': rgb2gray(float_image),
                'hsv': rgb2hsv(float_image),
                'lab': rgb2lab(float_image),
                'rgb_normalized': float_image
            }
            
            return color_spaces
            
        except Exception as e:
            logger.error(f"❌ Color space conversion error: {e}")
            return {
                'gray': cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) / 255.0,
                'hsv': cv2.cvtColor(image, cv2.COLOR_RGB2HSV) / 255.0,
                'lab': image.astype(np.float32) / 255.0,  # Fallback
                'rgb_normalized': image.astype(np.float32) / 255.0
            }

    def _advanced_denoising(self, image: np.ndarray, strategy: Dict) -> np.ndarray:
        """Apply advanced denoising based on strategy."""
        try:
            strength = strategy.get('denoising_strength', 'moderate')

            if strength == 'light':
                # Light Gaussian denoising
                denoised = gaussian(image, sigma=0.5, channel_axis=-1)
            elif strength == 'moderate':
                # Bilateral filtering to preserve edges
                denoised = np.zeros_like(image, dtype=np.float32)
                for i in range(3):
                    denoised[:, :, i] = denoise_bilateral(
                        image[:, :, i], sigma_color=0.1, sigma_spatial=1.0
                    )
            else:  # strong
                # Advanced wavelet denoising
                try:
                    denoised = denoise_wavelet(
                        image, sigma=None, wavelet='db4', channel_axis=-1,
                        method='BayesShrink', mode='soft'
                    )
                except ImportError:
                    # Fallback to bilateral if wavelet not available
                    denoised = np.zeros_like(image, dtype=np.float32)
                    for i in range(3):
                        denoised[:, :, i] = denoise_bilateral(
                            image[:, :, i], sigma_color=0.15, sigma_spatial=2.0
                        )

            # Ensure proper format
            if denoised.max() <= 1.0:
                denoised = (denoised * 255).astype(np.uint8)
            else:
                denoised = np.clip(denoised, 0, 255).astype(np.uint8)

            return denoised

        except Exception as e:
            logger.error(f"❌ Denoising error: {e}")
            return image

    def _correct_illumination(self, image: np.ndarray, strategy: Dict) -> np.ndarray:
        """Correct illumination variations."""
        try:
            # Simple background estimation and correction
            corrected = np.zeros_like(image, dtype=np.float32)
            
            for i in range(3):
                channel = image[:, :, i].astype(np.float32)
                
                # Estimate background using morphological opening
                kernel_size = min(image.shape[:2]) // 20
                kernel = disk(kernel_size)
                background = opening(channel, kernel)
                background = gaussian(background, sigma=kernel_size//2)
                
                # Correct illumination
                corrected_channel = channel - background + np.mean(background)
                corrected[:, :, i] = np.clip(corrected_channel, 0, 255)
            
            return corrected.astype(np.uint8)
            
        except Exception as e:
            logger.error(f"❌ Illumination correction error: {e}")
            return image

    def _enhance_biological_features(self, image: np.ndarray, hsv: np.ndarray) -> Dict:
        """Enhance biological features specific to Wolffia."""
        try:
            # Normalize channels
            r, g, b = image[:, :, 0]/255.0, image[:, :, 1]/255.0, image[:, :, 2]/255.0
            
            # Create advanced green mask
            green_mask = self._create_green_mask(hsv)
            
            # Enhanced green channel with biological relevance
            green_enhanced = g * (1 + 0.3 * green_mask)
            green_enhanced = np.clip(green_enhanced, 0, 1)
            
            # Advanced chlorophyll estimation
            # Multiple chlorophyll indices combined
            
            # 1. Green Leaf Index (GLI)
            gli = (2*g - r - b) / (2*g + r + b + 1e-8)
            
            # 2. Visible Atmospherically Resistant Index (VARI)
            vari = (g - r) / (g + r - b + 1e-8)
            
            # 3. Excess Green Index
            egi = 2*g - r - b
            
            # Combined chlorophyll index
            chlorophyll_enhanced = 0.4 * gli + 0.3 * vari + 0.3 * egi
            chlorophyll_enhanced = np.clip(chlorophyll_enhanced, -1, 1)
            
            # Normalize to 0-1 range
            chlorophyll_enhanced = (chlorophyll_enhanced + 1) / 2
            
            # Apply green mask to focus on relevant regions
            chlorophyll_enhanced = chlorophyll_enhanced * green_mask
            green_enhanced = green_enhanced * green_mask
            
            return {
                'green_enhanced': green_enhanced,
                'chlorophyll_enhanced': chlorophyll_enhanced,
                'green_mask': green_mask,
                'indices': {
                    'gli': gli,
                    'vari': vari,
                    'egi': egi
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Biological enhancement error: {e}")
            # Fallback enhancement
            g = image[:, :, 1] / 255.0
            r, b = image[:, :, 0]/255.0, image[:, :, 2]/255.0
            chlorophyll_simple = g - 0.5 * (r + b)
            return {
                'green_enhanced': g,
                'chlorophyll_enhanced': np.clip(chlorophyll_simple, 0, 1),
                'green_mask': np.ones_like(g, dtype=bool)
            }

    def _create_green_mask(self, hsv_image: np.ndarray) -> np.ndarray:
        """Create sophisticated green mask for biological relevance."""
        try:
            hsv_image = hsv_image.astype(np.float32)
            h, s, v = hsv_image[:, :, 0], hsv_image[:, :, 1], hsv_image[:, :, 2]

            green_hue = (h >= 0.25) & (h <= 0.45)
            sufficient_saturation = s >= 0.2
            sufficient_brightness = v >= 0.15
            not_too_bright = v <= 0.95
            balanced_saturation = (s >= 0.1) & (s <= 0.9)

            mask = (green_hue & sufficient_saturation & sufficient_brightness &
                    not_too_bright & balanced_saturation)

            try:
                kernel = disk(2)
                mask = opening(mask, kernel)
                mask = closing(mask, disk(3))
            except:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                mask = mask.astype(np.uint8) * 255
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                mask = mask > 0

            return mask

        except Exception as e:
            logger.error(f"❌ Green mask error: {e}")
            try:
                hsv_uint8 = (hsv_image * 255).astype(np.uint8)
                lower_green = np.array([35, 50, 40])
                upper_green = np.array([85, 255, 255])
                mask = cv2.inRange(hsv_uint8, lower_green, upper_green) > 0
                return mask
            except:
                return np.ones(hsv_image.shape[:2], dtype=bool)


    def _enhance_contrast_adaptive(self, gray: np.ndarray, green: np.ndarray, 
                                 chlorophyll: np.ndarray, strategy: Dict) -> Dict:
        """Apply adaptive contrast enhancement."""
        try:
            method = strategy.get('contrast_method', 'adaptive')
            
            if method == 'clahe':
                # Contrast Limited Adaptive Histogram Equalization
                gray_enhanced = equalize_adapthist(gray, clip_limit=0.03)
                green_enhanced = equalize_adapthist(green, clip_limit=0.03)
                chlorophyll_enhanced = equalize_adapthist(chlorophyll, clip_limit=0.03)
                
            elif method == 'histogram':
                # Standard histogram equalization
                gray_enhanced = equalize_adapthist(gray, clip_limit=0.01)
                green_enhanced = equalize_adapthist(green, clip_limit=0.01)
                chlorophyll_enhanced = equalize_adapthist(chlorophyll, clip_limit=0.01)
                
            else:  # adaptive
                # Gentle adaptive enhancement
                gray_enhanced = rescale_intensity(gray, out_range=(0, 1))
                green_enhanced = rescale_intensity(green, out_range=(0, 1))
                chlorophyll_enhanced = rescale_intensity(chlorophyll, out_range=(0, 1))
            
            return {
                'gray': gray_enhanced,
                'green': green_enhanced,
                'chlorophyll': chlorophyll_enhanced
            }
            
        except Exception as e:
            logger.error(f"❌ Contrast enhancement error: {e}")
            return {
                'gray': gray,
                'green': green,
                'chlorophyll': chlorophyll
            }

    def _assess_processing_quality(self, original: np.ndarray, gray: np.ndarray, 
                                 green: np.ndarray, chlorophyll: np.ndarray) -> float:
        """Assess the quality of processed images."""
        try:
            quality_factors = []
            
            # Check dynamic range improvement
            original_range = np.std(rgb2gray(original.astype(np.float32) / 255.0))
            processed_range = np.std(gray)
            range_improvement = min(processed_range / (original_range + 1e-8), 2.0) / 2.0
            quality_factors.append(range_improvement)
            
            # Check feature enhancement
            chlorophyll_contrast = np.std(chlorophyll)
            feature_quality = min(chlorophyll_contrast * 2, 1.0)
            quality_factors.append(feature_quality)
            
            # Check preservation of details
            detail_preservation = min(np.mean(green), 1.0)
            quality_factors.append(detail_preservation)
            
            # Overall processing quality
            return float(np.mean(quality_factors))
            
        except Exception as e:
            logger.error(f"❌ Processing quality assessment error: {e}")
            return 0.5

    def _extract_image_metadata(self, image_path: str) -> Dict:
        """Extract comprehensive image metadata."""
        try:
            path = Path(image_path)
            stat = path.stat()
            
            metadata = {
                'filename': path.name,
                'file_size_bytes': stat.st_size,
                'file_size_mb': round(stat.st_size / (1024 * 1024), 2),
                'creation_time': stat.st_ctime,
                'modification_time': stat.st_mtime,
                'format': path.suffix.lower(),
                'processing_timestamp': datetime.now().isoformat()
            }
            
            # Try to get additional EXIF data if available
            try:
                import PIL.ExifTags
                import PIL.Image
                
                with PIL.Image.open(image_path) as img:
                    metadata.update({
                        'pil_format': img.format,
                        'pil_mode': img.mode,
                        'dimensions': img.size
                    })
                    
                    # Extract EXIF if available
                    exifdata = img.getexif()
                    if exifdata:
                        for tag_id, value in exifdata.items():
                            tag = PIL.ExifTags.TAGS.get(tag_id, tag_id)
                            if isinstance(value, (str, int, float)):
                                metadata[f'exif_{tag}'] = value
                                
            except ImportError:
                logger.info("PIL not available for EXIF extraction")
            except Exception as exif_error:
                logger.warning(f"EXIF extraction failed: {exif_error}")
            
            return metadata
            
        except Exception as e:
            logger.error(f"❌ Metadata extraction error: {e}")
            return {'error': str(e)}

    # Enhanced utility methods
    def get_image_info(self, image_input: Union[str, np.ndarray]) -> Optional[Dict]:
        """Get comprehensive information about an image."""
        try:
            load_result = self._load_and_validate_image(image_input)
            if not load_result['success']:
                return None

            image = load_result['image']
            metadata = load_result['metadata']

            # Calculate comprehensive statistics
            color_stats = {}
            for i, channel in enumerate(['red', 'green', 'blue']):
                channel_data = image[:, :, i]
                color_stats[channel] = {
                    'mean': float(np.mean(channel_data)),
                    'std': float(np.std(channel_data)),
                    'min': int(np.min(channel_data)),
                    'max': int(np.max(channel_data)),
                    'median': float(np.median(channel_data))
                }

            # Ensure compatible type for OpenCV color conversion
            image_safe = image.astype(np.uint8) if image.dtype != np.uint8 else image
            hsv = cv2.cvtColor(image_safe, cv2.COLOR_RGB2HSV)
            green_mask = self._create_green_mask(hsv.astype(np.float32) / 255.0)

            # Quality assessment
            quality_metrics = self._assess_image_quality(image)

            info = {
                'dimensions': {
                    'width': image.shape[1],
                    'height': image.shape[0],
                    'channels': image.shape[2],
                    'total_pixels': image.shape[0] * image.shape[1]
                },
                'color_statistics': color_stats,
                'quality_metrics': quality_metrics,
                'green_content': {
                    'green_pixels': int(np.sum(green_mask)),
                    'green_percentage': float(np.sum(green_mask) / green_mask.size * 100)
                },
                'metadata': metadata,
                'processing_recommendations': self._generate_processing_recommendations(quality_metrics)
            }

            return info

        except Exception as e:
            logger.error(f"❌ Error getting image info: {str(e)}")
            return None

    def _generate_processing_recommendations(self, quality_metrics: Dict) -> List[str]:
        """Generate processing recommendations based on quality assessment."""
        recommendations = []
        
        try:
            scores = quality_metrics.get('scores', {})
            
            if scores.get('sharpness', 1.0) < 0.5:
                recommendations.append("Apply sharpening filter to improve edge definition")
            
            if scores.get('contrast', 1.0) < 0.5:
                recommendations.append("Use CLAHE (Contrast Limited Adaptive Histogram Equalization)")
            
            if scores.get('noise', 1.0) < 0.5:
                recommendations.append("Apply bilateral or wavelet denoising")
            
            if scores.get('brightness', 1.0) < 0.5:
                recommendations.append("Correct illumination variations")
            
            if scores.get('green_content', 1.0) < 0.3:
                recommendations.append("Enhance biological features and green channel")
            
            if not recommendations:
                recommendations.append("Image quality is good - standard processing recommended")
                
        except Exception as e:
            logger.error(f"❌ Recommendation generation error: {e}")
            recommendations = ["Standard processing recommended"]
        
        return recommendations

    def batch_validate_images(self, image_paths: List[str]) -> Dict:
        """Validate multiple image paths for batch processing."""
        valid_paths = []
        invalid_paths = []
        validation_details = []

        logger.info(f"🔍 Validating {len(image_paths)} image paths...")

        for path in image_paths:
            try:
                if self._validate_image_path(path):
                    valid_paths.append(path)
                    validation_details.append({
                        'path': path,
                        'status': 'valid',
                        'size_mb': round(Path(path).stat().st_size / (1024*1024), 2)
                    })
                else:
                    invalid_paths.append(path)
                    validation_details.append({
                        'path': path,
                        'status': 'invalid',
                        'error': 'Validation failed'
                    })
            except Exception as e:
                invalid_paths.append(path)
                validation_details.append({
                    'path': path,
                    'status': 'error',
                    'error': str(e)
                })

        result = {
            'valid_paths': valid_paths,
            'invalid_paths': invalid_paths,
            'validation_details': validation_details,
            'valid_count': len(valid_paths),
            'invalid_count': len(invalid_paths),
            'success_rate': len(valid_paths) / len(image_paths) * 100 if image_paths else 0,
            'total_size_mb': sum(d.get('size_mb', 0) for d in validation_details if d['status'] == 'valid')
        }

        logger.info(f"✅ Validation complete: {len(valid_paths)}/{len(image_paths)} valid images")
        if invalid_paths:
            logger.warning(f"❌ Invalid paths: {len(invalid_paths)}")
            for detail in validation_details:
                if detail['status'] != 'valid':
                    logger.warning(f"   - {detail['path']}: {detail.get('error', 'Unknown error')}")

        return result


# Testing and validation
if __name__ == "__main__":
    print("🧪 Testing Professional Image Processor...")
    
    try:
        # Test initialization
        processor = ImageProcessor()
        
        # Test with a simple synthetic image
        test_image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        
        # Test preprocessing
        result = processor.preprocess_image(test_image)
        
        if result:
            original, gray, green_channel, chlorophyll_enhanced, hsv = result
            print("✅ Preprocessing successful:")
            print(f"   Original shape: {original.shape}")
            print(f"   Gray range: {gray.min():.3f} - {gray.max():.3f}")
            print(f"   Green channel range: {green_channel.min():.3f} - {green_channel.max():.3f}")
            print(f"   Chlorophyll range: {chlorophyll_enhanced.min():.3f} - {chlorophyll_enhanced.max():.3f}")
        else:
            print("❌ Preprocessing failed")
        
        # Test image info
        info = processor.get_image_info(test_image)
        if info:
            print(f"📊 Image info extracted: Quality grade = {info['quality_metrics']['quality_grade']}")
        
        print("✅ Professional Image Processor test complete")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()