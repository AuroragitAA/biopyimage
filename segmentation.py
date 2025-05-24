"""
segmentation.py

Enhanced cell segmentation module for Wolffia bioimage analysis.
Compatible with multiple scikit-image versions.

Key Features:
- Watershed segmentation for overlapping cells
- Color-based segmentation for green organisms
- Adaptive thresholding for varying lighting
- Morphological operations for noise reduction
- Size filtering for relevant cell detection

Dependencies: OpenCV, scikit-image, numpy, scipy
"""

import cv2
import numpy as np
from scipy import ndimage
from skimage.segmentation import watershed
from skimage.measure import label, regionprops
from skimage.morphology import disk, binary_erosion, binary_dilation, remove_small_objects
from skimage.filters import threshold_otsu, gaussian
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Handle different scikit-image versions for peak_local_max
try:
    from skimage.feature import peak_local_max
    print("✅ Using skimage.feature.peak_local_max")
except ImportError:
    try:
        from skimage.segmentation import peak_local_max
        print("✅ Using skimage.segmentation.peak_local_max")
    except ImportError:
        print("⚠️ peak_local_max not found, using fallback implementation")
        
        def peak_local_max(image, min_distance=1, threshold_abs=None, indices=True):
            """
            Fallback implementation of peak_local_max for compatibility.
            
            Parameters:
            -----------
            image : np.ndarray
                Input image
            min_distance : int
                Minimum distance between peaks
            threshold_abs : float, optional
                Minimum threshold for peaks
            indices : bool
                If True, return indices; if False, return boolean mask
                
            Returns:
            --------
            np.ndarray : Peak locations or boolean mask
            """
            from scipy.ndimage import maximum_filter
            
            # Create a structure for the maximum filter
            size = min_distance * 2 + 1
            
            # Find local max using maximum filter
            local_max = maximum_filter(image, size=size) == image
            
            # Remove peaks at the border
            border = min_distance
            local_max[:border] = False
            local_max[-border:] = False
            local_max[:, :border] = False
            local_max[:, -border:] = False
            
            # Apply threshold if provided
            if threshold_abs is not None:
                local_max = local_max & (image >= threshold_abs)
            
            if indices:
                return np.where(local_max)
            else:
                return local_max


class ColorSegmenter:
    """
    Specialized color-based segmentation for green photosynthetic organisms.
    """
    
    def __init__(self):
        """Initialize color segmentation parameters"""
        self.color_profiles = {
            'green_wolffia': {
                'hue_range': (35, 85),      # Green hue range in HSV
                'saturation_min': 0.3,      # Minimum saturation
                'value_min': 0.2,           # Minimum brightness
                'description': 'Standard green Wolffia detection'
            },
            'bright_green': {
                'hue_range': (30, 90),
                'saturation_min': 0.4,
                'value_min': 0.3,
                'description': 'Bright green organisms'
            },
            'dark_green': {
                'hue_range': (40, 80),
                'saturation_min': 0.2,
                'value_min': 0.1,
                'description': 'Dark or shadowed green organisms'
            }
        }
        print("🎨 Color Segmenter initialized")

    def segment_by_color(self, image, color_name='green_wolffia'):
        """
        Segment image based on color profile.
        
        Parameters:
        -----------
        image : np.ndarray
            RGB image array
        color_name : str
            Color profile name to use
            
        Returns:
        --------
        dict : Segmentation results with labels and visualization
        """
        try:
            if color_name not in self.color_profiles:
                print(f"❌ Unknown color profile: {color_name}")
                return self._empty_result()
            
            profile = self.color_profiles[color_name]
            print(f"🔍 Segmenting with profile: {profile['description']}")
            
            # Convert to HSV color space
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            hsv_normalized = hsv.astype(np.float32) / 255.0
            
            # Create color mask
            h_min, h_max = profile['hue_range']
            mask = (
                (hsv_normalized[:, :, 0] * 360 >= h_min) & 
                (hsv_normalized[:, :, 0] * 360 <= h_max) &
                (hsv_normalized[:, :, 1] >= profile['saturation_min']) &
                (hsv_normalized[:, :, 2] >= profile['value_min'])
            )
            
            # Clean up mask with morphological operations
            try:
                kernel = disk(2)
                mask = binary_erosion(mask, kernel)
                mask = binary_dilation(mask, kernel)
                mask = remove_small_objects(mask, min_size=50)
            except Exception as e:
                print(f"⚠️ Morphological operations failed, using basic mask: {e}")
                # Fallback: simple cleanup
                kernel = np.ones((3, 3), np.uint8)
                mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel) > 0
            
            # Label connected components
            labels = label(mask.astype(int))
            cell_count = np.max(labels)
            
            # Create visualization
            visualization = self._create_color_visualization(image, labels)
            
            print(f"✅ Color segmentation complete: {cell_count} regions found")
            
            return {
                'labels': labels,
                'cell_count': cell_count,
                'visualization': visualization,
                'mask': mask.astype(np.uint8) * 255,
                'method': f'color_{color_name}'
            }
            
        except Exception as e:
            print(f"❌ Color segmentation error: {str(e)}")
            return self._empty_result()

    def _create_color_visualization(self, original, labels):
        """Create colored visualization of segmentation results"""
        try:
            # Create colored overlay
            overlay = np.zeros_like(original)
            
            # Assign different colors to different labels
            unique_labels = np.unique(labels)
            if len(unique_labels) <= 1:  # Only background
                return original
            
            # Use matplotlib colormap for colors
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))[:, :3]
            
            for i, label_val in enumerate(unique_labels[1:], 1):  # Skip background
                color = colors[i % len(colors)]
                overlay[labels == label_val] = (color * 255).astype(np.uint8)
            
            # Blend with original
            result = cv2.addWeighted(original, 0.6, overlay, 0.4, 0)
            return result
            
        except Exception as e:
            print(f"❌ Visualization error: {str(e)}")
            return original

    def _empty_result(self):
        """Return empty result structure"""
        return {
            'labels': np.zeros((100, 100), dtype=np.int32),  # Small empty array
            'cell_count': 0,
            'visualization': None,
            'mask': None,
            'method': 'failed'
        }


class EnhancedCellSegmentation:
    """
    Advanced cell segmentation system with multiple strategies.
    
    This class provides various segmentation approaches optimized for different
    image conditions and cell types, with automatic method selection.
    """

    def __init__(self, min_area=30, max_area=8000):
        """
        Initialize segmentation system.
        
        Parameters:
        -----------
        min_area : int
            Minimum cell area in pixels
        max_area : int
            Maximum cell area in pixels
        """
        self.min_area = min_area
        self.max_area = max_area
        self.color_segmenter = ColorSegmenter()
        
        print("🔬 Enhanced Cell Segmentation initialized")
        print(f"   Cell size range: {min_area}-{max_area} pixels")

    def segment_cells(self, gray_image, green_channel, chlorophyll_enhanced, method='auto'):
        """
        Main segmentation method with automatic strategy selection.
        
        Parameters:
        -----------
        gray_image : np.ndarray
            Grayscale version of the image
        green_channel : np.ndarray
            Enhanced green channel
        chlorophyll_enhanced : np.ndarray
            Chlorophyll-specific enhancement
        method : str
            Segmentation method ('auto', 'watershed', 'threshold', 'adaptive')
            
        Returns:
        --------
        np.ndarray : Labeled image with cell regions
        """
        try:
            print(f"🔍 Starting segmentation with method: {method}")
            
            # Ensure all inputs are numpy arrays
            if not isinstance(gray_image, np.ndarray):
                gray_image = np.array(gray_image)
            if not isinstance(green_channel, np.ndarray):
                green_channel = np.array(green_channel)
            if not isinstance(chlorophyll_enhanced, np.ndarray):
                chlorophyll_enhanced = np.array(chlorophyll_enhanced)
            
            # Choose segmentation strategy
            if method == 'auto':
                labels = self._auto_segment(gray_image, green_channel, chlorophyll_enhanced)
            elif method == 'watershed':
                labels = self._watershed_segment(chlorophyll_enhanced)
            elif method == 'threshold':
                labels = self._threshold_segment(chlorophyll_enhanced)
            elif method == 'adaptive':
                labels = self._adaptive_segment(gray_image, chlorophyll_enhanced)
            else:
                print(f"❌ Unknown method: {method}, using threshold")
                labels = self._threshold_segment(chlorophyll_enhanced)
            
            # Post-process results
            labels = self._post_process_labels(labels)
            
            cell_count = np.max(labels)
            print(f"✅ Segmentation complete: {cell_count} cells detected")
            
            return labels
            
        except Exception as e:
            print(f"❌ Segmentation error: {str(e)}")
            import traceback
            traceback.print_exc()
            # Return empty labels with same shape as input
            return np.zeros_like(gray_image, dtype=np.int32)

    def _auto_segment(self, gray_image, green_channel, chlorophyll_enhanced):
        """
        Automatic segmentation method selection based on image characteristics.
        """
        try:
            # Analyze image characteristics
            contrast = np.std(chlorophyll_enhanced)
            density = np.sum(chlorophyll_enhanced > 0.3) / chlorophyll_enhanced.size
            
            print(f"   Image analysis: contrast={contrast:.3f}, density={density:.3f}")
            
            # Choose method based on characteristics
            if contrast > 0.2 and density < 0.3:
                # High contrast, sparse cells -> watershed
                print("   → Using watershed segmentation")
                return self._watershed_segment(chlorophyll_enhanced)
            elif density > 0.5:
                # Dense cells -> adaptive thresholding
                print("   → Using adaptive segmentation")
                return self._adaptive_segment(gray_image, chlorophyll_enhanced)
            else:
                # Default -> threshold segmentation
                print("   → Using threshold segmentation")
                return self._threshold_segment(chlorophyll_enhanced)
                
        except Exception as e:
            print(f"❌ Auto segmentation error: {str(e)}")
            return self._threshold_segment(chlorophyll_enhanced)

    def _watershed_segment(self, image):
        """
        Watershed segmentation for overlapping or touching cells.
        """
        try:
            # Prepare image
            if image.max() <= 1.0:
                image_uint8 = (image * 255).astype(np.uint8)
            else:
                image_uint8 = image.astype(np.uint8)
            
            # Create binary mask using Otsu thresholding
            try:
                threshold = threshold_otsu(image_uint8)
            except:
                threshold = 127  # Fallback threshold
            
            binary = image_uint8 > threshold
            
            if np.sum(binary) == 0:
                print("   ⚠️ No foreground pixels found")
                return np.zeros_like(image, dtype=np.int32)
            
            # Distance transform
            distance = ndimage.distance_transform_edt(binary)
            
            if distance.max() == 0:
                print("   ⚠️ Distance transform failed")
                return label(binary.astype(int))
            
            # Find local max as seeds
            try:
                local_max = peak_local_max(
                    distance, 
                    min_distance=10, 
                    threshold_abs=distance.max() * 0.3,
                    indices=False
                )
            except Exception as e:
                print(f"   ⚠️ peak_local_max failed: {e}, using fallback")
                # Simple fallback: use maximum filter
                from scipy.ndimage import maximum_filter
                local_max = maximum_filter(distance, size=21) == distance
                local_max = local_max & (distance > distance.max() * 0.3)
            
            # Create markers
            markers = label(local_max)
            
            if np.max(markers) == 0:
                print("   ⚠️ No markers found, using simple labeling")
                return label(binary.astype(int))
            
            # Watershed segmentation
            labels = watershed(-distance, markers, mask=binary)
            
            return labels
            
        except Exception as e:
            print(f"❌ Watershed error: {str(e)}")
            return self._threshold_segment(image)

    def _threshold_segment(self, image):
        """
        Simple threshold-based segmentation.
        """
        try:
            # Normalize image to 0-255
            if image.max() <= 1.0:
                image_uint8 = (image * 255).astype(np.uint8)
            else:
                image_uint8 = image.astype(np.uint8)
            
            # Apply Gaussian smoothing
            try:
                smoothed = gaussian(image_uint8, sigma=1.0)
                smoothed_uint8 = (smoothed * 255).astype(np.uint8)
            except:
                # Fallback: use OpenCV Gaussian blur
                smoothed_uint8 = cv2.GaussianBlur(image_uint8, (5, 5), 1.0)
            
            # Otsu thresholding
            try:
                threshold = threshold_otsu(smoothed_uint8)
                binary = smoothed_uint8 > threshold
            except:
                # Fallback: use OpenCV threshold
                _, binary = cv2.threshold(smoothed_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                binary = binary > 0
            
            # Morphological cleaning
            try:
                kernel = disk(2)
                binary = binary_erosion(binary, kernel)
                binary = binary_dilation(binary, kernel)
                binary = remove_small_objects(binary, min_size=self.min_area)
            except:
                # Fallback: use OpenCV morphology
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                binary = cv2.morphologyEx(binary.astype(np.uint8), cv2.MORPH_CLOSE, kernel) > 0
            
            # Label connected components
            labels = label(binary)
            
            return labels
            
        except Exception as e:
            print(f"❌ Threshold segmentation error: {str(e)}")
            # Ultimate fallback: simple thresholding
            try:
                binary = image > (image.mean() + image.std())
                return label(binary.astype(int))
            except:
                return np.zeros_like(image, dtype=np.int32)

    def _adaptive_segment(self, gray_image, chlorophyll_enhanced):
        """
        Adaptive thresholding for varying illumination conditions.
        """
        try:
            # Convert to uint8
            if gray_image.max() <= 1.0:
                gray_uint8 = (gray_image * 255).astype(np.uint8)
            else:
                gray_uint8 = gray_image.astype(np.uint8)
                
            if chlorophyll_enhanced.max() <= 1.0:
                chlor_uint8 = (chlorophyll_enhanced * 255).astype(np.uint8)
            else:
                chlor_uint8 = chlorophyll_enhanced.astype(np.uint8)
            
            # Adaptive thresholding
            adaptive_thresh = cv2.adaptiveThreshold(
                chlor_uint8, 255, 
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 
                blockSize=21, 
                C=10
            )
            
            # Convert to binary
            binary = adaptive_thresh > 0
            
            # Morphological operations
            try:
                kernel = disk(3)
                binary = binary_erosion(binary, kernel)
                binary = binary_dilation(binary, disk(4))
                binary = remove_small_objects(binary, min_size=self.min_area)
            except:
                # Fallback morphology
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
                binary = cv2.morphologyEx(binary.astype(np.uint8), cv2.MORPH_CLOSE, kernel) > 0
            
            # Label components
            labels = label(binary)
            
            return labels
            
        except Exception as e:
            print(f"❌ Adaptive segmentation error: {str(e)}")
            return self._threshold_segment(chlorophyll_enhanced)

    def _post_process_labels(self, labels):
        """
        Post-process segmentation results to filter by size and clean up.
        """
        try:
            if np.max(labels) == 0:
                return labels
            
            # Get region properties
            regions = regionprops(labels)
            
            # Filter by size
            valid_labels = []
            for region in regions:
                if self.min_area <= region.area <= self.max_area:
                    valid_labels.append(region.label)
            
            # Create new label image with only valid regions
            new_labels = np.zeros_like(labels)
            for i, old_label in enumerate(valid_labels, 1):
                new_labels[labels == old_label] = i
            
            print(f"   Size filtering: {len(regions)} → {len(valid_labels)} cells")
            
            return new_labels
            
        except Exception as e:
            print(f"❌ Post-processing error: {str(e)}")
            return labels

    def get_segmentation_stats(self, labels):
        """Get comprehensive statistics about segmentation results."""
        try:
            if np.max(labels) == 0:
                return {
                    'total_cells': 0,
                    'areas': [],
                    'mean_area': 0,
                    'std_area': 0,
                    'coverage': 0
                }
            
            regions = regionprops(labels)
            areas = [region.area for region in regions]
            
            stats = {
                'total_cells': len(regions),
                'areas': areas,
                'mean_area': np.mean(areas) if areas else 0,
                'std_area': np.std(areas) if areas else 0,
                'coverage': np.sum(areas) / labels.size * 100 if labels.size > 0 else 0,
                'size_distribution': {
                    'small': len([a for a in areas if a < 100]),
                    'medium': len([a for a in areas if 100 <= a < 500]),
                    'large': len([a for a in areas if a >= 500])
                }
            }
            
            return stats
            
        except Exception as e:
            print(f"❌ Stats calculation error: {str(e)}")
            return {
                'total_cells': 0,
                'areas': [],
                'mean_area': 0,
                'std_area': 0,
                'coverage': 0
            }


# Standalone segmentation functions for backward compatibility
def segment_cells_simple(image, method='otsu'):
    """
    Simple segmentation function for basic use cases.
    """
    try:
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Normalize to 0-255
        if gray.max() <= 1.0:
            gray = (gray * 255).astype(np.uint8)
        
        if method == 'otsu':
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif method == 'adaptive':
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
        else:
            # Default fallback
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        return binary > 0
        
    except Exception as e:
        print(f"❌ Simple segmentation error: {str(e)}")
        return np.zeros_like(image, dtype=bool)


# Main execution for testing
if __name__ == "__main__":
    print("🔬 Testing Enhanced Cell Segmentation...")
    
    try:
        # Create test instance
        segmenter = EnhancedCellSegmentation(min_area=50, max_area=5000)
        print("✅ Segmentation module loaded successfully")
        
        # Test with simple synthetic data
        test_image = np.random.rand(100, 100)
        labels = segmenter._threshold_segment(test_image)
        stats = segmenter.get_segmentation_stats(labels)
        
        print("📊 Test Results:")
        print(f"   Total cells: {stats['total_cells']}")
        print(f"   Mean area: {stats['mean_area']:.1f} pixels")
        
        print("✅ Segmentation module test complete")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()