"""
Enhanced Wolffia-Specific Segmentation for Real Petri Dish Analysis
Optimized for small, round green specimens against light backgrounds
"""

import logging

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from skimage import filters, measure, morphology, segmentation
from skimage.feature import peak_local_max

logger = logging.getLogger(__name__)

class WolffiaSpecificSegmentation:
    """Specialized segmentation tuned for Wolffia arrhiza in petri dishes"""
    
    def __init__(self, min_area=20, max_area=2000, debug_mode=False):
        self.min_area = min_area  # Smaller minimum for tiny Wolffia
        self.max_area = max_area  # Reasonable maximum 
        self.debug_mode = debug_mode
        self.debug_images = {}
        
    def segment_wolffia_cells(self, image_rgb, output_debug_path=None):
        """
        Segment Wolffia cells with diagnostic visualization
        
        Args:
            image_rgb: RGB image array
            output_debug_path: Path to save debug visualizations
            
        Returns:
            labels: Labeled image with detected cells
            debug_info: Dictionary with intermediate results
        """
        try:
            logger.info("🔬 Starting Wolffia-specific segmentation")
            
            # Step 1: Convert and enhance for green organisms
            lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
            hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
            
            # Enhanced green extraction for Wolffia
            green_enhanced = self._extract_wolffia_green(image_rgb, hsv, lab)
            self.debug_images['01_green_enhanced'] = green_enhanced
            
            # Step 2: Advanced preprocessing
            preprocessed = self._preprocess_for_wolffia(green_enhanced)
            self.debug_images['02_preprocessed'] = preprocessed
            
            # Step 3: Adaptive thresholding
            binary_mask = self._adaptive_threshold_wolffia(preprocessed)
            self.debug_images['03_binary_mask'] = binary_mask
            
            # Step 4: Morphological cleanup
            cleaned_mask = self._morphological_cleanup_wolffia(binary_mask)
            self.debug_images['04_cleaned_mask'] = cleaned_mask
            
            # Step 5: Watershed segmentation for overlapping cells
            labels = self._watershed_segment_wolffia(cleaned_mask, preprocessed)
            self.debug_images['05_labels'] = labels
            
            # Step 6: Quality filtering
            filtered_labels = self._quality_filter_wolffia(labels, image_rgb.shape)
            self.debug_images['06_filtered_labels'] = filtered_labels
            
            # Step 7: Generate visualizations if requested
            if output_debug_path or self.debug_mode:
                self._save_debug_visualizations(image_rgb, filtered_labels, output_debug_path)
            
            cell_count = np.max(filtered_labels)
            logger.info(f"✅ Wolffia segmentation complete: {cell_count} cells detected")
            
            return filtered_labels, {
                'cell_count': cell_count,
                'debug_images': self.debug_images,
                'processing_steps': [
                    'green_extraction', 'preprocessing', 'thresholding', 
                    'morphological_cleanup', 'watershed', 'quality_filtering'
                ]
            }
            
        except Exception as e:
            logger.error(f"❌ Wolffia segmentation error: {str(e)}")
            return np.zeros(image_rgb.shape[:2], dtype=np.int32), {'error': str(e)}
    
    def _extract_wolffia_green(self, rgb, hsv, lab):
        """Extract green channel optimized for Wolffia specimens"""
        try:
            # Multiple green extraction methods
            
            # Method 1: Enhanced green from RGB
            r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
            green_excess = g.astype(np.float32) - 0.5 * (r.astype(np.float32) + b.astype(np.float32))
            green_excess = np.clip(green_excess, 0, 255)
            
            # Method 2: HSV-based green extraction
            h, s, v = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
            green_hue_mask = ((h >= 35) & (h <= 85)) & (s > 30) & (v > 20)
            green_hsv = np.zeros_like(g, dtype=np.float32)
            green_hsv[green_hue_mask] = v[green_hue_mask]
            
            # Method 3: LAB a-channel (green-red axis)
            a_channel = lab[:,:,1].astype(np.float32)
            green_lab = np.clip(128 - a_channel, 0, 255)  # More negative = more green
            
            # Combine methods with weighting
            combined_green = (
                0.4 * green_excess + 
                0.4 * green_hsv + 
                0.2 * green_lab
            )
            
            # Normalize to 0-1 range
            combined_green = combined_green / (combined_green.max() + 1e-8)
            
            return combined_green
            
        except Exception as e:
            logger.error(f"❌ Green extraction error: {str(e)}")
            return rgb[:,:,1] / 255.0
    
    def _preprocess_for_wolffia(self, green_enhanced):
        """Preprocessing optimized for small Wolffia specimens"""
        try:
            # Gentle denoising to preserve small objects
            denoised = filters.gaussian(green_enhanced, sigma=0.8)
            
            # Contrast enhancement
            p2, p98 = np.percentile(denoised, (2, 98))
            enhanced = np.clip((denoised - p2) / (p98 - p2 + 1e-8), 0, 1)
            
            # Subtle sharpening for small objects
            kernel = np.array([[-0.1, -0.1, -0.1],
                              [-0.1,  1.8, -0.1],
                              [-0.1, -0.1, -0.1]])
            sharpened = cv2.filter2D(enhanced, -1, kernel)
            sharpened = np.clip(sharpened, 0, 1)
            
            return sharpened
            
        except Exception as e:
            logger.error(f"❌ Preprocessing error: {str(e)}")
            return green_enhanced
    
    def _adaptive_threshold_wolffia(self, preprocessed):
        """Adaptive thresholding specifically for Wolffia"""
        try:
            # Convert to uint8 for thresholding
            img_uint8 = (preprocessed * 255).astype(np.uint8)
            
            # Multiple thresholding approaches
            
            # Otsu thresholding
            otsu_thresh = filters.threshold_otsu(preprocessed)
            binary_otsu = preprocessed > otsu_thresh
            
            # Local adaptive thresholding
            local_thresh = filters.threshold_local(img_uint8, block_size=25, offset=0.02)
            binary_local = img_uint8 > local_thresh
            
            # Percentile-based thresholding (good for varying illumination)
            percentile_thresh = np.percentile(preprocessed[preprocessed > 0], 75)
            binary_percentile = preprocessed > percentile_thresh
            
            # Combine thresholds with voting
            vote_sum = binary_otsu.astype(int) + binary_local.astype(int) + binary_percentile.astype(int)
            binary_combined = vote_sum >= 2  # At least 2/3 methods agree
            
            return binary_combined
            
        except Exception as e:
            logger.error(f"❌ Thresholding error: {str(e)}")
            return preprocessed > 0.5
    
    def _morphological_cleanup_wolffia(self, binary_mask):
        """Morphological operations tuned for Wolffia size and shape"""
        try:
            # Remove small noise
            cleaned = morphology.remove_small_objects(binary_mask, min_size=self.min_area//2)
            
            # Fill small holes in cells
            filled = morphology.remove_small_holes(cleaned, area_threshold=self.min_area//4)
            
            # Gentle opening to separate slightly touching cells
            selem_open = morphology.disk(1)  # Very small for tiny Wolffia
            opened = morphology.opening(filled, selem_open)
            
            # Closing to restore cell shape
            selem_close = morphology.disk(2)
            closed = morphology.closing(opened, selem_close)
            
            return closed
            
        except Exception as e:
            logger.error(f"❌ Morphological cleanup error: {str(e)}")
            return binary_mask
    
    def _watershed_segment_wolffia(self, binary_mask, intensity_image):
        """Watershed segmentation optimized for overlapping Wolffia"""
        try:
            # Distance transform
            distance = ndimage.distance_transform_edt(binary_mask)
            
            # Find local maxima (cell centers)
            # Use smaller min_distance for tiny Wolffia cells
            min_distance = max(3, int(np.sqrt(self.min_area) / 2))
            
            try:
                local_maxima = peak_local_max(
                    distance, 
                    min_distance=min_distance,
                    threshold_abs=0.3 * distance.max(),
                    indices=False
                )
            except:
                # Fallback if peak_local_max not available
                from scipy.ndimage import maximum_filter
                maxima_mask = maximum_filter(distance, size=min_distance*2+1) == distance
                local_maxima = maxima_mask & (distance > 0.3 * distance.max())
            
            # Create markers
            markers = measure.label(local_maxima)
            
            # Watershed segmentation
            labels = segmentation.watershed(-distance, markers, mask=binary_mask)
            
            return labels
            
        except Exception as e:
            logger.error(f"❌ Watershed error: {str(e)}")
            return measure.label(binary_mask)
    
    def _quality_filter_wolffia(self, labels, image_shape):
        """Quality filtering specific to Wolffia characteristics"""
        try:
            if np.max(labels) == 0:
                return labels
            
            regions = measure.regionprops(labels)
            valid_labels = []
            
            # Calculate image border region (5% from edges)
            border_width = min(image_shape[0], image_shape[1]) * 0.05
            
            for region in regions:
                # Size filtering
                if not (self.min_area <= region.area <= self.max_area):
                    continue
                
                # Shape filtering for Wolffia (roughly circular)
                if region.eccentricity > 0.9:  # Too elongated
                    continue
                
                if region.solidity < 0.6:  # Too irregular
                    continue
                
                # Border filtering
                min_row, min_col, max_row, max_col = region.bbox
                if (min_row < border_width or min_col < border_width or 
                    max_row > image_shape[0] - border_width or 
                    max_col > image_shape[1] - border_width):
                    continue
                
                # Aspect ratio check (Wolffia should be roughly round)
                if region.major_axis_length / (region.minor_axis_length + 1e-8) > 2.5:
                    continue
                
                valid_labels.append(region.label)
            
            # Create filtered label image
            filtered_labels = np.zeros_like(labels)
            for i, old_label in enumerate(valid_labels, 1):
                filtered_labels[labels == old_label] = i
            
            return filtered_labels
            
        except Exception as e:
            logger.error(f"❌ Quality filtering error: {str(e)}")
            return labels
    
    def _save_debug_visualizations(self, original_image, labels, output_path):
        """Create comprehensive debug visualizations"""
        try:
            if not self.debug_mode and not output_path:
                return
            
            fig, axes = plt.subplots(3, 3, figsize=(15, 15))
            fig.suptitle('Wolffia Segmentation Debug Analysis', fontsize=16)
            
            # Original image
            axes[0,0].imshow(original_image)
            axes[0,0].set_title('Original Image')
            axes[0,0].axis('off')
            
            # Green enhanced
            if '01_green_enhanced' in self.debug_images:
                axes[0,1].imshow(self.debug_images['01_green_enhanced'], cmap='viridis')
                axes[0,1].set_title('Green Enhanced')
                axes[0,1].axis('off')
            
            # Preprocessed
            if '02_preprocessed' in self.debug_images:
                axes[0,2].imshow(self.debug_images['02_preprocessed'], cmap='gray')
                axes[0,2].set_title('Preprocessed')
                axes[0,2].axis('off')
            
            # Binary mask
            if '03_binary_mask' in self.debug_images:
                axes[1,0].imshow(self.debug_images['03_binary_mask'], cmap='gray')
                axes[1,0].set_title('Binary Mask')
                axes[1,0].axis('off')
            
            # Cleaned mask
            if '04_cleaned_mask' in self.debug_images:
                axes[1,1].imshow(self.debug_images['04_cleaned_mask'], cmap='gray')
                axes[1,1].set_title('Cleaned Mask')
                axes[1,1].axis('off')
            
            # Watershed labels
            if '05_labels' in self.debug_images:
                axes[1,2].imshow(self.debug_images['05_labels'], cmap='nipy_spectral')
                axes[1,2].set_title('Watershed Labels')
                axes[1,2].axis('off')
            
            # Final segmentation overlay
            overlay = original_image.copy()
            if np.max(labels) > 0:
                # Create colored overlay
                colored_labels = np.zeros_like(original_image)
                unique_labels = np.unique(labels)[1:]  # Skip background
                
                colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
                for i, label_val in enumerate(unique_labels):
                    mask = labels == label_val
                    colored_labels[mask] = colors[i][:3] * 255
                
                overlay = cv2.addWeighted(original_image, 0.7, 
                                        colored_labels.astype(np.uint8), 0.3, 0)
            
            axes[2,0].imshow(overlay)
            axes[2,0].set_title(f'Final Result ({np.max(labels)} cells)')
            axes[2,0].axis('off')
            
            # Cell size histogram
            if np.max(labels) > 0:
                regions = measure.regionprops(labels)
                areas = [r.area for r in regions]
                axes[2,1].hist(areas, bins=20, edgecolor='black')
                axes[2,1].set_title('Cell Size Distribution')
                axes[2,1].set_xlabel('Area (pixels)')
                axes[2,1].set_ylabel('Count')
            
            # Detection overlay with numbers
            axes[2,2].imshow(original_image)
            if np.max(labels) > 0:
                regions = measure.regionprops(labels)
                for region in regions:
                    y, x = region.centroid
                    axes[2,2].plot(x, y, 'r+', markersize=8, markeredgewidth=2)
                    axes[2,2].text(x+5, y-5, str(region.label), 
                                  color='red', fontsize=8, fontweight='bold')
            axes[2,2].set_title('Detected Cells with IDs')
            axes[2,2].axis('off')
            
            plt.tight_layout()
            
            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                logger.info(f"📊 Debug visualization saved to: {output_path}")
            
            if self.debug_mode:
                plt.show()
            else:
                plt.close()
                
        except Exception as e:
            logger.error(f"❌ Debug visualization error: {str(e)}")

def run_pipeline(image_path_or_array, debug_mode=True, output_debug_path=None):
    """
    Standalone function to run Wolffia segmentation pipeline
    Compatible with existing run_pipeline calls
    """
    try:
        # Handle both file paths and arrays
        if isinstance(image_path_or_array, str):
            image = cv2.imread(image_path_or_array)
            if image is None:
                raise ValueError(f"Could not load image: {image_path_or_array}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = image_path_or_array
            # Ensure proper format
            if len(image.shape) != 3 or image.shape[2] != 3:
                raise ValueError(f"Invalid image shape: {image.shape}")
        
        # Initialize segmenter
        segmenter = WolffiaSpecificSegmentation(
            min_area=20,     # Adjusted for small Wolffia
            max_area=2000,   # Reasonable maximum
            debug_mode=debug_mode
        )
        
        # Run segmentation
        labels, debug_info = segmenter.segment_wolffia_cells(image, output_debug_path)
        
        # Extract cell measurements (compatible with existing code)
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        regions = measure.regionprops(labels, intensity_image=gray_image)
        
        results = {
            "cell_id": [],
            "cell_area": [],
            "int_mem_mean": [],  # Mean intensity
            "int_mean": [],      # Mean intensity (duplicate for compatibility)
            "cell_edge": []      # Perimeter
        }
        
        for region in regions:
            results["cell_id"].append(region.label)
            results["cell_area"].append(region.area)
            results["int_mem_mean"].append(region.mean_intensity)
            results["int_mean"].append(region.mean_intensity)
            results["cell_edge"].append(region.perimeter)
        
        logger.info(f"✅ Segmentation pipeline complete: {len(results['cell_id'])} cells")
        return labels, results
        
    except Exception as e:
        logger.error(f"❌ Pipeline error: {str(e)}")
        # Return properly sized empty results to prevent crashes
        default_shape = (400, 400) if isinstance(image_path_or_array, str) else image_path_or_array.shape[:2]
        empty_labels = np.zeros(default_shape, dtype=np.int32)
        empty_results = {
            "cell_id": [],
            "cell_area": [],
            "int_mem_mean": [],
            "int_mean": [],
            "cell_edge": []
        }
        return empty_labels, empty_results
    
# Test the pipeline
if __name__ == "__main__":
    print("🧪 Testing Wolffia-specific segmentation pipeline...")
    
    # Create test image similar to Wolffia plate
    test_image = np.ones((500, 500, 3), dtype=np.uint8) * 240  # Light background
    
    # Add some green circular objects (simulating Wolffia)
    centers = [(100, 100), (200, 150), (300, 200), (150, 300), (400, 350)]
    for center in centers:
        cv2.circle(test_image, center, 15, (50, 150, 50), -1)  # Green circles
        cv2.circle(test_image, center, 12, (30, 180, 30), -1)  # Brighter center
    
    # Test the pipeline
    labels, results = run_pipeline(test_image, debug_mode=True)
    
    print(f"✅ Test complete: {len(results['cell_id'])} cells detected")
    print(f"   Cell areas: {results['cell_area']}")