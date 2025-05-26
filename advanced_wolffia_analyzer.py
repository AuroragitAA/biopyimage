"""
advanced_wolffia_analyzer.py

Professional-grade Wolffia bioimage analysis system with advanced algorithms,
statistical analysis, and comprehensive biological feature extraction.

This system implements state-of-the-art techniques for:
- Multi-scale image analysis
- Advanced morphological characterization  
- Growth dynamics quantification
- Health assessment algorithms
- Statistical validation and quality control
- Machine learning-enhanced classification

Author: Senior Bioinformatics Developer
Version: 2.0.0
"""

import json
import logging
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage, stats
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from skimage import feature, filters, measure, morphology, restoration, segmentation
from skimage.color import lab2rgb, rgb2hsv, rgb2lab
from skimage.util import img_as_float, img_as_ubyte
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from analysis_config import AnalysisConfig

warnings.filterwarnings('ignore')

# Configure professional logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('wolffia_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

import os

os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/bioimagin_app.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)



class AdvancedImageProcessor:
    """Advanced image processing with multi-scale analysis and quality assessment."""
    
    def __init__(self, config: AnalysisConfig):
        self.config = AnalysisConfig
        logger.info("🔬 Advanced Image Processor initialized")
    
    def process_image_advanced(self, image_input: Union[str, np.ndarray]) -> Dict:
        """
        Advanced image processing with comprehensive quality assessment.
        
        Returns:
            Dict containing processed images, quality metrics, and metadata
        """
        try:
            # Load and validate image
            image_data = self._load_and_validate_image(image_input)
            if image_data is None:
                return None
            
            original = image_data['image']
            quality_score = image_data['quality_score']
            
            logger.info(f"📊 Image quality score: {quality_score:.3f}")
            
            # Multi-scale decomposition
            multi_scale = self._multi_scale_decomposition(original)
            
            # Advanced color space analysis
            color_analysis = self._advanced_color_analysis(original)
            
            # Enhanced preprocessing pipeline
            processed = self._enhanced_preprocessing(original, multi_scale, color_analysis)
            
            # Texture and structural analysis
            texture_features = self._extract_texture_features(processed['gray'])
            
            result = {
                'original': original,
                'processed': processed,
                'multi_scale': multi_scale,
                'color_analysis': color_analysis,
                'texture_features': texture_features,
                'quality_score': quality_score,
                'metadata': {
                    'processing_timestamp': datetime.now().isoformat(),
                    'image_shape': original.shape,
                    'pixel_to_micron': self.config.pixel_to_micron
                }
            }
            
            logger.info("✅ Advanced image processing complete")
            return result
            
        except Exception as e:
            logger.error(f"❌ Advanced processing error: {str(e)}")
            return None
    
    def _load_and_validate_image(self, image_input: Union[str, np.ndarray]) -> Dict:
        """Load image with comprehensive validation and quality assessment."""
        try:
            if isinstance(image_input, str):
                image = cv2.imread(image_input)
                if image is None:
                    raise ValueError(f"Cannot load image: {image_input}")
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image = self._validate_array(image_input)
            
            # Calculate quality metrics
            quality_metrics = self._assess_image_quality(image)
            
            return {
                'image': image,
                'quality_score': quality_metrics['overall_score'],
                'quality_details': quality_metrics
            }
            
        except Exception as e:
            logger.error(f"❌ Image loading error: {str(e)}")
            return None
    
    def _assess_image_quality(self, image: np.ndarray) -> Dict:
        """Comprehensive image quality assessment."""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Sharpness (Laplacian variance)
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness_score = min(sharpness / 1000.0, 1.0)  # Normalize
            
            # Contrast (standard deviation)
            contrast = np.std(gray)
            contrast_score = min(contrast / 128.0, 1.0)  # Normalize
            
            # Brightness balance
            brightness = np.mean(gray)
            brightness_score = 1.0 - abs(brightness - 128) / 128.0
            
            # Noise estimation (using high-frequency content)
            high_freq = filters.sobel(gray)
            noise_level = np.std(high_freq)
            noise_score = max(0, 1.0 - noise_level / 50.0)
            
            # Color balance (RGB channel variance)
            rgb_means = [np.mean(image[:,:,i]) for i in range(3)]
            color_balance = 1.0 - np.std(rgb_means) / 128.0
            
            # Overall quality score (weighted combination)
            weights = {
                'sharpness': 0.3,
                'contrast': 0.25,
                'brightness': 0.2,
                'noise': 0.15,
                'color_balance': 0.1
            }
            
            overall_score = (
                weights['sharpness'] * sharpness_score +
                weights['contrast'] * contrast_score +
                weights['brightness'] * brightness_score +
                weights['noise'] * noise_score +
                weights['color_balance'] * color_balance
            )
            
            return {
                'overall_score': overall_score,
                'sharpness': sharpness_score,
                'contrast': contrast_score,
                'brightness': brightness_score,
                'noise': noise_score,
                'color_balance': color_balance
            }
            
        except Exception as e:
            logger.error(f"❌ Quality assessment error: {str(e)}")
            return {'overall_score': 0.5}  # Default moderate quality
    
    def _multi_scale_decomposition(self, image: np.ndarray) -> Dict:
        """Multi-scale image decomposition for enhanced analysis."""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            scales = {}
            
            for level in range(self.config.multi_scale_levels):
                sigma = 2 ** level
                
                # Gaussian pyramid
                smoothed = filters.gaussian(gray, sigma=sigma)
                
                # Difference of Gaussians for edge detection
                if level > 0:
                    dog = scales[f'smoothed_{level-1}'] - smoothed
                    scales[f'dog_{level}'] = dog
                
                scales[f'smoothed_{level}'] = smoothed
                
                # Laplacian for detail enhancement
                laplacian = filters.laplace(smoothed)
                scales[f'laplacian_{level}'] = laplacian
            
            logger.info(f"📈 Multi-scale decomposition: {len(scales)} scales")
            return scales
            
        except Exception as e:
            logger.error(f"❌ Multi-scale decomposition error: {str(e)}")
            return {}
    
    def _advanced_color_analysis(self, image: np.ndarray) -> Dict:
        """Advanced color space analysis for biological features."""
        try:
            # Multiple color space conversions
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            lab = rgb2lab(image)
            
            # Chlorophyll-specific analysis
            chlorophyll_index = self._calculate_chlorophyll_index(image)
            
            # Green vegetation index
            gvi = self._calculate_green_vegetation_index(image)
            
            # Color distribution analysis
            color_hist = self._analyze_color_distribution(image, hsv)
            
            # Spectral analysis
            spectral_features = self._extract_spectral_features(image)
            
            return {
                'hsv': hsv,
                'lab': lab,
                'chlorophyll_index': chlorophyll_index,
                'green_vegetation_index': gvi,
                'color_histogram': color_hist,
                'spectral_features': spectral_features
            }
            
        except Exception as e:
            logger.error(f"❌ Color analysis error: {str(e)}")
            return {}
    
    def _calculate_chlorophyll_index(self, image: np.ndarray) -> np.ndarray:
        """Calculate advanced chlorophyll content index."""
        try:
            # Normalize channels
            r, g, b = image[:,:,0]/255.0, image[:,:,1]/255.0, image[:,:,2]/255.0
            
            # Multiple chlorophyll indices
            # Green Leaf Index (GLI)
            gli = (2*g - r - b) / (2*g + r + b + 1e-8)
            
            # Visible Atmospherically Resistant Index (VARI)
            vari = (g - r) / (g + r - b + 1e-8)
            
            # Combined chlorophyll index
            chlorophyll_index = 0.6 * gli + 0.4 * vari
            
            return np.clip(chlorophyll_index, -1, 1)
            
        except Exception as e:
            logger.error(f"❌ Chlorophyll index calculation error: {str(e)}")
            return np.zeros_like(image[:,:,0])
    
    def _calculate_green_vegetation_index(self, image: np.ndarray) -> np.ndarray:
        """Calculate Green Vegetation Index for photosynthetic activity."""
        try:
            r, g, b = image[:,:,0], image[:,:,1], image[:,:,2]
            
            # Excess Green Index
            gvi = 2*g - r - b
            
            # Normalize
            gvi = (gvi - gvi.min()) / (gvi.max() - gvi.min() + 1e-8)
            
            return gvi
            
        except Exception as e:
            logger.error(f"❌ GVI calculation error: {str(e)}")
            return np.zeros_like(image[:,:,0])
    
    def _analyze_color_distribution(self, image: np.ndarray, hsv: np.ndarray) -> Dict:
        """Analyze color distribution for biological insights."""
        try:
            # HSV histograms
            h_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
            s_hist = cv2.calcHist([hsv], [1], None, [256], [0, 256])
            v_hist = cv2.calcHist([hsv], [2], None, [256], [0, 256])
            
            # Dominant colors using K-means
            pixels = image.reshape(-1, 3)
            kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
            kmeans.fit(pixels)
            dominant_colors = kmeans.cluster_centers_
            
            # Green content analysis
            green_mask = (hsv[:,:,0] >= 35) & (hsv[:,:,0] <= 85) & (hsv[:,:,1] > 50)
            green_percentage = np.sum(green_mask) / green_mask.size * 100
            
            return {
                'hue_histogram': h_hist.flatten(),
                'saturation_histogram': s_hist.flatten(),
                'value_histogram': v_hist.flatten(),
                'dominant_colors': dominant_colors,
                'green_percentage': green_percentage
            }
            
        except Exception as e:
            logger.error(f"❌ Color distribution analysis error: {str(e)}")
            return {}
    
    def _extract_spectral_features(self, image: np.ndarray) -> Dict:
        """Extract spectral features for advanced analysis."""
        try:
            # RGB ratios
            r, g, b = image[:,:,0], image[:,:,1], image[:,:,2]
            total = r + g + b + 1e-8
            
            r_ratio = np.mean(r / total)
            g_ratio = np.mean(g / total)
            b_ratio = np.mean(b / total)
            
            # Color moments
            rgb_means = [np.mean(image[:,:,i]) for i in range(3)]
            rgb_stds = [np.std(image[:,:,i]) for i in range(3)]
            rgb_skews = [stats.skew(image[:,:,i].flatten()) for i in range(3)]
            
            return {
                'rgb_ratios': [r_ratio, g_ratio, b_ratio],
                'rgb_means': rgb_means,
                'rgb_stds': rgb_stds,
                'rgb_skewness': rgb_skews
            }
            
        except Exception as e:
            logger.error(f"❌ Spectral feature extraction error: {str(e)}")
            return {}
    
    def _enhanced_preprocessing(self, image: np.ndarray, multi_scale: Dict, color_analysis: Dict) -> Dict:
        """Enhanced preprocessing pipeline with adaptive parameters."""
        try:
            # Adaptive denoising based on image quality
            denoised = restoration.denoise_bilateral(
                image, 
                sigma_color=0.1, 
                sigma_spatial=self.config.noise_reduction_sigma
            )
            
            # Multi-scale edge enhancement
            enhanced_edges = np.zeros_like(image[:,:,0])
            for level in range(self.config.multi_scale_levels):
                if f'laplacian_{level}' in multi_scale:
                    weight = 1.0 / (2 ** level)
                    enhanced_edges += weight * multi_scale[f'laplacian_{level}']
            
            # Adaptive contrast enhancement
            gray = cv2.cvtColor(denoised, cv2.COLOR_RGB2GRAY)
            clahe = cv2.createCLAHE(clipLimit=self.config.contrast_enhancement_clip, tileGridSize=(8,8))
            enhanced_gray = clahe.apply(img_as_ubyte(gray))
            
            # Enhanced green channel with biological relevance
            green_enhanced = color_analysis.get('chlorophyll_index', image[:,:,1])
            
            return {
                'denoised': denoised,
                'gray': enhanced_gray,
                'green_enhanced': green_enhanced,
                'edge_enhanced': enhanced_edges,
                'chlorophyll_map': color_analysis.get('chlorophyll_index', np.zeros_like(gray))
            }
            
        except Exception as e:
            logger.error(f"❌ Enhanced preprocessing error: {str(e)}")
            return {'gray': cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)}
    
    def _extract_texture_features(self, gray_image: np.ndarray) -> Dict:
        """Extract comprehensive texture features."""
        try:
            # Gray Level Co-occurrence Matrix (GLCM) features
            from skimage.feature import greycomatrix, greycoprops
            
            # Normalize to 0-255 and reduce levels for GLCM
            normalized = img_as_ubyte(gray_image)
            reduced = normalized // 4  # Reduce to 64 levels
            
            # Calculate GLCM for multiple directions and distances
            glcm = greycomatrix(reduced, [1, 2], [0, 45, 90, 135], levels=64, symmetric=True, normed=True)
            
            # Extract texture properties
            contrast = greycoprops(glcm, 'contrast').mean()
            dissimilarity = greycoprops(glcm, 'dissimilarity').mean()
            homogeneity = greycoprops(glcm, 'homogeneity').mean()
            energy = greycoprops(glcm, 'energy').mean()
            correlation = greycoprops(glcm, 'correlation').mean()
            
            # Local Binary Pattern (LBP)
            from skimage.feature import local_binary_pattern
            lbp = local_binary_pattern(gray_image, P=8, R=1, method='uniform')
            lbp_hist, _ = np.histogram(lbp.ravel(), bins=10)
            lbp_uniformity = np.sum(lbp_hist ** 2) / (lbp_hist.sum() ** 2)
            
            return {
                'glcm_contrast': contrast,
                'glcm_dissimilarity': dissimilarity,
                'glcm_homogeneity': homogeneity,
                'glcm_energy': energy,
                'glcm_correlation': correlation,
                'lbp_uniformity': lbp_uniformity,
                'entropy': measure.shannon_entropy(gray_image)
            }
            
        except Exception as e:
            logger.error(f"❌ Texture feature extraction error: {str(e)}")
            return {}
    
    def _validate_array(self, array: np.ndarray) -> np.ndarray:
        """Validate and normalize input array."""
        if len(array.shape) != 3 or array.shape[2] != 3:
            raise ValueError(f"Invalid array shape: {array.shape}")
        
        if array.max() <= 1.0:
            return (array * 255).astype(np.uint8)
        return array.astype(np.uint8)


class BiologicalFeatureExtractor:
    """Advanced biological feature extraction with machine learning enhancement."""
    
    def __init__(self, config: AnalysisConfig):
        self.config = AnalysisConfig
        self.scaler = StandardScaler()
        logger.info("🧬 Biological Feature Extractor initialized")
    
    def extract_comprehensive_features(self, labels: np.ndarray, image_data: Dict) -> pd.DataFrame:
        """Extract comprehensive biological features for each detected cell."""
        try:
            if np.max(labels) == 0:
                return pd.DataFrame()
            
            regions = measure.regionprops(labels, intensity_image=image_data['processed']['gray'])
            
            features_list = []
            for region in regions:
                features = self._extract_single_cell_features(region, image_data, labels)
                if features:
                    features_list.append(features)
            
            if not features_list:
                return pd.DataFrame()
            
            df = pd.DataFrame(features_list)
            
            # Add derived features
            df = self._add_derived_features(df)
            
            # Calculate health scores
            df = self._calculate_health_scores(df)
            
            # Detect outliers
            df = self._detect_outliers(df)
            
            logger.info(f"🔬 Extracted features for {len(df)} cells")
            return df
            
        except Exception as e:
            logger.error(f"❌ Feature extraction error: {str(e)}")
            return pd.DataFrame()
    
    def _extract_single_cell_features(self, region, image_data: Dict, labels: np.ndarray) -> Dict:
        """Extract comprehensive features for a single cell."""
        try:
            # Basic morphological features
            features = {
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
                'euler_number': region.euler_number
            }
            
            # Shape descriptors
            features.update(self._calculate_shape_descriptors(region))
            
            # Intensity features
            features.update(self._calculate_intensity_features(region, image_data))
            
            # Texture features
            features.update(self._calculate_local_texture_features(region, image_data))
            
            # Biological indices
            features.update(self._calculate_biological_indices(region, image_data))
            
            return features
            
        except Exception as e:
            logger.error(f"❌ Single cell feature extraction error: {str(e)}")
            return {}
    
    def _calculate_shape_descriptors(self, region) -> Dict:
        """Calculate advanced shape descriptors."""
        try:
            # Aspect ratio
            aspect_ratio = region.major_axis_length / (region.minor_axis_length + 1e-8)
            
            # Compactness (isoperimetric quotient)
            compactness = 4 * np.pi * region.area / (region.perimeter ** 2 + 1e-8)
            
            # Convex hull features
            convex_area = region.convex_area if hasattr(region, 'convex_area') else region.area
            convexity = region.area / (convex_area + 1e-8)
            
            # Circularity
            circularity = 4 * np.pi * region.area / (region.perimeter ** 2 + 1e-8)
            
            # Roundness
            roundness = 4 * region.area / (np.pi * region.major_axis_length ** 2 + 1e-8)
            
            return {
                'aspect_ratio': aspect_ratio,
                'compactness': compactness,
                'convexity': convexity,
                'circularity': circularity,
                'roundness': roundness,
                'shape_factor': region.perimeter / (2 * np.sqrt(np.pi * region.area) + 1e-8)
            }
            
        except Exception as e:
            logger.error(f"❌ Shape descriptor calculation error: {str(e)}")
            return {}
    
    def _calculate_intensity_features(self, region, image_data: Dict) -> Dict:
        """Calculate intensity-based features."""
        try:
            processed = image_data['processed']
            
            # Mean intensities in different channels
            mean_intensity = region.mean_intensity
            
            # Chlorophyll content
            chlorophyll_map = processed.get('chlorophyll_map', np.zeros_like(processed['gray']))
            mask = region.image
            
            if mask.size > 0:
                chlorophyll_values = chlorophyll_map[region.slice][mask]
                chlorophyll_mean = np.mean(chlorophyll_values) if len(chlorophyll_values) > 0 else 0
                chlorophyll_std = np.std(chlorophyll_values) if len(chlorophyll_values) > 0 else 0
            else:
                chlorophyll_mean = 0
                chlorophyll_std = 0
            
            # Green vegetation index
            if 'green_enhanced' in processed:
                green_values = processed['green_enhanced'][region.slice][mask]
                green_intensity = np.mean(green_values) if len(green_values) > 0 else 0
            else:
                green_intensity = mean_intensity
            
            return {
                'mean_intensity': mean_intensity,
                'chlorophyll_content': chlorophyll_mean,
                'chlorophyll_variability': chlorophyll_std,
                'green_intensity': green_intensity,
                'intensity_uniformity': 1.0 / (1.0 + np.std(chlorophyll_values) + 1e-8) if 'chlorophyll_values' in locals() else 0.5
            }
            
        except Exception as e:
            logger.error(f"❌ Intensity feature calculation error: {str(e)}")
            return {}
    
    def _calculate_local_texture_features(self, region, image_data: Dict) -> Dict:
        """Calculate local texture features for the cell region."""
        try:
            # Extract cell region
            mask = region.image
            gray_region = image_data['processed']['gray'][region.slice][mask]
            
            if len(gray_region) < 5:  # Too small for texture analysis
                return {
                    'local_contrast': 0,
                    'local_entropy': 0,
                    'local_uniformity': 0
                }
            
            # Local contrast
            local_contrast = np.std(gray_region)
            
            # Local entropy
            hist, _ = np.histogram(gray_region, bins=16, range=(0, 255))
            hist = hist / (hist.sum() + 1e-8)
            local_entropy = -np.sum(hist * np.log2(hist + 1e-8))
            
            # Local uniformity (inverse of variance)
            local_uniformity = 1.0 / (1.0 + np.var(gray_region))
            
            return {
                'local_contrast': local_contrast,
                'local_entropy': local_entropy,
                'local_uniformity': local_uniformity
            }
            
        except Exception as e:
            logger.error(f"❌ Local texture calculation error: {str(e)}")
            return {}
    
    def _calculate_biological_indices(self, region, image_data: Dict) -> Dict:
        """Calculate biologically relevant indices."""
        try:
            # Cell integrity score (based on shape regularity)
            integrity_score = region.solidity * (1.0 - region.eccentricity)
            
            # Photosynthetic activity estimate
            chlorophyll_content = region.mean_intensity  # Simplified
            photosynthetic_activity = min(chlorophyll_content / 255.0, 1.0)
            
            # Cell maturity estimate (based on size and shape)
            size_factor = min(region.area / 1000.0, 1.0)  # Normalize by expected mature size
            shape_factor = 1.0 - abs(region.eccentricity - 0.5)  # Optimal eccentricity around 0.5
            maturity_estimate = (size_factor + shape_factor) / 2.0
            
            # Stress indicators
            shape_irregularity = 1.0 - region.solidity
            size_deviation = abs(region.area - 500) / 500.0  # Deviation from typical size
            stress_indicator = (shape_irregularity + min(size_deviation, 1.0)) / 2.0
            
            return {
                'cell_integrity': integrity_score,
                'photosynthetic_activity': photosynthetic_activity,
                'maturity_estimate': maturity_estimate,
                'stress_indicator': stress_indicator,
                'health_potential': (integrity_score + photosynthetic_activity) / 2.0
            }
            
        except Exception as e:
            logger.error(f"❌ Biological indices calculation error: {str(e)}")
            return {}
    
    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived features based on population statistics."""
        try:
            if len(df) == 0:
                return df
            
            # Size categories
            df['size_category'] = pd.cut(df['area'], bins=3, labels=['small', 'medium', 'large'])
            
            # Normalized features (z-scores)
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col not in ['cell_id', 'centroid_x', 'centroid_y']:
                    df[f'{col}_zscore'] = (df[col] - df[col].mean()) / (df[col].std() + 1e-8)
            
            # Spatial features
            if 'centroid_x' in df.columns and 'centroid_y' in df.columns:
                # Distance from image center
                center_x, center_y = df['centroid_x'].mean(), df['centroid_y'].mean()
                df['distance_from_center'] = np.sqrt(
                    (df['centroid_x'] - center_x)**2 + (df['centroid_y'] - center_y)**2
                )
                
                # Local density (number of neighbors within certain radius)
                df['local_density'] = self._calculate_local_density(df)
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Derived features calculation error: {str(e)}")
            return df
    
    def _calculate_local_density(self, df: pd.DataFrame, radius: float = 100.0) -> List[int]:
        """Calculate local cell density for each cell."""
        try:
            coords = df[['centroid_x', 'centroid_y']].values
            distances = squareform(pdist(coords))
            
            local_densities = []
            for i in range(len(df)):
                neighbors = np.sum(distances[i] < radius) - 1  # Exclude self
                local_densities.append(neighbors)
            
            return local_densities
            
        except Exception as e:
            logger.error(f"❌ Local density calculation error: {str(e)}")
            return [0] * len(df)
    
    def _calculate_health_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive health scores."""
        try:
            if len(df) == 0:
                return df
            
            weights = self.config.health_score_weights
            
            # Normalize features to 0-1 scale for health score calculation
            features_for_health = {}
            
            if 'chlorophyll_content' in df.columns:
                features_for_health['chlorophyll'] = df['chlorophyll_content'] / df['chlorophyll_content'].max()
            
            if 'cell_integrity' in df.columns:
                features_for_health['integrity'] = df['cell_integrity']
            
            if 'area' in df.columns:
                # Size consistency (closer to median is better)
                median_size = df['area'].median()
                features_for_health['size_consistency'] = 1.0 - np.abs(df['area'] - median_size) / median_size
            
            if 'local_uniformity' in df.columns:
                features_for_health['texture'] = df['local_uniformity']
            
            if 'circularity' in df.columns:
                features_for_health['shape'] = df['circularity']
            
            # Calculate weighted health score
            health_scores = []
            for i in range(len(df)):
                score = 0
                total_weight = 0
                
                for feature, weight_key in [
                    ('chlorophyll', 'chlorophyll_content'),
                    ('integrity', 'cell_integrity'),
                    ('size_consistency', 'size_consistency'),
                    ('texture', 'texture_uniformity'),
                    ('shape', 'shape_regularity')
                ]:
                    if feature in features_for_health:
                        weight = weights.get(weight_key, 0.2)
                        score += weight * features_for_health[feature].iloc[i]
                        total_weight += weight
                
                final_score = score / total_weight if total_weight > 0 else 0.5
                health_scores.append(final_score)
            
            df['health_score'] = health_scores
            
            # Health categories
            df['health_category'] = pd.cut(
                df['health_score'], 
                bins=[0, 0.3, 0.7, 1.0], 
                labels=['poor', 'moderate', 'excellent']
            )
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Health score calculation error: {str(e)}")
            return df
    
    def _detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect outlier cells using machine learning."""
        try:
            if len(df) < 10:  # Need minimum samples for outlier detection
                df['is_outlier'] = False
                return df
            
            # Select numerical features for outlier detection
            numeric_features = df.select_dtypes(include=[np.number]).columns
            feature_cols = [col for col in numeric_features if 'zscore' not in col and col not in ['cell_id', 'centroid_x', 'centroid_y']]
            
            if len(feature_cols) == 0:
                df['is_outlier'] = False
                return df
            
            # Use Isolation Forest for outlier detection
            isolation_forest = IsolationForest(
                contamination=self.config.outlier_detection_contamination,
                random_state=42
            )
            
            X = df[feature_cols].fillna(0)  # Handle any NaN values
            outlier_predictions = isolation_forest.fit_predict(X)
            
            df['is_outlier'] = outlier_predictions == -1
            df['outlier_score'] = isolation_forest.decision_function(X)
            
            logger.info(f"🔍 Outlier detection: {np.sum(df['is_outlier'])} outliers found")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Outlier detection error: {str(e)}")
            df['is_outlier'] = False
            return df


class StatisticalAnalyzer:
    """Advanced statistical analysis and reporting."""
    
    def __init__(self, config: AnalysisConfig):
        self.config = AnalysisConfig
        logger.info("📊 Statistical Analyzer initialized")
    
    def generate_comprehensive_report(self, results_list: List[Dict]) -> Dict:
        """Generate comprehensive statistical report from multiple analyses."""
        try:
            if not results_list:
                return self._empty_report()
            
            successful_results = [r for r in results_list if r.get('success', False)]
            
            if not successful_results:
                return self._empty_report()
            
            # Aggregate data
            all_cells_data = []
            for result in successful_results:
                if 'cell_data' in result:
                    all_cells_data.extend(result['cell_data'])
            
            if not all_cells_data:
                return self._empty_report()
            
            df = pd.DataFrame(all_cells_data)
            
            # Basic statistics
            basic_stats = self._calculate_basic_statistics(df)
            
            # Population analysis
            population_analysis = self._analyze_population_dynamics(df)
            
            # Health assessment
            health_assessment = self._assess_population_health(df)
            
            # Quality metrics
            quality_metrics = self._calculate_quality_metrics(successful_results)
            
            # Trends and patterns
            temporal_analysis = self._analyze_temporal_patterns(successful_results)
            
            # Confidence intervals
            confidence_intervals = self._calculate_confidence_intervals(df)
            
            report = {
                'summary': {
                    'total_images_analyzed': len(successful_results),
                    'total_cells_detected': len(df),
                    'analysis_period': self._get_analysis_period(successful_results),
                    'success_rate': len(successful_results) / len(results_list) * 100
                },
                'basic_statistics': basic_stats,
                'population_analysis': population_analysis,
                'health_assessment': health_assessment,
                'quality_metrics': quality_metrics,
                'temporal_analysis': temporal_analysis,
                'confidence_intervals': confidence_intervals,
                'recommendations': self._generate_recommendations(df, health_assessment)
            }
            
            logger.info("📋 Comprehensive report generated")
            return report
            
        except Exception as e:
            logger.error(f"❌ Report generation error: {str(e)}")
            return self._empty_report()
    
    def _calculate_basic_statistics(self, df: pd.DataFrame) -> Dict:
        """Calculate basic descriptive statistics."""
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            stats = {}
            for col in numeric_cols:
                if col not in ['cell_id', 'centroid_x', 'centroid_y']:
                    stats[col] = {
                        'mean': float(df[col].mean()),
                        'median': float(df[col].median()),
                        'std': float(df[col].std()),
                        'min': float(df[col].min()),
                        'max': float(df[col].max()),
                        'q25': float(df[col].quantile(0.25)),
                        'q75': float(df[col].quantile(0.75))
                    }
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Basic statistics calculation error: {str(e)}")
            return {}
    
    def _analyze_population_dynamics(self, df: pd.DataFrame) -> Dict:
        """Analyze population-level dynamics."""
        try:
            # Size distribution analysis
            size_distribution = {
                'small_cells': len(df[df['area'] < 100]),
                'medium_cells': len(df[(df['area'] >= 100) & (df['area'] < 500)]),
                'large_cells': len(df[df['area'] >= 500])
            }
            
            # Spatial distribution
            spatial_stats = {}
            if 'centroid_x' in df.columns and 'centroid_y' in df.columns:
                spatial_stats = {
                    'spatial_dispersion': float(np.std(df['centroid_x']) + np.std(df['centroid_y'])),
                    'clustering_coefficient': self._calculate_clustering_coefficient(df)
                }
            
            # Morphological diversity
            morphological_diversity = {}
            if 'eccentricity' in df.columns and 'aspect_ratio' in df.columns:
                morphological_diversity = {
                    'shape_diversity_index': float(np.std(df['eccentricity'])),
                    'size_diversity_index': float(np.std(df['area']) / np.mean(df['area']))
                }
            
            return {
                'size_distribution': size_distribution,
                'spatial_distribution': spatial_stats,
                'morphological_diversity': morphological_diversity,
                'population_density': len(df) / (df['area'].sum() / 1000.0) if 'area' in df.columns else 0
            }
            
        except Exception as e:
            logger.error(f"❌ Population analysis error: {str(e)}")
            return {}
    
    def _assess_population_health(self, df: pd.DataFrame) -> Dict:
        """Assess overall population health."""
        try:
            health_metrics = {}
            
            # Health score distribution
            if 'health_score' in df.columns:
                health_metrics['health_distribution'] = {
                    'excellent': len(df[df['health_score'] > 0.7]),
                    'moderate': len(df[(df['health_score'] >= 0.3) & (df['health_score'] <= 0.7)]),
                    'poor': len(df[df['health_score'] < 0.3]),
                    'average_health_score': float(df['health_score'].mean())
                }
            
            # Chlorophyll assessment
            if 'chlorophyll_content' in df.columns:
                health_metrics['chlorophyll_assessment'] = {
                    'high_chlorophyll_percentage': len(df[df['chlorophyll_content'] > self.config.chlorophyll_threshold]) / len(df) * 100,
                    'average_chlorophyll': float(df['chlorophyll_content'].mean()),
                    'chlorophyll_uniformity': 1.0 / (1.0 + df['chlorophyll_content'].std())
                }
            
            # Stress indicators
            if 'stress_indicator' in df.columns:
                health_metrics['stress_assessment'] = {
                    'high_stress_cells': len(df[df['stress_indicator'] > 0.7]),
                    'average_stress_level': float(df['stress_indicator'].mean()),
                    'stress_distribution': df['stress_indicator'].describe().to_dict()
                }
            
            # Population vitality
            vitality_factors = []
            if 'health_score' in df.columns:
                vitality_factors.append(df['health_score'].mean())
            if 'chlorophyll_content' in df.columns:
                vitality_factors.append(df['chlorophyll_content'].mean())
            if 'cell_integrity' in df.columns:
                vitality_factors.append(df['cell_integrity'].mean())
            
            health_metrics['population_vitality'] = float(np.mean(vitality_factors)) if vitality_factors else 0.5
            
            return health_metrics
            
        except Exception as e:
            logger.error(f"❌ Health assessment error: {str(e)}")
            return {}
    
    def _calculate_clustering_coefficient(self, df: pd.DataFrame) -> float:
        """Calculate spatial clustering coefficient."""
        try:
            if len(df) < 3:
                return 0.0
            
            coords = df[['centroid_x', 'centroid_y']].values
            distances = squareform(pdist(coords))
            
            # Calculate average nearest neighbor distance
            avg_nn_distance = np.mean([np.sort(row)[1] for row in distances])  # Second closest (first is self)
            
            # Compare to random distribution
            area = (df['centroid_x'].max() - df['centroid_x'].min()) * (df['centroid_y'].max() - df['centroid_y'].min())
            expected_distance = 0.5 * np.sqrt(area / len(df))
            
            clustering_coefficient = expected_distance / (avg_nn_distance + 1e-8)
            
            return float(clustering_coefficient)
            
        except Exception as e:
            logger.error(f"❌ Clustering coefficient calculation error: {str(e)}")
            return 0.0
    
    def _calculate_quality_metrics(self, results_list: List[Dict]) -> Dict:
        """Calculate analysis quality metrics."""
        try:
            quality_scores = []
            processing_times = []
            detection_rates = []
            
            for result in results_list:
                if 'quality_score' in result:
                    quality_scores.append(result['quality_score'])
                
                if 'processing_time' in result:
                    processing_times.append(result['processing_time'])
                
                if 'total_cells' in result:
                    detection_rates.append(result['total_cells'])
            
            return {
                'average_image_quality': float(np.mean(quality_scores)) if quality_scores else 0.0,
                'quality_consistency': float(1.0 - np.std(quality_scores)) if len(quality_scores) > 1 else 1.0,
                'average_processing_time': float(np.mean(processing_times)) if processing_times else 0.0,
                'detection_consistency': float(1.0 / (1.0 + np.std(detection_rates))) if len(detection_rates) > 1 else 1.0
            }
            
        except Exception as e:
            logger.error(f"❌ Quality metrics calculation error: {str(e)}")
            return {}
    
    def _analyze_temporal_patterns(self, results_list: List[Dict]) -> Dict:
        """Analyze temporal patterns in the data."""
        try:
            if len(results_list) < 2:
                return {'message': 'Insufficient data for temporal analysis'}
            
            # Extract timestamps and cell counts
            temporal_data = []
            for result in results_list:
                if 'timestamp' in result and 'total_cells' in result:
                    try:
                        timestamp = datetime.fromisoformat(result['timestamp'].replace('_', 'T'))
                        temporal_data.append({
                            'timestamp': timestamp,
                            'cell_count': result['total_cells'],
                            'health_score': np.mean([cell.get('health_score', 0.5) for cell in result.get('cell_data', [])])
                        })
                    except:
                        continue
            
            if len(temporal_data) < 2:
                return {'message': 'Insufficient temporal data'}
            
            # Sort by timestamp
            temporal_data.sort(key=lambda x: x['timestamp'])
            
            # Calculate trends
            cell_counts = [d['cell_count'] for d in temporal_data]
            health_scores = [d['health_score'] for d in temporal_data]
            
            # Linear regression for trends
            x = np.arange(len(cell_counts))
            cell_trend_slope, _, _, _, _ = stats.linregress(x, cell_counts)
            health_trend_slope, _, _, _, _ = stats.linregress(x, health_scores)
            
            return {
                'observation_period_hours': (temporal_data[-1]['timestamp'] - temporal_data[0]['timestamp']).total_seconds() / 3600,
                'cell_count_trend': 'increasing' if cell_trend_slope > 0 else 'decreasing' if cell_trend_slope < 0 else 'stable',
                'health_trend': 'improving' if health_trend_slope > 0 else 'declining' if health_trend_slope < 0 else 'stable',
                'growth_rate_cells_per_hour': float(cell_trend_slope),
                'health_change_rate': float(health_trend_slope)
            }
            
        except Exception as e:
            logger.error(f"❌ Temporal analysis error: {str(e)}")
            return {}
    
    def _calculate_confidence_intervals(self, df: pd.DataFrame) -> Dict:
        """Calculate confidence intervals for key metrics."""
        try:
            confidence_level = self.config.confidence_interval
            alpha = 1 - confidence_level
            
            intervals = {}
            numeric_cols = ['area', 'health_score', 'chlorophyll_content']
            
            for col in numeric_cols:
                if col in df.columns:
                    data = df[col].dropna()
                    if len(data) > 1:
                        mean = np.mean(data)
                        sem = stats.sem(data)  # Standard error of mean
                        ci = stats.t.interval(confidence_level, len(data)-1, loc=mean, scale=sem)
                        
                        intervals[col] = {
                            'mean': float(mean),
                            'confidence_interval': [float(ci[0]), float(ci[1])],
                            'margin_of_error': float(ci[1] - mean)
                        }
            
            return intervals
            
        except Exception as e:
            logger.error(f"❌ Confidence interval calculation error: {str(e)}")
            return {}
    
    def _generate_recommendations(self, df: pd.DataFrame, health_assessment: Dict) -> List[str]:
        """Generate actionable recommendations based on analysis."""
        try:
            recommendations = []
            
            # Health-based recommendations
            if 'health_distribution' in health_assessment:
                health_dist = health_assessment['health_distribution']
                total_cells = sum(health_dist.values())
                
                if health_dist['poor'] / total_cells > 0.3:
                    recommendations.append("⚠️ High proportion of cells showing poor health indicators. Consider environmental stress factors.")
                
                if health_dist['excellent'] / total_cells > 0.7:
                    recommendations.append("✅ Population shows excellent health indicators. Current conditions are optimal.")
            
            # Chlorophyll-based recommendations
            if 'chlorophyll_assessment' in health_assessment:
                chlor_assess = health_assessment['chlorophyll_assessment']
                
                if chlor_assess['high_chlorophyll_percentage'] < 50:
                    recommendations.append("🌱 Low chlorophyll content detected. Consider light conditions and nutrient availability.")
                
                if chlor_assess['chlorophyll_uniformity'] < 0.5:
                    recommendations.append("🔄 High variability in chlorophyll content suggests heterogeneous conditions.")
            
            # Size distribution recommendations
            if 'area' in df.columns:
                size_cv = df['area'].std() / df['area'].mean()  # Coefficient of variation
                
                if size_cv > 0.8:
                    recommendations.append("📏 High size variability indicates mixed growth stages or stress conditions.")
                
                if df['area'].mean() < 200:
                    recommendations.append("📈 Small average cell size may indicate early growth stage or nutrient limitation.")
            
            # Outlier recommendations
            if 'is_outlier' in df.columns:
                outlier_rate = df['is_outlier'].sum() / len(df)
                
                if outlier_rate > 0.2:
                    recommendations.append("🔍 High outlier rate detected. Review image quality and processing parameters.")
            
            # Default recommendation if none generated
            if not recommendations:
                recommendations.append("📊 Population appears stable. Continue monitoring for trend analysis.")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"❌ Recommendation generation error: {str(e)}")
            return ["📊 Analysis complete. Monitor trends over time for better insights."]
    
    def _get_analysis_period(self, results_list: List[Dict]) -> Dict:
        """Get analysis time period information."""
        try:
            timestamps = []
            for result in results_list:
                if 'timestamp' in result:
                    try:
                        timestamp = datetime.fromisoformat(result['timestamp'].replace('_', 'T'))
                        timestamps.append(timestamp)
                    except:
                        continue
            
            if not timestamps:
                return {'message': 'No valid timestamps found'}
            
            timestamps.sort()
            
            return {
                'start_time': timestamps[0].isoformat(),
                'end_time': timestamps[-1].isoformat(),
                'duration_hours': (timestamps[-1] - timestamps[0]).total_seconds() / 3600,
                'total_timepoints': len(timestamps)
            }
            
        except Exception as e:
            logger.error(f"❌ Analysis period calculation error: {str(e)}")
            return {}
    
    def _empty_report(self) -> Dict:
        """Return empty report structure."""
        return {
            'summary': {
                'total_images_analyzed': 0,
                'total_cells_detected': 0,
                'success_rate': 0
            },
            'message': 'No data available for analysis'
        }


# Example usage and testing
if __name__ == "__main__":
    print("🚀 Testing Professional Wolffia Analysis System...")
    
    try:
        # Initialize configuration
        config = AnalysisConfig()
        
        # Initialize components
        image_processor = AdvancedImageProcessor(config)
        feature_extractor = BiologicalFeatureExtractor(config)
        statistical_analyzer = StatisticalAnalyzer(config)
        
        print("✅ All components initialized successfully")
        print("🔬 Professional Wolffia Analysis System ready for deployment")
        
    except Exception as e:
        print(f"❌ System initialization failed: {str(e)}")
        import traceback
        traceback.print_exc()