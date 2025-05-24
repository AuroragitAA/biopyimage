    # bioimaging.py
    
    import cv2
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy import ndimage
    from skimage import filters, morphology, measure, segmentation, feature
    from skimage.color import rgb2gray, rgb2hsv
    from skimage.exposure import equalize_adapthist
    from skimage.filters import gaussian, threshold_otsu, threshold_local
    from skimage.morphology import remove_small_objects, disk, opening, closing  # <-- without watershed
    from skimage.segmentation import watershed
    from skimage.segmentation import clear_border
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    import os
    import json
    from datetime import datetime
    import warnings
    warnings.filterwarnings('ignore')

    class WolffiaAnalyzer:
    """
    Advanced Wolffia bioimage analysis pipeline for automated cell counting,
    morphological analysis, chlorophyll content assessment, and biomass estimation.
    """

    def __init__(self, pixel_to_micron_ratio=1.0, chlorophyll_threshold=0.6):
    """
    Initialize Wolffia analyzer

    Parameters:
    pixel_to_micron_ratio: float, conversion factor from pixels to microns
    chlorophyll_threshold: float, threshold for high chlorophyll content (0-1)
    """
    self.pixel_to_micron = pixel_to_micron_ratio
    self.chlorophyll_threshold = chlorophyll_threshold
    self.results_history = []

       def preprocess_image(self, image_path, enhance_contrast=True, denoise=True):
        """
        Enhanced image preprocessing for Wolffia detection.
        Focuses on isolating green floating structures and suppressing background/dust.
        """
        try:
            # Load image
            if isinstance(image_path, str):
                image = cv2.imread(image_path)
                if image is None:
                    raise ValueError(f"Could not load image from {image_path}")
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image = image_path

            original = image.copy()

            # Convert to color spaces
            gray = rgb2gray(image)
            hsv = rgb2hsv(image)

            # Extract color channels (normalized)
            red_channel = image[:, :, 0] / 255.0
            green_channel = image[:, :, 1] / 255.0
            blue_channel = image[:, :, 2] / 255.0

            # Create green color mask using HSV
            lower_green_h = 35 / 360  # Hue range in [0, 1]
            upper_green_h = 85 / 360
            green_mask = (
                (hsv[:, :, 0] >= lower_green_h) & (hsv[:, :, 0] <= upper_green_h) &
                (hsv[:, :, 1] >= 0.3) &  # minimum saturation
                (hsv[:, :, 2] >= 0.2)    # minimum brightness
            )

            # Enhanced chlorophyll detection
            chlorophyll_enhanced = green_channel - 0.5 * (red_channel + blue_channel)
            chlorophyll_enhanced = np.clip(chlorophyll_enhanced, 0, 1)

            # Apply green mask to focus only on likely Wolffia regions
            chlorophyll_enhanced = chlorophyll_enhanced * green_mask
            green_channel = green_channel * green_mask
            gray = gray * green_mask

            # Adaptive contrast enhancement
            if enhance_contrast:
                gray = equalize_adapthist(gray, clip_limit=0.03)
                green_channel = equalize_adapthist(green_channel, clip_limit=0.03)
                chlorophyll_enhanced = equalize_adapthist(chlorophyll_enhanced, clip_limit=0.03)

            # Denoising
            if denoise:
                gray = gaussian(gray, sigma=0.5)
                green_channel = gaussian(green_channel, sigma=0.5)
                chlorophyll_enhanced = gaussian(chlorophyll_enhanced, sigma=0.5)

            return original, gray, green_channel, chlorophyll_enhanced, hsv

        except Exception as e:
            print(f"Error in preprocessing: {str(e)}")
            return None, None, None, None, None

    def advanced_segmentation(self, gray_image, green_channel, chlorophyll_enhanced, 
        min_cell_area=30, max_cell_area=8000):
    """
    Advanced multi-modal segmentation for better cell detection
    """
    try:
    # Multi-threshold approach
    thresh_gray = threshold_otsu(gray_image)
    thresh_green = threshold_otsu(green_channel)
    thresh_chl = threshold_otsu(chlorophyll_enhanced)

    # Adaptive thresholding for local variations
    adaptive_thresh = threshold_local(gray_image, block_size=35, offset=0.01)

    # Create multiple binary masks
    binary_gray = gray_image > thresh_gray
    binary_green = green_channel > thresh_green
    binary_chl = chlorophyll_enhanced > thresh_chl
    binary_adaptive = gray_image > adaptive_thresh

    # Combine segmentations with weighted voting
    combined_binary = (binary_gray.astype(int) + 
            binary_green.astype(int) + 
            binary_chl.astype(int) * 2 +  # Weight chlorophyll more
            binary_adaptive.astype(int)) >= 2

    # Clean up binary image
    combined_binary = remove_small_objects(combined_binary, min_size=min_cell_area)
    combined_binary = clear_border(combined_binary)

    # Advanced morphological operations
    kernel_small = disk(1)
    kernel_medium = disk(3)

    # Remove noise and smooth boundaries
    combined_binary = opening(combined_binary, kernel_small)
    combined_binary = closing(combined_binary, kernel_medium)

    # Watershed segmentation for separating touching cells
    distance = ndimage.distance_transform_edt(combined_binary)

    # Find local maxima as seeds
    local_maxima = feature.peak_local_max(
    distance, 
    min_distance=8, 
    threshold_abs=3,
    exclude_border=True
    )

    # Create markers
    markers = np.zeros(distance.shape, dtype=bool)
    if local_maxima.size > 0:
    for r, c in local_maxima:
    markers[r, c] = True
    markers, _ = ndimage.label(markers)


    # Apply watershed
    labels = watershed(-distance, markers, mask=combined_binary)

    # Filter by size and shape
    props = measure.regionprops(labels)
    filtered_labels = np.zeros_like(labels)
    valid_label = 1

    for prop in props:
    # Size filtering
    if min_cell_area <= prop.area <= max_cell_area:
    # Shape filtering (remove very elongated objects)
    if prop.eccentricity < 0.95:  # Remove very elongated objects
    mask = labels == prop.label
    filtered_labels[mask] = valid_label
    valid_label += 1

    return filtered_labels

    except Exception as e:
    print(f"Error in segmentation: {str(e)}")
    return np.zeros_like(gray_image)

    def extract_comprehensive_features(self, labels, original_image, green_channel, chlorophyll_enhanced):
    """
    Extract comprehensive morphological and biochemical features
    """
    try:
    props = measure.regionprops(labels, intensity_image=green_channel)
    chl_props = measure.regionprops(labels, intensity_image=chlorophyll_enhanced)

    cell_data = []

    for i, (prop, chl_prop) in enumerate(zip(props, chl_props)):
    # Basic morphological features
    area_pixels = prop.area
    area_microns = area_pixels * (self.pixel_to_micron ** 2)
    perimeter = prop.perimeter * self.pixel_to_micron
    equivalent_diameter = prop.equivalent_diameter * self.pixel_to_micron
    major_axis = prop.major_axis_length * self.pixel_to_micron
    minor_axis = prop.minor_axis_length * self.pixel_to_micron

    # Shape descriptors
    aspect_ratio = major_axis / minor_axis if minor_axis > 0 else 0
    circularity = (4 * np.pi * area_pixels) / (perimeter ** 2) if perimeter > 0 else 0
    roundness = 4 * area_pixels / (np.pi * major_axis ** 2) if major_axis > 0 else 0

    # Intensity features from green channel
    mean_green_intensity = prop.mean_intensity
    max_green_intensity = prop.max_intensity
    min_green_intensity = prop.min_intensity

    # Enhanced chlorophyll features
    mean_chl_intensity = chl_prop.mean_intensity
    max_chl_intensity = chl_prop.max_intensity

    # Calculate intensity statistics
    cell_mask = labels == prop.label
    green_pixels = green_channel[cell_mask]
    chl_pixels = chlorophyll_enhanced[cell_mask]

    green_std = np.std(green_pixels)
    chl_std = np.std(chl_pixels)

    # Extract RGB values for comprehensive color analysis
    cell_rgb_pixels = original_image[cell_mask]
    mean_rgb = np.mean(cell_rgb_pixels, axis=0)
    std_rgb = np.std(cell_rgb_pixels, axis=0)

    # Advanced chlorophyll metrics
    chlorophyll_content = mean_chl_intensity
    chlorophyll_density = chlorophyll_content / area_microns if area_microns > 0 else 0
    is_high_chlorophyll = chlorophyll_content > self.chlorophyll_threshold

    # Biomass estimation using multiple factors
    # More sophisticated model considering area, chlorophyll, and cell density
    biomass_estimate = (area_microns * 
                (0.7 * chlorophyll_content + 0.3 * mean_green_intensity) * 
                (1 + 0.1 * circularity))  # Shape factor

    # Cell health indicators
    color_variation = np.mean(std_rgb) / np.mean(mean_rgb) if np.mean(mean_rgb) > 0 else 0

    cell_data.append({
    'cell_id': i + 1,
    'centroid_x': prop.centroid[1] * self.pixel_to_micron,
    'centroid_y': prop.centroid[0] * self.pixel_to_micron,
    'area_pixels': area_pixels,
    'area_microns_sq': area_microns,
    'perimeter_microns': perimeter,
    'equivalent_diameter_microns': equivalent_diameter,
    'major_axis_microns': major_axis,
    'minor_axis_microns': minor_axis,
    'aspect_ratio': aspect_ratio,
    'eccentricity': prop.eccentricity,
    'solidity': prop.solidity,
    'circularity': circularity,
    'roundness': roundness,
    'mean_green_intensity': mean_green_intensity,
    'max_green_intensity': max_green_intensity,
    'min_green_intensity': min_green_intensity,
    'green_intensity_std': green_std,
    'mean_chlorophyll_intensity': mean_chl_intensity,
    'max_chlorophyll_intensity': max_chl_intensity,
    'chlorophyll_intensity_std': chl_std,
    'chlorophyll_density': chlorophyll_density,
    'mean_red': mean_rgb[0] / 255.0,
    'mean_green': mean_rgb[1] / 255.0,
    'mean_blue': mean_rgb[2] / 255.0,
    'std_red': std_rgb[0] / 255.0,
    'std_green': std_rgb[1] / 255.0,
    'std_blue': std_rgb[2] / 255.0,
    'color_variation': color_variation,
    'high_chlorophyll': is_high_chlorophyll,
    'biomass_estimate': biomass_estimate
    })

    return pd.DataFrame(cell_data)

    except Exception as e:
    print(f"Error in feature extraction: {str(e)}")
    return pd.DataFrame()

    def advanced_cell_classification(self, df):
    """
    Advanced cell classification using multiple features and clustering
    """
    if len(df) == 0:
    return df

    try:
    # Multi-dimensional classification
    df['cell_type'] = 'unknown'
    df['health_status'] = 'unknown'
    df['size_category'] = 'unknown'
    df['chlorophyll_category'] = 'unknown'

    # Size-based classification (tertiles)
    size_33 = df['area_microns_sq'].quantile(0.33)
    size_67 = df['area_microns_sq'].quantile(0.67)

    df.loc[df['area_microns_sq'] <= size_33, 'size_category'] = 'small'
    df.loc[(df['area_microns_sq'] > size_33) & (df['area_microns_sq'] <= size_67), 'size_category'] = 'medium'
    df.loc[df['area_microns_sq'] > size_67, 'size_category'] = 'large'

    # Chlorophyll-based classification
    chl_33 = df['mean_chlorophyll_intensity'].quantile(0.33)
    chl_67 = df['mean_chlorophyll_intensity'].quantile(0.67)

    df.loc[df['mean_chlorophyll_intensity'] <= chl_33, 'chlorophyll_category'] = 'low'
    df.loc[(df['mean_chlorophyll_intensity'] > chl_33) & (df['mean_chlorophyll_intensity'] <= chl_67), 'chlorophyll_category'] = 'medium'
    df.loc[df['mean_chlorophyll_intensity'] > chl_67, 'chlorophyll_category'] = 'high'

    # Combined classification
    df['cell_type'] = df['size_category'] + '_' + df['chlorophyll_category']

    # Health status based on multiple indicators
    # Healthy cells: high circularity, medium-high chlorophyll, low color variation
    health_score = (df['circularity'] * 0.3 + 
        df['mean_chlorophyll_intensity'] * 0.4 + 
        (1 - df['color_variation']) * 0.3)

    health_threshold_low = health_score.quantile(0.33)
    health_threshold_high = health_score.quantile(0.67)

    df.loc[health_score <= health_threshold_low, 'health_status'] = 'stressed'
    df.loc[(health_score > health_threshold_low) & (health_score <= health_threshold_high), 'health_status'] = 'moderate'
    df.loc[health_score > health_threshold_high, 'health_status'] = 'healthy'

    # Advanced clustering for cell type discovery
    if len(df) >= 10:  # Only if we have enough cells
    features_for_clustering = ['area_microns_sq', 'mean_chlorophyll_intensity', 
                        'circularity', 'aspect_ratio', 'chlorophyll_density']

    # Prepare data for clustering
    X = df[features_for_clustering].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # K-means clustering
    n_clusters = min(5, len(df) // 3)  # Adaptive number of clusters
    if n_clusters >= 2:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['cluster_id'] = kmeans.fit_predict(X_scaled)
    else:
    df['cluster_id'] = 0
    else:
    df['cluster_id'] = 0

    return df

    except Exception as e:
    print(f"Error in classification: {str(e)}")
    return df

    def analyze_single_image(self, image_path, timestamp=None, save_visualization=True):
    """
    Complete analysis pipeline for a single image with error handling
    """
    if timestamp is None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"Analyzing image: {image_path}")

    try:
    # Preprocess
    result = self.preprocess_image(image_path)
    if result[0] is None:
    print("Failed to preprocess image")
    return None

    original, gray, green_channel, chlorophyll_enhanced, hsv = result

    # Segment cells
    labels = self.advanced_segmentation(gray, green_channel, chlorophyll_enhanced)

    if np.max(labels) == 0:
    print("No cells detected after segmentation!")
    return None

    # Extract features
    df = self.extract_comprehensive_features(labels, original, green_channel, chlorophyll_enhanced)

    if len(df) == 0:
    print("No features extracted!")
    return None

    # Classify cell types
    df = self.advanced_cell_classification(df)

    # Add metadata
    df['timestamp'] = timestamp
    df['image_path'] = str(image_path)

    # Calculate summary statistics
    summary = self.calculate_comprehensive_stats(df)

    # Visualization
    if save_visualization:
    self.create_comprehensive_visualization(original, labels, df, 
                                    output_path=f"analysis_{timestamp}.png")

    # Store results
    result = {
    'timestamp': timestamp,
    'image_path': str(image_path),
    'cell_data': df,
    'summary': summary,
    'total_cells': len(df)
    }

    self.results_history.append(result)
    print(f"Analysis complete: {len(df)} cells detected")

    return result

    except Exception as e:
    print(f"Error in analysis: {str(e)}")
    return None

    def calculate_comprehensive_stats(self, df):
    """
    Calculate comprehensive summary statistics
    """
    if len(df) == 0:
    return {}

    try:
    summary = {
    # Basic counts
    'total_cell_count': len(df),
    'healthy_cell_count': len(df[df['health_status'] == 'healthy']),
    'stressed_cell_count': len(df[df['health_status'] == 'stressed']),

    # Size statistics
    'mean_cell_area_microns': float(df['area_microns_sq'].mean()),
    'std_cell_area_microns': float(df['area_microns_sq'].std()),
    'median_cell_area_microns': float(df['area_microns_sq'].median()),
    'min_cell_area_microns': float(df['area_microns_sq'].min()),
    'max_cell_area_microns': float(df['area_microns_sq'].max()),

    # Chlorophyll statistics
    'mean_chlorophyll_intensity': float(df['mean_chlorophyll_intensity'].mean()),
    'std_chlorophyll_intensity': float(df['mean_chlorophyll_intensity'].std()),
    'median_chlorophyll_intensity': float(df['mean_chlorophyll_intensity'].median()),
    'mean_chlorophyll_density': float(df['chlorophyll_density'].mean()),

    # Biomass statistics
    'total_biomass_estimate': float(df['biomass_estimate'].sum()),
    'mean_biomass_per_cell': float(df['biomass_estimate'].mean()),
    'biomass_density': float(df['biomass_estimate'].sum() / df['area_microns_sq'].sum()),

    # Shape statistics
    'mean_aspect_ratio': float(df['aspect_ratio'].mean()),
    'mean_circularity': float(df['circularity'].mean()),
    'mean_roundness': float(df['roundness'].mean()),

    # Classification distributions
    'cell_type_distribution': df['cell_type'].value_counts().to_dict(),
    'health_status_distribution': df['health_status'].value_counts().to_dict(),
    'size_category_distribution': df['size_category'].value_counts().to_dict(),
    'chlorophyll_category_distribution': df['chlorophyll_category'].value_counts().to_dict(),

    # Quality metrics
    'high_chlorophyll_percentage': float((df['high_chlorophyll'].sum() / len(df)) * 100),
    'healthy_cell_percentage': float((len(df[df['health_status'] == 'healthy']) / len(df)) * 100),
    'mean_color_variation': float(df['color_variation'].mean())
    }

    return summary

    except Exception as e:
    print(f"Error calculating statistics: {str(e)}")
    return {}

    def analyze_time_series(self, image_paths, timestamps=None):
    """
    Analyze multiple images for comprehensive time-series analysis
    """
    if timestamps is None:
    timestamps = [f"t_{i:03d}" for i in range(len(image_paths))]

    results = []

    print(f"Starting time series analysis of {len(image_paths)} images...")

    for i, (path, timestamp) in enumerate(zip(image_paths, timestamps)):
    print(f"Processing image {i+1}/{len(image_paths)}: {timestamp}")
    result = self.analyze_single_image(path, timestamp, save_visualization=(i==0))
    if result:
    results.append(result)
    else:
    print(f"Failed to analyze image {i+1}")

    # Calculate temporal changes
    if len(results) > 1:
    self.calculate_temporal_changes(results)
    self.create_time_series_plots(results)

    print(f"Time series analysis complete: {len(results)} images processed successfully")
    return results

    def calculate_temporal_changes(self, results):
    """
    Calculate comprehensive temporal changes and growth rates
    """
    try:
    timestamps = [r['timestamp'] for r in results]
    cell_counts = [r['total_cells'] for r in results]
    biomass_totals = [r['summary']['total_biomass_estimate'] for r in results]
    mean_areas = [r['summary']['mean_cell_area_microns'] for r in results]
    mean_chlorophyll = [r['summary']['mean_chlorophyll_intensity'] for r in results]

    # Calculate rates of change
    if len(results) > 1:
    cell_count_changes = np.diff(cell_counts)
    biomass_changes = np.diff(biomass_totals)
    area_changes = np.diff(mean_areas)
    chlorophyll_changes = np.diff(mean_chlorophyll)

    # Calculate relative changes (percentages)
    cell_count_rel_changes = (cell_count_changes / np.array(cell_counts[:-1])) * 100
    biomass_rel_changes = (biomass_changes / np.array(biomass_totals[:-1])) * 100

    print("\nTemporal Analysis Results:")
    print(f"Cell count changes: {cell_count_changes}")
    print(f"Biomass changes: {biomass_changes}")
    print(f"Relative cell count changes (%): {cell_count_rel_changes}")
    print(f"Relative biomass changes (%): {biomass_rel_changes}")

    # Add to results
    for i, result in enumerate(results[1:], 1):
    result['cell_count_change'] = float(cell_count_changes[i-1])
    result['biomass_change'] = float(biomass_changes[i-1])
    result['cell_count_rel_change'] = float(cell_count_rel_changes[i-1])
    result['biomass_rel_change'] = float(biomass_rel_changes[i-1])
    result['mean_area_change'] = float(area_changes[i-1])
    result['mean_chlorophyll_change'] = float(chlorophyll_changes[i-1])

    except Exception as e:
    print(f"Error in temporal analysis: {str(e)}")

    def create_comprehensive_visualization(self, original_image, labels, df, output_path=None):
    """
    Create comprehensive visualization with multiple panels
    """
    try:
    fig, axes = plt.subplots(3, 3, figsize=(20, 16))

    # Original image
    axes[0,0].imshow(original_image)
    axes[0,0].set_title('Original Image', fontsize=12)
    axes[0,0].axis('off')

    # Segmentation overlay
    axes[0,1].imshow(original_image)
    axes[0,1].imshow(labels, alpha=0.4, cmap='tab20')
    axes[0,1].set_title(f'Segmentation ({len(df)} cells)', fontsize=12)
    axes[0,1].axis('off')

    # Cell outlines
    axes[0,2].imshow(original_image)
    for _, cell in df.iterrows():
    y, x = cell['centroid_y'] / self.pixel_to_micron, cell['centroid_x'] / self.pixel_to_micron
    axes[0,2].plot(x, y, 'ro', markersize=3)
    axes[0,2].text(x, y, str(int(cell['cell_id'])), fontsize=8, color='white')
    axes[0,2].set_title('Cell Centers & IDs', fontsize=12)
    axes[0,2].axis('off')

    if len(df) > 0:
    # Size distribution
    axes[1,0].hist(df['area_microns_sq'], bins=20, alpha=0.7, edgecolor='black', color='skyblue')
    axes[1,0].set_xlabel('Cell Area (μm²)')
    axes[1,0].set_ylabel('Frequency')
    axes[1,0].set_title('Cell Size Distribution')
    axes[1,0].grid(True, alpha=0.3)

    # Chlorophyll distribution
    axes[1,1].hist(df['mean_chlorophyll_intensity'], bins=20, alpha=0.7, 
            color='green', edgecolor='black')
    axes[1,1].set_xlabel('Chlorophyll Intensity')
    axes[1,1].set_ylabel('Frequency')
    axes[1,1].set_title('Chlorophyll Distribution')
    axes[1,1].grid(True, alpha=0.3)

    # Size vs Chlorophyll scatter
    scatter = axes[1,2].scatter(df['area_microns_sq'], df['mean_chlorophyll_intensity'], 
                        c=df['biomass_estimate'], cmap='viridis', alpha=0.7, s=30)
    axes[1,2].set_xlabel('Cell Area (μm²)')
    axes[1,2].set_ylabel('Chlorophyll Intensity')
    axes[1,2].set_title('Size vs Chlorophyll')
    plt.colorbar(scatter, ax=axes[1,2], label='Biomass Estimate')

    # Cell type distribution
    if 'cell_type' in df.columns:
    type_counts = df['cell_type'].value_counts()
    axes[2,0].pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%')
    axes[2,0].set_title('Cell Type Distribution')

    # Health status distribution
    if 'health_status' in df.columns:
    health_counts = df['health_status'].value_counts()
    colors = ['red', 'orange', 'green'][:len(health_counts)]
    axes[2,1].pie(health_counts.values, labels=health_counts.index, 
                autopct='%1.1f%%', colors=colors)
    axes[2,1].set_title('Health Status Distribution')

    # Biomass distribution
    axes[2,2].hist(df['biomass_estimate'], bins=20, alpha=0.7, 
            color='purple', edgecolor='black')
    axes[2,2].set_xlabel('Biomass Estimate')
    axes[2,2].set_ylabel('Frequency')
    axes[2,2].set_title('Biomass Distribution')
    axes[2,2].grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Comprehensive visualization saved to: {output_path}")

    plt.show()

    except Exception as e:
    print(f"Error creating visualization: {str(e)}")

    def create_time_series_plots(self, results):
    """
    Create time series plots for temporal analysis
    """
    try:
    if len(results) < 2:
    return

    timestamps = [r['timestamp'] for r in results]
    cell_counts = [r['total_cells'] for r in results]
    biomass_totals = [r['summary']['total_biomass_estimate'] for r in results]
    mean_areas = [r['summary']['mean_cell_area_microns'] for r in results]
    mean_chlorophyll = [r['summary']['mean_chlorophyll_intensity'] for r in results]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Cell count over time
    axes[0,0].plot(range(len(timestamps)), cell_counts, 'bo-', linewidth=2, markersize=8)
    axes[0,0].set_xlabel('Time Point')
    axes[0,0].set_ylabel('Cell Count')
    axes[0,0].set_title('Cell Count Over Time')
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].set_xticks(range(len(timestamps)))
    axes[0,0].set_xticklabels(timestamps, rotation=45)

    # Biomass over time
    axes[0,1].plot(range(len(timestamps)), biomass_totals, 'go-', linewidth=2, markersize=8)
    axes[0,1].set_xlabel('Time Point')
    axes[0,1].set_ylabel('Total Biomass')
    axes[0,1].set_title('Biomass Over Time')
    axes[0,1].grid(True, alpha=0.3)
    axes[0,1].set_xticks(range(len(timestamps)))
    axes[0,1].set_xticklabels(timestamps, rotation=45)

    # Mean cell area over time
    axes[1,0].plot(range(len(timestamps)), mean_areas, 'ro-', linewidth=2, markersize=8)
    axes[1,0].set_xlabel('Time Point')
    axes[1,0].set_ylabel('Mean Cell Area (μm²)')
    axes[1,0].set_title('Mean Cell Size Over Time')
    axes[1,0].grid(True, alpha=0.3)
    axes[1,0].set_xticks(range(len(timestamps)))
    axes[1,0].set_xticklabels(timestamps, rotation=45)

    # Mean chlorophyll over time
    axes[1,1].plot(range(len(timestamps)), mean_chlorophyll, 'mo-', linewidth=2, markersize=8)
    axes[1,1].set_xlabel('Time Point')
    axes[1,1].set_ylabel('Mean Chlorophyll Intensity')
    axes[1,1].set_title('Chlorophyll Content Over Time')
    axes[1,1].grid(True, alpha=0.3)
    axes[1,1].set_xticks(range(len(timestamps)))
    axes[1,1].set_xticklabels(timestamps, rotation=45)

    plt.tight_layout()
    plt.savefig(f"time_series_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png", 
    dpi=300, bbox_inches='tight')
    plt.show()

    except Exception as e:
    print(f"Error creating time series plots: {str(e)}")

    def export_comprehensive_results(self, output_dir="wolffia_analysis_results"):
    """
    Export comprehensive analysis results to multiple formats
    """
    try:
    if not os.path.exists(output_dir):
    os.makedirs(output_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Export individual results
    for i, result in enumerate(self.results_history):
    result_timestamp = result['timestamp']

    # Export detailed cell data
    result['cell_data'].to_csv(
    f"{output_dir}/detailed_cells_{result_timestamp}.csv", index=False
    )

    # Export summary statistics
    with open(f"{output_dir}/summary_{result_timestamp}.json", 'w') as f:
    # Convert numpy types to Python types for JSON serialization
    summary_clean = {}
    for key, value in result['summary'].items():
    if isinstance(value, (np.integer, np.floating)):
        summary_clean[key] = float(value)
    elif isinstance(value, dict):
        summary_clean[key] = {k: float(v) if isinstance(v, (np.integer, np.floating)) else v 
                            for k, v in value.items()}
    else:
        summary_clean[key] = value
    json.dump(summary_clean, f, indent=2)

    # Export combined time series data
    if len(self.results_history) > 1:
    # Combine all cell data
    all_cells = pd.concat([r['cell_data'] for r in self.results_history], 
                ignore_index=True)
    all_cells.to_csv(f"{output_dir}/all_cells_timeseries_{timestamp}.csv", index=False)

    # Time series summary
    time_summary = []
    for result in self.results_history:
    summary = result['summary'].copy()
    summary['timestamp'] = result['timestamp']
    summary['image_path'] = result['image_path']
    summary['total_cells'] = result['total_cells']

    # Add change metrics if available
    if 'cell_count_change' in result:
    summary['cell_count_change'] = result['cell_count_change']
    summary['biomass_change'] = result['biomass_change']
    summary['cell_count_rel_change'] = result['cell_count_rel_change']
    summary['biomass_rel_change'] = result['biomass_rel_change']

    time_summary.append(summary)

    time_df = pd.DataFrame(time_summary)
    time_df.to_csv(f"{output_dir}/time_series_summary_{timestamp}.csv", index=False)

    # Export growth analysis
    self.export_growth_analysis(output_dir, timestamp)

    # Create comprehensive report
    self.create_analysis_report(output_dir, timestamp)

    print(f"Comprehensive results exported to: {output_dir}")

    except Exception as e:
    print(f"Error exporting results: {str(e)}")

    def export_growth_analysis(self, output_dir, timestamp):
    """
    Export detailed growth analysis
    """
    try:
    if len(self.results_history) < 2:
    return

    growth_data = []
    for i, result in enumerate(self.results_history):
    growth_entry = {
    'timepoint': i,
    'timestamp': result['timestamp'],
    'total_cells': result['total_cells'],
    'total_biomass': result['summary']['total_biomass_estimate'],
    'mean_cell_area': result['summary']['mean_cell_area_microns'],
    'mean_chlorophyll': result['summary']['mean_chlorophyll_intensity'],
    'healthy_cells': result['summary']['healthy_cell_count'],
    'stressed_cells': result['summary']['stressed_cell_count'],
    'healthy_percentage': result['summary']['healthy_cell_percentage']
    }

    if 'cell_count_change' in result:
    growth_entry.update({
    'cell_count_change': result['cell_count_change'],
    'biomass_change': result['biomass_change'],
    'cell_count_rel_change': result['cell_count_rel_change'],
    'biomass_rel_change': result['biomass_rel_change']
    })

    growth_data.append(growth_entry)

    growth_df = pd.DataFrame(growth_data)
    growth_df.to_csv(f"{output_dir}/growth_analysis_{timestamp}.csv", index=False)

    except Exception as e:
    print(f"Error exporting growth analysis: {str(e)}")

    def create_analysis_report(self, output_dir, timestamp):
    """
    Create a comprehensive analysis report
    """
    try:
    report_path = f"{output_dir}/analysis_report_{timestamp}.txt"

    with open(report_path, 'w') as f:
    f.write("WOLFFIA BIOIMAGE ANALYSIS REPORT\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Total Images Analyzed: {len(self.results_history)}\n\n")

    # Overall statistics
    if self.results_history:
    f.write("OVERALL STATISTICS\n")
    f.write("-" * 20 + "\n")

    total_cells = sum(r['total_cells'] for r in self.results_history)
    total_biomass = sum(r['summary']['total_biomass_estimate'] for r in self.results_history)

    f.write(f"Total Cells Detected: {total_cells}\n")
    f.write(f"Total Biomass Estimate: {total_biomass:.2f}\n")
    f.write(f"Average Cells per Image: {total_cells / len(self.results_history):.1f}\n\n")

    # Individual image results
    f.write("INDIVIDUAL IMAGE RESULTS\n")
    f.write("-" * 25 + "\n")

    for i, result in enumerate(self.results_history):
    f.write(f"\nImage {i+1}: {result['timestamp']}\n")
    f.write(f"  Path: {result['image_path']}\n")
    f.write(f"  Cells Detected: {result['total_cells']}\n")
    f.write(f"  Mean Cell Area: {result['summary']['mean_cell_area_microns']:.2f} μm²\n")
    f.write(f"  Mean Chlorophyll: {result['summary']['mean_chlorophyll_intensity']:.3f}\n")
    f.write(f"  Total Biomass: {result['summary']['total_biomass_estimate']:.2f}\n")
    f.write(f"  Healthy Cells: {result['summary']['healthy_cell_percentage']:.1f}%\n")

    if 'cell_count_change' in result:
    f.write(f"  Cell Count Change: {result['cell_count_change']:+.0f} ({result['cell_count_rel_change']:+.1f}%)\n")
    f.write(f"  Biomass Change: {result['biomass_change']:+.2f} ({result['biomass_rel_change']:+.1f}%)\n")

    # Time series analysis
    if len(self.results_history) > 1:
    f.write(f"\n\nTIME SERIES ANALYSIS\n")
    f.write("-" * 20 + "\n")

    cell_counts = [r['total_cells'] for r in self.results_history]
    biomass_totals = [r['summary']['total_biomass_estimate'] for r in self.results_history]

    total_cell_change = cell_counts[-1] - cell_counts[0]
    total_biomass_change = biomass_totals[-1] - biomass_totals[0]

    f.write(f"Total Cell Count Change: {total_cell_change:+.0f}\n")
    f.write(f"Total Biomass Change: {total_biomass_change:+.2f}\n")
    f.write(f"Average Growth Rate (cells/timepoint): {np.mean(np.diff(cell_counts)):.1f}\n")
    f.write(f"Average Biomass Rate (biomass/timepoint): {np.mean(np.diff(biomass_totals)):.2f}\n")

    f.write(f"\n\nANALYSIS PARAMETERS\n")
    f.write("-" * 20 + "\n")
    f.write(f"Pixel to Micron Ratio: {self.pixel_to_micron}\n")
    f.write(f"Chlorophyll Threshold: {self.chlorophyll_threshold}\n")

    f.write(f"\n\nFILES GENERATED\n")
    f.write("-" * 15 + "\n")
    f.write("- detailed_cells_[timestamp].csv: Individual cell measurements\n")
    f.write("- summary_[timestamp].json: Summary statistics per image\n")
    f.write("- all_cells_timeseries_[timestamp].csv: Combined cell data\n")
    f.write("- time_series_summary_[timestamp].csv: Time series statistics\n")
    f.write("- growth_analysis_[timestamp].csv: Growth metrics\n")
    f.write("- analysis_[timestamp].png: Visualization plots\n")

    print(f"Analysis report saved to: {report_path}")

    except Exception as e:
    print(f"Error creating analysis report: {str(e)}")


    # Advanced usage functions
    def batch_analyze_directory(directory_path, analyzer=None, pattern="*.jpg"):
    """
    Batch analyze all images in a directory
    """
    if analyzer is None:
    analyzer = WolffiaAnalyzer(pixel_to_micron_ratio=0.5)

    import glob
    image_paths = glob.glob(os.path.join(directory_path, pattern))
    image_paths.extend(glob.glob(os.path.join(directory_path, "*.png")))
    image_paths.extend(glob.glob(os.path.join(directory_path, "*.tif")))
    image_paths.extend(glob.glob(os.path.join(directory_path, "*.tiff")))
    image_paths.extend(glob.glob(os.path.join(directory_path, "*.bmp")))
    image_paths.extend(glob.glob(os.path.join(directory_path, "*.jfif")))

    image_paths = sorted(list(set(image_paths)))  # Remove duplicates and sort

    if not image_paths:
    print(f"No images found in {directory_path}")
    return None

    print(f"Found {len(image_paths)} images for batch analysis")

    # Generate timestamps based on file names or use sequential
    timestamps = [f"batch_{i:03d}" for i in range(len(image_paths))]

    results = analyzer.analyze_time_series(image_paths, timestamps)

    # Export results
    output_dir = os.path.join(directory_path, "wolffia_analysis_results")
    analyzer.export_comprehensive_results(output_dir)

    return results

    def calibrate_pixel_to_micron(image_path, known_distance_pixels, known_distance_microns):
    """
    Helper function to calibrate pixel to micron conversion
    """
    ratio = known_distance_microns / known_distance_pixels
    print(f"Calibrated pixel to micron ratio: {ratio:.4f}")
    return ratio

    # Example usage and demonstration
    def demo_comprehensive_analysis():
    """
    Comprehensive demonstration of the Wolffia analysis pipeline
    """
    print("""
    🔬 WOLFFIA BIOIMAGE ANALYSIS PIPELINE - FINAL VERSION
    ===================================================

    This advanced pipeline provides:

    ✅ AUTOMATED CELL DETECTION & COUNTING
    - Multi-modal segmentation (grayscale + chlorophyll enhanced)
    - Watershed algorithm for separating touching cells
    - Advanced morphological filtering

    ✅ COMPREHENSIVE MORPHOLOGICAL ANALYSIS
    - Area, perimeter, diameter measurements
    - Shape descriptors (circularity, aspect ratio, roundness)
    - Size distribution analysis

    ✅ CHLOROPHYLL CONTENT ANALYSIS
    - Enhanced chlorophyll detection algorithm
    - Chlorophyll density calculations
    - Health status assessment

    ✅ ADVANCED CELL CLASSIFICATION
    - Multi-dimensional classification (size × chlorophyll)
    - Health status (healthy/moderate/stressed)
    - Machine learning clustering for type discovery

    ✅ BIOMASS ESTIMATION
    - Sophisticated biomass model
    - Total biomass tracking
    - Growth rate calculations

    ✅ TIME SERIES ANALYSIS
    - Automated temporal change detection
    - Growth rate calculations
    - Comprehensive visualizations

    ✅ COMPREHENSIVE REPORTING
    - Detailed CSV exports
    - JSON summaries
    - Automated report generation
    - Multiple visualization formats
    """)

    # Example usage code
    example_code = '''
    # BASIC USAGE
    analyzer = WolffiaAnalyzer(pixel_to_micron_ratio=0.5, chlorophyll_threshold=0.6)

    # Single image analysis
    result = analyzer.analyze_single_image("wolffia_image.jpg")

    # Time series analysis
    image_paths = ["t0.jpg", "t1.jpg", "t2.jpg", "t3.jpg"]
    timestamps = ["0min", "30min", "60min", "90min"]
    results = analyzer.analyze_time_series(image_paths, timestamps)

    # Export comprehensive results
    analyzer.export_comprehensive_results("output_directory")

    # BATCH ANALYSIS
    results = batch_analyze_directory("path/to/images/", analyzer)

    # CALIBRATION
    ratio = calibrate_pixel_to_micron("calibration_image.jpg", 100, 50.0)
    analyzer = WolffiaAnalyzer(pixel_to_micron_ratio=ratio)
    '''

    print("\n📋 EXAMPLE CODE:")
    print(example_code)

    # Real example with your paths
    real_example = '''
    # YOUR SPECIFIC EXAMPLE
    analyzer = WolffiaAnalyzer(pixel_to_micron_ratio=0.5, chlorophyll_threshold=0.6)

    image_paths = [
    r"C:\\Users\\Aun\\Desktop\\Projects\\bioimaging\\1-2205.jfif",
    r"C:\\Users\\Aun\\Desktop\\Projects\\bioimaging\\1-2305.jfif", 
    r"C:\\Users\\Aun\\Desktop\\Projects\\bioimaging\\1-2305 - Copy.jfif",
    r"C:\\Users\\Aun\\Desktop\\Projects\\bioimaging\\2-2205.jfif"
    ]

    timestamps = ["Day1_Morning", "Day2_Morning", "Day2_Evening", "Day3_Morning"]
    results = analyzer.analyze_time_series(image_paths, timestamps)

    # Export everything
    analyzer.export_comprehensive_results("C:/Users/Aun/Desktop/Projects/bioimaging/wolffia_results")

    # Or batch analyze entire directory
    results = batch_analyze_directory("C:/Users/Aun/Desktop/Projects/bioimaging/", analyzer)
    '''

    print("\n🚀 YOUR SPECIFIC USAGE:")
    print(real_example)

    print("""
    📊 OUTPUT FILES GENERATED:
    - detailed_cells_[timestamp].csv: Individual cell measurements
    - summary_[timestamp].json: Summary statistics per image  
    - all_cells_timeseries_[timestamp].csv: Combined time series data
    - time_series_summary_[timestamp].csv: Temporal statistics
    - growth_analysis_[timestamp].csv: Growth metrics
    - analysis_[timestamp].png: Comprehensive visualizations
    - time_series_analysis_[timestamp].png: Temporal plots
    - analysis_report_[timestamp].txt: Detailed text report

    🔧 CUSTOMIZABLE PARAMETERS:
    - pixel_to_micron_ratio: Calibrate for accurate measurements
    - chlorophyll_threshold: Adjust for chlorophyll classification
    - min_cell_area, max_cell_area: Filter cell size range
    - Classification criteria: Modify cell type definitions
    """)

    if __name__ == "__main__":
    analyzer = WolffiaAnalyzer(pixel_to_micron_ratio=0.5, chlorophyll_threshold=0.6)

    image_paths = [
    r"C:\Users\Aun\Desktop\Projects\bioimaging\1-2205.jfif",
    r"C:\Users\Aun\Desktop\Projects\bioimaging\1-2305.jfif",
    r"C:\Users\Aun\Desktop\Projects\bioimaging\1-2305 - Copy.jfif",
    r"C:\Users\Aun\Desktop\Projects\bioimaging\2-2205.jfif"
    ]

    timestamps = ["Day1_Morning", "Day2_Morning", "Day2_Evening", "Day3_Morning"]

    results = analyzer.analyze_time_series(image_paths, timestamps)
    analyzer.export_comprehensive_results("C:/Users/Aun/Desktop/Projects/bioimaging/wolffia_results")
