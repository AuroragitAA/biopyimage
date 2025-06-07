# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BIOIMAGIN is a focused bioimage analysis system specifically designed for core analysis of *Wolffia arrhiza* (the world's smallest flowering plant). The system provides reliable, automated cell detection, counting, measurements, and time series tracking with professional-quality results.

### Core Architecture

- **bioimaging.py**: Main analysis engine containing the `WolffiaAnalyzer` class with core image processing pipeline
- **web_integration.py**: Flask web server providing streamlined REST API for upload, analysis, and results
- **templates/index.html**: Simplified web interface for core analysis workflow
- **static/styles.css**: Styling for the web interface

### Key Components

1. **Core Image Processing Pipeline**:
   - Multi-scale morphological enhancement with adaptive filtering
   - Multi-method segmentation (Otsu, adaptive, K-means, Felzenszwalb, SLIC)
   - Reliable cell detection and boundary identification

2. **Essential Measurements**:
   - Accurate cell counting and size measurements
   - Biomass estimation using proven mathematical models
   - Green cell identification through chlorophyll detection
   - Basic spectral analysis for vegetation indices

3. **Core Analysis Capabilities**:
   - Single image analysis with essential quantitative parameters
   - Time series analysis for growth tracking
   - Cell-level data export and visualization
   - Professional reporting and data export

## Common Development Commands

### Installation and Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Initialize system
python -c "from bioimaging import WolffiaAnalyzer; print('System initialized successfully')"

# Launch web interface
python web_integration.py
```

### Running Analysis
```bash
# Run web server (main interface)
python web_integration.py

# Access web interface at http://localhost:5000
```

### Testing the System
```bash
# Quick test with sample analysis
python -c "
from bioimaging import WolffiaAnalyzer
analyzer = WolffiaAnalyzer()
print('Analyzer ready:', analyzer.get_current_parameters())
"
```

## Development Guidelines

### Code Organization
- Main analysis logic is in `WolffiaAnalyzer` class in bioimaging.py
- Web API endpoints are in web_integration.py with `/api/` prefix
- Analysis results are stored in `results/` directory
- Uploaded images go to `uploads/` directory
- Training data and annotations in respective folders

### Key Classes and Methods
- `WolffiaAnalyzer.analyze_single_image()`: Core single image analysis
- `analyze_uploaded_image()`: Main analysis function for web uploads
- `process_core_analysis()`: Background processing for web interface
- `WolffiaAnalyzer.advanced_preprocess_image()`: Image preprocessing pipeline
- `WolffiaAnalyzer.multi_method_segmentation()`: Cell segmentation methods

### File Paths and Structure
- Analysis results: `/results/`
- Uploaded files: `/uploads/`
- Time series results: `/wolffia_results/`
- Model outputs: Saved as CSV, JSON, and ZIP formats

### Important Parameters
- `pixel_to_micron_ratio`: Spatial calibration (default: 0.5)
- `chlorophyll_threshold`: Green cell detection (default: 0.6) 
- `min_area_microns`: Minimum cell area filter (default: 30)
- `max_area_microns`: Maximum cell area filter (default: 12000)

### API Endpoints
- `POST /api/upload`: Upload images for analysis
- `GET /api/analyze/<id>`: Get analysis status/results
- `POST /api/set_parameters`: Update analysis parameters
- `POST /api/calibrate`: Calibrate pixel-to-micron ratio
- `GET /api/export/<id>/<format>`: Export results (csv/json/zip)
- `GET /api/health_check`: System status check

### Error Handling
The system includes comprehensive error handling with detailed logging. Analysis progress is tracked and results are JSON-serializable for web interface compatibility.

### Core Features Only
- Focus on reliable cell detection, counting, and measurement
- Green cell identification through chlorophyll analysis
- Time series tracking for growth analysis
- Professional data export and visualization
- No advanced ML training or complex optimization features

## Data Formats

### Input
- Supported image formats: PNG, JPG, JPEG, TIFF, BMP, JFIF
- Multiple images for time series analysis
- Maximum file size: 50MB

### Output
- CSV: Individual cell measurements with all features
- JSON: Complete analysis results with metadata
- ZIP: Comprehensive package with visualizations and reports
- Training data format for ML model development