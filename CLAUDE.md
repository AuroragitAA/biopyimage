# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BIOIMAGIN is a focused bioimage analysis system specifically designed for core analysis of *Wolffia arrhiza* (the world's smallest flowering plant). The system provides reliable, automated cell detection, counting, measurements, and time series tracking with professional-quality results.

### Core Architecture

- **bioimaging.py**: Legacy analysis engine with basic `WolffiaAnalyzer` class 
- **bioimaging_professional.py**: Original professional analysis pipeline with CellPose + SimpleITK integration
- **bioimaging_professional_improved.py**: Refined professional pipeline with enhanced error handling, learning system, and better visualizations
- **web_integration.py**: Flask web server that automatically chooses best available pipeline (improved → professional → legacy)
- **templates/index.html**: Enhanced web interface with professional parameter controls
- **static/styles.css**: Updated styling for professional interface
- **models/**: Pre-trained CellPose models (cyto_*.pla, nuclei_*.pla) for segmentation

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
# Install dependencies (includes CellPose, torch, SimpleITK, etc.)
pip install -r requirements.txt

# Initialize system (tests improved professional pipeline first)
python -c "
try:
    from bioimaging_professional_improved import WolffiaAnalyzer
    print('✅ Improved professional pipeline ready')
except ImportError:
    try:
        from bioimaging_professional import WolffiaAnalyzer
        print('✅ Professional pipeline ready')
    except ImportError:
        from bioimaging import WolffiaAnalyzer  
        print('⚠️ Using legacy pipeline')
"

# Launch web interface (auto-detects best available pipeline)
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
# Test improved professional pipeline (preferred)
python -c "
from bioimaging_professional_improved import WolffiaAnalyzer
analyzer = WolffiaAnalyzer()
print('Improved professional analyzer ready')
print('Engine status:', analyzer.get_current_parameters()['engines_status'])
"

# Test original professional pipeline
python -c "
from bioimaging_professional import WolffiaAnalyzer
analyzer = WolffiaAnalyzer()
print('Professional analyzer ready with CellPose integration')
"

# Test legacy pipeline
python -c "
from bioimaging import WolffiaAnalyzer
analyzer = WolffiaAnalyzer()
print('Legacy analyzer ready')
"

# Test web integration with pipeline detection
python -c "
import requests
import time
import subprocess
subprocess.Popen(['python', 'web_integration.py'])
time.sleep(3)
response = requests.get('http://localhost:5000/api/health_check')
if response.status_code == 200:
    data = response.json()
    print(f'Web server status: {data[\"status\"]}')
    print(f'Pipeline version: {data[\"version\"]}')
else:
    print('Web server failed to start')
"
```

## Development Guidelines

### Code Organization
- **Professional Pipeline**: `WolffiaAnalyzer` class in bioimaging_professional.py (preferred)
- **Legacy Pipeline**: `WolffiaAnalyzer` class in bioimaging.py (fallback)
- **Web Integration**: web_integration.py automatically detects and uses professional pipeline
- **API Endpoints**: All endpoints prefixed with `/api/` in web_integration.py  
- **File Structure**:
  - `uploads/`: Temporary uploaded images
  - `results/`: Individual analysis results (CSV format)
  - `wolffia_results/`: Time series and comprehensive analysis outputs
  - `training_data/`: ML training data in JSON format
  - `models/`: CellPose pre-trained models (.pla files)
  - `annotations/`: Manual annotations for training
  - `imagepy/`: Documentation and notes on different analysis methods

### Key Classes and Methods

**Improved Professional Pipeline (bioimaging_professional_improved.py)**:
- `WolffiaAnalyzer.analyze_image_professional()`: Main professional analysis method with enhanced parameters
- Modular engine architecture: restoration, segmentation, feature extraction, quality assessment, learning
- Advanced error handling and fallback mechanisms
- Real-time learning system for model improvement
- Professional visualizations similar to CellPose GUI
- Support for configurable diameter and flow threshold parameters

**Original Professional Pipeline (bioimaging_professional.py)**:
- Uses CellPose for advanced cell segmentation
- Integrates with cellpose-planer for model management
- Copies models from `models/` directory to cellpose-planer installation

**Legacy Pipeline (bioimaging.py)**:
- `WolffiaAnalyzer.analyze_single_image()`: Basic single image analysis
- `WolffiaAnalyzer.advanced_preprocess_image()`: Image preprocessing pipeline  
- `WolffiaAnalyzer.multi_method_segmentation()`: Multi-algorithm segmentation
- `WolffiaAnalyzer.extract_ml_features()`: Feature extraction for ML
- `WolffiaAnalyzer.ml_classify_cells()`: Cell classification using Random Forest

**Web Integration (web_integration.py)**:
- `process_core_analysis()`: Background processing with automatic pipeline selection
- Enhanced parameter API endpoints supporting professional controls
- Automatic pipeline detection and fallback (improved → professional → legacy)

### File Paths and Structure
- Analysis results: `/results/`
- Uploaded files: `/uploads/`
- Time series results: `/wolffia_results/`
- Model outputs: Saved as CSV, JSON, and ZIP formats

### Important Parameters

**Basic Parameters**:
- `pixel_to_micron_ratio`: Spatial calibration (default: 0.5)
- `chlorophyll_threshold`: Green cell detection (default: 0.6) 
- `min_area_microns`: Minimum cell area filter (default: 30)
- `max_area_microns`: Maximum cell area filter (default: 12000)

**Professional CellPose Parameters** (improved pipeline only):
- `diameter`: Expected cell diameter in pixels for CellPose (default: 30)
- `flow_threshold`: CellPose flow threshold for segmentation strictness (default: 0.4)
- `restoration_mode`: Image restoration method ('auto', 'denoise', 'enhance', 'full', 'none')
- `segmentation_model`: CellPose model selection ('auto', 'cyto3', 'nuclei', 'custom')
- `learn_from_analysis`: Enable learning system for model improvement (default: True)

### API Endpoints
- `POST /api/upload`: Upload images for analysis
- `GET /api/analyze/<id>`: Get analysis status/results
- `POST /api/set_parameters`: Update analysis parameters (supports professional CellPose parameters)
- `GET /api/get_parameters`: Get current analysis parameters
- `POST /api/calibrate`: Calibrate pixel-to-micron ratio
- `GET /api/export/<id>/<format>`: Export results (csv/json/zip)
- `GET /api/health_check`: System status check with pipeline detection

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
- Maximum file size: 50MB per image
- Recommended: High-resolution microscopy images of Wolffia cultures

### Output
- **CSV**: Individual cell measurements (`results/*.csv`)
- **JSON**: Complete analysis results with metadata (`training_data/*.json`)
- **ZIP**: Comprehensive packages with visualizations and reports
- **Time Series**: Population dynamics data (`wolffia_results/*.csv`)

## Development Workflow

### Debugging Issues
```bash
# Check which pipeline is being used
python -c "
from web_integration import analyzer
print(f'Using: {analyzer.__class__.__module__}.{analyzer.__class__.__name__}')
"

# Test CellPose models availability
python -c "
import cellpose_planer as cellpp
cellpp.search_models()
print('Available models:', cellpp.list_models())
"

# Check model files
ls -la models/

# View recent analysis results
ls -la results/ | tail -5
ls -la training_data/ | tail -5
```

### Adding New Models
```bash
# Copy new CellPose models to models directory
cp new_model.pla models/

# Restart web server to reload models
# Models are automatically copied to cellpose-planer on startup
```

### Working with Analysis Results
- Individual cell data: `results/<analysis_id>_cells.csv`
- Time series data: `wolffia_results/`
- Training data: `training_data/analysis_<timestamp>.json`
- Web uploads: `uploads/` (temporary storage)