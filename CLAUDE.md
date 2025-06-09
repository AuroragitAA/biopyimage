# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BIOIMAGIN OPTIMIZED is a streamlined, high-performance bioimage analysis system specifically designed for *Wolffia arrhiza* cell analysis. The system delivers fast, accurate cell detection with smart multi-method segmentation, essential visualizations, and AI-powered tophat training capabilities. Built from the ground up for simplicity, speed, and accuracy.

### Core Architecture

- **bioimaging.py**: Optimized analysis engine with smart multi-method detection (OptimizedWolffiaAnalyzer)
- **web_integration.py**: Streamlined Flask API with essential endpoints and tophat training
- **templates/index.html**: Clean, responsive interface focused on core functionality
- **run_optimized.py**: Simple launcher with dependency checking and system setup
- **models/**: Pre-trained CellPose models and tophat AI training models

### Key Components

1. **Smart Detection Pipeline**:
   - Adaptive preprocessing based on image quality assessment
   - Multi-method segmentation: Optimized Watershed + CellPose + Tophat AI
   - Intelligent fusion of detection results with duplicate removal
   - Wolffia-specific size filtering (50-1200 pixels area)

2. **Essential Outputs**:
   - Cell count with high accuracy
   - Total and average cell area measurements  
   - Single comprehensive labeled visualization
   - Fast processing (typically <5 seconds per image)

3. **AI Training System**:
   - Tophat model training with user corrections
   - Interactive annotation interface for marking correct/incorrect cells
   - Random Forest classifier for improved detection
   - Persistent model storage and reuse

4. **Simplified Interface**:
   - Drag-and-drop file upload
   - Real-time progress tracking
   - Essential metrics display only
   - CSV/JSON export functionality

## Common Development Commands

### Installation and Setup
```bash
# Install dependencies
pip install opencv-python numpy matplotlib pandas scikit-learn flask flask-cors

# Install optional but recommended packages
pip install cellpose scikit-image

# Quick start with optimized launcher (recommended)
python run_optimized.py

# Alternative: Direct web server launch
python web_integration.py
```

### Running Analysis
```bash
# Start optimized system (checks dependencies automatically)
python run_optimized.py

# Access web interface at http://localhost:5000
# Features: Upload images, analyze with/without tophat, train AI model
```

### System Testing
```bash
# Test optimized analyzer
python -c "from bioimaging import WolffiaAnalyzer; analyzer = WolffiaAnalyzer(); print('✅ Optimized analyzer ready')"

# Test specific analysis on sample image
python -c "
from bioimaging import WolffiaAnalyzer
analyzer = WolffiaAnalyzer()
# result = analyzer.analyze_image('path/to/image.jpg')
print('✅ Analysis method available')
"

# Check tophat model status
python -c "
from bioimaging import WolffiaAnalyzer
analyzer = WolffiaAnalyzer()
print(f'Tophat model available: {analyzer.tophat_model is not None}')
"
```

### Advanced Testing
```bash
# Test main analysis pipeline
python -c "
from bioimaging import WolffiaAnalyzer
analyzer = WolffiaAnalyzer()
print('✅ Main analyzer ready')
print('Features: Multi-method segmentation, CellPose integration, biomass estimation')
"

# Test web server health
python -c "
import requests
import time
import subprocess
subprocess.Popen(['python', 'web_integration.py'])
time.sleep(3)
response = requests.get('http://localhost:5000/api/health_check')
if response.status_code == 200:
    data = response.json()
    print(f'✅ Web server status: {data[\"status\"]}')
else:
    print('❌ Web server failed to start')
"
```

## Development Guidelines

### Code Organization
- **Main Pipeline**: `WolffiaAnalyzer` class in bioimaging.py (primary analysis engine)
- **Web Integration**: web_integration.py with Flask API and background processing
- **API Endpoints**: All endpoints prefixed with `/api/` in web_integration.py  
- **File Structure**:
  - `uploads/`: Temporary uploaded images
  - `results/`: Individual analysis results (CSV format)
  - `wolffia_results/`: Time series and comprehensive analysis outputs
  - `training_data/`: ML training data in JSON format
  - `learning_system/`: AI learning system data and model performance tracking
  - `models/`: CellPose pre-trained models (.pla files)
  - `annotations/`: Manual annotations for training
  - `imagepy/`: Documentation and notes on different analysis methods

### Key Classes and Methods

**Optimized Analysis Pipeline (bioimaging.py)**:
- `WolffiaAnalyzer.analyze_image()`: Main analysis method - fast, accurate, simple
- `WolffiaAnalyzer.smart_preprocess()`: Adaptive preprocessing based on image quality
- `WolffiaAnalyzer.smart_detect_cells()`: Multi-method detection with intelligent fusion
- `WolffiaAnalyzer.watershed_detection()`: Optimized watershed segmentation for Wolffia
- `WolffiaAnalyzer.cellpose_detection()`: CellPose integration (if available)
- `WolffiaAnalyzer.tophat_detection()`: AI-powered detection using trained model
- `WolffiaAnalyzer.fuse_detections()`: Smart fusion removes duplicates, keeps best results
- `WolffiaAnalyzer.create_essential_visualization()`: Single labeled cell image with stats

**Web Integration (web_integration.py)**:
- `upload_files()`: Multi-file upload with validation
- `analyze_image()`: Background analysis with progress tracking  
- `get_analysis_status()`: Real-time status and results retrieval
- `export_results()`: CSV/JSON export functionality

**Tophat AI Training (bioimaging.py)**:
- `start_tophat_training()`: Initialize training session with multiple images
- `save_user_annotations()`: Store user corrections for model training
- `train_tophat_model()`: Train Random Forest classifier from annotations
- `collect_training_data()`: Prepare features and labels from user feedback

### File Paths and Structure
- Analysis results: `/results/`
- Uploaded files: `/uploads/`
- Time series results: `/wolffia_results/`
- Model outputs: Saved as CSV, JSON, and ZIP formats

### Important Parameters

**Core Parameters** (automatically optimized):
- Cell area range: 50-1200 pixels (optimized for Wolffia)
- CellPose diameter: 25 pixels (Wolffia-specific)
- Detection methods: Watershed + CellPose + Tophat AI
- Quality assessment: Automatic contrast/brightness adaptation
- Size filtering: Wolffia-specific morphological constraints

**User Options**:
- `use_tophat`: Enable AI-trained tophat model (if available)
- File formats: PNG, JPG, JPEG, BMP, TIFF, JFIF (max 50MB each)
- Export formats: CSV (cell data), JSON (complete results)

### API Endpoints

**Core Analysis**:
- `POST /api/upload`: Upload multiple images for analysis
- `POST /api/analyze/<file_id>`: Start background analysis of specific image
- `GET /api/status/<analysis_id>`: Get real-time analysis progress and results
- `GET /api/export/<analysis_id>/<format>`: Export results (csv/json)
- `GET /api/health`: System health check and version info

**Tophat AI Training**:
- `POST /api/tophat/start_training`: Initialize training session with images
- `POST /api/tophat/save_annotations`: Save user corrections for training
- `POST /api/tophat/train_model`: Train AI model from annotations
- `GET /api/tophat/model_status`: Check if tophat model is available

### Key Features

**Optimized Performance**:
- Fast analysis: Typically <5 seconds per image
- Smart preprocessing: Automatic quality assessment and enhancement
- Multi-method detection: Combines best of watershed, CellPose, and AI
- Intelligent fusion: Eliminates duplicates, keeps highest confidence detections

**Essential Simplicity**:
- Single comprehensive visualization per image
- Core metrics only: cell count, total area, average area
- Clean, responsive web interface
- Real-time progress tracking

**AI Training Capability**:
- Interactive tophat model training
- User-guided annotation system
- Persistent model improvement
- Immediate deployment of trained models

## Data Formats

### Input
- **Formats**: PNG, JPG, JPEG, BMP, TIFF, JFIF (max 50MB each)
- **Quality**: Any quality - system adapts automatically
- **Quantity**: Single images or multiple files for batch processing
- **Optimized for**: Wolffia arrhiza microscopy images

### Output
- **Essential Metrics**: Cell count, total area, average area, processing time
- **Visualization**: Single labeled image with numbered cells and statistics
- **Export Formats**: CSV (cell data), JSON (complete results)
- **Training Data**: Stored automatically for tophat model training

## Development Workflow

### Quick Debugging
```bash
# Check system status with optimized launcher
python run_optimized.py
# This automatically checks dependencies and starts the system

# Test analysis pipeline directly
python -c "
from bioimaging import WolffiaAnalyzer
analyzer = WolffiaAnalyzer()
print('✅ Optimized analyzer ready')
print(f'Tophat model: {\"Available\" if analyzer.tophat_model else \"Not trained\"}')
"

# Check web server health
curl http://localhost:5000/api/health
```

### Tophat Model Management
```bash
# Check tophat model status
python -c "
from bioimaging import WolffiaAnalyzer
analyzer = WolffiaAnalyzer()
model_path = analyzer.dirs['models'] / 'tophat_model.pkl'
print(f'Model file exists: {model_path.exists()}')
"

# View training sessions
ls -la tophat_training/

# View annotations
ls -la annotations/
```

### Results and Data
- **Analysis results**: `results/<analysis_id>_result.json`
- **Cell data exports**: `results/<analysis_id>_cells.csv` 
- **Training sessions**: `tophat_training/session_<timestamp>.json`
- **User annotations**: `annotations/<session>_<image>_annotation.json`
- **Uploaded files**: `uploads/` (temporary)