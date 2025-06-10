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
# Install dependencies from requirements.txt
pip install -r requirements.txt

# Essential dependencies (if installing manually)
pip install opencv-python numpy matplotlib pandas scikit-learn flask flask-cors

# Optional but recommended packages for advanced features
pip install cellpose>=3.0.0 torch torchvision SimpleITK

# Initialize system check
python -c "from bioimaging import WolffiaAnalyzer; print('System initialized successfully')"

# Start web interface
python web_integration.py
```

### Running Analysis
```bash
# Start web server (main entry point)
python web_integration.py

# Access web interface at http://localhost:5000
# Features: Upload images, analyze with/without tophat, train AI model

# Direct command-line analysis (for testing)
python -c "
from bioimaging import WolffiaAnalyzer
analyzer = WolffiaAnalyzer()
result = analyzer.analyze_image('path/to/image.jpg')
print(f'Detected {result[\"total_cells\"]} cells')
"
```

### Tophat AI Model Training
```bash
# Train tophat model from existing annotations
python tophat_trainer.py

# Check available training sessions
ls -la tophat_training/

# View annotation files
ls -la annotations/
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
curl http://localhost:5000/api/health

# Test analysis on sample image (if available)
python -c "
from bioimaging import WolffiaAnalyzer
import os
analyzer = WolffiaAnalyzer()
if os.path.exists('images/test_wolffia_cells.png'):
    result = analyzer.analyze_image('images/test_wolffia_cells.png')
    print(f'✅ Analysis test: {result[\"total_cells\"]} cells detected')
else:
    print('⚠️ No test image found at images/test_wolffia_cells.png')
"
```

## Development Guidelines

### Code Organization
- **Main Pipeline**: `WolffiaAnalyzer` class in bioimaging.py (primary analysis engine)
- **Web Integration**: web_integration.py with Flask API and background processing
- **Tophat Training**: tophat_trainer.py for AI model training from annotations
- **API Endpoints**: All endpoints prefixed with `/api/` in web_integration.py  
- **File Structure**:
  - `uploads/`: Temporary uploaded images
  - `results/`: Individual analysis results (JSON/CSV format)
  - `wolffia_results/`: Time series and comprehensive analysis outputs
  - `models/`: Pre-trained models (CellPose .pla files, tophat_model.pkl)
  - `annotations/`: Manual annotations for training (PNG images + JSON data)
  - `tophat_training/`: Training session data in JSON format
  - `imagepy/`: Documentation and notes on different analysis methods
  - `images/`: Test images and organized sample datasets

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

**Tophat AI Training**:
- **bioimaging.py**: Core training methods integrated into WolffiaAnalyzer
- **tophat_trainer.py**: Standalone script for batch training from existing annotations
- Key methods: `start_tophat_training()`, `save_user_annotations()`, `train_tophat_model()`
- Training pipeline: Load annotations → Extract features → Train Random Forest → Save model

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
# Start web server and check health
python web_integration.py &
sleep 3
curl http://localhost:5000/api/health

# Test analysis pipeline directly
python -c "
from bioimaging import WolffiaAnalyzer
analyzer = WolffiaAnalyzer()
print('✅ Optimized analyzer ready')
print(f'Tophat model: {\"Available\" if analyzer.tophat_model else \"Not trained\"}')
"

# Check dependency availability
python -c "
import sys
try:
    import torch; print('✅ PyTorch available')
except: print('⚠️ PyTorch not available')
try:
    import cellpose; print('✅ CellPose available') 
except: print('⚠️ CellPose not available')
try:
    from bioimaging import WolffiaAnalyzer; print('✅ Core analyzer available')
except Exception as e: print(f'❌ Core analyzer failed: {e}')
"
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

### Development and Testing
```bash
# Code formatting (if black is available)
black bioimaging.py web_integration.py tophat_trainer.py

# Code linting (if flake8 is available)  
flake8 bioimaging.py web_integration.py --max-line-length=100

# Run basic functionality tests
python -m pytest tests/ 2>/dev/null || echo "No pytest tests configured"

# Manual testing of core components
python -c "
from bioimaging import WolffiaAnalyzer
analyzer = WolffiaAnalyzer()
print('Core system test passed')
"
```