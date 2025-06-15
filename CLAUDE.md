# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BIOIMAGIN OPTIMIZED (Version 3.0 - Enhanced) is a cutting-edge bioimage analysis system specifically designed for *Wolffia arrhiza* (the world's smallest flowering plant) cell detection and analysis. Part of "BinAqua: Klimafreundliche Herstellung vollwertiger veganer Proteinpulver durch die Co-Kultivierung von Mikroalgen und Wasserlinsen" at BTU Cottbus-Senftenberg.

The system combines state-of-the-art deep learning, realistic synthetic data generation, and user-guided training to achieve CellPose-level precision with custom adaptability. Built from the ground up for simplicity, speed, and accuracy.

### Core Architecture

- **bioimaging.py**: Optimized analysis engine with smart multi-method detection (WolffiaAnalyzer)
- **web_integration.py**: Streamlined Flask API with essential endpoints and tophat training
- **wolffia_cnn_model.py**: Lightweight CNN architecture for enhanced Wolffia cell detection
- **train_wolffia_cnn.py**: User-friendly CNN training script with synthetic data generation
- **templates/index.html**: Clean, responsive interface focused on core functionality
- **models/**: Pre-trained CellPose models, tophat AI training models, and CNN models

### Key Components

1. **Smart Detection Pipeline**:
   - Adaptive preprocessing based on image quality assessment
   - Multi-method segmentation: Optimized Watershed + CellPose + Tophat AI + Wolffia CNN
   - Intelligent fusion of detection results with duplicate removal
   - Wolffia-specific size filtering (50-1200 pixels area)
   - AI-enhanced detection using trained CNN model

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

## Quick Start Guide

For new installations and immediate usage:

```bash
# 1. Install core dependencies
pip install -r requirements.txt

# 2. Install optional but recommended packages for enhanced features
pip install torch torchvision cellpose>=3.0.0

# 3. Train all available models (recommended)
python train_all_models.py

# 4. Start the web interface
python web_integration.py
# Navigate to http://localhost:5000 to access the analysis interface

# 5. Test the system
python -c "from bioimaging import WolffiaAnalyzer; print('✅ System ready')"
```

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

### CNN Model Training
```bash
# Train ALL available models (RECOMMENDED for new installations)
python train_all_models.py

# Train basic Wolffia CNN model with synthetic data
python train_wolffia_cnn.py

# Train ENHANCED multi-task CNN with edge detection + watershed
python train_enhanced_wolffia_cnn.py

# Quick CNN training (faster, less comprehensive)
python quick_train_cnn.py

# Test CNN integration
python test_cnn_integration.py

# Test enhanced CNN specifically
python test_enhanced_cnn.py

# Extract training data from real images
python real_image_analyzer.py

# REALISTIC synthetic data generation (RECOMMENDED)
python realistic_wolffia_generator.py

# Train ML models using tophat annotation data
python tophat_ml_trainer.py

# Test all new enhancements
python test_realistic_generator.py

# Visualize synthetic training data
python visualize_training_data.py

# Evaluate detection methods (compare all methods)
python evaluate_detection_methods.py
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

# Run comprehensive test suite
python test_all_enhancements.py

# Test deployment readiness
python test_deployment.py

# Test recent fixes and improvements
python test_fixes.py
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
  - `docs/`: Comprehensive documentation (Architecture, Usage Guide, API Reference)
  - `evaluation_results/`: Performance comparison data and plots
  - `extracted_training_data/`: Training data extracted from real images
  - `backup/`: Backup copies of important files
  - `python_for_microscopists-master/`: Reference materials and examples for microscopy analysis

### Key Classes and Methods

**Optimized Analysis Pipeline (bioimaging.py)**:
- `WolffiaAnalyzer.analyze_image()`: Main analysis method - fast, accurate, simple
- `WolffiaAnalyzer.smart_preprocess()`: Adaptive preprocessing based on image quality
- `WolffiaAnalyzer.smart_detect_cells()`: Multi-method detection with intelligent fusion
- `WolffiaAnalyzer.watershed_detection()`: Optimized watershed segmentation for Wolffia
- `WolffiaAnalyzer.cellpose_detection()`: CellPose integration (if available)
- `WolffiaAnalyzer.tophat_detection()`: AI-powered detection using trained model
- `WolffiaAnalyzer.wolffia_cnn_detection()`: CNN-based detection using trained Wolffia model
- `WolffiaAnalyzer.enhanced_wolffia_cnn_detection()`: Enhanced multi-task CNN with edge detection and watershed
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

**CNN Model Architecture**:
- **wolffia_cnn_model.py**: Lightweight CNN optimized for Wolffia cell detection
- **enhanced_wolffia_cnn.py**: Multi-task CNN with edge detection and watershed capabilities
- **realistic_wolffia_generator.py**: Highly realistic synthetic data generator with lighting effects
- **train_wolffia_cnn.py**: User-friendly training script with synthetic data generation
- **train_enhanced_wolffia_cnn.py**: Enhanced training with multi-task learning
- **train_all_models.py**: Universal model training script (trains all available models)
- **real_image_analyzer.py**: Extract training data from real Wolffia images
- **tophat_ml_trainer.py**: Train ML models using tophat annotation data
- **WolffiaCNN**: Custom PyTorch model with 3 conv layers + batch norm + dropout
- **EnhancedWolffiaCNN**: U-Net architecture with mask + edge + distance prediction
- **RealisticWolffiaGenerator**: Advanced generator with Poisson disc sampling and lighting
- **TophatAnnotationAnalyzer**: Extracts training data from tophat annotation sessions

### File Paths and Structure
- Analysis results: `/results/`
- Uploaded files: `/uploads/`
- Time series results: `/wolffia_results/`
- Model outputs: Saved as CSV, JSON, and ZIP formats

### Important Parameters

**Core Parameters** (automatically optimized):
- Cell area range: 50-1200 pixels (optimized for Wolffia)
- CellPose diameter: 25 pixels (Wolffia-specific)
- Detection methods: Watershed + CellPose + Tophat AI + Wolffia CNN
- Quality assessment: Automatic contrast/brightness adaptation
- Size filtering: Wolffia-specific morphological constraints

**User Options**:
- `use_tophat`: Enable AI-trained tophat model (if available)
- `use_wolffia_cnn`: Enable CNN-based detection (if model trained)
- `use_enhanced_cnn`: Enable Enhanced multi-task CNN (if model trained)
- `use_celldetection`: Enable CellDetection method (if available)
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
- Multi-method detection: Combines best of watershed, CellPose, Tophat AI, and CNN
- Intelligent fusion: Eliminates duplicates, keeps highest confidence detections
- GPU acceleration: Supports CUDA for CNN inference when available

**Essential Simplicity**:
- Single comprehensive visualization per image
- Core metrics only: cell count, total area, average area
- Clean, responsive web interface
- Real-time progress tracking

**AI Training Capability**:
- Interactive tophat model training
- User-guided annotation system
- Automated CNN training with synthetic data
- Persistent model improvement
- Immediate deployment of trained models
- Performance comparison between detection methods

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

## CNN Enhancement Features

### Wolffia CNN Model
The system includes a lightweight CNN specifically trained for Wolffia cell detection:

**Key Features**:
- **Synthetic Training Data**: Generates realistic Wolffia-like training samples automatically
- **Optimized Architecture**: 3-layer CNN with batch normalization and dropout
- **Fast Training**: Typically 5-15 minutes on modern hardware
- **High Accuracy**: >85% detection accuracy on synthetic test data
- **GPU Support**: Automatic CUDA detection and acceleration

**Training Process**:
1. **Synthetic Data Generation**: Creates thousands of realistic cell-like patches
2. **Augmentation**: Random rotations, flips, and color variations
3. **Training**: Supervised learning with early stopping
4. **Evaluation**: Comprehensive testing with performance metrics
5. **Integration**: Automatic integration into analysis pipeline

**CNN Training Commands**:
```bash
# Train ALL available models (RECOMMENDED for new installations)
python train_all_models.py

# Basic CNN training with synthetic data
python train_wolffia_cnn.py

# ENHANCED CNN training (RECOMMENDED for best results)
python train_enhanced_wolffia_cnn.py

# Extract training data from your real images
python real_image_analyzer.py

# Quick training for testing
python quick_train_cnn.py

# Visualize synthetic training data
python visualize_training_data.py

# Test CNN integration
python test_cnn_integration.py

# Test enhanced CNN specifically
python test_enhanced_cnn.py
```

**Enhanced CNN Workflow**:
1. **Generate Realistic Data**: `python realistic_wolffia_generator.py` - Creates highly realistic synthetic training data
2. **Extract Real Data**: `python real_image_analyzer.py` - Analyzes your Wolffia images to create high-quality training targets
3. **Train Enhanced Model**: `python train_enhanced_wolffia_cnn.py` - Trains multi-task CNN with edge detection and watershed
4. **Test Integration**: `python test_enhanced_cnn.py` - Verifies the enhanced model works correctly
5. **Enable in Analysis**: Use `use_enhanced_cnn=True` parameter in analysis calls

**Tophat ML Workflow** (NEW):
1. **Create Annotations**: Use tophat training interface to mark correct/incorrect cells
2. **Train Tophat ML**: `python tophat_ml_trainer.py` - Trains models using your annotation data
3. **Use Trained Models**: Models automatically integrate into detection pipeline

### Enhanced Detection Methods
The system now supports multiple complementary detection methods:

1. **Shape Index Method**: Original watershed-based detection
2. **CellPose Integration**: Professional cell segmentation (if installed)
3. **CellDetection Method**: Advanced AI-powered detection (if installed)
4. **Tophat AI Model**: User-trained model from manual annotations
5. **Wolffia CNN**: Custom-trained CNN for Wolffia-specific detection
6. **Enhanced Wolffia CNN**: Multi-task CNN with edge detection and watershed

**Method Selection Priority** (when multiple methods enabled):
1. Enhanced Wolffia CNN (highest precision, if available)
2. Wolffia CNN (high accuracy, if available)
3. CellDetection (AI-powered, if available)
4. Tophat AI Model (user-trained, if available)
5. Shape Index Method (classical, always available)

**Method Features**:
- Methods are automatically detected based on available models/packages
- Users can enable/disable methods via web interface
- Results are intelligently fused for optimal accuracy
- Performance comparison tools available for evaluation
- Enhanced CNN provides precise boundaries via watershed post-processing

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
    import celldetection; print('✅ CellDetection available')
except: print('⚠️ CellDetection not available')
try:
    from bioimaging import WolffiaAnalyzer; print('✅ Core analyzer available')
except Exception as e: print(f'❌ Core analyzer failed: {e}')
"
```

### Model Management
```bash
# Check tophat model status
python -c "
from bioimaging import WolffiaAnalyzer
analyzer = WolffiaAnalyzer()
model_path = analyzer.dirs['models'] / 'tophat_model.pkl'
print(f'Tophat model exists: {model_path.exists()}')
"

# Check CNN model status
python -c "
from bioimaging import WolffiaAnalyzer
from pathlib import Path
analyzer = WolffiaAnalyzer()
cnn_path = Path('models/wolffia_cnn_best.pth')
print(f'CNN model exists: {cnn_path.exists()}')
print(f'CNN available: {analyzer.wolffia_cnn_available}')
"

# View training sessions and annotations
ls -la tophat_training/
ls -la annotations/
ls -la models/
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
black bioimaging.py web_integration.py tophat_trainer.py wolffia_cnn_model.py

# Code linting (if flake8 is available)  
flake8 bioimaging.py web_integration.py --max-line-length=100

# Test CNN integration
python test_cnn_integration.py

# Test improved detection methods
python test_improvements.py

# Evaluate all detection methods
python evaluate_detection_methods.py

# Run comprehensive enhancement tests
python test_all_enhancements.py

# Test system deployment
python test_deployment.py

# Run basic functionality tests
python -m pytest tests/ 2>/dev/null || echo "No pytest tests configured"

# Manual testing of core components
python -c "
from bioimaging import WolffiaAnalyzer
analyzer = WolffiaAnalyzer()
print('Core system test passed')
print(f'CNN Available: {analyzer.wolffia_cnn_available}')
print(f'CellDetection Available: {analyzer.celldetection_available}')
print(f'Tophat Available: {analyzer.tophat_model is not None}')
"
```

## Performance Benchmarks

### Speed (typical 1024x1024 image)
- **Enhanced CNN**: 2-3 seconds (GPU), 8-12 seconds (CPU)
- **Regular CNN**: 1-2 seconds (GPU), 4-6 seconds (CPU)
- **Tophat ML**: 0.5-1 seconds
- **Watershed**: 0.2-0.5 seconds

### Accuracy (on test datasets)
- **Enhanced CNN**: Highest precision, excellent edge detection
- **Tophat ML**: Customized to your specific image characteristics
- **CellPose**: Good general-purpose performance
- **Watershed**: Fast but lower precision

## Documentation Reference

For comprehensive documentation, see:
- **`docs/ARCHITECTURE.md`**: System design and components
- **`docs/USAGE_GUIDE.md`**: Detailed usage instructions and examples
- **`docs/API_REFERENCE.md`**: Complete API documentation