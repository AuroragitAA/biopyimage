# BIOIMAGIN Architecture Documentation

## System Overview

BIOIMAGIN is a high-performance bioimage analysis system specifically designed for *Wolffia arrhiza* cell analysis. The system features an enhanced CNN architecture with realistic synthetic data generation, tophat-based machine learning, and priority-based detection methods.

## Core Components

### 1. Main Analysis Engine (`bioimaging.py`)
- **WolffiaAnalyzer**: Core analysis class with smart multi-method detection
- **Priority-based method selection**: Enhanced CNN → Regular CNN → CellPose → Tophat → Watershed
- **Smart preprocessing**: Adaptive image enhancement based on quality assessment
- **Intelligent fusion**: Combines results from multiple detection methods

### 2. Enhanced CNN System (`wolffia_cnn.py`)
- **Multi-task U-Net architecture**: Mask + Edge + Distance prediction
- **CellPose-inspired design**: Skip connections and feature pyramid
- **Enhanced training pipeline**: Realistic synthetic data integration
- **Watershed post-processing**: Distance transform-based cell separation

### 3. Realistic Data Generator (`realistic_wolffia_generator.py`)
- **Poisson disc sampling**: Natural cell placement without overlap
- **Realistic backgrounds**: Gradients, noise textures, real image patches
- **Wolffia-like morphology**: Blobby ellipses with internal structures
- **Lighting effects**: Shadows, highlights, and brightness gradients

### 4. Tophat ML Training (`tophat_ml_trainer.py`)
- **Annotation-based training**: Uses real user feedback data
- **Classical ML models**: Random Forest with extracted features
- **Enhanced CNN training**: PyTorch-based deep learning
- **Quality assessment**: High/medium/low annotation quality

### 5. Web Interface (`web_integration.py`)
- **Flask-based API**: RESTful endpoints for analysis
- **Background processing**: Non-blocking analysis with progress tracking
- **Real-time updates**: WebSocket-style status monitoring
- **Tophat training interface**: Interactive annotation system

## Architecture Flow

```
Input Image
     ↓
Smart Preprocessing
     ↓
Priority Detection Pipeline:
  1. Enhanced CNN (if available)
  2. Regular CNN (if available)  
  3. CellPose (if available)
  4. Tophat ML (if trained)
  5. Watershed (fallback)
     ↓
Intelligent Result Fusion
     ↓
Post-processing & Visualization
     ↓
Output (Count, Areas, Labeled Image)
```

## Key Features

### Enhanced CNN Architecture
- **U-Net with multiple outputs**: Simultaneous mask, edge, and distance prediction
- **Skip connections**: Preserve fine-grained details
- **Multi-scale feature extraction**: Captures cells at different sizes
- **Watershed-ready**: Distance transforms enable precise segmentation

### Realistic Synthetic Data Generation
- **Natural cell placement**: Poisson disc sampling prevents overlap
- **Authentic backgrounds**: Real image patches and procedural generation
- **Wolffia-specific morphology**: Internal dots for chloroplasts
- **Lighting simulation**: Shadows and highlights for realism

### Tophat-Based Training
- **User feedback integration**: Learn from annotation corrections
- **Balanced datasets**: Equal positive/negative samples
- **Feature engineering**: Color, texture, and geometric features
- **Continuous improvement**: Models update with new annotations

### Smart Detection Pipeline
- **Adaptive preprocessing**: Contrast and brightness optimization
- **Method prioritization**: Best available method selected automatically
- **Result fusion**: Combines predictions with confidence weighting
- **Quality filtering**: Size and circularity constraints

## Data Flow

### Training Data Flow
```
Real Images → Annotation Interface → Tophat Training → Model Updates
     ↓
Synthetic Generator → Enhanced CNN Training → Model Deployment
```

### Analysis Data Flow
```
Input Image → Preprocessing → Multi-method Detection → Fusion → Results
```

## Model Storage

### File Structure
```
models/
├── wolffia_cnn_best.pth    # Enhanced CNN model
├── wolffia_cnn_best.pth             # Regular CNN model  
├── tophat_model.pkl                 # Tophat ML model
├── tophat_model_info.json           # Tophat model metadata
├── cyto_*.pla                       # CellPose models
└── training_history.json            # Training metrics
```

### Model Types
1. **Enhanced CNN**: Multi-task deep learning model
2. **Regular CNN**: Basic classification model
3. **Tophat ML**: Random Forest based on annotations
4. **CellPose**: Pre-trained cell segmentation models

## Performance Characteristics

### Speed
- **Enhanced CNN**: ~2-3 seconds per image (GPU)
- **Regular CNN**: ~1-2 seconds per image
- **Tophat ML**: ~0.5-1 seconds per image
- **Watershed**: ~0.2-0.5 seconds per image

### Accuracy
- **Enhanced CNN**: Highest precision, best edge detection
- **Tophat ML**: Customized to specific image types
- **CellPose**: Good general-purpose performance
- **Watershed**: Fast but lower precision

## Integration Points

### Web API Endpoints
- `POST /api/analyze/<file_id>`: Start analysis
- `GET /api/status/<analysis_id>`: Check progress
- `POST /api/tophat/start_training`: Begin annotation session
- `POST /api/tophat/save_annotations`: Save user feedback

### Training Interfaces
- **Enhanced CNN**: `train_wolffia_cnn.py`
- **Tophat ML**: `tophat_ml_trainer.py`
- **Data generation**: `realistic_wolffia_generator.py`

## Dependencies

### Core Libraries
- **OpenCV**: Image processing and computer vision
- **scikit-image**: Advanced image analysis
- **NumPy/SciPy**: Numerical computing
- **scikit-learn**: Machine learning utilities

### Deep Learning (Optional)
- **PyTorch**: Enhanced CNN training and inference
- **torchvision**: Computer vision utilities
- **CellPose**: Pre-trained cell segmentation

### Web Framework
- **Flask**: Web server and API
- **Flask-CORS**: Cross-origin request handling

## Extensibility

### Adding New Detection Methods
1. Implement detection function in `WolffiaAnalyzer`
2. Add method to priority list in `smart_detect_cells`
3. Include availability check in initialization

### Custom Training Data
1. Add new generator in `realistic_wolffia_generator.py`
2. Update dataset class in `wolffia_cnn.py`
3. Retrain models with new data

### API Extensions
1. Add new endpoints in `web_integration.py`
2. Update frontend interface as needed
3. Document new functionality

## Configuration

### Analysis Parameters
- **Cell size range**: 50-1200 pixels area
- **CellPose diameter**: 25 pixels
- **Detection confidence**: Method-specific thresholds
- **Fusion weights**: Confidence-based combination

### Training Parameters
- **Enhanced CNN**: 40 epochs, 0.0005 learning rate
- **Batch size**: 16 samples
- **Data augmentation**: Rotation, flip, color jitter
- **Early stopping**: 20 epochs patience

This architecture provides a robust, extensible framework for high-precision Wolffia cell analysis with continuous learning capabilities.