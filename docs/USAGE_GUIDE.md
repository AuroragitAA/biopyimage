# BIOIMAGIN Usage Guide

## Quick Start

### 1. Installation
```bash
# Install required dependencies
pip install -r requirements.txt

# Optional: Install PyTorch for enhanced CNN features
pip install torch torchvision

# Optional: Install CellPose for advanced segmentation
pip install cellpose>=3.0.0
```

### 2. Basic Analysis
```bash
# Start the web interface
python web_integration.py

# Access the interface at http://localhost:5000
# Upload images and analyze with default settings
```

### 3. Command Line Analysis
```python
from bioimaging import WolffiaAnalyzer

# Initialize analyzer
analyzer = WolffiaAnalyzer()

# Analyze single image
result = analyzer.analyze_image('path/to/image.jpg')
print(f"Detected {result['total_cells']} cells")
```

## Web Interface Guide

### Image Upload
1. **Drag and drop** files onto the upload area
2. **Click to browse** and select multiple files
3. **Supported formats**: PNG, JPG, JPEG, BMP, TIFF, JFIF
4. **File size limit**: 50MB per file

### Analysis Options
- **Standard Analysis**: Uses best available detection method
- **Enhanced CNN**: Uses multi-task CNN (if trained)
- **Tophat Training**: Enable annotation interface for model training

### Results Interpretation
- **Cell Count**: Total number of detected cells
- **Total Area**: Sum of all cell areas in pixels
- **Average Area**: Mean cell size
- **Labeled Image**: Visualization with numbered cells

### Export Options
- **CSV**: Cell-by-cell data with coordinates and measurements
- **JSON**: Complete analysis results with metadata
- **ZIP**: Bulk export of multiple analyses

## Training Guide

### Enhanced CNN Training

#### Prerequisites
- PyTorch installed
- Real Wolffia images in `images/` directory
- At least 4GB RAM (8GB+ recommended)

#### Training Process
```bash
# Run enhanced training script
python train_wolffia_cnn.py

# Choose training mode:
# 1. Quick (8K samples, ~10 minutes)
# 2. Standard (15K samples, ~20 minutes)
# 3. Professional (25K samples, ~35 minutes)
# 4. Research-grade (40K samples, ~60 minutes)
```

#### Training Features
- **Realistic synthetic data**: Poisson disc sampling, lighting effects
- **Multi-task learning**: Mask + edge + distance prediction
- **Real image integration**: Uses backgrounds from your images
- **Automatic validation**: Early stopping and best model saving

### Tophat ML Training

#### Annotation Process
1. **Start annotation session**: Upload images for training
2. **Mark corrections**: Click cells to mark as correct/incorrect/missing
3. **Save annotations**: Training data is automatically stored
4. **Train model**: Run `python tophat_ml_trainer.py`

#### Annotation Guidelines
- **Correct cells**: Green circles around properly detected cells
- **False positives**: Red X on incorrectly detected objects
- **Missing cells**: Blue + on cells that were missed
- **Quality matters**: Focus on clear, unambiguous cases

#### Training Output
- **Random Forest model**: Classical ML based on features
- **Enhanced CNN model**: Deep learning (if PyTorch available)
- **Performance metrics**: Accuracy and classification reports

### Realistic Data Generation

#### Generate Samples
```bash
# Create realistic synthetic data
python realistic_wolffia_generator.py

# Output: realistic_samples/ directory with examples
```

#### Features Generated
- **Natural backgrounds**: Gradients, noise, real image patches
- **Wolffia morphology**: Blobby ellipses with internal structures
- **Lighting effects**: Shadows and highlights
- **Cell placement**: Poisson disc sampling prevents overlap

## Advanced Usage

### Method Selection
```python
# Force specific detection method
result = analyzer.analyze_image('image.jpg', use_cnn=True)
result = analyzer.analyze_image('image.jpg', use_tophat=True)
```

### Batch Processing
```python
import os
from pathlib import Path

# Process multiple images
image_dir = Path('my_images/')
results = []

for img_path in image_dir.glob('*.jpg'):
    result = analyzer.analyze_image(str(img_path))
    results.append(result)
```

### Custom Parameters
```python
# Initialize with custom settings
analyzer = WolffiaAnalyzer(
    min_cell_area=30,      # Smaller minimum cell size
    max_cell_area=1500,    # Larger maximum cell size
    cellpose_diameter=30   # Larger CellPose diameter
)
```

## API Reference

### Core Analysis
```python
# Main analysis method
result = analyzer.analyze_image(
    image_path,           # Path to image file
    use_cnn=False,  # Use enhanced CNN if available
    use_tophat=False,     # Use tophat model if available
    save_result=True      # Save result to results/ directory
)
```

### Tophat Training
```python
# Start training session
session_id = analyzer.start_tophat_training(image_paths)

# Save user annotations
analyzer.save_user_annotations(session_id, annotations)

# Train model from annotations
analyzer.train_tophat_model()
```

### Result Format
```python
{
    'total_cells': 42,           # Number of detected cells
    'total_area': 15680.5,       # Total area in pixels
    'average_area': 373.3,       # Average cell area
    'processing_time': 2.34,     # Analysis time in seconds
    'method_used': 'cnn', # Detection method
    'cells': [                   # Individual cell data
        {
            'id': 1,
            'center': [245, 178],
            'area': 385.2,
            'circularity': 0.87
        },
        # ... more cells
    ],
    'labeled_image_path': 'path/to/labeled.png'
}
```

## Troubleshooting

### Common Issues

#### No Cells Detected
- **Check image quality**: Ensure cells are visible and well-lit
- **Adjust parameters**: Try different detection methods
- **Train tophat model**: Create custom model for your images

#### Poor Detection Accuracy
- **Use enhanced CNN**: Train with realistic synthetic data
- **Create annotations**: Use tophat training for customization
- **Check image preprocessing**: Ensure proper contrast/brightness

#### Slow Performance
- **Use GPU**: Install PyTorch with CUDA support
- **Optimize image size**: Resize large images before analysis
- **Choose faster methods**: Tophat ML is faster than CNN

#### Memory Errors
- **Reduce batch size**: In CNN training scripts
- **Close other applications**: Free up system memory
- **Use smaller images**: Resize before processing

### Performance Optimization

#### For Speed
1. **Use tophat ML**: Train custom model for fastest results
2. **Resize images**: Process at lower resolution if acceptable
3. **Enable GPU**: For CNN-based methods

#### For Accuracy
1. **Train enhanced CNN**: With realistic synthetic data
2. **Create tophat annotations**: Custom model for your images
3. **Use method fusion**: Combine multiple detection approaches

### File Organization

#### Project Structure
```
bioimagin/
├── bioimaging.py              # Core analysis engine
├── wolffia_cnn.py    # Enhanced CNN implementation
├── realistic_wolffia_generator.py # Synthetic data generator
├── tophat_ml_trainer.py       # Tophat-based training
├── web_integration.py         # Web interface
├── train_wolffia_cnn.py # Training script
├── images/                    # Input images for training
├── models/                    # Trained models
├── results/                   # Analysis outputs
├── annotations/               # Tophat training data
└── docs/                      # Documentation
```

#### Data Management
- **Regular cleanup**: Remove old results and uploads
- **Model versioning**: Keep track of training iterations
- **Backup annotations**: Tophat training data is valuable

## Best Practices

### Image Preparation
- **Consistent lighting**: Uniform illumination across images
- **Good contrast**: Clear separation between cells and background
- **Appropriate resolution**: Balance between detail and processing speed

### Model Training
- **Quality over quantity**: Better to have fewer high-quality annotations
- **Diverse examples**: Include various lighting and cell conditions
- **Regular retraining**: Update models as you collect more data

### Analysis Workflow
1. **Start with defaults**: Use standard analysis first
2. **Evaluate results**: Check detection accuracy
3. **Train if needed**: Create custom models for better performance
4. **Validate outputs**: Spot-check results for quality assurance

This guide provides comprehensive coverage of BIOIMAGIN's capabilities and should help you achieve optimal results for your Wolffia cell analysis needs.