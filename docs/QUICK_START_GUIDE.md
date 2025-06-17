# BIOIMAGIN - Quick Start Guide

## 🚀 Get Started in 5 Minutes

### Prerequisites
- Python 3.8+
- 8GB RAM (16GB recommended)
- Windows 10+, macOS 10.15+, or Linux

### 1. Installation

```bash
# Clone repository
git clone https://github.com/AuroragitAA/bioimagin.git
cd bioimagin

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "from bioimaging import WolffiaAnalyzer; print('✅ Installation successful')"
```

### 2. Launch Web Interface

```bash
python web_integration.py
```

Open browser to: `http://localhost:5000`

### 3. Analyze Your First Image

1. **Upload**: Drag and drop Wolffia microscopy images
2. **Configure**: Select detection methods:
   - ✅ Watershed (always available)
   - ✅ Tophat ML (if trained)
   - ✅ Enhanced CNN (if available)
3. **Analyze**: Click "Start Analysis"
4. **Results**: View cell counts, areas, and labeled images
5. **Export**: Download CSV, JSON, or ZIP packages

### 4. Train Custom Models (Optional)

#### Quick CNN Training
```bash
python train_wolffia_cnn.py
# Choose: Quick (10 min) → Standard (20 min) → Professional (35 min)
```

#### Interactive Tophat Training
1. Web Interface → "Tophat Training"
2. Upload 5-10 representative images
3. Mark correct/incorrect/missed cells
4. Train model with your annotations

### 5. Python API Usage

```python
from bioimaging import WolffiaAnalyzer

# Initialize analyzer
analyzer = WolffiaAnalyzer()

# Analyze single image
result = analyzer.analyze_image('wolffia_sample.jpg')
print(f"Detected {result['total_cells']} cells")
print(f"Total area: {result['total_area']:.1f} pixels")

# Use specific methods
result = analyzer.analyze_image(
    'image.jpg',
    use_tophat=True,    # Custom trained model
    use_cnn=True,       # CNN detection
    use_celldetection=False  # Skip if not needed
)

# Access individual cell data
for cell in result['cells']:
    print(f"Cell {cell['id']}: area={cell['area']:.1f}, center={cell['center']}")
```

## 🔧 Configuration Options

### Analysis Parameters
```python
analyzer = WolffiaAnalyzer(
    min_cell_area=15,        # Minimum cell size
    max_cell_area=1200,      # Maximum cell size
    cellpose_diameter=25     # CellPose parameter
)
```

### Method Selection
- **Watershed**: Fast, reliable baseline (always available)
- **Enhanced CNN**: Highest accuracy, requires training
- **Tophat ML**: Customizable, learns from your annotations
- **CellDetection**: General-purpose, good for diverse samples

## 📊 Understanding Results

### Output Structure
```json
{
    "total_cells": 42,
    "total_area": 15680.5,
    "average_area": 373.3,
    "processing_time": 2.34,
    "method_used": ["watershed", "cnn"],
    "cells": [
        {
            "id": 1,
            "center": [245, 178],
            "area": 385.2,
            "circularity": 0.87,
            "eccentricity": 0.23
        }
    ]
}
```

### Key Metrics
- **total_cells**: Number of detected Wolffia cells
- **total_area**: Sum of all cell areas (pixels)
- **average_area**: Mean cell area for population analysis
- **processing_time**: Analysis duration in seconds

## ⚡ Performance Tips

### For Speed
- Use watershed only: `use_cnn=False, use_tophat=False`
- Process smaller images: resize to 1024×1024
- Disable detailed visualizations

### For Accuracy
- Enable all methods: `use_cnn=True, use_tophat=True`
- Train custom tophat model with your images
- Use multi-method fusion results

### For Batch Processing
- Use GPU acceleration if available
- Process 10-20 images per batch
- Enable result caching

## 🔍 Troubleshooting

### Common Issues

**"Module not found" errors**:
```bash
pip install -r requirements.txt --upgrade
```

**Images not displaying in training**:
- System automatically converts TIFF to PNG
- Check browser console for errors

**Poor detection accuracy**:
- Train custom tophat model with your images
- Adjust cell size parameters
- Check image quality and contrast

**Slow performance**:
- Install PyTorch for GPU acceleration
- Reduce image resolution
- Use fewer detection methods

### Getting Help
- Check full documentation: `docs/COMPREHENSIVE_GUIDE.md`
- API reference: `docs/API_REFERENCE.md`
- Architecture details: `docs/ARCHITECTURE.md`
- Create GitHub issue for bugs

## 📈 Next Steps

1. **Explore Advanced Features**:
   - Multi-method comparison
   - Batch processing workflows
   - Debug visualizations

2. **Customize for Your Data**:
   - Train tophat models
   - Adjust detection parameters
   - Create custom analysis pipelines

3. **Integrate with Your Workflow**:
   - Use Python API in scripts
   - Export results to external tools
   - Set up automated processing

---

**Ready to analyze Wolffia cells with precision and speed!** 🔬✨