# BIOIMAGIN - Enhanced Wolffia Cell Detection System

<div align="center">

![BIOIMAGIN Logo](https://img.shields.io/badge/BIOIMAGIN-v3.0--Enhanced-2F855A?style=for-the-badge&logo=microscope)

**Professional-grade automated bioimage analysis with enhanced CNN and realistic synthetic data**

[![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange?style=flat-square&logo=pytorch)](https://pytorch.org)
[![Flask](https://img.shields.io/badge/Flask-2.0+-green?style=flat-square&logo=flask)](https://flask.palletsprojects.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-red?style=flat-square&logo=opencv)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

</div>

---

## 🔬 Overview

BIOIMAGIN is a cutting-edge bioimage analysis system specifically designed for *Wolffia arrhiza* (the world's smallest flowering plant) cell detection and analysis. The system combines state-of-the-art deep learning, realistic synthetic data generation, and user-guided training to achieve CellPose-level precision with custom adaptability.

**Project Context**: Part of "BinAqua: Klimafreundliche Herstellung vollwertiger veganer Proteinpulver durch die Co-Kultivierung von Mikroalgen und Wasserlinsen" at BTU Cottbus-Senftenberg.

### ✨ Key Features

- **🧠 Enhanced CNN Architecture**: Multi-task U-Net with mask, edge, and distance prediction
- **🎨 Realistic Synthetic Data**: Poisson disc sampling with natural lighting and cell morphology
- **🎯 Tophat ML Training**: User annotation-based custom model training
- **⚡ Smart Detection Pipeline**: Priority-based method selection for optimal results
- **🌐 Web Interface**: Intuitive drag-and-drop analysis with real-time progress
- **📊 Comprehensive Outputs**: Cell counts, areas, labeled visualizations, and export options

---

## 🚀 New in Version 3.0

### Enhanced CNN System
- **Multi-task Learning**: Simultaneous mask, edge, and distance transform prediction
- **CellPose-inspired Architecture**: U-Net with skip connections and feature pyramids
- **Watershed Post-processing**: Distance transform-based precise cell separation
- **Real Image Integration**: Uses backgrounds from your actual images for training

### Realistic Synthetic Data Generator
- **Natural Cell Placement**: Poisson disc sampling prevents unrealistic overlap
- **Authentic Morphology**: Blobby ellipses with internal chloroplast structures
- **Dynamic Lighting**: Shadows, highlights, and brightness gradients
- **Background Variety**: Soft gradients, noise textures, and real image patches

### Tophat-Based Training
- **User Feedback Integration**: Learn from your annotation corrections
- **Quality-Aware Training**: Balanced datasets with confidence weighting
- **Continuous Improvement**: Models update with new annotation sessions
- **Classical + Deep Learning**: Both Random Forest and enhanced CNN training

### Smart Detection Pipeline
```
Enhanced CNN → Regular CNN → CellPose → Tophat ML → Watershed
(Priority-based automatic selection)
```

---

## 📋 System Requirements

### Minimum Requirements
- **Python**: 3.8 or higher
- **RAM**: 8GB (16GB recommended for training)
- **Storage**: 2GB free space
- **OS**: Windows 10+, macOS 10.15+, Linux (Ubuntu 18.04+)

### Optional for Enhanced Performance
- **GPU**: CUDA-compatible for enhanced CNN training
- **PyTorch**: For deep learning features
- **CellPose**: For advanced baseline comparison

---

## 🛠️ Installation

### 1. Clone Repository
```bash
git clone https://github.com/your-org/bioimagin.git
cd bioimagin
```

### 2. Install Dependencies
```bash
# Core dependencies
pip install -r requirements.txt

# Optional: Enhanced CNN features
pip install torch torchvision

# Optional: CellPose integration
pip install cellpose>=3.0.0
```

### 3. Verify Installation
```bash
python -c "from bioimaging import WolffiaAnalyzer; print('✅ System ready')"
```

### 4. Launch Web Interface
```bash
python web_integration.py
```
Open browser to `http://localhost:5000`

---

## 📖 Quick Start

### Web Interface (Recommended)
1. **Launch**: `python web_integration.py`
2. **Upload**: Drag and drop your Wolffia images
3. **Analyze**: Click "Start Analysis" with default or custom settings
4. **Results**: View cell counts, areas, and labeled images
5. **Export**: Download CSV, JSON, or ZIP packages

### Python API
```python
from bioimaging import WolffiaAnalyzer

# Initialize analyzer
analyzer = WolffiaAnalyzer()

# Analyze single image
result = analyzer.analyze_image('wolffia_image.jpg')
print(f"Detected {result['total_cells']} cells")

# Use enhanced CNN (if trained)
result = analyzer.analyze_image('image.jpg', use_cnn=True)

# Use custom tophat model (if trained)
result = analyzer.analyze_image('image.jpg', use_tophat=True)
```

### Training Enhanced Models

#### Enhanced CNN Training
```bash
# Interactive training script
python train_wolffia_cnn.py

# Choose from:
# 1. Quick (8K samples, ~10 min)
# 2. Standard (15K samples, ~20 min)  
# 3. Professional (25K samples, ~35 min)
# 4. Research-grade (40K samples, ~60 min)
```

#### Tophat Model Training
1. **Web Interface**: Start tophat training session
2. **Annotate**: Mark correct/incorrect/missing cells
3. **Train**: Run `python tophat_ml_trainer.py`
4. **Use**: Models automatically available in analysis

---

## 🏗️ Architecture

### Core Components

#### 1. Enhanced CNN (`wolffia_cnn.py`)
- **Multi-task U-Net**: Simultaneous mask, edge, distance prediction
- **Skip Connections**: Preserve fine details through decoder
- **Realistic Training**: Uses synthetic data with natural characteristics

#### 2. Realistic Data Generator (`realistic_wolffia_generator.py`)
- **Poisson Disc Sampling**: Natural cell placement without overlap
- **Wolffia Morphology**: Blobby shapes with internal structures
- **Lighting Simulation**: Shadows, highlights, gradients
- **Background Integration**: Real image patches for authenticity

#### 3. Tophat ML Trainer (`tophat_ml_trainer.py`)
- **Annotation Analysis**: Extracts training data from user feedback
- **Balanced Datasets**: Equal positive/negative samples
- **Dual Training**: Random Forest + Enhanced CNN options

#### 4. Main Analysis Engine (`bioimaging.py`)
- **Smart Detection**: Priority-based method selection
- **Result Fusion**: Intelligent combination of multiple methods
- **Adaptive Preprocessing**: Quality-based image enhancement

#### 5. Web Interface (`web_integration.py`)
- **Real-time Analysis**: Background processing with progress updates
- **Interactive Training**: Tophat annotation interface
- **Export Options**: Multiple formats and visualization

### Detection Pipeline Flow
```
Input Image
     ↓
Smart Preprocessing
     ↓
Priority-Based Detection:
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
Output Results
```

---

## 📊 Key Capabilities

### Detection Methods
- **Enhanced CNN**: Multi-task deep learning with precise edge detection
- **Regular CNN**: Fast classification-based detection
- **CellPose**: Pre-trained general cell segmentation
- **Tophat ML**: Custom models trained on your annotations
- **Watershed**: Classical morphological approach

### Data Generation
- **Realistic Synthetic Data**: Natural-looking training samples
- **Poisson Disc Sampling**: Prevents unrealistic cell overlap
- **Lighting Effects**: Shadows, highlights, brightness variation
- **Morphological Accuracy**: Wolffia-specific cell shapes

### Training Systems
- **Enhanced CNN Training**: Multi-task learning with realistic data
- **Tophat ML Training**: User annotation-based custom models
- **Continuous Learning**: Models improve with new annotations
- **Quality Assessment**: Confidence scoring for all methods

### Analysis Features
- **Cell Counting**: Accurate detection and enumeration
- **Area Measurement**: Individual and total cell areas
- **Visualization**: Labeled images with numbered cells
- **Export Options**: CSV (cell data), JSON (full results), ZIP (complete package)

---

## 📈 Performance

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

---

## 📚 Documentation

### Comprehensive Guides
- **[Architecture Overview](docs/ARCHITECTURE.md)**: System design and components
- **[Usage Guide](docs/USAGE_GUIDE.md)**: Detailed usage instructions and examples
- **[API Reference](docs/API_REFERENCE.md)**: Complete API documentation

### Core Workflows
- **Basic Analysis**: Upload → Analyze → Export
- **Enhanced CNN Training**: Data generation → Model training → Deployment
- **Tophat Training**: Annotation → Training → Custom model usage
- **Batch Processing**: Multiple image analysis workflows

---

## 🔧 Configuration

### Analysis Parameters
```python
analyzer = WolffiaAnalyzer(
    min_cell_area=15,        # Minimum cell area (pixels)
    max_cell_area=1200,      # Maximum cell area (pixels)
    cellpose_diameter=25,    # CellPose size parameter
    enhance_contrast=True    # Automatic preprocessing
)
```

### Training Configuration
- **Enhanced CNN**: Epochs, learning rate, batch size
- **Synthetic Data**: Cell count, size range, lighting parameters
- **Tophat ML**: Feature extraction, model parameters

---

## 📄 Output Formats

### Analysis Results
```python
{
    'total_cells': 42,
    'total_area': 15680.5,
    'average_area': 373.3,
    'processing_time': 2.34,
    'method_used': 'cnn',
    'cells': [
        {
            'id': 1,
            'center': [245, 178],
            'area': 385.2,
            'circularity': 0.87
        }
        # ... more cells
    ],
    'labeled_image_path': 'results/labeled_image.png'
}
```

### Export Options
- **CSV**: Individual cell measurements
- **JSON**: Complete analysis results with metadata
- **ZIP**: Labeled images + data + summary reports

---

## 🤝 Contributing

We welcome contributions! Areas for improvement:
- **Algorithm enhancements**: New detection methods
- **Training data**: Additional realistic generation features
- **Performance**: Speed and memory optimizations
- **Documentation**: Usage examples and tutorials

### Development Setup
```bash
# Install development dependencies
pip install -r requirements.txt

# Run tests
python test_realistic_generator.py

# Format code
black *.py
```

---

## 📚 Citation

If you use BIOIMAGIN in your research, please cite:

```bibtex
@software{bioimagin2025,
  title={BIOIMAGIN: Enhanced Wolffia Cell Detection with CNN and Synthetic Data},
  author={BIOIMAGIN Development Team},
  version={3.0-Enhanced},
  year={2025},
  url={https://github.com/your-org/bioimagin}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🆘 Support

### Getting Help
- **Documentation**: Check `docs/` directory for comprehensive guides
- **Issues**: [GitHub Issues](https://github.com/your-org/bioimagin/issues)
- **Questions**: Create a discussion or issue

### Common Solutions
- **Installation Issues**: Check Python version and dependencies
- **Performance**: Enable GPU support for enhanced CNN features
- **Accuracy**: Train custom tophat models for your specific images

---

## 🙏 Acknowledgments

- **PyTorch Team** for deep learning framework
- **OpenCV Community** for computer vision tools
- **CellPose Developers** for inspiration and baseline methods
- **Scientific Community** for validation and feedback

---

<div align="center">

**BIOIMAGIN - Precision Cell Detection Through AI Enhancement**

![Footer](https://img.shields.io/badge/Made%20with-🧠%20AI%20%26%20🔬%20Science-green?style=for-the-badge)

</div>