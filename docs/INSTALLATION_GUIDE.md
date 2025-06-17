# BIOIMAGIN - Installation Guide

## System Requirements

### Minimum Requirements
- **Operating System**: Windows 10+, macOS 10.15+, Ubuntu 18.04+
- **Python**: 3.8 or higher
- **RAM**: 8GB (16GB recommended for training)
- **Storage**: 2GB free space
- **Network**: Internet connection for package downloads

### Recommended for Enhanced Performance
- **GPU**: CUDA-compatible (RTX 2070+) for CNN acceleration
- **RAM**: 16GB+ for large batch processing
- **CPU**: Multi-core processor (Intel i7+ or AMD Ryzen 7+)
- **Storage**: SSD for faster I/O operations

## Installation Methods

### Method 1: Standard Installation (Recommended)

#### Step 1: Clone Repository
```bash
git clone https://github.com/AuroragitAA/bioimagin.git
cd bioimagin
```

#### Step 2: Create Virtual Environment (Recommended)
```bash
# Using venv
python -m venv bioimagin_env

# Activate environment
# Windows:
bioimagin_env\Scripts\activate
# macOS/Linux:
source bioimagin_env/bin/activate
```

#### Step 3: Install Core Dependencies
```bash
pip install -r requirements.txt
```

#### Step 4: Install Optional Enhancements
```bash
# For CNN features (highly recommended)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CellPose integration (optional)
pip install cellpose>=3.0.0

# For additional scientific packages (optional)
pip install SimpleITK scikit-learn pandas matplotlib seaborn
```

#### Step 5: Verify Installation
```bash
python -c "from bioimaging import WolffiaAnalyzer; print('✅ BIOIMAGIN installed successfully')"
```

### Method 2: Quick Development Setup

For developers who want to get started immediately:

```bash
# Clone and install in one command
git clone https://github.com/AuroragitAA/bioimagin.git && cd bioimagin && pip install -r requirements.txt

# Quick verification
python web_integration.py
```

### Method 3: Docker Installation (Experimental)

```bash
# Build Docker image
docker build -t bioimagin:latest .

# Run container
docker run -p 5000:5000 -v $(pwd)/uploads:/app/uploads -v $(pwd)/results:/app/results bioimagin:latest
```

## Platform-Specific Instructions

### Windows 10/11

#### Prerequisites
1. Install Python 3.8+ from [python.org](https://python.org)
2. Install Git from [git-scm.com](https://git-scm.com)
3. Install Microsoft Visual C++ Redistributable

#### Installation
```cmd
# Open Command Prompt or PowerShell as Administrator
git clone https://github.com/AuroragitAA/bioimagin.git
cd bioimagin
pip install -r requirements.txt

# For GPU support (optional)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### Common Windows Issues
- **Path Issues**: Ensure Python and pip are in system PATH
- **Permission Errors**: Run as Administrator or use `--user` flag
- **Visual Studio Build Tools**: Install if compilation errors occur

### macOS (Intel/Apple Silicon)

#### Prerequisites
```bash
# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python and Git
brew install python@3.11 git
```

#### Installation
```bash
# Clone repository
git clone https://github.com/AuroragitAA/bioimagin.git
cd bioimagin

# Create virtual environment
python3 -m venv bioimagin_env
source bioimagin_env/bin/activate

# Install dependencies
pip install -r requirements.txt

# For Apple Silicon Macs, use conda for PyTorch
# conda install pytorch torchvision -c pytorch
```

### Linux (Ubuntu/Debian)

#### Prerequisites
```bash
# Update package list
sudo apt update

# Install Python, pip, and development tools
sudo apt install python3.11 python3.11-pip python3.11-venv git build-essential

# Install system dependencies for image processing
sudo apt install libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6 libfontconfig1
```

#### Installation
```bash
# Clone repository
git clone https://github.com/AuroragitAA/bioimagin.git
cd bioimagin

# Create virtual environment
python3 -m venv bioimagin_env
source bioimagin_env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

## Dependency Details

### Core Dependencies (Always Required)
```
opencv-python>=4.8.0
numpy>=1.21.0
scipy>=1.7.0
scikit-image>=0.19.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
pandas>=1.3.0
flask>=2.0.0
flask-cors>=3.0.0
pillow>=8.3.0
```

### Enhanced Features (Optional)
```
torch>=2.0.0              # CNN models and training
torchvision>=0.15.0        # Computer vision utilities
cellpose>=3.0.0            # Professional cell segmentation
SimpleITK>=2.2.0           # Advanced image processing
```

### Development Dependencies (For Contributors)
```
black>=22.0.0              # Code formatting
flake8>=5.0.0              # Code linting
pytest>=7.0.0              # Testing framework
pytest-cov>=4.0.0          # Coverage reporting
sphinx>=5.0.0              # Documentation generation
```

## GPU Setup (Optional but Recommended)

### NVIDIA GPU Support

#### 1. Install NVIDIA Drivers
- Download from [NVIDIA Driver Downloads](https://www.nvidia.com/drivers/)
- Ensure version 520+ for modern GPUs

#### 2. Install CUDA Toolkit
```bash
# For CUDA 11.8 (recommended)
# Windows: Download from NVIDIA website
# Linux:
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
sudo apt update
sudo apt install cuda-toolkit-11-8
```

#### 3. Install PyTorch with CUDA
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### 4. Verify GPU Support
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

## Verification and Testing

### System Health Check
```bash
# Launch web interface
python web_integration.py

# In another terminal, test API
curl http://localhost:5000/api/health
```

### Feature Verification
```python
from bioimaging import WolffiaAnalyzer
import json

analyzer = WolffiaAnalyzer()

# Check available features
features = {
    "watershed": True,  # Always available
    "cnn": analyzer.wolffia_cnn_available,
    "tophat": analyzer.tophat_model is not None,
    "celldetection": analyzer.celldetection_available
}

print("Available Features:")
print(json.dumps(features, indent=2))
```

### Performance Test
```python
import time
from bioimaging import WolffiaAnalyzer

analyzer = WolffiaAnalyzer()

# Create test image
import numpy as np
test_image = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)

# Time analysis
start_time = time.time()
# result = analyzer.analyze_image(test_image)  # Use with real image
end_time = time.time()

print(f"Analysis completed in {end_time - start_time:.2f} seconds")
```

## Troubleshooting Installation Issues

### Common Problems and Solutions

#### 1. Package Installation Failures
```bash
# Upgrade pip and setuptools
pip install --upgrade pip setuptools wheel

# Install with no cache
pip install --no-cache-dir -r requirements.txt

# Install specific problem packages individually
pip install opencv-python scikit-image scipy
```

#### 2. OpenCV Import Errors
```bash
# Uninstall conflicting packages
pip uninstall opencv-python opencv-contrib-python opencv-python-headless

# Reinstall clean version
pip install opencv-python==4.8.1.78
```

#### 3. PyTorch Installation Issues
```bash
# For CPU-only installation
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# For specific CUDA version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

#### 4. Permission Errors
```bash
# Install to user directory
pip install --user -r requirements.txt

# Or use virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

#### 5. Memory Issues During Installation
```bash
# Increase pip timeout and use no cache
pip install --timeout 300 --no-cache-dir -r requirements.txt

# Install packages one by one
pip install numpy scipy scikit-image opencv-python matplotlib pandas flask
```

### Environment Variables

Set these if needed:
```bash
# For custom model paths
export BIOIMAGIN_MODEL_PATH="/path/to/models"

# For CUDA memory management
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"

# For OpenCV threading
export OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS=0
```

## Development Setup

For contributors and advanced users:

```bash
# Clone with development branch
git clone -b develop https://github.com/AuroragitAA/bioimagin.git
cd bioimagin

# Install in development mode
pip install -e .

# Install development dependencies
pip install -r requirements-dev.txt

# Set up pre-commit hooks
pre-commit install

# Run tests
python -m pytest tests/

# Generate documentation
cd docs && make html
```

## Uninstallation

To completely remove BIOIMAGIN:

```bash
# If using virtual environment
deactivate
rm -rf bioimagin_env

# Remove repository
rm -rf bioimagin

# Clean pip cache
pip cache purge
```

---

## Next Steps

After successful installation:

1. 📖 Read the [Quick Start Guide](QUICK_START_GUIDE.md)
2. 🔬 Try the [Tutorial Examples](TUTORIAL.md)
3. 📚 Explore the [API Reference](API_REFERENCE.md)
4. 🎯 Train models with your data using the [Training Guide](TRAINING_GUIDE.md)

**Installation complete! Ready to analyze Wolffia cells! 🔬✨**