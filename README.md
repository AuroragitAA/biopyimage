# 🌱 Wolffia Bioimage Analysis System

A professional, web-based bioimage analysis system for automated Wolffia cell counting, morphological analysis, and biomass estimation.

## ✨ Features

- **🔍 Automated Cell Detection** - Advanced segmentation algorithms for accurate cell identification
- **📊 Morphological Analysis** - Comprehensive cell size, shape, and chlorophyll content analysis  
- **🎨 Multiple Segmentation Methods** - Watershed, adaptive thresholding, and color-based segmentation
- **📈 Real-time Statistics** - Live calculation of cell counts, biomass estimates, and chlorophyll ratios
- **🖥️ Web Interface** - Professional, responsive web dashboard
- **📁 Batch Processing** - Analyze multiple images simultaneously
- **💾 Data Export** - Export results to CSV format
- **📚 Analysis History** - Track and review previous analyses

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Web Interface                      │
│                 (HTML + JavaScript)                 │
├─────────────────────────────────────────────────────┤
│                  Flask Backend                      │
│                    (app.py)                         │
├─────────────────────────────────────────────────────┤
│               Analysis Pipeline                     │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐   │
│  │Image        │ │Segmentation │ │Feature      │   │
│  │Processor    │ │Engine       │ │Extraction   │   │
│  └─────────────┘ └─────────────┘ └─────────────┘   │
├─────────────────────────────────────────────────────┤
│              Core Libraries                         │
│         OpenCV • scikit-image • NumPy              │
└─────────────────────────────────────────────────────┘
```

## 📋 Requirements

### System Requirements
- Python 3.8 or higher
- 4GB RAM minimum (8GB recommended)
- 1GB free disk space

### Python Dependencies
```
Flask==2.3.3
opencv-python==4.8.1.78
numpy==1.24.3
scipy==1.11.4
scikit-image==0.21.0
matplotlib==3.7.2
pandas==2.0.3
Pillow==10.0.1
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or download the project
git clone <repository-url>
cd wolffia-analysis

# Install dependencies
pip install -r requirements.txt
```

### 2. Automated Setup

```bash
# Run the setup script (recommended)
python setup_and_run.py
```

### 3. Manual Setup

```bash
# Create required directories
mkdir temp_uploads templates static results logs

# Copy your files to the correct locations
# - app.py (main Flask application)
# - image_processor.py (image preprocessing)
# - segmentation.py (cell segmentation)
# - wolffia_analyzer.py (main analyzer)
# - templates/index.html (web interface)
# - static/main.js (frontend JavaScript)

# Start the server
python app.py
```

### 4. Access the Application

Open your web browser and navigate to:
```
http://localhost:5000
```

## 📖 Usage Guide

### Basic Analysis Workflow

1. **Upload Images**
   - Drag and drop images or click to browse
   - Supported formats: JPG, PNG, TIF, BMP
   - Maximum file size: 16MB per image

2. **Configure Analysis**
   - Choose segmentation method (Auto recommended)
   - Adjust parameters if needed:
     - Pixel to micron ratio
     - Chlorophyll threshold
     - Cell size limits

3. **Run Analysis**
   - Single image: Click "Analyze Image"
   - Multiple images: Click "Batch Analysis"

4. **Review Results**
   - Statistics tab: Cell counts and measurements
   - Visualization tab: Original vs segmented images
   - Details tab: Individual cell data
   - History tab: Previous analyses

5. **Export Data**
   - Click "Export Results" to download CSV

### Analysis Methods

#### Auto (Recommended)
Automatically selects the best segmentation method based on image characteristics.

#### Watershed
Ideal for images with overlapping or touching cells. Uses distance transform and local max detection.

#### Threshold  
Simple binary thresholding with morphological cleaning. Good for high-contrast images.

#### Adaptive
Adaptive thresholding for images with varying illumination conditions.

#### Color-based
Segments cells based on color profiles:
- **Green Wolffia**: Standard green organism detection
- **Bright Green**: For well-illuminated samples  
- **Dark Green**: For shadowed or darker samples

## 🔧 Configuration

### Advanced Parameters

- **Pixel to Micron Ratio**: Conversion factor for accurate size measurements
- **Chlorophyll Threshold**: Threshold for high chlorophyll classification (0-1)
- **Min Cell Area**: Minimum cell size in pixels for detection
- **Max Cell Area**: Maximum cell size in pixels to exclude debris

### File Structure

```
wolffia-analysis/
├── app.py                    # Main Flask application
├── image_processor.py        # Image preprocessing module
├── segmentation.py          # Cell segmentation algorithms
├── wolffia_analyzer.py      # Main analysis orchestrator
├── requirements.txt         # Python dependencies
├── setup_and_run.py        # Setup and run script
├── README.md               # This documentation
├── templates/
│   └── index.html          # Web interface template
├── static/
│   └── main.js            # Frontend JavaScript
├── temp_uploads/          # Temporary file storage
├── results/               # Analysis results
└── logs/                  # Application logs
```

## 🧪 API Endpoints

### Health Check
```
GET /health
```
Returns system component status.

### Image Analysis
```
POST /analyze
Content-Type: multipart/form-data

Parameters:
- image: Image file
- analysis_method: Segmentation method
- pixel_ratio: Pixel to micron conversion
- chlorophyll_threshold: Chlorophyll threshold
- min_cell_area: Minimum cell area
- max_cell_area: Maximum cell area
```

### Export Results
```
POST /export
Content-Type: application/json

Body: Analysis results data
```

## 📊 Output Data

### Statistics
- **Total Cells**: Number of detected cells
- **Average Area**: Mean cell area in pixels/μm²
- **Biomass Estimate**: Estimated total biomass
- **Chlorophyll Ratio**: Percentage of high-chlorophyll cells

### Cell Data (CSV Export)
```csv
analysis_timestamp,image_path,cell_id,area,perimeter,chlorophyll,classification,centroid_x,centroid_y
2024-01-15_14:30:22,sample1.jpg,1,245.6,62.3,0.73,healthy,123.4,456.7
```

## 🛠️ Troubleshooting

### Common Issues

#### Import Errors
```bash
# Ensure all dependencies are installed
pip install -r requirements.txt

# Check Python version
python --version  # Should be 3.8+
```

#### Memory Issues
- Reduce image size before analysis
- Process images in smaller batches
- Increase system RAM if possible

#### No Cells Detected
- Check image quality and contrast
- Adjust segmentation method
- Modify cell size parameters
- Ensure proper lighting in images

#### Server Won't Start
```bash
# Check if port 5000 is available
netstat -an | grep 5000

# Try different port
python app.py --port 5001
```

### Debug Mode

Enable debug logging by setting environment variable:
```bash
export FLASK_DEBUG=1
python app.py
```

## 🔬 Scientific Background

### Wolffia Analysis
Wolffia is the world's smallest flowering plant, making accurate quantification challenging. This system addresses key measurement needs:

- **Cell Counting**: Automated detection reduces manual counting errors
- **Size Distribution**: Quantifies population heterogeneity  
- **Chlorophyll Content**: Assesses photosynthetic capacity
- **Growth Monitoring**: Tracks population changes over time

### Segmentation Algorithms

1. **Watershed**: Separates touching objects using topological approach
2. **Adaptive Thresholding**: Handles varying illumination conditions
3. **Color Segmentation**: Leverages chlorophyll's green pigmentation
4. **Morphological Operations**: Cleans noise and connects fragmented cells

## 👥 Contributing

### Development Setup
```bash
# Install development dependencies
pip install -r requirements.txt

# Run tests
python -m pytest tests/

# Code formatting
black *.py

# Linting
flake8 *.py
```

### Adding New Features
1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Submit pull request

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## 📞 Support

For issues and questions:
- Check the troubleshooting section above
- Review system logs in `logs/` directory  
- Open an issue on the project repository

## 🔄 Version History

### v1.0.0 (Current)
- Initial release
- Basic cell detection and analysis
- Web interface
- Batch processing
- CSV export

### Planned Features
- Advanced morphological measurements
- Time-series analysis  
- Machine learning classification
- Multi-species support
- Cloud deployment options

---

*Built with ❤️ for the scientific community*