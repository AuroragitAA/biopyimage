# BIOIMAGIN API Reference

## Web API Endpoints

### Core Analysis Endpoints

#### Upload Files
```http
POST /api/upload
Content-Type: multipart/form-data
```

**Parameters:**
- `files`: Multiple image files (PNG, JPG, JPEG, BMP, TIFF, JFIF)
- `max_size`: 50MB per file

**Response:**
```json
{
  "status": "success",
  "files": [
    {
      "file_id": "uuid-string",
      "filename": "image.jpg",
      "size": 1234567
    }
  ]
}
```

#### Start Analysis
```http
POST /api/analyze/<file_id>
Content-Type: application/json
```

**Parameters:**
```json
{
  "use_cnn": false,
  "use_tophat": false,
  "save_result": true
}
```

**Response:**
```json
{
  "status": "started",
  "analysis_id": "uuid-string",
  "estimated_time": 5.2
}
```

#### Get Analysis Status
```http
GET /api/status/<analysis_id>
```

**Response (In Progress):**
```json
{
  "status": "processing",
  "progress": 65,
  "stage": "cell_detection",
  "estimated_remaining": 2.1
}
```

**Response (Completed):**
```json
{
  "status": "completed",
  "result": {
    "total_cells": 42,
    "total_area": 15680.5,
    "average_area": 373.3,
    "processing_time": 2.34,
    "method_used": "cnn",
    "cells": [...],
    "labeled_image_path": "results/analysis_id_labeled.png"
  }
}
```

#### Export Results
```http
GET /api/export/<analysis_id>/<format>
```

**Formats:**
- `csv`: Cell data in CSV format
- `json`: Complete results in JSON format
- `zip`: All files in ZIP archive

**Response:**
- File download with appropriate Content-Type

### Tophat Training Endpoints

#### Start Training Session
```http
POST /api/tophat/start_training
Content-Type: application/json
```

**Parameters:**
```json
{
  "image_files": ["file_id_1", "file_id_2"],
  "detection_method": "watershed"
}
```

**Response:**
```json
{
  "status": "started",
  "session_id": "uuid-string",
  "initial_detections": [
    {
      "image_id": "file_id_1",
      "detections": [
        {
          "id": 1,
          "center": [245, 178],
          "area": 385.2,
          "confidence": 0.85
        }
      ]
    }
  ]
}
```

#### Save Annotations
```http
POST /api/tophat/save_annotations
Content-Type: application/json
```

**Parameters:**
```json
{
  "session_id": "uuid-string",
  "annotations": [
    {
      "detection_id": 1,
      "type": "correct_cell",
      "user_feedback": "good_detection"
    },
    {
      "detection_id": 2,
      "type": "false_positive",
      "user_feedback": "not_a_cell"
    }
  ]
}
```

**Response:**
```json
{
  "status": "saved",
  "annotations_count": 25,
  "session_progress": 0.6
}
```

#### Train Model
```http
POST /api/tophat/train_model
Content-Type: application/json
```

**Parameters:**
```json
{
  "session_ids": ["session_1", "session_2"],
  "model_type": "random_forest"
}
```

**Response:**
```json
{
  "status": "training_started",
  "training_id": "uuid-string",
  "estimated_time": 30.0
}
```

#### Get Model Status
```http
GET /api/tophat/model_status
```

**Response:**
```json
{
  "model_available": true,
  "model_info": {
    "type": "random_forest",
    "accuracy": 0.92,
    "training_samples": 150,
    "last_updated": "2025-06-13T12:34:56Z"
  }
}
```

### System Status Endpoints

#### Health Check
```http
GET /api/health
```

**Response:**
```json
{
  "status": "healthy",
  "version": "2.0.0",
  "capabilities": {
    "cnn": true,
    "regular_cnn": true,
    "cellpose": false,
    "tophat_ml": true,
    "gpu_available": true
  },
  "system_info": {
    "python_version": "3.9.7",
    "memory_usage": "2.1GB",
    "disk_space": "45.2GB"
  }
}
```

## Python API Reference

### WolffiaAnalyzer Class

#### Initialization
```python
from bioimaging import WolffiaAnalyzer

analyzer = WolffiaAnalyzer(
    min_cell_area=50,        # Minimum cell area in pixels
    max_cell_area=1200,      # Maximum cell area in pixels
    cellpose_diameter=25,    # CellPose diameter parameter
    enhance_contrast=True,   # Enable automatic contrast enhancement
    gpu=True                # Use GPU if available
)
```

#### Core Methods

##### analyze_image()
```python
result = analyzer.analyze_image(
    image_path: str,           # Path to image file
    use_cnn: bool = False,  # Use enhanced CNN
    use_tophat: bool = False,  # Use tophat model
    save_result: bool = True   # Save to results directory
) -> dict
```

**Returns:**
```python
{
    'total_cells': int,
    'total_area': float,
    'average_area': float,
    'processing_time': float,
    'method_used': str,
    'confidence': float,
    'cells': [
        {
            'id': int,
            'center': [x, y],
            'area': float,
            'circularity': float,
            'intensity': float,
            'green_intensity': float
        }
    ],
    'labeled_image_path': str,
    'analysis_id': str
}
```

##### smart_detect_cells()
```python
detections = analyzer.smart_detect_cells(
    image: np.ndarray,         # Input image array
    method_priority: list = None  # Override method priority
) -> list
```

##### wolffia_cnn_detection()
```python
detections = analyzer.wolffia_cnn_detection(
    image: np.ndarray          # Input image array
) -> list
```

#### Tophat Training Methods

##### start_tophat_training()
```python
session_id = analyzer.start_tophat_training(
    image_paths: list,         # List of image file paths
    detection_method: str = 'watershed'  # Initial detection method
) -> str
```

##### save_user_annotations()
```python
analyzer.save_user_annotations(
    session_id: str,           # Training session ID
    annotations: list          # User annotation data
) -> bool
```

##### train_tophat_model()
```python
model_info = analyzer.train_tophat_model(
    session_ids: list = None,  # Specific sessions to use
    model_type: str = 'random_forest'  # Model type
) -> dict
```

### Enhanced CNN Classes

#### EnhancedWolffiaCNN
```python
from wolffia_cnn import EnhancedWolffiaCNN

model = EnhancedWolffiaCNN(
    input_channels=3,          # RGB input
    base_filters=32           # Base filter count
)

# Forward pass
outputs = model(input_tensor)
# Returns: {'mask': tensor, 'edge': tensor, 'distance': tensor}
```

#### EnhancedWolffiaCNNTrainer
```python
from wolffia_cnn import EnhancedWolffiaCNNTrainer

trainer = EnhancedWolffiaCNNTrainer(
    model_save_dir='models',   # Model save directory
    real_images_dir='images'   # Real images for training
)

# Create datasets
trainer.create_enhanced_datasets(
    train_samples=10000,
    val_samples=1500,
    test_samples=1500,
    batch_size=16
)

# Initialize model
trainer.initialize_model(base_filters=32)

# Train model
history = trainer.train_model(
    epochs=40,
    learning_rate=0.0005,
    patience=15
)
```

### Realistic Data Generator

#### RealisticWolffiaGenerator
```python
from realistic_wolffia_generator import RealisticWolffiaGenerator

generator = RealisticWolffiaGenerator(
    patch_size=64,             # Output patch size
    real_images_dir='images'   # Directory with real images
)

# Generate single patch
sample = generator.generate_realistic_patch()
# Returns: {'image': array, 'mask': array, 'edge': array, 'distance': array}

# Generate multiple samples
generator.save_sample_images('output_dir', num_samples=20)
```

### Tophat ML Training

#### TophatAnnotationAnalyzer
```python
from tophat_ml_trainer import TophatAnnotationAnalyzer

analyzer = TophatAnnotationAnalyzer(
    tophat_training_dir='tophat_training',
    annotations_dir='annotations'
)

# Load training sessions
sessions = analyzer.load_tophat_sessions()

# Extract training data
training_data = analyzer.extract_training_patches(sessions)

# Create balanced dataset
X_train, y_train, X_test, y_test, _, _ = analyzer.create_training_dataset(training_data)
```

## Error Handling

### HTTP Status Codes
- `200 OK`: Request successful
- `400 Bad Request`: Invalid parameters
- `404 Not Found`: Resource not found
- `413 Payload Too Large`: File size exceeded
- `415 Unsupported Media Type`: Invalid file format
- `500 Internal Server Error`: Processing error

### Error Response Format
```json
{
  "status": "error",
  "error_code": "INVALID_FILE_FORMAT",
  "message": "Unsupported file format. Please use PNG, JPG, JPEG, BMP, TIFF, or JFIF.",
  "details": {
    "filename": "image.gif",
    "supported_formats": ["png", "jpg", "jpeg", "bmp", "tiff", "jfif"]
  }
}
```

### Common Error Codes
- `INVALID_FILE_FORMAT`: Unsupported image format
- `FILE_TOO_LARGE`: File exceeds size limit
- `PROCESSING_FAILED`: Analysis error
- `MODEL_NOT_AVAILABLE`: Required model not found
- `INSUFFICIENT_MEMORY`: Out of memory error
- `INVALID_PARAMETERS`: Bad request parameters

## Rate Limiting

### Limits
- **File uploads**: 10 files per minute
- **Analysis requests**: 5 per minute per IP
- **Training requests**: 1 per hour

### Headers
```http
X-RateLimit-Limit: 5
X-RateLimit-Remaining: 3
X-RateLimit-Reset: 1623456789
```

## Authentication

Currently, the API does not require authentication. For production deployment, consider implementing:
- API key authentication
- JWT tokens
- User session management

## WebSocket Support

For real-time updates during long-running operations:

```javascript
const ws = new WebSocket('ws://localhost:5000/ws/analysis/' + analysis_id);

ws.onmessage = function(event) {
    const update = JSON.parse(event.data);
    console.log('Progress:', update.progress);
};
```

## SDK Examples

### Python SDK Usage
```python
import requests
import json

# Upload file
files = {'files': open('image.jpg', 'rb')}
response = requests.post('http://localhost:5000/api/upload', files=files)
file_info = response.json()

# Start analysis
analysis_data = {'use_cnn': True}
response = requests.post(
    f'http://localhost:5000/api/analyze/{file_info["files"][0]["file_id"]}',
    json=analysis_data
)
analysis_info = response.json()

# Check status
response = requests.get(
    f'http://localhost:5000/api/status/{analysis_info["analysis_id"]}'
)
result = response.json()
```

### JavaScript SDK Usage
```javascript
async function analyzeImage(file) {
    // Upload file
    const formData = new FormData();
    formData.append('files', file);
    
    const uploadResponse = await fetch('/api/upload', {
        method: 'POST',
        body: formData
    });
    const uploadResult = await uploadResponse.json();
    
    // Start analysis
    const analysisResponse = await fetch(`/api/analyze/${uploadResult.files[0].file_id}`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({use_cnn: true})
    });
    const analysisResult = await analysisResponse.json();
    
    // Poll for results
    const pollResults = async () => {
        const statusResponse = await fetch(`/api/status/${analysisResult.analysis_id}`);
        const status = await statusResponse.json();
        
        if (status.status === 'completed') {
            return status.result;
        } else {
            setTimeout(pollResults, 1000);
        }
    };
    
    return await pollResults();
}
```

This API reference provides comprehensive coverage of all BIOIMAGIN endpoints and methods for effective integration and usage.