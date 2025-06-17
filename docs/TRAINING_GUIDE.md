# BIOIMAGIN - Training Guide

## Overview

BIOIMAGIN offers two powerful training systems to customize detection for your specific Wolffia images:

1. **Tophat ML Training**: Interactive annotation-based training
2. **Enhanced CNN Training**: Deep learning with synthetic data

## Tophat ML Training (Recommended for Beginners)

### When to Use Tophat Training

✅ **Perfect for**:
- New to the system
- Have 5-20 representative images
- Want quick, customized results
- Images differ from defaults

❌ **Not ideal for**:
- Less than 5 images
- Extremely diverse image types
- Need highest possible accuracy

### Step-by-Step Tophat Training

#### 1. Prepare Your Images

**Image Requirements**:
- **Count**: 5-20 images minimum, 10-30 recommended
- **Quality**: Clear, well-focused microscopy images
- **Diversity**: Include various cell densities and conditions
- **Format**: Any format (PNG, JPG, TIFF) - system auto-converts

**Image Selection Tips**:
```
Good Training Images:
✅ Representative of your typical samples
✅ Mix of high/medium/low cell density
✅ Various lighting conditions
✅ Different cell arrangements

Avoid:
❌ Extremely blurry or low contrast
❌ All identical conditions
❌ Purely background/no cells
❌ Corrupted or damaged files
```

#### 2. Start Training Session

```bash
# Launch web interface
python web_integration.py
# Navigate to http://localhost:5000
```

**Web Interface Steps**:
1. Click **"Tophat Training"** tab
2. Click **"Start Training Session"**
3. Upload your prepared images
4. Wait for initial detection analysis

#### 3. Annotation Interface Guide

The system shows you its current detection results for annotation:

**Visual Elements**:
- **Blue circles**: Current detections (what system found)
- **Your job**: Mark these as correct/incorrect and add missing cells

**Annotation Colors and Meanings**:

🟢 **Green Annotations** (Correct Detections):
```
Mark detections that are:
✅ Properly identified Wolffia cells
✅ Good cell boundaries
✅ Appropriate size (not too large/small)
✅ Clear green coloration
✅ Single cells (not merged groups)
```

🔵 **Blue Annotations** (False Positives):
```
Mark detections that are:
❌ Background areas incorrectly detected
❌ Debris, artifacts, or bubbles
❌ Non-biological objects
❌ Merged cell clusters counted as single
❌ Areas without green coloration
```

🔴 **Red Annotations** (Missed Cells):
```
Mark cells that were:
⭕ Not detected at all by the system
⭕ Partially detected (poor boundaries)
⭕ Split into multiple detections
⭕ Clearly visible Wolffia cells system missed
```

#### 4. Annotation Best Practices

**Quality Annotation Strategy**:

**Complete Coverage**:
- Annotate ALL visible Wolffia cells in each image
- Mark ALL false positives (background detections)
- Be consistent with your criteria across images

**Annotation Guidelines**:
```python
# Good Wolffia Cell Criteria:
- Size: 15-800 pixels area
- Shape: Roughly circular to oval
- Color: Visible green content
- Boundaries: Clear, defined edges
- Context: Floating in aquatic environment

# Mark as FALSE POSITIVE:
- Large background regions
- Debris or air bubbles
- Non-green objects
- Extremely large or small detections
- Clearly artificial artifacts
```

**Annotation Workflow**:
1. **Quick Pass**: Mark obvious false positives (blue)
2. **Careful Review**: Mark all correct detections (green)
3. **Missing Cells**: Add any cells system missed (red)
4. **Final Check**: Review consistency across images

#### 5. Training Process

After annotations are complete:

```bash
# Automatic training triggered by interface
# OR manual training:
python tophat_trainer.py
```

**Training Output Example**:
```
🎯 Tophat ML Training Session: session_20250616_143502
========================================================
Images Processed: 12
Total Annotations: 847
- ✅ Correct Detections (Green): 523 (61.7%)
- ❌ False Positives (Blue): 201 (23.7%)
- ⭕ Missed Cells (Red): 123 (14.5%)

🔬 Feature Engineering:
Extracted 34 features per annotation:
- Morphological: area, perimeter, eccentricity, solidity
- Intensity: mean, std, percentiles, contrast
- Texture: LBP, GLCM properties
- Shape: circularity, aspect ratio, orientation

🤖 Model Training:
Algorithm: Random Forest (100 trees)
Training Accuracy: 89.3%
Cross-Validation: 87.1% ± 3.2%
Feature Importance: area (23.4%), circularity (18.7%)

✅ Training completed in 23.4 seconds
💾 Model saved: models/tophat_model.pkl
```

#### 6. Testing Your Trained Model

```python
from bioimaging import WolffiaAnalyzer

analyzer = WolffiaAnalyzer()

# Check if model loaded
print(f"Tophat model available: {analyzer.tophat_model is not None}")

# Test on new image
result = analyzer.analyze_image('test_image.jpg', use_tophat=True)
print(f"Detected {result['total_cells']} cells with custom model")
```

## Enhanced CNN Training

### When to Use CNN Training

✅ **Perfect for**:
- Want highest accuracy
- Have GPU for training
- Need to process many images
- Research-grade requirements

❌ **Consider alternatives**:
- Limited computational resources
- Very few training images
- Need quick results

### CNN Training Options

#### Quick Training (Recommended Start)
```bash
python train_wolffia_cnn.py

# Interactive menu:
# 1. Quick (8K samples, ~10 min)
# 2. Standard (15K samples, ~20 min)
# 3. Professional (25K samples, ~35 min)
# 4. Research-grade (40K samples, ~60 min)
```

#### Advanced Training Script
```bash
# Train with custom parameters
python enhanced_wolffia_trainer.py --samples 25000 --epochs 50 --gpu
```

### CNN Training Process

#### 1. Synthetic Data Generation

**Realistic Data Creation**:
```python
# Automatic generation includes:
- Poisson disc sampling for natural placement
- Wolffia-specific morphology (size, shape)
- Realistic coloration (green variations)
- Natural lighting effects (shadows, gradients)
- Background integration (aquatic environments)
```

**Training Data Structure**:
```
Generated Training Data:
├── Cell Samples: 25,000 synthetic cells
├── Background Samples: 15,000 negative examples
├── Augmentation: 10x variation per sample
├── Validation Set: 20% held out
└── Test Set: 10% final evaluation
```

#### 2. Training Configuration

**Model Architecture**:
```python
Enhanced Wolffia CNN:
- Input: 64×64 grayscale patches
- Encoder: 4-layer CNN with batch normalization
- Decoder: U-Net style with skip connections
- Output: Multi-task (mask + edge + distance)
- Parameters: ~2.1M trainable parameters
```

**Training Parameters**:
```python
Training Configuration:
- Optimizer: Adam (lr=0.001, decay=1e-4)
- Loss: Combined BCE + Dice + MSE
- Batch Size: 32 (GPU), 16 (CPU)
- Epochs: 50 with early stopping
- Validation: 20% of training data
```

#### 3. Training Monitoring

**Real-time Progress**:
```
Epoch 15/50:
  Train Loss: 0.142, Val Loss: 0.163
  Mask Acc: 92.4%, Edge Acc: 89.8%
  Best model saved! (validation improved)
  
Training Visualizations:
├── Loss curves: training_visualizations/loss_curves.png
├── Accuracy plots: training_visualizations/accuracy.png
├── Sample predictions: training_visualizations/samples.png
└── Confusion matrix: training_visualizations/confusion.png
```

#### 4. Model Evaluation

```python
# Automatic evaluation after training
Final Model Performance:
- Validation Accuracy: 93.4%
- Test Set Accuracy: 91.7%
- Precision: 0.912
- Recall: 0.873
- F1-Score: 0.892

Model saved to: models/enhanced_wolffia_cnn_best.pth
```

## Advanced Training Techniques

### Real Image Training Data Extraction

Extract training data from your own annotated images:

```bash
# Analyze real images to create training targets
python real_image_analyzer.py --input_dir your_images/ --output_dir extracted_data/

# Use extracted data for CNN training
python enhanced_wolffia_trainer.py --real_data extracted_data/ --synthetic_ratio 0.7
```

### Hybrid Training (Recommended for Best Results)

Combine multiple training approaches:

```bash
# 1. Start with tophat training for quick customization
# Web interface → Tophat Training → Annotate 10-15 images

# 2. Extract features from tophat annotations
python tophat_ml_trainer.py --export_features

# 3. Train enhanced CNN with real + synthetic data
python enhanced_wolffia_trainer.py --use_tophat_data --samples 30000
```

### Transfer Learning

Use pre-trained models as starting points:

```python
from wolffia_cnn_model import WolffiaCNN

# Load pre-trained model
model = WolffiaCNN.load_pretrained('models/base_wolffia_model.pth')

# Fine-tune on your data
trainer = EnhancedTrainer(model, fine_tune=True)
trainer.train(your_data, epochs=20, learning_rate=0.0001)
```

## Training Data Management

### Data Organization

```
Recommended Directory Structure:
bioimagin/
├── training_data/
│   ├── raw_images/           # Original microscopy images
│   ├── annotations/          # Tophat annotation sessions
│   ├── synthetic/           # Generated training data
│   ├── extracted/           # Features from real images
│   └── models/              # Trained model files
├── validation/
│   ├── test_images/         # Held-out test images
│   └── ground_truth/        # Manual annotations for validation
└── results/
    ├── training_logs/       # Training progress logs
    ├── evaluations/         # Model performance reports
    └── visualizations/      # Training charts and plots
```

### Quality Control

**Training Data Validation**:
```python
# Check annotation quality
python validate_annotations.py --session_dir tophat_training/

# Verify synthetic data quality
python inspect_synthetic_data.py --output_dir training_visualizations/

# Evaluate model performance
python evaluate_models.py --test_dir validation/test_images/
```

## Training Best Practices

### Data Quality

**Annotation Guidelines**:
1. **Consistency**: Apply same criteria across all images
2. **Completeness**: Mark ALL cells and false positives
3. **Representative**: Include diverse conditions and densities
4. **Balanced**: Aim for 50-70% positive examples

**Image Quality Standards**:
```
Training Image Checklist:
✅ Sharp focus on cells
✅ Adequate contrast
✅ Representative cell density
✅ Minimal motion blur
✅ Good lighting conditions
✅ Minimal compression artifacts
```

### Training Strategies

**Incremental Training**:
1. Start with 5-10 images for tophat training
2. Test on new images
3. Add more annotations if needed
4. Consider CNN training for highest accuracy

**Evaluation Protocol**:
1. Hold out 20% of images for testing
2. Never train on test images
3. Use cross-validation for model selection
4. Test on completely new image batches

### Troubleshooting Training Issues

**Common Problems and Solutions**:

**Low Training Accuracy**:
```bash
# Check annotation quality
python analyze_annotations.py --session session_id

# Increase training data
# Add more diverse images with careful annotations

# Adjust model parameters
# For tophat: increase max_depth, n_estimators
# For CNN: increase epochs, adjust learning rate
```

**Overfitting**:
```python
# Reduce model complexity
model = WolffiaCNN(features=[16, 32, 64])  # Fewer features

# Add regularization
trainer.train(dropout=0.3, weight_decay=1e-4)

# Increase training data diversity
```

**Poor Generalization**:
```bash
# Add more diverse training images
# Include various lighting conditions, cell densities
# Test on completely different image batches
# Consider domain adaptation techniques
```

## Integration After Training

### Using Trained Models

```python
from bioimaging import WolffiaAnalyzer

# Initialize with trained models
analyzer = WolffiaAnalyzer()

# Models are automatically detected and loaded
result = analyzer.analyze_image(
    'new_image.jpg',
    use_tophat=True,    # Uses your trained tophat model
    use_cnn=True,       # Uses trained CNN model
    use_celldetection=False
)

# Compare different methods
methods = ['watershed', 'tophat', 'cnn', 'fusion']
for method in methods:
    if method in result['method_results']:
        cells = result['method_results'][method]['cells_detected']
        print(f"{method}: {cells} cells")
```

### Model Management

```bash
# Check available models
ls models/
# tophat_model.pkl
# enhanced_wolffia_cnn_best.pth
# base_cnn_model.pth

# Backup important models
cp models/tophat_model.pkl models/backup/tophat_model_$(date +%Y%m%d).pkl

# Model performance comparison
python compare_models.py --test_dir validation/
```

### Continuous Improvement

**Iterative Refinement**:
1. Collect challenging images where models fail
2. Add targeted annotations for these cases
3. Retrain with expanded dataset
4. Evaluate improvement on test set
5. Deploy updated models

**Performance Monitoring**:
```python
# Track model performance over time
performance_log = {
    'model_version': 'tophat_v2.1',
    'training_date': '2025-06-16',
    'test_accuracy': 0.891,
    'images_trained_on': 847,
    'notes': 'Added diverse lighting conditions'
}
```

## Advanced Topics

### Custom Loss Functions

For specialized detection requirements:

```python
class WolffiaLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.2, gamma=0.1):
        super().__init__()
        self.alpha = alpha  # Mask loss weight
        self.beta = beta    # Edge loss weight  
        self.gamma = gamma  # Size consistency weight
        
    def forward(self, pred, target):
        mask_loss = F.binary_cross_entropy(pred['mask'], target['mask'])
        edge_loss = F.mse_loss(pred['edges'], target['edges'])
        size_loss = F.mse_loss(pred['sizes'], target['sizes'])
        
        return self.alpha * mask_loss + self.beta * edge_loss + self.gamma * size_loss
```

### Multi-Scale Training

For handling various cell sizes:

```python
# Train on multiple scales
scales = [0.5, 0.75, 1.0, 1.25, 1.5]
for scale in scales:
    scaled_data = resize_training_data(training_data, scale)
    model.train_epoch(scaled_data)
```

### Domain Adaptation

For different microscopy setups:

```bash
# Adapt model to new imaging conditions
python domain_adaptation.py \
    --source_model models/base_model.pth \
    --target_images new_setup_images/ \
    --output models/adapted_model.pth
```

---

**Training complete! Your models are now customized for optimal Wolffia detection! 🎯🔬**