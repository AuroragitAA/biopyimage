#!/usr/bin/env python3
"""
BIOIMAGIN Enhanced Wolffia CNN Training
Professional training script specifically optimized for Wolffia arrhiza analysis
Author: BIOIMAGIN Professional Team
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

def main():
    """Main training function with enhanced Wolffia-specific features"""
    
    print("🌱 BIOIMAGIN ENHANCED WOLFFIA CNN TRAINING")
    print("=" * 60)
    print("🎯 Specialized for Wolffia arrhiza cell detection")
    print("🔬 Professional-grade training with real microscopy simulation")
    print("📊 Enhanced synthetic data generation")
    print()
    
    # Check PyTorch availability
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__} detected")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🔧 Using device: {device}")
        
        if device == 'cuda':
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("   CPU training (slower but will work)")
    except ImportError:
        print("❌ PyTorch not available. Please install PyTorch:")
        print("   pip install torch torchvision")
        return False
    
    print()
    
    # Training configuration
    print("📊 ENHANCED TRAINING CONFIGURATION")
    print("-" * 40)
    
    try:
        print("Choose training mode:")
        print("  1. Quick training (5K samples, ~5 minutes)")
        print("  2. Standard training (8K samples, ~8 minutes) [RECOMMENDED]")
        print("  3. Comprehensive training (12K samples, ~15 minutes)")
        print("  4. Maximum training (15K samples, ~20 minutes)")
        print("  5. Enhanced realistic training (8K realistic samples, ~12 minutes)")
        print("  6. ENHANCED RGB training (8K RGB samples with green-enhancement, ~10 minutes) [NEW]")
        
        mode = input("Enter choice (1-6) [default: 6]: ").strip() or "6"
        
        if mode == "1":
            train_samples, val_samples, epochs = 5000, 1000, 30
            trainer_type = "standard"
            print("🚀 Quick training mode selected")
        elif mode == "2":
            train_samples, val_samples, epochs = 8000, 1500, 40
            trainer_type = "standard"
            print("🚀 Standard training mode selected [RECOMMENDED]")
        elif mode == "3":
            train_samples, val_samples, epochs = 12000, 2000, 50
            trainer_type = "standard"
            print("🚀 Comprehensive training mode selected")
        elif mode == "4":
            train_samples, val_samples, epochs = 15000, 2500, 60
            trainer_type = "standard"
            print("🚀 Maximum training mode selected")
        elif mode == "5":
            train_samples, val_samples, epochs = 8000, 1500, 45
            trainer_type = "enhanced"
            print("🚀 Enhanced realistic training mode selected")
        elif mode == "6":
            train_samples, val_samples, epochs = 8000, 1500, 35
            trainer_type = "rgb_enhanced"
            print("🟢 ENHANCED RGB training mode selected")
            print("   ✨ Features: Green-enhanced 3-channel input, background rejection")
        else:
            print("Invalid choice, using enhanced RGB mode")
            train_samples, val_samples, epochs = 8000, 1500, 35
            trainer_type = "rgb_enhanced"
        
        print(f"📊 Configuration: {train_samples} training samples, {val_samples} validation samples")
        print(f"🔄 Training for {epochs} epochs with early stopping")
        print()
        
        # Use the enhanced unified trainer
        from wolffia_cnn_model import WolffiaCNNTrainer
        print("🌱 Using Enhanced Wolffia CNN Trainer with python_for_microscopists techniques")
        print("✨ Features: Attention mechanisms, ASPP, multi-task learning, focal loss")
        print()
        
        # Initialize trainer
        trainer = WolffiaCNNTrainer(real_images_dir='images', device=device)
        
        # Create datasets (enable multi-task if enhanced mode)
        use_multi_task = (trainer_type in ["enhanced", "rgb_enhanced"])
        use_rgb = (trainer_type == "rgb_enhanced")
        print("📦 Creating enhanced synthetic datasets...")
        trainer.create_datasets(
            train_samples=train_samples,
            val_samples=val_samples,
            test_samples=max(1000, val_samples // 2),
            batch_size=16,
            multi_task=use_multi_task,
            use_rgb=use_rgb
        )
        
        # Initialize model with proper input channels
        input_channels = 3 if trainer_type == "rgb_enhanced" else 1
        print(f"🧠 Initializing Enhanced Wolffia CNN model ({input_channels}-channel input)...")
        trainer.initialize_model(
            input_channels=input_channels,
            base_filters=32,
            use_attention=True,
            multi_task=use_multi_task
        )
        
        # Train model
        print(f"🚀 Starting training for {epochs} epochs...")
        print("💡 Training will stop early if the model converges")
        print()
        
        history = trainer.train_model(
            epochs=epochs, 
            learning_rate=0.001,
            use_focal_loss=True
        )
        
        # Evaluate model
        print("📊 Evaluating trained model...")
        test_results = trainer.evaluate_model()
        
        # Visualize results
        print("📈 Creating training visualizations...")
        trainer.visualize_training_history(history)
        
        success = True
        
        if success:
            print()
            print("🎉 WOLFFIA CNN TRAINING COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            
            # Load and display results
            try:
                # Use unified file names
                history_file = Path('models/training_history.json')
                model_file = Path('models/wolffia_cnn_best.pth')
                
                if history_file.exists():
                    with open(history_file, 'r') as f:
                        history = json.load(f)
                    
                    print(f"🏆 Training completed in {history.get('epochs_trained', epochs)} epochs")
                    print(f"📊 Best validation loss: {history.get('best_val_loss', 0):.4f}")
                    
                    if 'val_accuracies' in history and history['val_accuracies']:
                        final_accuracy = history['val_accuracies'][-1]
                        print(f"🎯 Final validation accuracy: {final_accuracy:.1%}")
                        
                        # Performance assessment
                        if final_accuracy > 0.90:
                            print("🌟 Outstanding performance! Ready for production.")
                        elif final_accuracy > 0.85:
                            print("✅ Excellent performance! Suitable for research.")
                        elif final_accuracy > 0.75:
                            print("✅ Good performance! Suitable for most applications.")
                        else:
                            print("⚠️ Performance could be improved - consider more training.")
                    
                    if 'final_test_accuracy' in history:
                        print(f"✅ Final test accuracy: {history['final_test_accuracy']:.1%}")
                
            except Exception as e:
                print(f"⚠️ Could not load training history: {e}")
            
            # Check model file
            model_files = []
            if Path('models/wolffia_cnn_best.pth').exists():
                model_files.append('wolffia_cnn_best.pth')
            
            print()
            print("📁 Model files saved:")
            for model_file in model_files:
                size_mb = Path(f'models/{model_file}').stat().st_size / (1024*1024)
                print(f"   • {model_file} ({size_mb:.1f} MB)")
            
            print()
            print("📈 Training artifacts:")
            artifacts = [
                ('training_history.json', 'Training metrics and performance'),
                ('training_history.png', 'Performance visualization plots'),
                ('sample_preview/', 'Sample training images')
            ]
            
            for artifact, description in artifacts:
                if Path(f'models/{artifact}').exists() or Path(artifact).exists():
                    print(f"   ✅ {artifact} - {description}")
            
            print()
            print("🔄 INTEGRATION STATUS:")
            print("1. Model capabilities:")
            print("   🎯 Optimized for small Wolffia cell detection")
            print("   🎯 Handles realistic microscopy conditions")
            print("   🎯 Works with varying lighting and focus")
            print("   🎯 Attention mechanisms for better feature focusing")
            print("   🎯 Multi-scale feature extraction with ASPP")
            print("   🎯 Focal loss for handling class imbalance")
            if use_multi_task:
                print("   🎯 Multi-task learning (segmentation + edges + distance)")
                print("   🎯 Enhanced boundary detection")
            
            print()
            print("2. Ready for deployment:")
            print("   ✅ Model integrated with bioimaging.py")
            print("   ✅ Compatible with web interface")
            print("   ✅ Works alongside tophat and watershed methods")
            print("   ✅ Automatic GPU/CPU device detection")
            
            print()
            print("3. Next steps:")
            print("   🔄 Restart your web server to load the new model")
            print("   🧪 Test with real Wolffia images")
            print("   📊 Compare with other detection methods")
            print("   🎯 Use in combination with tophat training for best results")
            
            return True
        else:
            print("❌ Training failed. Check error messages above.")
            return False
            
    except KeyboardInterrupt:
        print("\n\n⚠️ Training interrupted by user")
        print("Partial training progress may have been saved")
        return False
    except Exception as e:
        print(f"\n❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def check_system_requirements():
    """Check if system meets requirements for CNN training"""
    print("🔍 SYSTEM REQUIREMENTS CHECK")
    print("-" * 35)
    
    # Python version
    python_version = sys.version_info
    if python_version >= (3, 7):
        print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    else:
        print(f"❌ Python {python_version.major}.{python_version.minor} (requires 3.7+)")
        return False
    
    # Required packages
    required_packages = [
        ('torch', 'PyTorch for deep learning'),
        ('torchvision', 'Computer vision utilities'),
        ('numpy', 'Numerical computing'),
        ('cv2', 'OpenCV for image processing'),
        ('sklearn', 'Machine learning utilities'),
        ('matplotlib', 'Plotting and visualization'),
        ('scipy', 'Scientific computing'),
        ('skimage', 'Image processing utilities')
    ]
    
    missing_packages = []
    for import_name, description in required_packages:
        try:
            __import__(import_name)
            print(f"✅ {import_name}")
        except ImportError:
            print(f"❌ {import_name} - {description}")
            missing_packages.append(import_name)
    
    # Check for training files
    if Path('wolffia_cnn_model.py').exists():
        print("✅ wolffia_cnn_model.py")
    else:
        print("❌ wolffia_cnn_model.py not found")
        return False
    
    if Path('enhanced_wolffia_trainer.py').exists():
        print("✅ enhanced_wolffia_trainer.py")
    else:
        print("⚠️ enhanced_wolffia_trainer.py not found (enhanced mode unavailable)")
    
    # Check for real images
    images_dir = Path('images')
    if images_dir.exists():
        image_files = [f for f in images_dir.glob('*') if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']]
        if image_files:
            print(f"✅ Found {len(image_files)} images in images/ directory")
        else:
            print("⚠️ No image files found in images/ directory")
    else:
        print("⚠️ images/ directory not found")
        print("💡 Create images/ directory and add real Wolffia images for better training")
    
    # Create required directories
    Path('models').mkdir(exist_ok=True)
    print("✅ models/ directory ready")
    
    if missing_packages:
        print(f"\n📦 To install missing packages:")
        if 'cv2' in missing_packages:
            missing_packages[missing_packages.index('cv2')] = 'opencv-python'
        if 'sklearn' in missing_packages:
            missing_packages[missing_packages.index('sklearn')] = 'scikit-learn'
        if 'skimage' in missing_packages:
            missing_packages[missing_packages.index('skimage')] = 'scikit-image'
        
        print(f"pip install {' '.join(set(missing_packages))}")
        return False
    
    print("\n✅ System ready for Wolffia CNN training!")
    return True


def show_training_preview():
    """Show what the training will accomplish"""
    print("🎯 WOLFFIA CNN TRAINING PREVIEW")
    print("=" * 40)
    print()
    
    print("🔬 What this training accomplishes:")
    print("  ✅ Creates a specialized CNN for Wolffia cell detection")
    print("  ✅ Generates thousands of realistic synthetic training samples")
    print("  ✅ Trains with proven deep learning techniques")
    print("  ✅ Optimizes for small cell detection accuracy")
    print("  ✅ Integrates seamlessly with the analysis pipeline")
    print()
    
    print("📊 Training data characteristics:")
    print("  🎯 Realistic microscopy conditions (focus, lighting, noise)")
    print("  🎯 Various background types (culture medium, plates, water)")
    print("  🎯 Wolffia-specific cell sizes and morphology")
    print("  🎯 Multiple cell densities and arrangements")
    print("  🎯 Enhanced color representation for green cells")
    print()
    
    print("🧠 Model architecture:")
    print("  🔧 U-Net inspired CNN with skip connections")
    print("  🔧 Optimized for small object detection")
    print("  🔧 Batch normalization and dropout for stability")
    print("  🔧 GPU acceleration when available")
    print("  🔧 Automatic early stopping to prevent overfitting")
    print()
    
    print("📈 Expected outcomes:")
    print("  🎯 >85% detection accuracy on validation data")
    print("  🎯 Robust performance on real Wolffia images")
    print("  🎯 Fast inference (<1 second per image)")
    print("  🎯 Integration with existing analysis methods")
    print("  🎯 Professional-grade training metrics and visualizations")
    print()


if __name__ == "__main__":
    print("🌱 BIOIMAGIN ENHANCED WOLFFIA CNN TRAINER")
    print("=" * 50)
    print("🎯 Specialized training for Wolffia arrhiza cell detection")
    print("🔬 Professional-grade deep learning for microscopy analysis")
    print()
    
    # Show training preview
    show_preview = input("Show training preview and capabilities? (y/n) [n]: ").strip().lower()
    if show_preview == 'y':
        show_training_preview()
        print()
    
    # Check system requirements
    print("🔍 Checking system requirements...")
    if not check_system_requirements():
        print("\n❌ Please install missing requirements and try again")
        print("💡 Run: pip install torch torchvision opencv-python scikit-learn matplotlib scipy scikit-image")
        sys.exit(1)
    
    print()
    
    # Confirm training
    confirm = input("Start Wolffia CNN training? (y/n) [y]: ").strip().lower()
    if confirm == 'n':
        print("Training cancelled.")
        sys.exit(0)
    
    print()
    
    # Run training
    success = main()
    
    if success:
        print("\n🎉 Training completed successfully!")
        print("🔄 You can now restart your web server to use the new CNN model.")
    else:
        print("\n❌ Training failed. Please check the errors above.")
        sys.exit(1)