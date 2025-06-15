#!/usr/bin/env python3
"""
BIOIMAGIN Model Training Script - Deployment Version
Simple script to train all models using proven patterns
Author: BIOIMAGIN Professional Team
"""

import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from wolffia_cnn_model import train_wolffia_cnn
from tophat_trainer_fixed import train_tophat_model


def main():
    """Main training function - trains all available models"""
    
    print("🚀 BIOIMAGIN Model Training - Deployment Version")
    print("=" * 60)
    
    # Check PyTorch availability for CNN training
    try:
        import torch
        pytorch_available = True
        print("✅ PyTorch available - CNN training enabled")
    except ImportError:
        pytorch_available = False
        print("⚠️ PyTorch not available - CNN training disabled")
    
    # Training results
    results = {}
    
    # 1. Train CNN Model (if PyTorch available)
    if pytorch_available:
        print("\n📈 Training Wolffia CNN Model...")
        print("-" * 40)
        try:
            success = train_wolffia_cnn(num_samples=2000, epochs=30)
            results['cnn'] = success
            if success:
                print("✅ CNN training completed successfully")
            else:
                print("❌ CNN training failed")
        except Exception as e:
            print(f"❌ CNN training error: {e}")
            results['cnn'] = False
    else:
        results['cnn'] = False
        print("⏭️ Skipping CNN training (PyTorch not available)")
    
    # 2. Train Tophat ML Model
    print("\n🎯 Training Tophat ML Model...")
    print("-" * 40)
    try:
        success = train_tophat_model(n_estimators=100)
        results['tophat'] = success
        if success:
            print("✅ Tophat ML training completed successfully")
        else:
            print("❌ Tophat ML training failed (no annotation data)")
    except Exception as e:
        print(f"❌ Tophat ML training error: {e}")
        results['tophat'] = False
    
    # Summary
    print("\n📊 Training Summary")
    print("=" * 60)
    
    total_models = len(results)
    successful_models = sum(results.values())
    
    for model_name, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{model_name.upper():<10} : {status}")
    
    print(f"\nModels trained successfully: {successful_models}/{total_models}")
    
    if successful_models > 0:
        print("\n🎉 Training completed! Your models are ready for deployment.")
        print("\nNext steps:")
        print("1. Test the system: python bioimaging.py")
        print("2. Start web interface: python web_integration.py")
        print("3. Upload Wolffia images for analysis")
    else:
        print("\n⚠️ No models were trained successfully.")
        print("Please check error messages above and ensure:")
        print("1. PyTorch is installed for CNN training")
        print("2. Annotation data is available for Tophat ML training")
    
    return successful_models > 0


if __name__ == "__main__":
    main()