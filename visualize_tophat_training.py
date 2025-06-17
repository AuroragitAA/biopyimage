#!/usr/bin/env python3
"""
Standalone script to visualize tophat training data and annotations
"""

import sys
from pathlib import Path
from enhanced_tophat_trainer import EnhancedTophatTrainer

def main():
    """Create comprehensive visualizations of tophat training data"""
    
    print("📷 TOPHAT TRAINING DATA VISUALIZATION")
    print("=" * 50)
    print("🎯 Showcase training images and annotations")
    print("📊 Generate comprehensive training statistics")
    print()
    
    # Initialize trainer
    trainer = EnhancedTophatTrainer()
    
    # Load training data
    print("🔍 Loading annotation data...")
    training_data = trainer.load_annotation_data_enhanced()
    
    if not training_data:
        print("❌ No training data found.")
        print()
        print("💡 To create training data:")
        print("   1. Use the web interface to upload images")
        print("   2. Start a tophat training session")
        print("   3. Mark missed and incorrect cells")
        print("   4. Save annotations")
        print("   5. Run this script again")
        return False
    
    print(f"✅ Found {len(training_data)} training images")
    
    # Show data summary
    total_missed = 0
    total_correct = 0
    total_false_pos = 0
    
    for data in training_data:
        annotations = data['annotations']
        total_missed += len(annotations.get('missed', []))
        total_correct += len(annotations.get('correct', []))
        total_false_pos += len(annotations.get('false_positive', []))
    
    print(f"📊 Training Data Summary:")
    print(f"   Training images: {len(training_data)}")
    print(f"   Missed cells (need detection): {total_missed}")
    print(f"   Correct cells (good detection): {total_correct}")
    print(f"   False positives (over detection): {total_false_pos}")
    print(f"   Average missed per image: {total_missed/len(training_data):.1f}")
    print()
    
    if total_missed == 0:
        print("⚠️ Warning: No missed cells found in annotations.")
        print("   This means no positive training examples for the model.")
        print("   Consider creating more training annotations.")
        print()
    
    # Create visualizations
    print("📷 Creating training sample visualizations...")
    trainer.visualize_training_samples(training_data, max_samples=8)
    
    print("📊 Creating annotation statistics...")
    trainer.create_annotation_statistics(training_data)
    
    # List created files
    artifacts_dir = Path('training_artifacts')
    if artifacts_dir.exists():
        print(f"\n📁 Generated visualization files:")
        
        files_to_check = [
            ('tophat_training_samples.png', 'Training images with annotations'),
            ('tophat_annotation_statistics.png', 'Annotation statistics and distributions'),
            ('tophat_training_visualization.png', 'Training progress plots (if model trained)'),
            ('tophat_training_history.json', 'Training metrics data')
        ]
        
        for filename, description in files_to_check:
            file_path = artifacts_dir / filename
            if file_path.exists():
                size_kb = file_path.stat().st_size / 1024
                print(f"   ✅ {filename} ({size_kb:.1f} KB) - {description}")
            else:
                print(f"   ⚠️ {filename} - {description} (not created)")
        
        print(f"\n📂 All files saved in: {artifacts_dir.absolute()}")
    
    print("\n🎉 Visualization completed successfully!")
    print("\n💡 Next steps:")
    print("   1. Review the generated visualizations")
    print("   2. Check if training data is sufficient")
    print("   3. Run enhanced_tophat_trainer.py to train the model")
    print("   4. Test the trained model with your images")
    
    return True

def show_help():
    """Show help information"""
    print("📖 TOPHAT TRAINING VISUALIZATION HELP")
    print("=" * 40)
    print()
    print("This script creates comprehensive visualizations of your tophat training data.")
    print()
    print("📷 Generated Visualizations:")
    print("   • Training samples with annotation overlays")
    print("   • Statistical analysis of annotations")
    print("   • Distribution plots of cell counts")
    print("   • Training data quality assessment")
    print()
    print("📋 Prerequisites:")
    print("   • Training annotations must exist in annotations/ directory")
    print("   • Original images must be available in uploads/ directory")
    print("   • At least one tophat training session must have been completed")
    print()
    print("🔧 Usage:")
    print("   python visualize_tophat_training.py")
    print("   python visualize_tophat_training.py --help")
    print()

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ['--help', '-h', 'help']:
        show_help()
        sys.exit(0)
    
    success = main()
    sys.exit(0 if success else 1)