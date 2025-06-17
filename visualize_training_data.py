#!/usr/bin/env python3
"""
TRAINING DATA VISUALIZER
Shows the synthetic Wolffia cell images that the CNN is training on
"""

import matplotlib.pyplot as plt
import numpy as np
from wolffia_cnn_model import SyntheticWolffiaDataset


def visualize_training_samples(num_samples=20, save_plot=True):
    """Visualize synthetic training data samples"""
    print("🔍 SYNTHETIC TRAINING DATA VISUALIZATION")
    print("=" * 50)
    
    # Create dataset
    dataset = SyntheticWolffiaDataset(num_samples=num_samples, patch_size=64)
    
    # Create figure
    fig, axes = plt.subplots(4, 5, figsize=(15, 12))
    fig.suptitle('Synthetic Wolffia Training Data Samples', fontsize=16, fontweight='bold')
    
    cell_count = 0
    no_cell_count = 0
    
    for i in range(20):
        patch, label = dataset[i]
        
        # Convert tensor back to numpy if needed
        if hasattr(patch, 'numpy'):
            patch_np = patch.permute(1, 2, 0).numpy()
        else:
            patch_np = patch
        
        # Ensure values are in [0,1] range for display
        patch_np = np.clip(patch_np, 0, 1)
        
        row = i // 5
        col = i % 5
        
        axes[row, col].imshow(patch_np)
        
        # Add title with label
        if label == 1:
            axes[row, col].set_title(f'Cell Present', color='green', fontweight='bold')
            cell_count += 1
        else:
            axes[row, col].set_title(f'No Cell', color='red', fontweight='bold')
            no_cell_count += 1
        
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    # Add statistics
    fig.text(0.02, 0.02, f'Cell samples: {cell_count} | No-cell samples: {no_cell_count}', 
             fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    
    if save_plot:
        plt.savefig('training_data_samples.png', dpi=300, bbox_inches='tight')
        print("💾 Visualization saved as 'training_data_samples.png'")
    
    plt.show()
    
    print(f"\n📊 Sample Statistics:")
    print(f"   Cell samples: {cell_count}")
    print(f"   No-cell samples: {no_cell_count}")
    print(f"   Image size: 64x64 pixels")
    print(f"   Color channels: RGB")

def visualize_cell_generation_process():
    """Show step-by-step cell generation process"""
    print("\n🧪 CELL GENERATION PROCESS")
    print("=" * 40)
    
    dataset = SyntheticWolffiaDataset(num_samples=1)
    
    # Generate multiple samples to show variety
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Synthetic Wolffia Cell Generation Process', fontsize=16, fontweight='bold')
    
    for i in range(8):
        patch, label = dataset[i]
        
        if hasattr(patch, 'numpy'):
            patch_np = patch.permute(1, 2, 0).numpy()
        else:
            patch_np = patch
        
        patch_np = np.clip(patch_np, 0, 1)
        
        row = i // 4
        col = i % 4
        
        axes[row, col].imshow(patch_np)
        
        if label == 1:
            axes[row, col].set_title(f'Sample {i+1}: Cell', color='green', fontweight='bold')
        else:
            axes[row, col].set_title(f'Sample {i+1}: Background', color='red', fontweight='bold')
        
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig('cell_generation_process.png', dpi=300, bbox_inches='tight')
    print("💾 Process visualization saved as 'cell_generation_process.png'")
    plt.show()

def analyze_training_data_statistics(num_samples=1000):
    """Analyze the distribution of training data"""
    print("\n📊 TRAINING DATA STATISTICS")
    print("=" * 40)
    
    dataset = SyntheticWolffiaDataset(num_samples=num_samples)
    
    cell_samples = 0
    background_samples = 0
    
    print(f"Analyzing {num_samples} samples...")
    
    for i in range(num_samples):
        _, label = dataset[i]
        if label == 1:
            cell_samples += 1
        else:
            background_samples += 1
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{num_samples} samples...")
    
    print(f"\n📈 Results:")
    print(f"   Cell samples: {cell_samples} ({cell_samples/num_samples*100:.1f}%)")
    print(f"   Background samples: {background_samples} ({background_samples/num_samples*100:.1f}%)")
    print(f"   Balance ratio: {min(cell_samples, background_samples)/max(cell_samples, background_samples):.2f}")
    
    # Create distribution plot
    plt.figure(figsize=(8, 6))
    labels = ['Background', 'Cell']
    sizes = [background_samples, cell_samples]
    colors = ['lightcoral', 'lightgreen']
    
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.title('Training Data Distribution')
    plt.axis('equal')
    plt.savefig('training_data_distribution.png', dpi=300, bbox_inches='tight')
    print("💾 Distribution plot saved as 'training_data_distribution.png'")
    plt.show()

def main():
    """Main visualization function"""
    print("🧬 BIOIMAGIN TRAINING DATA VISUALIZER")
    print("=" * 50)
    print("This tool shows the synthetic data your CNN is learning from")
    print()
    
    try:
        # Show sample images
        visualize_training_samples(num_samples=20)
        
        # Show generation process
        visualize_cell_generation_process()
        
        # Analyze statistics
        analyze_training_data_statistics(num_samples=500)
        
        print("\n🎉 VISUALIZATION COMPLETE!")
        print("✅ Check the saved PNG files to see your training data")
        print("💡 These synthetic images teach the CNN to recognize Wolffia cells")
        
    except Exception as e:
        print(f"❌ Visualization failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()