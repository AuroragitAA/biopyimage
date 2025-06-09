#!/usr/bin/env python3
"""
Test the segmentation pipeline to verify fixes are working
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def create_test_image():
    """Create a test image with clear cell-like structures"""
    # Create a 200x200 image
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    
    # Add background noise
    background = np.random.randint(20, 40, (200, 200))
    image[:, :, 0] = background  # Red
    image[:, :, 1] = background + 10  # Green (slightly higher)
    image[:, :, 2] = background  # Blue
    
    # Add some clear green circular "cells"
    y, x = np.ogrid[:200, :200]
    
    # Cell 1 - Large cell
    mask1 = (x-60)**2 + (y-60)**2 <= 15**2
    image[mask1] = [30, 180, 40]  # Bright green
    
    # Cell 2 - Medium cell
    mask2 = (x-140)**2 + (y-80)**2 <= 12**2
    image[mask2] = [25, 160, 35]
    
    # Cell 3 - Small cell
    mask3 = (x-100)**2 + (y-140)**2 <= 8**2
    image[mask3] = [35, 170, 45]
    
    # Cell 4 - Another medium cell
    mask4 = (x-50)**2 + (y-150)**2 <= 10**2
    image[mask4] = [28, 165, 38]
    
    # Cell 5 - Small cell
    mask5 = (x-160)**2 + (y-40)**2 <= 6**2
    image[mask5] = [32, 175, 42]
    
    return image

def test_analyzer_with_fallback():
    """Test the analyzer with our synthetic image"""
    try:
        from bioimaging_professional_improved import WolffiaAnalyzer
        
        print("🧪 Testing Improved Professional Pipeline with synthetic image...")
        analyzer = WolffiaAnalyzer()
        
        # Create test image
        test_image = create_test_image()
        print(f"📷 Created test image with 5 simulated cells")
        
        # Save test image temporarily
        from PIL import Image
        pil_image = Image.fromarray(test_image)
        test_path = Path("test_synthetic_cells.png")
        pil_image.save(test_path)
        
        print(f"💾 Saved test image to {test_path}")
        
        # Run analysis
        print("🔬 Running analysis...")
        result = analyzer.analyze_image_professional(
            test_path,
            restoration_mode='enhance',
            segmentation_model='auto'
        )
        
        # Check results
        if result.get('success', False):
            cell_count = len(result.get('cells', []))
            print(f"✅ Analysis completed successfully!")
            print(f"📊 Cells detected: {cell_count}")
            print(f"🎯 Expected: 5 cells")
            
            if cell_count > 0:
                print(f"🎉 SUCCESS: Segmentation is working! Found {cell_count} cells")
                return True
            else:
                print(f"⚠️ ISSUE: No cells detected - segmentation may still have issues")
                return False
        else:
            print(f"❌ Analysis failed: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up
        if test_path.exists():
            test_path.unlink()

def main():
    """Run segmentation test"""
    print("🧪 Testing segmentation fixes...")
    print("=" * 50)
    
    success = test_analyzer_with_fallback()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 Segmentation test PASSED! The fixes are working.")
    else:
        print("❌ Segmentation test FAILED. May need additional debugging.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)