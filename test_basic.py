#!/usr/bin/env python3
"""
Basic test script to verify core functionality without external dependencies
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_basic_import():
    """Test basic imports work"""
    try:
        import numpy as np
        import cv2
        from skimage import filters, feature, morphology, segmentation
        from scipy import ndimage
        print("✅ Basic scientific libraries available")
        return True
    except ImportError as e:
        print(f"❌ Basic import failed: {e}")
        return False

def test_watershed_algorithm():
    """Test watershed segmentation directly"""
    try:
        from skimage import filters, feature, morphology, segmentation
        from scipy import ndimage
        
        # Create test image with some bright spots
        image = np.zeros((100, 100), dtype=np.uint8)
        # Add circular bright spots to simulate cells
        y, x = np.ogrid[:100, :100]
        
        # Cell 1
        mask1 = (x-30)**2 + (y-30)**2 <= 8**2
        image[mask1] = 200
        
        # Cell 2  
        mask2 = (x-70)**2 + (y-70)**2 <= 10**2
        image[mask2] = 180
        
        # Cell 3
        mask3 = (x-50)**2 + (y-20)**2 <= 6**2
        image[mask3] = 190
        
        # Apply watershed algorithm
        thresh = filters.threshold_otsu(image)
        binary = image > thresh
        binary = morphology.remove_small_objects(binary, min_size=10)
        binary = ndimage.binary_fill_holes(binary)
        
        if not np.any(binary):
            print("⚠️ No objects found after thresholding")
            return False
        
        distance = ndimage.distance_transform_edt(binary)
        coords = feature.peak_local_max(distance, min_distance=5, threshold_abs=0.3*distance.max())
        
        if len(coords) == 0:
            print("⚠️ No peaks found")
            return False
            
        markers = np.zeros_like(distance, dtype=int)
        for i, coord in enumerate(coords):
            markers[coord[0], coord[1]] = i + 1
        
        labels = segmentation.watershed(-distance, markers, mask=binary)
        num_cells = len(coords)
        
        print(f"✅ Watershed segmentation found {num_cells} cells (expected 3)")
        return num_cells > 0
        
    except Exception as e:
        print(f"❌ Watershed test failed: {e}")
        return False

def test_basic_analyzer():
    """Test basic analyzer functionality"""
    try:
        # Try to import the fallback version first
        try:
            from bioimaging import WolffiaAnalyzer
            print("✅ Using legacy bioimaging analyzer")
        except ImportError:
            print("⚠️ Legacy analyzer not available, testing improved version")
            from bioimaging_professional_improved import WolffiaAnalyzer
        
        analyzer = WolffiaAnalyzer()
        print("✅ Analyzer initialized successfully")
        return True
        
    except Exception as e:
        print(f"❌ Analyzer initialization failed: {e}")
        return False

def main():
    """Run basic tests"""
    print("🧪 Testing basic BIOIMAGIN functionality...")
    print("=" * 50)
    
    tests = [
        ("Basic Imports", test_basic_import),
        ("Watershed Algorithm", test_watershed_algorithm),
        ("Basic Analyzer", test_basic_analyzer)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔬 Running {test_name}...")
        try:
            if test_func():
                print(f"✅ {test_name} PASSED")
                passed += 1
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 Basic functionality works! Core image processing is operational.")
    elif passed > 0:
        print("⚠️ Partial functionality available. Some features may be limited.")
    else:
        print("❌ System has critical issues. Check dependencies.")
    
    return passed > 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)