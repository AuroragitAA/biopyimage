#!/usr/bin/env python3
"""
Quick test script to verify the fixes are working
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_import():
    """Test that the improved pipeline imports correctly"""
    try:
        from bioimaging_professional_improved import WolffiaAnalyzer, get_system_status
        print("✅ Successfully imported improved professional pipeline")
        return True
    except ImportError as e:
        print(f"❌ Failed to import improved pipeline: {e}")
        return False

def test_initialization():
    """Test analyzer initialization"""
    try:
        from bioimaging_professional_improved import WolffiaAnalyzer
        analyzer = WolffiaAnalyzer()
        print("✅ Analyzer initialized successfully")
        
        # Test health check
        health = analyzer.health_check()
        print(f"✅ Health check completed: {health['status']}")
        if health['issues']:
            print(f"⚠️ Issues found: {health['issues']}")
        if health['recommendations']:
            print(f"💡 Recommendations: {health['recommendations']}")
        
        return True
    except Exception as e:
        print(f"❌ Analyzer initialization failed: {e}")
        return False

def test_fallback_segmentation():
    """Test that the fallback segmentation works"""
    try:
        from bioimaging_professional_improved import WolffiaAnalyzer
        analyzer = WolffiaAnalyzer()
        
        # Create a dummy image for testing
        dummy_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        # Add some bright spots to simulate cells
        dummy_image[30:40, 30:40] = [0, 255, 0]  # Green spot
        dummy_image[60:70, 60:70] = [0, 200, 0]  # Another green spot
        
        # Test restoration engine first
        if hasattr(analyzer, 'restoration_engine') and analyzer.restoration_engine:
            restoration_result = analyzer.restoration_engine.enhance_image(dummy_image, mode='enhance')
            print("✅ Image restoration working")
            
            # Test segmentation engine fallback
            if hasattr(analyzer, 'segmentation_engine') and analyzer.segmentation_engine:
                seg_result = analyzer.segmentation_engine._fallback_segmentation(restoration_result)
                print(f"✅ Fallback segmentation working - found {seg_result['num_cells']} cells")
                return seg_result['num_cells'] > 0
            else:
                print("⚠️ Segmentation engine not available")
                return False
        else:
            print("⚠️ Restoration engine not available")
            return False
            
    except Exception as e:
        print(f"❌ Fallback segmentation test failed: {e}")
        return False

def test_web_integration():
    """Test web integration health check"""
    try:
        from web_integration import app
        with app.test_client() as client:
            response = client.get('/api/health_check')
            if response.status_code == 200:
                data = response.get_json()
                print(f"✅ Web integration health check: {data.get('status', 'unknown')}")
                print(f"   Pipeline: {data.get('version', 'unknown')}")
                return True
            else:
                print(f"❌ Health check returned status {response.status_code}")
                return False
    except Exception as e:
        print(f"❌ Web integration test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing BIOIMAGIN fixes...")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_import),
        ("Initialization Test", test_initialization),
        ("Fallback Segmentation Test", test_fallback_segmentation),
        ("Web Integration Test", test_web_integration)
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
        print("🎉 All tests passed! System is ready for production.")
        return True
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)