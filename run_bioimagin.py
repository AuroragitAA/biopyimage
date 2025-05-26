#!/usr/bin/env python3
"""
BIOIMAGIN Wolffia Analysis System - Simple Startup Script

This script provides an easy way to start the BIOIMAGIN system with proper
dependency checking and error handling.

Usage:
    python run_bioimagin.py
"""

import importlib
import os
import subprocess
import sys
from pathlib import Path


def print_header():
    """Print startup header"""
    print("=" * 70)
    print("🌱 BIOIMAGIN WOLFFIA ANALYSIS SYSTEM")
    print("   Professional Bioimage Analysis Platform")
    print("=" * 70)
    print()

def check_python_version():
    """Check Python version compatibility"""
    print("🐍 Checking Python version...")
    
    if sys.version_info < (3, 7):
        print(f"❌ Python {sys.version_info.major}.{sys.version_info.minor} detected")
        print("   This system requires Python 3.7 or higher")
        return False
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} - Compatible")
    return True

def check_dependencies():
    """Check if required dependencies are available"""
    print("\n📦 Checking dependencies...")
    
    required_packages = {
        'flask': 'Flask',
        'flask_socketio': 'Flask-SocketIO', 
        'cv2': 'opencv-python',
        'numpy': 'numpy',
        'pandas': 'pandas',
        'scipy': 'scipy',
        'skimage': 'scikit-image',
        'matplotlib': 'matplotlib',
        'PIL': 'Pillow'
    }
    
    missing_packages = []
    
    for module_name, package_name in required_packages.items():
        try:
            importlib.import_module(module_name)
            print(f"✅ {package_name}")
        except ImportError:
            print(f"❌ {package_name} - Missing")
            missing_packages.append(package_name)
    
    # Check optional packages
    optional_packages = {
        'sklearn': 'scikit-learn (ML features)',
        'xgboost': 'xgboost (Advanced ML)',
        'joblib': 'joblib (Model persistence)'
    }
    
    print("\n📊 Optional packages (for advanced features):")
    for module_name, package_name in optional_packages.items():
        try:
            importlib.import_module(module_name)
            print(f"✅ {package_name}")
        except ImportError:
            print(f"⚠️ {package_name} - Not installed (optional)")
    
    if missing_packages:
        print(f"\n❌ Missing required packages: {', '.join(missing_packages)}")
        print("\n💡 Install missing packages with:")
        print(f"   pip install {' '.join(missing_packages)}")
        return False
    
    print("\n✅ All required dependencies are available")
    return True

def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating directories...")
    
    directories = [
        'temp_uploads', 'logs', 'results', 'exports', 'outputs',
        'outputs/debug_images', 'outputs/results', 'outputs/exports',
        'static/temp_images'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ {directory}")

def check_system_files():
    """Check if core system files are present"""
    print("\n📄 Checking system files...")
    
    required_files = [
        'app.py',
        'wolffia_analyzer.py',
        'segmentation.py',
        'image_processor.py',
        'templates/index.html',
        'static/main.js'
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - Missing")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n❌ Missing system files: {', '.join(missing_files)}")
        return False
    
    print("\n✅ All system files present")
    return True

def test_imports():
    """Test if core modules can be imported"""
    print("\n🧪 Testing module imports...")
    
    import_results = {}
    
    try:
        # Test core analyzer
        from wolffia_analyzer import WolffiaAnalyzer
        print("✅ Core analyzer")
        import_results['analyzer'] = True
        
        # Test if analyzer can be instantiated
        test_analyzer = WolffiaAnalyzer(debug_mode=False)
        print("✅ Analyzer instantiation")
        
    except Exception as e:
        print(f"❌ Core analyzer error: {str(e)}")
        import_results['analyzer'] = False
    
    try:
        # Test segmentation
        from segmentation import WolffiaSpecificSegmentation, run_pipeline
        print("✅ Segmentation module")
        import_results['segmentation'] = True
        
        # Test basic segmentation
        test_seg = WolffiaSpecificSegmentation(debug_mode=False)
        print("✅ Segmentation instantiation")
        
    except Exception as e:
        print(f"⚠️ Segmentation warning: {str(e)}")
        import_results['segmentation'] = False
    
    try:
        # Test image processing
        from image_processor import ImageProcessor
        print("✅ Image processor")
        import_results['image_processor'] = True
    except Exception as e:
        print(f"⚠️ Image processor warning: {str(e)}")
        import_results['image_processor'] = False
    
    try:
        # Test Flask app
        from app import app
        print("✅ Flask application")
        import_results['flask_app'] = True
    except Exception as e:
        print(f"❌ Flask app error: {str(e)}")
        import_results['flask_app'] = False
        return False
    
    # Check if we have at least the minimum required components
    if import_results['analyzer'] and import_results['flask_app']:
        print("\n✅ Minimum system requirements met")
        return True
    else:
        print("\n❌ Critical components missing")
        return False

def start_server():
    """Start the BIOIMAGIN server"""
    print("\n🚀 Starting BIOIMAGIN Server...")
    print("   🌐 Server will be available at: http://localhost:5000")
    print("   📊 Live analysis dashboard: http://localhost:5000/")
    print("   Press Ctrl+C to stop the server")
    print("-" * 70)
    
    try:
        # Import and run the Flask app
        from app import app, socketio
        socketio.run(
            app,
            host='0.0.0.0',
            port=5000,
            debug=False,
            use_reloader=False
        )
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
        print("✅ BIOIMAGIN System shutdown complete")
    except Exception as e:
        print(f"\n❌ Server error: {str(e)}")
        return False
    
    return True

def main():
    """Main startup function"""
    print_header()
    
    # Step 1: Check Python version
    if not check_python_version():
        return False
    
    # Step 2: Check dependencies
    if not check_dependencies():
        return False
    
    # Step 3: Create directories
    create_directories()
    
    # Step 4: Check system files
    if not check_system_files():
        return False
    
    # Step 5: Test imports
    if not test_imports():
        return False
    
def run_quick_test():
    """Run a quick system test"""
    print("\n🚀 Running quick system test...")
    
    try:
        import numpy as np

        from wolffia_analyzer import WolffiaAnalyzer
        
        # Create test analyzer
        analyzer = WolffiaAnalyzer(debug_mode=False)
        
        # Create a simple test image
        test_image = np.ones((200, 200, 3), dtype=np.uint8) * 200
        
        # Add a few test objects
        import cv2
        cv2.circle(test_image, (50, 50), 10, (100, 200, 100), -1)
        cv2.circle(test_image, (150, 150), 12, (120, 220, 120), -1)
        
        print("   🔬 Running test analysis...")
        result = analyzer.analyze_single_image(test_image)
        
        if result.get('success'):
            print(f"   ✅ Test analysis successful!")
            print(f"   📊 Detected {result.get('total_cells', 0)} test objects")
            print(f"   ⏱️ Processing time: {result.get('processing_time', 0):.2f}s")
            print(f"   🎯 Quality score: {result.get('quality_score', 0):.3f}")
            return True
        else:
            print(f"   ❌ Test analysis failed: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"   ❌ Test error: {str(e)}")
        return False
    
    # Ask user if they want to start the server
    while True:
        try:
            choice = input("\n🚀 Start the server now? (y/n): ").lower().strip()
            if choice in ['y', 'yes', '']:
                start_server()
                break
            elif choice in ['n', 'no']:
                print("\n💡 To start the server later, run:")
                print("   python run_bioimagin.py")
                print("   or")
                print("   python app.py")
                break
            else:
                print("   Please enter 'y' or 'n'")
        except KeyboardInterrupt:
            print("\n👋 Startup cancelled")
            break
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Startup failed: {str(e)}")
        sys.exit(1)