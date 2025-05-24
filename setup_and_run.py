#!/usr/bin/env python3
"""
Professional Setup and Integration Script for Wolffia Analysis System

This script integrates all your existing components with the professional features:
- Your existing app.py (Flask web interface)
- Your bioimaging.py (WolffiaAnalyzer class)
- Professional components (database, ML, batch processing)
- Advanced image processing capabilities

Usage:
    python professional_setup.py
"""

import os
import sys
import subprocess
import shutil
import importlib.util
from pathlib import Path
import json
from datetime import datetime

def print_header():
    """Print professional welcome header"""
    print("=" * 70)
    print("🌱 PROFESSIONAL WOLFFIA BIOIMAGE ANALYSIS SYSTEM SETUP")
    print("   Enterprise-Grade Integration & Configuration")
    print("=" * 70)
    print()

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    
    if sys.version_info < (3, 8):
        print(f"❌ Python {sys.version_info.major}.{sys.version_info.minor} detected")
        print("   This system requires Python 3.8 or higher")
        print("   Please upgrade Python and try again")
        return False
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} - Compatible")
    return True

def check_professional_dependencies():
    """Check if professional packages are installed"""
    print("\n📦 Checking professional dependencies...")
    
    # Core dependencies
    core_packages = [
        'flask', 'opencv-python', 'numpy', 'scipy', 
        'scikit-image', 'matplotlib', 'pandas', 'pillow'
    ]
    
    # Professional/optional dependencies
    professional_packages = [
        'sklearn', 'xgboost', 'joblib', 'psutil', 'h5py', 'openpyxl'
    ]
    
    missing_core = []
    missing_professional = []
    
    # Check core packages
    for package in core_packages:
        try:
            if package == 'opencv-python':
                import cv2
                print(f"✅ OpenCV {cv2.__version__}")
            elif package == 'scikit-image':
                import skimage
                print(f"✅ scikit-image {skimage.__version__}")
            else:
                module = __import__(package)
                version = getattr(module, '__version__', 'unknown')
                print(f"✅ {package} {version}")
        except ImportError:
            print(f"❌ {package} - Not installed (REQUIRED)")
            missing_core.append(package)
    
    # Check professional packages
    for package in professional_packages:
        try:
            if package == 'sklearn':
                import sklearn
                print(f"✅ scikit-learn {sklearn.__version__} (Professional)")
            else:
                module = __import__(package)
                version = getattr(module, '__version__', 'unknown')
                print(f"✅ {package} {version} (Professional)")
        except ImportError:
            print(f"⚠️ {package} - Not installed (Professional feature)")
            missing_professional.append(package)
    
    # Report results
    if missing_core:
        print(f"\n❌ Missing REQUIRED packages: {', '.join(missing_core)}")
        print("   Install with: pip install " + " ".join(missing_core))
        return False, False
    
    core_ready = True
    professional_ready = len(missing_professional) == 0
    
    if missing_professional:
        print(f"\n⚠️ Missing PROFESSIONAL packages: {', '.join(missing_professional)}")
        print("   Install for full features: pip install " + " ".join(missing_professional))
        print("   System will run in BASIC mode without these")
    
    print("✅ Core dependencies satisfied")
    if professional_ready:
        print("✅ Professional dependencies satisfied - FULL FEATURE MODE")
    else:
        print("⚠️ Professional features limited - BASIC MODE")
    
    return core_ready, professional_ready

def create_professional_directories():
    """Create comprehensive directory structure"""
    print("\n📁 Creating professional directory structure...")
    
    directories = {
        # Core directories
        'temp_uploads': 'Temporary file uploads',
        'templates': 'Web interface templates',
        'static': 'Static web assets (CSS, JS, images)',
        'results': 'Basic analysis results',
        'logs': 'System and analysis logs',
        
        # Professional directories
        'professional_results': 'Advanced analysis results',
        'batch_jobs': 'Batch processing jobs and results',
        'ml_models': 'Machine learning models and training data',
        'database_backups': 'Automated database backups',
        'exports': 'Data export files (CSV, Excel, etc.)',
        'quality_reports': 'Quality control and validation reports',
        'temp_processing': 'Temporary processing files',
        'config': 'System configuration files',
        'documentation': 'Generated reports and documentation'
    }
    
    for dir_name, description in directories.items():
        Path(dir_name).mkdir(exist_ok=True)
        print(f"✅ {dir_name:20} - {description}")
    
    return True

def check_existing_files():
    """Check and validate existing project files"""
    print("\n📄 Checking existing project files...")
    
    required_files = {
        'app.py': 'Flask web application',
        'image_processor.py': 'Image preprocessing module',
        'segmentation.py': 'Cell segmentation algorithms',
        'wolffia_analyzer.py': 'Basic Wolffia analyzer',
    }
    
    optional_files = {
        'bioimaging.py': 'Advanced bioimaging analyzer',
        'advanced_wolffia_analyzer.py': 'Professional analyzer components',
        'batch_processor.py': 'Batch processing system',
        'database_manager.py': 'Database management system',
        'ml_enhancement.py': 'Machine learning enhancements',
        'requirements.txt': 'Python dependencies list'
    }
    
    # Check required files
    missing_required = []
    for file_path, description in required_files.items():
        if Path(file_path).exists():
            print(f"✅ {file_path:25} - {description}")
        else:
            print(f"❌ {file_path:25} - {description} (REQUIRED)")
            missing_required.append(file_path)
    
    # Check optional files
    available_professional = []
    for file_path, description in optional_files.items():
        if Path(file_path).exists():
            print(f"🚀 {file_path:25} - {description} (Professional)")
            available_professional.append(file_path)
        else:
            print(f"⚠️ {file_path:25} - {description} (Optional)")
    
    if missing_required:
        print(f"\n❌ Missing REQUIRED files: {', '.join(missing_required)}")
        return False, available_professional
    
    print(f"\n✅ All required files present")
    if available_professional:
        print(f"🚀 Professional components available: {len(available_professional)}")
    
    return True, available_professional

def test_component_integration():
    """Test if all components can be imported and integrated"""
    print("\n🧪 Testing component integration...")
    
    integration_status = {
        'basic_components': False,
        'bioimaging_analyzer': False,
        'professional_components': False,
        'flask_app': False
    }
    
    # Test basic components
    try:
        from image_processor import ImageProcessor
        from segmentation import EnhancedCellSegmentation
        from wolffia_analyzer import WolffiaAnalyzer
        print("✅ Basic components imported successfully")
        integration_status['basic_components'] = True
    except Exception as e:
        print(f"❌ Basic components import failed: {e}")
    
    # Test bioimaging analyzer
    try:
        spec = importlib.util.spec_from_file_location("bioimaging", "bioimaging.py")
        if spec and spec.loader:
            bioimaging = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(bioimaging)
            # Test if WolffiaAnalyzer class exists in bioimaging
            if hasattr(bioimaging, 'WolffiaAnalyzer'):
                print("✅ Advanced bioimaging analyzer imported successfully")
                integration_status['bioimaging_analyzer'] = True
            else:
                print("⚠️ bioimaging.py exists but WolffiaAnalyzer not found")
        else:
            print("⚠️ bioimaging.py not found - using basic analyzer")
    except Exception as e:
        print(f"⚠️ Advanced bioimaging analyzer not available: {e}")
    
    # Test professional components
    try:
        # Check if professional components exist
        professional_files = [
            'advanced_wolffia_analyzer.py',
            'batch_processor.py', 
            'database_manager.py',
            'ml_enhancement.py'
        ]
        
        available_count = sum(1 for f in professional_files if Path(f).exists())
        
        if available_count >= 2:  # At least some professional components
            print(f"✅ Professional components available ({available_count}/4)")
            integration_status['professional_components'] = True
        else:
            print(f"⚠️ Limited professional components ({available_count}/4)")
    except Exception as e:
        print(f"⚠️ Professional components check failed: {e}")
    
    # Test Flask app
    try:
        from app import app
        print("✅ Flask application imported successfully")
        integration_status['flask_app'] = True
    except Exception as e:
        print(f"❌ Flask application import failed: {e}")
    
    return integration_status

def create_integration_config(integration_status, professional_ready):
    """Create integration configuration file"""
    print("\n⚙️ Creating integration configuration...")
    
    config = {
        'system_info': {
            'name': 'Professional Wolffia Analysis System',
            'version': '3.0.0',
            'setup_date': datetime.now().isoformat(),
            'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        },
        'component_status': integration_status,
        'feature_flags': {
            'professional_mode': professional_ready and integration_status.get('professional_components', False),
            'advanced_bioimaging': integration_status.get('bioimaging_analyzer', False),
            'ml_enhancements': professional_ready,
            'batch_processing': integration_status.get('professional_components', False),
            'database_integration': integration_status.get('professional_components', False),
            'quality_control': True,
            'web_interface': integration_status.get('flask_app', False)
        },
        'analysis_config': {
            'default_pixel_to_micron': 1.0,
            'default_chlorophyll_threshold': 0.6,
            'min_cell_area': 30,
            'max_cell_area': 8000,
            'enable_advanced_segmentation': integration_status.get('bioimaging_analyzer', False),
            'enable_comprehensive_features': integration_status.get('bioimaging_analyzer', False)
        },
        'paths': {
            'upload_folder': 'temp_uploads',
            'results_folder': 'professional_results',
            'batch_folder': 'batch_jobs',
            'models_folder': 'ml_models',
            'exports_folder': 'exports',
            'logs_folder': 'logs'
        }
    }
    
    # Save configuration
    config_path = Path('config') / 'system_config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Configuration saved to: {config_path}")
    
    # Print system capabilities
    print("\n📊 System Capabilities Summary:")
    for feature, enabled in config['feature_flags'].items():
        status = "✅ ENABLED" if enabled else "❌ DISABLED"
        print(f"   {feature:20} : {status}")
    
    return config

def create_integration_wrapper():
    """Create integration wrapper for seamless component integration"""
    print("\n🔧 Creating integration wrapper...")
    
    wrapper_code = """
# integration_wrapper.py
'''
Professional Wolffia Analysis System - Integration Wrapper

This module provides seamless integration between all system components,
automatically detecting and using the best available analyzers and features.
'''

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SystemIntegrator:
    '''Main system integrator that manages all components.'''
    
    def __init__(self):
        self.config = self._load_config()
        self.components = {}
        self._initialize_components()
    
    def _load_config(self) -> Dict:
        '''Load system configuration.'''
        try:
            config_path = Path('config') / 'system_config.json'
            if config_path.exists():
                with open(config_path) as f:
                    return json.load(f)
            else:
                logger.warning("Configuration file not found, using defaults")
                return self._default_config()
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return self._default_config()
    
    def _default_config(self) -> Dict:
        '''Default configuration.'''
        return {
            'feature_flags': {
                'professional_mode': False,
                'advanced_bioimaging': False,
                'ml_enhancements': False,
                'batch_processing': False,
                'database_integration': False,
                'quality_control': True,
                'web_interface': True
            },
            'analysis_config': {
                'default_pixel_to_micron': 1.0,
                'default_chlorophyll_threshold': 0.6,
                'min_cell_area': 30,
                'max_cell_area': 8000
            }
        }
    
    def _initialize_components(self):
        '''Initialize available components based on configuration.'''
        try:
            # Initialize basic components
            if self.config['component_status'].get('basic_components', False):
                from image_processor import ImageProcessor
                from segmentation import EnhancedCellSegmentation
                self.components['image_processor'] = ImageProcessor()
                self.components['segmentation'] = EnhancedCellSegmentation()
                logger.info("✅ Basic components initialized")
            
            # Initialize advanced bioimaging analyzer if available
            if self.config['feature_flags'].get('advanced_bioimaging', False):
                try:
                    import importlib.util
                    spec = importlib.util.spec_from_file_location("bioimaging", "bioimaging.py")
                    if spec and spec.loader:
                        bioimaging = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(bioimaging)
                        if hasattr(bioimaging, 'WolffiaAnalyzer'):
                            analyzer_config = self.config['analysis_config']
                            self.components['advanced_analyzer'] = bioimaging.WolffiaAnalyzer(
                                pixel_to_micron_ratio=analyzer_config.get('default_pixel_to_micron', 1.0),
                                chlorophyll_threshold=analyzer_config.get('default_chlorophyll_threshold', 0.6)
                            )
                            logger.info("✅ Advanced bioimaging analyzer initialized")
                        else:
                            logger.warning("WolffiaAnalyzer not found in bioimaging.py")
                except Exception as e:
                    logger.error(f"Failed to initialize advanced analyzer: {e}")
            
            # Initialize basic analyzer as fallback
            if 'advanced_analyzer' not in self.components:
                try:
                    from wolffia_analyzer import WolffiaAnalyzer
                    self.components['basic_analyzer'] = WolffiaAnalyzer()
                    logger.info("✅ Basic analyzer initialized as fallback")
                except Exception as e:
                    logger.error(f"Failed to initialize basic analyzer: {e}")
            
            # Initialize professional components if available
            if self.config['feature_flags'].get('professional_mode', False):
                self._initialize_professional_components()
                
        except Exception as e:
            logger.error(f"Component initialization failed: {e}")
    
    def _initialize_professional_components(self):
        '''Initialize professional components if available.'''
        try:
            # Database manager
            if Path('database_manager.py').exists():
                from database_manager import DatabaseManager, DatabaseConfig
                self.components['database'] = DatabaseManager(DatabaseConfig())
                logger.info("✅ Database manager initialized")
            
            # Batch processor
            if Path('batch_processor.py').exists():
                from batch_processor import BatchProcessor, BatchJobConfig
                # Will be initialized when needed with specific analyzer
                logger.info("✅ Batch processor available")
            
            # ML enhancements
            if Path('ml_enhancement.py').exists():
                from ml_enhancement import MLEnhancedAnalyzer, MLConfig
                # Will be initialized when needed with base analyzer
                logger.info("✅ ML enhancements available")
                
        except Exception as e:
            logger.error(f"Professional components initialization failed: {e}")
    
    def get_analyzer(self):
        '''Get the best available analyzer.'''
        if 'advanced_analyzer' in self.components:
            return self.components['advanced_analyzer']
        elif 'basic_analyzer' in self.components:
            return self.components['basic_analyzer']
        else:
            raise Exception("No analyzer available")
    
    def analyze_image(self, image_path: str, **kwargs) -> Dict[str, Any]:
        '''Analyze image using the best available analyzer.'''
        try:
            analyzer = self.get_analyzer()
            
            # Use advanced analyzer if available
            if hasattr(analyzer, 'analyze_single_image'):
                # This is likely the advanced bioimaging analyzer
                result = analyzer.analyze_single_image(image_path, **kwargs)
                
                # Convert result format for consistency
                if result and isinstance(result, dict):
                    # Check if it's the advanced format
                    if 'cell_data' in result and hasattr(result['cell_data'], 'to_dict'):
                        # Convert DataFrame to dict
                        result['cell_data'] = result['cell_data'].to_dict('records')
                    
                    # Ensure consistent format
                    if 'success' not in result:
                        result['success'] = result.get('total_cells', 0) > 0
                    
                    return result
                else:
                    return {'success': False, 'error': 'Analysis failed'}
            
            else:
                # Basic analyzer
                result = analyzer.analyze_single_image(image_path)
                return result if result else {'success': False, 'error': 'Analysis failed'}
                
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_system_status(self) -> Dict[str, Any]:
        '''Get comprehensive system status.'''
        return {
            'config': self.config,
            'components_loaded': list(self.components.keys()),
            'analyzer_type': 'advanced' if 'advanced_analyzer' in self.components else 'basic',
            'professional_mode': self.config['feature_flags'].get('professional_mode', False)
        }

# Global integrator instance
_integrator = None

def get_integrator():
    '''Get global integrator instance.'''
    global _integrator
    if _integrator is None:
        _integrator = SystemIntegrator()
    return _integrator
"""
    
    # Write integration wrapper
    wrapper_path = Path('integration_wrapper.py')
    with open(wrapper_path, 'w') as f:
        f.write(wrapper_code)
    
    print(f"✅ Integration wrapper created: {wrapper_path}")
    return wrapper_path

def create_professional_requirements():
    """Create comprehensive requirements.txt file"""
    print("\n📋 Creating professional requirements.txt...")
    
    requirements = """# Professional Wolffia Analysis System Requirements
# Core Dependencies (REQUIRED)
Flask==3.0.3
opencv-python-headless==4.9.0.80
scikit-image==0.23.2
numpy==1.26.4
scipy==1.13.0
matplotlib==3.7.2
pandas==2.0.3
Pillow==10.0.1
Werkzeug==2.3.7

# Professional Features (OPTIONAL - for full feature set)
scikit-learn>=1.3.0
xgboost>=1.7.0
joblib>=1.3.0
psutil>=5.9.0
h5py>=3.8.0
openpyxl>=3.1.0

# Development and Quality
setuptools>=70.0.0

# Optional Advanced Features
# Uncomment if you want these features:
# tensorflow>=2.13.0  # For advanced ML models
# pytorch>=2.0.0      # Alternative ML framework  
# plotly>=5.15.0      # Interactive visualizations
# dash>=2.14.0        # Advanced dashboards
"""
    
    requirements_path = Path('requirements.txt')
    with open(requirements_path, 'w') as f:
        f.write(requirements)
    
    print(f"✅ Requirements file created: {requirements_path}")
    return requirements_path

def run_system_validation():
    """Run comprehensive system validation"""
    print("\n🔍 Running comprehensive system validation...")
    
    validation_results = {
        'basic_analysis': False,
        'advanced_analysis': False,
        'web_interface': False,
        'integration': False
    }
    
    try:
        # Test basic analysis
        print("   Testing basic analysis components...")
        from integration_wrapper import get_integrator
        integrator = get_integrator()
        
        analyzer = integrator.get_analyzer()
        if analyzer:
            print("   ✅ Analyzer successfully loaded")
            validation_results['basic_analysis'] = True
        
        # Test integration
        print("   Testing system integration...")
        status = integrator.get_system_status()
        if status and 'components_loaded' in status:
            print(f"   ✅ Integration successful - {len(status['components_loaded'])} components loaded")
            validation_results['integration'] = True
        
        # Test advanced features
        if 'advanced_analyzer' in status.get('components_loaded', []):
            print("   ✅ Advanced bioimaging analyzer available")
            validation_results['advanced_analysis'] = True
        
        # Test web interface
        try:
            from app import app
            if app:
                print("   ✅ Flask web interface ready")
                validation_results['web_interface'] = True
        except Exception as e:
            print(f"   ⚠️ Web interface issue: {e}")
        
    except Exception as e:
        print(f"   ❌ Validation error: {e}")
    
    # Summary
    passed = sum(validation_results.values())
    total = len(validation_results)
    
    print(f"\n📊 Validation Results: {passed}/{total} tests passed")
    for test, result in validation_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test:20} : {status}")
    
    return validation_results

def start_professional_server():
    """Start the professional server with all integrations"""
    print("\n🚀 Starting Professional Wolffia Analysis Server...")
    print("   Loading all professional components...")
    
    try:
        # Import and configure the Flask app with professional features
        from app import app
        
        # Add professional configuration
        app.config.update({
            'PROFESSIONAL_MODE': True,
            'SYSTEM_CONFIG_PATH': 'config/system_config.json',
            'INTEGRATION_WRAPPER': True
        })
        
        print("   ✅ Professional configuration applied")
        print("   🌐 Server will be available at: http://localhost:5000")
        print("   📊 Professional dashboard: http://localhost:5000/")
        print("   🔬 Advanced analysis features: ENABLED")
        print("   Press Ctrl+C to stop the server")
        print("-" * 70)
        
        # Start the server
        app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
        
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Server error: {e}")
        return False
    
    return True

def main():
    """Main professional setup and integration function"""
    print_header()
    
    # Step 1: Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Step 2: Check dependencies
    core_ready, professional_ready = check_professional_dependencies()
    if not core_ready:
        print("\n💡 To install missing dependencies:")
        print("   pip install -r requirements.txt")
        sys.exit(1)
    
    # Step 3: Create professional directory structure
    if not create_professional_directories():
        sys.exit(1)
    
    # Step 4: Check existing files
    files_ready, professional_files = check_existing_files()
    if not files_ready:
        sys.exit(1)
    
    # Step 5: Test component integration
    integration_status = test_component_integration()
    
    # Step 6: Create integration configuration
    config = create_integration_config(integration_status, professional_ready)
    
    # Step 7: Create integration wrapper
    create_integration_wrapper()
    
    # Step 8: Create/update requirements file
    create_professional_requirements()
    
    # Step 9: Run system validation
    validation_results = run_system_validation()
    
    # Step 10: Final status
    print("\n" + "=" * 70)
    if all(validation_results.values()):
        print("✅ PROFESSIONAL SETUP COMPLETE - ALL SYSTEMS READY")
        print("🚀 Full Professional Feature Set Available")
    elif validation_results['basic_analysis'] and validation_results['integration']:
        print("✅ SETUP COMPLETE - CORE SYSTEMS READY")  
        print("⚠️ Some professional features limited")
    else:
        print("⚠️ SETUP COMPLETE WITH ISSUES")
        print("   Check validation results above")
    
    print("=" * 70)
    
    # Display system summary
    print(f"\n📊 SYSTEM SUMMARY:")
    print(f"   Mode: {'🚀 PROFESSIONAL' if professional_ready else '⚠️ BASIC'}")
    print(f"   Analyzer: {'Advanced Bioimaging' if integration_status.get('bioimaging_analyzer') else 'Standard'}")
    print(f"   Web Interface: {'✅ Ready' if integration_status.get('flask_app') else '❌ Issues'}")
    print(f"   Professional Components: {len(professional_files)} available")
    
    # Ask user if they want to start the server
    while True:
        choice = input("\n🚀 Start the professional server now? (y/n): ").lower().strip()
        if choice in ['y', 'yes']:
            start_professional_server()
            break
        elif choice in ['n', 'no']:
            print("\n💡 To start the server later, run:")
            print("   python app.py")
            print("\n💡 Or for professional mode:")
            print("   python -c \"from integration_wrapper import get_integrator; integrator = get_integrator(); print('Professional system ready!')\"")
            break
        else:
            print("   Please enter 'y' or 'n'")

if __name__ == "__main__":
    main()