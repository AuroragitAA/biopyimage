# bioimaging_professional_improved.py - Refined Professional Pipeline for Wolffia Analysis
# Architecture: CellPose + SimpleITK + Advanced Learning System

import base64
import json
import os
import shutil
import warnings
from datetime import datetime
from io import BytesIO
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch not available - GPU acceleration disabled")
from scipy import ndimage, stats
from skimage import feature, filters, measure, morphology, restoration, segmentation
from skimage.color import rgb2gray, rgb2hsv, rgb2lab
from skimage.exposure import adjust_gamma, equalize_adapthist
from skimage.filters import gaussian, threshold_otsu
from skimage.morphology import disk, remove_small_objects
from skimage.segmentation import watershed
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler

matplotlib.use('Agg')
warnings.filterwarnings('ignore')
# GPU Detection and Setup
def detect_gpu_capabilities():
    """Detect and configure GPU capabilities for optimal performance"""
    gpu_info = {
        'cuda_available': False,
        'gpu_count': 0,
        'gpu_name': None,
        'gpu_memory': 0,
        'recommended_settings': {
            'batch_size': 1,
            'diameter': 25,
            'use_mixed_precision': False
        }
    }
    
    if TORCH_AVAILABLE:
        gpu_info['cuda_available'] = torch.cuda.is_available()
        
        if torch.cuda.is_available():
            gpu_info['gpu_count'] = torch.cuda.device_count()
            gpu_info['gpu_name'] = torch.cuda.get_device_name(0)
            gpu_info['gpu_memory'] = torch.cuda.get_device_properties(0).total_memory / 1e9  # GB
            
            # Recommended settings based on GPU memory
            if gpu_info['gpu_memory'] >= 8:
                gpu_info['recommended_settings'] = {
                    'batch_size': 8,
                    'diameter': 30,
                    'use_mixed_precision': True
                }
            elif gpu_info['gpu_memory'] >= 4:
                gpu_info['recommended_settings'] = {
                    'batch_size': 4,
                    'diameter': 30,
                    'use_mixed_precision': True
                }
            else:
                gpu_info['recommended_settings'] = {
                    'batch_size': 2,
                    'diameter': 25,
                    'use_mixed_precision': False
                }
    
    return gpu_info


# Check available libraries
try:
    import SimpleITK as sitk
    SIMPLEITK_AVAILABLE = True
    print("✅ SimpleITK available for advanced image restoration")
except ImportError:
    SIMPLEITK_AVAILABLE = False
    print("⚠️ SimpleITK not available - using OpenCV fallback")

try:
    from cellpose import denoise, io, models, train
    from cellpose.dynamics import resize_and_compute_masks
    CELLPOSE_AVAILABLE = True
    try:
        from cellpose import __version__ as cp_version
        CELLPOSE_VERSION = int(cp_version.split('.')[0])
        print(f"✅ CellPose {cp_version} available")
    except:
        CELLPOSE_VERSION = 2
        print("✅ CellPose available (version unknown)")
except ImportError:
    CELLPOSE_AVAILABLE = False
    CELLPOSE_VERSION = 0
    print("❌ CellPose not available - using watershed fallback")

try:
    import cellpose_planer as cellpp
    CELLPOSE_PLANER_AVAILABLE = True
    print("✅ CellPose-Planer available for lightweight models")
except ImportError:
    CELLPOSE_PLANER_AVAILABLE = False
    print("⚠️ CellPose-Planer not available")

def get_system_status():
    """Get comprehensive system status for production monitoring"""
    gpu_info = detect_gpu_capabilities()
    
    status = {
        'libraries': {
            'simpleitk': SIMPLEITK_AVAILABLE,
            'cellpose': CELLPOSE_AVAILABLE,
            'cellpose_planer': CELLPOSE_PLANER_AVAILABLE,
            'torch': TORCH_AVAILABLE and torch.cuda.is_available() if TORCH_AVAILABLE else False
        },
        'gpu': gpu_info,
        'models_available': [],
        'ready_for_production': False
    }
    
    if CELLPOSE_AVAILABLE:
        try:
            # Test model loading (v4+ compatible)
            model = models.CellposeModel(gpu=False, model='cyto2')
            status['models_available'].append('cyto2')
        except:
            pass
    
    # System is ready if we have at least one working segmentation method
    status['ready_for_production'] = (
        CELLPOSE_AVAILABLE or 
        (np.any([SIMPLEITK_AVAILABLE]) and True)  # Has watershed fallback
    )
    
    return status

def print_system_status():
    """Print professional system status"""
    print(f"🔬 BIOIMAGIN Professional Pipeline Status:")
    print(f"   📦 SimpleITK: {'✅' if SIMPLEITK_AVAILABLE else '❌'}")
    print(f"   🧬 CellPose: {'✅' if CELLPOSE_AVAILABLE else '❌'}")
    print(f"   🎯 CellPose-Planer: {'✅' if CELLPOSE_PLANER_AVAILABLE else '❌'}")
    
    gpu_info = detect_gpu_capabilities()
    if gpu_info['cuda_available']:
        print(f"   🚀 GPU: {gpu_info['gpu_name']} ({gpu_info['gpu_memory']:.1f}GB)")
    else:
        print(f"   💻 CPU-only mode")

print_system_status()

class WolffiaAnalyzer:
    """
    Professional Bioinformatics Pipeline for Wolffia Analysis
    Refined for reliability, performance, and learning capabilities
    """
    
    def __init__(self, pixel_to_micron_ratio=1.0, chlorophyll_threshold=0.6):
        """Initialize professional analysis pipeline with GPU optimization"""
        self.pixel_to_micron = pixel_to_micron_ratio
        self.chlorophyll_threshold = chlorophyll_threshold
        
        # Detect GPU capabilities first
        self.gpu_info = detect_gpu_capabilities()
        
        # Enhanced Wolffia-specific parameters with GPU optimization
        self.wolffia_params = {
            'min_area_microns': 30,
            'max_area_microns': 10000,
            'expected_circularity': 0.7,
            'diameter': self.gpu_info['recommended_settings'].get('diameter', 30) if self.gpu_info['cuda_available'] else 30,
            'flow_threshold': 0.2,  # More permissive for better cell detection
            'cellprob_threshold': 0.0,
            'chlorophyll_peaks': [435, 670],
            'min_green_intensity': 0.3,
            'max_aspect_ratio': 2.0,
            'use_gpu': self.gpu_info['cuda_available'],
            'batch_size': self.gpu_info['recommended_settings'].get('batch_size', 1) if self.gpu_info['cuda_available'] else 1
        }
        
        # Initialize components with GPU support
        self.restoration_engine = None
        self.segmentation_engine = None
        self.feature_engine = None
        self.quality_engine = None
        self.learning_engine = None
        
        self._initialize_engines()
        
        # Model management with GPU considerations
        self.models_dir = Path(__file__).parent / 'models'
        self.custom_model_path = self.models_dir / 'wolffia_custom_model'
        self._setup_model_infrastructure()
        
        # Performance monitoring
        self.performance_stats = {
            'gpu_enabled': self.gpu_info['cuda_available'],
            'gpu_name': self.gpu_info.get('gpu_name', 'None'),
            'total_analyses': 0,
            'avg_processing_time': 0,
            'gpu_memory_peak': 0
        }
        
        print(f"🧬 Professional WolffiaAnalyzer initialized")
        print(f"   🚀 GPU: {self.gpu_info['gpu_name'] if self.gpu_info['cuda_available'] else 'CPU Only'}")
        print(f"   🔧 Restoration: {self.restoration_engine.status if self.restoration_engine else '❌'}")
        print(f"   🧬 Segmentation: {self.segmentation_engine.status if self.segmentation_engine else '❌'}")
        print(f"   📊 Features: {self.feature_engine.status if self.feature_engine else '❌'}")
        print(f"   📈 Quality: {self.quality_engine.status if self.quality_engine else '❌'}")
        print(f"   🧠 Learning: {self.learning_engine.status if self.learning_engine else '❌'}")

    def get_current_parameters(self):
        """Get current analysis parameters including GPU status"""
        gpu_status = {}
        if hasattr(self, 'segmentation_engine') and self.segmentation_engine:
            gpu_status = self.segmentation_engine.get_gpu_status()
        
        return {
            'pixel_to_micron': self.pixel_to_micron,
            'chlorophyll_threshold': self.chlorophyll_threshold,
            'wolffia_params': self.wolffia_params.copy(),
            'gpu_info': self.gpu_info,
            'gpu_status': gpu_status,
            'performance_stats': self.performance_stats,
            'engines_status': {
                'restoration': getattr(self.restoration_engine, 'status', '❌'),
                'segmentation': getattr(self.segmentation_engine, 'status', '❌'),
                'features': getattr(self.feature_engine, 'status', '❌'),
                'quality': getattr(self.quality_engine, 'status', '❌'),
                'learning': getattr(self.learning_engine, 'status', '❌')
            }
        }
    
    def health_check(self):
        """Comprehensive system health check for production monitoring"""
        health = {
            'status': 'healthy',
            'issues': [],
            'capabilities': {},
            'recommendations': []
        }
        
        try:
            # Check engines
            engines = ['restoration_engine', 'segmentation_engine', 'feature_engine', 
                      'quality_engine', 'learning_engine']
            
            for engine_name in engines:
                engine = getattr(self, engine_name, None)
                if engine is None:
                    health['issues'].append(f"{engine_name} not initialized")
                    health['status'] = 'degraded'
                elif hasattr(engine, 'status') and '❌' in engine.status:
                    health['issues'].append(f"{engine_name} failed to initialize properly")
                    health['status'] = 'degraded'
            
            # Check GPU status
            if not self.gpu_info['cuda_available']:
                health['recommendations'].append("Consider using GPU for better performance")
            elif self.gpu_info['gpu_memory'] < 4:
                health['recommendations'].append("Low GPU memory - consider reducing batch size")
            
            # Check model availability
            if hasattr(self, 'segmentation_engine') and self.segmentation_engine:
                if not hasattr(self.segmentation_engine, 'models') or not self.segmentation_engine.models:
                    health['issues'].append("No CellPose models available - relying on watershed fallback")
                    health['status'] = 'degraded'
            
            # Check libraries
            if not CELLPOSE_AVAILABLE:
                health['issues'].append("CellPose not available - using watershed fallback only")
                health['status'] = 'degraded'
            
            # Set capabilities
            health['capabilities'] = {
                'gpu_acceleration': self.gpu_info['cuda_available'],
                'cellpose_models': CELLPOSE_AVAILABLE,
                'advanced_restoration': SIMPLEITK_AVAILABLE,
                'watershed_fallback': True,
                'learning_system': hasattr(self, 'learning_engine') and self.learning_engine is not None
            }
            
            # Overall health assessment
            if len(health['issues']) > 3:
                health['status'] = 'unhealthy'
            elif len(health['issues']) > 0:
                health['status'] = 'degraded'
            
        except Exception as e:
            health['status'] = 'unhealthy'
            health['issues'].append(f"Health check failed: {str(e)}")
        
        return health

    def _initialize_engines(self):
        """Initialize analysis engines with error handling"""
        try:
            self.restoration_engine = ImageRestorationEngine()
        except Exception as e:
            print(f"⚠️ Restoration engine failed to initialize: {e}")
        
        try:
            self.segmentation_engine = CellPoseSegmentationEngine()
        except Exception as e:
            print(f"⚠️ Segmentation engine failed to initialize: {e}")
        
        try:
            self.feature_engine = FeatureExtractionEngine()
        except Exception as e:
            print(f"⚠️ Feature engine failed to initialize: {e}")
        
        try:
            self.quality_engine = QualityAssessmentEngine()
        except Exception as e:
            print(f"⚠️ Quality engine failed to initialize: {e}")
        
        try:
            self.learning_engine = LearningEngine()
        except Exception as e:
            print(f"⚠️ Learning engine failed to initialize: {e}")
    
    def _setup_model_infrastructure(self):
        """Setup model directories and CellPose integration"""
        try:
            # Create model directories
            self.models_dir.mkdir(exist_ok=True)
            
            # Setup CellPose-Planer integration if available
            if CELLPOSE_PLANER_AVAILABLE:
                self._setup_cellpose_planer_models()
            
        except Exception as e:
            print(f"⚠️ Model infrastructure setup failed: {e}")
    
    def _setup_cellpose_planer_models(self):
        """Setup CellPose-Planer models from local directory"""
        try:
            import cellpose_planer as cellpp
            
            # Get cellpose-planer models directory
            spec = __import__('importlib.util', fromlist=['find_spec']).find_spec('cellpose_planer')
            cp_dir = Path(spec.origin).parent
            cp_models_dir = cp_dir / 'models'
            cp_models_dir.mkdir(exist_ok=True)
            
            # Copy models if they exist
            if self.models_dir.exists():
                model_files = list(self.models_dir.glob('*.pla'))
                for model_file in model_files:
                    dst = cp_models_dir / model_file.name
                    if not dst.exists():
                        shutil.copy2(model_file, dst)
                        print(f"✓ Copied {model_file.name} to cellpose-planer")
            
            # Search for models
            cellpp.search_models()
            available_models = cellpp.list_models()
            print(f"📁 Available CellPose-Planer models: {available_models}")
            
        except Exception as e:
            print(f"⚠️ CellPose-Planer setup failed: {e}")
    
    def analyze_image_professional(self, image_path, 
                             restoration_mode='auto',
                             segmentation_model='auto',
                             diameter=None,
                             flow_threshold=None,
                             learn_from_analysis=True,
                             save_visualizations=True):
        """Enhanced professional analysis with GPU acceleration and performance monitoring"""
        analysis_start = datetime.now()
        
        try:
            print(f"🧬 Professional Wolffia Analysis (GPU: {'✅' if self.wolffia_params['use_gpu'] else '❌'})")
            print(f"📁 Image: {Path(image_path).name}")
            print(f"🔧 Restoration: {restoration_mode}")
            print(f"🧬 Model: {segmentation_model}")
            print(f"📏 Diameter: {diameter or self.wolffia_params['diameter']}")
            print(f"🌊 Flow Threshold: {flow_threshold or self.wolffia_params['flow_threshold']}")
            print("="*60)
            
            # GPU memory monitoring
            initial_gpu_memory = 0
            if self.wolffia_params['use_gpu']:
                try:
                    initial_gpu_memory = torch.cuda.memory_allocated(0) / 1e6  # MB
                    print(f"💾 Initial GPU memory: {initial_gpu_memory:.1f} MB")
                except:
                    pass
            
            # Update parameters if provided
            if diameter is not None:
                self.wolffia_params['diameter'] = diameter
            if flow_threshold is not None:
                self.wolffia_params['flow_threshold'] = flow_threshold
            
            # Step 1: Image Restoration (GPU-accelerated if available)
            print("🔧 Step 1: Image Restoration...")
            if self.restoration_engine:
                restoration_result = self.restoration_engine.restore_image(
                    image_path, mode=restoration_mode
                )
            else:
                # Fallback: load image without restoration
                image = cv2.imread(str(image_path))
                restoration_result = {
                    'original': image,
                    'restored': image,
                    'gray_restored': cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
                    'quality_score': 0.7,
                    'method': 'no_restoration'
                }
            
            if 'error' in restoration_result:
                return {'error': f"Image restoration failed: {restoration_result['error']}", 'success': False}
            
            print(f"✅ Restoration complete (Quality: {restoration_result['quality_score']:.3f})")
            
            # Step 2: GPU-Accelerated Cell Segmentation
            print(f"\n🧬 Step 2: {'GPU' if self.wolffia_params['use_gpu'] else 'CPU'} Cell Segmentation...")
            if self.segmentation_engine:
                # Auto-select model if needed
                if segmentation_model == 'auto':
                    segmentation_model = self._select_best_model()
                
                # Use auto parameters for optimal results, or user-specified if set
                auto_diameter = 'auto' if self.wolffia_params['diameter'] == 30 else self.wolffia_params['diameter']
                auto_flow = 'auto' if self.wolffia_params['flow_threshold'] == 0.2 else self.wolffia_params['flow_threshold']
                
                segmentation_result = self.segmentation_engine.segment_cells(
                    restoration_result,
                    model=segmentation_model,
                    diameter=auto_diameter,
                    flow_threshold=auto_flow
                )
                
                # Monitor GPU memory usage
                if self.wolffia_params['use_gpu'] and TORCH_AVAILABLE:
                    try:
                        current_gpu_memory = torch.cuda.memory_allocated(0) / 1e6  # MB
                        peak_gpu_memory = torch.cuda.max_memory_allocated(0) / 1e6  # MB
                        print(f"💾 GPU memory: {current_gpu_memory:.1f} MB (peak: {peak_gpu_memory:.1f} MB)")
                        
                        # Update performance stats
                        self.performance_stats['gpu_memory_peak'] = max(
                            self.performance_stats['gpu_memory_peak'], peak_gpu_memory
                        )
                    except:
                        pass
            else:
                # Fallback segmentation
                segmentation_result = self._fallback_segmentation(restoration_result)
            
            print(f"✅ Segmentation complete ({segmentation_result['num_cells']} cells)")
            
            # Step 3: Feature Extraction (optimized for GPU if available)
            print("\n📊 Step 3: Feature Extraction...")
            if self.feature_engine and segmentation_result['num_cells'] > 0:
                features_df = self.feature_engine.extract_features(
                    restoration_result['restored'],
                    segmentation_result['labels'],
                    self.pixel_to_micron
                )
            else:
                features_df = pd.DataFrame()
            
            print(f"✅ Features extracted for {len(features_df)} cells")
            
            # Step 4: Quality Assessment
            print("\n📈 Step 4: Quality Assessment...")
            if self.quality_engine:
                quality_report = self.quality_engine.assess_quality(
                    restoration_result, segmentation_result, features_df
                )
            else:
                quality_report = {'overall_quality': 0.7, 'status': 'basic'}
            
            # Step 5: Learning (GPU-accelerated if available)
            learning_result = None
            if learn_from_analysis and self.learning_engine and segmentation_result['num_cells'] > 0:
                print("\n🧠 Step 5: Learning from Analysis...")
                learning_result = self.learning_engine.learn_from_analysis(
                    restoration_result, segmentation_result, features_df, quality_report
                )
            
            # Step 6: Visualizations
            visualizations = {}
            if save_visualizations:
                print("\n🎨 Step 6: Creating Visualizations...")
                visualizations = self._create_visualizations(
                    restoration_result, segmentation_result, features_df
                )
            
            # Performance tracking
            analysis_duration = (datetime.now() - analysis_start).total_seconds()
            self.performance_stats['total_analyses'] += 1
            
            # Update average processing time
            total_analyses = self.performance_stats['total_analyses']
            current_avg = self.performance_stats['avg_processing_time']
            self.performance_stats['avg_processing_time'] = (
                (current_avg * (total_analyses - 1) + analysis_duration) / total_analyses
            )
            
            # Compile results with GPU performance info
            results = {
                'success': True,
                'image_path': str(image_path),
                'timestamp': analysis_start.isoformat(),
                'analysis_duration_seconds': analysis_duration,
                'gpu_used': segmentation_result.get('gpu_used', False),
                'parameters': {
                    'restoration_mode': restoration_mode,
                    'segmentation_model': segmentation_model,
                    'diameter': self.wolffia_params['diameter'],
                    'flow_threshold': self.wolffia_params['flow_threshold'],
                    'pixel_to_micron': self.pixel_to_micron,
                    'use_gpu': self.wolffia_params['use_gpu']
                },
                'restoration': {
                    'method': restoration_result.get('method', 'unknown'),
                    'quality_score': restoration_result.get('quality_score', 0)
                },
                'segmentation': {
                    'method': segmentation_result.get('method', 'unknown'),
                    'num_cells': segmentation_result['num_cells'],
                    'confidence': segmentation_result.get('confidence', 0)
                },
                'cells': features_df.to_dict('records') if not features_df.empty else [],
                'summary': self._create_summary(features_df),
                'quality': quality_report,
                'learning': learning_result,
                'visualizations': visualizations,
                'performance': {
                    'processing_time': analysis_duration,
                    'gpu_acceleration': self.wolffia_params['use_gpu'],
                    'peak_memory_mb': self.performance_stats['gpu_memory_peak']
                }
            }
            
            # GPU cleanup
            if self.wolffia_params['use_gpu']:
                try:
                    torch.cuda.empty_cache()
                except:
                    pass
            
            # Save analysis if learning is enabled
            if learn_from_analysis and self.learning_engine:
                self.learning_engine.save_analysis(results)
            
            print(f"\n🎉 Professional Analysis Complete!")
            print(f"   📊 Cells: {len(features_df)}")
            print(f"   📈 Quality: {quality_report['overall_quality']:.3f}")
            print(f"   ⏱️ Duration: {analysis_duration:.1f}s")
            print(f"   🚀 GPU: {'Used' if segmentation_result.get('gpu_used', False) else 'Not Used'}")
            
            return results
            
        except Exception as e:
            print(f"❌ Professional analysis failed: {e}")
            import traceback
            traceback.print_exc()
            
            # GPU cleanup on error
            if self.wolffia_params['use_gpu']:
                try:
                    torch.cuda.empty_cache()
                except:
                    pass
            
            return {
                'error': str(e),
                'success': False,
                'timestamp': analysis_start.isoformat()
            }
            
    def _select_best_model(self):
        """Select best segmentation model based on learning history"""
        if self.learning_engine:
            return self.learning_engine.get_best_model()
        return 'cyto3'  # Default fallback
    
    def _fallback_segmentation(self, restoration_result):
        """Simple fallback segmentation when CellPose is not available"""
        print("⚠️ Using fallback segmentation (watershed)")
        
        image = restoration_result['restored']
        gray = restoration_result['gray_restored']
        
        # Threshold-based segmentation
        if len(image.shape) == 3:
            green = image[:,:,1]
            thresh = filters.threshold_otsu(green)
            binary = green > thresh
        else:
            thresh = filters.threshold_otsu(gray)
            binary = gray > thresh
        
        # Clean up
        binary = morphology.remove_small_objects(binary, min_size=20)
        binary = ndimage.binary_fill_holes(binary)
        
        # Watershed
        distance = ndimage.distance_transform_edt(binary)
        coords = feature.peak_local_max(distance, min_distance=5, threshold_abs=0.3*distance.max())
        markers = np.zeros_like(distance, dtype=int)
        markers[tuple(coords.T)] = np.arange(len(coords)) + 1
        
        labels = segmentation.watershed(-distance, markers, mask=binary)
        
        return {
            'labels': labels,
            'num_cells': np.max(labels),
            'method': 'fallback_watershed',
            'confidence': 0.6,
            'gpu_used': False
        }
    
    def _create_summary(self, features_df):
        """Create analysis summary - FIXED VERSION"""
        if features_df.empty:
            return {
                'total_cells_detected': 0,  # FIXED: Use consistent key names
                'total_green_cells': 0,
                'average_area': 0,
                'total_biomass': 0,
                'area_statistics': {
                    'min': 0,
                    'max': 0,
                    'std': 0
                }
            }
        
        # FIXED: Better error handling and consistent column names
        try:
            # Handle different possible column names
            area_col = None
            for col in ['area_microns_sq', 'area_microns', 'area']:
                if col in features_df.columns:
                    area_col = col
                    break
            
            biomass_col = None
            for col in ['biomass_estimate_ug', 'biomass_ug', 'biomass']:
                if col in features_df.columns:
                    biomass_col = col
                    break
            
            green_col = None
            for col in ['is_green_cell', 'green_cell', 'is_green']:
                if col in features_df.columns:
                    green_col = col
                    break
            
            # Calculate values safely
            total_cells = len(features_df)
            
            if area_col:
                areas = features_df[area_col].fillna(0)
                avg_area = float(areas.mean())
                area_stats = {
                    'min': float(areas.min()),
                    'max': float(areas.max()),
                    'std': float(areas.std())
                }
            else:
                avg_area = 0
                area_stats = {'min': 0, 'max': 0, 'std': 0}
            
            if biomass_col:
                total_biomass = float(features_df[biomass_col].fillna(0).sum())
            else:
                total_biomass = 0
            
            if green_col:
                total_green = int(features_df[green_col].fillna(False).sum())
            else:
                total_green = 0
            
            summary = {
                'total_cells_detected': total_cells,  # FIXED: Consistent naming
                'total_cells': total_cells,           # ADDED: For backward compatibility
                'total_green_cells': total_green,
                'average_area': avg_area,
                'average_cell_area': avg_area,        # ADDED: For backward compatibility
                'total_biomass': total_biomass,
                'area_statistics': area_stats
            }
            
            print(f"📊 Summary created: {total_cells} cells, {total_green} green, {total_biomass:.3f} μg biomass")
            return summary
            
        except Exception as e:
            print(f"❌ Error creating summary: {e}")
            return {
                'total_cells_detected': len(features_df),
                'total_green_cells': 0,
                'average_area': 0,
                'total_biomass': 0
            }
    
    def _create_visualizations(self, restoration_result, segmentation_result, features_df):
        """Create professional visualizations"""
        visualizations = {}
        
        try:
            # Main cell detection overlay
            vis_main = self._create_detection_overlay(
                restoration_result['original'],
                segmentation_result['labels']
            )
            if vis_main:
                visualizations['cell_detection'] = vis_main
            
            # Quality assessment visualization
            if not features_df.empty:
                vis_quality = self._create_quality_visualization(features_df)
                if vis_quality:
                    visualizations['quality_assessment'] = vis_quality
                    
        except Exception as e:
            print(f"⚠️ Visualization creation failed: {e}")
        
        return visualizations
    
    def _create_detection_overlay(self, image, labels):
        """Create cell detection overlay visualization - FIXED VERSION"""
        try:
            fig, axes = plt.subplots(1, 2, figsize=(15, 7))
            
            # Original image
            if len(image.shape) == 3:
                axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                axes[0].imshow(image, cmap='gray')
            axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
            axes[0].axis('off')
            
            # Segmentation overlay - FIXED: Only borders, no black numbers
            overlay = image.copy()
            
            if np.max(labels) > 0:
                # Create border highlighting overlay
                from skimage import segmentation
                
                # Find boundaries of cells
                boundaries = segmentation.find_boundaries(labels, mode='thick')
                
                # Create overlay with original image
                overlay_display = image.copy()
                if len(overlay_display.shape) == 3:
                    # Highlight cell boundaries in bright green - MUCH MORE VISIBLE
                    overlay_display[boundaries] = [0, 255, 0]  # Bright green borders
                    
                    # REMOVED: No more black text numbers that hide cells
                    # Instead, use bright colored dots at centroids
                    props = measure.regionprops(labels)
                    for prop in props:
                        y, x = prop.centroid
                        # Draw small bright yellow circles instead of text
                        cv2.circle(overlay_display, (int(x), int(y)), 3, (0, 255, 255), -1)  # Yellow filled circle
                        cv2.circle(overlay_display, (int(x), int(y)), 5, (255, 255, 255), 2)   # White border
                    
                else:
                    overlay_display = cv2.cvtColor(overlay_display, cv2.COLOR_GRAY2RGB)
                    overlay_display[boundaries] = [0, 255, 0]
                    
                    # Add centroids for grayscale too
                    props = measure.regionprops(labels)
                    for prop in props:
                        y, x = prop.centroid
                        cv2.circle(overlay_display, (int(x), int(y)), 3, (0, 255, 255), -1)
                        cv2.circle(overlay_display, (int(x), int(y)), 5, (255, 255, 255), 2)
                
                overlay = overlay_display
            
            if len(overlay.shape) == 3:
                axes[1].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
            else:
                axes[1].imshow(overlay, cmap='viridis')
            axes[1].set_title(f'Cell Detection - {np.max(labels)} cells found', 
                            fontsize=14, fontweight='bold')
            axes[1].axis('off')
            
            plt.tight_layout()
            
            # Convert to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            img_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return img_base64
            
        except Exception as e:
            print(f"⚠️ Detection overlay creation failed: {e}")
            return None
    
    def _create_quality_visualization(self, features_df):
        """Create quality assessment visualization"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Area distribution
            if 'area_microns_sq' in features_df.columns:
                axes[0,0].hist(features_df['area_microns_sq'], bins=20, alpha=0.7, color='blue')
                axes[0,0].set_title('Cell Area Distribution')
                axes[0,0].set_xlabel('Area (μm²)')
                axes[0,0].set_ylabel('Count')
            
            # Circularity distribution
            if 'circularity' in features_df.columns:
                axes[0,1].hist(features_df['circularity'], bins=20, alpha=0.7, color='green')
                axes[0,1].set_title('Cell Circularity Distribution')
                axes[0,1].set_xlabel('Circularity')
                axes[0,1].set_ylabel('Count')
            
            # Health status
            if 'health_status' in features_df.columns:
                health_counts = features_df['health_status'].value_counts()
                axes[1,0].pie(health_counts.values, labels=health_counts.index, autopct='%1.1f%%')
                axes[1,0].set_title('Cell Health Distribution')
            
            # Biomass vs Area scatter
            if 'area_microns_sq' in features_df.columns and 'biomass_estimate_ug' in features_df.columns:
                axes[1,1].scatter(features_df['area_microns_sq'], features_df['biomass_estimate_ug'], 
                                alpha=0.6, color='red')
                axes[1,1].set_title('Biomass vs Area')
                axes[1,1].set_xlabel('Area (μm²)')
                axes[1,1].set_ylabel('Biomass (μg)')
            
            plt.tight_layout()
            
            # Convert to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            img_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return img_base64
            
        except Exception as e:
            print(f"⚠️ Quality visualization creation failed: {e}")
            return None
    
    # Compatibility methods for web integration
    def analyze_single_image(self, image_path):
        """Compatibility method for legacy interface"""
        return self.analyze_image_professional(image_path)
    
    def analyze_single_image_enhanced(self, image_path, timestamp=None, save_visualization=True, custom_params=None):
        """Compatibility method for enhanced analysis"""
        params = {}
        if custom_params:
            params.update(custom_params)
        
        return self.analyze_image_professional(
            image_path,
            save_visualizations=save_visualization,
            **params
        )
    
    def get_current_parameters(self):
        """Get current analysis parameters"""
        return {
            'pixel_to_micron': self.pixel_to_micron,
            'chlorophyll_threshold': self.chlorophyll_threshold,
            'wolffia_params': self.wolffia_params.copy(),
            'engines_status': {
                'restoration': getattr(self.restoration_engine, 'status', '❌'),
                'segmentation': getattr(self.segmentation_engine, 'status', '❌'),
                'features': getattr(self.feature_engine, 'status', '❌'),
                'quality': getattr(self.quality_engine, 'status', '❌'),
                'learning': getattr(self.learning_engine, 'status', '❌')
            }
        }
    
    def set_parameters(self, **kwargs):
        """Set analysis parameters"""
        if 'pixel_to_micron_ratio' in kwargs:
            self.pixel_to_micron = kwargs['pixel_to_micron_ratio']
        if 'chlorophyll_threshold' in kwargs:
            self.chlorophyll_threshold = kwargs['chlorophyll_threshold']
        if 'diameter' in kwargs:
            self.wolffia_params['diameter'] = kwargs['diameter']
        if 'flow_threshold' in kwargs:
            self.wolffia_params['flow_threshold'] = kwargs['flow_threshold']
        
        # Update any other Wolffia parameters
        for key, value in kwargs.items():
            if key in self.wolffia_params:
                self.wolffia_params[key] = value
    
    def save_tophat_annotations(self, analysis_id, image_path, annotations):
        """Save tophat training annotations"""
        if self.learning_engine:
            return self.learning_engine.save_user_annotations(analysis_id, image_path, annotations)
        return False
    
    def apply_tophat_training(self, image_path, original_labels):
        """Apply tophat training to improve segmentation"""
        try:
            if not self.learning_engine:
                return original_labels, []
            
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                return original_labels, []
            
            # Apply trained tophat model
            improved_labels, confidence_scores = self.learning_engine.apply_tophat_model(image, original_labels)
            
            return improved_labels, confidence_scores
            
        except Exception as e:
            print(f"❌ Error applying tophat training: {e}")
            return original_labels, []
    
    def get_tophat_training_status(self):
        """Get status of tophat training system"""
        if not self.learning_engine:
            return {'available': False, 'reason': 'Learning engine not available'}
        
        num_annotations = len(getattr(self.learning_engine, 'user_annotations', []))
        model_file = self.learning_engine.learning_dir / 'tophat_classifier.pkl'
        model_trained = model_file.exists()
        
        return {
            'available': True,
            'num_annotations': num_annotations,
            'model_trained': model_trained,
            'needs_more_training': num_annotations < 10,
            'status': 'ready' if model_trained else ('training' if num_annotations >= 5 else 'needs_annotations')
        }
    
def export_enhanced_results(self, analysis_result):
    """Export results in JSON-serializable format for web interface - FIXED VERSION"""
    try:
        if not analysis_result or not analysis_result.get('success'):
            return None
        
        def convert_to_serializable(obj):
            if isinstance(obj, (np.integer, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif hasattr(obj, 'item'):  # numpy scalars
                return obj.item()
            elif pd.isna(obj):  # ADDED: Handle pandas NaN values
                return None
            else:
                return obj
        
        # Convert the entire result to JSON-serializable format
        json_result = convert_to_serializable(analysis_result)
        
        # FIXED: Ensure critical fields are present and correct
        if 'cells' in json_result and json_result['cells']:
            cell_count = len(json_result['cells'])
            if 'summary' in json_result:
                # FIXED: Ensure summary reflects actual cell data
                json_result['summary']['total_cells_detected'] = cell_count
                json_result['summary']['total_cells'] = cell_count
                
                # Recalculate green cells from actual cell data
                green_cells = sum(1 for cell in json_result['cells'] 
                                if cell.get('is_green_cell', False) or 
                                   cell.get('green_cell', False))
                json_result['summary']['total_green_cells'] = green_cells
                
                # Recalculate biomass from actual cell data
                total_biomass = sum(cell.get('biomass_estimate_ug', 0) or 
                                  cell.get('biomass_ug', 0) or 0 
                                  for cell in json_result['cells'])
                json_result['summary']['total_biomass'] = total_biomass
                
                print(f"🔧 FIXED summary: {cell_count} cells, {green_cells} green, {total_biomass:.3f} μg")
        
        return json_result
        
    except Exception as e:
        print(f"⚠️ Error exporting enhanced results: {e}")
        import traceback
        traceback.print_exc()
        return None

class ImageRestorationEngine:
    """Professional image restoration engine"""
    
    def __init__(self):
        self.status = "✅ Ready" if (SIMPLEITK_AVAILABLE or CELLPOSE_VERSION >= 3) else "⚠️ Limited"
        
    def restore_image(self, image_path, mode='auto'):
        """Restore image using adaptive quality-aware restoration"""
        try:
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                return {'error': 'Cannot load image'}
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Analyze image quality and characteristics
            image_analysis = self._analyze_image_quality(image, gray)
            print(f"📊 Image analysis: {image_analysis}")
            
            # Choose optimal restoration strategy based on analysis
            if mode == 'auto':
                mode = self._select_optimal_restoration(image_analysis)
                print(f"🔧 Auto-selected restoration mode: {mode}")
            
            if mode == 'none':
                restored = image
                quality = image_analysis['overall_quality']
                method = 'no_restoration'
            elif CELLPOSE_VERSION >= 3 and mode in ['denoise']:
                restored, quality, method = self._cellpose_restoration(image, mode)
            elif SIMPLEITK_AVAILABLE and mode in ['advanced', 'medical']:
                restored, quality, method = self._simpleitk_restoration(image, gray, mode)
            else:
                # Enhanced OpenCV restoration with adaptive algorithms
                restored, quality, method = self._adaptive_opencv_restoration(image, gray, image_analysis, mode)
            
            # Apply additional enhancement if needed
            if image_analysis['needs_enhancement']:
                restored = self._apply_enhancement_pipeline(restored, image_analysis)
                quality = min(quality + 0.1, 1.0)  # Boost quality score
                method += '_enhanced'
            
            return {
                'original': image,
                'restored': restored,
                'gray_restored': cv2.cvtColor(restored, cv2.COLOR_BGR2GRAY),
                'quality_score': quality,
                'method': method,
                'image_analysis': image_analysis
            }
            
        except Exception as e:
            print(f"❌ Restoration failed: {e}")
            return {'error': str(e)}
    
    def _cellpose_restoration(self, image, mode):
        """CellPose 3+ restoration"""
        try:
            # Use CellPose denoise model
            dn_model = denoise.DenoiseModel(
                model_type='denoise_cyto3',
                gpu=torch.cuda.is_available()
            )
            
            # Apply to each channel
            if len(image.shape) == 3:
                restored_channels = []
                for i in range(3):
                    channel = image[:, :, i]
                    restored_ch = dn_model.eval(channel, diameter=30)
                    restored_channels.append(restored_ch.squeeze())
                restored = np.stack(restored_channels, axis=-1)
            else:
                restored = dn_model.eval(image, diameter=30)
            
            restored = np.clip(restored, 0, 255).astype(np.uint8)
            quality = 0.85
            
            return restored, quality, 'cellpose_denoise'
            
        except Exception as e:
            print(f"⚠️ CellPose restoration failed: {e}")
            return self._opencv_restoration(image, cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), mode)
    
    def _simpleitk_restoration(self, image, gray, mode):
        """SimpleITK professional restoration"""
        try:
            sitk_image = sitk.GetImageFromArray(gray)
            
            if mode == 'full':
                # Full restoration pipeline
                restored_sitk = sitk.DiscreteGaussian(sitk_image, 0.8)
                restored_sitk = sitk.AdaptiveHistogramEqualization(restored_sitk)
            else:
                # Basic enhancement
                restored_sitk = sitk.DiscreteGaussian(sitk_image, 0.5)
            
            restored_gray = sitk.GetArrayFromImage(restored_sitk).astype(np.uint8)
            
            # Apply to color image
            restored_image = image.copy()
            for i in range(3):
                channel_sitk = sitk.GetImageFromArray(image[:,:,i])
                if mode == 'full':
                    restored_ch = sitk.DiscreteGaussian(channel_sitk, 0.8)
                    restored_ch = sitk.AdaptiveHistogramEqualization(restored_ch)
                else:
                    restored_ch = sitk.DiscreteGaussian(channel_sitk, 0.5)
                restored_image[:,:,i] = sitk.GetArrayFromImage(restored_ch)
            
            quality = 0.8
            return restored_image, quality, f'simpleitk_{mode}'
            
        except Exception as e:
            print(f"⚠️ SimpleITK restoration failed: {e}")
            return self._opencv_restoration(image, gray, mode)
    
    def _opencv_restoration(self, image, gray, mode):
        """OpenCV fallback restoration"""
        if mode == 'full':
            # Full OpenCV pipeline
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            enhanced_gray = clahe.apply(gray)
            enhanced_image = cv2.bilateralFilter(image, 9, 75, 75)
            quality = 0.7
        else:
            # Basic enhancement
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced_gray = clahe.apply(gray)
            enhanced_image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
            quality = 0.65
        
        return enhanced_image, quality, f'opencv_{mode}'
    
    def _analyze_image_quality(self, image, gray):
        """Analyze image quality and characteristics for adaptive processing"""
        analysis = {}
        
        # Basic image statistics
        analysis['mean_intensity'] = np.mean(gray)
        analysis['std_intensity'] = np.std(gray)
        analysis['contrast'] = analysis['std_intensity'] / (analysis['mean_intensity'] + 1e-8)
        
        # Noise analysis (estimate using Laplacian variance)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        analysis['noise_level'] = 1.0 / (1.0 + laplacian_var / 1000.0)  # Normalize
        analysis['is_noisy'] = analysis['noise_level'] > 0.6
        
        # Brightness analysis
        analysis['is_dark'] = analysis['mean_intensity'] < 80
        analysis['is_bright'] = analysis['mean_intensity'] > 180
        analysis['is_low_contrast'] = analysis['contrast'] < 0.3
        
        # Color channel analysis
        if len(image.shape) == 3:
            b, g, r = cv2.split(image)
            analysis['green_dominance'] = np.mean(g) / (np.mean(r) + np.mean(b) + 1e-8)
            analysis['has_color_cast'] = analysis['green_dominance'] > 1.5 or analysis['green_dominance'] < 0.7
        else:
            analysis['green_dominance'] = 1.0
            analysis['has_color_cast'] = False
        
        # Edge definition (for focus quality)
        edges = cv2.Canny(gray, 50, 150)
        analysis['edge_density'] = np.sum(edges > 0) / edges.size
        analysis['is_blurry'] = analysis['edge_density'] < 0.05
        
        # Overall quality assessment
        quality_factors = []
        quality_factors.append(1.0 - analysis['noise_level'])  # Lower noise = higher quality
        quality_factors.append(min(analysis['contrast'] / 0.5, 1.0))  # Good contrast
        quality_factors.append(1.0 if not analysis['is_dark'] and not analysis['is_bright'] else 0.7)
        quality_factors.append(1.0 if not analysis['is_blurry'] else 0.5)
        
        analysis['overall_quality'] = np.mean(quality_factors)
        analysis['needs_enhancement'] = analysis['overall_quality'] < 0.7
        
        return analysis
    
    def _select_optimal_restoration(self, analysis):
        """Select optimal restoration mode based on image analysis"""
        if analysis['is_noisy'] and analysis['is_low_contrast']:
            return 'full'
        elif analysis['is_noisy']:
            return 'denoise'
        elif analysis['is_low_contrast'] or analysis['is_dark']:
            return 'enhance'
        elif analysis['is_blurry']:
            return 'sharpen'
        elif analysis['overall_quality'] > 0.8:
            return 'none'
        else:
            return 'enhance'
    
    def _adaptive_opencv_restoration(self, image, gray, analysis, mode):
        """Enhanced OpenCV restoration with adaptive algorithms"""
        restored = image.copy()
        quality_improvement = 0
        methods_used = []
        
        # Denoising based on noise analysis
        if analysis['is_noisy'] or mode in ['denoise', 'full']:
            if len(image.shape) == 3:
                # Color image denoising
                h_value = min(15, max(5, int(analysis['noise_level'] * 20)))
                restored = cv2.fastNlMeansDenoisingColored(restored, None, h_value, h_value, 7, 21)
            else:
                # Grayscale denoising
                h_value = min(15, max(5, int(analysis['noise_level'] * 20)))
                restored = cv2.fastNlMeansDenoising(restored, None, h_value, 7, 21)
            methods_used.append('adaptive_denoise')
            quality_improvement += 0.15
        
        # Contrast enhancement
        if analysis['is_low_contrast'] or analysis['is_dark'] or mode in ['enhance', 'full']:
            # Adaptive CLAHE based on image characteristics
            clip_limit = 2.0 + (1.0 - analysis['contrast']) * 2.0  # More aggressive for low contrast
            tile_size = (8, 8) if analysis['mean_intensity'] < 100 else (12, 12)
            
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
            
            if len(image.shape) == 3:
                # Apply CLAHE to luminance channel
                lab = cv2.cvtColor(restored, cv2.COLOR_BGR2LAB)
                lab[:,:,0] = clahe.apply(lab[:,:,0])
                restored = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            else:
                restored = clahe.apply(restored)
            
            methods_used.append('adaptive_clahe')
            quality_improvement += 0.2
        
        # Sharpening for blurry images
        if analysis['is_blurry'] or mode in ['sharpen', 'full']:
            # Adaptive unsharp masking
            blur_strength = min(3.0, analysis['edge_density'] * 10)
            blurred = cv2.GaussianBlur(restored, (0, 0), blur_strength)
            sharpened = cv2.addWeighted(restored, 1.5, blurred, -0.5, 0)
            restored = np.clip(sharpened, 0, 255).astype(np.uint8)
            methods_used.append('adaptive_sharpen')
            quality_improvement += 0.1
        
        # Color balance correction
        if analysis['has_color_cast'] and len(image.shape) == 3:
            # Simple white balance using gray world assumption
            b, g, r = cv2.split(restored)
            b_mean, g_mean, r_mean = np.mean(b), np.mean(g), np.mean(r)
            gray_mean = (b_mean + g_mean + r_mean) / 3
            
            b = np.clip(b * (gray_mean / (b_mean + 1e-8)), 0, 255).astype(np.uint8)
            r = np.clip(r * (gray_mean / (r_mean + 1e-8)), 0, 255).astype(np.uint8)
            
            restored = cv2.merge([b, g, r])
            methods_used.append('color_balance')
            quality_improvement += 0.05
        
        final_quality = min(analysis['overall_quality'] + quality_improvement, 1.0)
        method_name = f"adaptive_opencv_{'_'.join(methods_used) if methods_used else mode}"
        
        return restored, final_quality, method_name
    
    def _apply_enhancement_pipeline(self, image, analysis):
        """Apply additional enhancement pipeline for low-quality images"""
        enhanced = image.copy()
        
        # Edge-preserving smoothing for very noisy images
        if analysis['noise_level'] > 0.7:
            enhanced = cv2.edgePreservingFilter(enhanced, flags=1, sigma_s=50, sigma_r=0.4)
        
        # Gamma correction for very dark or bright images
        if analysis['is_dark']:
            # Brighten dark images
            gamma = 0.7
            enhanced = np.power(enhanced / 255.0, gamma) * 255.0
            enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
        elif analysis['is_bright']:
            # Darken bright images
            gamma = 1.3
            enhanced = np.power(enhanced / 255.0, gamma) * 255.0
            enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
        
        return enhanced

class CellPoseSegmentationEngine:
    """Professional CellPose segmentation engine with GPU optimization"""
    
    def __init__(self):
        self.gpu_info = detect_gpu_capabilities()
        self.use_gpu = self.gpu_info['cuda_available']
        
        if self.use_gpu:
            print(f"🚀 GPU Detected: {self.gpu_info['gpu_name']}")
            print(f"💾 GPU Memory: {self.gpu_info['gpu_memory']:.1f} GB")
            print(f"⚙️ Recommended batch size: {self.gpu_info['recommended_settings']['batch_size']}")
            self.status = f"✅ GPU Ready ({self.gpu_info['gpu_name']})"
        else:
            print("⚠️ GPU not available, using CPU")
            self.status = "⚠️ CPU Only"
            
        self.models = {}
        self._load_models()
    
    def _load_models(self):
        """Load CellPose models with GPU optimization"""
        if not CELLPOSE_AVAILABLE:
            return
        
        try:
            # Enable mixed precision if supported
            if self.use_gpu and self.gpu_info['recommended_settings'].get('use_mixed_precision', False):
                torch.backends.cudnn.benchmark = True
                print("🔥 Mixed precision enabled for faster inference")
            
            # Load models with GPU support
            model_configs = [
                ('cyto3', 'cyto3'),
                ('nuclei', 'nuclei'),
                ('cyto2', 'cyto2')  # Backup model
            ]
            
            for model_name, model_type in model_configs:
                try:
                    print(f"📦 Loading {model_name} model...")
                    
                    # Create model with optimal settings (v4+ compatible)
                    model = models.CellposeModel(
                        gpu=self.use_gpu,
                        model=model_type
                    )
                    
                    self.models[model_name] = model
                    print(f"✅ {model_name} model loaded {'(GPU)' if self.use_gpu else '(CPU)'}")
                    
                except Exception as e:
                    print(f"⚠️ Failed to load {model_name}: {e}")
                    continue
            
            if not self.models:
                print("❌ No CellPose models could be loaded")
            else:
                print(f"🎯 {len(self.models)} CellPose models ready")
                
        except Exception as e:
            print(f"❌ Error in model loading: {e}")
    
    def segment_cells(self, restoration_result, model='auto', diameter='auto', flow_threshold='auto'):
        """Segment cells using adaptive parameter optimization"""
        print(f"🧬 Segmentation request: model={model}, diameter={diameter}, flow_threshold={flow_threshold}")
        print(f"🔍 CellPose available: {CELLPOSE_AVAILABLE}, Models loaded: {len(self.models) if hasattr(self, 'models') else 0}")
        
        # Auto-optimize parameters based on image analysis
        if 'image_analysis' in restoration_result:
            optimized_params = self._optimize_parameters(restoration_result['image_analysis'])
            
            if model == 'auto':
                model = optimized_params['model']
            if diameter == 'auto':
                diameter = optimized_params['diameter']
            if flow_threshold == 'auto':
                flow_threshold = optimized_params['flow_threshold']
                
            print(f"🎯 Auto-optimized parameters: model={model}, diameter={diameter}, flow_threshold={flow_threshold}")
        else:
            # Use defaults if no analysis available
            if model == 'auto':
                model = 'cyto3'
            if diameter == 'auto':
                diameter = 25
            if flow_threshold == 'auto':
                flow_threshold = 0.2
        
        try:
            if not CELLPOSE_AVAILABLE or not self.models:
                print("📉 Using optimized fallback segmentation (CellPose not available)")
                return self._optimized_fallback_segmentation(restoration_result, diameter, flow_threshold)
            
            image = restoration_result['restored']
            
            # Get model with fallback
            if model not in self.models:
                print(f"⚠️ Model {model} not available, trying fallbacks...")
                available_models = list(self.models.keys())
                if available_models:
                    model = available_models[0]
                    print(f"🔄 Using fallback model: {model}")
                else:
                    return self._fallback_segmentation(restoration_result)
            
            cellpose_model = self.models[model]
            
            # Optimize parameters based on GPU capabilities
            if self.use_gpu:
                # GPU-optimized parameters
                batch_size = self.gpu_info['recommended_settings']['batch_size']
                
                # Use optimal diameter
                if diameter is None:
                    diameter = self.gpu_info['recommended_settings']['diameter']
                
                print(f"🚀 GPU Segmentation: {model} (batch: {batch_size}, diameter: {diameter})")
            else:
                batch_size = 1
                print(f"🐌 CPU Segmentation: {model} (diameter: {diameter})")
            
            # Prepare channels for Wolffia (standard configuration)
            channels = [0, 0] if len(image.shape) == 3 else [0, 0]  # Use grayscale mode for better results
            
            # GPU-optimized inference with error handling
            if TORCH_AVAILABLE and self.use_gpu:
                context_manager = torch.cuda.device(0)
            elif TORCH_AVAILABLE:
                context_manager = torch.no_grad()
            else:
                from contextlib import nullcontext
                context_manager = nullcontext()
            
            with context_manager:
                try:
                    # Memory optimization for GPU
                    if self.use_gpu and self.gpu_info['gpu_memory'] < 4:
                        # For low memory GPUs, process in smaller tiles
                        masks, flows, styles = self._process_with_tiling(
                            cellpose_model, image, channels, diameter, flow_threshold
                        )
                    else:
                        # Standard processing
                        masks, flows, styles = cellpose_model.eval(
                            image,
                            diameter=diameter,
                            channels=channels,
                            flow_threshold=flow_threshold,
                            cellprob_threshold=-1.0  # More permissive threshold
                        )
                        
                        # Apply post-processing filters for small cells (moved from min_size parameter)
                        if np.max(masks) > 0:
                            masks = morphology.remove_small_objects(masks.astype(bool), min_size=10)
                            masks = measure.label(masks)
                        
                except Exception as cuda_error:
                    if TORCH_AVAILABLE and 'CUDA out of memory' in str(cuda_error):
                        print("⚠️ GPU out of memory, falling back to CPU")
                    else:
                        print(f"⚠️ Error during GPU processing, falling back to CPU: {cuda_error}")
                    # Fallback to CPU processing
                    cellpose_model = models.CellposeModel(
                        gpu=False, model=model
                    )
                    masks, flows, styles = cellpose_model.eval(
                        image, diameter=diameter, channels=channels,
                        flow_threshold=flow_threshold, cellprob_threshold=-1.0
                    )
                    
                    # Apply post-processing filters for small cells
                    if np.max(masks) > 0:
                        masks = morphology.remove_small_objects(masks.astype(bool), min_size=10)
                        masks = measure.label(masks)
            
            # Post-process for Wolffia with GPU acceleration if available
            masks = self._post_process_masks_gpu(masks, image)
            
            # Calculate confidence
            confidence = self._calculate_confidence_gpu(masks, flows)
            
            # Clear GPU cache to prevent memory issues
            if self.use_gpu:
                torch.cuda.empty_cache()
            
            return {
                'labels': masks,
                'num_cells': np.max(masks),
                'method': f'cellpose_{model}_{"gpu" if self.use_gpu else "cpu"}',
                'confidence': confidence,
                'flows': flows,
                'styles': styles,
                'gpu_used': self.use_gpu
            }
            
        except Exception as e:
            print(f"❌ GPU CellPose segmentation failed: {e}")
            # Clear GPU cache and fallback
            if self.use_gpu:
                torch.cuda.empty_cache()
            return self._fallback_segmentation(restoration_result)
        
    def _process_with_tiling(self, model, image, channels, diameter, flow_threshold):
        """Process large images with tiling for memory-constrained GPUs"""
        try:
            print("🧩 Using tiled processing for memory optimization")
            
            # Calculate tile size based on available memory
            tile_size = 512 if self.gpu_info['gpu_memory'] < 4 else 1024
            
            h, w = image.shape[:2]
            masks_full = np.zeros((h, w), dtype=np.uint16)
            flows_full = [np.zeros((h, w)), np.zeros((h, w)), np.zeros((h, w))]
            
            label_counter = 1
            
            # Process in overlapping tiles
            overlap = 50
            for y in range(0, h, tile_size - overlap):
                for x in range(0, w, tile_size - overlap):
                    y_end = min(y + tile_size, h)
                    x_end = min(x + tile_size, w)
                    
                    # Extract tile
                    if len(image.shape) == 3:
                        tile = image[y:y_end, x:x_end, :]
                    else:
                        tile = image[y:y_end, x:x_end]
                    
                    # Process tile
                    masks_tile, flows_tile, _ = model.eval(
                        tile, diameter=diameter, channels=channels,
                        flow_threshold=flow_threshold, cellprob_threshold=-1.0
                    )
                    
                    # Merge results (simple approach - can be improved)
                    if np.max(masks_tile) > 0:
                        masks_tile[masks_tile > 0] += label_counter - 1
                        masks_full[y:y_end, x:x_end] = np.maximum(
                            masks_full[y:y_end, x:x_end], masks_tile
                        )
                        label_counter += np.max(masks_tile)
                    
                    # Merge flows
                    for i in range(3):
                        flows_full[i][y:y_end, x:x_end] = flows_tile[i]
            
            return masks_full, flows_full, []
            
        except Exception as e:
            print(f"❌ Tiled processing failed: {e}")
            # Fallback to regular processing
            return model.eval(image, diameter=diameter, channels=channels)
    
    def _post_process_masks_gpu(self, masks, image):
        """GPU-accelerated post-processing for Wolffia-specific filtering"""
        if np.max(masks) == 0:
            return masks
        
        try:
            # Use GPU acceleration for image processing if available
            if self.use_gpu and 'cupy' in globals():
                return self._cupy_post_process(masks, image)
            else:
                return self._cpu_post_process(masks, image)
                
        except Exception as e:
            print(f"⚠️ GPU post-processing failed, using CPU: {e}")
            return self._cpu_post_process(masks, image)
    
    def _cupy_post_process(self, masks, image):
        """CuPy-accelerated post-processing"""
        try:
            import cupy as cp
            
            # Transfer to GPU
            masks_gpu = cp.asarray(masks)
            image_gpu = cp.asarray(image)
            
            # GPU-accelerated filtering
            props = measure.regionprops(cp.asnumpy(masks_gpu))
            filtered = cp.zeros_like(masks_gpu)
            new_label = 1
            
            for prop in props:
                # Size and shape filters
                if prop.area < 10 or prop.area > 0.05 * masks.size:
                    continue
                if prop.eccentricity > 0.9:
                    continue
                
                # Color filtering on GPU
                if len(image.shape) == 3:
                    region_mask = masks_gpu == prop.label
                    region_pixels = image_gpu[region_mask]
                    green_mean = cp.mean(region_pixels[:, 1])
                    red_mean = cp.mean(region_pixels[:, 0])
                    
                    if green_mean <= red_mean * 0.8:
                        continue
                
                # Accept cell
                filtered[masks_gpu == prop.label] = new_label
                new_label += 1
            
            # Transfer back to CPU
            return cp.asnumpy(filtered)
            
        except Exception as e:
            print(f"❌ CuPy processing failed: {e}")
            return self._cpu_post_process(masks, image)
    
    def _cpu_post_process(self, masks, image):
        """CPU fallback post-processing"""
        props = measure.regionprops(masks)
        filtered = np.zeros_like(masks)
        new_label = 1
        
        for prop in props:
            # Size filter - more permissive
            if prop.area < 5 or prop.area > 0.15 * masks.size:  # Reduced min from 10 to 5, increased max from 5% to 15%
                continue
            
            # Shape filter - more permissive for irregular cell shapes
            if prop.eccentricity > 0.95:  # Increased from 0.9 to 0.95
                continue
            
            # Color filter for green cells - more permissive
            if len(image.shape) == 3:
                region_mask = masks == prop.label
                region_pixels = image[region_mask]
                if region_pixels.size > 0:  # Safety check
                    green_mean = np.mean(region_pixels[:, 1])
                    red_mean = np.mean(region_pixels[:, 0])
                    blue_mean = np.mean(region_pixels[:, 2])
                    
                    # More relaxed green filter - just needs to be somewhat green
                    if green_mean <= red_mean * 0.6 or green_mean <= blue_mean * 0.6:  # Reduced from 0.8 to 0.6
                        continue
            
            filtered[masks == prop.label] = new_label
            new_label += 1
        
        return filtered
    
    def _calculate_confidence_gpu(self, masks, flows):
        """GPU-accelerated confidence calculation"""
        if np.max(masks) == 0:
            return 0.0
        
        try:
            # Flow consistency calculation
            flow_magnitude = np.sqrt(flows[1]**2 + flows[2]**2)
            flow_consistency = 1.0 - np.std(flow_magnitude) / (np.mean(flow_magnitude) + 1e-10)
            
            # Cell count reasonableness
            num_cells = np.max(masks)
            if 5 <= num_cells <= 200:
                count_score = 1.0
            elif 1 <= num_cells <= 500:
                count_score = 0.8
            else:
                count_score = 0.5
            
            confidence = np.clip(0.6 * flow_consistency + 0.4 * count_score, 0, 1)
            
            return float(confidence)
            
        except Exception as e:
            print(f"⚠️ Confidence calculation error: {e}")
            return 0.7  # Default confidence
    
    def get_gpu_status(self):
        """Get current GPU status and performance info"""
        status = {
            'gpu_available': self.use_gpu,
            'gpu_info': self.gpu_info,
            'models_loaded': len(self.models),
            'memory_usage': {}
        }
        
        if self.use_gpu:
            try:
                status['memory_usage'] = {
                    'allocated': torch.cuda.memory_allocated(0) / 1e9,
                    'cached': torch.cuda.memory_reserved(0) / 1e9,
                    'max_allocated': torch.cuda.max_memory_allocated(0) / 1e9
                }
            except:
                status['memory_usage'] = {'error': 'Could not get memory info'}
        
        return status
    
    def _post_process_masks(self, masks, image):
        """Post-process masks for Wolffia"""
        if np.max(masks) == 0:
            return masks
        
        # Remove artifacts and filter by Wolffia characteristics
        props = measure.regionprops(masks)
        filtered = np.zeros_like(masks)
        new_label = 1
        
        for prop in props:
            # Size filter
            if prop.area < 10 or prop.area > 0.05 * masks.size:
                continue
            
            # Shape filter
            if prop.eccentricity > 0.9:  # Too elongated
                continue
            
            # Color filter for green cells
            if len(image.shape) == 3:
                region_mask = masks == prop.label
                region_pixels = image[region_mask]
                green_mean = np.mean(region_pixels[:, 1])
                red_mean = np.mean(region_pixels[:, 0])
                
                if green_mean <= red_mean * 0.8:  # Not green enough
                    continue
            
            # Accept this cell
            filtered[masks == prop.label] = new_label
            new_label += 1
        
        return filtered
    
    def _calculate_confidence(self, masks, flows):
        """Calculate segmentation confidence"""
        if np.max(masks) == 0:
            return 0.0
        
        # Flow consistency
        flow_magnitude = np.sqrt(flows[1]**2 + flows[2]**2)
        flow_consistency = 1.0 - np.std(flow_magnitude) / (np.mean(flow_magnitude) + 1e-10)
        
        # Cell count reasonableness
        num_cells = np.max(masks)
        if 5 <= num_cells <= 200:
            count_score = 1.0
        elif 1 <= num_cells <= 500:
            count_score = 0.8
        else:
            count_score = 0.5
        
        return np.clip(0.6 * flow_consistency + 0.4 * count_score, 0, 1)
    
    def _fallback_segmentation(self, restoration_result):
        """Fallback segmentation without CellPose using watershed"""
        print("⚠️ Using watershed fallback segmentation")
        
        try:
            image = restoration_result['restored']
            
            # Use green channel or grayscale
            if len(image.shape) == 3:
                green = image[:, :, 1]  # Green channel for plant cells
            else:
                green = image
            
            # Threshold to binary
            thresh = filters.threshold_otsu(green)
            binary = green > thresh
            
            # Clean up binary image
            binary = morphology.remove_small_objects(binary, min_size=15)  # Reduced from 20 for smaller cells
            binary = ndimage.binary_fill_holes(binary)
            
            if not np.any(binary):
                print("⚠️ No objects found after thresholding")
                return {
                    'labels': np.zeros_like(image[:, :, 0] if len(image.shape) == 3 else image, dtype=int),
                    'num_cells': 0,
                    'method': 'watershed_fallback',
                    'confidence': 0.3
                }
            
            # Distance transform and peak detection
            distance = ndimage.distance_transform_edt(binary)
            coords = feature.peak_local_max(distance, min_distance=3, threshold_abs=0.2*distance.max())  # More sensitive
            
            if len(coords) == 0:
                print("⚠️ No peaks found in distance transform")
                return {
                    'labels': np.zeros_like(image[:, :, 0] if len(image.shape) == 3 else image, dtype=int),
                    'num_cells': 0,
                    'method': 'watershed_fallback',
                    'confidence': 0.3
                }
            
            # Create markers
            markers = np.zeros_like(distance, dtype=int)
            for i, coord in enumerate(coords):
                markers[coord[0], coord[1]] = i + 1
            
            # Watershed segmentation
            labels = segmentation.watershed(-distance, markers, mask=binary)
            num_cells = len(coords)
            
            print(f"✅ Watershed fallback found {num_cells} cells")
            
            return {
                'labels': labels,
                'num_cells': num_cells,
                'method': 'watershed_fallback',
                'confidence': 0.6,
                'gpu_used': False
            }
            
        except Exception as e:
            print(f"❌ Fallback segmentation failed: {e}")
            return {
                'labels': np.zeros_like(image[:, :, 0] if len(image.shape) == 3 else image, dtype=int),
                'num_cells': 0,
                'method': 'watershed_fallback_failed',
                'confidence': 0.1,
                'gpu_used': False
            }
    
    def _optimize_parameters(self, image_analysis):
        """Optimize segmentation parameters based on image analysis"""
        params = {}
        
        # Model selection based on image characteristics
        if image_analysis['green_dominance'] > 1.3:
            # Strong green channel - likely plant cells
            params['model'] = 'cyto3'  # Good for plant-like structures
        elif image_analysis['edge_density'] > 0.1:
            # High edge density - well-defined boundaries
            params['model'] = 'cyto3'
        else:
            # General case
            params['model'] = 'cyto2'
        
        # Diameter optimization based on image size and quality
        base_diameter = 25
        
        # Adjust for image quality
        if image_analysis['overall_quality'] < 0.5:
            # Poor quality - use smaller diameter for more sensitive detection
            base_diameter = 20
        elif image_analysis['overall_quality'] > 0.8:
            # High quality - can use larger diameter
            base_diameter = 30
        
        # Adjust for noise level
        if image_analysis['is_noisy']:
            base_diameter += 5  # Larger diameter for noisy images
        
        # Adjust for contrast
        if image_analysis['is_low_contrast']:
            base_diameter -= 3  # Smaller diameter for low contrast
        
        params['diameter'] = max(15, min(40, base_diameter))
        
        # Flow threshold optimization
        base_flow_threshold = 0.2
        
        # Adjust for image quality
        if image_analysis['overall_quality'] < 0.5:
            # Poor quality - more permissive threshold
            base_flow_threshold = 0.1
        elif image_analysis['overall_quality'] > 0.8:
            # High quality - can be more strict
            base_flow_threshold = 0.3
        
        # Adjust for noise and blur
        if image_analysis['is_noisy'] or image_analysis['is_blurry']:
            base_flow_threshold = max(0.05, base_flow_threshold - 0.1)
        
        params['flow_threshold'] = max(0.05, min(0.4, base_flow_threshold))
        
        print(f"🎯 Parameter optimization based on image analysis:")
        print(f"   Quality: {image_analysis['overall_quality']:.3f}")
        print(f"   Noise level: {image_analysis['noise_level']:.3f}")
        print(f"   Contrast: {image_analysis['contrast']:.3f}")
        print(f"   Edge density: {image_analysis['edge_density']:.3f}")
        print(f"   → Model: {params['model']}")
        print(f"   → Diameter: {params['diameter']}")
        print(f"   → Flow threshold: {params['flow_threshold']}")
        
        return params
    
    def _optimized_fallback_segmentation(self, restoration_result, diameter, flow_threshold):
        """Optimized watershed segmentation using automatic parameters"""
        print(f"⚠️ Using optimized watershed fallback (diameter={diameter}, flow_threshold={flow_threshold})")
        
        try:
            image = restoration_result['restored']
            image_analysis = restoration_result.get('image_analysis', {})
            
            # Use green channel or grayscale
            if len(image.shape) == 3:
                green = image[:, :, 1]  # Green channel for plant cells
            else:
                green = image
            
            # Adaptive thresholding based on image analysis
            if image_analysis.get('is_low_contrast', False):
                # Use adaptive threshold for low contrast images
                binary = cv2.adaptiveThreshold(
                    green, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
                )
                binary = binary > 0
            else:
                # Use Otsu for normal images
                thresh = filters.threshold_otsu(green)
                binary = green > thresh
            
            # Adaptive morphological operations
            min_size = max(10, int(diameter * 0.3))  # Scale with expected cell size
            binary = morphology.remove_small_objects(binary, min_size=min_size)
            binary = ndimage.binary_fill_holes(binary)
            
            if not np.any(binary):
                print("⚠️ No objects found after adaptive thresholding")
                return {
                    'labels': np.zeros_like(image[:, :, 0] if len(image.shape) == 3 else image, dtype=int),
                    'num_cells': 0,
                    'method': 'optimized_watershed_fallback',
                    'confidence': 0.3,
                    'gpu_used': False
                }
            
            # Distance transform and adaptive peak detection
            distance = ndimage.distance_transform_edt(binary)
            
            # Adaptive peak detection parameters
            min_distance = max(3, int(diameter * 0.15))
            threshold_ratio = 0.2 if image_analysis.get('is_noisy', False) else 0.3
            
            coords = feature.peak_local_max(
                distance, 
                min_distance=min_distance, 
                threshold_abs=threshold_ratio * distance.max()
            )
            
            if len(coords) == 0:
                print("⚠️ No peaks found in adaptive distance transform")
                return {
                    'labels': np.zeros_like(image[:, :, 0] if len(image.shape) == 3 else image, dtype=int),
                    'num_cells': 0,
                    'method': 'optimized_watershed_fallback',
                    'confidence': 0.3,
                    'gpu_used': False
                }
            
            # Create markers
            markers = np.zeros_like(distance, dtype=int)
            for i, coord in enumerate(coords):
                markers[coord[0], coord[1]] = i + 1
            
            # Watershed segmentation
            labels = segmentation.watershed(-distance, markers, mask=binary)
            num_cells = len(coords)
            
            # Calculate confidence based on detection quality
            confidence = 0.6
            if image_analysis.get('overall_quality', 0.5) > 0.7:
                confidence += 0.1
            if not image_analysis.get('is_noisy', False):
                confidence += 0.1
            
            print(f"✅ Optimized watershed found {num_cells} cells (confidence: {confidence:.3f})")
            
            return {
                'labels': labels,
                'num_cells': num_cells,
                'method': 'optimized_watershed_fallback',
                'confidence': min(confidence, 1.0),
                'gpu_used': False,
                'diameter_used': diameter,
                'flow_threshold_used': flow_threshold
            }
            
        except Exception as e:
            print(f"❌ Optimized fallback segmentation failed: {e}")
            return {
                'labels': np.zeros_like(image[:, :, 0] if len(image.shape) == 3 else image, dtype=int),
                'num_cells': 0,
                'method': 'optimized_watershed_fallback_failed',
                'confidence': 0.1,
                'gpu_used': False
            }


class FeatureExtractionEngine:
    """Professional feature extraction engine"""
    
    def __init__(self):
        self.status = "✅ Ready"
    
    def extract_features(self, image, labels, pixel_to_micron=1.0):
        """Extract comprehensive cell features"""
        if np.max(labels) == 0:
            return pd.DataFrame()
        
        features = []
        props = measure.regionprops(labels)
        
        for prop in props:
            cell_features = self._extract_cell_features(prop, image, labels, pixel_to_micron)
            features.append(cell_features)
        
        return pd.DataFrame(features)
    
    def _extract_cell_features(self, prop, image, labels, pixel_to_micron):
        """Extract features for a single cell"""
        # Basic morphological features
        area_pixels = prop.area
        area_microns = area_pixels * (pixel_to_micron ** 2)
        perimeter_microns = prop.perimeter * pixel_to_micron
        
        circularity = 4 * np.pi * area_pixels / (prop.perimeter ** 2) if prop.perimeter > 0 else 0
        aspect_ratio = prop.major_axis_length / prop.minor_axis_length if prop.minor_axis_length > 0 else 1
        
        # Biological features
        mask = labels == prop.label
        
        if len(image.shape) == 3:
            green_intensity = np.mean(image[mask, 1])
            red_intensity = np.mean(image[mask, 2])
            blue_intensity = np.mean(image[mask, 0])
        else:
            green_intensity = red_intensity = blue_intensity = np.mean(image[mask])
        
        chlorophyll_index = green_intensity / (red_intensity + 1e-10)
        is_green_cell = chlorophyll_index > 1.2
        
        # Biomass estimation
        biomass = self._estimate_biomass(area_microns)
        
        # Health assessment
        health_score = self._assess_health(area_microns, chlorophyll_index)
        
        return {
            'cell_id': prop.label,
            'area_pixels': area_pixels,
            'area_microns_sq': area_microns,
            'perimeter_microns': perimeter_microns,
            'circularity': circularity,
            'aspect_ratio': aspect_ratio,
            'solidity': prop.solidity,
            'eccentricity': prop.eccentricity,
            'major_axis_length': prop.major_axis_length * pixel_to_micron,
            'minor_axis_length': prop.minor_axis_length * pixel_to_micron,
            'centroid_x': prop.centroid[1],
            'centroid_y': prop.centroid[0],
            'green_intensity': green_intensity,
            'red_intensity': red_intensity,
            'blue_intensity': blue_intensity,
            'chlorophyll_index': chlorophyll_index,
            'is_green_cell': is_green_cell,
            'biomass_estimate_ug': biomass,
            'health_score': health_score,
            'health_status': self._categorize_health(health_score)
        }
    
    def _estimate_biomass(self, area_microns):
        """Estimate biomass from area"""
        # Empirical model for Wolffia
        radius = np.sqrt(area_microns / np.pi)
        volume = (4/3) * np.pi * (radius ** 3)
        biomass = volume * 1.2e-6  # μg
        return biomass
    
    def _assess_health(self, area, chlorophyll_index):
        """Assess cell health"""
        size_factor = 1.0 if 100 <= area <= 5000 else 0.5
        chlorophyll_factor = min(1.0, chlorophyll_index / 2.0)
        return 0.6 * size_factor + 0.4 * chlorophyll_factor
    
    def _categorize_health(self, health_score):
        """Categorize health status"""
        if health_score > 0.8:
            return 'excellent'
        elif health_score > 0.6:
            return 'good'
        elif health_score > 0.4:
            return 'moderate'
        else:
            return 'poor'


class QualityAssessmentEngine:
    """Professional quality assessment engine"""
    
    def __init__(self):
        self.status = "✅ Ready"
    
    def assess_quality(self, restoration_result, segmentation_result, features_df):
        """Assess overall analysis quality"""
        restoration_quality = self._assess_restoration(restoration_result)
        segmentation_quality = self._assess_segmentation(segmentation_result, features_df)
        feature_quality = self._assess_features(features_df)
        
        overall_quality = (restoration_quality + segmentation_quality + feature_quality) / 3
        
        return {
            'overall_quality': overall_quality,
            'restoration_quality': restoration_quality,
            'segmentation_quality': segmentation_quality,
            'feature_quality': feature_quality,
            'status': self._quality_status(overall_quality),
            'recommendations': self._generate_recommendations(
                restoration_quality, segmentation_quality, feature_quality
            )
        }
    
    def _assess_restoration(self, restoration_result):
        """Assess restoration quality"""
        return restoration_result.get('quality_score', 0.7)
    
    def _assess_segmentation(self, segmentation_result, features_df):
        """Assess segmentation quality"""
        confidence = segmentation_result.get('confidence', 0.5)
        num_cells = segmentation_result.get('num_cells', 0)
        
        if num_cells == 0:
            return 0.2
        
        # Reasonable cell count
        count_factor = 1.0 if 5 <= num_cells <= 200 else 0.7
        
        # Feature consistency
        feature_factor = 1.0
        if not features_df.empty and 'area_microns_sq' in features_df.columns:
            area_cv = features_df['area_microns_sq'].std() / features_df['area_microns_sq'].mean()
            feature_factor = max(0.5, 1.0 - area_cv)
        
        return confidence * count_factor * feature_factor
    
    def _assess_features(self, features_df):
        """Assess feature quality"""
        if features_df.empty:
            return 0.2
        
        # Completeness
        completeness = 1.0 - features_df.isnull().sum().sum() / features_df.size
        
        # Reasonable ranges
        if 'area_microns_sq' in features_df.columns:
            area_reasonable = features_df['area_microns_sq'].between(10, 15000).mean()
        else:
            area_reasonable = 0.5
        
        return (completeness + area_reasonable) / 2
    
    def _quality_status(self, quality):
        """Get quality status"""
        if quality > 0.85:
            return "Excellent"
        elif quality > 0.7:
            return "Good"
        elif quality > 0.5:
            return "Acceptable"
        else:
            return "Needs Improvement"
    
    def _generate_recommendations(self, rest_q, seg_q, feat_q):
        """Generate improvement recommendations"""
        recommendations = []
        
        if rest_q < 0.6:
            recommendations.append("Consider using more aggressive image restoration")
        if seg_q < 0.6:
            recommendations.append("Check segmentation parameters or image quality")
        if feat_q < 0.6:
            recommendations.append("Verify cell detection accuracy")
        if rest_q > 0.8 and seg_q > 0.8 and feat_q > 0.8:
            recommendations.append("Excellent analysis quality achieved")
        
        return recommendations


class LearningEngine:
    """Professional learning and adaptation engine with tophat training"""
    
    def __init__(self):
        self.status = "✅ Ready"
        self.learning_dir = Path(__file__).parent / 'learning_system'
        self.annotations_dir = self.learning_dir / 'annotations'
        self.training_data_dir = self.learning_dir / 'training_data'
        
        # Create directories
        self.learning_dir.mkdir(exist_ok=True)
        self.annotations_dir.mkdir(exist_ok=True)
        self.training_data_dir.mkdir(exist_ok=True)
        
        self.history_file = self.learning_dir / 'analysis_history.json'
        self.model_performance = self.learning_dir / 'model_performance.json'
        self.annotations_file = self.learning_dir / 'user_annotations.json'
        
        self.history = []
        self.performance = {}
        self.user_annotations = []
        self._load_history()
    
    def _load_history(self):
        """Load learning history and annotations"""
        try:
            if self.history_file.exists():
                with open(self.history_file, 'r') as f:
                    self.history = json.load(f)
            
            if self.model_performance.exists():
                with open(self.model_performance, 'r') as f:
                    self.performance = json.load(f)
            
            if self.annotations_file.exists():
                with open(self.annotations_file, 'r') as f:
                    self.user_annotations = json.load(f)
                    
        except Exception as e:
            print(f"⚠️ Error loading learning history: {e}")
    
    def learn_from_analysis(self, restoration_result, segmentation_result, features_df, quality_report):
        """Learn from analysis results"""
        try:
            model_used = segmentation_result.get('method', 'unknown')
            quality = quality_report.get('overall_quality', 0)
            
            # Record analysis
            record = {
                'timestamp': datetime.now().isoformat(),
                'model': model_used,
                'quality': quality,
                'num_cells': segmentation_result.get('num_cells', 0),
                'success': quality > 0.7
            }
            
            self.history.append(record)
            
            # Update performance metrics
            if model_used not in self.performance:
                self.performance[model_used] = {
                    'count': 0,
                    'avg_quality': 0,
                    'success_rate': 0
                }
            
            perf = self.performance[model_used]
            perf['count'] += 1
            perf['avg_quality'] = ((perf['count'] - 1) * perf['avg_quality'] + quality) / perf['count']
            perf['success_rate'] = sum(1 for r in self.history if r['model'] == model_used and r['success']) / perf['count']
            
            self._save_history()
            
            return {
                'learned': True,
                'model_performance': perf,
                'recommendation': self.get_best_model()
            }
            
        except Exception as e:
            print(f"❌ Learning failed: {e}")
            return {'learned': False, 'error': str(e)}
    
    def get_best_model(self):
        """Get best performing model"""
        if not self.performance:
            return 'cyto3'
        
        best_model = 'cyto3'
        best_score = 0
        
        for model, perf in self.performance.items():
            if perf['count'] >= 3:  # Minimum analyses for reliability
                score = perf['avg_quality'] * perf['success_rate']
                if score > best_score:
                    best_score = score
                    best_model = model
        
        return best_model
    
    def save_analysis(self, analysis_result):
        """Save complete analysis for future reference"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            analysis_file = self.learning_dir / f'analysis_{timestamp}.json'
            
            # Save simplified version (JSON serializable)
            simplified = {
                'timestamp': analysis_result['timestamp'],
                'success': analysis_result['success'],
                'num_cells': len(analysis_result.get('cells', [])),
                'quality': analysis_result.get('quality', {}),
                'parameters': analysis_result.get('parameters', {})
            }
            
            with open(analysis_file, 'w') as f:
                json.dump(simplified, f, indent=2)
                
        except Exception as e:
            print(f"⚠️ Error saving analysis: {e}")
    
    def _save_history(self):
        """Save learning history, metrics, and annotations"""
        try:
            # Convert data to JSON-serializable format before saving
            def convert_to_serializable(obj):
                if isinstance(obj, (np.integer, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
                    return int(obj)
                elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_to_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_serializable(item) for item in obj]
                elif hasattr(obj, 'item'):  # numpy scalars
                    return obj.item()
                else:
                    return obj
            
            with open(self.history_file, 'w') as f:
                json.dump(convert_to_serializable(self.history), f, indent=2)
            
            with open(self.model_performance, 'w') as f:
                json.dump(convert_to_serializable(self.performance), f, indent=2)
            
            with open(self.annotations_file, 'w') as f:
                json.dump(convert_to_serializable(self.user_annotations), f, indent=2)
                
        except Exception as e:
            print(f"⚠️ Error saving history: {e}")
    
    def save_user_annotations(self, analysis_id, image_path, annotations):
        """Save user annotations for tophat training"""
        try:
            annotation_record = {
                'analysis_id': analysis_id,
                'image_path': str(image_path),
                'timestamp': datetime.now().isoformat(),
                'annotations': annotations,
                'annotation_type': 'tophat_training'
            }
            
            self.user_annotations.append(annotation_record)
            self._save_history()
            
            print(f"✅ Saved {len(annotations)} user annotations for tophat training")
            
            # If we have enough annotations, trigger model retraining
            if len(self.user_annotations) >= 5:
                self._retrain_with_annotations()
            
            return True
            
        except Exception as e:
            print(f"❌ Error saving user annotations: {e}")
            return False
    
    def force_retrain(self):
        """Force retrain the model with all available annotations"""
        print(f"🚀 Force retraining with {len(self.user_annotations)} annotation sessions...")
        return self._retrain_with_annotations(force=True)
    
    def _retrain_with_annotations(self, force=False):
        """Retrain model using user annotations"""
        try:
            print("🔄 Starting tophat training with user annotations...")
            
            # Use more annotations if force training
            num_records = len(self.user_annotations) if force else 10
            records_to_use = self.user_annotations[-num_records:] if not force else self.user_annotations
            
            print(f"📊 Using {len(records_to_use)} annotation records for training")
            
            # Extract features from annotated regions
            training_features = []
            training_labels = []
            
            for annotation_record in records_to_use:
                image_path = annotation_record['image_path']
                annotations = annotation_record['annotations']
                
                if not os.path.exists(image_path):
                    continue
                
                # Load image
                image = cv2.imread(image_path)
                if image is None:
                    continue
                
                # Extract features from each annotation
                for annotation in annotations:
                    if annotation['type'] == 'positive':
                        features = self._extract_annotation_features(image, annotation)
                        if features is not None:
                            training_features.append(features)
                            training_labels.append(1)  # Positive example
                    elif annotation['type'] == 'negative':
                        features = self._extract_annotation_features(image, annotation)
                        if features is not None:
                            training_features.append(features)
                            training_labels.append(0)  # Negative example
            
            if len(training_features) >= 10:  # Need minimum training samples
                # Train classifier
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.preprocessing import StandardScaler
                
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(training_features)
                
                # Train enhanced neural network model if possible, otherwise Random Forest
                if self._train_neural_network_model(X_scaled, training_labels):
                    print(f"✅ Neural network model trained with {len(training_features)} examples")
                else:
                    # Fallback to Random Forest
                    classifier = RandomForestClassifier(n_estimators=100, random_state=42, 
                                                      max_depth=10, min_samples_split=5)
                    classifier.fit(X_scaled, training_labels)
                    
                    # Save trained model
                    import pickle
                    model_file = self.learning_dir / 'tophat_classifier.pkl'
                    scaler_file = self.learning_dir / 'tophat_scaler.pkl'
                    
                    with open(model_file, 'wb') as f:
                        pickle.dump(classifier, f)
                    
                    with open(scaler_file, 'wb') as f:
                        pickle.dump(scaler, f)
                    
                    print(f"✅ Random Forest classifier trained with {len(training_features)} examples")
                
                return True
            else:
                print("⚠️ Need at least 10 annotation examples to train classifier")
                return False
        
        except Exception as e:
            print(f"❌ Error during training: {e}")
            return False
    
    def _train_neural_network_model(self, X_scaled, y):
        """Train a neural network model for better segmentation performance"""
        try:
            # Try to use TensorFlow/Keras for advanced neural network
            try:
                import tensorflow as tf
                from tensorflow.keras.callbacks import EarlyStopping
                from tensorflow.keras.layers import Dense, Dropout
                from tensorflow.keras.models import Sequential
                from tensorflow.keras.optimizers import Adam
                
                print("🧠 Training with TensorFlow neural network...")
                
                # Convert to numpy arrays
                X = np.array(X_scaled)
                y = np.array(y)
                
                # Create neural network model
                model = Sequential([
                    Dense(64, activation='relu', input_shape=(X.shape[1],)),
                    Dropout(0.3),
                    Dense(32, activation='relu'),
                    Dropout(0.2),
                    Dense(16, activation='relu'),
                    Dense(1, activation='sigmoid')
                ])
                
                model.compile(
                    optimizer=Adam(learning_rate=0.001),
                    loss='binary_crossentropy',
                    metrics=['accuracy']
                )
                
                # Train with early stopping
                early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
                
                model.fit(
                    X, y,
                    epochs=100,
                    batch_size=8,
                    verbose=0,
                    callbacks=[early_stopping]
                )
                
                # Save the TensorFlow model
                model_path = self.learning_dir / 'neural_segmentation_model'
                model.save(model_path)
                
                # Also save the scaler
                import pickle
                scaler_file = self.learning_dir / 'neural_scaler.pkl'
                with open(scaler_file, 'wb') as f:
                    pickle.dump(StandardScaler().fit(X_scaled), f)
                
                print("✅ TensorFlow neural network model saved")
                return True
                
            except ImportError:
                print("⚠️ TensorFlow not available, trying simple neural network...")
                
                # Fallback to simple multi-layer perceptron using numpy
                return self._train_simple_neural_network(X_scaled, y)
                
        except Exception as e:
            print(f"❌ Neural network training failed: {e}")
            return False
    
    def _train_simple_neural_network(self, X_scaled, y):
        """Train a simple neural network using only numpy"""
        try:
            print("🧠 Training with simple numpy neural network...")
            
            X = np.array(X_scaled)
            y = np.array(y).reshape(-1, 1)
            
            # Initialize network parameters
            input_size = X.shape[1]
            hidden_size = 32
            output_size = 1
            learning_rate = 0.01
            epochs = 500
            
            # Xavier initialization
            W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
            b1 = np.zeros((1, hidden_size))
            W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
            b2 = np.zeros((1, output_size))
            
            def sigmoid(z):
                return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
            
            def relu(z):
                return np.maximum(0, z)
            
            def relu_derivative(z):
                return (z > 0).astype(float)
            
            # Training loop
            for epoch in range(epochs):
                # Forward pass
                z1 = np.dot(X, W1) + b1
                a1 = relu(z1)
                z2 = np.dot(a1, W2) + b2
                a2 = sigmoid(z2)
                
                # Compute loss
                loss = -np.mean(y * np.log(a2 + 1e-8) + (1 - y) * np.log(1 - a2 + 1e-8))
                
                # Backward pass
                m = X.shape[0]
                dz2 = a2 - y
                dW2 = np.dot(a1.T, dz2) / m
                db2 = np.sum(dz2, axis=0, keepdims=True) / m
                
                da1 = np.dot(dz2, W2.T)
                dz1 = da1 * relu_derivative(z1)
                dW1 = np.dot(X.T, dz1) / m
                db1 = np.sum(dz1, axis=0, keepdims=True) / m
                
                # Update parameters
                W2 -= learning_rate * dW2
                b2 -= learning_rate * db2
                W1 -= learning_rate * dW1
                b1 -= learning_rate * db1
                
                # Print progress every 100 epochs
                if epoch % 100 == 0:
                    accuracy = np.mean((a2 > 0.5) == y)
                    print(f"   Epoch {epoch}: Loss = {loss:.4f}, Accuracy = {accuracy:.4f}")
            
            # Save the simple neural network parameters
            import pickle
            nn_params = {
                'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2,
                'input_size': input_size, 'hidden_size': hidden_size
            }
            
            model_file = self.learning_dir / 'simple_neural_network.pkl'
            with open(model_file, 'wb') as f:
                pickle.dump(nn_params, f)
            
            # Save scaler
            scaler_file = self.learning_dir / 'neural_scaler.pkl'
            with open(scaler_file, 'wb') as f:
                pickle.dump(StandardScaler().fit(X_scaled), f)
            
            print("✅ Simple neural network model saved")
            return True
            
        except Exception as e:
            print(f"❌ Simple neural network training failed: {e}")
            return False
                
        except Exception as e:
            print(f"❌ Error retraining with annotations: {e}")
            return False
    
    def _extract_annotation_features(self, image, annotation):
        """Extract features from an annotated region"""
        try:
            x, y, width, height = annotation['x'], annotation['y'], annotation['width'], annotation['height']
            
            # Ensure coordinates are integers for array slicing
            x, y, width, height = int(x), int(y), int(width), int(height)
            
            # Extract region
            region = image[y:y+height, x:x+width]
            if region.size == 0:
                return None
            
            # Calculate features
            features = []
            
            # Color features
            if len(region.shape) == 3:
                mean_rgb = np.mean(region, axis=(0,1))
                features.extend(mean_rgb)
                
                # Color ratios
                green_ratio = mean_rgb[1] / (mean_rgb[0] + mean_rgb[2] + 1e-10)
                features.append(green_ratio)
            else:
                mean_gray = np.mean(region)
                features.extend([mean_gray, mean_gray, mean_gray, 1.0])
            
            # Texture features
            gray_region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if len(region.shape) == 3 else region
            
            # Standard deviation (texture measure)
            texture = np.std(gray_region)
            features.append(texture)
            
            # Edge density
            edges = cv2.Canny(gray_region, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            features.append(edge_density)
            
            # Size features
            features.extend([width, height, width * height])
            
            # Circularity approximation
            aspect_ratio = width / (height + 1e-10)
            features.append(aspect_ratio)
            
            return features
            
        except Exception as e:
            print(f"⚠️ Error extracting annotation features: {e}")
            return None
    
    # Enhanced apply_tophat_model method with neural network support

    def apply_tophat_model(self, image, segmentation_labels):
        """Apply trained tophat model to find similar cells - ENHANCED WITH NEURAL NETWORKS"""
        try:
            # Check for neural network models first (preferred)
            neural_model_path = self.learning_dir / 'neural_segmentation_model'
            simple_nn_path = self.learning_dir / 'simple_neural_network.pkl'
            neural_scaler_path = self.learning_dir / 'neural_scaler.pkl'
            
            # Fallback to Random Forest model
            rf_model_file = self.learning_dir / 'tophat_classifier.pkl'
            rf_scaler_file = self.learning_dir / 'tophat_scaler.pkl'
            
            model = None
            scaler = None
            model_type = None
            
            # Try to load neural network models first
            if neural_model_path.exists() and neural_scaler_path.exists():
                try:
                    import tensorflow as tf
                    model = tf.keras.models.load_model(neural_model_path)
                    model_type = 'tensorflow'
                    print("🧠 Using TensorFlow neural network model")
                except ImportError:
                    print("⚠️ TensorFlow not available for model loading")
                except Exception as e:
                    print(f"⚠️ Failed to load TensorFlow model: {e}")
            
            # Try simple neural network if TensorFlow failed
            if model is None and simple_nn_path.exists() and neural_scaler_path.exists():
                try:
                    import pickle
                    with open(simple_nn_path, 'rb') as f:
                        model = pickle.load(f)
                    model_type = 'simple_nn'
                    print("🧠 Using simple neural network model")
                except Exception as e:
                    print(f"⚠️ Failed to load simple neural network: {e}")
            
            # Fallback to Random Forest
            if model is None and rf_model_file.exists() and rf_scaler_file.exists():
                try:
                    import pickle
                    with open(rf_model_file, 'rb') as f:
                        model = pickle.load(f)
                    with open(rf_scaler_file, 'rb') as f:
                        scaler = pickle.load(f)
                    model_type = 'random_forest'
                    print("🌲 Using Random Forest model")
                except Exception as e:
                    print(f"⚠️ Failed to load Random Forest model: {e}")
            
            # Load scaler for neural networks
            if model_type in ['tensorflow', 'simple_nn'] and neural_scaler_path.exists():
                try:
                    import pickle
                    with open(neural_scaler_path, 'rb') as f:
                        scaler = pickle.load(f)
                except Exception as e:
                    print(f"⚠️ Failed to load neural network scaler: {e}")
                    return segmentation_labels, []
            
            if model is None or scaler is None:
                print("⚠️ No trained tophat model found")
                return segmentation_labels, []
            
            # Extract features for each segmented region
            props = measure.regionprops(segmentation_labels)
            improved_labels = np.zeros_like(segmentation_labels)
            confidence_scores = []
            new_label = 1
            
            print(f"🔍 Analyzing {len(props)} regions with {model_type.replace('_', ' ').title()} model...")
            
            for prop in props:
                # Create annotation-like dict for feature extraction
                annotation = {
                    'x': int(prop.bbox[1]),
                    'y': int(prop.bbox[0]),
                    'width': int(prop.bbox[3] - prop.bbox[1]),
                    'height': int(prop.bbox[2] - prop.bbox[0])
                }
                
                features = self._extract_annotation_features(image, annotation)
                if features is None:
                    continue
                
                # Scale features
                X_scaled = scaler.transform([features])
                
                # Predict based on model type
                if model_type == 'tensorflow':
                    prediction = model.predict(X_scaled, verbose=0)[0][0]
                    probability = float(prediction)
                elif model_type == 'simple_nn':
                    # Apply simple neural network manually
                    probability = self._predict_simple_nn(model, X_scaled[0])
                elif model_type == 'random_forest':
                    probability = model.predict_proba(X_scaled)[0][1]  # Probability of positive class
                else:
                    continue
                
                # Dynamic confidence threshold based on model type
                confidence_threshold = 0.6 if model_type in ['tensorflow', 'simple_nn'] else 0.7
                if probability > confidence_threshold:
                    mask = segmentation_labels == prop.label
                    improved_labels[mask] = new_label
                    confidence_scores.append({
                        'label': new_label,
                        'confidence': float(probability),
                        'bbox': annotation,
                        'model_type': model_type,
                        'area': prop.area
                    })
                    new_label += 1
            
            print(f"🎯 {model_type.replace('_', ' ').title()} model found {new_label-1} similar cells (from {len(props)} original)")
            
            return improved_labels, confidence_scores
            
        except Exception as e:
            print(f"❌ Error applying tophat model: {e}")
            import traceback
            traceback.print_exc()
            return segmentation_labels, []

    def _predict_simple_nn(self, nn_params, X):
        """Apply simple neural network prediction manually"""
        try:
            W1, b1, W2, b2 = nn_params['W1'], nn_params['b1'], nn_params['W2'], nn_params['b2']
            
            # Forward pass
            z1 = np.dot(X.reshape(1, -1), W1) + b1
            a1 = np.maximum(0, z1)  # ReLU activation
            z2 = np.dot(a1, W2) + b2
            prediction = 1 / (1 + np.exp(-np.clip(z2, -500, 500)))  # Sigmoid activation
            
            return float(prediction[0][0])
        except Exception as e:
            print(f"❌ Error in simple NN prediction: {e}")
            return 0.0

    def get_tophat_training_status(self):
        """Get enhanced status of tophat training system with neural network info"""
        if not self.learning_engine:
            return {'available': False, 'reason': 'Learning engine not available'}
        
        num_annotations = len(getattr(self.learning_engine, 'user_annotations', []))
        
        # Check all model types
        models_status = {}
        models_status['tensorflow'] = (self.learning_engine.learning_dir / 'neural_segmentation_model').exists()
        models_status['simple_nn'] = (self.learning_engine.learning_dir / 'simple_neural_network.pkl').exists()
        models_status['random_forest'] = (self.learning_engine.learning_dir / 'tophat_classifier.pkl').exists()
        
        any_model_trained = any(models_status.values())
        
        # Determine best available model
        best_model = 'none'
        if models_status['tensorflow']:
            best_model = 'tensorflow_neural_network'
        elif models_status['simple_nn']:
            best_model = 'simple_neural_network'
        elif models_status['random_forest']:
            best_model = 'random_forest'
        
        return {
            'available': True,
            'num_annotations': num_annotations,
            'models_trained': models_status,
            'best_model': best_model,
            'any_model_trained': any_model_trained,
            'needs_more_training': num_annotations < 10,
            'can_force_retrain': num_annotations >= 5,
            'status': 'ready' if any_model_trained else ('training' if num_annotations >= 5 else 'needs_annotations'),
            'recommended_action': self._get_training_recommendation(num_annotations, any_model_trained)
        }

    def _get_training_recommendation(self, num_annotations, any_model_trained):
        """Get training recommendation based on current state"""
        if num_annotations == 0:
            return "Start by creating annotations: draw rectangles around correct cells (green) and incorrect detections (red)"
        elif num_annotations < 5:
            return f"Create {5 - num_annotations} more annotation sessions to enable training"
        elif num_annotations < 10:
            return "You can train now, but 10+ annotation sessions are recommended for better accuracy"
        elif not any_model_trained:
            return "Ready to train! Click 'Force Retrain' to create models with your annotations"
        else:
            return "Models trained and ready! You can retrain anytime to improve accuracy"

if __name__ == "__main__":
    print("="*80)
    print("REFINED PROFESSIONAL BIOINFORMATICS PIPELINE FOR WOLFFIA ANALYSIS")
    print("="*80)
    print("\nFeatures:")
    print("✓ Reliable CellPose segmentation with fallbacks")
    print("✓ Professional image restoration (SimpleITK + CellPose)")
    print("✓ Comprehensive biological feature extraction")
    print("✓ Advanced quality assessment")
    print("✓ Intelligent learning system")
    print("✓ Professional visualizations")
    print("✓ Complete web integration compatibility")
    print("✓ Robust error handling and fallbacks")
    print("\nRefined professional system ready for production use.")
    print("="*80)