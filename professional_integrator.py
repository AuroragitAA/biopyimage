# professional_integrator.py
"""
Professional Wolffia System Integrator
Seamlessly connects all professional components into a unified workflow
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd

# Import all professional components
try:
    from advanced_wolffia_analyzer import AdvancedImageProcessor, BiologicalFeatureExtractor, StatisticalAnalyzer, AnalysisConfig
    from batch_processor import BatchProcessor, BatchJobConfig, QualityController, BatchExporter
    from database_manager import DatabaseManager, DatabaseConfig
    from ml_enhancement import MLEnhancedAnalyzer, MLConfig
    from wolffia_analyzer import WolffiaAnalyzer
    from image_processor import ImageProcessor
    from segmentation import EnhancedCellSegmentation
    PROFESSIONAL_COMPONENTS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Some professional components not available: {e}")
    PROFESSIONAL_COMPONENTS_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProfessionalWolffiaSystem:
    """
    Master integration system that orchestrates all professional components
    """
    
    def __init__(self, config_path: str = None):
        """Initialize the integrated professional system"""
        self.config = self._load_configuration(config_path)
        self.components = {}
        self.system_status = {
            'initialized': False,
            'database_ready': False,
            'ml_ready': False,
            'batch_ready': False,
            'analyzer_ready': False
        }
        
        logger.info("🚀 Initializing Professional Wolffia System Integration...")
        self._initialize_all_components()
        
    def _load_configuration(self, config_path: str) -> Dict:
        """Load system configuration"""
        default_config = {
            'analysis': {
                'pixel_to_micron': 1.0,
                'chlorophyll_threshold': 0.6,
                'min_cell_area': 30,
                'max_cell_area': 8000,
                'quality_threshold': 0.7
            },
            'database': {
                'enabled': True,
                'db_path': 'wolffia_professional.db',
                'backup_enabled': True
            },
            'ml': {
                'enabled': True,
                'auto_training': True,
                'confidence_threshold': 0.75
            },
            'batch': {
                'enabled': True,
                'max_workers': 4,
                'quality_control': True
            },
            'system': {
                'professional_mode': True,
                'auto_export': True,
                'monitoring_enabled': True
            }
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                # Merge configurations
                for section, settings in user_config.items():
                    if section in default_config:
                        default_config[section].update(settings)
                    else:
                        default_config[section] = settings
        
        return default_config
    
    def _initialize_all_components(self):
        """Initialize all professional components in correct order"""
        try:
            # 1. Initialize Core Analyzer
            self._initialize_core_analyzer()
            
            # 2. Initialize Database
            if self.config['database']['enabled']:
                self._initialize_database()
            
            # 3. Initialize ML Enhancement
            if self.config['ml']['enabled']:
                self._initialize_ml_enhancement()
            
            # 4. Initialize Batch Processing
            if self.config['batch']['enabled']:
                self._initialize_batch_processing()
            
            # 5. Setup integrated workflows
            self._setup_integrated_workflows()
            
            self.system_status['initialized'] = True
            logger.info("✅ Professional Wolffia System fully integrated and ready")
            
        except Exception as e:
            logger.error(f"❌ System initialization failed: {str(e)}")
            raise
    
    def _initialize_core_analyzer(self):
        """Initialize the core analysis system"""
        try:
            if PROFESSIONAL_COMPONENTS_AVAILABLE:
                # Use professional components
                analysis_config = AnalysisConfig(
                    pixel_to_micron=self.config['analysis']['pixel_to_micron'],
                    min_cell_area=self.config['analysis']['min_cell_area'],
                    max_cell_area=self.config['analysis']['max_cell_area'],
                    chlorophyll_threshold=self.config['analysis']['chlorophyll_threshold']
                )
                
                self.components['image_processor'] = AdvancedImageProcessor(analysis_config)
                self.components['feature_extractor'] = BiologicalFeatureExtractor(analysis_config)
                self.components['statistical_analyzer'] = StatisticalAnalyzer(analysis_config)
                
                # Enhanced analyzer with professional components
                self.components['analyzer'] = WolffiaAnalyzer(
                    pixel_to_micron_ratio=self.config['analysis']['pixel_to_micron'],
                    chlorophyll_threshold=self.config['analysis']['chlorophyll_threshold'],
                    min_cell_area=self.config['analysis']['min_cell_area'],
                    max_cell_area=self.config['analysis']['max_cell_area']
                )
                
                logger.info("✅ Professional analyzer components initialized")
            else:
                # Fallback to basic analyzer
                self.components['analyzer'] = WolffiaAnalyzer()
                logger.info("⚠️ Using basic analyzer (professional components not available)")
            
            self.system_status['analyzer_ready'] = True
            
        except Exception as e:
            logger.error(f"❌ Core analyzer initialization failed: {str(e)}")
            raise
    
    def _initialize_database(self):
        """Initialize database management"""
        try:
            db_config = DatabaseConfig(
                db_path=self.config['database']['db_path'],
                backup_enabled=self.config['database']['backup_enabled'],
                auto_cleanup=True
            )
            
            self.components['database'] = DatabaseManager(db_config)
            
            # Create default project if none exists
            self._ensure_default_project()
            
            self.system_status['database_ready'] = True
            logger.info("✅ Database system initialized")
            
        except Exception as e:
            logger.error(f"❌ Database initialization failed: {str(e)}")
            self.config['database']['enabled'] = False
    
# Fixes for professional_integrator.py

# Add this method to the ProfessionalWolffiaSystem class to fix the workflow creation:

    def _create_single_analysis_workflow(self):
        """Create optimized single image analysis workflow"""
        def workflow(image_path: str, **kwargs):
            try:
                # Choose best available analyzer with proper fallback chain
                analyzer = None
                
                # Try ML-enhanced analyzer first
                if self.system_status.get('ml_ready') and 'ml_analyzer' in self.components:
                    analyzer = self.components['ml_analyzer']
                    # Ensure it has the required method
                    if not hasattr(analyzer, 'analyze_single_image'):
                        # Use the base analyzer from ML analyzer
                        if hasattr(analyzer, 'base_analyzer'):
                            analyzer = analyzer.base_analyzer
                        else:
                            analyzer = None
                
                # Fallback to standard analyzer
                if not analyzer and 'analyzer' in self.components:
                    analyzer = self.components['analyzer']
                
                if not analyzer:
                    raise Exception("No analyzer available")
                
                # Perform analysis with proper method detection
                result = None
                
                # Try different method names in order of preference
                if hasattr(analyzer, 'analyze_single_image'):
                    result = analyzer.analyze_single_image(image_path, **kwargs)
                elif hasattr(analyzer, 'analyze_image_professional'):
                    result = analyzer.analyze_image_professional(image_path, **kwargs)
                elif hasattr(analyzer, 'analyze_image'):
                    result = analyzer.analyze_image(image_path, **kwargs)
                else:
                    raise Exception(f"Analyzer {type(analyzer).__name__} has no compatible analysis method")
                
                # Ensure result is valid
                if result is None:
                    result = {'success': False, 'error': 'Analysis returned None'}
                
                # Store in database if available
                if self.system_status.get('database_ready') and result.get('success'):
                    try:
                        self._store_analysis_result(result)
                    except Exception as db_error:
                        logger.warning(f"Database storage failed: {str(db_error)}")
                
                # Add system metadata
                result['system_info'] = {
                    'professional_mode': True,
                    'components_used': list(self.components.keys()),
                    'workflow': 'single_analysis',
                    'analyzer_type': type(analyzer).__name__
                }
                
                return result
                
            except Exception as e:
                logger.error(f"❌ Single analysis workflow error: {str(e)}")
                return {
                    'success': False,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
        
        return workflow

# Also update the _initialize_ml_enhancement method to properly set up the ML analyzer:

    def _initialize_ml_enhancement(self):
        """Initialize ML enhancement system"""
        try:
            if not PROFESSIONAL_COMPONENTS_AVAILABLE:
                logger.warning("⚠️ ML components not available")
                return
            
            ml_config = MLConfig(
                confidence_threshold=self.config['ml']['confidence_threshold'],
                enable_hyperparameter_tuning=True
            )
            
            base_analyzer = self.components.get('analyzer')
            if base_analyzer:
                # Ensure MLEnhancedAnalyzer has all required methods
                ml_analyzer = MLEnhancedAnalyzer(base_analyzer, ml_config)
                
                # Add missing methods if needed
                if not hasattr(ml_analyzer, 'analyze_single_image'):
                    # Bind the method dynamically
                    ml_analyzer.analyze_single_image = lambda image_path, **kwargs: ml_analyzer.analyze_with_ml_enhancement(image_path)
                
                self.components['ml_analyzer'] = ml_analyzer
                
                # Auto-training if enabled and we have data
                if self.config['ml']['auto_training']:
                    self._check_and_train_ml_models()
                
                self.system_status['ml_ready'] = True
                logger.info("✅ ML enhancement system initialized")
            else:
                logger.warning("⚠️ No base analyzer available for ML enhancement")
            
        except Exception as e:
            logger.error(f"❌ ML initialization failed: {str(e)}")
            self.config['ml']['enabled'] = False
    
    def _initialize_batch_processing(self):
        """Initialize batch processing system"""
        try:
            if not PROFESSIONAL_COMPONENTS_AVAILABLE:
                logger.warning("⚠️ Batch processing components not available")
                return
            
            batch_config = BatchJobConfig(
                max_workers=self.config['batch']['max_workers'],
                enable_qc=self.config['batch']['quality_control'],
                quality_threshold=self.config['analysis']['quality_threshold'],
                export_formats=['csv', 'excel', 'json']
            )
            
            analyzer_system = self.components.get('ml_analyzer') or self.components.get('analyzer')
            if analyzer_system:
                self.components['batch_processor'] = BatchProcessor(analyzer_system, batch_config)
                self.system_status['batch_ready'] = True
                logger.info("✅ Batch processing system initialized")
            
        except Exception as e:
            logger.error(f"❌ Batch processing initialization failed: {str(e)}")
            self.config['batch']['enabled'] = False
    
    def _setup_integrated_workflows(self):
        """Setup integrated workflows between components"""
        try:
            # Create workflow connections
            self.workflows = {
                'single_analysis': self._create_single_analysis_workflow(),
                'batch_analysis': self._create_batch_analysis_workflow(),
                'ml_enhanced_analysis': self._create_ml_enhanced_workflow(),
                'comprehensive_analysis': self._create_comprehensive_workflow()
            }
            
            logger.info("✅ Integrated workflows configured")
            
        except Exception as e:
            logger.error(f"❌ Workflow setup failed: {str(e)}")
    
    def _create_single_analysis_workflow(self):
        """Create optimized single image analysis workflow"""
        def workflow(image_path: str, **kwargs):
            try:
                # Choose best available analyzer
                analyzer = (self.components.get('ml_analyzer') or 
                           self.components.get('analyzer'))
                
                if not analyzer:
                    raise Exception("No analyzer available")
                
                # Perform analysis
                result = analyzer.analyze_single_image(image_path, **kwargs)
                
                # Store in database if available
                if self.system_status['database_ready'] and result.get('success'):
                    self._store_analysis_result(result)
                
                # Add system metadata
                result['system_info'] = {
                    'professional_mode': True,
                    'components_used': list(self.components.keys()),
                    'workflow': 'single_analysis'
                }
                
                return result
                
            except Exception as e:
                logger.error(f"❌ Single analysis workflow error: {str(e)}")
                return {'success': False, 'error': str(e)}
        
        return workflow
    
    def _create_batch_analysis_workflow(self):
        """Create optimized batch analysis workflow"""
        def workflow(image_paths: List[str], progress_callback=None, **kwargs):
            try:
                if not self.system_status['batch_ready']:
                    # Fallback to sequential processing
                    return self._fallback_batch_analysis(image_paths, progress_callback, **kwargs)
                
                batch_processor = self.components['batch_processor']
                
                # Run batch processing with professional features
                result = batch_processor.process_batch(image_paths, progress_callback)
                
                # Store batch results in database
                if self.system_status['database_ready']:
                    self._store_batch_results(result)
                
                return result
                
            except Exception as e:
                logger.error(f"❌ Batch analysis workflow error: {str(e)}")
                return {'success': False, 'error': str(e)}
        
        return workflow
    
    def _create_ml_enhanced_workflow(self):
        """Create ML-enhanced analysis workflow"""
        def workflow(image_path: str, **kwargs):
            try:
                if not self.system_status['ml_ready']:
                    # Fallback to standard analysis
                    return self.workflows['single_analysis'](image_path, **kwargs)
                
                ml_analyzer = self.components['ml_analyzer']
                result = ml_analyzer.analyze_with_ml_enhancement(image_path)
                
                # Store with ML metadata
                if self.system_status['database_ready'] and result.get('success'):
                    result['ml_enhanced'] = True
                    self._store_analysis_result(result)
                
                return result
                
            except Exception as e:
                logger.error(f"❌ ML enhanced workflow error: {str(e)}")
                return self.workflows['single_analysis'](image_path, **kwargs)
        
        return workflow
    
    def _create_comprehensive_workflow(self):
        """Create comprehensive analysis workflow with all features"""
        def workflow(image_path: str, **kwargs):
            try:
                # Use ML-enhanced workflow as base
                result = self.workflows['ml_enhanced_analysis'](image_path, **kwargs)
                
                if result.get('success'):
                    # Add comprehensive statistics
                    if PROFESSIONAL_COMPONENTS_AVAILABLE and 'statistical_analyzer' in self.components:
                        enhanced_stats = self.components['statistical_analyzer'].generate_comprehensive_report([result])
                        result['comprehensive_stats'] = enhanced_stats
                    
                    # Add quality assessment
                    result['quality_assessment'] = self._assess_comprehensive_quality(result)
                    
                    # Add recommendations
                    result['recommendations'] = self._generate_recommendations(result)
                
                result['workflow'] = 'comprehensive_analysis'
                return result
                
            except Exception as e:
                logger.error(f"❌ Comprehensive workflow error: {str(e)}")
                return self.workflows['single_analysis'](image_path, **kwargs)
        
        return workflow
    
    def _ensure_default_project(self):
        """Ensure a default project exists in database"""
        try:
            if 'database' in self.components:
                db = self.components['database']
                # Check if any projects exist
                with db.get_connection() as conn:
                    projects = conn.execute("SELECT COUNT(*) FROM projects").fetchone()[0]
                    
                if projects == 0:
                    # Create default project
                    project_id = db.create_project(
                        name="Wolffia Analysis Project",
                        description="Default project for Wolffia bioimage analysis",
                        operator_name="System",
                        metadata={"created_by": "professional_system", "auto_created": True}
                    )
                    
                    # Create default experiment
                    experiment_id = db.create_experiment(
                        project_id=project_id,
                        experiment_name="General Analysis",
                        description="General Wolffia analysis experiments"
                    )
                    
                    # Store for later use
                    self.default_project_id = project_id
                    self.default_experiment_id = experiment_id
                    
                    logger.info(f"✅ Default project created (ID: {project_id})")
        except Exception as e:
            logger.error(f"❌ Default project creation failed: {str(e)}")
    
    def _store_analysis_result(self, result: Dict):
        """Store analysis result in database"""
        try:
            if 'database' not in self.components:
                return
                
            db = self.components['database']
            
            # Store image first
            image_id = db.store_image(
                experiment_id=getattr(self, 'default_experiment_id', 1),
                filename=Path(result['image_path']).name,
                file_path=result['image_path']
            )
            
            # Store analysis result
            analysis_id = db.store_analysis_result(image_id, result)
            
            result['database_ids'] = {
                'image_id': image_id,
                'analysis_id': analysis_id
            }
            
        except Exception as e:
            logger.error(f"❌ Database storage error: {str(e)}")
    
    def _store_batch_results(self, batch_result: Dict):
        """Store batch analysis results"""
        try:
            if 'database' not in self.components or not batch_result.get('success'):
                return
                
            # Store each successful result
            for result in batch_result.get('individual_results', []):
                if result.get('success'):
                    self._store_analysis_result(result)
                    
        except Exception as e:
            logger.error(f"❌ Batch storage error: {str(e)}")
    
    def _check_and_train_ml_models(self):
        """Check if ML models need training and train them"""
        try:
            if 'ml_analyzer' not in self.components:
                return
            
            ml_analyzer = self.components['ml_analyzer']
            
            # Check if we have enough data for training
            if len(ml_analyzer.training_data) >= 50:
                logger.info("🎓 Starting ML model training...")
                training_result = ml_analyzer.train_ml_models()
                
                if training_result.get('error'):
                    logger.warning(f"⚠️ ML training had issues: {training_result['error']}")
                else:
                    logger.info("✅ ML models trained successfully")
            else:
                logger.info("ℹ️ Insufficient data for ML training (need 50+ samples)")
                
        except Exception as e:
            logger.error(f"❌ ML training check failed: {str(e)}")
    
    def _fallback_batch_analysis(self, image_paths: List[str], progress_callback=None, **kwargs):
        """Fallback batch analysis using basic analyzer"""
        try:
            analyzer = self.components.get('analyzer')
            if not analyzer:
                raise Exception("No analyzer available for batch processing")
            
            results = []
            total = len(image_paths)
            
            for i, image_path in enumerate(image_paths):
                try:
                    result = self.workflows['single_analysis'](image_path, **kwargs)
                    results.append(result)
                    
                    if progress_callback:
                        progress_callback({
                            'completed': i + 1,
                            'total': total,
                            'current_image': Path(image_path).name,
                            'success': result.get('success', False)
                        })
                        
                except Exception as e:
                    logger.error(f"❌ Batch item failed {image_path}: {str(e)}")
                    results.append({'success': False, 'error': str(e), 'image_path': image_path})
            
            # Calculate summary
            successful = [r for r in results if r.get('success')]
            
            return {
                'success': True,
                'batch_summary': {
                    'total_images': total,
                    'successful_analyses': len(successful),
                    'failed_analyses': total - len(successful),
                    'success_rate': len(successful) / total * 100 if total > 0 else 0
                },
                'individual_results': results,
                'workflow': 'fallback_batch'
            }
            
        except Exception as e:
            logger.error(f"❌ Fallback batch analysis failed: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _assess_comprehensive_quality(self, result: Dict) -> Dict:
        """Assess comprehensive quality of analysis"""
        try:
            quality_factors = []
            
            # Basic success factor
            if result.get('success'):
                quality_factors.append(1.0)
            else:
                return {'overall_quality': 0.0, 'grade': 'FAILED'}
            
            # Cell count factor
            cell_count = result.get('total_cells', 0)
            if cell_count > 0:
                count_factor = min(cell_count / 50.0, 1.0)  # Normalize to reasonable range
                quality_factors.append(count_factor)
            
            # Processing quality factor
            processing_quality = result.get('quality_score', 0.5)
            quality_factors.append(processing_quality)
            
            # Data completeness factor
            completeness = 1.0 if result.get('cell_data') else 0.5
            quality_factors.append(completeness)
            
            # ML enhancement factor (if available)
            if result.get('ml_enhancements'):
                ml_quality = 0.8  # ML enhanced results get bonus
                quality_factors.append(ml_quality)
            
            overall_quality = np.mean(quality_factors)
            
            # Determine grade
            if overall_quality >= 0.9:
                grade = 'EXCELLENT'
            elif overall_quality >= 0.75:
                grade = 'GOOD'
            elif overall_quality >= 0.6:
                grade = 'ACCEPTABLE'
            else:
                grade = 'POOR'
            
            return {
                'overall_quality': float(overall_quality),
                'grade': grade,
                'factors': {
                    'cell_detection': count_factor if 'count_factor' in locals() else 0,
                    'processing_quality': processing_quality,
                    'data_completeness': completeness,
                    'ml_enhanced': result.get('ml_enhancements') is not None
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Quality assessment error: {str(e)}")
            return {'overall_quality': 0.5, 'grade': 'UNKNOWN'}
    
    def _generate_recommendations(self, result: Dict) -> List[str]:
        """Generate actionable recommendations based on analysis"""
        try:
            recommendations = []
            
            if not result.get('success'):
                recommendations.append("❌ Analysis failed - check image quality and format")
                return recommendations
            
            cell_count = result.get('total_cells', 0)
            quality_score = result.get('quality_score', 0.5)
            
            # Cell count recommendations
            if cell_count == 0:
                recommendations.append("🔍 No cells detected - try adjusting segmentation parameters")
            elif cell_count < 5:
                recommendations.append("⚠️ Very few cells detected - verify image quality and parameters")
            elif cell_count > 200:
                recommendations.append("📊 High cell density detected - consider image cropping for better accuracy")
            
            # Quality recommendations
            if quality_score < 0.6:
                recommendations.append("📷 Image quality is low - consider better lighting or focus")
            elif quality_score > 0.9:
                recommendations.append("✅ Excellent image quality - ideal conditions for analysis")
            
            # ML recommendations
            if not result.get('ml_enhancements'):
                recommendations.append("🤖 Enable ML enhancements for improved accuracy and insights")
            
            # Database recommendations
            if not result.get('database_ids'):
                recommendations.append("💾 Enable database storage for better data management")
            
            # Default positive recommendation
            if not recommendations:
                recommendations.append("✅ Analysis completed successfully with good quality results")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"❌ Recommendations generation error: {str(e)}")
            return ["📊 Analysis complete - review results for insights"]
    
    # PUBLIC API METHODS
    
    def analyze_image(self, image_path: str, workflow: str = 'auto', **kwargs) -> Dict:
        """
        Main public API for image analysis
        
        Args:
            image_path: Path to image file
            workflow: 'auto', 'single', 'ml_enhanced', 'comprehensive'
            **kwargs: Additional parameters
        """
        try:
            # Auto-select best workflow
            if workflow == 'auto':
                if self.system_status['ml_ready']:
                    workflow = 'comprehensive'
                elif self.system_status['analyzer_ready']:
                    workflow = 'single_analysis'
                else:
                    raise Exception("No analysis workflow available")
            
            # Execute workflow
            if workflow in self.workflows:
                result = self.workflows[workflow](image_path, **kwargs)
            else:
                # Fallback to single analysis
                result = self.workflows['single_analysis'](image_path, **kwargs)
            
            # Add system timestamp
            result['system_timestamp'] = datetime.now().isoformat()
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Image analysis error: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'image_path': image_path,
                'system_timestamp': datetime.now().isoformat()
            }
    
    def analyze_batch(self, image_paths: List[str], progress_callback=None, **kwargs) -> Dict:
        """
        Batch analysis with professional features
        """
        try:
            return self.workflows['batch_analysis'](image_paths, progress_callback, **kwargs)
        except Exception as e:
            logger.error(f"❌ Batch analysis error: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def get_system_status(self) -> Dict:
        """Get comprehensive system status"""
        try:
            status = {
                'system_status': self.system_status.copy(),
                'components_available': list(self.components.keys()),
                'workflows_available': list(self.workflows.keys()) if hasattr(self, 'workflows') else [],
                'configuration': self.config,
                'professional_features': {
                    'database_integration': self.system_status['database_ready'],
                    'ml_enhancement': self.system_status['ml_ready'],
                    'batch_processing': self.system_status['batch_ready'],
                    'quality_control': self.config.get('batch', {}).get('quality_control', False),
                    'auto_export': self.config.get('system', {}).get('auto_export', False)
                }
            }
            
            # Add database statistics if available
            if self.system_status['database_ready']:
                try:
                    db_stats = self.components['database'].get_database_statistics()
                    status['database_statistics'] = db_stats
                except:
                    pass
            
            return status
            
        except Exception as e:
            logger.error(f"❌ System status error: {str(e)}")
            return {'error': str(e)}
    
    def export_results(self, format: str = 'csv', **kwargs) -> Optional[str]:
        """Export analysis results"""
        try:
            if self.system_status['database_ready']:
                # Export from database
                return self.components['database'].export_data(f'full_database', **kwargs)
            elif 'analyzer' in self.components:
                # Export from analyzer history
                return self.components['analyzer'].export_results(format, **kwargs)
            else:
                logger.warning("⚠️ No export source available")
                return None
                
        except Exception as e:
            logger.error(f"❌ Export error: {str(e)}")
            return None

# Global system instance
_professional_system = None

def get_professional_system(config_path: str = None) -> ProfessionalWolffiaSystem:
    """Get or create the global professional system instance"""
    global _professional_system
    
    if _professional_system is None:
        _professional_system = ProfessionalWolffiaSystem(config_path)
    
    return _professional_system

def initialize_professional_system(config_path: str = None) -> ProfessionalWolffiaSystem:
    """Initialize and return the professional system"""
    return get_professional_system(config_path)

# Convenience functions for direct use
def analyze_image_professional(image_path: str, **kwargs) -> Dict:
    """Direct professional image analysis"""
    system = get_professional_system()
    return system.analyze_image(image_path, **kwargs)

def analyze_batch_professional(image_paths: List[str], **kwargs) -> Dict:
    """Direct professional batch analysis"""
    system = get_professional_system()
    return system.analyze_batch(image_paths, **kwargs)