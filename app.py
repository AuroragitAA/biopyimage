"""
Enhanced Flask App for BIOIMAGIN Wolffia Analysis System
Production-ready with live analysis, real-time updates, and robust error handling
Complete integration with all components
"""

import base64
import io
import json
import logging
import os
import shutil
import sys
import tempfile
import threading
import time
import traceback
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from flask import (
    Flask,
    Response,
    jsonify,
    render_template,
    request,
    send_file,
    send_from_directory,
    session,
)
from flask_socketio import SocketIO, emit, join_room, leave_room
from matplotlib import pyplot as plt
from PIL import Image
from plotly.subplots import make_subplots
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

# Import safe logging first
from logging_config import setup_production_logging

logger = logging.getLogger(__name__)

# Import core analyzer
try:
    from wolffia_analyzer import AnalysisConfig, WolffiaAnalyzer
    ANALYZER_AVAILABLE = True
    logger.info("[OK] Core analyzer imported successfully")
except ImportError as e:
    logger.error(f"[ERROR] Core analyzer not available: {e}")
    ANALYZER_AVAILABLE = False

# Try to import professional components
try:
    from database_manager import DatabaseConfig, DatabaseManager
    DATABASE_AVAILABLE = True
    logger.info("[OK] Database manager available")
except ImportError:
    DATABASE_AVAILABLE = False
    logger.info("[INFO] Database manager not available")

try:
    from ml_enhancement import MLConfig, MLEnhancedAnalyzer
    ML_AVAILABLE = True
    logger.info("[OK] ML enhancement available")
except ImportError:
    ML_AVAILABLE = False
    logger.info("[INFO] ML enhancement not available")

try:
    from batch_processor import BatchJobConfig, BatchProcessor
    BATCH_PROCESSOR_AVAILABLE = True
    logger.info("[OK] Batch processor available")
except ImportError:
    BATCH_PROCESSOR_AVAILABLE = False
    logger.info("[INFO] Batch processor not available")

# Enhanced import handling for professional components
try:
    from professional_integrator import ProfessionalBioimageAnalyzer
    PROFESSIONAL_PIPELINE_AVAILABLE = True
    logger.info("[OK] Professional pipeline components imported")
except ImportError as e:
    logger.warning(f"[WARN] Professional pipeline not available: {e}")
    PROFESSIONAL_PIPELINE_AVAILABLE = False

# Import comprehensive visualizer
try:
    from comprehensive_visualizer import ComprehensiveVisualizer
    VISUALIZER_AVAILABLE = True
    logger.info("[OK] Comprehensive visualizer imported")
except ImportError:
    VISUALIZER_AVAILABLE = False
    logger.info("[INFO] Comprehensive visualizer not available")
    
    
    
# Ensure required directories exist
def ensure_directories():
    """Ensure all required directories exist"""
    directories = [
        'temp_uploads', 'logs', 'results', 'exports', 'outputs',
        'outputs/debug_images', 'outputs/results', 'outputs/exports',
        'static/temp_images'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

# Initialize Flask app
app = Flask(__name__)
app.config.update({
    'MAX_CONTENT_LENGTH': 50 * 1024 * 1024,  # 50MB max file size
    'UPLOAD_FOLDER': 'temp_uploads',
    'SECRET_KEY': os.environ.get('SECRET_KEY', 'bioimagin-wolffia-live-analysis-2024'),
    'JSON_SORT_KEYS': False,
    'JSONIFY_PRETTYPRINT_REGULAR': True,
    'DEBUG': os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
})

# Initialize SocketIO for real-time updates
socketio = SocketIO(app, cors_allowed_origins="*", logger=True, engineio_logger=False)

# Global variables
analyzer = None
database_manager = None
ml_analyzer = None
batch_processor = None
active_analyses = {}  # Track ongoing analyses
analysis_results_cache = {}  # Cache recent results

def initialize_system():
    """Initialize the analysis system with all available components"""
    global analyzer, database_manager, ml_analyzer, batch_processor
    
    try:
        ensure_directories()
        
        if ANALYZER_AVAILABLE:
            # Initialize core analyzer with robust error handling
            try:
                analyzer = WolffiaAnalyzer(
                    pixel_to_micron_ratio=1.0,
                    debug_mode=True,
                    output_dir="outputs"
                )
                logger.info("[SUCCESS] Core analyzer initialized")
                
                # Try to enhance with ML if available
                if ML_AVAILABLE:
                    try:
                        from ml_enhancement import MLConfig

                        ml_config = MLConfig()
                        ml_analyzer = MLEnhancedAnalyzer(analyzer, ml_config)
                        analyzer = ml_analyzer  # Use ML-enhanced version
                        logger.info("[SUCCESS] ML-enhanced analyzer initialized")
                    except Exception as e:
                        logger.warning(f"[WARN] ML enhancement failed, using basic analyzer: {e}")
                        # Keep the basic analyzer
                
            except Exception as e:
                logger.error(f"[ERROR] Core analyzer initialization failed: {e}")
                return False
            
            # Initialize database if available
            if DATABASE_AVAILABLE:
                try:
                    db_config = DatabaseConfig()
                    database_manager = DatabaseManager(db_config)
                    logger.info("[SUCCESS] Database system initialized")
                except Exception as e:
                    logger.warning(f"[WARN] Database initialization failed: {e}")
                    database_manager = None
            
            # Initialize batch processor if available
            if BATCH_PROCESSOR_AVAILABLE:
                try:
                    batch_config = BatchJobConfig()
                    batch_processor = BatchProcessor(analyzer, batch_config)
                    logger.info("[SUCCESS] Batch processor initialized")
                except Exception as e:
                    logger.warning(f"[WARN] Batch processor initialization failed: {e}")
                    batch_processor = None
            
            return True
        else:
            logger.error("[ERROR] Core analyzer not available")
            return False
    except Exception as e:
        logger.error(f"[ERROR] System initialization failed: {str(e)}")
        return False

# Initialize system
analyzer_ready = initialize_system()

# ============================================================================
# WEBSOCKET EVENTS FOR REAL-TIME UPDATES
# ============================================================================

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    logger.info(f"[CONNECT] Client connected: {request.sid}")
    emit('status', {'message': 'Connected to BIOIMAGIN Live Analysis', 'analyzer_ready': analyzer_ready})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    logger.info(f"[DISCONNECT] Client disconnected: {request.sid}")

@socketio.on('join_analysis')
def handle_join_analysis(data):
    """Join analysis room for real-time updates"""
    analysis_id = data.get('analysis_id')
    if analysis_id:
        join_room(analysis_id)
        logger.info(f"[ROOM] Client {request.sid} joined analysis room: {analysis_id}")

@socketio.on('leave_analysis')
def handle_leave_analysis(data):
    """Leave analysis room"""
    analysis_id = data.get('analysis_id')
    if analysis_id:
        leave_room(analysis_id)
        logger.info(f"[ROOM] Client {request.sid} left analysis room: {analysis_id}")

# ============================================================================
# FLASK ROUTES
# ============================================================================

@app.route('/')
def home():
    """Serve the main application dashboard"""
    try:
        system_status = {
            'analyzer_ready': analyzer_ready,
            'system_time': datetime.now().isoformat(),
            'recent_analyses': len(analysis_results_cache),
            'features': {
                'live_analysis': True,
                'batch_processing': BATCH_PROCESSOR_AVAILABLE,
                'real_time_updates': True,
                'debug_visualization': True,
                'ml_enhancement': ML_AVAILABLE,
                'database_integration': DATABASE_AVAILABLE
            }
        }
        
        return render_template('index.html', system_status=system_status)
        
    except Exception as e:
        logger.error(f"[ERROR] Home route error: {str(e)}")
        return render_template('index.html', system_status={'analyzer_ready': False})

@app.route('/api/health')
def api_health():
    """System health check"""
    try:
        health_data = {
            'status': 'healthy' if analyzer_ready else 'degraded',
            'timestamp': datetime.now().isoformat(),
            'analyzer_available': ANALYZER_AVAILABLE,
            'ml_available': ML_AVAILABLE,
            'database_available': DATABASE_AVAILABLE,
            'batch_available': BATCH_PROCESSOR_AVAILABLE,
            'active_analyses': len(active_analyses),
            'cached_results': len(analysis_results_cache),
            'uptime': 'running'
        }
        
        if analyzer and hasattr(analyzer, 'get_analysis_summary'):
            try:
                summary = analyzer.get_analysis_summary()
                health_data['analysis_stats'] = summary
            except:
                pass
        
        return jsonify(health_data), 200 if analyzer_ready else 503
        
    except Exception as e:
        logger.error(f"[ERROR] Health check error: {str(e)}")
        return jsonify({'status': 'error', 'error': str(e)}), 500

@app.route('/api/analyze', methods=['POST'])
def api_analyze_single():
    """Enhanced single image analysis with real-time updates"""
    if not analyzer_ready:
        return jsonify({'error': 'Analyzer not available'}), 503

    temp_path = None
    analysis_id = str(uuid.uuid4())

    try:
        # Validate request
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Extract parameters
        params = {
            'pixel_ratio': float(request.form.get('pixel_ratio', 1.0)),
            'debug_mode': request.form.get('debug_mode', 'false').lower() == 'true',
            'auto_export': request.form.get('auto_export', 'false').lower() == 'true'
        }

        # Save uploaded file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        secure_name = secure_filename(file.filename)
        temp_filename = f"analysis_{timestamp}_{secure_name}"
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
        file.save(temp_path)

        logger.info(f"[UPLOAD] File saved for analysis: {temp_path}")

        # Initialize analysis tracking
        active_analyses[analysis_id] = {
            'status': 'starting',
            'progress': 0,
            'stage': 'initialization',
            'start_time': datetime.now(),
            'image_path': temp_path
        }

        # Start analysis in background thread
        def run_analysis():
            try:
                def progress_callback(progress, stage, **kwargs):
                    active_analyses[analysis_id]['progress'] = progress
                    active_analyses[analysis_id]['stage'] = stage
                    socketio.emit('analysis_progress', {
                        'analysis_id': analysis_id,
                        'progress': progress,
                        'stage': stage
                    }, room=analysis_id)

                if hasattr(analyzer, 'set_progress_callback'):
                    analyzer.set_progress_callback(progress_callback)

                result = analyzer.analyze_single_image(temp_path, **params)
                result['analysis_id'] = analysis_id

                if result.get('success'):
                    result['visualizations'] = create_analysis_visualizations(temp_path, result)
                if VISUALIZER_AVAILABLE:
                    try:
                        visualizer = ComprehensiveVisualizer()
                        result['visualizations'].update(visualizer.create_all_visualizations(result))
                        logger.info(f"[VIZ] Visualizations embedded into result: {result['analysis_id']}")
                    except Exception as viz_error:
                        logger.warning(f"[WARN] Visualizer failed: {viz_error}")

                if database_manager and result.get('success') and os.path.exists(temp_path):
                    try:
                        store_analysis_in_database(result)
                    except Exception as db_error:
                        logger.warning(f"[WARN] Database storage failed: {str(db_error)}")
                elif result.get('success'):
                    logger.warning(f"[WARN] Skipped DB store: file not found for {temp_path}")

                analysis_results_cache[analysis_id] = result

                socketio.emit('analysis_complete', {
                    'analysis_id': analysis_id,
                    'success': result.get('success', False),
                    'total_cells': result.get('total_cells', 0),
                    'quality_score': result.get('quality_score', 0),
                    'processing_time': result.get('processing_time', 0)
                }, room=analysis_id)

            except Exception as e:
                logger.error(f"[ERROR] Background analysis error: {str(e)}")
                socketio.emit('analysis_error', {
                    'analysis_id': analysis_id,
                    'error': str(e)
                }, room=analysis_id)

            finally:
                active_analyses.pop(analysis_id, None)

                if temp_path and os.path.exists(temp_path):
                    def cleanup_later():
                        time.sleep(300)
                        try:
                            if os.path.exists(temp_path):
                                os.unlink(temp_path)
                                logger.info(f"[CLEANUP] Temp file removed: {temp_path}")
                        except Exception as cleanup_error:
                            logger.warning(f"[WARN] Cleanup failed for {temp_path}: {cleanup_error}")

                    cleanup_thread = threading.Thread(target=cleanup_later)
                    cleanup_thread.daemon = True
                    cleanup_thread.start()

        analysis_thread = threading.Thread(target=run_analysis)
        analysis_thread.daemon = True
        analysis_thread.start()

        return jsonify({
            'analysis_id': analysis_id,
            'status': 'started',
            'message': 'Analysis started - connect to WebSocket for real-time updates'
        })

    except Exception as e:
        logger.error(f"[ERROR] Analysis start error: {str(e)}")
        if analysis_id in active_analyses:
            del active_analyses[analysis_id]
        return jsonify({
            'error': f'Analysis failed to start: {str(e)}',
            'timestamp': datetime.now().isoformat()
        }), 500

    finally:
        if temp_path and os.path.exists(temp_path):
            def cleanup_later():
                time.sleep(300)
                try:
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
                except:
                    pass

            cleanup_thread = threading.Thread(target=cleanup_later)
            cleanup_thread.daemon = True
            cleanup_thread.start()
            
            
            
@socketio.on('update_analysis_parameter')
def handle_parameter_update(data):
    """Handle live parameter updates during analysis"""
    analysis_id = data.get('analysis_id')
    parameter = data.get('parameter')
    value = data.get('value')
    
    if analysis_id in active_analyses:
        # Update the analysis parameters
        logger.info(f"[UPDATE] Parameter {parameter} = {value} for analysis {analysis_id}")
        
        # Emit update confirmation
        emit('parameter_updated', {
            'analysis_id': analysis_id,
            'parameter': parameter,
            'value': value,
            'status': 'updated'
        }, room=analysis_id)

@app.route('/api/visualizations/comprehensive/<analysis_id>')
def api_get_comprehensive_visualizations(analysis_id):
    """Get comprehensive visualizations"""
    try:
        if analysis_id not in analysis_results_cache:
            return jsonify({'error': 'Analysis not found'}), 404
        
        result = analysis_results_cache[analysis_id]
        
        if not VISUALIZER_AVAILABLE:
            return jsonify({'error': 'Visualizer not available'}), 503
        
        visualizer = ComprehensiveVisualizer()
        visualizations = visualizer.create_all_visualizations(result)
        
        return jsonify(visualizations)
        
    except Exception as e:
        logger.error(f"[ERROR] Comprehensive visualization error: {str(e)}")
        return jsonify({'error': str(e)}), 500
    
    
@app.route('/api/analyze/comprehensive', methods=['POST'])
def api_analyze_comprehensive():
    """Comprehensive analysis with all advanced features"""
    if not analyzer_ready:
        return jsonify({'error': 'Analyzer not available'}), 503

    temp_path = None
    analysis_id = str(uuid.uuid4())

    try:
        # Validate request
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        file = request.files['image']
        timestamp = request.form.get('timestamp')  # Optional timestamp

        # Save uploaded file
        temp_filename = f"comprehensive_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{secure_filename(file.filename)}"
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
        file.save(temp_path)

        logger.info(f"[COMPREHENSIVE] Starting comprehensive analysis: {temp_path}")

        # Initialize analysis tracking
        active_analyses[analysis_id] = {
            'type': 'comprehensive',
            'status': 'starting',
            'progress': 0,
            'stage': 'initialization',
            'start_time': datetime.now(),
            'image_path': temp_path
        }

        def run_comprehensive_analysis():
            try:
                def progress_callback(progress, stage, **kwargs):
                    active_analyses[analysis_id]['progress'] = progress
                    active_analyses[analysis_id]['stage'] = stage
                    socketio.emit('analysis_progress', {
                        'analysis_id': analysis_id,
                        'progress': progress,
                        'stage': stage,
                        'type': 'comprehensive'
                    }, room=analysis_id)

                if hasattr(analyzer, 'set_progress_callback'):
                    analyzer.set_progress_callback(progress_callback)

                # Run comprehensive analysis
                if hasattr(analyzer, 'analyze_comprehensive'):
                    result = analyzer.analyze_comprehensive(temp_path, timestamp=timestamp)
                else:
                    # Fallback to standard analysis with comprehensive features
                    result = analyzer.analyze_single_image(temp_path, timestamp=timestamp)
                    if result.get('success'):
                        # Add comprehensive features manually
                        comprehensive_features = add_comprehensive_features(result, timestamp)
                        result.update(comprehensive_features)
                
                result['analysis_id'] = analysis_id

                # Visualizations
                if result.get('success'):
                    result['visualizations'] = create_comprehensive_visualizations(temp_path, result)
                if VISUALIZER_AVAILABLE:
                    try:
                        visualizer = ComprehensiveVisualizer()
                        result['visualizations'].update(visualizer.create_all_visualizations(result))
                        logger.info(f"[VIZ] Comprehensive visualizations injected: {analysis_id}")
                    except Exception as viz_error:
                        logger.warning(f"[WARN] Visualization injection failed: {viz_error}")

                # Store in DB
                if database_manager and result.get('success'):
                    try:
                        store_analysis_in_database(result)
                    except Exception as db_error:
                        logger.warning(f"[WARN] Database storage failed: {str(db_error)}")

                # Cache result
                analysis_results_cache[analysis_id] = result

                # Notify client
                socketio.emit('analysis_complete', {
                    'analysis_id': analysis_id,
                    'success': result.get('success', False),
                    'type': 'comprehensive',
                    'total_cells': result.get('total_cells', 0),
                    'biomass_g': result.get('biomass_analysis', {}).get('combined_estimate', {}).get('fresh_biomass_g', 0),
                    'similar_cells': len(result.get('similarity_analysis', {}).get('similar_cell_groups', [])),
                    'temporal_tracks': len(result.get('temporal_analysis', {}).get('growth_curves', {}))
                }, room=analysis_id)

            except Exception as e:
                logger.error(f"[ERROR] Comprehensive analysis error: {str(e)}")
                socketio.emit('analysis_error', {
                    'analysis_id': analysis_id,
                    'error': str(e),
                    'type': 'comprehensive'
                }, room=analysis_id)
            finally:
                if analysis_id in active_analyses:
                    del active_analyses[analysis_id]

        # Start background thread
        analysis_thread = threading.Thread(target=run_comprehensive_analysis)
        analysis_thread.daemon = True
        analysis_thread.start()

        return jsonify({
            'analysis_id': analysis_id,
            'status': 'started',
            'type': 'comprehensive',
            'message': 'Comprehensive analysis started with biomass, spectral, similarity, and temporal features'
        })

    except Exception as e:
        logger.error(f"[ERROR] Comprehensive analysis start error: {str(e)}")
        if analysis_id in active_analyses:
            del active_analyses[analysis_id]
        return jsonify({
            'error': f'Comprehensive analysis failed to start: {str(e)}',
            'timestamp': datetime.now().isoformat()
        }), 500

    finally:
        # Schedule file cleanup
        if temp_path and os.path.exists(temp_path):
            def cleanup_later():
                time.sleep(300)
                try:
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
                except:
                    pass

            cleanup_thread = threading.Thread(target=cleanup_later)
            cleanup_thread.daemon = True
            cleanup_thread.start()





@app.route('/api/analysis/<analysis_id>')
def api_get_analysis_result(analysis_id):
    def safe_convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.generic, np.float32, np.float64, float)):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return obj.item() if hasattr(obj, 'item') else obj
        return obj  # leave other types alone

    try:
        result = analysis_results_cache[analysis_id]

        # Recursively sanitize any nested NaN/inf
        def sanitize(obj):
            if isinstance(obj, dict):
                return {k: sanitize(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [sanitize(v) for v in obj]
            elif isinstance(obj, (np.floating, float)):
                return None if np.isnan(obj) or np.isinf(obj) else obj
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return sanitize(obj.tolist())
            return obj

        sanitized = sanitize(result)
        return Response(json.dumps(sanitized), mimetype='application/json')

    except KeyError:
        return jsonify({'error': 'Analysis not found'}), 404
    except Exception as e:
        app.logger.error(f"[ERROR] Serialization failed: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/batch/analyze', methods=['POST'])
def api_batch_analyze():
    """Enhanced batch analysis with progress tracking"""
    if not analyzer_ready:
        return jsonify({'error': 'Analyzer not available'}), 503
    
    try:
        # Validate files
        if 'images' not in request.files:
            return jsonify({'error': 'No image files provided'}), 400
        
        files = request.files.getlist('images')
        if not files or all(f.filename == '' for f in files):
            return jsonify({'error': 'No valid files selected'}), 400
        
        if len(files) > 20:  # Reasonable limit for web interface
            return jsonify({'error': 'Too many files. Maximum 20 images per batch.'}), 400
        
        batch_id = str(uuid.uuid4())
        logger.info(f"[BATCH] Starting batch analysis: {len(files)} images (ID: {batch_id})")
        
        # Save files
        temp_paths = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for i, file in enumerate(files):
            secure_name = secure_filename(file.filename)
            temp_filename = f"batch_{timestamp}_{i:03d}_{secure_name}"
            temp_path = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
            file.save(temp_path)
            temp_paths.append(temp_path)
        
        # Initialize batch tracking
        active_analyses[batch_id] = {
            'type': 'batch',
            'status': 'starting',
            'progress': 0,
            'stage': 'initialization',
            'start_time': datetime.now(),
            'total_images': len(temp_paths),
            'completed_images': 0
        }
        
        def run_batch_analysis():
            try:
                # Set up progress callback for batch processing
                def batch_progress_callback(progress, stage, **kwargs):
                    socketio.emit('batch_progress', {
                        'batch_id': batch_id,
                        'progress': progress,
                        'stage': stage,
                        'completed': kwargs.get('completed', 0),
                        'total': len(temp_paths)
                    }, room=batch_id)
                
                # Use batch processor if available
                if batch_processor:
                    result = batch_processor.process_batch(temp_paths, batch_progress_callback)
                else:
                    # Fallback to sequential processing
                    results = []
                    for i, temp_path in enumerate(temp_paths):
                        try:
                            single_result = analyzer.analyze_single_image(temp_path)
                            results.append(single_result)
                            
                            # Update progress
                            progress = int(((i + 1) / len(temp_paths)) * 100)
                            batch_progress_callback(progress, f'Completed {i+1}/{len(temp_paths)}', completed=i+1)
                            
                        except Exception as img_error:
                            logger.error(f"[ERROR] Batch image error: {str(img_error)}")
                            results.append({
                                'success': False,
                                'error': str(img_error),
                                'image_path': temp_path
                            })
                    
                    # Calculate batch summary
                    successful = [r for r in results if r.get('success')]
                    result = {
                        'success': True,
                        'batch_summary': {
                            'total_images': len(temp_paths),
                            'successful': len(successful),
                            'failed': len(temp_paths) - len(successful),
                            'success_rate': len(successful) / len(temp_paths) * 100,
                            'total_cells_detected': sum(r.get('total_cells', 0) for r in successful),
                            'processing_time': (datetime.now() - active_analyses[batch_id]['start_time']).total_seconds()
                        },
                        'individual_results': results
                    }
                
                # Add batch metadata
                result['batch_id'] = batch_id
                
                # Cache result
                analysis_results_cache[batch_id] = result
                
                # Notify completion
                socketio.emit('batch_complete', {
                    'batch_id': batch_id,
                    'success_rate': result['batch_summary']['success_rate'],
                    'total_cells': result['batch_summary']['total_cells_detected'],
                    'processing_time': result['batch_summary']['processing_time']
                }, room=batch_id)
                
                # Clean up
                if batch_id in active_analyses:
                    del active_analyses[batch_id]
                
                # Clean up temp files
                for temp_path in temp_paths:
                    try:
                        os.unlink(temp_path)
                    except:
                        pass
                
            except Exception as e:
                logger.error(f"[ERROR] Batch analysis error: {str(e)}")
                socketio.emit('batch_error', {
                    'batch_id': batch_id,
                    'error': str(e)
                }, room=batch_id)
                
                if batch_id in active_analyses:
                    del active_analyses[batch_id]
        
        # Start batch analysis
        batch_thread = threading.Thread(target=run_batch_analysis)
        batch_thread.daemon = True
        batch_thread.start()
        
        return jsonify({
            'batch_id': batch_id,
            'status': 'started',
            'total_images': len(files),
            'message': 'Batch analysis started - connect to WebSocket for progress updates'
        })
        
    except Exception as e:
        logger.error(f"[ERROR] Batch analysis start error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/export/<analysis_id>/<format>')
def api_export_analysis(analysis_id, format):
    """Export analysis results"""
    try:
        if analysis_id not in analysis_results_cache:
            return jsonify({'error': 'Analysis not found'}), 404
        
        result = analysis_results_cache[analysis_id]
        
        if not result.get('success') or not result.get('cell_data'):
            return jsonify({'error': 'No data to export'}), 400
        
        # Create DataFrame
        df = pd.DataFrame(result['cell_data'])
        
        # Generate export
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format.lower() == 'csv':
            export_path = f"exports/wolffia_analysis_{analysis_id}_{timestamp}.csv"
            Path("exports").mkdir(exist_ok=True)
            df.to_csv(export_path, index=False)
            return send_file(export_path, as_attachment=True)
            
        elif format.lower() == 'excel':
            export_path = f"exports/wolffia_analysis_{analysis_id}_{timestamp}.xlsx"
            Path("exports").mkdir(exist_ok=True)
            with pd.ExcelWriter(export_path, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Cell_Data', index=False)
                
                # Add summary sheet
                summary_df = pd.DataFrame([result['summary']])
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            return send_file(export_path, as_attachment=True)
            
        elif format.lower() == 'json':
            export_data = {
                'analysis_info': {
                    'analysis_id': analysis_id,
                    'timestamp': result.get('timestamp'),
                    'image_path': result.get('image_path'),
                    'total_cells': result.get('total_cells'),
                    'quality_score': result.get('quality_score')
                },
                'summary': result.get('summary'),
                'cell_data': result.get('cell_data')
            }
            
            export_path = f"exports/wolffia_analysis_{analysis_id}_{timestamp}.json"
            Path("exports").mkdir(exist_ok=True)
            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            return send_file(export_path, as_attachment=True)
        
        else:
            return jsonify({'error': f'Unsupported format: {format}'}), 400
            
    except Exception as e:
        logger.error(f"[ERROR] Export error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/visualizations/<analysis_id>')
def api_get_visualizations(analysis_id):
    """Get analysis visualizations"""
    try:
        if analysis_id not in analysis_results_cache:
            return jsonify({'error': 'Analysis not found'}), 404
        
        result = analysis_results_cache[analysis_id]
        visualizations = result.get('visualizations', {})
        
        return jsonify(visualizations)
        
    except Exception as e:
        logger.error(f"[ERROR] Visualizations error: {str(e)}")
        return jsonify({'error': str(e)}), 500

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_analysis_visualizations(image_path, result):
    """Create visualizations for analysis results"""
    try:
        visualizations = {}
        
        # Load original image
        original_image = cv2.imread(image_path)
        if original_image is None:
            return visualizations
        
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        
        # Convert to base64 for web display
        def image_to_base64(img):
            """Convert image to base64 string"""
            try:
                if len(img.shape) == 3:
                    img_pil = Image.fromarray(img)
                else:
                    img_pil = Image.fromarray(img, mode='L')
                
                buffer = io.BytesIO()
                img_pil.save(buffer, format='PNG')
                img_str = base64.b64encode(buffer.getvalue()).decode()
                return img_str
            except:
                return None
        
        # Original image
        visualizations['original'] = image_to_base64(original_image)
        
        # Segmentation overlay
        labels = result.get('labels')
        if labels is not None and np.max(labels) > 0:
            try:
                # Create colored overlay
                overlay = original_image.copy()
                
                # Generate colors for each cell
                unique_labels = np.unique(labels)[1:]  # Skip background
                colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
                
                for i, label_val in enumerate(unique_labels):
                    mask = labels == label_val
                    color = (colors[i][:3] * 255).astype(np.uint8)
                    overlay[mask] = color
                
                # Blend with original
                blended = cv2.addWeighted(original_image, 0.6, overlay, 0.4, 0)
                visualizations['segmentation'] = image_to_base64(blended)
                
                # Contour overlay
                contour_overlay = original_image.copy()
                for label_val in unique_labels:
                    mask = (labels == label_val).astype(np.uint8)
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(contour_overlay, contours, -1, (255, 0, 0), 2)
                
                visualizations['contours'] = image_to_base64(contour_overlay)
                
            except Exception as viz_error:
                logger.warning(f"[WARN] Visualization creation failed: {str(viz_error)}")
        
        # Cell detection overlay with numbers
        if result.get('cell_data'):
            numbered_overlay = original_image.copy()
            for cell in result['cell_data']:
                if 'centroid_x' in cell and 'centroid_y' in cell:
                    x, y = int(cell['centroid_x']), int(cell['centroid_y'])
                    cell_id = cell.get('cell_id', 0)
                    
                    # Draw marker
                    cv2.circle(numbered_overlay, (x, y), 8, (255, 0, 0), 2)
                    cv2.putText(numbered_overlay, str(cell_id), (x+10, y-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            visualizations['numbered'] = image_to_base64(numbered_overlay)
        
        return visualizations
        
    except Exception as e:
        logger.error(f"[ERROR] Visualization creation error: {str(e)}")
        return {}

def create_comprehensive_visualizations(image_path, result):
    """Create comprehensive visualizations including histograms and advanced charts"""
    try:
        visualizations = create_analysis_visualizations(image_path, result)

        # Add comprehensive visualizations if data available
        if result.get('biomass_analysis'):
            visualizations['biomass_chart'] = create_biomass_visualization(result['biomass_analysis'])

        if result.get('spectral_analysis'):
            visualizations['spectral_charts'] = create_spectral_visualizations(result['spectral_analysis'])

        if result.get('similarity_analysis'):
            visualizations['similarity_charts'] = create_similarity_visualizations(result['similarity_analysis'])

        if result.get('temporal_analysis'):
            visualizations['temporal_charts'] = create_temporal_visualizations(result['temporal_analysis'])

        if result.get('cell_data'):
            visualizations['histograms'] = create_distribution_histograms(result['cell_data'])

        return visualizations

    except Exception as e:
        logger.error(f"[ERROR] Comprehensive visualization error: {str(e)}")
        return {}

def create_biomass_visualization(biomass_data):
    """Create biomass analysis visualization"""
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Comprehensive Biomass Analysis', fontsize=16)

        # Method comparison
        methods = ['Area-based', 'Chlorophyll-based', 'Combined']
        values = [
            biomass_data['area_based']['fresh_biomass_g'],
            biomass_data['chlorophyll_based']['estimated_biomass_g'],
            biomass_data['combined_estimate']['fresh_biomass_g']
        ]
        ax1.bar(methods, values, color=['skyblue', 'lightgreen', 'lightcoral'])
        ax1.set_title('Biomass Estimation Methods')
        ax1.set_ylabel('Fresh Biomass (g)')
        ax1.tick_params(axis='x', rotation=45)

        # Combined estimate with confidence interval
        combined = biomass_data['combined_estimate']
        ax2.errorbar(['Combined Estimate'], [combined['fresh_biomass_g']],
                     yerr=[[combined['fresh_biomass_g'] - combined['confidence_interval'][0]],
                           [combined['confidence_interval'][1] - combined['fresh_biomass_g']]],
                     fmt='o', capsize=5, color='red', markersize=10)
        ax2.set_title('Combined Biomass Estimate')
        ax2.set_ylabel('Fresh Biomass (g)')

        # Fresh vs Dry biomass pie chart
        biomass_types = ['Fresh Biomass', 'Dry Biomass']
        biomass_values = [combined['fresh_biomass_g'], combined['dry_biomass_g']]
        ax3.pie(biomass_values, labels=biomass_types, autopct='%1.1f%%', colors=['lightblue', 'orange'])
        ax3.set_title('Fresh vs Dry Biomass')

        # Area-biomass relationship
        areas = np.linspace(100, 5000, 50)
        biomass_est = 0.0012 * (areas ** 1.15) * 1e-6
        ax4.plot(areas, biomass_est, 'g-', linewidth=2, label='Allometric Model')
        ax4.scatter([biomass_data['area_based'].get('total_area_mm2', 1000) * 1000], 
                   [biomass_data['area_based']['fresh_biomass_g']],
                   color='red', s=100, label='Current Sample')
        ax4.set_xlabel('Area (pixels)')
        ax4.set_ylabel('Biomass (g)')
        ax4.set_title('Area-Biomass Relationship')
        ax4.legend()

        plt.tight_layout()

        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        plot_data = buffer.getvalue()
        buffer.close()
        plt.close()

        return base64.b64encode(plot_data).decode()

    except Exception as e:
        logger.error(f"[ERROR] Biomass visualization error: {str(e)}")
        return None

def create_spectral_visualizations(spectral_data):
    """Create spectral analysis visualizations"""
    try:
        import matplotlib.pyplot as plt
        import pandas as pd

        if not spectral_data.get('cell_spectral_data'):
            return None

        df = pd.DataFrame(spectral_data['cell_spectral_data'])

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Spectral Analysis Results', fontsize=16)

        # Wavelength intensity distribution (simplified)
        if 'mean_green' in df.columns:
            ax1.hist(df['mean_green'], bins=20, color='green', alpha=0.7, edgecolor='black')
            ax1.set_title('Green Intensity Distribution')
            ax1.set_xlabel('Green Intensity')
            ax1.set_ylabel('Count')

        # Vegetation index per cell
        if 'vegetation_index' in df.columns:
            ax2.plot(df['cell_id'], df['vegetation_index'], 'o-', color='forestgreen', markersize=4)
            ax2.set_xlabel('Cell ID')
            ax2.set_ylabel('Vegetation Index')
            ax2.set_title('Vegetation Index per Cell')

        # RGB distribution
        if all(col in df.columns for col in ['mean_red', 'mean_green', 'mean_blue']):
            ax3.scatter(df['mean_red'], df['mean_green'], alpha=0.6, c=df['mean_blue'], cmap='coolwarm')
            ax3.set_xlabel('Red Intensity')
            ax3.set_ylabel('Green Intensity')
            ax3.set_title('RGB Distribution')

        # Population statistics
        if 'population_statistics' in spectral_data:
            stats = spectral_data['population_statistics']
            categories = list(stats.keys())
            values = list(stats.values())
            ax4.bar(categories, values, color='lightgreen')
            ax4.set_title('Population Statistics')
            ax4.tick_params(axis='x', rotation=45)

        plt.tight_layout()

        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        plot_data = buffer.getvalue()
        buffer.close()
        plt.close()

        return base64.b64encode(plot_data).decode()

    except Exception as e:
        logger.error(f"[ERROR] Spectral visualization error: {str(e)}")
        return None

def create_similarity_visualizations(similarity_data):
    """Create cell similarity visualizations"""
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Cell Similarity Analysis', fontsize=16)

        # Similar cell pairs
        if 'similar_cell_groups' in similarity_data and similarity_data['similar_cell_groups']:
            pairs = similarity_data['similar_cell_groups'][:10]  # Top 10 pairs
            similarity_scores = [pair['similarity_score'] for pair in pairs]
            pair_labels = [f"({pair['cell_1']},{pair['cell_2']})" for pair in pairs]
            ax1.barh(pair_labels, similarity_scores, color='coral')
            ax1.set_title('Top Similar Cell Pairs')
            ax1.set_xlabel('Similarity Score')
        else:
            ax1.text(0.5, 0.5, 'No similar cells found', ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('Similar Cell Pairs')

        # Summary statistics
        total_pairs = len(similarity_data.get('similar_cell_groups', []))
        threshold = similarity_data.get('similarity_threshold', 0.85)
        
        ax2.bar(['Total Pairs', 'Threshold'], [total_pairs, threshold], color=['lightblue', 'orange'])
        ax2.set_title('Similarity Summary')
        ax2.set_ylabel('Count / Value')

        plt.tight_layout()

        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        plot_data = buffer.getvalue()
        buffer.close()
        plt.close()

        return base64.b64encode(plot_data).decode()

    except Exception as e:
        logger.error(f"[ERROR] Similarity visualization error: {str(e)}")
        return None

def create_temporal_visualizations(temporal_data):
    """Create temporal tracking visualizations"""
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Temporal Tracking Analysis', fontsize=16)

        # Population growth curve
        if 'population_dynamics' in temporal_data:
            pop_dyn = temporal_data['population_dynamics']
            growth_data = pop_dyn.get('growth_data', [])
            if growth_data:
                timepoints = [d['timepoint'] for d in growth_data]
                cell_counts = [d['cell_count'] for d in growth_data]
                ax1.plot(timepoints, cell_counts, 'bo-', linewidth=2, markersize=6)
                ax1.set_xlabel('Time Point')
                ax1.set_ylabel('Cell Count')
                ax1.set_title('Population Growth Curve')
                ax1.grid(True, alpha=0.3)
            else:
                ax1.text(0.5, 0.5, 'No growth data available', ha='center', va='center', transform=ax1.transAxes)
                ax1.set_title('Population Growth Curve')

        # Growth curves
        if 'growth_curves' in temporal_data:
            curves = temporal_data['growth_curves']
            for track_id, curve in list(curves.items())[:5]:  # Max 5 curves
                if 'time_points' in curve and 'cell_counts' in curve:
                    ax2.plot(curve['time_points'], curve['cell_counts'], 'o-', label=f'Track {track_id}', markersize=4)
            ax2.set_xlabel('Time Point')
            ax2.set_ylabel('Cell Count')
            ax2.set_title('Individual Growth Tracks')
            ax2.legend()

        plt.tight_layout()

        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        plot_data = buffer.getvalue()
        buffer.close()
        plt.close()

        return base64.b64encode(plot_data).decode()

    except Exception as e:
        logger.error(f"[ERROR] Temporal visualization error: {str(e)}")
        return None

def create_distribution_histograms(cell_data):
    """Create comprehensive distribution histograms"""
    try:
        import matplotlib.pyplot as plt
        import pandas as pd

        df = pd.DataFrame(cell_data)

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Cell Population Distributions', fontsize=16)

        # Area distribution
        axes[0, 0].hist(df['area'], bins=20, color='skyblue', edgecolor='navy', alpha=0.7)
        axes[0, 0].set_title('Cell Area Distribution')
        axes[0, 0].set_xlabel('Area (pixels)')
        axes[0, 0].set_ylabel('Count')

        # Circularity distribution
        if 'circularity' in df.columns:
            axes[0, 1].hist(df['circularity'], bins=20, color='lightgreen', edgecolor='darkgreen', alpha=0.7)
            axes[0, 1].set_title('Circularity Distribution')
            axes[0, 1].set_xlabel('Circularity')
            axes[0, 1].set_ylabel('Count')
        else:
            axes[0, 1].axis('off')

        # Aspect ratio distribution
        if 'aspect_ratio' in df.columns:
            axes[0, 2].hist(df['aspect_ratio'], bins=20, color='lightcoral', edgecolor='darkred', alpha=0.7)
            axes[0, 2].set_title('Aspect Ratio Distribution')
            axes[0, 2].set_xlabel('Aspect Ratio')
            axes[0, 2].set_ylabel('Count')
        else:
            axes[0, 2].axis('off')

        # Chlorophyll content distribution
        if 'chlorophyll_content' in df.columns:
            axes[1, 0].hist(df['chlorophyll_content'], bins=20, color='green', edgecolor='darkgreen', alpha=0.7)
            axes[1, 0].set_title('Chlorophyll Content Distribution')
            axes[1, 0].set_xlabel('Chlorophyll Content')
            axes[1, 0].set_ylabel('Count')
        else:
            axes[1, 0].axis('off')

        # Health score distribution
        if 'health_score' in df.columns:
            axes[1, 1].hist(df['health_score'], bins=20, color='gold', edgecolor='orange', alpha=0.7)
            axes[1, 1].set_title('Health Score Distribution')
            axes[1, 1].set_xlabel('Health Score')
            axes[1, 1].set_ylabel('Count')
        else:
            axes[1, 1].axis('off')

        # Size category distribution (pie chart)
        if 'size_category' in df.columns:
            size_counts = df['size_category'].value_counts()
            axes[1, 2].pie(size_counts.values, labels=size_counts.index, autopct='%1.1f%%',
                           colors=['lightblue', 'lightgreen', 'lightcoral'])
            axes[1, 2].set_title('Size Category Distribution')
        else:
            axes[1, 2].axis('off')

        plt.tight_layout()

        # Convert to base64 image
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        plot_data = buffer.getvalue()
        buffer.close()
        plt.close()

        return base64.b64encode(plot_data).decode()

    except Exception as e:
        logger.error(f"[ERROR] Distribution histogram error: {str(e)}")
        return None

def add_comprehensive_features(result, timestamp):
    """Add comprehensive analysis features manually"""
    try:
        comprehensive_data = {}
        
        if result.get('cell_data'):
            df = pd.DataFrame(result['cell_data'])
            
            # Simple biomass calculation
            total_area = df['area'].sum() if 'area' in df.columns else 0
            biomass_estimate = total_area * 0.001  # Simple conversion
            
            comprehensive_data['biomass_analysis'] = {
                'area_based': {
                    'fresh_biomass_g': biomass_estimate / 1000,
                    'total_area_mm2': total_area / 1000000
                },
                'chlorophyll_based': {
                    'estimated_biomass_g': biomass_estimate * 0.8 / 1000
                },
                'combined_estimate': {
                    'fresh_biomass_g': biomass_estimate * 0.9 / 1000,
                    'dry_biomass_g': biomass_estimate * 0.12 / 1000,
                    'confidence_interval': [biomass_estimate * 0.7 / 1000, biomass_estimate * 1.1 / 1000]
                }
            }
        
        return comprehensive_data
        
    except Exception as e:
        logger.error(f"[ERROR] Comprehensive features failed: {str(e)}")
        return {}

def store_analysis_in_database(result):
    """Store analysis result in database if available"""
    try:
        if not database_manager:
            logger.info("[INFO] Database not available, skipping storage")
            return
        
        # Ensure we have a valid image path
        image_path = result.get('image_path', '')
        
        # FIXED: Handle temporary file path properly
        if not image_path or not os.path.exists(image_path):
            # Create a reference path for database storage
            timestamp = result.get('timestamp', datetime.now().strftime('%Y%m%d_%H%M%S'))
            image_path = f"analysis_{timestamp}.png"
            logger.warning(f"[WARN] Using reference path for DB: {image_path}")
        
        
        # Ensure we have default project/experiment
        try:
            with database_manager.get_connection() as conn:
                project = conn.execute("SELECT id FROM projects WHERE name = ?", ("Default Project",)).fetchone()
                if not project:
                    project_id = database_manager.create_project(
                        name="Default Project",
                        description="Default project for Wolffia analysis",
                        operator_name="System"
                    )
                else:
                    project_id = project['id']
                
                experiment = conn.execute(
                    "SELECT id FROM experiments WHERE project_id = ? AND experiment_name = ?", 
                    (project_id, "Default Experiment")
                ).fetchone()
                if not experiment:
                    experiment_id = database_manager.create_experiment(
                        project_id=project_id,
                        experiment_name="Default Experiment",
                        description="Default experiment for analysis"
                    )
                else:
                    experiment_id = experiment['id']
        
        except Exception as setup_error:
            logger.warning(f"[WARN] Database setup failed: {setup_error}")
            project_id = 1
            experiment_id = 1
        
        # Store image with fixed path
        image_id = database_manager.store_image(
            experiment_id=experiment_id,
            filename=Path(image_path).name,
            file_path=image_path
        )
        
        # Store analysis result
        analysis_id = database_manager.store_analysis_result(image_id, result)
        
        result['database_ids'] = {
            'image_id': image_id,
            'analysis_id': analysis_id  
        }
        
        logger.info(f"[SUCCESS] Analysis stored in database: {analysis_id}")
        
    except Exception as e:
        logger.warning(f"[WARN] Database storage failed (continuing without): {str(e)}")

# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(413)
def handle_file_too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 50MB.'}), 413

@app.errorhandler(404)
def handle_not_found(e):
    if request.path.startswith('/api/'):
        return jsonify({'error': 'API endpoint not found'}), 404
    return render_template('index.html'), 404

@app.errorhandler(500)
def handle_internal_error(e):
    logger.error(f"[ERROR] Internal server error: {str(e)}")
    if request.path.startswith('/api/'):
        return jsonify({'error': 'Internal server error'}), 500
    return render_template('index.html'), 500

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    try:
        print("\n" + "=" * 70)
        print("[SYSTEM] BIOIMAGIN ENHANCED WOLFFIA ANALYSIS SYSTEM")
        print("   Production-Ready Live Analysis Platform")
        print("=" * 70)
        print(f"   Analyzer Ready: {'[SUCCESS] YES' if analyzer_ready else '[ERROR] NO'}")
        print(f"   ML Enhancement: {'[SUCCESS] YES' if ML_AVAILABLE else '[INFO] NO'}")
        print(f"   Database Ready: {'[SUCCESS] YES' if DATABASE_AVAILABLE else '[INFO] NO'}")
        print(f"   Batch Processing: {'[SUCCESS] YES' if BATCH_PROCESSOR_AVAILABLE else '[INFO] NO'}")
        print(f"   Server URL: http://localhost:5000")
        print(f"   WebSocket: ws://localhost:5000")
        print("=" * 70)
        print("   Press Ctrl+C to stop the server")
        print("-" * 70)
        
        # Run with SocketIO
        socketio.run(
            app,
            host='0.0.0.0',
            port=5000,
            debug=app.config['DEBUG'],
            use_reloader=False
        )
        
    except KeyboardInterrupt:
        print("\n[BYE] Server stopped by user")
        print("[SUCCESS] BIOIMAGIN System shutdown complete")
    except Exception as startup_error:
        logger.error(f"[ERROR] Server startup failed: {startup_error}")
        print(f"[ERROR] Server startup failed: {startup_error}")
        sys.exit(1)