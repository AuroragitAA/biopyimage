"""
Enhanced Flask App for BIOIMAGIN Wolffia Analysis System
Production-ready with live analysis, real-time updates, and robust error handling
Fixed integration issues and standardized interfaces
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
from flask import (
    Flask,
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
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

# Import enhanced analyzer with proper error handling
try:
    from wolffia_analyzer import WolffiaAnalyzer
    ANALYZER_AVAILABLE = True
    print("✅ Core analyzer imported successfully")
except ImportError as e:
    print(f"❌ Core analyzer not available: {e}")
    ANALYZER_AVAILABLE = False

# Try to import professional components
try:
    from database_manager import DatabaseConfig, DatabaseManager
    DATABASE_AVAILABLE = True
    print("✅ Database manager available")
except ImportError:
    DATABASE_AVAILABLE = False
    print("⚠️ Database manager not available")

try:
    from ml_enhancement import MLConfig, MLEnhancedAnalyzer
    ML_AVAILABLE = True
    print("✅ ML enhancement available")
except ImportError:
    ML_AVAILABLE = False
    print("⚠️ ML enhancement not available")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/bioimagin_app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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
active_analyses = {}  # Track ongoing analyses
analysis_results_cache = {}  # Cache recent results

def ensure_directories():
    """Ensure all required directories exist"""
    directories = [
        'temp_uploads', 'logs', 'results', 'exports', 'outputs',
        'outputs/debug_images', 'outputs/results', 'outputs/exports',
        'static/temp_images'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

def initialize_system():
    """Initialize the analysis system with all available components"""
    global analyzer, database_manager
    
    try:
        if ANALYZER_AVAILABLE:
            # Initialize core analyzer with robust error handling
            try:
                analyzer = WolffiaAnalyzer(
                    pixel_to_micron_ratio=1.0,
                    debug_mode=True,
                    output_dir="outputs"
                )
                logger.info("✅ Core analyzer initialized")
                
                # Try to enhance with ML if available
                if ML_AVAILABLE:
                    try:
                        ml_config = MLConfig()
                        # Test if ML analyzer can be created
                        ml_analyzer = MLEnhancedAnalyzer(analyzer, ml_config)
                        analyzer = ml_analyzer
                        logger.info("✅ ML-enhanced analyzer initialized")
                    except Exception as e:
                        logger.warning(f"⚠️ ML enhancement failed, using basic analyzer: {e}")
                        # Keep the basic analyzer
                
            except Exception as e:
                logger.error(f"❌ Core analyzer initialization failed: {e}")
                return False
            
            # Initialize database if available
            if DATABASE_AVAILABLE:
                try:
                    db_config = DatabaseConfig()
                    database_manager = DatabaseManager(db_config)
                    logger.info("✅ Database system initialized")
                except Exception as e:
                    logger.warning(f"⚠️ Database initialization failed: {e}")
                    database_manager = None
            
            return True
        else:
            logger.error("❌ Core analyzer not available")
            return False
    except Exception as e:
        logger.error(f"❌ System initialization failed: {str(e)}")
        return False

# Initialize system
ensure_directories()
analyzer_ready = initialize_system()

# ============================================================================
# WEBSOCKET EVENTS FOR REAL-TIME UPDATES
# ============================================================================

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    logger.info(f"🔗 Client connected: {request.sid}")
    emit('status', {'message': 'Connected to BIOIMAGIN Live Analysis', 'analyzer_ready': analyzer_ready})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    logger.info(f"🔌 Client disconnected: {request.sid}")

@socketio.on('join_analysis')
def handle_join_analysis(data):
    """Join analysis room for real-time updates"""
    analysis_id = data.get('analysis_id')
    if analysis_id:
        join_room(analysis_id)
        logger.info(f"👥 Client {request.sid} joined analysis room: {analysis_id}")

@socketio.on('leave_analysis')
def handle_leave_analysis(data):
    """Leave analysis room"""
    analysis_id = data.get('analysis_id')
    if analysis_id:
        leave_room(analysis_id)
        logger.info(f"👤 Client {request.sid} left analysis room: {analysis_id}")

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
                'batch_processing': True,
                'real_time_updates': True,
                'debug_visualization': True,
                'ml_enhancement': ML_AVAILABLE,
                'database_integration': DATABASE_AVAILABLE
            }
        }
        
        return render_template('index.html', system_status=system_status)
        
    except Exception as e:
        logger.error(f"❌ Home route error: {str(e)}")
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
        logger.error(f"❌ Health check error: {str(e)}")
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
        
        logger.info(f"📁 File saved for analysis: {temp_path}")
        
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
                # Set up progress callback for WebSocket updates
                def progress_callback(progress, stage, **kwargs):
                    active_analyses[analysis_id]['progress'] = progress
                    active_analyses[analysis_id]['stage'] = stage
                    socketio.emit('analysis_progress', {
                        'analysis_id': analysis_id,
                        'progress': progress,
                        'stage': stage
                    }, room=analysis_id)
                
                # Set progress callback on analyzer
                if hasattr(analyzer, 'set_progress_callback'):
                    analyzer.set_progress_callback(progress_callback)
                
                # Run analysis
                result = analyzer.analyze_single_image(temp_path, **params)
                
                # Add analysis metadata
                result['analysis_id'] = analysis_id
                
                # Add visualizations
                if result.get('success'):
                    result['visualizations'] = create_analysis_visualizations(temp_path, result)
                
                # Store in database if available
                if database_manager and result.get('success'):
                    try:
                        store_analysis_in_database(result)
                    except Exception as db_error:
                        logger.warning(f"Database storage failed: {str(db_error)}")
                
                # Cache result
                analysis_results_cache[analysis_id] = result
                
                # Notify completion
                socketio.emit('analysis_complete', {
                    'analysis_id': analysis_id,
                    'success': result.get('success', False),
                    'total_cells': result.get('total_cells', 0),
                    'quality_score': result.get('quality_score', 0),
                    'processing_time': result.get('processing_time', 0)
                }, room=analysis_id)
                
                # Clean up
                if analysis_id in active_analyses:
                    del active_analyses[analysis_id]
                
            except Exception as e:
                logger.error(f"❌ Background analysis error: {str(e)}")
                
                # Notify error
                socketio.emit('analysis_error', {
                    'analysis_id': analysis_id,
                    'error': str(e)
                }, room=analysis_id)
                
                # Clean up
                if analysis_id in active_analyses:
                    del active_analyses[analysis_id]
        
        # Start background analysis
        analysis_thread = threading.Thread(target=run_analysis)
        analysis_thread.daemon = True
        analysis_thread.start()
        
        return jsonify({
            'analysis_id': analysis_id,
            'status': 'started',
            'message': 'Analysis started - connect to WebSocket for real-time updates'
        })
        
    except Exception as e:
        logger.error(f"❌ Analysis start error: {str(e)}")
        
        # Clean up
        if analysis_id in active_analyses:
            del active_analyses[analysis_id]
        
        return jsonify({
            'error': f'Analysis failed to start: {str(e)}',
            'timestamp': datetime.now().isoformat()
        }), 500
    
    finally:
        # Schedule cleanup of temp file
        if temp_path and os.path.exists(temp_path):
            def cleanup_later():
                time.sleep(300)  # Keep for 5 minutes
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
    """Get analysis result by ID"""
    try:
        if analysis_id in analysis_results_cache:
            result = analysis_results_cache[analysis_id]
            return jsonify(result)
        elif analysis_id in active_analyses:
            return jsonify({
                'analysis_id': analysis_id,
                'status': 'running',
                'progress': active_analyses[analysis_id]['progress'],
                'stage': active_analyses[analysis_id]['stage']
            })
        else:
            return jsonify({'error': 'Analysis not found'}), 404
            
    except Exception as e:
        logger.error(f"❌ Get analysis result error: {str(e)}")
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
        logger.info(f"🔄 Starting batch analysis: {len(files)} images (ID: {batch_id})")
        
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
                
                # Use the analyzer's batch processing if available
                if hasattr(analyzer, 'batch_analyze_images'):
                    result = analyzer.batch_analyze_images(temp_paths, batch_progress_callback)
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
                            logger.error(f"❌ Batch image error: {str(img_error)}")
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
                logger.error(f"❌ Batch analysis error: {str(e)}")
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
        logger.error(f"❌ Batch analysis start error: {str(e)}")
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
            df.to_csv(export_path, index=False)
            return send_file(export_path, as_attachment=True)
            
        elif format.lower() == 'excel':
            export_path = f"exports/wolffia_analysis_{analysis_id}_{timestamp}.xlsx"
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
            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            return send_file(export_path, as_attachment=True)
        
        else:
            return jsonify({'error': f'Unsupported format: {format}'}), 400
            
    except Exception as e:
        logger.error(f"❌ Export error: {str(e)}")
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
        logger.error(f"❌ Visualizations error: {str(e)}")
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
                logger.warning(f"⚠️ Visualization creation failed: {str(viz_error)}")
        
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
        logger.error(f"❌ Visualization creation error: {str(e)}")
        return {}

def store_analysis_in_database(result):
    """Store analysis result in database if available"""
    try:
        if not database_manager:
            logger.info("📝 Database not available, skipping storage")
            return
        
        # Ensure we have default project/experiment
        try:
            # Try to create or get default project
            with database_manager.get_connection() as conn:
                # Check if default project exists
                project = conn.execute("SELECT id FROM projects WHERE name = ?", ("Default Project",)).fetchone()
                if not project:
                    project_id = database_manager.create_project(
                        name="Default Project",
                        description="Default project for Wolffia analysis",
                        operator_name="System"
                    )
                else:
                    project_id = project['id']
                
                # Check if default experiment exists
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
            logger.warning(f"⚠️ Database setup failed: {setup_error}")
            # Use fallback IDs
            project_id = 1
            experiment_id = 1
        
        # Store image
        image_id = database_manager.store_image(
            experiment_id=experiment_id,
            filename=Path(result['image_path']).name,
            file_path=result['image_path']
        )
        
        # Store analysis result
        analysis_id = database_manager.store_analysis_result(image_id, result)
        
        result['database_ids'] = {
            'image_id': image_id,
            'analysis_id': analysis_id
        }
        
        logger.info(f"✅ Analysis stored in database: {analysis_id}")
        
    except Exception as e:
        logger.warning(f"⚠️ Database storage failed (continuing without): {str(e)}")
        # Don't raise - system should continue working without database

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
    logger.error(f"Internal server error: {str(e)}")
    if request.path.startswith('/api/'):
        return jsonify({'error': 'Internal server error'}), 500
    return render_template('index.html'), 500

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    try:
        print("\n" + "=" * 70)
        print("🌱 BIOIMAGIN ENHANCED WOLFFIA ANALYSIS SYSTEM")
        print("   🔬 Production-Ready Live Analysis Platform")
        print("=" * 70)
        print(f"   📊 Analyzer Ready: {'✅ YES' if analyzer_ready else '❌ NO'}")
        print(f"   🤖 ML Enhancement: {'✅ YES' if ML_AVAILABLE else '❌ NO'}")
        print(f"   🗄️ Database Ready: {'✅ YES' if DATABASE_AVAILABLE else '❌ NO'}")
        print(f"   🌐 Server URL: http://localhost:5000")
        print(f"   📡 WebSocket: ws://localhost:5000")
        print(f"   🎯 Features: Live Analysis, Real-time Updates, Debug Mode")
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
        print("\n👋 Server stopped by user")
        print("✅ BIOIMAGIN System shutdown complete")
    except Exception as startup_error:
        logger.error(f"❌ Server startup failed: {startup_error}")
        print(f"❌ Server startup failed: {startup_error}")
        sys.exit(1)