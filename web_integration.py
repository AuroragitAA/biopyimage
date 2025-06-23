#!/usr/bin/env python3
"""
BIOIMAGIN Web Integration - DEPLOYMENT VERSION  
Professional Flask backend integrated with streamlined analysis methods
Author: BIOIMAGIN Professional Team
"""

import base64
import json
import os
import queue
import shutil
import threading
import uuid
import zipfile
from datetime import datetime
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from flask import (
    Flask,
    Response,
    jsonify,
    make_response,
    render_template,
    request,
    send_file,
)
from flask_cors import CORS
from PIL import Image
from skimage.color import label2rgb
from werkzeug.utils import secure_filename

# Import our streamlined analyzer
from bioimaging import WolffiaAnalyzer

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'tif', 'jfif'}
SESSIONS_FILE = Path('data/training_sessions.json')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size

# Create directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('annotations', exist_ok=True)
os.makedirs('tophat_training', exist_ok=True)
os.makedirs('results/time_series', exist_ok=True)  # New for time-series

# Global analyzer instance - streamlined initialization
analyzer = WolffiaAnalyzer()

# Analysis management
analysis_queue = queue.Queue()
analysis_results = {}
analysis_progress = {}
uploaded_files_store = {}
training_sessions = {}

def load_training_sessions():
    """Load training sessions from file"""
    try:
        if SESSIONS_FILE.exists():
            with open(SESSIONS_FILE, 'r') as f:
                return json.load(f)
        return {}
    except Exception as e:
        print(f"⚠️ Failed to load training sessions: {e}")
        return {}

training_sessions = load_training_sessions()
time_series_sessions = {}  # New for time-series tracking

def convert_numpy_types(obj):
    """
    Convert numpy types and Path objects to JSON-serializable Python types
    FIXED: Now handles WindowsPath and all Path objects properly
    """
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (Path, os.PathLike)):  # 🔧 FIX: Handle Path objects
        return str(obj)
    elif hasattr(obj, '__fspath__'):  # 🔧 FIX: Handle any path-like object
        return str(obj)
    else:
        return obj

def safe_json_dump(data, file_path, **kwargs):
    """
    Safely dump data to JSON with proper error handling
    FIXED: Uses enhanced convert_numpy_types function
    """
    try:
        # Convert all non-serializable types
        serializable_data = convert_numpy_types(data)
        
        with open(file_path, 'w') as f:
            json.dump(serializable_data, f, **kwargs)
        
        print(f"💾 Successfully saved JSON to {file_path}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to save JSON to {file_path}: {e}")
        
        # Fallback: try with default=str for any remaining issues
        try:
            with open(file_path, 'w') as f:
                json.dump(data, f, default=str, **kwargs)
            print(f"💾 Saved JSON with fallback method to {file_path}")
            return True
        except Exception as fallback_error:
            print(f"❌ Even fallback JSON save failed: {fallback_error}")
            return False

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """Main interface"""
    return render_template('index.html')

@app.route('/uploads/<filename>')
def serve_uploaded_file(filename):
    """Serve uploaded files"""
    try:
        return send_file(os.path.join(UPLOAD_FOLDER, filename))
    except FileNotFoundError:
        return jsonify({'error': 'File not found'}), 404

@app.route('/uploads/display/<filename>')
def serve_display_image(filename):
    """Serve browser-compatible version of uploaded images"""
    try:
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        
        # Check if file exists
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
        
        # Read image with OpenCV (handles TIFF and other formats)
        img = cv2.imread(file_path)
        if img is None:
            return jsonify({'error': 'Cannot read image file'}), 400
        
        # Convert to PNG for browser compatibility
        is_success, buffer = cv2.imencode('.png', img)
        if not is_success:
            return jsonify({'error': 'Cannot convert image'}), 500
        
        # Create response with proper headers
        response = make_response(buffer.tobytes())
        response.headers['Content-Type'] = 'image/png'
        response.headers['Content-Disposition'] = f'inline; filename="{filename}.png"'
        
        return response
        
    except Exception as e:
        print(f"❌ Error serving display image {filename}: {e}")
        return jsonify({'error': f'Failed to serve image: {str(e)}'}), 500

@app.route('/api/health')
def health_check():
    """Enhanced system health check with comprehensive status"""
    try:
        available_features = []
        if analyzer.load_tophat_model():
            available_features.append('Enhanced Tophat AI Model')
        if analyzer.load_cnn_model():
            available_features.append('Enhanced Wolffia CNN')
        if analyzer.celldetection_available:
            available_features.append('CellDetection AI with Patch Processing')
        available_features.append('Enhanced Watershed Segmentation')
        available_features.append('Comprehensive Biomass Analysis')
        available_features.append('Color Wavelength Analysis')
        available_features.append('Time-Series Tracking')
        
        status = {
            'status': 'healthy',
            'version': '3.0-Enhanced-Professional',
            'timestamp': datetime.now().isoformat(),
            'features': available_features,
            'available_methods': {
                'enhanced_watershed': True,
                'enhanced_tophat': analyzer.load_tophat_model(),
                'enhanced_cnn': analyzer.load_cnn_model(),
                'enhanced_celldetection': analyzer.celldetection_available,
            },
            'enhanced_capabilities': {
                'patch_processing': True,
                'biomass_calculation': True,
                'wavelength_analysis': True,
                'time_series_tracking': True,
                'comprehensive_visualization': True,
                'morphometric_analysis': True,
                'spatial_analysis': True,
                'health_assessment': True
            },
            'ai_status': {
                'wolffia_cnn_available': analyzer.wolffia_cnn_available,
                'tophat_model_available': analyzer.tophat_model is not None,
                'celldetection_available': analyzer.celldetection_available
            }
        }
        
        return jsonify(status)
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/upload', methods=['POST'])
def upload_files():
    """Enhanced upload with time-series support"""
    try:
        if 'files' not in request.files:
            return jsonify({'error': 'No files provided'}), 400
        
        files = request.files.getlist('files')
        if not files or files[0].filename == '':
            return jsonify({'error': 'No files selected'}), 400
        
        # Check for time-series metadata
        time_series_data = request.form.get('time_series_data')
        series_id = request.form.get('series_id')
        
        if time_series_data:
            try:
                time_series_info = json.loads(time_series_data)
            except:
                time_series_info = {}
        else:
            time_series_info = {}
        
        uploaded_files = []
        for i, file in enumerate(files):
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                
                original_filename = filename
                base_name = os.path.splitext(filename)[0]
                tiff_filename = f"{base_name}.tiff"
                unique_filename = f"{uuid.uuid4()}_{tiff_filename}"
                file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
                
                temp_path = os.path.join(UPLOAD_FOLDER, f"temp_{uuid.uuid4()}_{filename}")
                file.save(temp_path)
                
                try:
                    import cv2
                    img = cv2.imread(temp_path, cv2.IMREAD_UNCHANGED)
                    preview_b64 = None

                    if img is not None:
                        if img.dtype != np.uint8:
                            img = cv2.convertScaleAbs(img)

                        if len(img.shape) == 2:
                            print("⚠️ Upload image was grayscale — converting to 3-channel RGB")
                            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

                        elif len(img.shape) == 3 and img.shape[2] == 4:
                            print("⚠️ Upload image had alpha — dropping to 3 channels RGB")
                            img = img[:, :, :3]

                        if len(img.shape) == 3 and img.shape[2] == 3:
                            # ✅ Save a 3-channel TIFF version for internal use
                            # ✅ Save the original bytes directly without conversion
                            shutil.copy2(temp_path, file_path)

                            # ✅ Generate preview directly from original bytes for fidelity
                            with open(temp_path, 'rb') as f:
                                raw_bytes = f.read()
                                nparr = np.frombuffer(raw_bytes, np.uint8)
                                img_orig = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                                preview_b64 = None
                                if img_orig is not None:
                                    _, buffer = cv2.imencode('.png', img_orig)
                                    preview_b64 = base64.b64encode(buffer).decode('utf-8')


                        else:
                            raise ValueError(f"Invalid image format: {img.shape}")

                    else:
                        # Couldn't decode — fallback to storing as-is
                        shutil.copy2(temp_path, file_path)
                        print(f"⚠️ Could not convert {original_filename}, keeping original format")

                    os.remove(temp_path)

                except Exception as conv_error:
                    print(f"⚠️ TIFF conversion failed for {original_filename}: {conv_error}")
                    shutil.move(temp_path, file_path)
                    unique_filename = f"{uuid.uuid4()}_{filename}"
                    file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
                    preview_b64 = None

                
                # Enhanced file info with time-series support
                file_info = {
                    'id': str(uuid.uuid4()),
                    'filename': filename,
                    'path': file_path,
                    'upload_order': i + 1,
                    'size': os.path.getsize(file_path),
                    'upload_time': datetime.now().isoformat(),
                    'preview_b64': preview_b64,
                    'series_id': series_id,
                    'timestamp': time_series_info.get('timestamps', [None])[i] if i < len(time_series_info.get('timestamps', [])) else None,
                    'time_point': time_series_info.get('time_points', [None])[i] if i < len(time_series_info.get('time_points', [])) else None
                }
                uploaded_files.append(file_info)
                uploaded_files_store[file_info['id']] = file_info
        
        if not uploaded_files:
            return jsonify({'error': 'No valid files uploaded'}), 400
        
        # Initialize time-series session if applicable
        if series_id and any(f.get('timestamp') for f in uploaded_files):
            time_series_sessions[series_id] = {
                'created': datetime.now().isoformat(),
                'files': uploaded_files,
                'analysis_results': {}
            }
            print(f"📊 Created time-series session: {series_id}")
        
        return jsonify({
            'success': True,
            'files': uploaded_files,
            'series_id': series_id,
            'time_series_enabled': bool(series_id),
            'message': f'{len(uploaded_files)} files uploaded successfully'
        })
    
    except Exception as e:
        print(f"❌ Upload error: {str(e)}")
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/api/analyze/<file_id>', methods=['POST'])
def analyze_image(file_id):
    """Enhanced analysis with comprehensive features and time-series support"""
    try:
        request_data = request.get_json() or {}
        
        # Enhanced analysis options
        use_tophat = request_data.get('use_tophat', True)
        use_cnn = request_data.get('use_cnn', False) or request_data.get('use_wolffia_cnn', False)
        use_celldetection = request_data.get('use_celldetection', False)
        enable_patch_processing = request_data.get('enable_patch_processing', True)
        enable_time_series = request_data.get('enable_time_series', False)
        
        if file_id not in uploaded_files_store:
            return jsonify({'error': 'File not found. Please upload the file again.'}), 404
        
        file_info = uploaded_files_store[file_id]
        file_path = file_info['path']
        
        if not os.path.exists(file_path):
            return jsonify({'error': 'File no longer exists on server.'}), 404
        
        # Enhanced analysis function
        def run_enhanced_analysis():
            try:
                analysis_results[file_id] = {'status': 'processing', 'progress': 10}
                
                print(f"🔬 Starting ENHANCED analysis for: {file_info.get('filename', 'Unknown')}")
                print(f"📝 Options: tophat={use_tophat}, cnn={use_cnn}, celldetection={use_celldetection}")
                print(f"🚀 Enhanced features: patch_processing={enable_patch_processing}, time_series={enable_time_series}")
                
                analysis_results[file_id]['progress'] = 25
                
                start_time = datetime.now()
                
                # Enhanced analysis with time-series support
                analysis_params = {
                    'use_tophat': use_tophat,
                    'use_cnn': use_cnn,
                    'use_celldetection': use_celldetection
                }
                
                # Add time-series parameters if applicable
                if enable_time_series and file_info.get('series_id') and file_info.get('timestamp'):
                    analysis_params.update({
                        'timestamp': file_info['timestamp'],
                        'image_series_id': file_info['series_id']
                    })
                
                # Run enhanced analysis
                result = analyzer.analyze_image(file_path, **analysis_params)
                
                if not isinstance(result, dict) or 'error' in result:
                    raise Exception(result.get('error', 'Analysis failed'))
                
                analysis_results[file_id]['progress'] = 85
                
                # Enhanced result processing
                end_time = datetime.now()
                processing_time = (end_time - start_time).total_seconds()
                result['processing_time'] = processing_time
                
                # Enhanced file info
                result['file_info'] = {
                    'filename': file_info['filename'],
                    'upload_time': file_info['upload_time'],
                    'file_size': file_info['size'],
                    'series_id': file_info.get('series_id'),
                    'timestamp': file_info.get('timestamp'),
                    'time_point': file_info.get('time_point')
                }
                
                print(f"✅ Enhanced analysis completed in {processing_time:.2f} seconds")
                print(f"📊 Cells detected: {result.get('total_cells', 0)}")
                print(f"🧬 Biomass: {result.get('quantitative_analysis', {}).get('biomass_analysis', {}).get('total_biomass_mg', 0):.3f} mg")
                print(f"🟢 Green content: {result.get('quantitative_analysis', {}).get('color_analysis', {}).get('green_percentage', 0):.1f}%")
                
                # Save enhanced results
                result_file = Path(RESULTS_FOLDER) / f"{file_id}_enhanced_result.json"
                try:
                    with open(result_file, 'w') as f:
                        json.dump(convert_numpy_types(result), f, indent=2)
                    print(f"💾 Enhanced results saved to {result_file}")
                except Exception as save_error:
                    print(f"⚠️ Failed to save results: {save_error}")
                
                # Save enhanced cell data as CSV
                if result.get('cells') and len(result['cells']) > 0:
                    csv_file = Path(RESULTS_FOLDER) / f"{file_id}_enhanced_cells.csv"
                    try:
                        df = pd.DataFrame(result['cells'])
                        df.to_csv(csv_file, index=False)
                        result['csv_export_path'] = str(csv_file)
                    except Exception as csv_error:
                        print(f"⚠️ Failed to save enhanced CSV: {csv_error}")
                
                # Update time-series session if applicable
                if enable_time_series and file_info.get('series_id'):
                    series_id = file_info['series_id']
                    if series_id in time_series_sessions:
                        time_series_sessions[series_id]['analysis_results'][file_id] = {
                            'timestamp': file_info.get('timestamp'),
                            'result_summary': {
                                'cell_count': result.get('total_cells', 0),
                                'biomass_mg': result.get('quantitative_analysis', {}).get('biomass_analysis', {}).get('total_biomass_mg', 0),
                                'green_percentage': result.get('quantitative_analysis', {}).get('color_analysis', {}).get('green_percentage', 0)
                            }
                        }
                
                analysis_results[file_id] = {
                    'status': 'completed',
                    'progress': 100,
                    'result': convert_numpy_types(result)
                }
                
            except Exception as e:
                print(f"❌ Enhanced analysis error for {file_id}: {e}")
                import traceback
                traceback.print_exc()
                
                analysis_results[file_id] = {
                    'status': 'error',
                    'progress': 0,
                    'error': str(e),
                    'details': traceback.format_exc()
                }
        
        # Start enhanced analysis thread
        thread = threading.Thread(target=run_enhanced_analysis, daemon=True)
        thread.start()
        
        return jsonify({
            'success': True,
            'analysis_id': file_id,
            'status': 'started',
            'enhanced_features': {
                'patch_processing': enable_patch_processing,
                'biomass_analysis': True,
                'wavelength_analysis': True,
                'time_series': enable_time_series,
                'comprehensive_visualization': True
            },
            'message': 'Enhanced analysis started with professional features'
        })
    
    except Exception as e:
        print(f"❌ Enhanced analysis setup error: {str(e)}")
        return jsonify({'error': f'Enhanced analysis failed to start: {str(e)}'}), 500

@app.route('/api/status/<analysis_id>')
def get_analysis_status(analysis_id):
    """Get analysis status and results"""
    try:
        if analysis_id not in analysis_results:
            return jsonify({'error': 'Analysis not found'}), 404
        
        return jsonify({
            'success': True,
            'analysis': analysis_results[analysis_id]
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze_batch', methods=['POST'])
def analyze_batch():
    """Enhanced batch analysis with time-series and comprehensive features"""
    try:
        request_data = request.get_json() or {}
        files_data = request_data.get('files', [])
        
        if not files_data:
            return jsonify({'error': 'No files provided for batch analysis'}), 400
        
        # Enhanced analysis options
        use_tophat = request_data.get('use_tophat', True)
        use_cnn = request_data.get('use_cnn', False) or request_data.get('use_wolffia_cnn', False)
        use_celldetection = request_data.get('use_celldetection', False)
        enable_time_series = request_data.get('enable_time_series', False)
        series_id = request_data.get('series_id')
        
        batch_id = str(uuid.uuid4())
        
        def run_enhanced_batch_analysis():
            try:
                analysis_results[batch_id] = {'status': 'processing', 'progress': 0, 'results': []}
                
                print(f"🔬 Starting ENHANCED batch analysis for {len(files_data)} files")
                
                batch_results = []
                total_files = len(files_data)
                
                # Enhanced batch processing with time-series support
                for i, file_data in enumerate(files_data):
                    file_id = file_data.get('id')
                    if file_id not in uploaded_files_store:
                        continue
                    
                    file_info = uploaded_files_store[file_id]
                    file_path = file_info['path']
                    
                    if not os.path.exists(file_path):
                        continue
                    
                    try:
                        progress = int((i / total_files) * 90)
                        analysis_results[batch_id]['progress'] = progress
                        
                        print(f"📁 Enhanced analysis {i+1}/{total_files}: {file_info['filename']}")
                        
                        start_time = datetime.now()
                        
                        # Enhanced analysis parameters
                        analysis_params = {
                            'use_tophat': use_tophat,
                            'use_cnn': use_cnn,
                            'use_celldetection': use_celldetection
                        }
                        
                        # Add time-series parameters if applicable
                        if enable_time_series and file_info.get('timestamp') and series_id:
                            analysis_params.update({
                                'timestamp': file_info['timestamp'],
                                'image_series_id': series_id
                            })
                        
                        # Run enhanced analysis
                        result = analyzer.analyze_image(file_path, **analysis_params)
                        
                        end_time = datetime.now()
                        result['processing_time'] = (end_time - start_time).total_seconds()
                        result['file_info'] = file_info
                        
                        # Enhanced result summary
                        result_summary = {
                            'file_id': file_id,
                            'filename': file_info['filename'],
                            'result': convert_numpy_types(result),
                            'enhanced_metrics': {
                                'biomass_mg': result.get('quantitative_analysis', {}).get('biomass_analysis', {}).get('total_biomass_mg', 0),
                                'green_percentage': result.get('quantitative_analysis', {}).get('color_analysis', {}).get('green_percentage', 0),
                                'cell_density': result.get('quantitative_analysis', {}).get('biomass_analysis', {}).get('cell_density_per_mm2', 0),
                                'health_score': result.get('quantitative_analysis', {}).get('health_assessment', {}).get('health_score', 0)
                            }
                        }
                        
                        batch_results.append(result_summary)
                        
                        # Save individual enhanced result
                        result_file = Path(RESULTS_FOLDER) / f"{file_id}_batch_enhanced_result.json"
                        with open(result_file, 'w') as f:
                            json.dump(convert_numpy_types(result), f, indent=2)
                        
                    except Exception as file_error:
                        print(f"❌ Error in enhanced analysis {file_info['filename']}: {file_error}")
                        batch_results.append({
                            'file_id': file_id,
                            'filename': file_info['filename'],
                            'error': str(file_error)
                        })
                
                # Enhanced batch summary
                successful_results = [r for r in batch_results if 'error' not in r]
                
                # Calculate batch statistics
                batch_statistics = {}
                if successful_results:
                    total_cells = sum(r['result']['total_cells'] for r in successful_results)
                    total_biomass = sum(r['enhanced_metrics']['biomass_mg'] for r in successful_results)
                    avg_green_content = np.mean([r['enhanced_metrics']['green_percentage'] for r in successful_results])
                    avg_health_score = np.mean([r['enhanced_metrics']['health_score'] for r in successful_results])
                    
                    batch_statistics = {
                        'total_cells_all_images': total_cells,
                        'total_biomass_mg': total_biomass,
                        'average_green_content_percent': float(avg_green_content),
                        'average_health_score': float(avg_health_score),
                        'images_processed': len(successful_results),
                        'processing_success_rate': len(successful_results) / total_files * 100
                    }
                
                # Save enhanced batch results
                batch_file = Path(RESULTS_FOLDER) / f"enhanced_batch_{batch_id}_results.json"
                enhanced_batch_data = {
                    'batch_id': batch_id,
                    'created': datetime.now().isoformat(),
                    'files_processed': total_files,
                    'results': batch_results,
                    'batch_statistics': batch_statistics,
                    'time_series_enabled': enable_time_series,
                    'series_id': series_id
                }
                
                with open(batch_file, 'w') as f:
                    json.dump(convert_numpy_types(enhanced_batch_data), f, indent=2)
                
                analysis_results[batch_id] = {
                    'status': 'completed',
                    'progress': 100,
                    'results': batch_results,
                    'enhanced_summary': {
                        'total_files': total_files,
                        'successful': len(successful_results),
                        'failed': len([r for r in batch_results if 'error' in r]),
                        'batch_statistics': batch_statistics
                    }
                }
                
            except Exception as e:
                print(f"❌ Enhanced batch analysis error: {e}")
                analysis_results[batch_id] = {
                    'status': 'error',
                    'progress': 0,
                    'error': str(e)
                }
        
        # Start enhanced batch analysis thread
        thread = threading.Thread(target=run_enhanced_batch_analysis, daemon=True)
        thread.start()
        
        return jsonify({
            'success': True,
            'batch_id': batch_id,
            'status': 'started',
            'enhanced_features': {
                'comprehensive_analysis': True,
                'batch_statistics': True,
                'time_series_support': enable_time_series
            },
            'message': f'Enhanced batch analysis started for {len(files_data)} files'
        })
    
    except Exception as e:
        print(f"❌ Enhanced batch analysis setup error: {str(e)}")
        return jsonify({'error': f'Enhanced batch analysis failed to start: {str(e)}'}), 500


@app.route('/api/time_series/<series_id>')
def get_time_series_analysis(series_id):
    """Get comprehensive time-series analysis results"""
    try:
        if series_id not in time_series_sessions:
            return jsonify({'error': 'Time-series session not found'}), 404
        
        session = time_series_sessions[series_id]
        
        # Check if we have enough analysis results
        analysis_results_data = session.get('analysis_results', {})
        if len(analysis_results_data) < 2:
            return jsonify({
                'series_id': series_id,
                'status': 'insufficient_data',
                'message': 'Need at least 2 analyzed images for time-series analysis',
                'current_analyses': len(analysis_results_data)
            })
        
        # Generate comprehensive time-series analysis
        time_points = []
        for file_id, analysis_data in analysis_results_data.items():
            time_points.append({
                'timestamp': analysis_data['timestamp'],
                'file_id': file_id,
                **analysis_data['result_summary']
            })
        
        # Sort by timestamp
        time_points.sort(key=lambda x: x['timestamp'])
        
        # Calculate trends and statistics
        cell_counts = [tp['cell_count'] for tp in time_points]
        biomass_values = [tp['biomass_mg'] for tp in time_points]
        green_percentages = [tp['green_percentage'] for tp in time_points]
        
        trends = {
            'cell_count_trend': analyzer.calculate_trend(cell_counts),
            'biomass_trend': analyzer.calculate_trend(biomass_values),
            'green_content_trend': analyzer.calculate_trend(green_percentages),
            'growth_rate_percent': analyzer.calculate_growth_rate(time_points)
        }
        
        # Generate time-series visualization
        viz_data = analyzer.create_time_series_visualization(time_points, series_id)
        
        return jsonify({
            'success': True,
            'series_id': series_id,
            'time_points': time_points,
            'trends': trends,
            'statistics': {
                'total_time_points': len(time_points),
                'time_span': f"{time_points[0]['timestamp']} to {time_points[-1]['timestamp']}",
                'max_cell_count': max(cell_counts),
                'max_biomass_mg': max(biomass_values),
                'avg_green_content': float(np.mean(green_percentages))
            },
            'visualization_data': viz_data
        })
        
    except Exception as e:
        print(f"❌ Time-series analysis error: {e}")
        return jsonify({'error': str(e)}), 500
    
    

@app.route('/api/export_enhanced/<analysis_id>/<format>')
@app.route('/api/export_enhanced/<analysis_id>/<format>/<method>')
def export_enhanced_results(analysis_id, format, method=None):
    """Enhanced export with comprehensive data and visualizations - supports method-specific exports"""
    try:
        if analysis_id not in analysis_results:
            return jsonify({'error': 'Analysis not found'}), 404
        
        analysis = analysis_results[analysis_id]
        if analysis['status'] != 'completed':
            return jsonify({'error': 'Analysis not completed'}), 400
        
        result = analysis['result']
        
        if format == 'json':
            # Enhanced JSON export with all analysis data
            enhanced_data = {
                'analysis_metadata': {
                    'analysis_id': analysis_id,
                    'export_timestamp': datetime.now().isoformat(),
                    'bioimagin_version': '3.0-Enhanced-Professional'
                },
                'results': result,
                'enhanced_features': {
                    'biomass_analysis': True,
                    'wavelength_analysis': True,
                    'morphometric_analysis': True,
                    'spatial_analysis': True,
                    'health_assessment': True
                }
            }
            
            json_data = json.dumps(convert_numpy_types(enhanced_data), indent=2)
            
            buffer = BytesIO()
            buffer.write(json_data.encode())
            buffer.seek(0)
            
            return send_file(
                buffer,
                as_attachment=True,
                download_name=f'enhanced_analysis_{analysis_id}.json',
                mimetype='application/json'
            )
        
        elif format == 'csv':
            # Enhanced CSV export with comprehensive cell data - method-specific support
            cells_data = None
            export_method = 'Combined Analysis'
            
            # Check if specific method is requested
            if method and method != 'all':
                method_results = result.get('detection_results', {}).get('method_results', {})
                if method in method_results and 'cells_data' in method_results[method]:
                    cells_data = method_results[method]['cells_data']
                    export_method = method_results[method]['method_name']
                else:
                    return jsonify({'error': f'Method "{method}" not found or has no cell data'}), 400
            else:
                # Use combined results
                cells_data = result.get('cells', [])
            
            if not cells_data:
                return jsonify({'error': f'No cell data available for {export_method}'}), 400
            
            df = pd.DataFrame(cells_data)
            
            # Add summary statistics as comments
            summary_stats = result.get('quantitative_analysis', {})
            csv_content = f"# BIOIMAGIN Enhanced Analysis Results\n"
            csv_content += f"# Analysis ID: {analysis_id}\n"
            csv_content += f"# Export Date: {datetime.now().isoformat()}\n"
            csv_content += f"# Detection Method: {export_method}\n"
            csv_content += f"# Total Cells ({export_method}): {len(cells_data)}\n"
            
            if 'biomass_analysis' in summary_stats:
                biomass = summary_stats['biomass_analysis']
                csv_content += f"# Total Biomass: {biomass.get('total_biomass_mg', 0):.3f} mg\n"
                csv_content += f"# Cell Density: {biomass.get('cell_density_per_mm2', 0):.2f} cells/mm²\n"
            
            if 'color_analysis' in summary_stats:
                color = summary_stats['color_analysis']
                csv_content += f"# Green Content: {color.get('green_percentage', 0):.1f}%\n"
            
            csv_content += "#\n"
            csv_content += df.to_csv(index=False)
            
            buffer = BytesIO()
            buffer.write(csv_content.encode())
            buffer.seek(0)
            
            # Create method-specific filename
            method_suffix = f"_{method}" if method and method != 'all' else ""
            
            return send_file(
                buffer,
                as_attachment=True,
                download_name=f'enhanced_cells_{analysis_id}{method_suffix}.csv',
                mimetype='text/csv'
            )
        
        elif format == 'zip':
            # Enhanced ZIP export with all visualizations and data
            buffer = BytesIO()
            
            with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                # Add enhanced JSON results
                enhanced_data = {
                    'analysis_metadata': {
                        'analysis_id': analysis_id,
                        'export_timestamp': datetime.now().isoformat(),
                        'bioimagin_version': '3.0-Enhanced-Professional'
                    },
                    'results': result
                }
                json_data = json.dumps(convert_numpy_types(enhanced_data), indent=2)
                zip_file.writestr(f'enhanced_analysis_{analysis_id}.json', json_data)
                
                # Add enhanced CSV if available
                if result.get('cells'):
                    df = pd.DataFrame(result['cells'])
                    csv_data = df.to_csv(index=False)
                    zip_file.writestr(f'enhanced_cells_{analysis_id}.csv', csv_data)
                
                # Add all visualizations
                visualizations = result.get('visualizations', {})
                for viz_name, viz_path in visualizations.items():
                    if viz_path and isinstance(viz_path, str) and os.path.exists(viz_path):
                        zip_file.write(viz_path, f'visualizations/{viz_name}_{analysis_id}.png')
                
                # Add histogram files if they exist
                histogram_paths = visualizations.get('histogram_paths', {})
                for hist_name, hist_path in histogram_paths.items():
                    if hist_path and isinstance(hist_path, str) and os.path.exists(hist_path):
                        zip_file.write(hist_path, f'histograms/{hist_name}_{analysis_id}.png')
                
                # Add summary report
                summary_text = generate_summary_report(result, analysis_id)
                zip_file.writestr(f'summary_report_{analysis_id}.txt', summary_text)
            
            buffer.seek(0)
            
            return send_file(
                buffer,
                as_attachment=True,
                download_name=f'enhanced_analysis_package_{analysis_id}.zip',
                mimetype='application/zip'
            )
        
        else:
            return jsonify({'error': 'Unsupported export format. Use: json, csv, zip'}), 400
    
    except Exception as e:
        print(f"❌ Enhanced export error: {str(e)}")
        return jsonify({'error': f'Enhanced export failed: {str(e)}'}), 500

def generate_summary_report(result, analysis_id):
    """Generate comprehensive text summary report"""
    report = f"""
BIOIMAGIN ENHANCED ANALYSIS REPORT
==================================
Analysis ID: {analysis_id}
Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
BIOIMAGIN Version: 3.0-Enhanced-Professional

DETECTION RESULTS
-----------------
Total Cells Detected: {result.get('total_cells', 0)}
Total Cell Area: {result.get('total_area', 0):.2f} μm²
Average Cell Area: {result.get('average_area', 0):.2f} μm²
Detection Methods Used: {', '.join(result.get('method_used', []))}

BIOMASS ANALYSIS
----------------
"""
    
    biomass = result.get('quantitative_analysis', {}).get('biomass_analysis', {})
    if biomass:
        report += f"""Fresh Weight Biomass: {biomass.get('total_biomass_mg', 0):.3f} mg
Dry Weight Biomass: {biomass.get('dry_biomass_mg', 0):.3f} mg
Cell Density: {biomass.get('cell_density_per_mm2', 0):.2f} cells/mm²
Average Cell Biomass: {biomass.get('average_cell_biomass_mg', 0):.6f} mg
"""
    
    report += """
COLOR ANALYSIS
--------------
"""
    
    color = result.get('quantitative_analysis', {}).get('color_analysis', {})
    if color:
        report += f"""Green Content: {color.get('green_percentage', 0):.1f}%
Chlorophyll Content: {color.get('chlorophyll_percentage', 0):.1f}%
Color Intensity (mean): {color.get('green_intensity', {}).get('mean', 0):.1f}
Color Uniformity: {color.get('color_uniformity', 0):.3f}
"""
    
    health = result.get('quantitative_analysis', {}).get('health_assessment', {})
    if health:
        report += f"""
HEALTH ASSESSMENT
-----------------
Overall Health: {health.get('overall_health', 'unknown').title()}
Health Score: {health.get('health_score', 0):.2f}/1.0

Recommendations:
"""
        for rec in health.get('recommendations', []):
            report += f"• {rec}\n"
    
    morphometric = result.get('quantitative_analysis', {}).get('morphometric_analysis', {})
    if morphometric and 'area_statistics' in morphometric:
        area_stats = morphometric['area_statistics']
        report += f"""
MORPHOMETRIC ANALYSIS
---------------------
Area Statistics:
  Mean: {area_stats.get('mean', 0):.2f} μm²
  Standard Deviation: {area_stats.get('std', 0):.2f} μm²
  Range: {area_stats.get('min', 0):.2f} - {area_stats.get('max', 0):.2f} μm²
  Median: {area_stats.get('median', 0):.2f} μm²

Size Distribution:
  Small Cells: {morphometric.get('size_distribution', {}).get('small_cells', 0)}
  Medium Cells: {morphometric.get('size_distribution', {}).get('medium_cells', 0)}
  Large Cells: {morphometric.get('size_distribution', {}).get('large_cells', 0)}
"""
    
    spatial = result.get('quantitative_analysis', {}).get('spatial_analysis', {})
    if spatial and 'nearest_neighbor_distance_um' in spatial:
        nn_dist = spatial['nearest_neighbor_distance_um']
        report += f"""
SPATIAL ANALYSIS
----------------
Nearest Neighbor Distance: {nn_dist.get('mean', 0):.2f} ± {nn_dist.get('std', 0):.2f} μm
Clustering Pattern: {spatial.get('clustering_interpretation', 'unknown').title()}
Clustering Index: {spatial.get('clustering_index', 0):.2f}
"""
    
    report += f"""
PROCESSING INFORMATION
----------------------
Processing Time: {result.get('processing_time', 0):.2f} seconds
File Information: {result.get('file_info', {}).get('filename', 'Unknown')}

This report was generated by BIOIMAGIN Enhanced Professional Analysis System.
For more information, visit: https://github.com/AuroragitAA/bioimagin
"""
    
    return report

    
@app.route('/api/export/<analysis_id>/<format>')
def export_results(analysis_id, format):
    """Export analysis results in various formats"""
    try:
        if analysis_id not in analysis_results:
            return jsonify({'error': 'Analysis not found'}), 404
        
        analysis = analysis_results[analysis_id]
        if analysis['status'] != 'completed':
            return jsonify({'error': 'Analysis not completed'}), 400
        
        result = analysis['result']
        
        if format == 'json':
            # Export as JSON
            json_data = json.dumps(convert_numpy_types(result), indent=2)
            
            buffer = BytesIO()
            buffer.write(json_data.encode())
            buffer.seek(0)
            
            return send_file(
                buffer,
                as_attachment=True,
                download_name=f'analysis_{analysis_id}.json',
                mimetype='application/json'
            )
        
        elif format == 'csv':
            # Export cell data as CSV
            if not result.get('cells'):
                return jsonify({'error': 'No cell data available for CSV export'}), 400
            
            df = pd.DataFrame(result['cells'])
            
            buffer = BytesIO()
            df.to_csv(buffer, index=False)
            buffer.seek(0)
            
            return send_file(
                buffer,
                as_attachment=True,
                download_name=f'cells_{analysis_id}.csv',
                mimetype='text/csv'
            )
        
        elif format == 'zip':
            # Export complete package as ZIP
            buffer = BytesIO()
            
            with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                # Add JSON results
                json_data = json.dumps(convert_numpy_types(result), indent=2)
                zip_file.writestr(f'analysis_{analysis_id}.json', json_data)
                
                # Add CSV if available
                if result.get('cells'):
                    df = pd.DataFrame(result['cells'])
                    csv_data = df.to_csv(index=False)
                    zip_file.writestr(f'cells_{analysis_id}.csv', csv_data)
                
                # Add labeled image if available
                if result.get('labeled_image_path') and os.path.exists(result['labeled_image_path']):
                    zip_file.write(result['labeled_image_path'], f'labeled_image_{analysis_id}.png')
            
            buffer.seek(0)
            
            return send_file(
                buffer,
                as_attachment=True,
                download_name=f'analysis_package_{analysis_id}.zip',
                mimetype='application/zip'
            )
        
        else:
            return jsonify({'error': 'Unsupported export format'}), 400
    
    except Exception as e:
        print(f"❌ Export error: {str(e)}")
        return jsonify({'error': f'Export failed: {str(e)}'}), 500
    
@app.route('/api/refresh_models', methods=['POST'])
def refresh_models():
    """Refresh AI model status"""
    try:
        print("🔄 Refreshing enhanced model status...")
        status = analyzer.refresh_model_status()
        
        return jsonify({
            'success': True,
            'message': 'Enhanced model status refreshed',
            'ai_status': {
                'celldetection_available': status['celldetection_available'],
                'tophat_model_available': status['tophat_available'],
                'wolffia_cnn_available': status['wolffia_cnn_available']
            },
            'enhanced_features': {
                'patch_processing': True,
                'biomass_analysis': True,
                'wavelength_analysis': True,
                'time_series_tracking': True
            },
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        print(f"❌ Enhanced model refresh failed: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500
    

@app.route('/api/celldetection/status')
def celldetection_status():
    """Get CellDetection model status"""
    try:
        status = analyzer.get_celldetection_status()
        return jsonify({
            'success': True,
            'status': status,
            'enhanced_features': {
                'patch_processing': True,
                'overlapping_tiles': True
            }
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'status': {
                'available': False,
                'model_loaded': False,
                'device': 'unknown',
                'model_name': None
            }
        })



@app.route('/api/train_models', methods=['POST'])
def train_models():
    """Train available models"""
    try:
        request_data = request.get_json() or {}
        train_cnn = request_data.get('train_cnn', True)
        train_tophat = request_data.get('train_tophat', True)
        
        training_id = str(uuid.uuid4())
        
        def run_training():
            try:
                training_results = {
                    'status': 'processing',
                    'progress': 0,
                    'results': {}
                }
                analysis_results[training_id] = training_results
                
                # Train CNN if requested and PyTorch available
                if train_cnn:
                    try:
                        from wolffia_cnn_model import train_wolffia_cnn
                        training_results['progress'] = 10
                        print("🤖 Training CNN model...")
                        
                        success = train_wolffia_cnn(num_samples=2000, epochs=30)
                        training_results['results']['cnn'] = {
                            'success': success,
                            'message': 'CNN training completed' if success else 'CNN training failed'
                        }
                        training_results['progress'] = 50
                        
                    except Exception as e:
                        training_results['results']['cnn'] = {
                            'success': False,
                            'error': str(e)
                        }
                
                # Train Tophat if requested
                if train_tophat:
                    try:
                        from tophat_trainer import train_tophat_model
                        training_results['progress'] = 60
                        print("🎯 Training Tophat ML model...")
                        
                        success = train_tophat_model()
                        training_results['results']['tophat'] = {
                            'success': success,
                            'message': 'Tophat training completed' if success else 'Tophat training failed (no annotation data)'
                        }
                        training_results['progress'] = 90
                        
                    except Exception as e:
                        training_results['results']['tophat'] = {
                            'success': False,
                            'error': str(e)
                        }
                
                training_results['status'] = 'completed'
                training_results['progress'] = 100
                
            except Exception as e:
                training_results['status'] = 'error'
                training_results['error'] = str(e)
        
        # Start training thread
        thread = threading.Thread(target=run_training, daemon=True)
        thread.start()
        
        return jsonify({
            'success': True,
            'training_id': training_id,
            'message': 'Training started in background'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500






def save_training_sessions():
    """Save training sessions to file"""
    try:
        SESSIONS_FILE.parent.mkdir(exist_ok=True)
        with open(SESSIONS_FILE, 'w') as f:
            json.dump(training_sessions, f, indent=2)
    except Exception as e:
        print(f"⚠️ Failed to save training sessions: {e}")

# Load sessions at startup

# TOPHAT TRAINING ENDPOINTS 
@app.route('/api/tophat/start_training', methods=['POST'])
def start_tophat_training():
    """Start tophat training session with enhanced support for both file_ids and files"""
    try:
        print("🎯 Tophat training endpoint called")
        request_data = request.get_json() or {}
        
        # Support both formats: file_ids array and files array
        file_ids = request_data.get('file_ids', [])
        files = request_data.get('files', [])
        use_tophat = request_data.get('use_tophat', True)
        fallback = request_data.get('fallback_to_watershed', False)

        print(f"🔍 Received file_ids: {file_ids}")
        print(f"🔍 Received files: {files}")
        print(f"🔍 Available in uploaded_files_store: {list(uploaded_files_store.keys())}")

        # Handle file_ids format (recommended)
        if file_ids:
            print("📝 Processing file_ids format")
            files_to_process = []
            for file_id in file_ids:
                if file_id in uploaded_files_store:
                    stored_info = uploaded_files_store[file_id]
                    files_to_process.append({
                        'id': file_id,
                        'path': stored_info['path'],
                        'filename': stored_info['filename'],
                        'original_filename': stored_info.get('original_filename', stored_info['filename'])
                    })
                    print(f"✅ Found file_id {file_id}: {stored_info['filename']}")
                else:
                    print(f"❌ file_id {file_id} not found in uploaded_files_store")
        
        # Handle files format (legacy support)
        elif files:
            print("📝 Processing files format")
            files_to_process = files
        
        else:
            print("❌ No files provided for training")
            return jsonify({'error': 'No files selected for training'}), 400

        if not files_to_process:
            return jsonify({'error': 'No valid files found for training'}), 400

        # Process files into file_infos format
        file_infos = []
        for file_info in files_to_process:
            if isinstance(file_info, dict):
                file_path = file_info.get('path')
                if file_path and os.path.exists(file_path):
                    server_filename = os.path.basename(file_path)
                    file_infos.append({
                        'server_path': file_path,
                        'upload_filename': server_filename,
                        'original_filename': file_info.get('filename', server_filename)
                    })
                    print(f"✅ Added file: {file_info.get('filename', server_filename)}")
                elif 'id' in file_info and file_info['id'] in uploaded_files_store:
                    stored_info = uploaded_files_store[file_info['id']]
                    if os.path.exists(stored_info['path']):
                        server_filename = os.path.basename(stored_info['path'])
                        file_infos.append({
                            'server_path': stored_info['path'],
                            'upload_filename': server_filename,
                            'original_filename': stored_info.get('filename', server_filename)
                        })
                        print(f"✅ Added stored file: {stored_info.get('filename', server_filename)}")

        print(f"📁 Found {len(file_infos)} valid files for training")
        if not file_infos:
            return jsonify({'error': 'No valid files found for training'}), 400

        # Create training images with original display
        images = []
        for file_entry in file_infos:
            image_path = file_entry['server_path']
            bgr_image = cv2.imread(image_path)
            if bgr_image is None:
                print(f"⚠️ Could not load image: {image_path}")
                continue

            # Convert BGR to RGB for proper display
            rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

            # Encode original RGB image directly as PNG
            _, buffer = cv2.imencode('.png', cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
            overlay_b64 = base64.b64encode(buffer).decode('utf-8')

            # Also keep the original binary file as raw input
            try:
                with open(image_path, 'rb') as f:
                    raw_input_b64 = base64.b64encode(f.read()).decode('utf-8')
            except Exception as e:
                print(f"⚠️ Could not read raw file {image_path}: {e}")
                raw_input_b64 = overlay_b64  # Fallback to overlay

            images.append({
                'id': str(uuid.uuid4()),
                'filename': file_entry['original_filename'],
                'path': image_path,  # Keep path for session persistence
                'method_used': 'Original Image Only',
                'image_data': {
                    'training_overlay_b64': overlay_b64,
                    'raw_input_b64': raw_input_b64,
                    'visualizations': {
                        'detection_overview': overlay_b64
                    }
                }
            })
            print(f"✅ Processed training image: {file_entry['original_filename']}")

        if not images:
            return jsonify({'error': 'No training images processed'}), 400

        # Create training session
        session_id = str(uuid.uuid4())
        session = {
            'id': session_id,
            'created': datetime.now().isoformat(),
            'images': images,
            'file_infos': file_infos,  # Store original file info
            'status': 'active',
            'use_tophat': use_tophat,
            'fallback_used': fallback,
            'annotations': {}  # Initialize annotations storage
        }

        # Store session in memory
        training_sessions[session_id] = session
        
        # Save sessions to file for persistence
        try:
            save_training_sessions()
            print(f"💾 Saved training session to file")
        except Exception as e:
            print(f"⚠️ Could not save training sessions: {e}")

        print(f"✅ Started training session {session_id} with {len(images)} images")
        return jsonify({
            'success': True,
            'session_id': session_id,
            'images_count': len(images),
            'session': session
        })

    except Exception as e:
        print(f"❌ Training start error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Training failed to start: {str(e)}'}), 500


    
@app.route('/api/tophat/save_annotations', methods=['POST'])
def save_tophat_annotations():
    """Save user drawing annotations for training with session persistence"""
    try:
        data = request.json
        session_id = data.get('session_id')
        image_filename = data.get('image_filename')
        image_index = data.get('image_index', 0)
        annotations = data.get('annotations', {})
        annotated_image = data.get('annotated_image', '')
        
        if not all([session_id, image_filename]):
            return jsonify({'error': 'Missing required fields (session_id, image_filename)'}), 400
        
        # Validate annotations structure
        valid_annotation_types = ['correct', 'false_positive', 'missed']
        for ann_type in annotations:
            if ann_type not in valid_annotation_types:
                return jsonify({'error': f'Invalid annotation type: {ann_type}'}), 400
        
        # Save drawing annotations with image data
        annotation = analyzer.save_drawing_annotations(
            session_id, image_filename, image_index, annotations, annotated_image
        )
        
        # Save sessions after annotation update
        save_training_sessions()
        
        return jsonify({
            'success': True,
            'annotation_saved': True,
            'annotation': annotation
        })
    
    except Exception as e:
        print(f"❌ Save annotations error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Failed to save annotations: {str(e)}'}), 500
    
@app.route('/api/tophat/train_model', methods=['POST'])
def train_tophat_model():
    """Train the tophat AI model"""
    try:
        session_id = request.json.get('session_id')
        if not session_id:
            return jsonify({'error': 'Session ID required'}), 400
        
        if session_id not in training_sessions:
            return jsonify({'error': 'Training session not found'}), 404
        
        success = analyzer.train_tophat_model(session_id)
        
        return jsonify({
            'success': success,
            'message': 'Enhanced model trained successfully' if success else 'Training failed - check logs for details'
        })
    
    except Exception as e:
        print(f"❌ Model training error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Model training failed: {str(e)}'}), 500

@app.route('/api/tophat/status')
def tophat_model_status():
    """Check if tophat model is available"""
    try:
        status = analyzer.get_tophat_status()
        return jsonify({
            'success': True,
            'model_available': status['model_available'],
            'model_trained': status['model_trained'],
            'training_sessions_active': len(training_sessions)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'model_available': False,
            'model_trained': False
        })


if __name__ == '__main__':
    print("🚀 BIOIMAGIN Web Interface - Deployment Version")
    print("=" * 60)
    print("✅ Professional analysis system ready")
    print("✅ Streamlined backend integrated")
    print("✅ All methods working smoothly")
    print("=" * 60)
    print("🌐 Starting server on http://localhost:5000")
    print("📝 Upload Wolffia images and start analyzing!")
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)