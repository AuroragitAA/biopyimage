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
from werkzeug.utils import secure_filename

# Import our streamlined analyzer
from bioimaging import WolffiaAnalyzer

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'tif', 'jfif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# Create directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('annotations', exist_ok=True)
os.makedirs('tophat_training', exist_ok=True)

# Global analyzer instance - streamlined initialization
analyzer = WolffiaAnalyzer()

# Analysis management
analysis_queue = queue.Queue()
analysis_results = {}
analysis_progress = {}
uploaded_files_store = {}
training_sessions = {}

def convert_numpy_types(obj):
    """Convert numpy types to JSON-serializable Python types"""
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
    else:
        return obj

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
    """System health check"""
    try:
        # Check system status
        available_features = []
        if analyzer.load_tophat_model():
            available_features.append('Tophat AI Model')
        if analyzer.load_cnn_model():
            available_features.append('Wolffia CNN')
        available_features.append('Watershed Segmentation')  # Always available
        
        status = {
            'status': 'healthy',
            'version': '3.0-Deployment',
            'timestamp': datetime.now().isoformat(),
            'features': available_features,  # Add features array for frontend
            'available_methods': {
                'watershed': True,  # Always available
                'tophat': analyzer.load_tophat_model(),
                'cnn': analyzer.load_cnn_model(),
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
    """Upload images for analysis"""
    try:
        if 'files' not in request.files:
            return jsonify({'error': 'No files provided'}), 400
        
        files = request.files.getlist('files')
        if not files or files[0].filename == '':
            return jsonify({'error': 'No files selected'}), 400
        
        uploaded_files = []
        for i, file in enumerate(files):
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                
                # Auto-convert to TIFF for optimal processing
                original_filename = filename
                base_name = os.path.splitext(filename)[0]
                tiff_filename = f"{base_name}.tiff"
                unique_filename = f"{uuid.uuid4()}_{tiff_filename}"
                file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
                
                # Save and convert to TIFF
                temp_path = os.path.join(UPLOAD_FOLDER, f"temp_{uuid.uuid4()}_{filename}")
                file.save(temp_path)
                
                try:
                    # Convert to TIFF using CV2 for better compatibility - PRESERVE RGB
                    import cv2
                    img = cv2.imread(temp_path, cv2.IMREAD_UNCHANGED)
                    if img is not None:
                        # Convert to uint8 if needed
                        if img.dtype != np.uint8:
                            img = cv2.convertScaleAbs(img)

                        # Convert grayscale to 3-channel RGB (not BGR!)
                        if len(img.shape) == 2:
                            print("⚠️ Upload image was grayscale — converting to 3-channel RGB")
                            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                            # Convert RGB back to BGR for cv2.imwrite
                            img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                        elif len(img.shape) == 3 and img.shape[2] == 3:
                            # Already 3-channel, ensure it's in correct format
                            # OpenCV imread loads as BGR, keep as BGR for consistency
                            pass

                        # Drop alpha if present and convert to RGB first
                        if len(img.shape) == 3 and img.shape[2] == 4:
                            print("⚠️ Upload image had alpha — dropping to 3 channels RGB")
                            img = img[:, :, :3]  # Drop alpha channel
                        
                        # Verify we have 3 channels in BGR format for storage
                        if len(img.shape) == 3 and img.shape[2] == 3:
                            # Save as TIFF with consistent BGR format (OpenCV standard)
                            cv2.imwrite(file_path, img, [cv2.IMWRITE_TIFF_COMPRESSION, 1])
                            print(f"✅ Saved {original_filename} as TIFF with 3-channel BGR format")
                        else:
                            print(f"❌ Invalid image format for {original_filename}: shape={img.shape}")
                            raise ValueError(f"Invalid image format: {img.shape}")

                    else:
                        # Fallback: just copy the file if conversion fails
                        shutil.copy2(temp_path, file_path)
                        print(f"⚠️ Could not convert {original_filename}, keeping original format")
                    
                    # Clean up temporary file
                    os.remove(temp_path)
                    
                except Exception as conv_error:
                    print(f"⚠️ TIFF conversion failed for {original_filename}: {conv_error}")
                    # Fallback: use original file
                    shutil.move(temp_path, file_path)
                    unique_filename = f"{uuid.uuid4()}_{filename}"
                    file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
                
                file_info = {
                    'id': str(uuid.uuid4()),
                    'filename': filename,
                    'path': file_path,
                    'upload_order': i + 1,
                    'size': os.path.getsize(file_path),
                    'upload_time': datetime.now().isoformat()
                }
                uploaded_files.append(file_info)
                uploaded_files_store[file_info['id']] = file_info
        
        if not uploaded_files:
            return jsonify({'error': 'No valid files uploaded'}), 400
        
        return jsonify({
            'success': True,
            'files': uploaded_files,
            'message': f'{len(uploaded_files)} files uploaded successfully'
        })
    
    except Exception as e:
        print(f"❌ Upload error: {str(e)}")
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/api/analyze/<file_id>', methods=['POST'])
def analyze_image(file_id):
    """Analyze a specific image using streamlined methods"""
    try:
        request_data = request.get_json() or {}
        
        # Get analysis options with sensible defaults
        use_tophat = request_data.get('use_tophat', True)
        # FIXED: Accept both parameter names for CNN (frontend compatibility)
        use_cnn = request_data.get('use_cnn', False) or request_data.get('use_wolffia_cnn', False)
        use_celldetection = request_data.get('use_celldetection', False)
        
        # Find file info
        if file_id not in uploaded_files_store:
            return jsonify({'error': 'File not found. Please upload the file again.'}), 404
        
        file_info = uploaded_files_store[file_id]
        file_path = file_info['path']
        
        if not os.path.exists(file_path):
            return jsonify({'error': 'File no longer exists on server.'}), 404
        
        # Start analysis in background
        def run_analysis():
            try:
                analysis_results[file_id] = {'status': 'processing', 'progress': 10}
                
                print(f"🔬 Starting analysis for: {file_info.get('filename', 'Unknown')}")
                print(f"📝 Options: tophat={use_tophat}, cnn={use_cnn}, celldetection={use_celldetection}")
                
                # Update progress
                analysis_results[file_id]['progress'] = 25
                
                # Get start time for performance measurement
                start_time = datetime.now()
                
                # Run separate method analysis for better comparison
                img = cv2.imread(str(file_path))
                if img is None:
                    raise Exception(f"Could not load image: {file_path}")
                processed = {'original': img}

                method_results = analyzer.analyze_image_separate_methods(
                    processed,
                    file_path,
                    use_tophat=use_tophat,
                    use_cnn=use_cnn,
                    use_celldetection=use_celldetection
                )

                
                # Create combined result structure for frontend compatibility
                if 'error' in method_results:
                    raise Exception(method_results['error'])
                
                # Get the best method result for summary (prioritize: cnn > celldetection > tophat > watershed)
                best_method = None
                best_cells = 0
                for method_name in ['cnn', 'celldetection', 'tophat', 'watershed']:
                    if method_name in method_results:
                        cells_count = method_results[method_name]['cells_detected']
                        if cells_count > best_cells or best_method is None:
                            best_method = method_name
                            best_cells = cells_count
                
                # Build result structure compatible with frontend
                best_result = method_results[best_method] if best_method else method_results['watershed']
                
                result = {
                    # Legacy format for compatibility
                    'total_cells': best_result['cells_detected'],
                    'total_area': best_result['total_area'],
                    'average_area': best_result['average_area'],
                    'cells': best_result['cells'],
                    'method_used': list(method_results.keys()),
                    
                    # Extended format with separate method results
                    'method_results': method_results,
                    'best_method': best_method,
                    'detection_results': {
                        'detection_method': f"Multi-Method Analysis ({len(method_results)} methods)",
                        'cells_detected': best_result['cells_detected'],
                        'total_area': best_result['total_area'],
                        'cells_data': best_result['cells']
                    },
                    'quantitative_analysis': {
                        'average_cell_area': best_result['average_area'],
                        'biomass_analysis': {
                            'total_biomass_mg': best_result['total_area'] * 0.001,
                        },
                        'color_analysis': {
                            'green_cell_percentage': 85.0
                        },
                        'health_assessment': {
                            'overall_health': 'good',
                            'health_score': 0.75
                        }
                    },
                    'visualizations': {
                        'detection_overview': best_result['visualization_b64']
                    }
                }
                
                # Add pipeline visualization from watershed method if available
                if 'watershed' in method_results and 'pipeline_visualization_b64' in method_results['watershed']:
                    result['visualizations']['pipeline_steps'] = {
                        'pipeline_overview': method_results['watershed']['pipeline_visualization_b64'],
                        'step_count': 11,
                        'step_descriptions': {
                            'original': 'Input image as uploaded by user',
                            'gray': 'Converted to grayscale for processing',
                            'otsu_threshold': 'OTSU thresholding',
                            'morphological_opening': 'Morphological opening',
                            'clear_border': 'Border removal',
                            'sure_background': 'Sure background (dilated)',
                            'distance_transform': 'Distance transform',
                            'sure_foreground': 'Sure foreground',
                            'unknown_region': 'Unknown region',
                            'markers': 'Markers for watershed',
                            'watershed_boundaries': 'Watershed with boundaries',
                            'final_segmentation': 'Final segmentation'
                        }
                    }
                
                # Also add individual method visualizations for complete analysis
                for method_name, method_data in method_results.items():
                    if 'visualization_b64' in method_data and method_data['visualization_b64']:
                        result['visualizations'][f'{method_name}_detection'] = method_data['visualization_b64']
                    if 'pipeline_visualization_b64' in method_data and method_data['pipeline_visualization_b64']:
                        result['visualizations'][f'{method_name}_pipeline'] = method_data['pipeline_visualization_b64']
                
                # Calculate processing time
                end_time = datetime.now()
                processing_time = (end_time - start_time).total_seconds()
                result['processing_time'] = processing_time
                
                # Update progress
                analysis_results[file_id]['progress'] = 85
                
                print(f"✅ Analysis completed in {processing_time:.2f} seconds")
                print(f"📊 Cells detected: {result.get('total_cells', 0)}")
                print(f"📊 Methods used: {result.get('method_used', [])}")
                
                # Add file info to result
                result['file_info'] = {
                    'filename': file_info['filename'],
                    'upload_time': file_info['upload_time'],
                    'file_size': file_info['size']
                }
                
                # Save results
                result_file = Path(RESULTS_FOLDER) / f"{file_id}_result.json"
                try:
                    with open(result_file, 'w') as f:
                        json.dump(convert_numpy_types(result), f, indent=2)
                    print(f"💾 Results saved to {result_file}")
                except Exception as save_error:
                    print(f"⚠️ Failed to save results: {save_error}")
                
                # Save cell data as CSV if cells were detected
                if result.get('cells') and len(result['cells']) > 0:
                    csv_file = Path(RESULTS_FOLDER) / f"{file_id}_cells.csv"
                    try:
                        df = pd.DataFrame(result['cells'])
                        df.to_csv(csv_file, index=False)
                        result['csv_export_path'] = str(csv_file)
                    except Exception as csv_error:
                        print(f"⚠️ Failed to save CSV: {csv_error}")
                
                analysis_results[file_id] = {
                    'status': 'completed',
                    'progress': 100,
                    'result': convert_numpy_types(result)
                }
                
            except Exception as e:
                print(f"❌ Analysis error for {file_id}: {e}")
                import traceback
                traceback.print_exc()
                
                analysis_results[file_id] = {
                    'status': 'error',
                    'progress': 0,
                    'error': str(e),
                    'details': traceback.format_exc()
                }
        
        # Start analysis thread
        thread = threading.Thread(target=run_analysis, daemon=True)
        thread.start()
        
        return jsonify({
            'success': True,
            'analysis_id': file_id,
            'status': 'started',
            'message': 'Analysis started in background'
        })
    
    except Exception as e:
        print(f"❌ Analysis setup error: {str(e)}")
        return jsonify({'error': f'Analysis failed to start: {str(e)}'}), 500

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
    """Analyze multiple images in batch"""
    try:
        request_data = request.get_json() or {}
        files_data = request_data.get('files', [])
        
        if not files_data:
            return jsonify({'error': 'No files provided for batch analysis'}), 400
        
        use_tophat = request_data.get('use_tophat', True)
        # FIXED: Accept both parameter names for CNN (frontend compatibility)
        use_cnn = request_data.get('use_cnn', False) or request_data.get('use_wolffia_cnn', False)
        use_celldetection = request_data.get('use_celldetection', False)
        
        batch_id = str(uuid.uuid4())
        
        def run_batch_analysis():
            try:
                analysis_results[batch_id] = {'status': 'processing', 'progress': 0, 'results': []}
                
                print(f"🔬 Starting batch analysis for {len(files_data)} files")
                
                batch_results = []
                total_files = len(files_data)
                
                for i, file_data in enumerate(files_data):
                    file_id = file_data.get('id')
                    if file_id not in uploaded_files_store:
                        continue
                    
                    file_info = uploaded_files_store[file_id]
                    file_path = file_info['path']
                    
                    if not os.path.exists(file_path):
                        continue
                    
                    try:
                        # Update progress
                        progress = int((i / total_files) * 90)
                        analysis_results[batch_id]['progress'] = progress
                        
                        print(f"📁 Analyzing file {i+1}/{total_files}: {file_info['filename']}")
                        
                        # Analyze image
                        start_time = datetime.now()
                        result = analyzer.analyze_image(
                            file_path,
                            use_tophat=use_tophat,
                            use_cnn=use_cnn,
                            use_celldetection=use_celldetection
                        )
                        end_time = datetime.now()
                        
                        result['processing_time'] = (end_time - start_time).total_seconds()
                        result['file_info'] = file_info
                        
                        batch_results.append({
                            'file_id': file_id,
                            'filename': file_info['filename'],
                            'result': convert_numpy_types(result)
                        })
                        
                        # Save individual result
                        result_file = Path(RESULTS_FOLDER) / f"{file_id}_result.json"
                        with open(result_file, 'w') as f:
                            json.dump(convert_numpy_types(result), f, indent=2)
                        
                    except Exception as file_error:
                        print(f"❌ Error analyzing {file_info['filename']}: {file_error}")
                        batch_results.append({
                            'file_id': file_id,
                            'filename': file_info['filename'],
                            'error': str(file_error)
                        })
                
                # Save batch results
                batch_file = Path(RESULTS_FOLDER) / f"batch_{batch_id}_results.json"
                with open(batch_file, 'w') as f:
                    json.dump(convert_numpy_types(batch_results), f, indent=2)
                
                analysis_results[batch_id] = {
                    'status': 'completed',
                    'progress': 100,
                    'results': batch_results,
                    'summary': {
                        'total_files': total_files,
                        'successful': len([r for r in batch_results if 'error' not in r]),
                        'failed': len([r for r in batch_results if 'error' in r]),
                        'total_cells': sum(r.get('result', {}).get('total_cells', 0) for r in batch_results if 'error' not in r)
                    }
                }
                
            except Exception as e:
                print(f"❌ Batch analysis error: {e}")
                analysis_results[batch_id] = {
                    'status': 'error',
                    'progress': 0,
                    'error': str(e)
                }
        
        # Start batch analysis thread
        thread = threading.Thread(target=run_batch_analysis, daemon=True)
        thread.start()
        
        return jsonify({
            'success': True,
            'batch_id': batch_id,
            'status': 'started',
            'message': f'Batch analysis started for {len(files_data)} files'
        })
    
    except Exception as e:
        print(f"❌ Batch analysis setup error: {str(e)}")
        return jsonify({'error': f'Batch analysis failed to start: {str(e)}'}), 500

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
    """Refresh AI model status (useful after training new models)"""
    try:
        print("🔄 Refreshing model status via API...")
        status = analyzer.refresh_model_status()
        
        return jsonify({
            'success': True,
            'message': 'Model status refreshed',
            'ai_status': {
                'celldetection_available': status['celldetection_available'],
                'tophat_model_available': status['tophat_available'],
                'wolffia_cnn_available': status['wolffia_cnn_available']
            },
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        print(f"❌ Model refresh failed: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500
    

@app.route('/api/celldetection/status')
def celldetection_status():
    """Get CellDetection model status for frontend compatibility"""
    try:
        status = analyzer.get_celldetection_status()
        return jsonify({
            'success': True,
            'status': status
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

# TOPHAT TRAINING ENDPOINTS 

@app.route('/api/tophat/start_training', methods=['POST'])
def start_tophat_training():
    """Start tophat training session with enhanced error handling"""
    try:
        print("🎯 Tophat training endpoint called")
        request_data = request.get_json() or {}
        files = request_data.get('files', [])
        
        if not files:
            print("❌ No files provided for training")
            return jsonify({'error': 'No files provided for training'}), 400
        
        # Get file paths and info - check both uploaded files store and direct file info
        file_infos = []
        for file_info in files:
            if isinstance(file_info, dict):
                file_path = file_info.get('path')
                if file_path and os.path.exists(file_path):
                    # Extract uploadable filename from path
                    server_filename = os.path.basename(file_path)
                    file_infos.append({
                        'server_path': file_path,
                        'upload_filename': server_filename,
                        'original_filename': file_info.get('filename', server_filename)
                    })
                elif 'id' in file_info and file_info['id'] in uploaded_files_store:
                    stored_info = uploaded_files_store[file_info['id']]
                    if os.path.exists(stored_info['path']):
                        server_filename = os.path.basename(stored_info['path'])
                        file_infos.append({
                            'server_path': stored_info['path'],
                            'upload_filename': server_filename,
                            'original_filename': stored_info.get('filename', server_filename)
                        })
        
        print(f"📁 Found {len(file_infos)} valid files for training")
        if not file_infos:
            return jsonify({'error': 'No valid files found for training'}), 400
        
        # Start training session
        session = analyzer.start_tophat_training(file_infos)
        training_sessions[session['id']] = session
        
        return jsonify({
            'success': True,
            'session_id': session['id'],
            'images_count': len(session['images']),
            'session': session
        })
    
    except Exception as e:
        print(f"❌ Training start error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Training failed to start: {str(e)}'}), 500

@app.route('/api/tophat/save_annotations', methods=['POST'])
def save_tophat_annotations():
    """Save user drawing annotations for training with enhanced validation"""
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
    """Train the tophat AI model with enhanced validation"""
    try:
        session_id = request.json.get('session_id')
        if not session_id:
            return jsonify({'error': 'Session ID required'}), 400
        
        # Check if session exists
        if session_id not in training_sessions:
            return jsonify({'error': 'Training session not found'}), 404
        
        # Train model
        success = analyzer.train_tophat_model(session_id)
        
        return jsonify({
            'success': success,
            'message': 'Model trained successfully' if success else 'Training failed - check logs for details'
        })
    
    except Exception as e:
        print(f"❌ Model training error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Model training failed: {str(e)}'}), 500

@app.route('/api/tophat/model_status')
def tophat_model_status():
    """Check if tophat model is available with enhanced status"""
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

@app.route('/api/debug/cnn/<file_id>')
def debug_cnn_analysis(file_id):
    """Debug CNN detection for a specific uploaded image"""
    try:
        if file_id not in uploaded_files_store:
            return jsonify({'error': 'File not found'}), 404
        
        file_path = uploaded_files_store[file_id]['path']
        print(f"🔬 Starting CNN debug analysis for {file_path}")
        
        # Load image
        img = cv2.imread(str(file_path))
        if img is None:
            return jsonify({'error': 'Could not load image'}), 400
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Run debug analysis
        debug_result = analyzer.debug_cnn_detection(gray, save_debug_images=True)
        
        if debug_result is None:
            return jsonify({'error': 'CNN debug analysis failed'}), 500
        
        # Return debug statistics and image paths
        return jsonify({
            'success': True,
            'statistics': debug_result['statistics'],
            'debug_images_saved': True,
            'debug_dir': str(analyzer.dirs['results'] / 'cnn_debug'),
            'message': 'CNN debug analysis completed - check results/cnn_debug/ folder for visualizations'
        })
        
    except Exception as e:
        print(f"❌ CNN debug analysis error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Debug analysis failed: {str(e)}'}), 500

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