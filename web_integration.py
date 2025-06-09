#!/usr/bin/env python3
"""
BIOIMAGIN Enhanced Flask backend for ML-powered Wolffia analysis
Simple, fast, effective web interface with tophat training
"""

import base64
import json
import os
import queue
import threading
import uuid
import zipfile
from datetime import datetime
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, Response, jsonify, render_template, request, send_file
from flask_cors import CORS
from PIL import Image
from werkzeug.utils import secure_filename

from bioimaging import WolffiaAnalyzer, analyze_multiple_images, analyze_uploaded_image

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'tif', 'jfif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Global analyzer instance
analyzer = WolffiaAnalyzer(pixel_to_micron_ratio=0.5, chlorophyll_threshold=0.6)

# Analysis queue for real-time updates
analysis_queue = queue.Queue()
analysis_results = {}
analysis_progress = {}

# Store uploaded files globally for access
uploaded_files_store = {}
training_sessions = {}

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

@app.route('/api/upload', methods=['POST'])
def upload_files():
    """Upload images for analysis"""
    try:
        if 'files' not in request.files:
            return jsonify({'error': 'No files provided'}), 400
        
        files = request.files.getlist('files')
        if not files or files[0].filename == '':
            return jsonify({'error': 'No files selected'}), 400
        
        use_tophat = request.form.get('use_tophat', 'false').lower() == 'true'
        
        uploaded_files = []
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                unique_filename = f"{uuid.uuid4()}_{filename}"
                file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
                file.save(file_path)
                
                file_info = {
                    'id': str(uuid.uuid4()),
                    'filename': filename,
                    'path': file_path,
                    'use_tophat': use_tophat
                }
                uploaded_files.append(file_info)
                # Store globally for later access
                uploaded_files_store[file_info['id']] = file_info
        
        if not uploaded_files:
            return jsonify({'error': 'No valid files uploaded'}), 400
        
        return jsonify({
            'success': True,
            'files': uploaded_files,
            'message': f'{len(uploaded_files)} files uploaded successfully'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============================================================================
# ADD this new endpoint to web_integration.py
# ============================================================================

@app.route('/api/celldetection/status')
def celldetection_status():
    """Get CellDetection model status"""
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

# ============================================================================
# UPDATE the existing analyze_image endpoint in web_integration.py
# ============================================================================

@app.route('/api/analyze/<file_id>', methods=['POST'])
def analyze_image(file_id):
    """Analyze a specific image with CellDetection option"""
    try:
        # Get file info from request or from stored upload data
        request_data = request.get_json() or {}
        files_data = request_data.get('files', [])
        
        # Get analysis options
        use_tophat = request_data.get('use_tophat', False)
        use_celldetection = request_data.get('use_celldetection', True)  # Default to True
        
        # Find file info
        file_info = None
        
        if files_data:
            for file_data in files_data:
                if isinstance(file_data, dict) and file_data.get('id') == file_id:
                    file_info = file_data
                    break
        
        if not file_info and file_id in uploaded_files_store:
            file_info = uploaded_files_store[file_id].copy()
            file_info['use_tophat'] = use_tophat
            file_info['use_celldetection'] = use_celldetection
        
        if not file_info:
            return jsonify({'error': 'File info not found. Please upload the file again.'}), 404
        
        # Start analysis in background
        def run_analysis():
            try:
                analysis_results[file_id] = {'status': 'processing', 'progress': 0}
                
                print(f"🔬 Starting analysis for file: {file_info.get('path', 'Unknown path')}")
                print(f"📝 Analysis options: tophat={use_tophat}, celldetection={use_celldetection}")
                
                # Validate file path
                file_path = file_info.get('path')
                if not file_path or not os.path.exists(file_path):
                    raise FileNotFoundError(f"File not found: {file_path}")
                
                # Run analysis with CellDetection option
                result = analyzer.analyze_image(
                    file_path, 
                    use_tophat=use_tophat,
                    use_celldetection=use_celldetection  # NEW parameter
                )
                
                print(f"📊 Analysis result: {result.get('success', False)}")
                
                # Save results
                result_file = Path('results') / f"{file_id}_result.json"
                
                try:
                    json_str = json.dumps(result, indent=2, default=str)
                    with open(result_file, 'w') as f:
                        f.write(json_str)
                    print(f"💾 Results saved to {result_file}")
                except Exception as save_error:
                    print(f"⚠️ Failed to save results: {save_error}")
                
                analysis_results[file_id] = {
                    'status': 'completed',
                    'progress': 100,
                    'result': result
                }
                
            except Exception as e:
                print(f"❌ Analysis error: {e}")
                analysis_results[file_id] = {
                    'status': 'error',
                    'progress': 0,
                    'error': str(e)
                }
        
        # Start analysis thread
        thread = threading.Thread(target=run_analysis)
        thread.start()
        
        return jsonify({
            'success': True,
            'analysis_id': file_id,
            'status': 'started'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/api/status/<analysis_id>')
def get_analysis_status(analysis_id):
    """Get analysis status and results"""
    try:
        if analysis_id not in analysis_results:
            return jsonify({'error': 'Analysis not found'}), 404
        
        analysis = analysis_results[analysis_id]
        
        response = {
            'analysis_id': analysis_id,
            'status': analysis['status'],
            'progress': analysis['progress']
        }
        
        if analysis['status'] == 'completed':
            response['result'] = analysis['result']
        elif analysis['status'] == 'error':
            response['error'] = analysis['error']
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health')
def health_check():
    """System health check"""
    return jsonify({
        'status': 'healthy',
        'version': 'optimized',
        'analyzer': 'WolffiaAnalyzer',
        'features': [
            'smart_detection',
            'tophat_training',
            'essential_visualization'
        ],
        'timestamp': datetime.now().isoformat()
    })

# TOPHAT TRAINING ENDPOINTS

@app.route('/api/tophat/start_training', methods=['POST'])
def start_tophat_training():
    """Start tophat training session"""
    try:
        print("🎯 Tophat training endpoint called")
        request_data = request.get_json() or {}
        files = request_data.get('files', [])
        
        if not files:
            print("❌ No files provided for training")
            return jsonify({'error': 'No files provided for training'}), 400
        
        # Get file paths - check both uploaded files store and direct file info
        file_paths = []
        for file_info in files:
            if isinstance(file_info, dict):
                file_path = file_info.get('path')
                if file_path and os.path.exists(file_path):
                    file_paths.append(file_path)
                elif 'id' in file_info and file_info['id'] in uploaded_files_store:
                    stored_info = uploaded_files_store[file_info['id']]
                    if os.path.exists(stored_info['path']):
                        file_paths.append(stored_info['path'])
        
        print(f"📁 Found {len(file_paths)} valid files for training")
        if not file_paths:
            return jsonify({'error': 'No valid files found for training'}), 400
        
        # Start training session
        session = analyzer.start_tophat_training(file_paths)
        training_sessions[session['id']] = session
        
        return jsonify({
            'success': True,
            'session_id': session['id'],
            'images_count': len(session['images']),
            'session': session
        })
    
    except Exception as e:
        print(f"❌ Training start error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/tophat/save_annotations', methods=['POST'])
def save_tophat_annotations():
    """Save user drawing annotations for training"""
    try:
        data = request.json
        session_id = data.get('session_id')
        image_filename = data.get('image_filename')
        image_index = data.get('image_index', 0)
        annotations = data.get('annotations', {})
        annotated_image = data.get('annotated_image', '')
        
        if not all([session_id, image_filename]):
            return jsonify({'error': 'Missing required fields'}), 400
        
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
        return jsonify({'error': str(e)}), 500

@app.route('/api/tophat/train_model', methods=['POST'])
def train_tophat_model():
    """Train the tophat AI model"""
    try:
        session_id = request.json.get('session_id')
        if not session_id:
            return jsonify({'error': 'Session ID required'}), 400
        
        # Train model
        success = analyzer.train_tophat_model(session_id)
        
        return jsonify({
            'success': success,
            'message': 'Model trained successfully' if success else 'Training failed'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/tophat/model_status')
def tophat_model_status():
    """Check if tophat model is available"""
    return jsonify({
        'model_available': analyzer.tophat_model is not None,
        'model_trained': analyzer.tophat_model is not None
    })

# EXPORT ENDPOINTS

@app.route('/api/export/<analysis_id>/<format>')
def export_results(analysis_id, format):
    """Export analysis results"""
    try:
        if analysis_id not in analysis_results:
            return jsonify({'error': 'Analysis not found'}), 404
        
        analysis = analysis_results[analysis_id]
        if analysis['status'] != 'completed':
            return jsonify({'error': 'Analysis not completed'}), 400
        
        result = analysis['result']
        
        if format == 'json':
            # Export as JSON
            output_file = f"results/{analysis_id}_export.json"
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            
            return send_file(output_file, as_attachment=True)
        
        elif format == 'csv':
            # Enhanced CSV export with biomass data
            if result['cells_data']:
                # Prepare enhanced cell data
                enhanced_cells = []
                for cell in result['cells_data']:
                    # Basic cell data
                    cell_row = {
                        'Cell_ID': cell.get('id', ''),
                        'Area_Pixels': cell.get('area', 0),
                        'Intensity': cell.get('intensity', 0),
                        'Center_X': cell.get('center', [0, 0])[0],
                        'Center_Y': cell.get('center', [0, 0])[1],
                        'Method': cell.get('method', 'unknown')
                    }
                    
                    # Add biomass data if available
                    if 'biomass' in cell:
                        biomass = cell['biomass']
                        cell_row.update({
                            'Fresh_Weight_mg': biomass.get('fresh_weight_mg', 0),
                            'Dry_Weight_mg': biomass.get('dry_weight_mg', 0),
                            'Chlorophyll_mg': biomass.get('chlorophyll_content_mg', 0),
                            'Volume_Microns³': biomass.get('volume_microns_cubed', 0)
                        })
                    
                    # Add color analysis if available
                    if 'color_analysis' in cell:
                        color = cell['color_analysis']
                        cell_row.update({
                            'Green_Ratio': color.get('green_ratio', 0),
                            'Green_Intensity': color.get('green_intensity', 0)
                        })
                    
                    # Add health status if available
                    if 'health_status' in cell:
                        health = cell['health_status']
                        cell_row.update({
                            'Health_Status': health.get('status', 'unknown'),
                            'Health_Score': health.get('score', 0)
                        })
                    
                    enhanced_cells.append(cell_row)
                
                # Create DataFrame with enhanced data
                df = pd.DataFrame(enhanced_cells)
                output_file = f"results/{analysis_id}_enhanced_cells.csv"
                df.to_csv(output_file, index=False)
                
                return send_file(output_file, 
                                as_attachment=True, 
                                download_name=f"wolffia_analysis_{analysis_id}.csv")
            else:
                return jsonify({'error': 'No cell data to export'}), 400
        
        else:
            return jsonify({'error': 'Unsupported format'}), 400
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# LEGACY ENDPOINTS FOR COMPATIBILITY

@app.route('/api/export/<analysis_id>/biomass_report')
def export_biomass_report(analysis_id):
    """Export comprehensive biomass analysis report"""
    try:
        if analysis_id not in analysis_results:
            return jsonify({'error': 'Analysis not found'}), 404
        
        analysis = analysis_results[analysis_id]
        if analysis['status'] != 'completed':
            return jsonify({'error': 'Analysis not completed'}), 400
        
        result = analysis['result']
        
        # Create comprehensive biomass report
        report_data = create_biomass_report(result)
        
        # Generate PDF or detailed CSV
        output_file = f"results/{analysis_id}_biomass_report.csv"
        
        # Create detailed CSV with all biomass metrics
        import pandas as pd
        
        # Prepare data for CSV
        csv_data = []
        
        if 'cells_data' in result and result['cells_data']:
            for cell in result['cells_data']:
                row = {
                    'Cell_ID': cell.get('id', ''),
                    'Area_Pixels': cell.get('area', 0),
                    'Area_Microns_Squared': cell.get('area_microns_squared', 0),
                    'Intensity': cell.get('intensity', 0),
                    'Center_X': cell.get('center', [0, 0])[0],
                    'Center_Y': cell.get('center', [0, 0])[1],
                }
                
                # Add biomass data if available
                if 'biomass' in cell:
                    biomass = cell['biomass']
                    row.update({
                        'Fresh_Weight_mg': biomass.get('fresh_weight_mg', 0),
                        'Dry_Weight_mg': biomass.get('dry_weight_mg', 0),
                        'Volume_Microns_Cubed': biomass.get('volume_microns_cubed', 0),
                        'Chlorophyll_Content_mg': biomass.get('chlorophyll_content_mg', 0),
                        'Biomass_Density': biomass.get('biomass_density_mg_per_micron_sq', 0)
                    })
                
                # Add color analysis if available
                if 'color_analysis' in cell:
                    color = cell['color_analysis']
                    row.update({
                        'Green_Ratio': color.get('green_ratio', 0),
                        'Green_Intensity': color.get('green_intensity', 0),
                        'Estimated_Chlorophyll_Level': color.get('estimated_chlorophyll_level', 'unknown')
                    })
                
                # Add health status if available
                if 'health_status' in cell:
                    health = cell['health_status']
                    row.update({
                        'Health_Status': health.get('status', 'unknown'),
                        'Health_Score': health.get('score', 0)
                    })
                
                csv_data.append(row)
        
        # Create DataFrame and save
        if csv_data:
            df = pd.DataFrame(csv_data)
            df.to_csv(output_file, index=False)
            
            return send_file(output_file, 
                           as_attachment=True, 
                           download_name=f"biomass_analysis_{analysis_id}.csv",
                           mimetype='text/csv')
        else:
            return jsonify({'error': 'No cell data available for biomass report'}), 400
    
    except Exception as e:
        print(f"❌ Biomass report export failed: {e}")
        return jsonify({'error': str(e)}), 500

def create_biomass_report(result):
    """Create comprehensive biomass analysis report data"""
    try:
        report = {
            'summary': {
                'total_cells': result.get('cells_detected', 0),
                'analysis_timestamp': result.get('timestamp', ''),
                'processing_time': result.get('processing_time', 0)
            }
        }
        
        # Extract enhanced metrics if available
        if 'enhanced_metrics' in result:
            enhanced = result['enhanced_metrics']
            
            if 'biomass_analysis' in enhanced:
                report['biomass_summary'] = enhanced['biomass_analysis']
            
            if 'color_analysis' in enhanced:
                report['color_summary'] = enhanced['color_analysis']
            
            if 'population_analysis' in enhanced:
                report['population_summary'] = enhanced['population_analysis']
        
        return report
        
    except Exception as e:
        print(f"⚠️ Report creation failed: {e}")
        return {}
    
    
@app.route('/api/set_parameters', methods=['POST'])
def set_parameters():
    """Set analysis parameters (legacy compatibility)"""
    try:
        params = request.json
        # Update analyzer parameters if needed
        if 'pixel_to_micron_ratio' in params:
            analyzer.pixel_to_micron_ratio = params['pixel_to_micron_ratio']
        if 'chlorophyll_threshold' in params:
            analyzer.chlorophyll_threshold = params['chlorophyll_threshold']
        
        return jsonify({'success': True, 'message': 'Parameters updated'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/get_parameters')
def get_parameters():
    """Get current analysis parameters (legacy compatibility)"""
    return jsonify({
        'pixel_to_micron_ratio': analyzer.pixel_to_micron_ratio,
        'chlorophyll_threshold': analyzer.chlorophyll_threshold,
        'wolffia_params': analyzer.wolffia_params
    })

# Background processing function for analysis
def process_core_analysis():
    """Process analysis requests from the queue"""
    while True:
        try:
            if not analysis_queue.empty():
                task = analysis_queue.get()
                
                analysis_id = task['id']
                image_path = task['image_path']
                use_tophat = task.get('use_tophat', False)
                
                # Update progress
                analysis_progress[analysis_id] = 10
                
                # Run analysis
                result = analyzer.analyze_image(image_path, use_tophat=use_tophat)
                
                # Update progress
                analysis_progress[analysis_id] = 100
                
                # Store result
                analysis_results[analysis_id] = result
                
                analysis_queue.task_done()
        except Exception as e:
            print(f"Background processing error: {e}")

if __name__ == '__main__':
    print("🚀 Starting BIOIMAGIN Enhanced Web Server...")
    print("✅ Wolffia Analyzer loaded with optimized features")
    print("🌐 Server running at http://localhost:5000")
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True
    )