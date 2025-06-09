# web_integration.py - Enhanced Flask backend for ML-powered Wolffia analysis

import base64
import json
import mimetypes
import os
import queue
import threading
import uuid
import warnings
import zipfile
from datetime import datetime
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd

# Add after your existing imports
from complete_advanced_integration import (
    UltimateWolffiaAnalyzer,
)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
import cv2
from flask import Flask, Response, jsonify, render_template, request, send_file
from flask_cors import CORS
from PIL import Image
from werkzeug.utils import safe_join, secure_filename

try:
    import numexpr
    NUMEXPR_AVAILABLE = True
except ImportError:
    NUMEXPR_AVAILABLE = False

try:
    import cupy
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

try:
    import cellpose
    CELLPOSE_AVAILABLE = True
except ImportError:
    CELLPOSE_AVAILABLE = False


# Try to import from the improved professional module first, then fallbacks
try:
    from bioimaging_professional_improved import WolffiaAnalyzer
    PIPELINE_TYPE = "professional_improved"
    print("✅ Using Improved Professional Bioinformatics Pipeline")
except ImportError:
    try:
        from bioimaging_professional import WolffiaAnalyzer
        PIPELINE_TYPE = "professional"
        print("✅ Using Professional Bioinformatics Pipeline")
    except ImportError:
        try:
            from bioimaging import WolffiaAnalyzer
            PIPELINE_TYPE = "legacy"
            print("⚠️ Using Legacy Bioimaging Module")
        except ImportError:
            print("❌ No bioimaging module found")
            raise ImportError("Cannot import WolffiaAnalyzer")

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

# Global analyzer instance with professional enhancement
print("🔬 Initializing Professional Wolffia Analyzer...")
analyzer = WolffiaAnalyzer(pixel_to_micron_ratio=0.5, chlorophyll_threshold=0.6)
print("✅ Professional analyzer ready for web integration")

# Analysis queue for real-time updates
analysis_queue = queue.Queue()
analysis_results = {}
analysis_progress = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def convert_professional_result_to_json(result, timestamp):
    """Convert professional analysis result to JSON-serializable format - FIXED VERSION"""
    try:
        if not result or not result.get('success'):
            return None
        
        # FIXED: Better data extraction and conversion
        cells_data = result.get('cells', [])
        summary_data = result.get('summary', {})
        
        # Ensure cells is a list and properly formatted
        if isinstance(cells_data, pd.DataFrame):
            cells_data = cells_data.to_dict('records')
        elif not isinstance(cells_data, list):
            cells_data = []
        
        # FIXED: Create proper summary with actual cell counts
        if cells_data and len(cells_data) > 0:
            # Recalculate summary from actual cell data
            total_cells = len(cells_data)
            
            # Count green cells
            green_cells = sum(1 for cell in cells_data 
                            if cell.get('is_green_cell', False) or 
                               cell.get('green_cell', False))
            
            # Calculate total biomass
            total_biomass = sum(float(cell.get('biomass_estimate_ug', 0) or 
                                    cell.get('biomass_ug', 0) or 0) 
                              for cell in cells_data)
            
            # Calculate average area
            areas = [float(cell.get('area_microns_sq', 0) or 
                          cell.get('area_microns', 0) or 0) 
                    for cell in cells_data]
            avg_area = sum(areas) / len(areas) if areas else 0
            
            # FIXED: Create corrected summary
            corrected_summary = {
                'total_cells_detected': total_cells,
                'total_cells': total_cells,
                'total_green_cells': green_cells,
                'total_biomass': total_biomass,
                'average_area': avg_area,
                'average_cell_area': avg_area
            }
            
            print(f"🔧 Data conversion: {total_cells} cells → {green_cells} green → {total_biomass:.3f} μg")
        else:
            corrected_summary = {
                'total_cells_detected': 0,
                'total_cells': 0,
                'total_green_cells': 0,
                'total_biomass': 0,
                'average_area': 0,
                'average_cell_area': 0
            }
        
        json_result = {
            'timestamp': timestamp,
            'image_path': result.get('image_path', ''),
            'success': result.get('success', False),
            'cells': convert_to_json_serializable(cells_data),  # FIXED: Ensure conversion
            'summary': corrected_summary,  # FIXED: Use corrected summary
            'quality': {
                'overall_quality': result.get('quality', {}).get('overall_quality', 0),
                'status': result.get('quality', {}).get('status', 'Unknown'),
                'restoration_quality': result.get('quality', {}).get('restoration_quality', 0),
                'segmentation_quality': result.get('quality', {}).get('segmentation_quality', 0),
                'feature_quality': result.get('quality', {}).get('feature_quality', 0)
            },
            'technical_details': result.get('technical_details', {}),
            'visualizations': result.get('visualizations', {})
        }
        
        return json_result
        
    except Exception as e:
        print(f"❌ Error converting professional result to JSON: {e}")
        import traceback
        traceback.print_exc()
        return None

def convert_to_json_serializable(obj):
    """Convert any object to JSON-serializable format - ENHANCED VERSION"""
    try:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(convert_to_json_serializable(item) for item in obj)
        elif pd.isna(obj) or obj is None:  # ADDED: Handle pandas NaN
            return None
        elif isinstance(obj, (bool, np.bool_)):
            return bool(obj)
        elif hasattr(obj, 'item'):  # For numpy scalars
            return obj.item()
        elif isinstance(obj, pd.DataFrame):  # ADDED: Handle DataFrames
            return obj.to_dict('records')
        else:
            return obj
    except:
        return str(obj)  # fallback to string representation

def create_enhanced_summary(results, all_cells):
    """Create enhanced summary from analysis results"""
    try:
        if not results or not all_cells:
            return {
                'total_cells_detected': 0,
                'total_green_cells': 0,
                'average_area': 0,
                'total_biomass': 0,
                'images_processed': len(results) if results else 0
            }
        
        # Convert to DataFrame for easier processing
        cells_df = pd.DataFrame(all_cells)
        
        summary = {
            'total_cells_detected': len(all_cells),
            'images_processed': len(results),
            'average_area': float(cells_df.get('area_microns_sq', pd.Series([0])).mean()),
            'total_biomass': float(cells_df.get('biomass_estimate_ug', pd.Series([0])).sum()),
            'total_green_cells': int(cells_df.get('is_green_cell', pd.Series([False])).sum()),
            'area_statistics': {
                'min': float(cells_df.get('area_microns_sq', pd.Series([0])).min()),
                'max': float(cells_df.get('area_microns_sq', pd.Series([0])).max()),
                'std': float(cells_df.get('area_microns_sq', pd.Series([0])).std())
            },
            'quality_statistics': {
                'average_quality': float(np.mean([r.get('quality', {}).get('overall_quality', 0) for r in results if r.get('success')])),
                'successful_analyses': sum(1 for r in results if r.get('success'))
            }
        }
        
        return summary
        
    except Exception as e:
        print(f"⚠️ Error creating enhanced summary: {e}")
        return {
            'total_cells_detected': len(all_cells) if all_cells else 0,
            'images_processed': len(results) if results else 0,
            'error': str(e)
        }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle single or multiple file uploads with progress tracking"""
    try:
        if 'files' not in request.files:
            return jsonify({'error': 'No files provided'}), 400
        
        files = request.files.getlist('files')
        if not files or files[0].filename == '':
            return jsonify({'error': 'No files selected'}), 400
        
        uploaded_files = []
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                unique_filename = f"{timestamp}_{filename}"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
                file.save(filepath)
                uploaded_files.append({
                    'filename': unique_filename,
                    'original_name': filename,
                    'path': filepath
                })
        
        if not uploaded_files:
            return jsonify({'error': 'No valid files uploaded'}), 400
        
        # Generate analysis ID
        analysis_id = str(uuid.uuid4())
        
        # Initialize progress tracking
        analysis_progress[analysis_id] = {
            'status': 'queued',
            'progress': 0,
            'current_step': 'Initializing',
            'total_images': len(uploaded_files),
            'processed_images': 0
        }
        
        # Start analysis in background
        thread = threading.Thread(target=process_core_analysis, 
                                args=(analysis_id, uploaded_files))
        thread.start()
        
        return jsonify({
            'success': True,
            'analysis_id': analysis_id,
            'files': uploaded_files,
            'message': f'{len(uploaded_files)} file(s) uploaded successfully'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Professional analysis status endpoint with enhanced error handling
@app.route('/api/analyze/<analysis_id>', methods=['GET'])
def get_analysis_status(analysis_id):
    """Get analysis status with progress updates"""
    try:
        if analysis_id in analysis_results:
            # Ensure the result is JSON-serializable
            result = analysis_results[analysis_id]
            return jsonify(convert_to_json_serializable(result))
        elif analysis_id in analysis_progress:
            return jsonify(analysis_progress[analysis_id]), 202
        else:
            return jsonify({'status': 'not_found', 'error': 'Analysis ID not found'}), 404
    except Exception as e:
        import traceback
        error_message = str(e)
        error_traceback = traceback.format_exc()
        print(f"Error in get_analysis_status: {error_message}")
        print(f"Traceback: {error_traceback}")
        
        return jsonify({
            'status': 'error',
            'error': error_message,
            'error_details': error_traceback
        }), 500

@app.route('/api/get_parameters', methods=['GET'])
def get_parameters():
    """Get current analyzer parameters"""
    try:
        return jsonify({
            'success': True,
            'parameters': analyzer.get_current_parameters(),
            'pipeline_type': PIPELINE_TYPE,
            'enhanced_features': PIPELINE_TYPE == "professional_improved"
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/export/<analysis_id>/<format>', methods=['GET'])
def export_results(analysis_id, format):
    """Export analysis results in various formats with ML insights"""
    try:
        if analysis_id not in analysis_results:
            return jsonify({'error': 'Analysis not found'}), 404
        
        results = analysis_results[analysis_id]['results']
        
        if format == 'csv':
            # Export comprehensive cell data as CSV
            all_cells = []
            for result in results:
                if 'cells' in result:
                    cells_df = pd.DataFrame(result['cells'])
                    cells_df['image'] = result['image_path']
                    cells_df['timestamp'] = result['timestamp']
                    all_cells.append(cells_df)
            
            if all_cells:
                combined_df = pd.concat(all_cells, ignore_index=True)
                csv_path = os.path.join(app.config['RESULTS_FOLDER'], f'{analysis_id}_cells.csv')
                combined_df.to_csv(csv_path, index=False)
                return send_file(csv_path, as_attachment=True, 
                               download_name='wolffia_ml_analysis_cells.csv')
            
        elif format == 'json':
            # Export complete results with ML insights as JSON
            json_path = os.path.join(app.config['RESULTS_FOLDER'], f'{analysis_id}_results.json')
            with open(json_path, 'w') as f:
                json.dump(results, f, indent=2)
            return send_file(json_path, as_attachment=True, 
                           download_name='wolffia_ml_analysis_results.json')
            
            
        elif format == 'zip':
            # Create comprehensive ZIP with all results
            memory_file = BytesIO()
            with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
                # Add JSON results
                zf.writestr('analysis_results.json', json.dumps(results, indent=2))
                
                # Add CSV for each image
                for i, result in enumerate(results):
                    if 'cells' in result:
                        cells_df = pd.DataFrame(result['cells'])
                        csv_content = cells_df.to_csv(index=False)
                        zf.writestr(f'cells_{result["timestamp"]}.csv', csv_content)
                
                # Add comprehensive report
                report = generate_enhanced_analysis_report(results)
                zf.writestr('comprehensive_report.txt', report)
                
                # Add population dynamics if available
                if results and 'population_dynamics' in results[-1]:
                    pop_dyn = json.dumps(results[-1]['population_dynamics'], indent=2)
                    zf.writestr('population_dynamics.json', pop_dyn)
            
            memory_file.seek(0)
            return send_file(memory_file, as_attachment=True, 
                           download_name=f'wolffia_ml_analysis_{analysis_id}.zip')
            
        else:
            return jsonify({'error': 'Invalid format'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500




@app.route('/api/calibrate', methods=['POST'])
def calibrate():
    """Calibrate pixel to micron ratio with validation"""
    try:
        data = request.json
        known_distance_pixels = float(data.get('known_distance_pixels', 1))
        known_distance_microns = float(data.get('known_distance_microns', 1))
        
        if known_distance_pixels <= 0 or known_distance_microns <= 0:
            return jsonify({'error': 'Invalid calibration values'}), 400
        
        ratio = known_distance_microns / known_distance_pixels
        analyzer.pixel_to_micron = ratio
        
        # Validate calibration with expected cell sizes
        expected_min = analyzer.wolffia_params['min_area_microns']
        expected_max = analyzer.wolffia_params['max_area_microns']
        
        # Calculate pixel area range
        pixel_area_min = expected_min / (ratio ** 2)
        pixel_area_max = expected_max / (ratio ** 2)
        
        return jsonify({
            'success': True,
            'pixel_to_micron_ratio': ratio,
            'message': f'Calibration successful. New ratio: {ratio:.4f}',
            'expected_pixel_range': {
                'min_area_pixels': pixel_area_min,
                'max_area_pixels': pixel_area_max
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500




# Replace analyzer initialization
analyzer = UltimateWolffiaAnalyzer(pixel_to_micron_ratio=0.5, chlorophyll_threshold=0.6)


# Add ultimate analysis endpoint
@app.route('/api/analyze_ultimate', methods=['POST'])
def analyze_ultimate():
    """Ultimate analysis endpoint using the best pipeline"""
    try:
        if 'files' not in request.files:
            return jsonify({'error': 'No files provided'}), 400
        
        files = request.files.getlist('files')
        config = request.form.to_dict()
        
        # Extract configuration
        analysis_mode = config.get('analysis_mode', 'auto')
        confidence_threshold = float(config.get('confidence_threshold', 0.05))
        detailed_analysis = config.get('detailed_analysis', 'true').lower() == 'true'
        force_advanced = config.get('force_advanced', 'false').lower() == 'true'
        generate_visualizations = config.get('generate_visualizations', 'true').lower() == 'true'
        
        # Save uploaded files
        uploaded_files = []
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                unique_filename = f"{timestamp}_{filename}"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
                file.save(filepath)
                uploaded_files.append(filepath)
        
        # Analyze each file
        results = []
        for filepath in uploaded_files:
            result = analyzer.analyze_image_ultimate(
                filepath,
                mode=analysis_mode,
                confidence_threshold=confidence_threshold,
                force_advanced=force_advanced,
                detailed_analysis=detailed_analysis,
                save_visualizations=generate_visualizations
            )
            
            if result:
                json_result = convert_to_json_serializable(result)
                results.append(json_result)
        
        # Generate summary
        summary = generate_ultimate_summary(results)
        
        return jsonify({
            'success': True,
            'results': results,
            'summary': summary,
            'analysis_type': 'ultimate_multi_pipeline',
            'configuration': config,
            'total_images': len(uploaded_files)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/system_status', methods=['GET'])
def get_system_status():
    """Get comprehensive system status"""
    try:
        health = analyzer.health_check()
        params = analyzer.get_current_parameters()
        
        return jsonify({
            'health': health,
            'parameters': params,
            'gpu_available': params['gpu_info']['cuda_available'],
            'advanced_pipeline_available': params['available_pipelines']['advanced_8_stage'],
            'system_capabilities': params['system_capabilities'],
            'performance_stats': params['performance_stats']
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def generate_ultimate_summary(results):
    """Generate summary for ultimate results"""
    if not results:
        return {}
    
    total_cells = sum(len(r.get('cells', [])) for r in results)
    total_green = sum(r.get('summary', {}).get('total_green_cells', 0) for r in results)
    total_biomass = sum(r.get('summary', {}).get('total_biomass', 0) for r in results)
    
    # Advanced metrics
    all_confidences = []
    viable_cells = 0
    
    for result in results:
        for cell in result.get('cells', []):
            all_confidences.append(cell.get('confidence', 0))
            if cell.get('viability_estimate') == 'high':
                viable_cells += 1
    
    return {
        'total_images_analyzed': len(results),
        'total_cells_detected': total_cells,
        'total_green_cells': total_green,
        'total_biomass': total_biomass,
        'green_cell_percentage': (total_green / total_cells * 100) if total_cells else 0,
        'average_cells_per_image': total_cells / len(results) if results else 0,
        'confidence_metrics': {
            'mean_confidence': np.mean(all_confidences) if all_confidences else 0,
            'high_confidence_cells': sum(1 for c in all_confidences if c > 0.8)
        },
        'quality_indicators': {
            'viable_cells': viable_cells,
            'viability_rate': (viable_cells / total_cells * 100) if total_cells else 0
        },
        'pipeline_statistics': {
            'advanced_pipeline_used': sum(1 for r in results if r.get('advanced_pipeline', False)),
            'professional_pipeline_used': sum(1 for r in results if not r.get('advanced_pipeline', True)),
            'hybrid_analysis_used': sum(1 for r in results if r.get('hybrid_analysis', False))
        }
    }

# Add this route to web_integration.py to serve uploaded images

# Add these imports at the top if not present
import mimetypes

from werkzeug.utils import safe_join


# Add this route to serve uploaded files
@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    """Serve uploaded files for tophat training and visualization"""
    try:
        safe_filename = secure_filename(filename)
        file_path = safe_join(app.config['UPLOAD_FOLDER'], safe_filename)
        
        print(f"🔍 Serving file request: {filename}")
        print(f"📁 Looking for file at: {file_path}")
        
        if file_path and os.path.exists(file_path):
            mime_type, _ = mimetypes.guess_type(file_path)
            if mime_type is None:
                mime_type = 'application/octet-stream'
            
            print(f"✅ File found, serving: {file_path}")
            return send_file(file_path, mimetype=mime_type)
        else:
            print(f"❌ File not found: {file_path}")
            if os.path.exists(app.config['UPLOAD_FOLDER']):
                available_files = os.listdir(app.config['UPLOAD_FOLDER'])
                print(f"📋 Available files: {available_files}")
            return jsonify({'error': 'File not found', 'requested': filename}), 404
            
    except Exception as e:
        print(f"❌ Error serving file {filename}: {e}")
        return jsonify({'error': str(e)}), 500

# Add debug endpoint
@app.route('/api/debug/uploads', methods=['GET'])
def debug_uploads():
    """Debug endpoint to check uploaded files"""
    try:
        upload_dir = app.config['UPLOAD_FOLDER']
        if os.path.exists(upload_dir):
            files = []
            for filename in os.listdir(upload_dir):
                file_path = os.path.join(upload_dir, filename)
                if os.path.isfile(file_path):
                    files.append({
                        'filename': filename,
                        'size': os.path.getsize(file_path),
                        'modified': datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()
                    })
            
            return jsonify({
                'upload_directory': upload_dir,
                'files': files,
                'total_files': len(files)
            })
        else:
            return jsonify({'error': 'Upload directory does not exist', 'path': upload_dir}), 404
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def process_core_analysis(analysis_id, uploaded_files):
    """Enhanced core analysis processing with better error handling and data validation"""
    try:
        # Update status
        analysis_progress[analysis_id] = {
            'status': 'processing',
            'progress': 10,
            'current_step': 'Starting Professional Analysis',
            'total_images': len(uploaded_files),
            'processed_images': 0
        }

        file_paths = [f['path'] for f in uploaded_files]
        timestamps = [f"T{i}" for i in range(len(file_paths))]

        results = []
        
        for i, (path, timestamp) in enumerate(zip(file_paths, timestamps)):
            analysis_progress[analysis_id]['current_step'] = f'Analyzing Image {i+1}/{len(file_paths)}'
            analysis_progress[analysis_id]['progress'] = 10 + (80 * i / len(file_paths))
            
            print(f"\n{'='*60}")
            print(f"🧬 Professional Analysis {i+1}/{len(file_paths)}")
            print(f"📁 Path: {path}")
            print(f"⏰ Timestamp: {timestamp}")
            print(f"{'='*60}")
            
            # Use the appropriate analyzer method based on pipeline type
            if PIPELINE_TYPE == "professional_improved":
                # Use improved professional pipeline
                result = analyzer.analyze_image_professional(
                    path, 
                    restoration_mode='auto',
                    segmentation_model='auto',
                    diameter=analyzer.wolffia_params.get('diameter'),
                    flow_threshold=analyzer.wolffia_params.get('flow_threshold'),
                    learn_from_analysis=True,
                    save_visualizations=True
                )
            elif hasattr(analyzer, 'analyze_image_professional'):
                # Use original professional pipeline
                result = analyzer.analyze_image_professional(path, model='auto', restoration='auto')
            else:
                # Fallback to legacy method
                result = analyzer.analyze_single_image_enhanced(path, timestamp, save_visualization=True)
            
            if result and result.get('success'):
                # ENHANCED: Better data validation and conversion
                print(f"📊 Raw result summary: {result.get('summary', {})}")
                print(f"🔬 Raw cells count: {len(result.get('cells', []))}")
                
                # Convert to JSON-serializable format for web
                json_result = convert_professional_result_to_json(result, timestamp)
                if json_result:
                    # VALIDATION: Check if data was converted correctly
                    converted_cells = len(json_result.get('cells', []))
                    converted_summary = json_result.get('summary', {})
                    
                    print(f"✅ Converted result - Cells: {converted_cells}, Summary: {converted_summary.get('total_cells_detected', 0)}")
                    
                    # ENSURE: Summary matches actual cell data
                    if converted_cells > 0 and converted_summary.get('total_cells_detected', 0) == 0:
                        print("🔧 Fixing summary mismatch...")
                        json_result['summary']['total_cells_detected'] = converted_cells
                        json_result['summary']['total_cells'] = converted_cells
                    
                    results.append(json_result)
                    print(f"✅ Professional analysis {i+1} completed successfully")
                else:
                    print(f"❌ Failed to serialize results for image {i+1}")
            else:
                error_msg = result.get('error', 'Unknown error') if result else 'Analysis failed'
                print(f"❌ Professional analysis {i+1} failed: {error_msg}")
                
            analysis_progress[analysis_id]['processed_images'] = i + 1

        if results:
            # Generate comprehensive summary with VALIDATION
            print(f"\n🔍 Generating summary from {len(results)} results...")
            
            for i, result in enumerate(results):
                cell_count = len(result.get('cells', []))
                summary_count = result.get('summary', {}).get('total_cells_detected', 0)
                print(f"Result {i+1}: {cell_count} cells in data, {summary_count} in summary")
            
            summary = generate_professional_summary(results)
            print(f"📈 Generated summary: {summary}")
            
            # Store results with validation
            analysis_results[analysis_id] = {
                'status': 'completed',
                'message': 'Professional analysis completed successfully',
                'results': results,
                'summary': summary,
                'timestamp': datetime.now().isoformat(),
                'analysis_type': 'professional_time_series' if len(results) > 1 else 'professional_single_image',
                'total_images_processed': len(results),
                'pipeline_version': 'Professional_v1.0_Fixed'
            }
            
            # Final validation
            final_cell_count = summary.get('total_cells_detected', 0)
            print(f"🎯 FINAL VALIDATION: {final_cell_count} total cells across all images")
            
            # Update progress
            analysis_progress[analysis_id] = {
                'status': 'completed',
                'progress': 100,
                'current_step': 'Analysis complete',
                'total_images': len(uploaded_files),
                'processed_images': len(results)
            }
            
            print(f"\n{'='*60}")
            print(f"🎉 PROFESSIONAL ANALYSIS COMPLETE")
            print(f"🖼️ Images: {len(results)}")
            print(f"🔬 Total cells: {final_cell_count}")
            print(f"🌱 Green cells: {summary.get('total_green_cells', 0)}")
            print(f"⚖️ Biomass: {summary.get('total_biomass', 0):.2f} μg")
            print(f"📈 Quality: {summary.get('average_quality', 0):.3f}")
            print(f"{'='*60}\n")
            
        else:
            analysis_results[analysis_id] = {
                'status': 'completed_with_warnings',
                'message': 'Analysis completed but no valid results generated',
                'results': [],
                'timestamp': datetime.now().isoformat(),
                'pipeline_version': 'Professional_v1.0_Fixed'
            }
            
            analysis_progress[analysis_id]['status'] = 'completed_with_warnings'
            analysis_progress[analysis_id]['current_step'] = 'Analysis complete - no valid results'

    except Exception as e:
        import traceback
        error_message = str(e)
        error_traceback = traceback.format_exc()
        
        print(f"\n{'='*60}")
        print(f"❌ CRITICAL ERROR in Professional Analysis")
        print(f"Error: {error_message}")
        print(f"{'='*60}")
        print(error_traceback)
        
        analysis_results[analysis_id] = {
            'status': 'failed',
            'message': 'Professional analysis failed with critical error',
            'error': error_message,
            'timestamp': datetime.now().isoformat(),
            'pipeline_version': 'Professional_v1.0_Fixed'
        }
        
        analysis_progress[analysis_id] = {
            'status': 'failed',
            'current_step': f'Error: {error_message}',
            'progress': 0
        }

# Enhanced web integration with GPU status reporting

@app.route('/api/health_check', methods=['GET'])
def health_check():
    """Enhanced health check with GPU status and performance monitoring"""
    try:
        params = analyzer.get_current_parameters()
        
        # Get GPU information
        gpu_info = params.get('gpu_info', {})
        gpu_status = params.get('gpu_status', {})
        performance_stats = params.get('performance_stats', {})
        
        # Get tophat training status if available
        tophat_status = {}
        if hasattr(analyzer, 'get_tophat_training_status'):
            tophat_status = analyzer.get_tophat_training_status()
        
        # Determine overall GPU status
        gpu_enabled = gpu_info.get('cuda_available', False)
        gpu_name = gpu_info.get('gpu_name', 'None')
        gpu_memory = gpu_info.get('gpu_memory', 0)
        
        # GPU status message
        if gpu_enabled:
            gpu_status_msg = f"✅ {gpu_name} ({gpu_memory:.1f} GB)"
        else:
            gpu_status_msg = "❌ CPU Only (Install CUDA + GPU PyTorch for acceleration)"
        
        # Performance recommendations
        recommendations = []
        if not gpu_enabled:
            recommendations.append("Install CUDA and GPU-enabled PyTorch for 5-10x speed improvement")
        
        if not NUMEXPR_AVAILABLE:
            recommendations.append("Install numexpr for faster NumPy operations: pip install numexpr")
        
        if gpu_enabled and not CUPY_AVAILABLE:
            recommendations.append("Install CuPy for GPU array operations: pip install cupy-cuda11x")
        
        # Memory usage info
        memory_info = {}
        if gpu_enabled and 'memory_usage' in gpu_status:
            memory_usage = gpu_status['memory_usage']
            memory_info = {
                'gpu_allocated_gb': memory_usage.get('allocated', 0),
                'gpu_cached_gb': memory_usage.get('cached', 0),
                'gpu_peak_gb': memory_usage.get('max_allocated', 0)
            }
        
        return jsonify({
            'status': 'healthy',
            'version': '2.0-Professional-GPU',
            'pipeline': 'Professional_Bioinformatics_GPU_v1.0',
            'pipeline_type': PIPELINE_TYPE,
            'analyzer_status': 'ready',
            'gpu_acceleration': {
                'enabled': gpu_enabled,
                'status': gpu_status_msg,
                'device_name': gpu_name,
                'memory_gb': gpu_memory,
                'memory_usage': memory_info,
                'cuda_version': torch.version.cuda if (TORCH_AVAILABLE and gpu_enabled) else None,
                'pytorch_version': torch.__version__ if TORCH_AVAILABLE else 'Not installed'
            },
            'performance': {
                'total_analyses': performance_stats.get('total_analyses', 0),
                'avg_processing_time': performance_stats.get('avg_processing_time', 0),
                'gpu_memory_peak_mb': performance_stats.get('gpu_memory_peak', 0)
            },
            'modules_status': params.get('engines_status', params.get('modules_status', {})),
            'optimization_libraries': {
                'numexpr': '✅ Available' if NUMEXPR_AVAILABLE else '❌ Not installed',
                'cupy': '✅ Available' if CUPY_AVAILABLE else '❌ Not installed (GPU only)',
                'tensorrt': '❓ Optional'
            },
            'recommendations': recommendations,
            'tophat_training': tophat_status,
            'parameters': {
                'pixel_to_micron': analyzer.pixel_to_micron,
                'chlorophyll_threshold': analyzer.chlorophyll_threshold,
                'use_gpu': analyzer.wolffia_params.get('use_gpu', False),
                'gpu_batch_size': analyzer.wolffia_params.get('batch_size', 1)
            }
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'version': '2.0-Professional-GPU',
            'error': str(e),
            'gpu_acceleration': {
                'enabled': False,
                'status': '❌ Error checking GPU status'
            }
        }), 500

@app.route('/api/gpu_status', methods=['GET'])
def get_gpu_status():
    """Detailed GPU status endpoint"""
    try:
        if hasattr(analyzer, 'segmentation_engine') and analyzer.segmentation_engine:
            gpu_status = analyzer.segmentation_engine.get_gpu_status()
            
            # Add real-time GPU monitoring
            if gpu_status['gpu_available']:
                try:
                    import subprocess
                    result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu', '--format=csv,noheader,nounits'], 
                                          capture_output=True, text=True, timeout=5)
                    if result.returncode == 0:
                        gpu_util, mem_used, mem_total, temp = result.stdout.strip().split(', ')
                        gpu_status['real_time'] = {
                            'utilization_percent': int(gpu_util),
                            'memory_used_mb': int(mem_used),
                            'memory_total_mb': int(mem_total),
                            'temperature_c': int(temp)
                        }
                except:
                    gpu_status['real_time'] = {'error': 'nvidia-smi not available'}
            
            return jsonify({
                'success': True,
                'gpu_status': gpu_status,
                'optimization_tips': [
                    "Use batch processing for multiple images",
                    "Monitor GPU memory usage to prevent OOM errors",
                    "Clear GPU cache between analyses if needed",
                    "Consider mixed precision for faster inference"
                ]
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Segmentation engine not available'
            }), 500
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/optimize_gpu', methods=['POST'])
def optimize_gpu_settings():
    """Optimize GPU settings based on current hardware"""
    try:
        data = request.json or {}
        
        if not hasattr(analyzer, 'gpu_info') or not analyzer.gpu_info['cuda_available']:
            return jsonify({
                'success': False,
                'error': 'GPU not available'
            }), 400
        
        # Get current GPU memory usage
        try:
            current_memory = torch.cuda.memory_allocated(0) / 1e9 if TORCH_AVAILABLE else 0  # GB
            total_memory = analyzer.gpu_info['gpu_memory']
            
            # Optimize batch size based on available memory
            available_memory = total_memory - current_memory
            
            if available_memory > 6:
                recommended_batch = 8
                recommended_diameter = 35
            elif available_memory > 3:
                recommended_batch = 4
                recommended_diameter = 30
            else:
                recommended_batch = 2
                recommended_diameter = 25
            
            # Apply optimizations
            analyzer.wolffia_params['batch_size'] = recommended_batch
            analyzer.wolffia_params['diameter'] = recommended_diameter
            
            # Clear GPU cache
            if TORCH_AVAILABLE:
                torch.cuda.empty_cache()
            
            return jsonify({
                'success': True,
                'optimizations_applied': {
                    'batch_size': recommended_batch,
                    'diameter': recommended_diameter,
                    'cache_cleared': True
                },
                'gpu_memory': {
                    'total_gb': total_memory,
                    'available_gb': available_memory,
                    'current_usage_gb': current_memory
                },
                'message': f'GPU settings optimized for {analyzer.gpu_info["gpu_name"]}'
            })
            
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Failed to optimize GPU settings: {str(e)}'
            }), 500
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Enhanced parameter setting with GPU optimization
@app.route('/api/set_parameters', methods=['POST'])
def set_parameters():
    """Update analyzer parameters with GPU-aware optimization"""
    try:
        data = request.json
        
        # Basic parameters
        if 'pixel_to_micron' in data:
            analyzer.pixel_to_micron = float(data['pixel_to_micron'])
        
        if 'chlorophyll_threshold' in data:
            analyzer.chlorophyll_threshold = float(data['chlorophyll_threshold'])
        
        # Enhanced professional parameters
        if hasattr(analyzer, 'set_parameters'):
            analyzer.set_parameters(**data)
        else:
            # Fallback for legacy parameters
            if 'min_area_microns' in data:
                analyzer.wolffia_params['min_area_microns'] = float(data['min_area_microns'])
            
            if 'max_area_microns' in data:
                analyzer.wolffia_params['max_area_microns'] = float(data['max_area_microns'])
            
            if 'expected_circularity' in data:
                analyzer.wolffia_params['expected_circularity'] = float(data['expected_circularity'])
        
        # GPU-specific parameters
        if 'diameter' in data and hasattr(analyzer, 'wolffia_params'):
            diameter = float(data['diameter'])
            # Validate diameter for GPU memory constraints
            if analyzer.wolffia_params.get('use_gpu', False):
                max_diameter = analyzer.gpu_info['recommended_settings'].get('diameter', 30) + 10
                if diameter > max_diameter:
                    diameter = max_diameter
                    print(f"⚠️ Diameter limited to {max_diameter} for GPU memory constraints")
            
            analyzer.wolffia_params['diameter'] = diameter
        
        if 'flow_threshold' in data and hasattr(analyzer, 'wolffia_params'):
            analyzer.wolffia_params['flow_threshold'] = float(data['flow_threshold'])
        
        if 'batch_size' in data and hasattr(analyzer, 'wolffia_params'):
            batch_size = int(data['batch_size'])
            # Validate batch size for GPU memory
            if analyzer.wolffia_params.get('use_gpu', False):
                max_batch = analyzer.gpu_info['recommended_settings'].get('batch_size', 4)
                if batch_size > max_batch:
                    batch_size = max_batch
                    print(f"⚠️ Batch size limited to {max_batch} for GPU memory constraints")
            
            analyzer.wolffia_params['batch_size'] = batch_size
        
        # Clear GPU cache after parameter changes
        if analyzer.wolffia_params.get('use_gpu', False):
            try:
                if TORCH_AVAILABLE:
                    torch.cuda.empty_cache()
            except:
                pass
        
        return jsonify({
            'success': True,
            'parameters': analyzer.get_current_parameters(),
            'pipeline_type': PIPELINE_TYPE,
            'gpu_optimized': analyzer.wolffia_params.get('use_gpu', False),
            'message': 'Parameters updated successfully'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Add GPU installation check helper
@app.route('/api/check_gpu_setup', methods=['GET'])
def check_gpu_setup():
    """Check GPU setup and provide installation guidance"""
    try:
        setup_status = {
            'cuda_available': torch.cuda.is_available() if TORCH_AVAILABLE else False,
            'pytorch_version': torch.__version__ if TORCH_AVAILABLE else 'Not installed',
            'torch_cuda_version': torch.version.cuda if TORCH_AVAILABLE else None,
            'gpu_count': torch.cuda.device_count() if (TORCH_AVAILABLE and torch.cuda.is_available()) else 0,
            'recommendations': []
        }
        
        if TORCH_AVAILABLE and torch.cuda.is_available():
            setup_status['gpu_name'] = torch.cuda.get_device_name(0)
            setup_status['gpu_memory'] = torch.cuda.get_device_properties(0).total_memory / 1e9
        elif not TORCH_AVAILABLE:
            setup_status['recommendations'].extend([
                "1. Install PyTorch: pip install torch torchvision torchaudio",
                "2. For GPU support: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118",
                "3. Install NVIDIA GPU drivers if not already installed",
                "4. Install CUDA Toolkit from developer.nvidia.com/cuda-downloads"
            ])
        else:
            setup_status['recommendations'].extend([
                "1. Install NVIDIA GPU drivers from nvidia.com",
                "2. Install CUDA Toolkit from developer.nvidia.com/cuda-downloads",
                "3. Uninstall current PyTorch: pip uninstall torch torchvision torchaudio",
                "4. Install GPU PyTorch: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118",
                "5. Install performance packages: pip install numexpr cupy-cuda11x"
            ])
        
        # Check for performance libraries
        setup_status['libraries'] = {
            'numexpr': NUMEXPR_AVAILABLE,
            'cupy': CUPY_AVAILABLE,
            'cellpose_version': getattr(cellpose, '__version__', 'unknown') if CELLPOSE_AVAILABLE else 'not_installed'
        }
        
        return jsonify(setup_status)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/tophat/save_annotations', methods=['POST'])
def save_tophat_annotations():
    """Save user annotations for tophat training"""
    try:
        data = request.json
        analysis_id = data.get('analysis_id')
        image_path = data.get('image_path')
        annotations = data.get('annotations', [])
        
        if not analysis_id or not image_path or not annotations:
            return jsonify({'error': 'Missing required fields'}), 400
        
        # Save annotations using analyzer
        if hasattr(analyzer, 'save_tophat_annotations'):
            success = analyzer.save_tophat_annotations(analysis_id, image_path, annotations)
            
            if success:
                # Get updated training status
                training_status = analyzer.get_tophat_training_status()
                
                return jsonify({
                    'success': True,
                    'message': f'Saved {len(annotations)} annotations',
                    'training_status': training_status
                })
            else:
                return jsonify({'error': 'Failed to save annotations'}), 500
        else:
            return jsonify({'error': 'Tophat training not available'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/tophat/apply_training/<analysis_id>', methods=['POST'])
def apply_tophat_training(analysis_id):
    """Apply tophat training to improve analysis results"""
    try:
        if analysis_id not in analysis_results:
            return jsonify({'error': 'Analysis not found'}), 404
        
        # Get the analysis results
        result_data = analysis_results[analysis_id]
        if not result_data.get('results'):
            return jsonify({'error': 'No analysis results found'}), 400
        
        # Apply tophat training to each image in the results
        improved_results = []
        
        for result in result_data['results']:
            if not result.get('success'):
                improved_results.append(result)
                continue
            
            image_path = result.get('image_path')
            if not image_path or not os.path.exists(image_path):
                improved_results.append(result)
                continue
            
            # Apply tophat training if available
            if hasattr(analyzer, 'apply_tophat_training'):
                # Get original segmentation (we need to re-run analysis with tophat)
                if PIPELINE_TYPE == "professional_improved":
                    tophat_result = analyzer.analyze_image_professional(
                        image_path,
                        restoration_mode='auto',
                        segmentation_model='auto',
                        learn_from_analysis=False,  # Don't learn during tophat application
                        save_visualizations=True
                    )
                    
                    if tophat_result and tophat_result.get('success'):
                        # Convert to JSON serializable and add tophat flag
                        tophat_result['tophat_applied'] = True
                        tophat_result['original_cells'] = len(result.get('cells', []))
                        tophat_result['improved_cells'] = len(tophat_result.get('cells', []))
                        
                        json_result = convert_professional_result_to_json(tophat_result, result.get('timestamp'))
                        if json_result:
                            improved_results.append(json_result)
                        else:
                            improved_results.append(result)
                    else:
                        improved_results.append(result)
                else:
                    # Fallback for non-improved pipelines
                    improved_results.append(result)
            else:
                improved_results.append(result)
        
        # Update the stored results
        result_data['results'] = improved_results
        result_data['tophat_applied'] = True
        result_data['tophat_timestamp'] = datetime.now().isoformat()
        
        # Recalculate summary
        all_cells = []
        for result in improved_results:
            if result.get('success') and result.get('cells'):
                all_cells.extend(result['cells'])
        
        summary = create_enhanced_summary(improved_results, all_cells)
        result_data['summary'] = summary
        
        return jsonify({
            'success': True,
            'message': f'Tophat training applied to {len(improved_results)} images',
            'results': convert_to_json_serializable(result_data),
            'training_status': analyzer.get_tophat_training_status() if hasattr(analyzer, 'get_tophat_training_status') else {}
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/tophat/status', methods=['GET'])
def get_tophat_status():
    """Get tophat training system status"""
    try:
        if hasattr(analyzer, 'get_tophat_training_status'):
            status = analyzer.get_tophat_training_status()
            return jsonify({
                'success': True,
                'status': status
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Tophat training not available'
            })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
    
    # Enhanced web endpoints for neural network training
# Add these endpoints to web_integration.py

@app.route('/api/tophat/force_retrain', methods=['POST'])
def force_retrain_tophat():
    """Force retrain the tophat models with all available annotations"""
    try:
        if not hasattr(analyzer, 'learning_engine') or not analyzer.learning_engine:
            return jsonify({'error': 'Learning engine not available'}), 400
        
        # Check if we have enough annotations
        num_annotations = len(analyzer.learning_engine.user_annotations)
        if num_annotations < 5:
            return jsonify({
                'error': f'Need at least 5 annotation sessions, have {num_annotations}',
                'current_annotations': num_annotations,
                'min_required': 5
            }), 400
        
        print(f"🚀 Force retraining initiated with {num_annotations} annotation sessions...")
        
        # Trigger force retraining
        success = analyzer.learning_engine.force_retrain()
        
        if success:
            # Get updated training status
            training_status = analyzer.get_tophat_training_status()
            
            return jsonify({
                'success': True,
                'message': f'Successfully retrained models with {num_annotations} annotation sessions',
                'training_status': training_status,
                'models_created': training_status.get('models_trained', {}),
                'best_model': training_status.get('best_model', 'unknown'),
                'recommendations': training_status.get('recommended_action', '')
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Retraining failed - check server logs for details',
                'num_annotations': num_annotations
            }), 500
            
    except Exception as e:
        print(f"❌ Error during force retrain: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e),
            'error_details': traceback.format_exc()
        }), 500

@app.route('/api/tophat/enhanced_status', methods=['GET'])
def get_enhanced_tophat_status():
    """Get enhanced tophat training system status with neural network info"""
    try:
        if hasattr(analyzer, 'get_tophat_training_status'):
            status = analyzer.get_tophat_training_status()
            
            # Add system capabilities info
            status['system_capabilities'] = {
                'tensorflow_available': False,
                'sklearn_available': True,
                'can_train_neural_networks': True,
                'supported_models': ['tensorflow_neural_network', 'simple_neural_network', 'random_forest']
            }
            
            # Check TensorFlow availability
            try:
                import tensorflow as tf
                status['system_capabilities']['tensorflow_available'] = True
                status['system_capabilities']['tensorflow_version'] = tf.__version__
            except ImportError:
                pass
            
            # Add annotation details
            if hasattr(analyzer.learning_engine, 'user_annotations'):
                annotations = analyzer.learning_engine.user_annotations
                annotation_stats = {
                    'total_sessions': len(annotations),
                    'total_examples': sum(len(ann.get('annotations', [])) for ann in annotations),
                    'positive_examples': 0,
                    'negative_examples': 0
                }
                
                for session in annotations:
                    for ann in session.get('annotations', []):
                        if ann.get('type') == 'positive':
                            annotation_stats['positive_examples'] += 1
                        elif ann.get('type') == 'negative':
                            annotation_stats['negative_examples'] += 1
                
                status['annotation_stats'] = annotation_stats
            
            return jsonify({
                'success': True,
                'status': status
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Enhanced tophat training not available'
            })
    except Exception as e:
        return jsonify({
            'success': False, 
            'error': str(e)
        }), 500

@app.route('/api/tophat/apply_neural_training/<analysis_id>', methods=['POST'])
def apply_neural_tophat_training(analysis_id):
    """Apply neural network-enhanced tophat training to improve analysis results"""
    try:
        if analysis_id not in analysis_results:
            return jsonify({'error': 'Analysis not found'}), 404
        
        # Get the analysis results
        result_data = analysis_results[analysis_id]
        if not result_data.get('results'):
            return jsonify({'error': 'No analysis results found'}), 400
        
        # Check if any neural network models are available
        training_status = analyzer.get_tophat_training_status()
        if not training_status.get('any_model_trained', False):
            return jsonify({
                'error': 'No trained models available',
                'training_status': training_status,
                'suggestion': 'Create annotations and train models first'
            }), 400
        
        print(f"🧠 Applying neural network-enhanced tophat training to analysis {analysis_id}")
        print(f"🎯 Best available model: {training_status.get('best_model', 'unknown')}")
        
        # Apply enhanced tophat training to each image in the results
        improved_results = []
        improvement_stats = {
            'total_images': 0,
            'improved_images': 0,
            'cells_before': 0,
            'cells_after': 0,
            'confidence_scores': []
        }
        
        for result in result_data['results']:
            improvement_stats['total_images'] += 1
            
            if not result.get('success'):
                improved_results.append(result)
                continue
            
            image_path = result.get('image_path')
            if not image_path or not os.path.exists(image_path):
                improved_results.append(result)
                continue
            
            # Apply enhanced tophat training
            if PIPELINE_TYPE == "professional_improved":
                enhanced_result = analyzer.analyze_image_professional(
                    image_path,
                    restoration_mode='auto',
                    segmentation_model='auto',
                    learn_from_analysis=False,  # Don't learn during enhancement
                    save_visualizations=True
                )
                
                if enhanced_result and enhanced_result.get('success'):
                    # Apply the neural network model
                    original_labels = enhanced_result.get('segmentation', {}).get('labels')
                    if original_labels is not None:
                        # Load image for tophat application
                        image = cv2.imread(image_path)
                        if image is not None:
                            improved_labels, confidence_scores = analyzer.learning_engine.apply_tophat_model(
                                image, original_labels
                            )
                            
                            # Update the segmentation results
                            if np.max(improved_labels) > 0:
                                enhanced_result['segmentation']['labels'] = improved_labels
                                enhanced_result['tophat_enhanced'] = True
                                enhanced_result['tophat_confidence'] = confidence_scores
                                enhanced_result['improvement_info'] = {
                                    'original_cells': len(result.get('cells', [])),
                                    'enhanced_cells': len(enhanced_result.get('cells', [])),
                                    'model_used': training_status.get('best_model', 'unknown'),
                                    'confidence_scores': confidence_scores
                                }
                                
                                improvement_stats['improved_images'] += 1
                                improvement_stats['cells_before'] += len(result.get('cells', []))
                                improvement_stats['cells_after'] += len(enhanced_result.get('cells', []))
                                improvement_stats['confidence_scores'].extend(confidence_scores)
                    
                    # Convert to JSON serializable and add enhancement flags
                    json_result = convert_professional_result_to_json(enhanced_result, result.get('timestamp'))
                    if json_result:
                        json_result['neural_enhanced'] = True
                        improved_results.append(json_result)
                    else:
                        improved_results.append(result)
                else:
                    improved_results.append(result)
            else:
                # Fallback for non-improved pipelines
                improved_results.append(result)
        
        # Update the stored results
        result_data['results'] = improved_results
        result_data['neural_enhanced'] = True
        result_data['enhancement_timestamp'] = datetime.now().isoformat()
        result_data['enhancement_stats'] = improvement_stats
        
        # Recalculate summary
        all_cells = []
        for result in improved_results:
            if result.get('success') and result.get('cells'):
                all_cells.extend(result['cells'])
        
        summary = create_enhanced_summary(improved_results, all_cells)
        result_data['summary'] = summary
        
        return jsonify({
            'success': True,
            'message': f'Neural network enhancement applied to {improvement_stats["total_images"]} images',
            'results': convert_to_json_serializable(result_data),
            'enhancement_stats': improvement_stats,
            'training_status': training_status,
            'model_used': training_status.get('best_model', 'unknown')
        })
        
    except Exception as e:
        print(f"❌ Error applying neural tophat training: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e),
            'error_details': traceback.format_exc()
        }), 500

@app.route('/api/tophat/model_info', methods=['GET'])
def get_tophat_model_info():
    """Get detailed information about available tophat models"""
    try:
        if not hasattr(analyzer, 'learning_engine') or not analyzer.learning_engine:
            return jsonify({'error': 'Learning engine not available'}), 400
        
        learning_dir = analyzer.learning_engine.learning_dir
        model_info = {
            'tensorflow_model': {
                'available': (learning_dir / 'neural_segmentation_model').exists(),
                'path': str(learning_dir / 'neural_segmentation_model'),
                'type': 'deep_neural_network',
                'description': 'TensorFlow/Keras deep neural network with multiple layers and dropout'
            },
            'simple_nn_model': {
                'available': (learning_dir / 'simple_neural_network.pkl').exists(),
                'path': str(learning_dir / 'simple_neural_network.pkl'),
                'type': 'simple_neural_network',
                'description': 'NumPy-based neural network with ReLU and sigmoid activations'
            },
            'random_forest_model': {
                'available': (learning_dir / 'tophat_classifier.pkl').exists(),
                'path': str(learning_dir / 'tophat_classifier.pkl'),
                'type': 'ensemble_classifier',
                'description': 'Random Forest ensemble classifier with 100 trees'
            }
        }
        
        # Add file sizes and modification times
        for model_name, info in model_info.items():
            if info['available']:
                try:
                    model_path = Path(info['path'])
                    if model_path.exists():
                        if model_path.is_dir():  # TensorFlow model directory
                            total_size = sum(f.stat().st_size for f in model_path.rglob('*') if f.is_file())
                            info['size_mb'] = round(total_size / 1024 / 1024, 2)
                        else:  # Pickle file
                            info['size_mb'] = round(model_path.stat().st_size / 1024 / 1024, 2)
                        info['last_modified'] = datetime.fromtimestamp(model_path.stat().st_mtime).isoformat()
                except Exception as e:
                    info['error'] = str(e)
        
        return jsonify({
            'success': True,
            'models': model_info,
            'total_models': sum(1 for m in model_info.values() if m['available']),
            'learning_directory': str(learning_dir)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


def process_enhanced_analysis(analysis_id, uploaded_files):
    """Process analysis with ML enhancement and progress updates"""
    try:
        # Update status
        analysis_progress[analysis_id] = {
            'status': 'processing',
            'progress': 10,
            'current_step': 'Initializing ML models',
            'total_images': len(uploaded_files),
            'processed_images': 0
        }

        file_paths = [f['path'] for f in uploaded_files]
        timestamps = [f"T{i}" for i in range(len(file_paths))]

        if len(file_paths) == 1:
            # Single image analysis
            analysis_progress[analysis_id]['current_step'] = 'Analyzing image with ML'
            analysis_progress[analysis_id]['progress'] = 50
            
            result = analyzer.analyze_single_image(file_paths[0], timestamps[0], save_visualization=True)
            
            if result:
                # Convert to JSON-serializable format
                json_result = analyzer.export_enhanced_results(result)
                results = [json_result] if json_result else []
            else:
                results = []
                
            analysis_progress[analysis_id]['processed_images'] = 1
            analysis_progress[analysis_id]['progress'] = 90
            
        else:
            # Multiple image analysis - keep both raw and JSON results
            raw_results = []  # For population dynamics analysis
            json_results = []  # For final output
            
            for i, (path, timestamp) in enumerate(zip(file_paths, timestamps)):
                analysis_progress[analysis_id]['current_step'] = f'Analyzing image {i+1}/{len(file_paths)}'
                analysis_progress[analysis_id]['progress'] = 10 + (70 * i / len(file_paths))
                
                print(f"\n{'='*60}")
                print(f"Processing image {i+1}/{len(file_paths)}: {timestamp}")
                print(f"Path: {path}")
                print(f"{'='*60}")
                
                # Analyze single image
                # Check if the method exists and use enhanced version
                if hasattr(analyzer, 'analyze_single_image_enhanced'):
                    result = analyzer.analyze_single_image_enhanced(path, timestamp, save_visualization=True)
                else:
                    result = analyzer.analyze_single_image(path, timestamp, save_visualization=True)
                    
                    # Add spectral analysis if not included
                    if result and 'spectral_data' not in result:
                        try:
                            # Get the preprocessed data and labels
                            preprocessed = analyzer.advanced_preprocess_image(path)
                            labels, _ = analyzer.multi_method_segmentation(preprocessed)
                            
                            if np.max(labels) > 0:
                                spectral_df, spectral_viz = analyzer.analyze_chlorophyll_spectrum(
                                    preprocessed['original'], 
                                    labels
                                )
                                
                                result['spectral_data'] = spectral_df
                                result['spectral_analysis'] = spectral_df.to_dict('records')
                                result['spectral_visualization'] = spectral_viz
                                result['spectral_report'] = analyzer.generate_spectral_report(spectral_df)
                        except Exception as e:
                            print(f"Error adding spectral analysis: {str(e)}")                
                if result:
                    # Keep raw result for population dynamics
                    raw_results.append(result)
                    
                    # Convert to JSON-serializable format for storage
                    json_result = analyzer.export_enhanced_results(result)
                    if json_result:
                        json_results.append(json_result)
                    else:
                        print(f"⚠️ Failed to export results for image {i+1}")
                else:
                    print(f"⚠️ Failed to analyze image {i+1}")
                    
                analysis_progress[analysis_id]['processed_images'] = i + 1
            
            # Population dynamics analysis using raw results
            if len(raw_results) > 1:
                analysis_progress[analysis_id]['current_step'] = 'Analyzing population dynamics'
                analysis_progress[analysis_id]['progress'] = 85
                
                print("\n" + "="*60)
                print("POPULATION DYNAMICS ANALYSIS")
                print("="*60)
                
                try:
                    # Use raw results for population dynamics
                    population_dynamics = analyzer.analyze_population_dynamics(raw_results)
                    
                    if population_dynamics:
                        # Convert population dynamics to JSON-serializable
                        pop_dyn_json = convert_to_json_serializable(population_dynamics)
                        if json_results:  # Add to last result
                            json_results[-1]['population_dynamics'] = pop_dyn_json
                        print("✅ Population dynamics analysis complete")
                    else:
                        print("⚠️ Population dynamics analysis returned no results")
                        
                except Exception as e:
                    print(f"❌ Error in population dynamics: {str(e)}")
                    import traceback
                    traceback.print_exc()
                
                # Time series visualizations using raw results
                analysis_progress[analysis_id]['current_step'] = 'Creating time series visualizations'
                analysis_progress[analysis_id]['progress'] = 90
                
                try:
                    time_series_viz = analyzer.create_enhanced_time_series_plots(raw_results, return_base64=True)
                    if time_series_viz and json_results:
                        json_results[-1]['time_series_visualizations'] = time_series_viz
                        print("Time series visualizations created")
                except Exception as e:
                    print(f"❌ Error creating time series visualizations: {str(e)}")
                
                # Parameter optimization using raw results
                analysis_progress[analysis_id]['current_step'] = 'Optimizing parameters'
                analysis_progress[analysis_id]['progress'] = 95
                
                try:
                    optimized_params = analyzer.optimize_parameters(raw_results)
                    if optimized_params and json_results:
                        json_results[-1]['optimized_parameters'] = optimized_params
                        print("✅ Parameters optimized")
                except Exception as e:
                    print(f"❌ Error optimizing parameters: {str(e)}")
            
            results = json_results

        if results:
            # Final processing
            analysis_progress[analysis_id]['current_step'] = 'Finalizing results'
            analysis_progress[analysis_id]['progress'] = 98
            
            # Generate comprehensive summary
            summary = generate_enhanced_summary(results)
            
            # Store results
            analysis_results[analysis_id] = {
                'status': 'completed',
                'message': 'Analysis completed successfully with ML enhancement',
                'results': results,
                'summary': summary,
                'timestamp': datetime.now().isoformat(),
                'analysis_type': 'time_series' if len(results) > 1 else 'single_image',
                'total_images_processed': len(results)
            }
            
            # Update progress
            analysis_progress[analysis_id] = {
                'status': 'completed',
                'progress': 100,
                'current_step': 'Analysis complete',
                'total_images': len(uploaded_files),
                'processed_images': len(results)
            }
            
            print(f"\n{'='*60}")
            print(f"ANALYSIS COMPLETE")
            print(f"Total images processed: {len(results)}")
            print(f"Total cells detected: {summary.get('total_cells_detected', 0)}")
            print(f"{'='*60}\n")
            
        else:
            analysis_results[analysis_id] = {
                'status': 'failed',
                'message': 'No results generated',
                'results': [],
                'timestamp': datetime.now().isoformat()
            }
            
            analysis_progress[analysis_id]['status'] = 'failed'
            analysis_progress[analysis_id]['current_step'] = 'Analysis failed - no results'

    except Exception as e:
        import traceback
        error_message = str(e)
        error_traceback = traceback.format_exc()
        
        print(f"\n{'='*60}")
        print(f"CRITICAL ERROR in process_enhanced_analysis")
        print(f"Error: {error_message}")
        print(f"{'='*60}")
        print(error_traceback)
        print(f"{'='*60}\n")
        
        analysis_results[analysis_id] = {
            'status': 'failed',
            'message': 'Analysis failed',
            'error': error_message,
            'error_details': error_traceback,
            'timestamp': datetime.now().isoformat()
        }
        
        analysis_progress[analysis_id] = {
            'status': 'failed',
            'current_step': f'Error: {error_message}',
            'progress': 0
        }


# Add this helper function to convert numpy/pandas objects to JSON-serializable format
def convert_to_json_serializable(obj):
    """Convert numpy/pandas objects to JSON-serializable format"""
    import numpy as np
    import pandas as pd
    
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict('records')
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    elif isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_json_serializable(item) for item in obj)
    elif pd.isna(obj) or obj is None:
        return None
    elif isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    elif hasattr(obj, 'item'):  # For numpy scalars
        return obj.item()
    else:
        return obj


def generate_enhanced_summary(results):
    """Generate core analysis summary"""
    try:
        if not results:
            return {}
        
        total_images = len(results)
        total_cells = sum(r.get('total_cells', 0) for r in results)

        # Core aggregates
        all_biomass = []
        green_cell_counts = []
        all_chlorophyll = []
        all_areas = []

        for result in results:
            if 'summary' in result:
                summary = result['summary']
                all_biomass.append(summary.get('total_biomass_ug', 0))
                green_cell_counts.append(summary.get('green_cell_count', 0))
                all_chlorophyll.append(summary.get('mean_chlorophyll_intensity', 0))
                all_areas.append(summary.get('mean_cell_area_microns', 0))

        # Growth metrics for time series
        growth_metrics = {}
        if len(results) > 1:
            biomass_change = all_biomass[-1] - all_biomass[0]
            biomass_growth_rate = (biomass_change / all_biomass[0] * 100) if all_biomass[0] > 0 else 0

            cell_change = results[-1].get('total_cells', 0) - results[0].get('total_cells', 0)
            cell_growth_rate = (cell_change / results[0].get('total_cells', 1) * 100)

            growth_metrics = {
                'biomass_change': biomass_change,
                'biomass_growth_rate': biomass_growth_rate,
                'cell_count_change': cell_change,
                'cell_growth_rate': cell_growth_rate
            }

        return {
            'total_images_analyzed': total_images,
            'total_cells_detected': total_cells,
            'total_green_cells': sum(green_cell_counts),
            'total_biomass': sum(all_biomass),
            'average_biomass_per_image': sum(all_biomass) / len(all_biomass) if all_biomass else 0,
            'average_chlorophyll': np.mean(all_chlorophyll) if all_chlorophyll else 0,
            'average_cell_area': np.mean(all_areas) if all_areas else 0,
            'growth_metrics': growth_metrics if growth_metrics else None
        }

    except Exception as e:
        print(f"Error generating summary: {str(e)}")
        return {}


def generate_enhanced_analysis_report(results):
    """Generate comprehensive analysis report"""
    try:
        report = []
        report.append("BIOIMAGIN - WOLFFIA ANALYSIS REPORT")
        report.append("=" * 50)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total Images Analyzed: {len(results)}")
        report.append("")
        
        # Overall summary
        summary = generate_enhanced_summary(results)
        report.append("OVERALL SUMMARY")
        report.append("-" * 30)
        report.append(f"Total Cells Detected: {summary.get('total_cells_detected', 0)}")
        report.append(f"Total Green Cells: {summary.get('total_green_cells', 0)}")
        report.append(f"Total Biomass (μg): {summary.get('total_biomass', 0):.2f}")
        report.append(f"Average Cell Area: {summary.get('average_cell_area', 0):.2f} μm²")
        report.append(f"Average Chlorophyll: {summary.get('average_chlorophyll', 0):.3f}")
        report.append("")
        
        # Growth analysis for time series
        if len(results) > 1 and 'growth_metrics' in summary:
            report.append("GROWTH ANALYSIS")
            report.append("-" * 30)
            gm = summary['growth_metrics']
            report.append(f"Cell Count Change: {gm.get('cell_count_change', 0)}")
            report.append(f"Cell Growth Rate: {gm.get('cell_growth_rate', 0):.1f}%")
            report.append(f"Biomass Change: {gm.get('biomass_change', 0):.2f} μg")
            report.append(f"Biomass Growth Rate: {gm.get('biomass_growth_rate', 0):.1f}%")
            report.append("")
        
        # Individual image results
        report.append("INDIVIDUAL IMAGE RESULTS")
        report.append("-" * 30)
        
        for i, result in enumerate(results):
            report.append(f"\nImage {i+1}: {result.get('timestamp', 'N/A')}")
            report.append(f"  File: {os.path.basename(result.get('image_path', 'N/A'))}")
            report.append(f"  Total Cells: {result.get('total_cells', 0)}")
            
            if 'summary' in result:
                s = result['summary']
                report.append(f"  Green Cells: {s.get('green_cell_count', 0)}")
                report.append(f"  Mean Cell Area: {s.get('mean_cell_area_microns', 0):.2f} μm²")
                report.append(f"  Mean Chlorophyll: {s.get('mean_chlorophyll_intensity', 0):.3f}")
                report.append(f"  Total Biomass: {s.get('total_biomass_ug', 0):.2f} μg")
        
        # Analysis parameters
        report.append("")
        report.append("ANALYSIS PARAMETERS")
        report.append("-" * 30)
        report.append(f"Pixel to Micron Ratio: {analyzer.pixel_to_micron}")
        report.append(f"Chlorophyll Threshold: {analyzer.chlorophyll_threshold}")
        
        return "\n".join(report)
        
    except Exception as e:
        return f"Error generating report: {str(e)}"


# Helper functions for professional pipeline integration

def convert_professional_result_to_json(result, timestamp):
    """Convert professional analysis result to JSON-serializable format"""
    try:
        json_result = {
            'timestamp': timestamp,
            'image_path': result.get('image_path', ''),
            'success': result.get('success', False),
            'cells': result.get('cells', []),
            'summary': result.get('summary', {}),
            'quality': {
                'overall_quality': result.get('quality', {}).get('overall_quality', 0),
                'status': result.get('quality', {}).get('status', 'Unknown'),
                'restoration_quality': result.get('quality', {}).get('restoration_quality', 0),
                'segmentation_quality': result.get('quality', {}).get('segmentation_quality', 0),
                'feature_quality': result.get('quality', {}).get('feature_quality', 0)
            },
            'technical_details': result.get('technical_details', {}),
            'visualizations': result.get('visualizations', {})
        }
        
        # Ensure all values are JSON serializable
        return convert_to_json_serializable(json_result)
        
    except Exception as e:
        print(f"Error converting professional result to JSON: {e}")
        return None

def generate_time_series_summary(results):
    """Generate time series summary for multiple images"""
    try:
        successful_results = [r for r in results if r.get('success', False)]
        
        if len(successful_results) < 2:
            return None
            
        # Calculate growth metrics
        cell_counts = [r.get('summary', {}).get('total_cells_detected', 0) for r in successful_results]
        biomass_values = [r.get('summary', {}).get('total_biomass', 0) for r in successful_results]
        
        time_series = {
            'total_timepoints': len(successful_results),
            'cell_count_trend': {
                'initial': cell_counts[0] if cell_counts else 0,
                'final': cell_counts[-1] if cell_counts else 0,
                'change': cell_counts[-1] - cell_counts[0] if len(cell_counts) >= 2 else 0,
                'percent_change': ((cell_counts[-1] - cell_counts[0]) / cell_counts[0] * 100) if cell_counts and cell_counts[0] > 0 else 0
            },
            'biomass_trend': {
                'initial': biomass_values[0] if biomass_values else 0,
                'final': biomass_values[-1] if biomass_values else 0,
                'change': biomass_values[-1] - biomass_values[0] if len(biomass_values) >= 2 else 0,
                'percent_change': ((biomass_values[-1] - biomass_values[0]) / biomass_values[0] * 100) if biomass_values and biomass_values[0] > 0 else 0
            }
        }
        
        return time_series
        
    except Exception as e:
        print(f"Error generating time series summary: {e}")
        return None

def generate_professional_summary(results):
    """Generate comprehensive summary for professional analysis - FIXED VERSION"""
    try:
        successful_results = [r for r in results if r.get('success', False)]
        
        if not successful_results:
            return {
                'total_cells_detected': 0,
                'total_green_cells': 0,
                'total_biomass': 0,
                'average_quality': 0,
                'analysis_success_rate': 0
            }
        
        # FIXED: Aggregate metrics across all images properly
        total_cells = 0
        total_green_cells = 0
        total_biomass = 0
        all_areas = []
        
        for result in successful_results:
            # FIXED: Handle different summary key names
            summary = result.get('summary', {})
            
            # Get cell count
            cell_count = (summary.get('total_cells_detected', 0) or 
                         summary.get('total_cells', 0) or 
                         len(result.get('cells', [])))
            total_cells += cell_count
            
            # Get green cell count
            green_count = (summary.get('total_green_cells', 0) or
                          summary.get('green_cells', 0))
            if green_count == 0 and result.get('cells'):
                # Fallback: count from actual cell data
                green_count = sum(1 for cell in result['cells'] 
                                if cell.get('is_green_cell', False))
            total_green_cells += green_count
            
            # Get biomass
            biomass = (summary.get('total_biomass', 0) or
                      summary.get('biomass', 0))
            if biomass == 0 and result.get('cells'):
                # Fallback: sum from actual cell data
                biomass = sum(cell.get('biomass_estimate_ug', 0) or 0 
                            for cell in result['cells'])
            total_biomass += biomass
            
            # Collect areas for average calculation
            if result.get('cells'):
                for cell in result['cells']:
                    area = (cell.get('area_microns_sq', 0) or 
                           cell.get('area_microns', 0) or 0)
                    if area > 0:
                        all_areas.append(area)
        
        # Calculate average area
        average_cell_area = sum(all_areas) / len(all_areas) if all_areas else 0
        
        # Quality metrics
        quality_scores = [r.get('quality', {}).get('overall_quality', 0) for r in successful_results]
        average_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        
        success_rate = len(successful_results) / len(results) if results else 0
        
        summary = {
            'total_images_analyzed': len(results),
            'successful_analyses': len(successful_results),
            'analysis_success_rate': success_rate,
            'total_cells_detected': total_cells,
            'total_green_cells': total_green_cells,
            'total_biomass': total_biomass,
            'average_cell_area': average_cell_area,
            'average_quality': average_quality,
            'quality_distribution': {
                'excellent': sum(1 for q in quality_scores if q > 0.85),
                'good': sum(1 for q in quality_scores if 0.7 < q <= 0.85),
                'acceptable': sum(1 for q in quality_scores if 0.5 < q <= 0.7),
                'needs_improvement': sum(1 for q in quality_scores if q <= 0.5)
            }
        }
        
        print(f"📈 FINAL SUMMARY: {total_cells} cells, {total_green_cells} green, {total_biomass:.3f} μg")
        return summary
        
    except Exception as e:
        print(f"❌ Error generating professional summary: {e}")
        import traceback
        traceback.print_exc()
        return {
            'error': str(e),
            'total_cells_detected': 0,
            'total_green_cells': 0,
            'total_biomass': 0
        }

if __name__ == '__main__':
    print("=" * 60)
    print("PROFESSIONAL BIOIMAGIN Web Server v2.0")
    print("=" * 60)
    print("🧬 Professional Bioinformatics Pipeline")
    print("🔬 CellPose + SimpleITK Integration")
    print("📊 Advanced Quality Assessment")
    print("=" * 60)
    print("Starting Flask server...")
    print("Core features enabled:")
    print("✓ Cell detection and counting")
    print("✓ Size and biomass measurements")
    print("✓ Green cell identification")
    print("✓ Time series tracking")
    print("✓ Comprehensive reporting")
    print("=" * 50)
    
    app.run(debug=True, port=5000, threaded=True)