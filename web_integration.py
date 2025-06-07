# web_integration.py - Enhanced Flask backend for ML-powered Wolffia analysis

import base64
import json
import os
import queue
import threading
import uuid
import zipfile
from datetime import datetime
from io import BytesIO

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

# Global analyzer instance with ML enhancement
analyzer = WolffiaAnalyzer(pixel_to_micron_ratio=0.5, chlorophyll_threshold=0.6)

# Analysis queue for real-time updates
analysis_queue = queue.Queue()
analysis_results = {}
analysis_progress = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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

# Update the get_analysis_status endpoint to handle errors better:
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

@app.route('/api/set_parameters', methods=['POST'])
def set_parameters():
    """Update analyzer parameters including ML settings"""
    try:
        data = request.json
        
        if 'pixel_to_micron' in data:
            analyzer.pixel_to_micron = float(data['pixel_to_micron'])
        
        if 'chlorophyll_threshold' in data:
            analyzer.chlorophyll_threshold = float(data['chlorophyll_threshold'])
        
        if 'min_area_microns' in data:
            analyzer.wolffia_params['min_area_microns'] = float(data['min_area_microns'])
        
        if 'max_area_microns' in data:
            analyzer.wolffia_params['max_area_microns'] = float(data['max_area_microns'])
        
        if 'expected_circularity' in data:
            analyzer.wolffia_params['expected_circularity'] = float(data['expected_circularity'])
        
        return jsonify({
            'success': True,
            'parameters': analyzer.get_current_parameters()
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



@app.route('/api/health_check', methods=['GET'])
def health_check():
    """API health check"""
    return jsonify({
        'status': 'healthy',
        'version': '2.0-Core',
        'analyzer_status': 'ready',
        'parameters': {
            'pixel_to_micron': analyzer.pixel_to_micron,
            'chlorophyll_threshold': analyzer.chlorophyll_threshold
        }
    })

# In web_integration.py, update the process_enhanced_analysis function:

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

def process_core_analysis(analysis_id, uploaded_files):
    """Process core analysis with essential functionality only"""
    try:
        # Update status
        analysis_progress[analysis_id] = {
            'status': 'processing',
            'progress': 10,
            'current_step': 'Starting analysis',
            'total_images': len(uploaded_files),
            'processed_images': 0
        }

        file_paths = [f['path'] for f in uploaded_files]
        timestamps = [f"T{i}" for i in range(len(file_paths))]

        results = []
        
        for i, (path, timestamp) in enumerate(zip(file_paths, timestamps)):
            analysis_progress[analysis_id]['current_step'] = f'Analyzing image {i+1}/{len(file_paths)}'
            analysis_progress[analysis_id]['progress'] = 10 + (80 * i / len(file_paths))
            
            # Core analysis using bioimaging module
            result = analyze_uploaded_image(path, analyzer)
            
            if result:
                # Convert to JSON-serializable format
                json_result = convert_to_json_serializable(result)
                if json_result:
                    json_result['timestamp'] = timestamp
                    json_result['image_path'] = path
                    results.append(json_result)
                    
            analysis_progress[analysis_id]['processed_images'] = i + 1

        if results:
            # Generate summary
            summary = generate_enhanced_summary(results)
            
            # Store results
            analysis_results[analysis_id] = {
                'status': 'completed',
                'message': 'Analysis completed successfully',
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
        
        print(f"Error in process_core_analysis: {error_message}")
        print(f"Traceback: {error_traceback}")
        
        analysis_results[analysis_id] = {
            'status': 'failed',
            'message': 'Analysis failed',
            'error': error_message,
            'timestamp': datetime.now().isoformat()
        }
        
        analysis_progress[analysis_id] = {
            'status': 'failed',
            'current_step': f'Error: {error_message}',
            'progress': 0
        }

if __name__ == '__main__':
    print("=" * 50)
    print("BIOIMAGIN Web Server v2.0 - Core")
    print("=" * 50)
    print("Starting Flask server...")
    print("Core features enabled:")
    print("✓ Cell detection and counting")
    print("✓ Size and biomass measurements")
    print("✓ Green cell identification")
    print("✓ Time series tracking")
    print("✓ Comprehensive reporting")
    print("=" * 50)
    
    app.run(debug=True, port=5000, threaded=True)