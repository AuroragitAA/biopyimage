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
        thread = threading.Thread(target=process_enhanced_analysis, 
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
            
        elif format == 'ml_report':
            # Generate ML-specific report
            report = generate_ml_report(results)
            report_path = os.path.join(app.config['RESULTS_FOLDER'], f'{analysis_id}_ml_report.txt')
            with open(report_path, 'w') as f:
                f.write(report)
            return send_file(report_path, as_attachment=True, 
                           download_name='wolffia_ml_analysis_report.txt')
            
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
                
                # Add comprehensive reports
                report = generate_enhanced_analysis_report(results)
                zf.writestr('comprehensive_report.txt', report)
                
                ml_report = generate_ml_report(results)
                zf.writestr('ml_insights_report.txt', ml_report)
                
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

@app.route('/api/live_analysis', methods=['POST'])
def live_analysis():
    """Perform live ML-enhanced analysis on uploaded image"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file'}), 400
        
        # Save temporary file
        filename = secure_filename(file.filename)
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f'temp_{filename}')
        file.save(temp_path)
        
        # Update progress
        analysis_id = str(uuid.uuid4())
        analysis_progress[analysis_id] = {
            'status': 'processing',
            'progress': 50,
            'current_step': 'Analyzing with ML'
        }
        
        # Analyze with ML enhancement
        result = analyze_uploaded_image(temp_path, analyzer)
        
        # Clean up temp file
        os.remove(temp_path)
        
        if result:
            # Add ML insights summary
            ml_summary = extract_ml_summary(result)
            result['ml_summary'] = ml_summary
            
            return jsonify({
                'success': True,
                'result': result
            })
        else:
            return jsonify({'error': 'Analysis failed'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/ml_insights/<analysis_id>', methods=['GET'])
def get_ml_insights(analysis_id):
    """Get detailed ML insights for an analysis"""
    try:
        if analysis_id not in analysis_results:
            return jsonify({'error': 'Analysis not found'}), 404
        
        results = analysis_results[analysis_id]['results']
        
        # Extract ML-specific insights
        insights = {
            'feature_importance': None,
            'anomaly_summary': [],
            'population_predictions': None,
            'cell_classification_summary': {},
            'growth_stage_analysis': {},
            'health_trends': [],
            'biomass_model_comparison': {}
        }
        
        # Feature importance from last result
        if results and 'ml_metrics' in results[-1]:
            insights['feature_importance'] = results[-1]['ml_metrics'].get('feature_importance')
        
        # Anomaly summary across all images
        for result in results:
            if 'ml_metrics' in result:
                anomaly_count = result['ml_metrics'].get('anomalies_detected', 0)
                insights['anomaly_summary'].append({
                    'timestamp': result['timestamp'],
                    'anomalies': anomaly_count,
                    'percentage': (anomaly_count / result['total_cells'] * 100) if result['total_cells'] > 0 else 0
                })
        
        # Population predictions
        if results and 'population_dynamics' in results[-1]:
            insights['population_predictions'] = results[-1]['population_dynamics']
        
        # Cell classification summary
        for result in results:
            if 'summary' in result and 'ml_cell_type_distribution' in result['summary']:
                insights['cell_classification_summary'][result['timestamp']] = result['summary']['ml_cell_type_distribution']
        
        # Growth stage analysis
        for result in results:
            if 'summary' in result and 'ml_growth_stage_distribution' in result['summary']:
                insights['growth_stage_analysis'][result['timestamp']] = result['summary']['ml_growth_stage_distribution']
        
        # Health trends
        for result in results:
            if 'summary' in result:
                insights['health_trends'].append({
                    'timestamp': result['timestamp'],
                    'mean_health_score': result['summary'].get('mean_health_score', 0),
                    'healthy_percentage': result['summary'].get('healthy_cell_percentage', 0)
                })
        
        # Biomass model comparison
        for result in results:
            if 'summary' in result:
                insights['biomass_model_comparison'][result['timestamp']] = {
                    'volume': result['summary'].get('biomass_volume_total', 0),
                    'area': result['summary'].get('biomass_area_total', 0),
                    'allometric': result['summary'].get('biomass_allometric_total', 0),
                    'ml': result['summary'].get('biomass_ml_total', 0),
                    'ensemble': result['summary'].get('total_biomass_ug', 0)
                }
        
        return jsonify({
            'success': True,
            'insights': insights
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/compare_cells', methods=['POST'])
def compare_cells():
    """Enhanced cell comparison with ML tracking"""
    try:
        data = request.json
        analysis_id = data.get('analysis_id')
        
        if analysis_id not in analysis_results:
            return jsonify({'error': 'Analysis not found'}), 404
        
        results = analysis_results[analysis_id]['results']
        
        # Enhanced cell comparison with ML features
        comparison_data = []
        
        for i in range(len(results) - 1):
            current = results[i]
            next_result = results[i + 1]
            
            if 'cells' in current and 'cells' in next_result:
                current_cells = pd.DataFrame(current['cells'])
                next_cells = pd.DataFrame(next_result['cells'])
                
                # ML-enhanced matching
                matches = []
                for _, cell1 in current_cells.iterrows():
                    best_match = None
                    min_distance = float('inf')
                    best_similarity = 0
                    
                    for _, cell2 in next_cells.iterrows():
                        # Spatial distance
                        dist = ((cell1['centroid_x'] - cell2['centroid_x'])**2 + 
                               (cell1['centroid_y'] - cell2['centroid_y'])**2)**0.5
                        
                        # Feature similarity (area, chlorophyll, shape)
                        size_ratio = cell2['area_microns_sq'] / cell1['area_microns_sq'] if cell1['area_microns_sq'] > 0 else 0
                        chlor_diff = abs(cell2['chlorophyll_index'] - cell1['chlorophyll_index'])
                        circ_diff = abs(cell2['circularity'] - cell1['circularity'])
                        
                        # Combined similarity score
                        if 0.7 <= size_ratio <= 1.3 and dist < 100:  # Basic constraints
                            similarity = 1 / (1 + dist/50 + chlor_diff + circ_diff)
                            
                            if similarity > best_similarity:
                                best_similarity = similarity
                                min_distance = dist
                                best_match = cell2
                    
                    if best_match is not None and best_similarity > 0.5:
                        # Calculate ML-based changes
                        ml_health_change = 0
                        if 'ml_health_score' in cell1 and 'ml_health_score' in best_match:
                            ml_health_change = best_match['ml_health_score'] - cell1['ml_health_score']
                        
                        growth_stage_change = 'stable'
                        if 'ml_growth_stage' in cell1 and 'ml_growth_stage' in best_match:
                            if cell1['ml_growth_stage'] != best_match['ml_growth_stage']:
                                growth_stage_change = f"{cell1['ml_growth_stage']} → {best_match['ml_growth_stage']}"
                        
                        matches.append({
                            'timepoint_1': current['timestamp'],
                            'timepoint_2': next_result['timestamp'],
                            'cell_id_1': int(cell1['cell_id']),
                            'cell_id_2': int(best_match['cell_id']),
                            'area_change': float(best_match['area_microns_sq'] - cell1['area_microns_sq']),
                            'area_change_percent': float((best_match['area_microns_sq'] - cell1['area_microns_sq']) / cell1['area_microns_sq'] * 100),
                            'chlorophyll_change': float(best_match['chlorophyll_index'] - cell1['chlorophyll_index']),
                            'biomass_change': float(best_match['biomass_ensemble'] - cell1['biomass_ensemble']),
                            'ml_health_change': float(ml_health_change),
                            'growth_stage_change': growth_stage_change,
                            'similarity_score': float(best_similarity),
                            'distance_moved': float(min_distance)
                        })
                
                comparison_data.append({
                    'from': current['timestamp'],
                    'to': next_result['timestamp'],
                    'matched_cells': len(matches),
                    'total_cells_t1': len(current_cells),
                    'total_cells_t2': len(next_cells),
                    'match_percentage': len(matches) / len(current_cells) * 100 if len(current_cells) > 0 else 0,
                    'new_cells': len(next_cells) - len(matches),
                    'lost_cells': len(current_cells) - len(matches),
                    'matches': matches
                })
        
        return jsonify({
            'success': True,
            'comparisons': comparison_data
        })
        
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

@app.route('/api/optimize_parameters/<analysis_id>', methods=['POST'])
def optimize_parameters(analysis_id):
    """Trigger parameter optimization based on analysis results"""
    try:
        if analysis_id not in analysis_results:
            return jsonify({'error': 'Analysis not found'}), 404
        
        results = analysis_results[analysis_id]['results']
        
        # Use analyzer's optimization method
        optimized_params = analyzer.optimize_parameters(results)
        
        return jsonify({
            'success': True,
            'optimized_parameters': optimized_params,
            'message': 'Parameters optimized based on analysis results'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stream_progress/<analysis_id>')
def stream_progress(analysis_id):
    """Stream analysis progress updates using Server-Sent Events"""
    def generate():
        while True:
            if analysis_id in analysis_progress:
                progress = analysis_progress[analysis_id]
                yield f"data: {json.dumps(progress)}\n\n"
                
                if progress['status'] in ['completed', 'failed']:
                    break
            else:
                yield f"data: {json.dumps({'status': 'not_found'})}\n\n"
                break
            
            import time
            time.sleep(0.5)
    
    return Response(generate(), mimetype="text/event-stream")

@app.route('/api/health_check', methods=['GET'])
def health_check():
    """Enhanced API health check with ML status"""
    return jsonify({
        'status': 'healthy',
        'version': '2.0-ML-Enhanced',
        'analyzer_status': 'ready',
        'ml_components': {
            'classifier': 'RandomForest',
            'anomaly_detector': 'IsolationForest',
            'feature_extraction': 'active',
            'population_modeling': 'active'
        },
        'parameters': analyzer.get_current_parameters()
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
    """Generate enhanced summary with ML insights"""
    try:
        if not results:
            return {}
        
        total_images = len(results)
        total_cells = sum(r.get('total_cells', 0) for r in results)

        # Aggregates
        all_biomass = []
        all_health_scores = []
        all_anomalies = []
        green_cell_counts = []
        cell_type_evolution = {}
        growth_stage_evolution = {}

        for result in results:
            if 'summary' in result:
                summary = result['summary']
                all_biomass.append(summary.get('total_biomass_ug', 0))
                all_health_scores.append(summary.get('mean_health_score', 0))
                green_cell_counts.append(summary.get('green_cell_count', 0))  # ✅ Add green cell count

                if 'ml_metrics' in result:
                    all_anomalies.append(result['ml_metrics'].get('anomalies_detected', 0))

                # Track cell type evolution
                if 'ml_cell_type_distribution' in summary:
                    for cell_type, count in summary['ml_cell_type_distribution'].items():
                        cell_type_evolution.setdefault(cell_type, []).append(count)

                # Track growth stage evolution
                if 'ml_growth_stage_distribution' in summary:
                    for stage, count in summary['ml_growth_stage_distribution'].items():
                        growth_stage_evolution.setdefault(stage, []).append(count)

        # ML Metrics
        ml_metrics = {
            'total_anomalies_detected': sum(all_anomalies),
            'anomaly_rate': (sum(all_anomalies) / total_cells * 100) if total_cells > 0 else 0,
            'mean_population_health': np.mean(all_health_scores) if all_health_scores else 0,
            'health_trend': 'improving' if len(all_health_scores) > 1 and all_health_scores[-1] > all_health_scores[0] else 'stable',
            'dominant_cell_type': max(cell_type_evolution.keys(), key=lambda k: sum(cell_type_evolution[k])) if cell_type_evolution else 'unknown',
            'population_diversity': len(cell_type_evolution)
        }

        # Growth metrics
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

            if 'population_dynamics' in results[-1]:
                pop_dyn = results[-1]['population_dynamics'].get('growth_analysis', {})
                growth_metrics['exponential_growth_rate'] = pop_dyn.get('cell_count_growth_rate', 0)
                growth_metrics['doubling_time'] = pop_dyn.get('doubling_time_cells', 0)
                growth_metrics['carrying_capacity'] = pop_dyn.get('carrying_capacity')

        # Biomass models
        biomass_models = {}
        if results and 'summary' in results[-1]:
            s = results[-1]['summary']
            biomass_models = {
                'volume_model': s.get('biomass_volume_total', 0),
                'area_model': s.get('biomass_area_total', 0),
                'allometric_model': s.get('biomass_allometric_total', 0),
                'ml_model': s.get('biomass_ml_total', 0),
                'ensemble_model': s.get('total_biomass_ug', 0),
                'mean_uncertainty': s.get('mean_biomass_uncertainty', 0)
            }

        # Average chlorophyll and cell area
        average_chlorophyll = np.mean([
            r['summary'].get('mean_chlorophyll_intensity', 0)
            for r in results if 'summary' in r
        ])
        average_cell_area = np.mean([
            r['summary'].get('mean_cell_area_microns', 0)
            for r in results if 'summary' in r
        ])

        return {
            'total_images_analyzed': total_images,
            'total_cells_detected': total_cells,
            'total_green_cells': sum(green_cell_counts),
            'total_biomass': sum(all_biomass),
            'average_biomass_per_image': sum(all_biomass) / len(all_biomass) if all_biomass else 0,
            'average_chlorophyll': average_chlorophyll,
            'average_cell_area': average_cell_area,
            'ml_metrics': ml_metrics,
            'growth_metrics': growth_metrics,
            'biomass_models': biomass_models,
            'cell_type_evolution': cell_type_evolution,
            'growth_stage_evolution': growth_stage_evolution,
            'analysis_quality': {
                'mean_segmentation_confidence': np.mean([
                    r['summary'].get('segmentation_confidence', 0) for r in results if 'summary' in r
                ]),
                'mean_population_homogeneity': np.mean([
                    r['summary'].get('population_homogeneity', 0) for r in results if 'summary' in r
                ])
            }
        }

    except Exception as e:
        print(f"Error generating enhanced summary: {str(e)}")
        return {}


def generate_enhanced_analysis_report(results):
    """Generate comprehensive analysis report with ML insights"""
    try:
        report = []
        report.append("BIOIMAGIN - ENHANCED WOLFFIA ANALYSIS REPORT")
        report.append("=" * 60)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Analysis Version: 2.0 ML-Enhanced")
        report.append(f"Total Images Analyzed: {len(results)}")
        report.append("")
        
        # Overall summary
        summary = generate_enhanced_summary(results)
        report.append("OVERALL SUMMARY")
        report.append("-" * 30)
        report.append(f"Total Cells Detected: {summary.get('total_cells_detected', 0)}")
        report.append(f"Total Biomass (μg): {summary.get('total_biomass', 0):.2f}")
        report.append("")
        
        # ML Insights
        if 'ml_metrics' in summary:
            ml = summary['ml_metrics']
            report.append("MACHINE LEARNING INSIGHTS")
            report.append("-" * 30)
            report.append(f"Total Anomalies Detected: {ml.get('total_anomalies_detected', 0)}")
            report.append(f"Anomaly Rate: {ml.get('anomaly_rate', 0):.1f}%")
            report.append(f"Mean Population Health Score: {ml.get('mean_population_health', 0):.3f}")
            report.append(f"Health Trend: {ml.get('health_trend', 'unknown')}")
            report.append(f"Dominant Cell Type: {ml.get('dominant_cell_type', 'unknown')}")
            report.append(f"Population Diversity (types): {ml.get('population_diversity', 0)}")
        
        # Growth analysis
        if 'growth_metrics' in summary and summary['growth_metrics']:
            report.append("")
            report.append("GROWTH ANALYSIS")
            report.append("-" * 30)
            gm = summary['growth_metrics']
            report.append(f"Biomass Change: {gm.get('biomass_change', 0):.2f} μg")
            report.append(f"Biomass Growth Rate: {gm.get('biomass_growth_rate', 0):.1f}%")
            report.append(f"Cell Count Change: {gm.get('cell_count_change', 0)}")
            report.append(f"Cell Growth Rate: {gm.get('cell_growth_rate', 0):.1f}%")
            
            if 'exponential_growth_rate' in gm:
                report.append(f"Exponential Growth Rate: {gm['exponential_growth_rate']:.4f}")
                report.append(f"Doubling Time: {gm.get('doubling_time', 0):.1f} time units")
            
            if gm.get('carrying_capacity'):
                report.append(f"Estimated Carrying Capacity: {gm['carrying_capacity']:.0f} cells")
        
        # Biomass models
        if 'biomass_models' in summary and summary['biomass_models']:
            report.append("")
            report.append("BIOMASS MODEL COMPARISON")
            report.append("-" * 30)
            bm = summary['biomass_models']
            report.append(f"Volume Model: {bm.get('volume_model', 0):.2f} μg")
            report.append(f"Area Model: {bm.get('area_model', 0):.2f} μg")
            report.append(f"Allometric Model: {bm.get('allometric_model', 0):.2f} μg")
            report.append(f"ML Model: {bm.get('ml_model', 0):.2f} μg")
            report.append(f"Ensemble Model: {bm.get('ensemble_model', 0):.2f} μg")
            report.append(f"Mean Uncertainty: ±{bm.get('mean_uncertainty', 0):.3f} μg")
        
        # Individual image results with ML details
        report.append("")
        report.append("INDIVIDUAL IMAGE RESULTS")
        report.append("-" * 30)
        
        for i, result in enumerate(results):
            report.append(f"\nImage {i+1}: {result.get('timestamp', 'N/A')}")
            report.append(f"  File: {os.path.basename(result.get('image_path', 'N/A'))}")
            report.append(f"  Total Cells: {result.get('total_cells', 0)}")
            
            if 'summary' in result:
                s = result['summary']
                report.append(f"  Mean Cell Area: {s.get('mean_cell_area_microns', 0):.2f} μm²")
                report.append(f"  Mean Chlorophyll: {s.get('mean_chlorophyll_intensity', 0):.3f}")
                report.append(f"  Total Biomass: {s.get('total_biomass_ug', 0):.2f} μg")
                report.append(f"  Mean Health Score: {s.get('mean_health_score', 0):.3f}")
                report.append(f"  Healthy Cells: {s.get('healthy_cell_percentage', 0):.1f}%")
                
                # ML classifications
                if 'ml_cell_type_distribution' in s:
                    report.append("  ML Cell Types:")
                    for cell_type, count in s['ml_cell_type_distribution'].items():
                        report.append(f"    - {cell_type}: {count}")
                
                if 'ml_growth_stage_distribution' in s:
                    report.append("  Growth Stages:")
                    for stage, count in s['ml_growth_stage_distribution'].items():
                        report.append(f"    - {stage}: {count}")
                
                # Quality metrics
                report.append(f"  Segmentation Confidence: {s.get('segmentation_confidence', 0):.3f}")
                report.append(f"  Population Homogeneity: {s.get('population_homogeneity', 0):.3f}")
            
            if 'ml_metrics' in result:
                report.append(f"  Anomalies Detected: {result['ml_metrics'].get('anomalies_detected', 0)}")
        
        # Population dynamics
        if results and 'population_dynamics' in results[-1]:
            report.append("")
            report.append("POPULATION DYNAMICS")
            report.append("-" * 30)
            pop_dyn = results[-1]['population_dynamics']
            
            if 'alerts' in pop_dyn and pop_dyn['alerts']:
                report.append("Alerts:")
                for alert in pop_dyn['alerts']:
                    report.append(f"  - {alert['message']} (Severity: {alert['severity']})")
            else:
                report.append("No population alerts")
        
        # Analysis parameters
        report.append("")
        report.append("ANALYSIS PARAMETERS")
        report.append("-" * 30)
        params = analyzer.get_current_parameters()
        for param, value in params.items():
            report.append(f"{param}: {value}")
        
        # Optimized parameters if available
        if results and 'optimized_parameters' in results[-1]:
            report.append("")
            report.append("OPTIMIZED PARAMETERS")
            report.append("-" * 30)
            opt_params = results[-1]['optimized_parameters']
            for param, value in opt_params.items():
                report.append(f"{param}: {value}")
        
        return "\n".join(report)
        
    except Exception as e:
        return f"Error generating report: {str(e)}"

def generate_ml_report(results):
    """Generate ML-specific analysis report"""
    try:
        report = []
        report.append("MACHINE LEARNING ANALYSIS REPORT")
        report.append("=" * 50)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Feature importance
        if results and 'ml_metrics' in results[-1] and results[-1]['ml_metrics'].get('feature_importance'):
            report.append("FEATURE IMPORTANCE ANALYSIS")
            report.append("-" * 30)
            fi = results[-1]['ml_metrics']['feature_importance']
            if 'feature' in fi and 'importance' in fi:
                features = fi['feature']
                importances = fi['importance']
                for f, i in zip(features[:10], importances[:10]):  # Top 10
                    report.append(f"{f}: {i:.4f}")
            report.append("")
        
        # Anomaly analysis
        report.append("ANOMALY DETECTION SUMMARY")
        report.append("-" * 30)
        total_anomalies = 0
        anomaly_details = []
        
        for result in results:
            if 'ml_metrics' in result:
                anomalies = result['ml_metrics'].get('anomalies_detected', 0)
                total_anomalies += anomalies
                anomaly_details.append({
                    'timestamp': result['timestamp'],
                    'anomalies': anomalies,
                    'total_cells': result['total_cells']
                })
        
        report.append(f"Total Anomalies Across All Images: {total_anomalies}")
        for detail in anomaly_details:
            if detail['anomalies'] > 0:
                percentage = (detail['anomalies'] / detail['total_cells'] * 100) if detail['total_cells'] > 0 else 0
                report.append(f"  {detail['timestamp']}: {detail['anomalies']} anomalies ({percentage:.1f}%)")
        
        # Cell classification evolution
        if len(results) > 1:
            report.append("")
            report.append("CELL TYPE EVOLUTION")
            report.append("-" * 30)
            
            # Collect cell type data
            cell_type_timeline = {}
            for result in results:
                if 'summary' in result and 'ml_cell_type_distribution' in result['summary']:
                    timestamp = result['timestamp']
                    for cell_type, count in result['summary']['ml_cell_type_distribution'].items():
                        if cell_type not in cell_type_timeline:
                            cell_type_timeline[cell_type] = {}
                        cell_type_timeline[cell_type][timestamp] = count
            
            for cell_type, timeline in cell_type_timeline.items():
                report.append(f"\n{cell_type}:")
                for timestamp, count in timeline.items():
                    report.append(f"  {timestamp}: {count} cells")
        
        # Health progression
        report.append("")
        report.append("POPULATION HEALTH PROGRESSION")
        report.append("-" * 30)
        
        health_data = []
        for result in results:
            if 'summary' in result:
                health_data.append({
                    'timestamp': result['timestamp'],
                    'mean_health': result['summary'].get('mean_health_score', 0),
                    'healthy_percentage': result['summary'].get('healthy_cell_percentage', 0)
                })
        
        for data in health_data:
            report.append(f"{data['timestamp']}: Health Score={data['mean_health']:.3f}, Healthy={data['healthy_percentage']:.1f}%")
        
        # Growth stage analysis
        report.append("")
        report.append("GROWTH STAGE DISTRIBUTION")
        report.append("-" * 30)
        
        for result in results:
            if 'summary' in result and 'ml_growth_stage_distribution' in result['summary']:
                report.append(f"\n{result['timestamp']}:")
                for stage, count in result['summary']['ml_growth_stage_distribution'].items():
                    report.append(f"  {stage}: {count} cells")
        
        # Biomass model performance
        report.append("")
        report.append("BIOMASS MODEL PERFORMANCE")
        report.append("-" * 30)
        
        model_totals = {
            'volume': [],
            'area': [],
            'allometric': [],
            'ml': [],
            'ensemble': []
        }
        
        for result in results:
            if 'summary' in result:
                s = result['summary']
                model_totals['volume'].append(s.get('biomass_volume_total', 0))
                model_totals['area'].append(s.get('biomass_area_total', 0))
                model_totals['allometric'].append(s.get('biomass_allometric_total', 0))
                model_totals['ml'].append(s.get('biomass_ml_total', 0))
                model_totals['ensemble'].append(s.get('total_biomass_ug', 0))
        
        for model, values in model_totals.items():
            if values:
                mean_val = np.mean(values)
                std_val = np.std(values)
                report.append(f"{model.capitalize()} Model: {mean_val:.2f} ± {std_val:.2f} μg")
        
        # Spectral analysis summary
        report.append("")
        report.append("SPECTRAL INDICES SUMMARY")
        report.append("-" * 30)
        
        spectral_data = {
            'ndvi': [],
            'gci': [],
            'exg': []
        }
        
        for result in results:
            if 'summary' in result:
                s = result['summary']
                spectral_data['ndvi'].append(s.get('mean_ndvi', 0))
                spectral_data['gci'].append(s.get('mean_gci', 0))
                spectral_data['exg'].append(s.get('mean_exg', 0))
        
        for index, values in spectral_data.items():
            if values:
                report.append(f"Mean {index.upper()}: {np.mean(values):.3f} (range: {min(values):.3f} - {max(values):.3f})")
        
        # Population predictions
        if results and 'population_dynamics' in results[-1]:
            report.append("")
            report.append("POPULATION PREDICTIONS")
            report.append("-" * 30)
            
            pop_dyn = results[-1]['population_dynamics']
            if 'growth_analysis' in pop_dyn:
                ga = pop_dyn['growth_analysis']
                report.append(f"Cell Growth Rate: {ga.get('cell_count_growth_rate', 0):.4f}")
                report.append(f"Biomass Growth Rate: {ga.get('biomass_growth_rate', 0):.4f}")
                report.append(f"Doubling Time (cells): {ga.get('doubling_time_cells', 0):.1f} time units")
                report.append(f"Doubling Time (biomass): {ga.get('doubling_time_biomass', 0):.1f} time units")
                
                if ga.get('carrying_capacity'):
                    report.append(f"Estimated Carrying Capacity: {ga['carrying_capacity']:.0f} cells")
        
        return "\n".join(report)
        
    except Exception as e:
        return f"Error generating ML report: {str(e)}"

def extract_ml_summary(result):
    """Extract key ML insights for quick display"""
    summary = {
        'dominant_cell_type': 'unknown',
        'health_status': 'unknown',
        'anomalies': 0,
        'growth_stage': 'unknown',
        'biomass_confidence': 0
    }
    
    try:
        if 'summary' in result:
            s = result['summary']
            
            # Dominant cell type
            if 'ml_cell_type_distribution' in s and s['ml_cell_type_distribution']:
                summary['dominant_cell_type'] = max(s['ml_cell_type_distribution'].items(), 
                                                  key=lambda x: x[1])[0]
            
            # Health status
            if 'mean_health_score' in s:
                score = s['mean_health_score']
                if score > 0.7:
                    summary['health_status'] = 'healthy'
                elif score > 0.4:
                    summary['health_status'] = 'moderate'
                else:
                    summary['health_status'] = 'stressed'
            
            # Anomalies
            if 'ml_metrics' in result:
                summary['anomalies'] = result['ml_metrics'].get('anomalies_detected', 0)
            
            # Dominant growth stage
            if 'ml_growth_stage_distribution' in s and s['ml_growth_stage_distribution']:
                summary['growth_stage'] = max(s['ml_growth_stage_distribution'].items(), 
                                            key=lambda x: x[1])[0]
            
            # Biomass confidence
            if 'mean_biomass_uncertainty' in s and s.get('total_biomass_ug', 0) > 0:
                uncertainty_ratio = s['mean_biomass_uncertainty'] / s['total_biomass_ug']
                summary['biomass_confidence'] = max(0, 1 - uncertainty_ratio)
        
    except Exception as e:
        print(f"Error extracting ML summary: {str(e)}")
    
    return summary

@app.route('/api/segmentation_methods', methods=['GET'])
def get_segmentation_methods():
    """Get available segmentation methods"""
    return jsonify({
        'methods': [
            {
                'id': 'multi_otsu',
                'name': 'Multi-Otsu Thresholding',
                'description': 'Multiple threshold levels for better separation'
            },
            {
                'id': 'adaptive',
                'name': 'Adaptive Thresholding',
                'description': 'Local thresholding with multiple block sizes'
            },
            {
                'id': 'kmeans',
                'name': 'K-means Clustering',
                'description': 'Color-based segmentation using machine learning'
            },
            {
                'id': 'felzenszwalb',
                'name': 'Felzenszwalb Segmentation',
                'description': 'Graph-based image segmentation'
            },
            {
                'id': 'slic',
                'name': 'SLIC Superpixels',
                'description': 'Simple Linear Iterative Clustering'
            },
            {
                'id': 'ensemble',
                'name': 'Ensemble Method',
                'description': 'Combines all methods with weighted voting'
            }
        ],
        'current': 'ensemble'
    })

@app.route('/api/ml_models', methods=['GET'])
def get_ml_models():
    """Get information about ML models in use"""
    return jsonify({
        'classification': {
            'model': 'RandomForestClassifier',
            'features': 12,
            'description': 'Ensemble learning for cell classification'
        },
        'anomaly_detection': {
            'model': 'IsolationForest',
            'contamination': 0.1,
            'description': 'Detects unusual cell morphologies'
        },
        'biomass_models': [
            'Volume-based estimation',
            'Area-based with chlorophyll',
            'Allometric scaling',
            'ML-enhanced prediction',
            'Ensemble averaging'
        ],
        'population_models': [
            'Exponential growth',
            'Logistic growth',
            'Carrying capacity estimation'
        ]
    })





@app.route('/api/spectral_analysis/<analysis_id>', methods=['GET'])
def get_spectral_analysis(analysis_id):
    """Get detailed spectral analysis results"""
    try:
        if analysis_id not in analysis_results:
            return jsonify({'error': 'Analysis not found'}), 404
        
        results = analysis_results[analysis_id]['results']
        spectral_data = []
        
        for result in results:
            if 'spectral_analysis' in result:
                spectral_data.append({
                    'timestamp': result['timestamp'],
                    'data': result['spectral_analysis'],
                    'visualization': result.get('spectral_visualization'),
                    'report': result.get('spectral_report')
                })
        
        return jsonify({
            'success': True,
            'spectral_data': spectral_data
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/manual_annotation', methods=['POST'])
def save_manual_annotation():
    """Save manual cell annotations for training"""
    try:
        data = request.json
        image_id = data.get('image_id')
        annotations = data.get('annotations')
        
        # Save to annotation database
        annotation_file = f"annotations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(os.path.join('annotations', annotation_file), 'w') as f:
            json.dump({
                'image_id': image_id,
                'annotations': annotations,
                'timestamp': datetime.now().isoformat()
            }, f)
        
        return jsonify({
            'success': True,
            'annotation_file': annotation_file
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/train_model', methods=['POST'])
def train_ml_model():
    """Train ML model with manual annotations"""
    try:
        # Get all annotation files
        annotation_files = [
            os.path.join('annotations', f) 
            for f in os.listdir('annotations') 
            if f.endswith('.json')
        ]
        
        if not annotation_files:
            return jsonify({'error': 'No annotations found'}), 400
        
        # Train model
        accuracy = analyzer.train_from_annotations(annotation_files)
        
        return jsonify({
            'success': True,
            'accuracy': float(accuracy),
            'message': f'Model trained with {len(annotation_files)} annotation files',
            'training_date': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/live_capture', methods=['POST'])
def live_capture_analysis():
    """Analyze live captured frame"""
    try:
        data = request.json
        image_data = data.get('image_data')  # Base64 encoded image
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data.split(',')[1])
        image = Image.open(BytesIO(image_bytes))
        
        # Save temporary file
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f'live_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        image.save(temp_path)
        
        # Analyze with enhanced features
        result = analyzer.analyze_single_image(temp_path, save_visualization=True)
        
        if result:
            # Add spectral analysis
            spectral_df, spectral_viz = analyzer.analyze_chlorophyll_spectrum(
                result['preprocessed']['original'], 
                result['labels']
            )
            
            result['spectral_analysis'] = spectral_df.to_dict('records')
            result['spectral_visualization'] = spectral_viz
            
            # Export to JSON
            json_result = analyzer.export_enhanced_results(result)
            
            return jsonify({
                'success': True,
                'result': json_result,
                'instant_metrics': {
                    'cell_count': len(spectral_df),
                    'total_chlorophyll': float(spectral_df['total_chlorophyll_ug'].sum()),
                    'health_distribution': spectral_df['spectral_health'].value_counts().to_dict()
                }
            })
        else:
            return jsonify({'error': 'Analysis failed'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/batch_process', methods=['POST'])
def batch_process_with_settings():
    """Batch process with custom settings per image"""
    try:
        data = request.json
        settings = data.get('settings', {})
        
        # Apply custom settings
        if 'tophat_size' in settings:
            analyzer.tophat_size = settings['tophat_size']
        if 'segmentation_sensitivity' in settings:
            analyzer.segmentation_sensitivity = settings['segmentation_sensitivity']
        
        # Process batch
        results = []
        for file_info in data.get('files', []):
            result = analyzer.analyze_single_image(
                file_info['path'],
                timestamp=file_info.get('timestamp'),
                custom_params=file_info.get('params', {})
            )
            if result:
                results.append(result)
        
        return jsonify({
            'success': True,
            'results': results,
            'processed': len(results)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/export_training_data/<analysis_id>', methods=['GET'])
def export_training_data(analysis_id):
    """Export analysis results as training data"""
    try:
        if analysis_id not in analysis_results:
            return jsonify({'error': 'Analysis not found'}), 404
        
        results = analysis_results[analysis_id]['results']
        training_data = []
        
        for result in results:
            if 'cells' in result:
                for cell in result['cells']:
                    training_entry = {
                        'features': {
                            'area': cell.get('area_microns_sq'),
                            'chlorophyll': cell.get('chlorophyll_index'),
                            'circularity': cell.get('circularity'),
                            'ndvi': cell.get('mean_ndvi'),
                            'health_score': cell.get('ml_health_score')
                        },
                        'labels': {
                            'cell_type': cell.get('ml_cell_type'),
                            'health_status': cell.get('ml_health_status'),
                            'growth_stage': cell.get('ml_growth_stage')
                        },
                        'metadata': {
                            'timestamp': result['timestamp'],
                            'image_path': result['image_path']
                        }
                    }
                    training_data.append(training_entry)
        
        # Save as JSON
        output_path = os.path.join(app.config['RESULTS_FOLDER'], f'training_data_{analysis_id}.json')
        with open(output_path, 'w') as f:
            json.dump(training_data, f, indent=2)
        
        return send_file(output_path, as_attachment=True, 
                        download_name='wolffia_training_data.json')
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/upload_training_data', methods=['POST'])
def upload_training_data():
    """Upload and process bulk training data"""
    try:
        if 'training_files' not in request.files:
            return jsonify({'error': 'No training files provided'}), 400
        
        files = request.files.getlist('training_files')
        
        # Also check for annotation file if provided
        annotation_data = None
        if 'annotations' in request.files:
            ann_file = request.files['annotations']
            annotation_data = json.load(ann_file)
        
        # Save training images
        training_dir = os.path.join('training_data', datetime.now().strftime('%Y%m%d_%H%M%S'))
        os.makedirs(training_dir, exist_ok=True)
        
        saved_files = []
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(training_dir, filename)
                file.save(filepath)
                saved_files.append({
                    'filename': filename,
                    'path': filepath
                })
        
        # Process training data
        if annotation_data:
            # Use provided annotations
            training_result = process_annotated_training_data(saved_files, annotation_data)
        else:
            # Auto-generate training data from images
            training_result = auto_generate_training_data(saved_files)
        
        return jsonify({
            'success': True,
            'message': f'Uploaded {len(saved_files)} training images',
            'training_result': training_result
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def process_annotated_training_data(files, annotations):
    """Process training data with provided annotations"""
    try:
        # Create annotation files for each image
        annotation_files = []
        
        for file_data in files:
            filename = file_data['filename']
            if filename in annotations:
                ann_file = os.path.join('annotations', f"training_{filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
                with open(ann_file, 'w') as f:
                    json.dump({
                        'image_path': file_data['path'],
                        'annotations': annotations[filename],
                        'metadata': {
                            'pixel_to_micron': analyzer.pixel_to_micron,
                            'timestamp': datetime.now().isoformat()
                        }
                    }, f, indent=2)
                annotation_files.append(ann_file)
        
        # Train model with annotations
        if annotation_files:
            accuracy = analyzer.train_from_annotations(annotation_files)
            return {
                'training_complete': True,
                'accuracy': float(accuracy),
                'files_processed': len(annotation_files)
            }
        else:
            return {
                'training_complete': False,
                'message': 'No matching annotations found'
            }
            
    except Exception as e:
        return {'error': str(e)}

def auto_generate_training_data(files):
    """Auto-generate training data from images"""
    try:
        training_data = []

        for file_data in files:
            # Analyze image with enhanced pipeline
            result = analyzer.analyze_single_image_enhanced(file_data['path'], save_visualization=False)

            if result and 'cell_data' in result:
                df = result['cell_data']

                # Create training entries from detected cells
                for _, cell in df.iterrows():
                    training_entry = {
                        'features': {
                            'area': cell.get('area_microns_sq'),
                            'chlorophyll': cell.get('chlorophyll_index'),
                            'circularity': cell.get('circularity'),
                            'health_score': cell.get('ml_health_score', 0.5)
                        },
                        'labels': {
                            'cell_type': cell.get('ml_cell_type', 'unknown'),
                            'health_status': cell.get('ml_health_status', 'unknown'),
                            'growth_stage': cell.get('ml_growth_stage', 'unknown')
                        },
                        'metadata': {
                            'timestamp': cell.get('timestamp'),
                            'image_path': cell.get('image_path'),
                            'cell_id': cell.get('cell_id')
                        }
                    }
                    training_data.append(training_entry)

        return {
            'training_complete': True,
            'entries_created': len(training_data),
            'training_data': training_data
        }

    except Exception as e:
        return {
            'error': str(e),
            'training_complete': False
        }


if __name__ == '__main__':
    print("=" * 60)
    print("BIOIMAGIN Web Server v2.0 - ML Enhanced")
    print("=" * 60)
    print("Starting Flask server with ML capabilities...")
    print("Features enabled:")
    print("✓ Multi-method segmentation")
    print("✓ Machine learning classification")
    print("✓ Real-time progress tracking")
    print("✓ Population dynamics modeling")
    print("✓ Automated parameter optimization")
    print("✓ Comprehensive ML reports")
    print("=" * 60)
    
    app.run(debug=True, port=5000, threaded=True)