#!/usr/bin/env python3
"""
BIOIMAGIN Enhanced Flask backend for Focused Wolffia Analysis
Professional web interface with optimized features and temporal analysis
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
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# Create directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Global analyzer instance
analyzer = WolffiaAnalyzer(pixel_to_micron_ratio=0.5, chlorophyll_threshold=0.6)

# Analysis management
analysis_queue = queue.Queue()
analysis_results = {}
analysis_progress = {}
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
    """Upload images for analysis with temporal support"""
    try:
        if 'files' not in request.files:
            return jsonify({'error': 'No files provided'}), 400
        
        files = request.files.getlist('files')
        if not files or files[0].filename == '':
            return jsonify({'error': 'No files selected'}), 400
        
        use_celldetection = request.form.get('use_celldetection', 'true').lower() == 'true'
        enable_temporal = request.form.get('enable_temporal', 'false').lower() == 'true'
        
        uploaded_files = []
        for i, file in enumerate(files):
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                unique_filename = f"{uuid.uuid4()}_{filename}"
                file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
                file.save(file_path)
                
                # Create timestamp for temporal analysis
                timestamp = f"t{i+1:03d}_{datetime.now().strftime('%H%M%S')}" if enable_temporal else None
                
                file_info = {
                    'id': str(uuid.uuid4()),
                    'filename': filename,
                    'path': file_path,
                    'use_celldetection': use_celldetection,
                    'timestamp': timestamp,
                    'upload_order': i + 1
                }
                uploaded_files.append(file_info)
                uploaded_files_store[file_info['id']] = file_info
        
        if not uploaded_files:
            return jsonify({'error': 'No valid files uploaded'}), 400
        
        return jsonify({
            'success': True,
            'files': uploaded_files,
            'temporal_analysis_enabled': enable_temporal,
            'message': f'{len(uploaded_files)} files uploaded successfully'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

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

@app.route('/api/analyze/<file_id>', methods=['POST'])
def analyze_image(file_id):
    """Analyze a specific image"""
    try:
        request_data = request.get_json() or {}
        files_data = request_data.get('files', [])
        
        # Get analysis options
        use_celldetection = request_data.get('use_celldetection', True)
        use_tophat = request_data.get('use_tophat', False)
        
        # Find file info
        file_info = None
        
        if files_data:
            for file_data in files_data:
                if isinstance(file_data, dict) and file_data.get('id') == file_id:
                    file_info = file_data
                    break
        
        if not file_info and file_id in uploaded_files_store:
            file_info = uploaded_files_store[file_id].copy()
            file_info['use_celldetection'] = use_celldetection
            file_info['use_tophat'] = use_tophat
        
        if not file_info:
            return jsonify({'error': 'File info not found. Please upload the file again.'}), 404
        
        # Start analysis in background
        def run_analysis():
            try:
                analysis_results[file_id] = {'status': 'processing', 'progress': 0}
                
                print(f"🔬 Starting FOCUSED analysis for: {file_info.get('filename', 'Unknown')}")
                print(f"📝 Options: celldetection={use_celldetection}, tophat={use_tophat}")
                
                file_path = file_info.get('path')
                if not file_path or not os.path.exists(file_path):
                    raise FileNotFoundError(f"File not found: {file_path}")
                
                # Run focused analysis
                result = analyzer.analyze_image(
                    file_path, 
                    use_celldetection=use_celldetection,
                    use_tophat=use_tophat,
                    image_timestamp=file_info.get('timestamp')
                )
                
                print(f"📊 Analysis result: {result.get('success', False)}")
                
                # Save results
                result_file = Path('results') / f"{file_id}_result.json"
                
                try:
                    with open(result_file, 'w') as f:
                        json.dump(result, f, indent=2, default=str)
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

@app.route('/api/analyze_batch', methods=['POST'])
def analyze_batch():
    """Analyze multiple images with temporal analysis support"""
    try:
        request_data = request.get_json() or {}
        files_data = request_data.get('files', [])
        
        if not files_data:
            return jsonify({'error': 'No files provided for batch analysis'}), 400
        
        use_celldetection = request_data.get('use_celldetection', True)
        enable_temporal = request_data.get('enable_temporal', False)
        
        batch_id = str(uuid.uuid4())
        
        def run_batch_analysis():
            try:
                analysis_results[batch_id] = {'status': 'processing', 'progress': 0, 'results': []}
                
                print(f"🔬 Starting BATCH analysis for {len(files_data)} files")
                print(f"📝 Options: celldetection={use_celldetection}, temporal={enable_temporal}")
                
                # Prepare file paths
                file_paths = []
                file_infos = []
                
                for file_data in files_data:
                    file_id = file_data.get('id')
                    if file_id in uploaded_files_store:
                        file_info = uploaded_files_store[file_id]
                        file_paths.append(file_info['path'])
                        file_infos.append(file_info)
                
                if not file_paths:
                    raise ValueError("No valid file paths found")
                
                # Run analysis
                if enable_temporal and len(file_paths) > 1:
                    # Use temporal analysis for multiple images
                    results = analyze_multiple_images(file_paths, use_celldetection=use_celldetection)
                else:
                    # Individual analysis for each image
                    results = []
                    for i, file_path in enumerate(file_paths):
                        timestamp = file_infos[i].get('timestamp') if enable_temporal else None
                        result = analyzer.analyze_image(
                            file_path,
                            use_celldetection=use_celldetection,
                            image_timestamp=timestamp
                        )
                        results.append(result)
                        
                        # Update progress
                        progress = int(((i + 1) / len(file_paths)) * 100)
                        analysis_results[batch_id]['progress'] = progress
                
                # Save batch results
                batch_result_file = Path('results') / f"batch_{batch_id}_results.json"
                
                batch_summary = {
                    'batch_id': batch_id,
                    'total_files': len(file_paths),
                    'temporal_analysis': enable_temporal,
                    'timestamp': datetime.now().isoformat(),
                    'individual_results': results,
                    'batch_summary': create_batch_summary(results)
                }
                
                try:
                    with open(batch_result_file, 'w') as f:
                        json.dump(batch_summary, f, indent=2, default=str)
                    print(f"💾 Batch results saved to {batch_result_file}")
                except Exception as save_error:
                    print(f"⚠️ Failed to save batch results: {save_error}")
                
                analysis_results[batch_id] = {
                    'status': 'completed',
                    'progress': 100,
                    'result': batch_summary
                }
                
            except Exception as e:
                print(f"❌ Batch analysis error: {e}")
                analysis_results[batch_id] = {
                    'status': 'error',
                    'progress': 0,
                    'error': str(e)
                }
        
        # Start batch analysis thread
        thread = threading.Thread(target=run_batch_analysis)
        thread.start()
        
        return jsonify({
            'success': True,
            'batch_id': batch_id,
            'status': 'started',
            'files_count': len(files_data)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def create_batch_summary(results):
    """Create summary statistics for batch analysis"""
    try:
        if not results:
            return {}
        
        successful_results = [r for r in results if r.get('success', False)]
        
        if not successful_results:
            return {'total_files': len(results), 'successful_analyses': 0}
        
        # Aggregate statistics
        total_cells = sum(r.get('detection_results', {}).get('cells_detected', 0) for r in successful_results)
        total_biomass = sum(r.get('quantitative_analysis', {}).get('biomass_analysis', {}).get('total_biomass_mg', 0) for r in successful_results)
        avg_green_percentage = np.mean([r.get('quantitative_analysis', {}).get('color_analysis', {}).get('green_cell_percentage', 0) for r in successful_results])
        
        # Health distribution aggregate
        health_categories = {'excellent': 0, 'good': 0, 'moderate': 0, 'poor': 0}
        for result in successful_results:
            health_dist = result.get('quantitative_analysis', {}).get('health_assessment', {}).get('health_distribution', {})
            for category in health_categories:
                health_categories[category] += health_dist.get(category, 0)
        
        # Temporal analysis summary (if available)
        temporal_summary = None
        if len(successful_results) > 1:
            temporal_data = []
            for i, result in enumerate(successful_results):
                temporal_analysis = result.get('temporal_analysis')
                if temporal_analysis:
                    temporal_data.append(temporal_analysis)
            
            if temporal_data:
                temporal_summary = {
                    'time_points': len(temporal_data),
                    'has_temporal_data': True
                }
        
        return {
            'total_files': len(results),
            'successful_analyses': len(successful_results),
            'failed_analyses': len(results) - len(successful_results),
            'aggregate_statistics': {
                'total_cells_detected': total_cells,
                'total_biomass_mg': total_biomass,
                'average_green_cell_percentage': avg_green_percentage,
                'health_distribution_aggregate': health_categories
            },
            'temporal_analysis_summary': temporal_summary
        }
        
    except Exception as e:
        print(f"❌ Batch summary creation failed: {e}")
        return {'error': str(e)}

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
            output_file = f"results/{analysis_id}_export.json"
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            
            return send_file(output_file, as_attachment=True)
        
        elif format == 'csv':
            # Export comprehensive CSV
            output_file = export_comprehensive_csv(result, analysis_id)
            if output_file:
                return send_file(output_file, 
                               as_attachment=True, 
                               download_name=f"wolffia_analysis_{analysis_id}.csv")
            else:
                return jsonify({'error': 'No data to export'}), 400
        
        elif format == 'excel':
            # Export comprehensive Excel file
            output_file = export_comprehensive_excel(result, analysis_id)
            if output_file:
                return send_file(output_file, 
                               as_attachment=True, 
                               download_name=f"wolffia_analysis_{analysis_id}.xlsx")
            else:
                return jsonify({'error': 'No data to export'}), 400
        
        elif format == 'report':
            # Export detailed report
            output_file = export_detailed_report(result, analysis_id)
            if output_file:
                return send_file(output_file, 
                               as_attachment=True, 
                               download_name=f"wolffia_report_{analysis_id}.zip")
            else:
                return jsonify({'error': 'Report generation failed'}), 400
        
        else:
            return jsonify({'error': 'Unsupported format'}), 400
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def export_comprehensive_csv(result, analysis_id):
    """Export comprehensive CSV with all analysis data"""
    try:
        # Check if it's a single result or batch result
        if 'batch_summary' in result:
            # Batch result - export aggregate data
            individual_results = result.get('individual_results', [])
            all_cells_data = []
            
            for i, individual_result in enumerate(individual_results):
                if individual_result.get('success', False):
                    cells_data = individual_result.get('detection_results', {}).get('cells_data', [])
                    for cell in cells_data:
                        cell_row = create_cell_csv_row(cell, i + 1)
                        all_cells_data.append(cell_row)
            
            if all_cells_data:
                df = pd.DataFrame(all_cells_data)
                output_file = f"results/{analysis_id}_batch_cells.csv"
                df.to_csv(output_file, index=False)
                return output_file
        
        else:
            # Single result
            cells_data = result.get('detection_results', {}).get('cells_data', [])
            if cells_data:
                csv_data = []
                for cell in cells_data:
                    cell_row = create_cell_csv_row(cell)
                    csv_data.append(cell_row)
                
                df = pd.DataFrame(csv_data)
                output_file = f"results/{analysis_id}_cells.csv"
                df.to_csv(output_file, index=False)
                return output_file
        
        return None
        
    except Exception as e:
        print(f"❌ CSV export failed: {e}")
        return None

def create_cell_csv_row(cell, image_number=1):
    """Create CSV row for a single cell with comprehensive data"""
    try:
        row = {
            'Image_Number': image_number,
            'Cell_ID': cell.get('id', ''),
            'Area_Pixels': cell.get('area', 0),
            'Intensity': cell.get('intensity', 0),
            'Green_Intensity': cell.get('green_intensity', cell.get('intensity', 0)),
            'Center_X': cell.get('center', [0, 0])[0],
            'Center_Y': cell.get('center', [0, 0])[1],
            'Detection_Method': cell.get('method', 'unknown'),
            'Confidence': cell.get('confidence', 0),
            'Perimeter': cell.get('perimeter', 0),
            'Circularity': cell.get('circularity', 0)
        }
        
        # Add biomass data if available
        biomass_data = cell.get('biomass_data', {})
        if biomass_data:
            row.update({
                'Fresh_Weight_mg': biomass_data.get('fresh_weight_mg', 0),
                'Dry_Weight_mg': biomass_data.get('dry_weight_mg', 0),
                'Chlorophyll_mg': biomass_data.get('chlorophyll_mg', 0),
                'Protein_mg': biomass_data.get('protein_mg', 0),
                'Carbon_mg': biomass_data.get('carbon_mg', 0),
                'Volume_Microns3': biomass_data.get('volume_microns3', 0)
            })
        
        return row
        
    except Exception as e:
        print(f"❌ Cell CSV row creation failed: {e}")
        return {}

def export_comprehensive_excel(result, analysis_id):
    """Export comprehensive Excel file with multiple sheets"""
    try:
        output_file = f"results/{analysis_id}_comprehensive.xlsx"
        
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # Sheet 1: Summary
            summary_data = create_summary_sheet_data(result)
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Sheet 2: Cell Data
            if 'batch_summary' in result:
                # Batch results
                individual_results = result.get('individual_results', [])
                all_cells_data = []
                
                for i, individual_result in enumerate(individual_results):
                    if individual_result.get('success', False):
                        cells_data = individual_result.get('detection_results', {}).get('cells_data', [])
                        for cell in cells_data:
                            cell_row = create_cell_csv_row(cell, i + 1)
                            all_cells_data.append(cell_row)
                
                if all_cells_data:
                    cells_df = pd.DataFrame(all_cells_data)
                    cells_df.to_excel(writer, sheet_name='Cell_Data', index=False)
            else:
                # Single result
                cells_data = result.get('detection_results', {}).get('cells_data', [])
                if cells_data:
                    csv_data = [create_cell_csv_row(cell) for cell in cells_data]
                    cells_df = pd.DataFrame(csv_data)
                    cells_df.to_excel(writer, sheet_name='Cell_Data', index=False)
            
            # Sheet 3: Biomass Analysis
            biomass_data = create_biomass_sheet_data(result)
            if biomass_data:
                biomass_df = pd.DataFrame(biomass_data)
                biomass_df.to_excel(writer, sheet_name='Biomass_Analysis', index=False)
        
        return output_file
        
    except Exception as e:
        print(f"❌ Excel export failed: {e}")
        return None

def create_summary_sheet_data(result):
    """Create summary sheet data for Excel export"""
    try:
        if 'batch_summary' in result:
            batch_summary = result['batch_summary']
            return [{
                'Parameter': 'Analysis Type',
                'Value': 'Batch Analysis'
            }, {
                'Parameter': 'Total Files',
                'Value': batch_summary.get('total_files', 0)
            }, {
                'Parameter': 'Successful Analyses', 
                'Value': batch_summary.get('successful_analyses', 0)
            }, {
                'Parameter': 'Total Cells Detected',
                'Value': batch_summary.get('aggregate_statistics', {}).get('total_cells_detected', 0)
            }, {
                'Parameter': 'Total Biomass (mg)',
                'Value': batch_summary.get('aggregate_statistics', {}).get('total_biomass_mg', 0)
            }]
        else:
            quantitative = result.get('quantitative_analysis', {})
            return [{
                'Parameter': 'Analysis Type',
                'Value': 'Single Image Analysis'
            }, {
                'Parameter': 'Cells Detected',
                'Value': quantitative.get('cell_count', 0)
            }, {
                'Parameter': 'Total Biomass (mg)',
                'Value': quantitative.get('biomass_analysis', {}).get('total_biomass_mg', 0)
            }, {
                'Parameter': 'Average Cell Area (μm²)',
                'Value': quantitative.get('average_cell_area', 0)
            }, {
                'Parameter': 'Green Cell Percentage',
                'Value': quantitative.get('color_analysis', {}).get('green_cell_percentage', 0)
            }]
        
    except Exception as e:
        print(f"❌ Summary sheet creation failed: {e}")
        return []

def create_biomass_sheet_data(result):
    """Create biomass analysis sheet data"""
    try:
        biomass_rows = []
        
        if 'batch_summary' in result:
            # Batch biomass summary
            aggregate_stats = result.get('batch_summary', {}).get('aggregate_statistics', {})
            biomass_rows.append({
                'Metric': 'Total Biomass (mg)',
                'Value': aggregate_stats.get('total_biomass_mg', 0)
            })
        else:
            # Single result biomass
            biomass_analysis = result.get('quantitative_analysis', {}).get('biomass_analysis', {})
            for key, value in biomass_analysis.items():
                biomass_rows.append({
                    'Metric': key.replace('_', ' ').title(),
                    'Value': value
                })
        
        return biomass_rows
        
    except Exception as e:
        print(f"❌ Biomass sheet creation failed: {e}")
        return []

def export_detailed_report(result, analysis_id):
    """Export detailed report with all visualizations and data"""
    try:
        output_file = f"results/{analysis_id}_detailed_report.zip"
        
        with zipfile.ZipFile(output_file, 'w') as zip_file:
            # Add JSON data
            json_data = json.dumps(result, indent=2, default=str)
            zip_file.writestr(f"{analysis_id}_data.json", json_data)
            
            # Add CSV data
            csv_file = export_comprehensive_csv(result, analysis_id)
            if csv_file and os.path.exists(csv_file):
                zip_file.write(csv_file, f"{analysis_id}_cells.csv")
            
            # Add visualizations (if available)
            visualizations = result.get('visualizations', {})
            for viz_name, viz_data in visualizations.items():
                if viz_data and isinstance(viz_data, str):
                    # Decode base64 image
                    try:
                        image_data = base64.b64decode(viz_data)
                        zip_file.writestr(f"{analysis_id}_{viz_name}.png", image_data)
                    except Exception as e:
                        print(f"⚠️ Failed to add visualization {viz_name}: {e}")
            
            # Add summary report
            summary_text = create_text_summary(result)
            zip_file.writestr(f"{analysis_id}_summary.txt", summary_text)
        
        return output_file
        
    except Exception as e:
        print(f"❌ Detailed report export failed: {e}")
        return None

def create_text_summary(result):
    """Create text summary of analysis results"""
    try:
        summary = "BIOIMAGIN Wolffia Analysis Report\n"
        summary += "=" * 40 + "\n\n"
        
        # Basic info
        summary += f"Analysis Date: {result.get('timestamp', 'Unknown')}\n"
        summary += f"Processing Time: {result.get('processing_time', 0):.2f} seconds\n\n"
        
        if 'batch_summary' in result:
            # Batch summary
            batch_summary = result['batch_summary']
            summary += "BATCH ANALYSIS SUMMARY\n"
            summary += "-" * 25 + "\n"
            summary += f"Total Files Analyzed: {batch_summary.get('total_files', 0)}\n"
            summary += f"Successful Analyses: {batch_summary.get('successful_analyses', 0)}\n"
            
            aggregate_stats = batch_summary.get('aggregate_statistics', {})
            summary += f"Total Cells Detected: {aggregate_stats.get('total_cells_detected', 0)}\n"
            summary += f"Total Biomass: {aggregate_stats.get('total_biomass_mg', 0):.3f} mg\n"
            summary += f"Average Green Cell %: {aggregate_stats.get('average_green_cell_percentage', 0):.1f}%\n\n"
        else:
            # Single analysis summary
            detection_results = result.get('detection_results', {})
            quantitative = result.get('quantitative_analysis', {})
            
            summary += "DETECTION RESULTS\n"
            summary += "-" * 17 + "\n"
            summary += f"Cells Detected: {detection_results.get('cells_detected', 0)}\n"
            summary += f"Detection Method: {detection_results.get('detection_method', 'Unknown')}\n\n"
            
            summary += "QUANTITATIVE ANALYSIS\n"
            summary += "-" * 21 + "\n"
            
            biomass = quantitative.get('biomass_analysis', {})
            summary += f"Total Biomass: {biomass.get('total_biomass_mg', 0):.3f} mg\n"
            summary += f"Average Biomass per Cell: {biomass.get('avg_biomass_mg', 0):.4f} mg\n"
            summary += f"Total Chlorophyll: {biomass.get('total_chlorophyll_mg', 0):.4f} mg\n\n"
            
            color = quantitative.get('color_analysis', {})
            summary += f"Green Cell Percentage: {color.get('green_cell_percentage', 0):.1f}%\n"
            summary += f"Average Green Intensity: {color.get('avg_green_intensity', 0):.1f}\n\n"
            
            health = quantitative.get('health_assessment', {})
            summary += f"Population Health: {health.get('overall_health', 'Unknown').title()}\n"
            summary += f"Health Score: {health.get('health_score', 0):.2f}/1.0\n\n"
        
        summary += "Report generated by BIOIMAGIN Wolffia Analysis System\n"
        return summary
        
    except Exception as e:
        print(f"❌ Text summary creation failed: {e}")
        return "Summary generation failed."

@app.route('/api/health')
def health_check():
    """System health check"""
    return jsonify({
        'status': 'healthy',
        'version': 'focused_optimized',
        'analyzer': 'WolffiaAnalyzer',
        'features': [
            'focused_detection',
            'biomass_quantification',
            'temporal_analysis',
            'professional_visualization',
            'comprehensive_export'
        ],
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/set_parameters', methods=['POST'])
def set_parameters():
    """Set analysis parameters"""
    try:
        params = request.json
        if 'pixel_to_micron_ratio' in params:
            analyzer.pixel_to_micron_ratio = params['pixel_to_micron_ratio']
        if 'chlorophyll_threshold' in params:
            analyzer.chlorophyll_threshold = params['chlorophyll_threshold']
        
        return jsonify({'success': True, 'message': 'Parameters updated'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/get_parameters')
def get_parameters():
    """Get current analysis parameters"""
    return jsonify({
        'pixel_to_micron_ratio': analyzer.pixel_to_micron_ratio,
        'chlorophyll_threshold': analyzer.chlorophyll_threshold,
        'wolffia_params': analyzer.wolffia_params,
        'biomass_params': analyzer.biomass_params
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


if __name__ == '__main__':
    print("🚀 Starting BIOIMAGIN Focused Web Server...")
    print("✅ Wolffia Analyzer loaded with optimized features:")
    print("   • Focused cell detection (AI + classical fallback)")
    print("   • Comprehensive biomass quantification")
    print("   • Temporal analysis for time-series data")
    print("   • Professional visualizations")
    print("   • Multi-format export capabilities")
    print("🌐 Server running at http://localhost:5000")
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True
    )