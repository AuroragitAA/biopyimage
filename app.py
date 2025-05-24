"""
app.py

Professional Flask Application for Wolffia Bioimage Analysis System
Enhanced integration with advanced analysis features and comprehensive API.
"""

import os
import cv2
import numpy as np
import base64
import logging
import json
import traceback
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from flask import Flask, request, jsonify, render_template, send_file
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/wolffia_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app with enhanced configuration
app = Flask(__name__)
app.config.update({
    'MAX_CONTENT_LENGTH': 32 * 1024 * 1024,  # 32MB max file size
    'UPLOAD_FOLDER': 'temp_uploads',
    'SECRET_KEY': 'your-enhanced-secret-key-here',
    'JSON_SORT_KEYS': False,
    'JSONIFY_PRETTYPRINT_REGULAR': True
})

# Create necessary directories
for directory in ['temp_uploads', 'results', 'logs', 'exports']:
    Path(directory).mkdir(exist_ok=True)

# Initialize analysis components with comprehensive error handling
analyzer = None
component_status = {
    'analyzer': False,
    'image_processor': False,
    'segmentation': False,
    'feature_extractor': False,
    'startup_time': datetime.now().isoformat()
}

def initialize_analyzer():
    """Initialize the enhanced Wolffia analyzer with fallback handling."""
    global analyzer, component_status
    
    try:
        logger.info("🚀 Initializing Professional Wolffia Analyzer...")
        
        # Try to import and initialize the enhanced analyzer
        from wolffia_analyzer import WolffiaAnalyzer
        
        analyzer = WolffiaAnalyzer(
            pixel_to_micron_ratio=1.0,
            chlorophyll_threshold=0.6,
            min_cell_area=30,
            max_cell_area=8000
        )
        
        # Test analyzer functionality
        status = analyzer.get_component_status()
        component_status.update(status)
        component_status['analyzer'] = True
        
        logger.info("✅ Professional Wolffia Analyzer initialized successfully")
        logger.info(f"📊 Component status: {status}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize analyzer: {str(e)}")
        logger.error(f"📋 Traceback: {traceback.format_exc()}")
        
        # Try fallback initialization
        try:
            analyzer = create_fallback_analyzer()
            component_status['analyzer'] = True
            logger.warning("⚠️ Using fallback analyzer")
            return True
        except Exception as fallback_error:
            logger.error(f"❌ Fallback analyzer also failed: {str(fallback_error)}")
            return False

def create_fallback_analyzer():
    """Create a basic fallback analyzer if main system fails."""
    class FallbackAnalyzer:
        def __init__(self):
            self.results_history = []
            
        def analyze_single_image(self, image_path, **kwargs):
            try:
                # Basic image analysis
                image = cv2.imread(image_path)
                if image is None:
                    return self._failed_result(str(image_path), "Could not load image")
                
                # Very basic cell detection using simple thresholding
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                # Find contours
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Filter contours by area
                min_area, max_area = 30, 8000
                valid_contours = [c for c in contours if min_area <= cv2.contourArea(c) <= max_area]
                
                # Create basic cell data
                cell_data = []
                for i, contour in enumerate(valid_contours[:50]):  # Limit to 50 cells
                    area = cv2.contourArea(contour)
                    perimeter = cv2.arcLength(contour, True)
                    
                    # Calculate centroid
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                    else:
                        cx, cy = 0, 0
                    
                    cell_data.append({
                        'cell_id': i + 1,
                        'area': float(area),
                        'perimeter': float(perimeter),
                        'centroid_x': float(cx),
                        'centroid_y': float(cy),
                        'chlorophyll': 0.5,  # Default value
                        'classification': 'detected',
                        'health_score': 0.5
                    })
                
                # Calculate summary
                total_cells = len(cell_data)
                avg_area = np.mean([c['area'] for c in cell_data]) if cell_data else 0
                
                summary = {
                    'total_cells': total_cells,
                    'avg_area': avg_area,
                    'total_biomass_estimate': avg_area * total_cells * 0.001,
                    'chlorophyll_ratio': 50.0  # Default
                }
                
                result = {
                    'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
                    'image_path': str(image_path),
                    'cell_data': cell_data,
                    'summary': summary,
                    'total_cells': total_cells,
                    'success': True,
                    'processing_info': {'method': 'fallback'},
                    'quality_score': 0.5
                }
                
                self.results_history.append(result)
                return result
                
            except Exception as e:
                return self._failed_result(str(image_path), str(e))
        
        def _failed_result(self, image_path, error):
            return {
                'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
                'image_path': image_path,
                'cell_data': [],
                'summary': {'total_cells': 0, 'avg_area': 0, 'total_biomass_estimate': 0, 'chlorophyll_ratio': 0},
                'total_cells': 0,
                'success': False,
                'error': error
            }
        
        def get_component_status(self):
            return {'fallback_mode': True, 'analyzer': True}
        
        def get_analysis_summary(self):
            if not self.results_history:
                return {"message": "No analyses performed yet"}
            
            successful = [r for r in self.results_history if r.get('success')]
            return {
                'total_images_analyzed': len(successful),
                'total_cells_detected': sum(r['total_cells'] for r in successful),
                'success_rate': len(successful) / len(self.results_history) * 100 if self.results_history else 0
            }
    
    return FallbackAnalyzer()

# Initialize analyzer on startup
analyzer_initialized = initialize_analyzer()

# Enhanced utility functions
def validate_image_file(file) -> tuple[bool, str]:
    """Comprehensive image file validation."""
    if not file or file.filename == '':
        return False, "No file selected"
    
    # Check file extension
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.jfif'}
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in allowed_extensions:
        return False, f"Unsupported file format. Allowed: {', '.join(allowed_extensions)}"
    
    # Check file size (additional check beyond Flask's MAX_CONTENT_LENGTH)
    if hasattr(file, 'content_length') and file.content_length:
        if file.content_length > app.config['MAX_CONTENT_LENGTH']:
            return False, "File too large. Maximum size: 32MB"
    
    return True, "Valid"

def create_visualization(original: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Create enhanced visualization overlay from segmentation labels."""
    try:
        if len(labels.shape) == 2:  # Grayscale labels
            # Create colored visualization
            viz = np.zeros((*labels.shape, 3), dtype=np.uint8)
            
            # Use different colors for each cell
            unique_labels = np.unique(labels)
            colors = [
                [255, 0, 0],    # Red
                [0, 255, 0],    # Green
                [0, 0, 255],    # Blue
                [255, 255, 0],  # Yellow
                [255, 0, 255],  # Magenta
                [0, 255, 255],  # Cyan
                [255, 128, 0],  # Orange
                [128, 0, 255],  # Purple
                [0, 128, 255],  # Light Blue
                [128, 255, 0],  # Light Green
            ]
            
            for i, label in enumerate(unique_labels[1:], 1):  # Skip background (0)
                color = colors[(i - 1) % len(colors)]
                viz[labels == label] = color
        else:
            viz = labels.astype(np.uint8)
        
        # Create overlay with transparency
        if len(original.shape) == 3 and original.shape[2] == 3:
            # Blend original image with visualization
            overlay = cv2.addWeighted(original, 0.6, viz, 0.4, 0)
        else:
            overlay = viz
            
        return overlay
        
    except Exception as e:
        logger.error(f"❌ Visualization creation error: {e}")
        return original if original is not None else np.zeros((100, 100, 3), dtype=np.uint8)

def calculate_enhanced_statistics(analysis_result: Dict) -> Dict:
    """Calculate enhanced statistics from analysis results."""
    try:
        if not analysis_result.get('success') or not analysis_result.get('cell_data'):
            return {
                'total_cells': 0,
                'avg_area': 0,
                'total_biomass': 0,
                'chlorophyll_ratio': 0,
                'health_statistics': {},
                'size_distribution': {}
            }
        
        cell_data = analysis_result['cell_data']
        df = pd.DataFrame(cell_data) if cell_data else pd.DataFrame()
        
        if len(df) == 0:
            return analysis_result.get('summary', {})
        
        # Enhanced statistics
        stats = {
            'total_cells': len(df),
            'avg_area': float(df['area'].mean()),
            'std_area': float(df['area'].std()),
            'median_area': float(df['area'].median()),
            'total_area': float(df['area'].sum()),
            'total_biomass': float(df['area'].sum() * 0.001),  # Simple biomass estimate
        }
        
        # Chlorophyll analysis
        if 'chlorophyll' in df.columns or 'chlorophyll_content' in df.columns:
            chlorophyll_col = 'chlorophyll_content' if 'chlorophyll_content' in df.columns else 'chlorophyll'
            high_chlorophyll = len(df[df[chlorophyll_col] > 0.6])
            stats.update({
                'chlorophyll_ratio': float(high_chlorophyll / len(df) * 100),
                'mean_chlorophyll': float(df[chlorophyll_col].mean()),
                'std_chlorophyll': float(df[chlorophyll_col].std())
            })
        else:
            stats.update({
                'chlorophyll_ratio': 0.0,
                'mean_chlorophyll': 0.0,
                'std_chlorophyll': 0.0
            })
        
        # Health analysis
        if 'health_score' in df.columns:
            stats['health_statistics'] = {
                'mean_health': float(df['health_score'].mean()),
                'healthy_cells': int(len(df[df['health_score'] > 0.7])),
                'moderate_cells': int(len(df[(df['health_score'] >= 0.3) & (df['health_score'] <= 0.7)])),
                'poor_cells': int(len(df[df['health_score'] < 0.3]))
            }
        
        # Size distribution
        if 'area' in df.columns:
            stats['size_distribution'] = {
                'small_cells': int(len(df[df['area'] < 100])),
                'medium_cells': int(len(df[(df['area'] >= 100) & (df['area'] < 500)])),
                'large_cells': int(len(df[df['area'] >= 500]))
            }
        
        return stats
        
    except Exception as e:
        logger.error(f"❌ Enhanced statistics calculation error: {e}")
        return analysis_result.get('summary', {})

# Main Flask routes
@app.route('/')
def index():
    """Serve the main application page with system status."""
    try:
        system_status = {
            'analyzer_available': analyzer is not None,
            'component_status': component_status,
            'startup_time': component_status.get('startup_time'),
            'total_analyses': len(analyzer.results_history) if analyzer else 0
        }
        
        return render_template('index.html', system_status=system_status)
    except Exception as e:
        logger.error(f"❌ Index route error: {str(e)}")
        return render_template('index.html', system_status={'analyzer_available': False})

@app.route('/health')
def health_check():
    """Comprehensive health check endpoint."""
    try:
        health_status = {
            'status': 'healthy' if analyzer else 'degraded',
            'timestamp': datetime.now().isoformat(),
            'components': component_status.copy(),
            'analyzer_summary': analyzer.get_analysis_summary() if analyzer else None,
            'system_info': {
                'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                'opencv_version': cv2.__version__ if 'cv2' in globals() else 'Not available',
                'numpy_version': np.__version__,
                'flask_version': flask.__version__ if 'flask' in globals() else 'Unknown'
            }
        }
        
        status_code = 200 if analyzer else 503
        return jsonify(health_status), status_code
        
    except Exception as e:
        logger.error(f"❌ Health check error: {str(e)}")
        return jsonify({
            'status': 'error', 
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/analyze', methods=['POST'])
def analyze_image():
    """Enhanced image analysis endpoint with comprehensive error handling."""
    temp_path = None
    
    try:
        # Validate analyzer availability
        if not analyzer:
            return jsonify({
                'error': 'Analysis system not available',
                'details': 'Analyzer failed to initialize properly'
            }), 503
        
        # Validate request
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        
        # Validate file
        is_valid, message = validate_image_file(file)
        if not is_valid:
            return jsonify({'error': message}), 400
        
        # Extract and validate parameters
        try:
            params = {
                'analysis_method': request.form.get('analysis_method', 'auto'),
                'color_method': request.form.get('color_method', 'green_wolffia'),
                'pixel_ratio': float(request.form.get('pixel_ratio', 1.0)),
                'chlorophyll_threshold': float(request.form.get('chlorophyll_threshold', 0.6)),
                'min_cell_area': int(request.form.get('min_cell_area', 30)),
                'max_cell_area': int(request.form.get('max_cell_area', 8000))
            }
            
            # Validate parameter ranges
            if not (0.1 <= params['pixel_ratio'] <= 10.0):
                return jsonify({'error': 'Pixel ratio must be between 0.1 and 10.0'}), 400
            if not (0.0 <= params['chlorophyll_threshold'] <= 1.0):
                return jsonify({'error': 'Chlorophyll threshold must be between 0.0 and 1.0'}), 400
            if not (10 <= params['min_cell_area'] <= 1000):
                return jsonify({'error': 'Minimum cell area must be between 10 and 1000'}), 400
            if not (100 <= params['max_cell_area'] <= 50000):
                return jsonify({'error': 'Maximum cell area must be between 100 and 50000'}), 400
                
        except (ValueError, TypeError) as e:
            return jsonify({'error': f'Invalid parameter values: {str(e)}'}), 400
        
        logger.info(f"🔍 Starting analysis with params: {params}")
        
        # Save uploaded file temporarily with secure filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        secure_name = secure_filename(file.filename)
        temp_filename = f"{timestamp}_{secure_name}"
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
        
        file.save(temp_path)
        logger.info(f"📁 File saved: {temp_path}")
        
        # Update analyzer parameters if needed
        if hasattr(analyzer, 'pixel_to_micron'):
            analyzer.pixel_to_micron = params['pixel_ratio']
            analyzer.chlorophyll_threshold = params['chlorophyll_threshold']
            analyzer.min_cell_area = params['min_cell_area']
            analyzer.max_cell_area = params['max_cell_area']
        
        # Perform analysis
        analysis_start = datetime.now()
        result = analyzer.analyze_single_image(
            temp_path, 
            method=params['analysis_method']
        )
        analysis_time = (datetime.now() - analysis_start).total_seconds()
        
        if not result:
            return jsonify({
                'error': 'Analysis failed',
                'details': 'No result returned from analyzer'
            }), 500
        
        if not result.get('success', False):
            error_msg = result.get('error', 'Unknown analysis error')
            logger.error(f"❌ Analysis failed: {error_msg}")
            return jsonify({
                'error': error_msg,
                'stats': result.get('summary', {}),
                'processing_time': analysis_time
            }), 500
        
        logger.info(f"✅ Analysis successful: {result['total_cells']} cells detected")
        
        # Create enhanced visualizations
        visualization_b64 = None
        original_b64 = None
        
        try:
            # Load original image for visualization
            original_image = cv2.imread(temp_path)
            if original_image is not None:
                original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
                
                # Create visualization if labels available
                if 'labels' in result:
                    viz_image = create_visualization(original_image, result['labels'])
                    
                    # Encode visualization
                    _, viz_buffer = cv2.imencode('.png', cv2.cvtColor(viz_image, cv2.COLOR_RGB2BGR))
                    visualization_b64 = base64.b64encode(viz_buffer).decode('utf-8')
                
                # Encode original
                _, orig_buffer = cv2.imencode('.png', cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR))
                original_b64 = base64.b64encode(orig_buffer).decode('utf-8')
                
        except Exception as viz_error:
            logger.warning(f"⚠️ Visualization creation failed: {viz_error}")
        
        # Calculate enhanced statistics
        enhanced_stats = calculate_enhanced_statistics(result)
        
        # Prepare comprehensive response
        response_data = {
            'success': True,
            'stats': enhanced_stats,
            'summary': result.get('summary', enhanced_stats),
            'total_cells': result['total_cells'],
            'cell_data': result['cell_data'][:100],  # Limit for UI performance
            'processing_info': {
                'analysis_time': analysis_time,
                'method': params['analysis_method'],
                'parameters': params,
                'quality_score': result.get('quality_score', 0.5),
                'timestamp': result['timestamp']
            },
            'visualization': visualization_b64,
            'original_image': original_b64,
            'image_info': {
                'filename': secure_name,
                'size_mb': round(os.path.getsize(temp_path) / (1024*1024), 2) if os.path.exists(temp_path) else 0,
                'analysis_timestamp': result['timestamp']
            }
        }
        
        # Add quality assessment if available
        if 'quality_details' in result:
            response_data['quality_assessment'] = result['quality_details']
        
        logger.info(f"📤 Sending response: {result['total_cells']} cells, {analysis_time:.2f}s processing time")
        
        return jsonify(response_data)
    
    except RequestEntityTooLarge:
        return jsonify({'error': 'File too large. Maximum size: 32MB.'}), 413
    
    except Exception as e:
        logger.error(f"❌ Unexpected analysis error: {str(e)}")
        logger.error(f"📋 Traceback: {traceback.format_exc()}")
        
        return jsonify({
            'error': 'Internal server error occurred during analysis',
            'details': str(e) if app.debug else 'Check server logs for details',
            'stats': {
                'total_cells': 0,
                'avg_area': 0,
                'total_biomass': 0,
                'chlorophyll_ratio': 0
            }
        }), 500
    
    finally:
        # Clean up temporary file
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
                logger.info(f"🗑️ Temporary file cleaned up: {temp_path}")
            except Exception as cleanup_error:
                logger.error(f"❌ Failed to clean up temp file: {cleanup_error}")

@app.route('/batch_analyze', methods=['POST'])
def batch_analyze():
    """Batch analysis endpoint for multiple images."""
    try:
        if not analyzer:
            return jsonify({'error': 'Analysis system not available'}), 503
        
        # Check if files were provided
        if 'images' not in request.files:
            return jsonify({'error': 'No image files provided'}), 400
        
        files = request.files.getlist('images')
        if not files or all(f.filename == '' for f in files):
            return jsonify({'error': 'No valid files selected'}), 400
        
        # Validate file count
        if len(files) > 50:  # Limit batch size
            return jsonify({'error': 'Too many files. Maximum 50 images per batch.'}), 400
        
        logger.info(f"🔄 Starting batch analysis of {len(files)} images")
        
        # Process files
        results = []
        temp_paths = []
        
        try:
            for i, file in enumerate(files):
                # Validate file
                is_valid, message = validate_image_file(file)
                if not is_valid:
                    results.append({
                        'filename': file.filename,
                        'success': False,
                        'error': message
                    })
                    continue
                
                # Save file temporarily
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                secure_name = secure_filename(file.filename)
                temp_filename = f"batch_{i}_{timestamp}_{secure_name}"
                temp_path = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
                
                file.save(temp_path)
                temp_paths.append(temp_path)
                
                # Analyze image
                try:
                    result = analyzer.analyze_single_image(temp_path)
                    
                    if result and result.get('success'):
                        results.append({
                            'filename': secure_name,
                            'success': True,
                            'total_cells': result['total_cells'],
                            'summary': result.get('summary', {}),
                            'processing_time': result.get('processing_time', 0)
                        })
                    else:
                        error_msg = result.get('error', 'Analysis failed') if result else 'No result'
                        results.append({
                            'filename': secure_name,
                            'success': False,
                            'error': error_msg
                        })
                        
                except Exception as analysis_error:
                    results.append({
                        'filename': secure_name,
                        'success': False,
                        'error': str(analysis_error)
                    })
            
            # Calculate batch summary
            successful_results = [r for r in results if r.get('success')]
            total_cells = sum(r.get('total_cells', 0) for r in successful_results)
            
            batch_summary = {
                'total_images': len(files),
                'successful_analyses': len(successful_results),
                'failed_analyses': len(results) - len(successful_results),
                'success_rate': len(successful_results) / len(files) * 100 if files else 0,
                'total_cells_detected': total_cells,
                'average_cells_per_image': total_cells / len(successful_results) if successful_results else 0
            }
            
            logger.info(f"✅ Batch analysis complete: {batch_summary}")
            
            return jsonify({
                'success': True,
                'batch_summary': batch_summary,
                'individual_results': results
            })
            
        finally:
            # Clean up all temporary files
            for temp_path in temp_paths:
                if os.path.exists(temp_path):
                    try:
                        os.unlink(temp_path)
                    except Exception as cleanup_error:
                        logger.error(f"❌ Failed to clean up {temp_path}: {cleanup_error}")
    
    except Exception as e:
        logger.error(f"❌ Batch analysis error: {str(e)}")
        return jsonify({
            'error': 'Batch analysis failed',
            'details': str(e)
        }), 500

@app.route('/export', methods=['POST'])
def export_results():
    """Export analysis results in various formats."""
    try:
        if not analyzer:
            return jsonify({'error': 'Analysis system not available'}), 503
        
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        export_format = data.get('format', 'csv').lower()
        include_all_history = data.get('include_history', False)
        
        # Use analyzer's export functionality if available
        if hasattr(analyzer, 'export_results'):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"wolffia_analysis_{timestamp}.{export_format}"
            export_path = os.path.join('exports', filename)
            
            result_path = analyzer.export_results(format=export_format, output_path=export_path)
            
            if result_path and os.path.exists(result_path):
                return send_file(
                    result_path,
                    as_attachment=True,
                    download_name=filename,
                    mimetype='application/octet-stream'
                )
            else:
                return jsonify({'error': 'Export failed'}), 500
        else:
            return jsonify({'message': 'Export functionality not available in current configuration'})
        
    except Exception as e:
        logger.error(f"❌ Export error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/system/status')
def system_status():
    """Get detailed system status."""
    try:
        status = {
            'timestamp': datetime.now().isoformat(),
            'analyzer_status': analyzer.get_component_status() if analyzer else None,
            'analysis_summary': analyzer.get_analysis_summary() if analyzer else None,
            'system_info': component_status.copy(),
            'memory_usage': {
                'results_in_memory': len(analyzer.results_history) if analyzer else 0
            }
        }
        
        return jsonify(status)
        
    except Exception as e:
        logger.error(f"❌ System status error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/system/reset', methods=['POST'])
def reset_system():
    """Reset analysis history and clear memory."""
    try:
        if analyzer and hasattr(analyzer, 'reset_analysis_history'):
            analyzer.reset_analysis_history()
            return jsonify({'message': 'System reset successfully'})
        else:
            return jsonify({'error': 'Reset not available'}), 503
            
    except Exception as e:
        logger.error(f"❌ System reset error: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Error handlers
@app.errorhandler(413)
def too_large(e):
    """Handle file too large error."""
    return jsonify({'error': 'File too large. Maximum size is 32MB.'}), 413

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors."""
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    """Handle internal server errors."""
    logger.error(f"Internal server error: {str(e)}")
    return jsonify({'error': 'Internal server error occurred'}), 500

@app.errorhandler(Exception)
def handle_exception(e):
    """Handle unexpected exceptions."""
    logger.error(f"Unhandled exception: {str(e)}")
    logger.error(f"Traceback: {traceback.format_exc()}")
    
    return jsonify({
        'error': 'An unexpected error occurred',
        'details': str(e) if app.debug else 'Check server logs'
    }), 500

# Startup information
if __name__ == '__main__':
    import sys
import pandas as pd
    
    print("🚀 Starting Professional Wolffia Bioimage Analysis Server...")
    print(f"📁 Upload directory: {app.config['UPLOAD_FOLDER']}")
    print(f"📊 System Components Status:")
    
    for component, status in component_status.items():
        status_icon = "✅" if status else "❌"
        print(f"   {component}: {status_icon}")
    
    if analyzer_initialized:
        print("🔬 Professional Wolffia Analyzer: ✅ Ready")
    else:
        print("🔬 Professional Wolffia Analyzer: ❌ Failed (using fallback)")
    
    print("🌐 Server will be available at: http://0.0.0.0:5000")
    print("📚 API Documentation: http://0.0.0.0:5000/health")
    print()
    
    try:
        app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as startup_error:
        logger.error(f"❌ Server startup failed: {startup_error}")
        sys.exit(1)