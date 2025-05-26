"""
batch_processor.py

Professional batch processing system for high-throughput Wolffia analysis.
Integrates with your existing dashboard to provide enterprise-grade capabilities.

Features:
- Multi-threaded batch processing
- Progress monitoring and ETA calculation
- Automatic quality control and filtering
- Export to multiple formats (CSV, Excel, JSON, HDF5)
- Integration with existing Flask dashboard
- Automated report generation
- Error handling and recovery
"""

import json
import logging
import os
import pickle
import threading
import time
import warnings
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from queue import Empty, Queue
from typing import Callable, Dict, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
import seaborn as sns
from flask import jsonify, request  # MAKE SURE these are imported
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import logging
import os

os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/bioimagin_app.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

try:
    from wolffia_analyzer import AnalysisConfig, WolffiaAnalyzer
    ANALYZER_AVAILABLE = True
    logger.info("[OK] Core analyzer imported successfully")
except ImportError as e:
    logger.error(f"[ERROR] Core analyzer not available: {e}")
    ANALYZER_AVAILABLE = False

os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/bioimagin_app.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)


@dataclass
class BatchJobConfig:
    """Configuration for batch processing jobs."""
    
    # Processing parameters
    max_workers: int = 4
    chunk_size: int = 10
    quality_threshold: float = 0.7
    
    # Output configuration
    output_directory: str = "batch_results"
    export_formats: List[str] = None
    include_visualizations: bool = True
    compress_results: bool = True
    
    # Quality control
    enable_qc: bool = True
    outlier_detection: bool = True
    consistency_checks: bool = True
    
    # Metadata
    job_name: str = ""
    operator_name: str = ""
    project_id: str = ""
    
    def __post_init__(self):
        if self.export_formats is None:
            self.export_formats = ['csv', 'excel']
        if not self.job_name:
            self.job_name = f"batch_job_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


class BatchProgressTracker:
    """Track progress and performance metrics for batch jobs."""
    
    def __init__(self, total_items: int):
        self.total_items = total_items
        self.completed_items = 0
        self.failed_items = 0
        self.start_time = datetime.now()
        self.item_times = []
        self.current_status = "Initializing"
        self.lock = threading.Lock()
        
    def update_progress(self, items_completed: int = 1, status: str = None):
        """Update progress tracking."""
        with self.lock:
            self.completed_items += items_completed
            if status:
                self.current_status = status
            self.item_times.append(time.time())
    
    def report_failure(self, items_failed: int = 1):
        """Report failed items."""
        with self.lock:
            self.failed_items += items_failed
    
    def get_progress_info(self) -> Dict:
        """Get current progress information."""
        with self.lock:
            elapsed_time = (datetime.now() - self.start_time).total_seconds()
            
            if self.completed_items > 0:
                avg_time_per_item = elapsed_time / self.completed_items
                remaining_items = self.total_items - self.completed_items - self.failed_items
                eta_seconds = remaining_items * avg_time_per_item
                eta = datetime.now() + timedelta(seconds=eta_seconds)
            else:
                eta = None
                avg_time_per_item = 0
            
            progress_percentage = (self.completed_items + self.failed_items) / self.total_items * 100
            
            return {
                'total_items': self.total_items,
                'completed_items': self.completed_items,
                'failed_items': self.failed_items,
                'progress_percentage': progress_percentage,
                'elapsed_time': elapsed_time,
                'eta': eta.isoformat() if eta else None,
                'current_status': self.current_status,
                'avg_time_per_item': avg_time_per_item,
                'items_per_minute': (self.completed_items / elapsed_time * 60) if elapsed_time > 0 else 0
            }


class QualityController:
    """Advanced quality control for batch processing."""
    
    def __init__(self, config: BatchJobConfig):
        self.config = AnalysisConfig
        self.quality_metrics = []
        self.outliers_detected = []
        
    def assess_image_quality(self, image_path: str) -> Dict:
        """Assess individual image quality."""
        try:
            import cv2
            from skimage import measure
            
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                return {'quality_score': 0.0, 'issues': ['Cannot load image']}
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Quality metrics
            issues = []
            
            # 1. Sharpness (Laplacian variance)
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness_score = min(sharpness / 1000.0, 1.0)
            
            if sharpness_score < 0.3:
                issues.append('Low sharpness detected')
            
            # 2. Contrast
            contrast = np.std(gray)
            contrast_score = min(contrast / 128.0, 1.0)
            
            if contrast_score < 0.3:
                issues.append('Low contrast detected')
            
            # 3. Brightness
            brightness = np.mean(gray)
            brightness_score = 1.0 - abs(brightness - 128) / 128.0
            
            if brightness < 50 or brightness > 200:
                issues.append('Poor brightness levels')
            
            # 4. Image size validation
            height, width = gray.shape
            if width < 500 or height < 500:
                issues.append('Image resolution too low')
            
            # 5. Check for motion blur
            motion_blur_score = self._detect_motion_blur(gray)
            if motion_blur_score > 0.7:
                issues.append('Motion blur detected')
            
            # Overall quality score
            overall_score = (sharpness_score + contrast_score + brightness_score) / 3.0
            
            quality_assessment = {
                'quality_score': overall_score,
                'sharpness': sharpness_score,
                'contrast': contrast_score,
                'brightness': brightness_score,
                'motion_blur': motion_blur_score,
                'issues': issues,
                'passes_qc': overall_score >= self.config.quality_threshold and len(issues) == 0
            }
            
            self.quality_metrics.append(quality_assessment)
            return quality_assessment
            
        except Exception as e:
            logger.error(f"Quality assessment error for {image_path}: {str(e)}")
            return {'quality_score': 0.0, 'issues': [f'Quality assessment failed: {str(e)}']}
    
    def _detect_motion_blur(self, gray_image: np.ndarray) -> float:
        """Detect motion blur using FFT analysis."""
        try:
            # Apply FFT
            f_transform = np.fft.fft2(gray_image)
            f_shift = np.fft.fftshift(f_transform)
            magnitude_spectrum = np.log(np.abs(f_shift) + 1)
            
            # Calculate the blur metric
            blur_metric = np.var(magnitude_spectrum)
            normalized_blur = 1.0 - min(blur_metric / 1000.0, 1.0)
            
            return normalized_blur
            
        except Exception as e:
            logger.error(f"Motion blur detection error: {str(e)}")
            return 0.0
    
    def detect_batch_outliers(self, results: List[Dict]) -> List[int]:
        """Detect outlier results in batch processing."""
        try:
            if len(results) < 10:  # Need minimum samples
                return []
            
            # Extract numeric features for outlier detection
            features = []
            for result in results:
                if result.get('success') and 'summary' in result:
                    summary = result['summary']
                    features.append([
                        summary.get('total_cells', 0),
                        summary.get('avg_area', 0),
                        summary.get('total_biomass_estimate', 0),
                        summary.get('chlorophyll_ratio', 0)
                    ])
            
            if len(features) < 5:
                return []
            
            # Use Isolation Forest for outlier detection
            from sklearn.ensemble import IsolationForest
            
            isolation_forest = IsolationForest(contamination=0.1, random_state=42)
            outlier_predictions = isolation_forest.fit_predict(features)
            
            # Get indices of outliers
            outlier_indices = [i for i, pred in enumerate(outlier_predictions) if pred == -1]
            
            self.outliers_detected = outlier_indices
            logger.info(f"Detected {len(outlier_indices)} outliers in batch results")
            
            return outlier_indices
            
        except Exception as e:
            logger.error(f"Outlier detection error: {str(e)}")
            return []
    
    def generate_qc_report(self) -> Dict:
        """Generate comprehensive quality control report."""
        try:
            if not self.quality_metrics:
                return {'message': 'No quality metrics available'}
            
            quality_scores = [qm['quality_score'] for qm in self.quality_metrics]
            
            report = {
                'total_images_assessed': len(self.quality_metrics),
                'average_quality_score': np.mean(quality_scores),
                'quality_distribution': {
                    'excellent': len([q for q in quality_scores if q > 0.8]),
                    'good': len([q for q in quality_scores if 0.6 <= q <= 0.8]),
                    'fair': len([q for q in quality_scores if 0.4 <= q < 0.6]),
                    'poor': len([q for q in quality_scores if q < 0.4])
                },
                'common_issues': self._analyze_common_issues(),
                'outliers_detected': len(self.outliers_detected),
                'quality_trend': self._calculate_quality_trend()
            }
            
            return report
            
        except Exception as e:
            logger.error(f"QC report generation error: {str(e)}")
            return {'error': str(e)}
    
    def _analyze_common_issues(self) -> Dict:
        """Analyze most common quality issues."""
        try:
            issue_counts = {}
            for qm in self.quality_metrics:
                for issue in qm.get('issues', []):
                    issue_counts[issue] = issue_counts.get(issue, 0) + 1
            
            # Sort by frequency
            sorted_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)
            
            return {
                'most_common_issues': sorted_issues[:5],
                'total_unique_issues': len(issue_counts)
            }
            
        except Exception as e:
            logger.error(f"Common issues analysis error: {str(e)}")
            return {}
    
    def _calculate_quality_trend(self) -> str:
        """Calculate quality trend over batch processing."""
        try:
            if len(self.quality_metrics) < 10:
                return 'insufficient_data'
            
            # Split into first and second half
            mid_point = len(self.quality_metrics) // 2
            first_half = [qm['quality_score'] for qm in self.quality_metrics[:mid_point]]
            second_half = [qm['quality_score'] for qm in self.quality_metrics[mid_point:]]
            
            first_avg = np.mean(first_half)
            second_avg = np.mean(second_half)
            
            if second_avg > first_avg + 0.05:
                return 'improving'
            elif second_avg < first_avg - 0.05:
                return 'declining'
            else:
                return 'stable'
                
        except Exception as e:
            logger.error(f"Quality trend calculation error: {str(e)}")
            return 'unknown'


class BatchExporter:
    """Handle export of batch results to multiple formats."""
    
    def __init__(self, config: BatchJobConfig):
        self.config = AnalysisConfig
        self.export_directory = Path(config.output_directory) / "exports"
        self.export_directory.mkdir(parents=True, exist_ok=True)
    
    def export_results(self, results: List[Dict], qc_report: Dict) -> Dict:
        """Export batch results in multiple formats."""
        try:
            export_info = {
                'timestamp': datetime.now().isoformat(),
                'job_name': self.config.job_name,
                'total_results': len(results),
                'exported_formats': [],
                'export_paths': {}
            }
            
            # Prepare consolidated data
            all_cell_data = []
            summary_data = []
            
            for i, result in enumerate(results):
                if result.get('success') and 'cell_data' in result:
                    # Add batch info to each cell record
                    for cell in result['cell_data']:
                        cell_record = cell.copy()
                        cell_record['batch_index'] = i
                        cell_record['image_path'] = result.get('image_path', '')
                        cell_record['analysis_timestamp'] = result.get('timestamp', '')
                        all_cell_data.append(cell_record)
                    
                    # Summary data
                    summary_record = result.get('summary', {}).copy()
                    summary_record['batch_index'] = i
                    summary_record['image_path'] = result.get('image_path', '')
                    summary_record['success'] = result.get('success', False)
                    summary_record['total_cells'] = result.get('total_cells', 0)
                    summary_data.append(summary_record)
            
            # Export in requested formats
            if 'csv' in self.config.export_formats:
                self._export_csv(all_cell_data, summary_data, export_info)
            
            if 'excel' in self.config.export_formats:
                self._export_excel(all_cell_data, summary_data, qc_report, export_info)
            
            if 'json' in self.config.export_formats:
                self._export_json(results, qc_report, export_info)
            
            if 'hdf5' in self.config.export_formats:
                self._export_hdf5(all_cell_data, summary_data, export_info)
            
            # Create comprehensive report
            self._create_comprehensive_report(results, qc_report, export_info)
            
            # Compress if requested
            if self.config.compress_results:
                self._compress_exports(export_info)
            
            logger.info(f"Batch export complete: {len(export_info['exported_formats'])} formats")
            return export_info
            
        except Exception as e:
            logger.error(f"Batch export error: {str(e)}")
            return {'error': str(e)}
    
    def _export_csv(self, cell_data: List[Dict], summary_data: List[Dict], export_info: Dict):
        """Export to CSV format."""
        try:
            if cell_data:
                cell_df = pd.DataFrame(cell_data)
                cell_path = self.export_directory / f"{self.config.job_name}_cell_data.csv"
                cell_df.to_csv(cell_path, index=False)
                export_info['export_paths']['cell_data_csv'] = str(cell_path)
            
            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                summary_path = self.export_directory / f"{self.config.job_name}_summary.csv"
                summary_df.to_csv(summary_path, index=False)
                export_info['export_paths']['summary_csv'] = str(summary_path)
            
            export_info['exported_formats'].append('csv')
            
        except Exception as e:
            logger.error(f"CSV export error: {str(e)}")
    
    def _export_excel(self, cell_data: List[Dict], summary_data: List[Dict], qc_report: Dict, export_info: Dict):
        """Export to Excel format with multiple sheets."""
        try:
            excel_path = self.export_directory / f"{self.config.job_name}_complete_analysis.xlsx"
            
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                # Cell data sheet
                if cell_data:
                    cell_df = pd.DataFrame(cell_data)
                    cell_df.to_excel(writer, sheet_name='Cell_Data', index=False)
                
                # Summary sheet
                if summary_data:
                    summary_df = pd.DataFrame(summary_data)
                    summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # QC report sheet
                if qc_report and 'error' not in qc_report:
                    qc_df = pd.DataFrame([qc_report])
                    qc_df.to_excel(writer, sheet_name='Quality_Control', index=False)
                
                # Metadata sheet
                metadata = {
                    'Job Name': [self.config.job_name],
                    'Export Timestamp': [datetime.now().isoformat()],
                    'Operator': [self.config.operator_name],
                    'Project ID': [self.config.project_id],
                    'Total Images': [export_info['total_results']],
                    'Export Formats': [', '.join(self.config.export_formats)]
                }
                metadata_df = pd.DataFrame(metadata)
                metadata_df.to_excel(writer, sheet_name='Metadata', index=False)
            
            export_info['export_paths']['excel'] = str(excel_path)
            export_info['exported_formats'].append('excel')
            
        except Exception as e:
            logger.error(f"Excel export error: {str(e)}")
    
    def _export_json(self, results: List[Dict], qc_report: Dict, export_info: Dict):
        """Export to JSON format."""
        try:
            json_data = {
                'metadata': {
                    'job_name': self.config.job_name,
                    'export_timestamp': datetime.now().isoformat(),
                    'total_results': len(results),
                    'configuration': asdict(self.config)
                },
                'results': results,
                'quality_control': qc_report
            }
            
            json_path = self.export_directory / f"{self.config.job_name}_complete.json"
            with open(json_path, 'w') as f:
                json.dump(json_data, f, indent=2, default=str)
            
            export_info['export_paths']['json'] = str(json_path)
            export_info['exported_formats'].append('json')
            
        except Exception as e:
            logger.error(f"JSON export error: {str(e)}")
    
    def _export_hdf5(self, cell_data: List[Dict], summary_data: List[Dict], export_info: Dict):
        """Export to HDF5 format for large datasets."""
        try:
            hdf5_path = self.export_directory / f"{self.config.job_name}_data.h5"
            
            with h5py.File(hdf5_path, 'w') as f:
                # Cell data
                if cell_data:
                    cell_df = pd.DataFrame(cell_data)
                    for col in cell_df.columns:
                        if cell_df[col].dtype == 'object':
                            # Convert object columns to strings
                            f.create_dataset(f'cell_data/{col}', data=cell_df[col].astype(str).values)
                        else:
                            f.create_dataset(f'cell_data/{col}', data=cell_df[col].values)
                
                # Summary data
                if summary_data:
                    summary_df = pd.DataFrame(summary_data)
                    for col in summary_df.columns:
                        if summary_df[col].dtype == 'object':
                            f.create_dataset(f'summary/{col}', data=summary_df[col].astype(str).values)
                        else:
                            f.create_dataset(f'summary/{col}', data=summary_df[col].values)
                
                # Metadata
                metadata_group = f.create_group('metadata')
                metadata_group.attrs['job_name'] = self.config.job_name
                metadata_group.attrs['export_timestamp'] = datetime.now().isoformat()
                metadata_group.attrs['total_results'] = export_info['total_results']
            
            export_info['export_paths']['hdf5'] = str(hdf5_path)
            export_info['exported_formats'].append('hdf5')
            
        except Exception as e:
            logger.error(f"HDF5 export error: {str(e)}")
    
    def _create_comprehensive_report(self, results: List[Dict], qc_report: Dict, export_info: Dict):
        """Create a comprehensive analysis report."""
        try:
            successful_results = [r for r in results if r.get('success', False)]
            
            report = {
                'executive_summary': {
                    'job_name': self.config.job_name,
                    'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'operator': self.config.operator_name,
                    'project_id': self.config.project_id,
                    'total_images_processed': len(results),
                    'successful_analyses': len(successful_results),
                    'success_rate_percentage': len(successful_results) / len(results) * 100 if results else 0
                },
                'aggregate_statistics': self._calculate_aggregate_statistics(successful_results),
                'quality_control_summary': qc_report,
                'recommendations': self._generate_batch_recommendations(successful_results, qc_report),
                'processing_performance': self._calculate_processing_performance(results),
                'export_information': export_info
            }
            
            # Save comprehensive report
            report_path = self.export_directory / f"{self.config.job_name}_comprehensive_report.json"
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            export_info['export_paths']['comprehensive_report'] = str(report_path)
            
        except Exception as e:
            logger.error(f"Comprehensive report creation error: {str(e)}")
    
    def _calculate_aggregate_statistics(self, results: List[Dict]) -> Dict:
        """Calculate aggregate statistics across all successful results."""
        try:
            if not results:
                return {}
            
            total_cells = sum(r.get('total_cells', 0) for r in results)
            
            # Collect all cell data
            all_areas = []
            all_chlorophyll = []
            
            for result in results:
                if 'cell_data' in result:
                    for cell in result['cell_data']:
                        if 'area' in cell:
                            all_areas.append(cell['area'])
                        if 'chlorophyll' in cell:
                            all_chlorophyll.append(cell['chlorophyll'])
            
            stats = {
                'total_cells_all_images': total_cells,
                'average_cells_per_image': total_cells / len(results),
                'cell_size_statistics': {
                    'mean_area': np.mean(all_areas) if all_areas else 0,
                    'median_area': np.median(all_areas) if all_areas else 0,
                    'std_area': np.std(all_areas) if all_areas else 0,
                    'min_area': np.min(all_areas) if all_areas else 0,
                    'max_area': np.max(all_areas) if all_areas else 0
                },
                'chlorophyll_statistics': {
                    'mean_chlorophyll': np.mean(all_chlorophyll) if all_chlorophyll else 0,
                    'high_chlorophyll_percentage': len([c for c in all_chlorophyll if c > 0.6]) / len(all_chlorophyll) * 100 if all_chlorophyll else 0
                }
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Aggregate statistics calculation error: {str(e)}")
            return {}
    
    def _generate_batch_recommendations(self, results: List[Dict], qc_report: Dict) -> List[str]:
        """Generate recommendations based on batch analysis."""
        try:
            recommendations = []
            
            if not results:
                return ["No successful analyses to evaluate"]
            
            # Success rate recommendations
            success_rate = len(results) / (len(results) + 1) * 100  # Simplified
            if success_rate < 80:
                recommendations.append("⚠️ Low success rate detected. Review image quality and processing parameters.")
            
            # Quality recommendations
            if qc_report and 'average_quality_score' in qc_report:
                avg_quality = qc_report['average_quality_score']
                if avg_quality < 0.6:
                    recommendations.append("📷 Image quality is below optimal. Consider improving lighting and focus.")
            
            # Cell count consistency
            cell_counts = [r.get('total_cells', 0) for r in results]
            if len(cell_counts) > 1:
                cv = np.std(cell_counts) / np.mean(cell_counts) if np.mean(cell_counts) > 0 else 0
                if cv > 0.8:
                    recommendations.append("📊 High variability in cell counts across images. Check for consistent sample preparation.")
            
            # Processing performance
            processing_times = [r.get('processing_time', 0) for r in results if 'processing_time' in r]
            if processing_times and np.mean(processing_times) > 30:
                recommendations.append("⚡ Consider optimizing processing parameters for faster analysis.")
            
            if not recommendations:
                recommendations.append("✅ Batch analysis completed successfully with consistent results.")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Batch recommendations generation error: {str(e)}")
            return ["Analysis complete. Review individual results for detailed insights."]
    
    def _calculate_processing_performance(self, results: List[Dict]) -> Dict:
        """Calculate processing performance metrics."""
        try:
            processing_times = [r.get('processing_time', 0) for r in results if 'processing_time' in r]
            
            if not processing_times:
                return {'message': 'No processing time data available'}
            
            return {
                'total_processing_time': sum(processing_times),
                'average_processing_time': np.mean(processing_times),
                'fastest_analysis': min(processing_times),
                'slowest_analysis': max(processing_times),
                'processing_efficiency': len([t for t in processing_times if t < 10]) / len(processing_times) * 100
            }
            
        except Exception as e:
            logger.error(f"Processing performance calculation error: {str(e)}")
            return {}
    
    def _compress_exports(self, export_info: Dict):
        """Compress exported files into a single archive."""
        try:
            zip_path = self.export_directory / f"{self.config.job_name}_complete_export.zip"
            
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for export_type, file_path in export_info['export_paths'].items():
                    if os.path.exists(file_path):
                        arcname = f"{export_type}_{os.path.basename(file_path)}"
                        zipf.write(file_path, arcname)
            
            export_info['compressed_export'] = str(zip_path)
            logger.info(f"Created compressed export: {zip_path}")
            
        except Exception as e:
            logger.error(f"Compression error: {str(e)}")


class BatchProcessor:
    """Main batch processing coordinator."""
    
    def __init__(self, analyzer_system, config: BatchJobConfig = None):
        """Initialize batch processor with analysis system."""
        self.analyzer_system = analyzer_system
        self.config = config or BatchJobConfig()
        self.quality_controller = QualityController(self.config)
        self.exporter = BatchExporter(self.config)
        self.progress_tracker = None
        self.is_running = False
        self.current_job_id = None
        
        # Results storage
        self.batch_results = []
        self.processing_errors = []
        
        logger.info(f"🚀 Batch Processor initialized: {self.config.job_name}")
    
    def process_batch(self, image_paths: List[str], progress_callback: Callable = None) -> Dict:
        """Process a batch of images with progress tracking and export."""
        try:
            self.is_running = True
            self.current_job_id = self.config.job_name
            self.batch_results = []
            self.processing_errors = []

            logger.info(f"🔄 Starting batch processing: {len(image_paths)} images")
            self.progress_tracker = BatchProgressTracker(len(image_paths))

            # Step 1: Quality pre-screening
            if self.config.enable_qc:
                screened_paths = self._quality_prescreening(image_paths, progress_callback)
                logger.info(f"✅ Quality screening complete: {len(screened_paths)}/{len(image_paths)} images approved")
            else:
                screened_paths = image_paths

            # Step 2: Handle empty batch post-QC
            if not screened_paths:
                logger.warning("⚠️ No images passed quality control. Skipping processing.")
                qc_report = self.quality_controller.generate_qc_report()
                export_info = self.exporter.export_results([], qc_report)

                self.is_running = False
                return {
                    'job_id': self.current_job_id,
                    'config': asdict(self.config),
                    'summary': {
                        'total_images': len(image_paths),
                        'successful_analyses': 0,
                        'failed_analyses': len(image_paths),
                        'success_rate': 0.0
                    },
                    'quality_control': qc_report,
                    'export_info': export_info,
                    'processing_time': (datetime.now() - self.progress_tracker.start_time).total_seconds()
                }

            # Step 3: Parallel analysis
            self._process_images_parallel(screened_paths, progress_callback)

            # Step 4: Post-analysis QC
            if self.config.enable_qc and self.config.outlier_detection:
                self.quality_controller.detect_batch_outliers(self.batch_results)

            # Step 5: Final QC report and export
            qc_report = self.quality_controller.generate_qc_report()
            export_info = self.exporter.export_results(self.batch_results, qc_report)

            # Step 6: Build and return final result
            successful = len([r for r in self.batch_results if r.get('success')])
            final_results = {
                'job_id': self.current_job_id,
                'config': asdict(self.config),
                'summary': {
                    'total_images': len(image_paths),
                    'successful_analyses': successful,
                    'failed_analyses': len(self.processing_errors) + (len(image_paths) - len(screened_paths)),
                    'success_rate': (successful / len(image_paths) * 100) if image_paths else 0.0
                },
                'quality_control': qc_report,
                'export_info': export_info,
                'processing_time': (datetime.now() - self.progress_tracker.start_time).total_seconds()
            }

            self.is_running = False
            logger.info(f"✅ Batch processing complete: {final_results['summary']['success_rate']:.1f}% success rate")
            return final_results

        except Exception as e:
            self.is_running = False
            logger.error(f"❌ Batch processing error: {str(e)}", exc_info=True)
            return {
                'job_id': self.current_job_id,
                'error': str(e),
                'completed_items': len(self.batch_results),
                'failed_items': len(self.processing_errors)
            }

    
    def _quality_prescreening(self, image_paths: List[str], progress_callback: Callable = None) -> List[str]:
        """Pre-screen images for quality before batch processing with tolerance for minor defects."""
        try:
            logger.info("🔍 Starting quality pre-screening...")
            approved_paths = []

            total_images = len(image_paths)
            for idx, image_path in enumerate(image_paths):
                try:
                    quality_assessment = self.quality_controller.assess_image_quality(image_path)
                    score = quality_assessment.get('quality_score', 0)
                    issues = quality_assessment.get('issues', [])

                    # Optimized decision logic: accept borderline images with minimal issues
                    passes_soft_qc = (
                        score >= self.config.quality_threshold or
                        (score >= (self.config.quality_threshold - 0.2) and len(issues) <= 2)
                    )

                    if passes_soft_qc:
                        approved_paths.append(image_path)
                        logger.info(f"✅ Image approved: {image_path} (Score: {score:.2f})")
                    else:
                        logger.warning(f"❌ Image failed QC: {image_path} - {issues} (Score: {score:.2f})")

                    if progress_callback:
                        progress_callback({
                            'stage': 'quality_screening',
                            'progress': (idx + 1) / total_images * 100,
                            'current_item': os.path.basename(image_path),
                            'approved_count': len(approved_paths),
                            'rejected_count': idx + 1 - len(approved_paths)
                        })

                except Exception as image_error:
                    logger.error(f"⚠️ Error assessing {image_path}: {str(image_error)}")
                    continue

            logger.info(f"✅ Quality screening complete: {len(approved_paths)}/{total_images} images approved")
            return approved_paths

        except Exception as e:
            logger.error(f"❌ Quality pre-screening failure: {str(e)}", exc_info=True)
            return image_paths

    
    def _process_images_parallel(self, image_paths: List[str], progress_callback: Callable):
        """Process images in parallel using ThreadPoolExecutor."""
        try:
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                # Submit all jobs
                future_to_path = {
                    executor.submit(self._process_single_image, path): path 
                    for path in image_paths
                }
                
                # Process completed jobs
                for future in as_completed(future_to_path):
                    image_path = future_to_path[future]
                    
                    try:
                        result = future.result()
                        self.batch_results.append(result)
                        
                        if result.get('success'):
                            self.progress_tracker.update_progress(status=f"Completed {os.path.basename(image_path)}")
                        else:
                            self.progress_tracker.report_failure()
                            self.processing_errors.append({
                                'image_path': image_path,
                                'error': result.get('error', 'Unknown error')
                            })
                        
                    except Exception as e:
                        logger.error(f"❌ Processing error for {image_path}: {str(e)}")
                        self.progress_tracker.report_failure()
                        self.processing_errors.append({
                            'image_path': image_path,
                            'error': str(e)
                        })
                    
                    # Update progress callback
                    if progress_callback:
                        progress_info = self.progress_tracker.get_progress_info()
                        progress_callback({
                            'stage': 'processing',
                            **progress_info,
                            'current_item': os.path.basename(image_path)
                        })
                        
        except Exception as e:
            logger.error(f"❌ Parallel processing error: {str(e)}")
    
    def _process_single_image(self, image_path: str) -> Dict:
        """
        Process a single image using the professional Wolffia analysis system.
        Falls back to run_pipeline if defined and analyzer signals Wolffia-compatible mode.
        """
        try:
            # Use primary analyzer if available
            if hasattr(self.analyzer_system, 'analyze_image_professional'):
                return self.analyzer_system.analyze_image_professional(image_path)

            # Custom Wolffia-compatible fallback using run_pipeline
            if getattr(self.analyzer_system, 'custom_segmentation_method', None) == 'wolffia_pipeline':
                import os
                from datetime import datetime

                import numpy as np

                from segmentation import run_pipeline  # Adjusted to real module

                dirpath, filename = os.path.split(image_path)
                segmentation, results = run_pipeline(dirpath, filename)

                return {
                    'image_path': image_path,
                    'success': True,
                    'timestamp': datetime.now().isoformat(),
                    'segmentation': segmentation.tolist(),
                    'summary': {
                        'total_cells': len(results["cell_id"]),
                        'avg_area': float(np.mean(results["cell_area"])) if results["cell_area"] else 0,
                        'chlorophyll_ratio': float(np.mean(results["int_mem_mean"])) if results["int_mem_mean"] else 0
                    },
                    'cell_data': [
                        {
                            'cell_id': int(cid),
                            'area': int(area),
                            'mem_intensity': float(mem),
                            'mean_intensity': float(mean),
                            'edge_length': int(edge)
                        }
                        for cid, area, mem, mean, edge in zip(
                            results["cell_id"],
                            results["cell_area"],
                            results["int_mem_mean"],
                            results["int_mean"],
                            results["cell_edge"]
                        )
                    ]
                }

            # Fallback
            return self.analyzer_system.analyze_single_image(image_path)

        except Exception as e:
            logger.error(f"❌ Single image processing error for {image_path}: {str(e)}")
            return {
                'image_path': image_path,
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
            }


    def get_progress_info(self) -> Dict:
        """Get current progress information."""
        if self.progress_tracker:
            return self.progress_tracker.get_progress_info()
        return {'message': 'No active batch job'}
    
    def cancel_batch(self):
        """Cancel current batch processing."""
        self.is_running = False
        logger.info("🛑 Batch processing cancelled by user")


# Integration with Flask for your existing dashboard
class BatchProcessorFlaskIntegration:
    """Flask integration for batch processing."""
    
    def __init__(self, app, analyzer_system):
        self.app = app
        self.analyzer_system = analyzer_system
        self.active_processors = {}  # job_id -> BatchProcessor
        
        # Add Flask routes
        self._add_batch_routes()
    
    def _add_batch_routes(self):
        """Add batch processing routes to Flask app."""

        from werkzeug.utils import secure_filename

        @self.app.route('/api/batch/start', methods=['POST'])
        def start_batch_job():
            try:
                if request.content_type.startswith('multipart/form-data'):
                    form = request.form
                    image_files = request.files.getlist('images')

                    job_name = form.get('job_name', f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                    max_workers = int(form.get('max_workers', 4))
                    enable_qc = form.get('enable_qc', 'true').lower() == 'true'

                    config = BatchJobConfig(
                        job_name=job_name,
                        max_workers=max_workers,
                        enable_qc=enable_qc
                    )

                    # Save images
                    image_paths = []
                    Path('temp_uploads').mkdir(parents=True, exist_ok=True)
                    for file in image_files:
                        filename = secure_filename(file.filename)
                        save_path = Path('temp_uploads') / f"{job_name}_{filename}"
                        file.save(save_path)
                        image_paths.append(str(save_path))

                    processor = BatchProcessor(self.analyzer_system, config)
                    self.active_processors[job_name] = processor

                    def run_job():
                        result = processor.process_batch(image_paths)
                        processor.final_result = result

                    threading.Thread(target=run_job).start()

                    return jsonify({
                        'job_id': job_name,
                        'status': 'started',
                        'total_images': len(image_paths)
                    })

                else:
                    return jsonify({'error': 'Unsupported Content-Type'}), 415

            except Exception as e:
                logger.error(f"❌ Batch start error: {str(e)}")
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/batch/progress/<job_id>')
        def get_batch_progress(job_id):
            """Get progress of a batch job."""
            try:
                if job_id not in self.active_processors:
                    return jsonify({'error': 'Job not found'}), 404

                processor = self.active_processors[job_id]
                progress_info = processor.get_progress_info()

                # Add job status
                progress_info['job_id'] = job_id
                progress_info['is_running'] = processor.is_running

                return jsonify(progress_info)

            except Exception as e:
                logger.error(f"❌ Progress retrieval error: {str(e)}")
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/batch/results/<job_id>')
        def get_batch_results(job_id):
            """Get results of a completed batch job."""
            try:
                if job_id not in self.active_processors:
                    return jsonify({'error': 'Job not found'}), 404

                processor = self.active_processors[job_id]

                if processor.is_running:
                    return jsonify({
                        'status': 'running',
                        'message': 'Job still in progress'
                    })

                if hasattr(processor, 'final_result'):
                    return jsonify(processor.final_result)
                else:
                    return jsonify({
                        'status': 'completed',
                        'results': processor.batch_results,
                        'errors': processor.processing_errors
                    })

            except Exception as e:
                logger.error(f"❌ Results retrieval error: {str(e)}")
                return jsonify({'error': str(e)}), 500


# Example usage
if __name__ == "__main__":
    print("🧪 Launching Wolffia Batch Processing System Test...")

    try:
        from pathlib import Path

        from wolffia_analyzer import WolffiaAnalyzer  # Real professional analyzer

        # Configuration
        job_name = f"test_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        input_dir = Path("sample_images")  # Change this to your real input directory
        image_paths = list(input_dir.glob("*.[jp][pn]g"))  # Matches .jpg, .jpeg, .png

        if not image_paths:
            raise FileNotFoundError("No test images found in 'sample_images' directory.")

        # Setup batch configuration
        config = BatchJobConfig(
            job_name=job_name,
            max_workers=2,
            export_formats=["csv", "json", "excel"],
            enable_qc=True,
            outlier_detection=True
        )

        analyzer = WolffiaAnalyzer()
        processor = BatchProcessor(analyzer, config)

        # Process
        result = processor.process_batch([str(p) for p in image_paths])

        # Output summary
        print("✅ Batch processing completed.")
        print(json.dumps(result["summary"], indent=2))
        print("📁 Exported files:")
        for fmt, path in result.get("export_info", {}).get("export_paths", {}).items():
            print(f" - {fmt}: {path}")

    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
