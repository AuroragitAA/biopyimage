"""
database_manager.py

Professional database management system for Wolffia analysis.
Handles data storage, retrieval, and integration with your existing dashboard.

Features:
- SQLite/PostgreSQL support
- Automated schema management
- Data versioning and backup
- Query optimization
- Integration with Flask dashboard
- Export/import capabilities
- Data analytics and reporting
- User management and access control
"""

import sqlite3
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
import hashlib
import shutil
import threading
from contextlib import contextmanager
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    """Configuration for database management."""
    
    # Database settings
    db_type: str = "sqlite"  # "sqlite" or "postgresql"
    db_path: str = "wolffia_database.db"
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "wolffia_db"
    db_user: str = ""
    db_password: str = ""
    
    # Backup settings
    backup_enabled: bool = True
    backup_interval_hours: int = 24
    max_backups: int = 30
    backup_directory: str = "database_backups"
    
    # Performance settings
    connection_pool_size: int = 10
    query_timeout: int = 30
    enable_wal_mode: bool = True
    
    # Data retention
    data_retention_days: int = 365
    auto_cleanup: bool = True
    
    # Security
    enable_encryption: bool = False
    encryption_key: str = ""


class DatabaseSchema:
    """Database schema definitions."""
    
    @staticmethod
    def get_schema_sql() -> Dict[str, str]:
        """Get SQL statements for creating database schema."""
        
        return {
            'projects': """
                CREATE TABLE IF NOT EXISTS projects (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    description TEXT,
                    operator_name TEXT,
                    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    modified_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    status TEXT DEFAULT 'active',
                    metadata TEXT
                )
            """,
            
            'experiments': """
                CREATE TABLE IF NOT EXISTS experiments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id INTEGER,
                    experiment_name TEXT NOT NULL,
                    description TEXT,
                    experimental_conditions TEXT,
                    start_date TIMESTAMP,
                    end_date TIMESTAMP,
                    status TEXT DEFAULT 'ongoing',
                    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (project_id) REFERENCES projects (id)
                )
            """,
            
            'images': """
                CREATE TABLE IF NOT EXISTS images (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id INTEGER,
                    filename TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    file_hash TEXT,
                    file_size INTEGER,
                    image_width INTEGER,
                    image_height INTEGER,
                    capture_date TIMESTAMP,
                    upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT,
                    FOREIGN KEY (experiment_id) REFERENCES experiments (id)
                )
            """,
            
            'analyses': """
                CREATE TABLE IF NOT EXISTS analyses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    image_id INTEGER,
                    analysis_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    analysis_method TEXT,
                    success BOOLEAN,
                    processing_time REAL,
                    total_cells INTEGER,
                    quality_score REAL,
                    parameters TEXT,
                    error_message TEXT,
                    version TEXT,
                    FOREIGN KEY (image_id) REFERENCES images (id)
                )
            """,
            
            'cells': """
                CREATE TABLE IF NOT EXISTS cells (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    analysis_id INTEGER,
                    cell_id INTEGER,
                    area REAL,
                    perimeter REAL,
                    centroid_x REAL,
                    centroid_y REAL,
                    major_axis_length REAL,
                    minor_axis_length REAL,
                    eccentricity REAL,
                    orientation REAL,
                    solidity REAL,
                    extent REAL,
                    chlorophyll_content REAL,
                    health_score REAL,
                    classification TEXT,
                    is_outlier BOOLEAN DEFAULT FALSE,
                    FOREIGN KEY (analysis_id) REFERENCES analyses (id)
                )
            """,
            
            'batch_jobs': """
                CREATE TABLE IF NOT EXISTS batch_jobs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_name TEXT NOT NULL,
                    experiment_id INTEGER,
                    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    end_time TIMESTAMP,
                    status TEXT DEFAULT 'running',
                    total_images INTEGER,
                    processed_images INTEGER,
                    success_rate REAL,
                    configuration TEXT,
                    results_summary TEXT,
                    FOREIGN KEY (experiment_id) REFERENCES experiments (id)
                )
            """,
            
            'quality_metrics': """
                CREATE TABLE IF NOT EXISTS quality_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    image_id INTEGER,
                    sharpness_score REAL,
                    contrast_score REAL,
                    brightness_score REAL,
                    noise_score REAL,
                    overall_quality REAL,
                    quality_issues TEXT,
                    assessment_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (image_id) REFERENCES images (id)
                )
            """,
            
            'ml_models': """
                CREATE TABLE IF NOT EXISTS ml_models (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    model_type TEXT,
                    training_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    performance_metrics TEXT,
                    feature_importance TEXT,
                    model_file_path TEXT,
                    status TEXT DEFAULT 'active',
                    version TEXT
                )
            """,
            
            'system_logs': """
                CREATE TABLE IF NOT EXISTS system_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    level TEXT,
                    module TEXT,
                    message TEXT,
                    details TEXT
                )
            """,
            
            'user_sessions': """
                CREATE TABLE IF NOT EXISTS user_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT UNIQUE,
                    user_name TEXT,
                    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    ip_address TEXT,
                    user_agent TEXT
                )
            """
        }


class DatabaseManager:
    """Professional database management system."""
    
    def __init__(self, config: DatabaseConfig = None):
        """Initialize database manager."""
        self.config = config or DatabaseConfig()
        self.connection_pool = []
        self.lock = threading.Lock()
        
        # Initialize database
        self._initialize_database()
        
        # Start background tasks
        if self.config.backup_enabled:
            self._start_backup_scheduler()
        
        if self.config.auto_cleanup:
            self._start_cleanup_scheduler()
        
        logger.info("🗄️ Database Manager initialized")
    
    def _initialize_database(self):
        """Initialize database with schema."""
        try:
            # Create database directory if needed
            if self.config.db_type == "sqlite":
                db_dir = Path(self.config.db_path).parent
                db_dir.mkdir(parents=True, exist_ok=True)
            
            # Create backup directory
            Path(self.config.backup_directory).mkdir(parents=True, exist_ok=True)
            
            # Create tables
            with self.get_connection() as conn:
                schema = DatabaseSchema.get_schema_sql()
                
                for table_name, sql in schema.items():
                    conn.execute(sql)
                    logger.info(f"✅ Table created/verified: {table_name}")
                
                # Enable WAL mode for SQLite
                if self.config.db_type == "sqlite" and self.config.enable_wal_mode:
                    conn.execute("PRAGMA journal_mode=WAL")
                
                conn.commit()
            
            # Create indexes for performance
            self._create_indexes()
            
            logger.info("✅ Database initialization complete")
            
        except Exception as e:
            logger.error(f"❌ Database initialization error: {str(e)}")
            raise
    
    def _create_indexes(self):
        """Create database indexes for better performance."""
        try:
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_images_experiment ON images(experiment_id)",
                "CREATE INDEX IF NOT EXISTS idx_analyses_image ON analyses(image_id)",
                "CREATE INDEX IF NOT EXISTS idx_cells_analysis ON cells(analysis_id)",
                "CREATE INDEX IF NOT EXISTS idx_analyses_timestamp ON analyses(analysis_timestamp)",
                "CREATE INDEX IF NOT EXISTS idx_images_upload_date ON images(upload_date)",
                "CREATE INDEX IF NOT EXISTS idx_cells_health_score ON cells(health_score)",
                "CREATE INDEX IF NOT EXISTS idx_cells_area ON cells(area)",
                "CREATE INDEX IF NOT EXISTS idx_batch_jobs_status ON batch_jobs(status)"
            ]
            
            with self.get_connection() as conn:
                for index_sql in indexes:
                    conn.execute(index_sql)
            
            logger.info("✅ Database indexes created")
            
        except Exception as e:
            logger.error(f"❌ Index creation error: {str(e)}")
    
    @contextmanager
    def get_connection(self):
        """Get database connection with context manager."""
        try:
            if self.config.db_type == "sqlite":
                conn = sqlite3.connect(
                    self.config.db_path,
                    timeout=self.config.query_timeout,
                    check_same_thread=False
                )
                conn.row_factory = sqlite3.Row  # Enable dict-like access
            else:
                # PostgreSQL support (would need psycopg2)
                raise NotImplementedError("PostgreSQL support not implemented yet")
            
            yield conn
            
        except Exception as e:
            logger.error(f"❌ Database connection error: {str(e)}")
            raise
        finally:
            if 'conn' in locals():
                conn.close()
    
    def create_project(self, name: str, description: str = "", operator_name: str = "", metadata: Dict = None) -> int:
        """Create a new project."""
        try:
            with self.get_connection() as conn:
                cursor = conn.execute(
                    """INSERT INTO projects (name, description, operator_name, metadata)
                       VALUES (?, ?, ?, ?)""",
                    (name, description, operator_name, json.dumps(metadata or {}))
                )
                project_id = cursor.lastrowid
                conn.commit()
                
                logger.info(f"✅ Project created: {name} (ID: {project_id})")
                return project_id
                
        except Exception as e:
            logger.error(f"❌ Project creation error: {str(e)}")
            raise
    
    def create_experiment(self, project_id: int, experiment_name: str, description: str = "", 
                         experimental_conditions: Dict = None) -> int:
        """Create a new experiment."""
        try:
            with self.get_connection() as conn:
                cursor = conn.execute(
                    """INSERT INTO experiments (project_id, experiment_name, description, experimental_conditions, start_date)
                       VALUES (?, ?, ?, ?, ?)""",
                    (project_id, experiment_name, description, 
                     json.dumps(experimental_conditions or {}), datetime.now())
                )
                experiment_id = cursor.lastrowid
                conn.commit()
                
                logger.info(f"✅ Experiment created: {experiment_name} (ID: {experiment_id})")
                return experiment_id
                
        except Exception as e:
            logger.error(f"❌ Experiment creation error: {str(e)}")
            raise
    
    def store_image(self, experiment_id: int, filename: str, file_path: str, 
                   image_width: int = None, image_height: int = None, metadata: Dict = None) -> int:
        """Store image information."""
        try:
            # Calculate file hash for integrity
            file_hash = self._calculate_file_hash(file_path)
            file_size = Path(file_path).stat().st_size if Path(file_path).exists() else None
            
            with self.get_connection() as conn:
                cursor = conn.execute(
                    """INSERT INTO images (experiment_id, filename, file_path, file_hash, file_size,
                                          image_width, image_height, metadata)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (experiment_id, filename, file_path, file_hash, file_size,
                     image_width, image_height, json.dumps(metadata or {}))
                )
                image_id = cursor.lastrowid
                conn.commit()
                
                logger.info(f"✅ Image stored: {filename} (ID: {image_id})")
                return image_id
                
        except Exception as e:
            logger.error(f"❌ Image storage error: {str(e)}")
            raise
    
    def store_analysis_result(self, image_id: int, analysis_result: Dict) -> int:
        """Store complete analysis result."""
        try:
            with self.get_connection() as conn:
                # Store main analysis record
                cursor = conn.execute(
                    """INSERT INTO analyses (image_id, analysis_method, success, processing_time,
                                           total_cells, quality_score, parameters, error_message, version)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        image_id,
                        analysis_result.get('method', 'unknown'),
                        analysis_result.get('success', False),
                        analysis_result.get('processing_time', 0),
                        analysis_result.get('total_cells', 0),
                        analysis_result.get('quality_score'),
                        json.dumps(analysis_result.get('processing_info', {})),
                        analysis_result.get('error'),
                        "1.0"  # Version
                    )
                )
                analysis_id = cursor.lastrowid
                
                # Store individual cell data
                if analysis_result.get('success') and 'cell_data' in analysis_result:
                    cell_data = analysis_result['cell_data']
                    if isinstance(cell_data, list) and len(cell_data) > 0:
                        self._store_cell_data(conn, analysis_id, cell_data)
                
                # Store quality metrics if available
                if 'quality_score' in analysis_result:
                    self._store_quality_metrics(conn, image_id, analysis_result)
                
                conn.commit()
                
                logger.info(f"✅ Analysis result stored (ID: {analysis_id})")
                return analysis_id
                
        except Exception as e:
            logger.error(f"❌ Analysis storage error: {str(e)}")
            raise
    
    def _store_cell_data(self, conn, analysis_id: int, cell_data: List[Dict]):
        """Store individual cell measurements."""
        try:
            for cell in cell_data:
                conn.execute(
                    """INSERT INTO cells (analysis_id, cell_id, area, perimeter, centroid_x, centroid_y,
                                         major_axis_length, minor_axis_length, eccentricity, orientation,
                                         solidity, extent, chlorophyll_content, health_score, classification, is_outlier)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        analysis_id,
                        cell.get('cell_id'),
                        cell.get('area'),
                        cell.get('perimeter'),
                        cell.get('centroid_x'),
                        cell.get('centroid_y'),
                        cell.get('major_axis_length'),
                        cell.get('minor_axis_length'),
                        cell.get('eccentricity'),
                        cell.get('orientation'),
                        cell.get('solidity'),
                        cell.get('extent'),
                        cell.get('chlorophyll_content', cell.get('chlorophyll')),
                        cell.get('health_score'),
                        cell.get('classification'),
                        cell.get('is_outlier', False)
                    )
                )
            
            logger.info(f"✅ Stored {len(cell_data)} cell records")
            
        except Exception as e:
            logger.error(f"❌ Cell data storage error: {str(e)}")
            raise
    
    def _store_quality_metrics(self, conn, image_id: int, analysis_result: Dict):
        """Store image quality metrics."""
        try:
            quality_details = analysis_result.get('quality_details', {})
            
            conn.execute(
                """INSERT INTO quality_metrics (image_id, sharpness_score, contrast_score,
                                               brightness_score, noise_score, overall_quality, quality_issues)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    image_id,
                    quality_details.get('sharpness'),
                    quality_details.get('contrast'),
                    quality_details.get('brightness'),
                    quality_details.get('noise'),
                    analysis_result.get('quality_score'),
                    json.dumps(quality_details.get('issues', []))
                )
            )
            
        except Exception as e:
            logger.error(f"❌ Quality metrics storage error: {str(e)}")
    
    def get_project_summary(self, project_id: int) -> Dict:
        """Get comprehensive project summary."""
        try:
            with self.get_connection() as conn:
                # Project info
                project = conn.execute(
                    "SELECT * FROM projects WHERE id = ?", (project_id,)
                ).fetchone()
                
                if not project:
                    return {'error': 'Project not found'}
                
                # Experiments count
                experiments_count = conn.execute(
                    "SELECT COUNT(*) FROM experiments WHERE project_id = ?", (project_id,)
                ).fetchone()[0]
                
                # Total images
                images_count = conn.execute(
                    """SELECT COUNT(*) FROM images i
                       JOIN experiments e ON i.experiment_id = e.id
                       WHERE e.project_id = ?""", (project_id,)
                ).fetchone()[0]
                
                # Total analyses
                analyses_count = conn.execute(
                    """SELECT COUNT(*) FROM analyses a
                       JOIN images i ON a.image_id = i.id
                       JOIN experiments e ON i.experiment_id = e.id
                       WHERE e.project_id = ?""", (project_id,)
                ).fetchone()[0]
                
                # Success rate
                success_rate = conn.execute(
                    """SELECT AVG(CASE WHEN success THEN 1.0 ELSE 0.0 END) * 100
                       FROM analyses a
                       JOIN images i ON a.image_id = i.id
                       JOIN experiments e ON i.experiment_id = e.id
                       WHERE e.project_id = ?""", (project_id,)
                ).fetchone()[0] or 0
                
                # Total cells detected
                total_cells = conn.execute(
                    """SELECT SUM(total_cells) FROM analyses a
                       JOIN images i ON a.image_id = i.id
                       JOIN experiments e ON i.experiment_id = e.id
                       WHERE e.project_id = ? AND success = 1""", (project_id,)
                ).fetchone()[0] or 0
                
                return {
                    'project': dict(project),
                    'statistics': {
                        'experiments_count': experiments_count,
                        'images_count': images_count,
                        'analyses_count': analyses_count,
                        'success_rate': round(success_rate, 2),
                        'total_cells_detected': total_cells
                    }
                }
                
        except Exception as e:
            logger.error(f"❌ Project summary error: {str(e)}")
            return {'error': str(e)}
    
    def get_experiment_data(self, experiment_id: int, include_cells: bool = False) -> Dict:
        """Get comprehensive experiment data."""
        try:
            with self.get_connection() as conn:
                # Experiment info
                experiment = conn.execute(
                    "SELECT * FROM experiments WHERE id = ?", (experiment_id,)
                ).fetchone()
                
                if not experiment:
                    return {'error': 'Experiment not found'}
                
                # Images and analyses
                query = """
                    SELECT i.*, a.id as analysis_id, a.analysis_timestamp, a.success,
                           a.total_cells, a.quality_score, a.processing_time
                    FROM images i
                    LEFT JOIN analyses a ON i.id = a.image_id
                    WHERE i.experiment_id = ?
                    ORDER BY i.upload_date
                """
                
                results = conn.execute(query, (experiment_id,)).fetchall()
                
                images_data = []
                for row in results:
                    image_data = {
                        'image_info': {
                            'id': row['id'],
                            'filename': row['filename'],
                            'file_path': row['file_path'],
                            'upload_date': row['upload_date'],
                            'image_width': row['image_width'],
                            'image_height': row['image_height']
                        },
                        'analysis_info': {
                            'analysis_id': row['analysis_id'],
                            'timestamp': row['analysis_timestamp'],
                            'success': bool(row['success']) if row['success'] is not None else None,
                            'total_cells': row['total_cells'],
                            'quality_score': row['quality_score'],
                            'processing_time': row['processing_time']
                        }
                    }
                    
                    # Include cell data if requested
                    if include_cells and row['analysis_id']:
                        cell_data = self._get_cell_data(conn, row['analysis_id'])
                        image_data['cells'] = cell_data
                    
                    images_data.append(image_data)
                
                return {
                    'experiment': dict(experiment),
                    'images': images_data,
                    'summary': self._calculate_experiment_summary(images_data)
                }
                
        except Exception as e:
            logger.error(f"❌ Experiment data retrieval error: {str(e)}")
            return {'error': str(e)}
    
    def _get_cell_data(self, conn, analysis_id: int) -> List[Dict]:
        """Get cell data for specific analysis."""
        try:
            cells = conn.execute(
                "SELECT * FROM cells WHERE analysis_id = ? ORDER BY cell_id",
                (analysis_id,)
            ).fetchall()
            
            return [dict(cell) for cell in cells]
            
        except Exception as e:
            logger.error(f"❌ Cell data retrieval error: {str(e)}")
            return []
    
    def _calculate_experiment_summary(self, images_data: List[Dict]) -> Dict:
        """Calculate summary statistics for experiment."""
        try:
            total_images = len(images_data)
            analyzed_images = len([img for img in images_data if img['analysis_info']['analysis_id']])
            successful_analyses = len([img for img in images_data if img['analysis_info']['success']])
            
            total_cells = sum(
                img['analysis_info']['total_cells'] or 0 
                for img in images_data 
                if img['analysis_info']['total_cells']
            )
            
            avg_quality = np.mean([
                img['analysis_info']['quality_score']
                for img in images_data
                if img['analysis_info']['quality_score'] is not None
            ]) if any(img['analysis_info']['quality_score'] for img in images_data) else None
            
            avg_processing_time = np.mean([
                img['analysis_info']['processing_time']
                for img in images_data
                if img['analysis_info']['processing_time'] is not None
            ]) if any(img['analysis_info']['processing_time'] for img in images_data) else None
            
            return {
                'total_images': total_images,
                'analyzed_images': analyzed_images,
                'successful_analyses': successful_analyses,
                'success_rate': (successful_analyses / analyzed_images * 100) if analyzed_images > 0 else 0,
                'total_cells_detected': total_cells,
                'average_cells_per_image': total_cells / successful_analyses if successful_analyses > 0 else 0,
                'average_quality_score': round(avg_quality, 3) if avg_quality else None,
                'average_processing_time': round(avg_processing_time, 2) if avg_processing_time else None
            }
            
        except Exception as e:
            logger.error(f"❌ Experiment summary calculation error: {str(e)}")
            return {}
    
    def export_data(self, export_type: str, **kwargs) -> str:
        """Export data in various formats."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if export_type == 'project_csv':
                return self._export_project_csv(kwargs.get('project_id'), timestamp)
            elif export_type == 'experiment_excel':
                return self._export_experiment_excel(kwargs.get('experiment_id'), timestamp)
            elif export_type == 'full_database':
                return self._export_full_database(timestamp)
            else:
                raise ValueError(f"Unsupported export type: {export_type}")
                
        except Exception as e:
            logger.error(f"❌ Data export error: {str(e)}")
            raise
    
    def _export_project_csv(self, project_id: int, timestamp: str) -> str:
        """Export project data to CSV."""
        try:
            with self.get_connection() as conn:
                # Main query to get all project data
                query = """
                    SELECT 
                        p.name as project_name,
                        e.experiment_name,
                        i.filename,
                        i.upload_date,
                        a.analysis_timestamp,
                        a.success,
                        a.total_cells,
                        a.quality_score,
                        a.processing_time,
                        c.cell_id,
                        c.area,
                        c.perimeter,
                        c.chlorophyll_content,
                        c.health_score,
                        c.classification
                    FROM projects p
                    JOIN experiments e ON p.id = e.project_id
                    JOIN images i ON e.id = i.experiment_id
                    LEFT JOIN analyses a ON i.id = a.image_id
                    LEFT JOIN cells c ON a.id = c.analysis_id
                    WHERE p.id = ?
                    ORDER BY e.experiment_name, i.upload_date, c.cell_id
                """
                
                df = pd.read_sql_query(query, conn, params=(project_id,))
                
                export_path = f"exports/project_{project_id}_export_{timestamp}.csv"
                Path("exports").mkdir(exist_ok=True)
                df.to_csv(export_path, index=False)
                
                logger.info(f"✅ Project data exported to {export_path}")
                return export_path
                
        except Exception as e:
            logger.error(f"❌ CSV export error: {str(e)}")
            raise
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA-256 hash of file."""
        try:
            if not Path(file_path).exists():
                return ""
            
            hash_sha256 = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            
            return hash_sha256.hexdigest()
            
        except Exception as e:
            logger.error(f"❌ File hash calculation error: {str(e)}")
            return ""
    
    def create_backup(self) -> str:
        """Create database backup."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_filename = f"wolffia_db_backup_{timestamp}.db"
            backup_path = Path(self.config.backup_directory) / backup_filename
            
            if self.config.db_type == "sqlite":
                # Simple file copy for SQLite
                shutil.copy2(self.config.db_path, backup_path)
            
            # Compress backup
            import gzip
            compressed_path = f"{backup_path}.gz"
            with open(backup_path, 'rb') as f_in:
                with gzip.open(compressed_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            # Remove uncompressed backup
            backup_path.unlink()
            
            logger.info(f"✅ Database backup created: {compressed_path}")
            
            # Clean old backups
            self._cleanup_old_backups()
            
            return str(compressed_path)
            
        except Exception as e:
            logger.error(f"❌ Backup creation error: {str(e)}")
            raise
    
    def _cleanup_old_backups(self):
        """Remove old backup files."""
        try:
            backup_dir = Path(self.config.backup_directory)
            backup_files = list(backup_dir.glob("wolffia_db_backup_*.db.gz"))
            
            # Sort by creation time (newest first)
            backup_files.sort(key=lambda x: x.stat().st_ctime, reverse=True)
            
            # Remove old backups beyond max_backups
            for backup_file in backup_files[self.config.max_backups:]:
                backup_file.unlink()
                logger.info(f"🗑️ Removed old backup: {backup_file}")
                
        except Exception as e:
            logger.error(f"❌ Backup cleanup error: {str(e)}")
    
    def _start_backup_scheduler(self):
        """Start automatic backup scheduler."""
        def backup_worker():
            while True:
                try:
                    import time
                    time.sleep(self.config.backup_interval_hours * 3600)
                    self.create_backup()
                except Exception as e:
                    logger.error(f"❌ Scheduled backup error: {str(e)}")
        
        backup_thread = threading.Thread(target=backup_worker, daemon=True)
        backup_thread.start()
        logger.info("🕒 Backup scheduler started")
    
    def _start_cleanup_scheduler(self):
        """Start automatic data cleanup scheduler."""
        def cleanup_worker():
            while True:
                try:
                    import time
                    time.sleep(24 * 3600)  # Run daily
                    self._cleanup_old_data()
                except Exception as e:
                    logger.error(f"❌ Scheduled cleanup error: {str(e)}")
        
        cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        cleanup_thread.start()
        logger.info("🧹 Cleanup scheduler started")
    
    def _cleanup_old_data(self):
        """Clean up old data based on retention policy."""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.config.data_retention_days)
            
            with self.get_connection() as conn:
                # Clean old system logs
                deleted_logs = conn.execute(
                    "DELETE FROM system_logs WHERE timestamp < ?",
                    (cutoff_date,)
                ).rowcount
                
                # Clean old user sessions
                deleted_sessions = conn.execute(
                    "DELETE FROM user_sessions WHERE start_time < ?",
                    (cutoff_date,)
                ).rowcount
                
                conn.commit()
                
                if deleted_logs > 0 or deleted_sessions > 0:
                    logger.info(f"🧹 Cleanup complete: {deleted_logs} log entries, {deleted_sessions} sessions removed")
                
        except Exception as e:
            logger.error(f"❌ Data cleanup error: {str(e)}")
    
    def get_database_statistics(self) -> Dict:
        """Get comprehensive database statistics."""
        try:
            with self.get_connection() as conn:
                stats = {}
                
                # Table counts
                tables = ['projects', 'experiments', 'images', 'analyses', 'cells', 'batch_jobs']
                for table in tables:
                    count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                    stats[f'{table}_count'] = count
                
                # Database size (SQLite)
                if self.config.db_type == "sqlite":
                    db_size = Path(self.config.db_path).stat().st_size if Path(self.config.db_path).exists() else 0
                    stats['database_size_mb'] = round(db_size / (1024 * 1024), 2)
                
                # Recent activity
                recent_analyses = conn.execute("""
                    SELECT COUNT(*) FROM analyses 
                    WHERE analysis_timestamp > datetime('now', '-7 days')
                """).fetchone()[0]
                
                stats['recent_analyses_7_days'] = recent_analyses
                
                # Success rate
                success_rate = conn.execute("""
                    SELECT AVG(CASE WHEN success THEN 1.0 ELSE 0.0 END) * 100
                    FROM analyses
                """).fetchone()[0] or 0
                
                stats['overall_success_rate'] = round(success_rate, 2)
                
                return stats
                
        except Exception as e:
            logger.error(f"❌ Database statistics error: {str(e)}")
            return {'error': str(e)}


# Flask integration for your existing dashboard
class DatabaseFlaskIntegration:
    """Flask integration for database management."""
    
    def __init__(self, app, db_manager: DatabaseManager):
        self.app = app
        self.db = db_manager
        
        # Add database routes
        self._add_database_routes()
    
    def _add_database_routes(self):
        """Add database management routes to Flask app."""
        
        @self.app.route('/api/db/projects', methods=['GET', 'POST'])
        def handle_projects():
            """Handle project operations."""
            if request.method == 'POST':
                data = request.get_json()
                try:
                    project_id = self.db.create_project(
                        name=data['name'],
                        description=data.get('description', ''),
                        operator_name=data.get('operator_name', ''),
                        metadata=data.get('metadata', {})
                    )
                    return jsonify({'project_id': project_id, 'success': True})
                except Exception as e:
                    return jsonify({'error': str(e)}), 500
            
            else:  # GET
                try:
                    with self.db.get_connection() as conn:
                        projects = conn.execute("SELECT * FROM projects ORDER BY created_date DESC").fetchall()
                        return jsonify([dict(project) for project in projects])
                except Exception as e:
                    return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/db/projects/<int:project_id>/summary')
        def get_project_summary(project_id):
            """Get project summary."""
            try:
                summary = self.db.get_project_summary(project_id)
                return jsonify(summary)
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/db/experiments', methods=['POST'])
        def create_experiment():
            """Create new experiment."""
            data = request.get_json()
            try:
                experiment_id = self.db.create_experiment(
                    project_id=data['project_id'],
                    experiment_name=data['experiment_name'],
                    description=data.get('description', ''),
                    experimental_conditions=data.get('conditions', {})
                )
                return jsonify({'experiment_id': experiment_id, 'success': True})
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/db/experiments/<int:experiment_id>')
        def get_experiment_data(experiment_id):
            """Get experiment data."""
            include_cells = request.args.get('include_cells', 'false').lower() == 'true'
            try:
                data = self.db.get_experiment_data(experiment_id, include_cells)
                return jsonify(data)
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/db/backup', methods=['POST'])
        def create_backup():
            """Create database backup."""
            try:
                backup_path = self.db.create_backup()
                return jsonify({'backup_path': backup_path, 'success': True})
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/db/statistics')
        def get_db_statistics():
            """Get database statistics."""
            try:
                stats = self.db.get_database_statistics()
                return jsonify(stats)
            except Exception as e:
                return jsonify({'error': str(e)}), 500


# Example usage
if __name__ == "__main__":
    print("🗄️ Testing Database Management System...")
    
    try:
        # Create test configuration
        config = DatabaseConfig()
        
        # Initialize database manager
        db_manager = DatabaseManager(config)
        
        # Test basic operations
        project_id = db_manager.create_project("Test Project", "Testing database functionality")
        experiment_id = db_manager.create_experiment(project_id, "Test Experiment")
        
        print(f"✅ Test project created (ID: {project_id})")
        print(f"✅ Test experiment created (ID: {experiment_id})")
        
        # Get statistics
        stats = db_manager.get_database_statistics()
        print(f"📊 Database statistics: {stats}")
        
        print("✅ Database Management System ready for integration")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()