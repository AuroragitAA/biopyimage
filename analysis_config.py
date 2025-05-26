"""
analysis_config.py - Fixed and Complete Analysis Configuration
Resolves all AttributeError issues with missing nested configurations
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import plotly.graph_objects as go
from plotly.subplots import make_subplots


@dataclass
class AnalysisConfig:
    """Complete production-ready configuration with all required nested fields."""
    
    # Core parameters
    pixel_to_micron: float = 1.0
    min_cell_area: int = 30
    max_cell_area: int = 8000
    chlorophyll_threshold: float = 0.6
    
    # Processing parameters
    noise_reduction_sigma: float = 0.8
    contrast_enhancement_clip: float = 0.03
    multi_scale_levels: int = 3
    watershed_min_distance: int = 8
    adaptive_block_size: int = 21
    
    # Quality control
    min_image_quality_score: float = 0.7
    outlier_detection_contamination: float = 0.1
    confidence_interval: float = 0.95
    
    # Health score weights
    health_score_weights: Dict[str, float] = field(default_factory=lambda: {
        'chlorophyll_content': 0.3,
        'cell_integrity': 0.25,
        'size_consistency': 0.2,
        'texture_uniformity': 0.15,
        'shape_regularity': 0.1
    })
    
    # FIXED: Add all missing nested configuration sections
    preprocessing: Dict[str, Any] = field(default_factory=lambda: {
        'noise_reduction': True,
        'contrast_enhancement': True,
        'illumination_correction': True,
        'denoise_strength': 'moderate',
        'contrast_method': 'clahe'
    })
    
    thresholding: Dict[str, Any] = field(default_factory=lambda: {
        'methods': ['otsu', 'adaptive', 'multiotsu', 'li'],
        'adaptive_block_size': 21,
        'adaptive_c': 2,
        'manual_threshold': None,
        'voting_threshold': 2
    })
    
    morphology: Dict[str, Any] = field(default_factory=lambda: {
        'opening_size': 2,
        'closing_size': 3,
        'remove_small_objects': 50,
        'fill_holes': True,
        'hole_fill_threshold': 200
    })
    
    postprocessing: Dict[str, Any] = field(default_factory=lambda: {
        'remove_border': True,
        'border_width': 10,
        'size_filter': True,
        'shape_filter': True,
        'min_area': 30,  # Will be set to min_cell_area in __post_init__
        'max_area': 8000,  # Will be set to max_cell_area in __post_init__
        'min_circularity': 0.3,
        'max_eccentricity': 0.95
    })
    
    # Biological analysis parameters
    biological_analysis: Dict[str, Any] = field(default_factory=lambda: {
        'enable_biomass_calculation': True,
        'enable_chlorophyll_analysis': True,
        'enable_temporal_tracking': True,
        'enable_similarity_analysis': True,
        'biomass_conversion_factor': 0.0012,  # mg/mm²
        'dry_weight_ratio': 0.12  # 12% dry weight
    })
    
    # Visualization parameters
    visualization: Dict[str, Any] = field(default_factory=lambda: {
        'create_debug_images': True,
        'save_intermediate_steps': True,
        'color_scheme': 'viridis',
        'figure_dpi': 150,
        'max_cells_to_label': 100
    })
    
    def __post_init__(self):
        """Ensure consistency between main parameters and nested configs."""
        # Sync main parameters with nested configs
        self.postprocessing['min_area'] = self.min_cell_area
        self.postprocessing['max_area'] = self.max_cell_area
        self.thresholding['adaptive_block_size'] = self.adaptive_block_size
        
        # Validate parameters
        if self.adaptive_block_size % 2 == 0:
            self.adaptive_block_size += 1
            self.thresholding['adaptive_block_size'] = self.adaptive_block_size
        
        # Set minimum values for critical parameters
        self.min_cell_area = max(self.min_cell_area, 10)
        self.max_cell_area = max(self.max_cell_area, self.min_cell_area * 2)
        self.watershed_min_distance = max(self.watershed_min_distance, 3)
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'AnalysisConfig':
        """Create AnalysisConfig from dictionary, handling missing keys gracefully."""
        # Create default instance
        config = cls()
        
        # Override with provided values
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        # Trigger __post_init__ to ensure consistency
        config.__post_init__()
        
        return config
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        import dataclasses
        return dataclasses.asdict(self)
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []
        
        if self.pixel_to_micron <= 0:
            issues.append("pixel_to_micron must be positive")
        
        if self.min_cell_area >= self.max_cell_area:
            issues.append("min_cell_area must be less than max_cell_area")
        
        if not (0 < self.chlorophyll_threshold < 1):
            issues.append("chlorophyll_threshold must be between 0 and 1")
        
        if self.adaptive_block_size < 3 or self.adaptive_block_size % 2 == 0:
            issues.append("adaptive_block_size must be odd and >= 3")
        
        return issues


def ensure_analysis_config(config_input) -> AnalysisConfig:
    """
    Utility function to ensure we always have a proper AnalysisConfig instance.
    Handles dict, None, or existing AnalysisConfig inputs.
    """
    if config_input is None:
        return AnalysisConfig()
    elif isinstance(config_input, dict):
        return AnalysisConfig.from_dict(config_input)
    elif isinstance(config_input, AnalysisConfig):
        return config_input
    else:
        # Fallback for any other type
        return AnalysisConfig()


"""
logging_config.py - Fixed Unicode-Safe Logging Configuration
Handles emoji characters properly on all platforms including Windows cmd/PowerShell
"""

import locale
import logging
import os
import sys
from datetime import datetime
from pathlib import Path


class SafeFormatter(logging.Formatter):
    """Safe formatter that handles emoji and Unicode characters properly."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Detect platform and terminal capabilities
        self.use_emojis = self._detect_emoji_support()
        
        # Emoji fallbacks for Windows cmd/PowerShell and other systems
        self.emoji_fallbacks = {
            '✅': '[OK]', '❌': '[ERROR]', '⚠️': '[WARN]', '🔬': '[ANALYSIS]',
            '📊': '[STATS]', '🗑️': '[CLEAN]', '📁': '[FILES]', '🔄': '[PROCESS]',
            '🚀': '[START]', '💾': '[SAVE]', '🎯': '[TARGET]', '📈': '[PROGRESS]',
            '🔍': '[SEARCH]', '🧬': '[BIO]', '🤖': '[ML]', '🌱': '[SYSTEM]',
            '📷': '[IMAGE]', '🎨': '[VIZ]', '👋': '[BYE]', '🔧': '[CONFIG]',
            '📋': '[REPORT]', '🔗': '[LINK]', '⚡': '[FAST]', '🏥': '[HEALTH]',
            '🌐': '[WEB]', '💡': '[INFO]', '🧪': '[TEST]', '📦': '[BACKUP]',
            '🎉': '[SUCCESS]', '🕒': '[TIME]', '🧹': '[CLEANUP]'
        }
    
    def _detect_emoji_support(self):
        """Detect if the current environment supports emojis."""
        try:
            # Always disable emojis on Windows to avoid encoding errors
            if os.name == 'nt':  # Windows
                return False
                
            # For Unix systems, test if we can encode emojis
            test_emoji = "✅"
            test_emoji.encode('utf-8')
            
            # Check if stdout supports UTF-8
            if hasattr(sys.stdout, 'encoding'):
                encoding = sys.stdout.encoding.lower()
                if 'utf' not in encoding and encoding != 'ascii':
                    return False
                    
            return True
            
        except (UnicodeEncodeError, AttributeError):
            return False
    
    def format(self, record):
        """Format log record with safe Unicode handling."""
        try:
            # Get the formatted message
            msg = super().format(record)
            
            # Replace emojis if not supported
            if not self.use_emojis:
                for emoji, fallback in self.emoji_fallbacks.items():
                    msg = msg.replace(emoji, fallback)
            
            # Ensure the message is safely encodable
            if isinstance(msg, str):
                # Try to encode and decode to catch any problematic characters
                try:
                    msg.encode('utf-8', errors='replace').decode('utf-8')
                except UnicodeError:
                    # Fallback to ASCII-safe version
                    msg = msg.encode('ascii', errors='replace').decode('ascii')
            
            return msg
            
        except Exception as e:
            # Ultimate fallback for any formatting issues
            try:
                safe_msg = str(record.msg).encode('ascii', 'replace').decode('ascii')
                return f"[{record.levelname}] {record.name}: {safe_msg}"
            except:
                return f"[{record.levelname}] LOGGING_ERROR: {str(e)}"


def setup_production_logging(log_level=logging.INFO, log_dir="logs"):
    """Setup production-ready logging system with Unicode safety."""
    
    # Ensure log directory exists
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Create formatters
    console_formatter = SafeFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    file_formatter = SafeFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler with safe encoding
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(console_formatter)
    
    # Force UTF-8 encoding on console handler if possible
    if hasattr(console_handler.stream, 'reconfigure'):
        try:
            console_handler.stream.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass  # Silently fail if reconfigure not supported
    
    root_logger.addHandler(console_handler)
    
    # File handler with UTF-8 encoding and error handling
    try:
        log_file = Path(log_dir) / f"bioimagin_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(
            log_file, 
            mode='a', 
            encoding='utf-8', 
            errors='replace'  # Replace problematic characters instead of crashing
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
        
        # Test log entry
        logger = logging.getLogger(__name__)
        if console_formatter.use_emojis:
            logger.info("[OK] Production logging system initialized with emoji support")
        else:
            logger.info("[OK] Production logging system initialized (compatibility mode)")
            
    except Exception as e:
        # If file logging fails, at least we have console logging
        print(f"Warning: Could not setup file logging: {e}")
        logger = logging.getLogger(__name__)
        logger.warning(f"File logging disabled due to error: {str(e)}")
    
    return root_logger


# Auto-initialize logging when imported, but only if not already configured
if not logging.getLogger().handlers:
    setup_production_logging()