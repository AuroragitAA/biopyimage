"""
logging_config.py - Production-Ready Cross-Platform Logging
Handles Unicode/emoji characters properly on all platforms including Windows cmd/PowerShell
"""

import locale
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

from analysis_config import AnalysisConfig


class SafeFormatter(logging.Formatter):
    """Safe formatter that handles emoji and Unicode characters properly."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Detect emoji support
        self.use_emojis = self._detect_emoji_support()
        
        # Emoji fallbacks for Windows cmd/PowerShell
        self.emoji_fallbacks = {
            '✅': '[OK]', '❌': '[ERROR]', '⚠️': '[WARN]', '🔬': '[ANALYSIS]',
            '📊': '[STATS]', '🗑️': '[CLEAN]', '📁': '[FILES]', '🔄': '[PROCESS]',
            '🚀': '[START]', '💾': '[SAVE]', '🎯': '[TARGET]', '📈': '[PROGRESS]',
            '🔍': '[SEARCH]', '🧬': '[BIO]', '🤖': '[ML]', '🌱': '[SYSTEM]',
            '📷': '[IMAGE]', '🎨': '[VIZ]', '👋': '[BYE]', '🔧': '[CONFIG]',
            '📋': '[REPORT]', '🔗': '[LINK]', '⚡': '[FAST]', '🏥': '[HEALTH]',
            '🌐': '[WEB]', '💡': '[INFO]', '🆘': '[FALLBACK]', '🧪': '[TEST]',
            '📦': '[BACKUP]', '🎉': '[SUCCESS]', '🕒': '[TIME]', '🧹': '[CLEANUP]'
        }
    
    def _detect_emoji_support(self):
        """Detect if the current environment supports emojis."""
        try:
            # Force disable emojis on Windows cmd/PowerShell to avoid encoding errors
            if os.name == 'nt':  # Windows
                console_encoding = getattr(sys.stdout, 'encoding', '').lower()
                if 'cp' in console_encoding or console_encoding in ['ascii', 'ansi']:
                    return False
                    
            # Test emoji encoding
            test_emoji = "✅"
            test_emoji.encode('utf-8')
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
            
            return msg
            
        except Exception:
            # Ultimate fallback for any encoding issues
            try:
                safe_msg = str(record.msg).encode('ascii', 'replace').decode('ascii')
                return f"[{record.levelname}] {record.name}: {safe_msg}"
            except:
                return f"[{record.levelname}] LOGGING_ERROR"


def setup_production_logging(log_level=logging.INFO, log_dir="logs"):
    """Setup production-ready logging system."""
    
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
    
    # Remove existing handlers
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
        except:
            pass
    
    root_logger.addHandler(console_handler)
    
    # File handler with UTF-8 encoding
    try:
        log_file = Path(log_dir) / f"bioimagin_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(
            log_file, 
            mode='a', 
            encoding='utf-8', 
            errors='replace'
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    except Exception as e:
        print(f"Warning: Could not setup file logging: {e}")
    
    # Test logging system
    logger = logging.getLogger(__name__)
    if console_formatter.use_emojis:
        logger.info("✅ Production logging system initialized with emoji support")
    else:
        logger.info("[OK] Production logging system initialized (compatibility mode)")
    
    return root_logger

# Auto-initialize when imported
if not logging.getLogger().handlers:
    setup_production_logging()