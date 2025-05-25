"""
logging_config.py

Robust cross-platform logging configuration for BIOIMAGIN system.
Handles emoji characters and Unicode properly on all platforms.
"""

import logging
import sys
import os
from pathlib import Path
from datetime import datetime
import traceback
import locale

class SafeFormatter(logging.Formatter):
    """Safe formatter that handles emoji and Unicode characters properly."""
    
    def __init__(self, *args, use_emojis=None, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Auto-detect emoji support if not specified
        if use_emojis is None:
            self.use_emojis = self._detect_emoji_support()
        else:
            self.use_emojis = use_emojis
            
        self.emoji_fallbacks = {
            '✅': '[OK]',
            '❌': '[ERROR]',
            '⚠️': '[WARN]',
            '🔬': '[ANALYSIS]',
            '📊': '[STATS]',
            '🗑️': '[CLEANUP]',
            '📁': '[FILES]',
            '🔄': '[PROCESS]',
            '🚀': '[START]',
            '💾': '[SAVE]',
            '🎯': '[TARGET]',
            '📈': '[PROGRESS]',
            '🔍': '[SEARCH]',
            '🧬': '[BIO]',
            '🤖': '[ML]',
            '🌱': '[SYSTEM]',
            '📷': '[IMAGE]',
            '🎨': '[VIZ]',
            '👋': '[BYE]',
            '🔧': '[CONFIG]',
            '📋': '[REPORT]',
            '🔗': '[LINK]',
            '⚡': '[FAST]',
            '🏥': '[HEALTH]',
            '🌐': '[WEB]',
            '💡': '[INFO]',
            '🆘': '[FALLBACK]',
            '🧪': '[TEST]',
            '📦': '[BACKUP]',
            '🎉': '[SUCCESS]',
            '🕒': '[TIME]',
            '🧹': '[CLEANUP]',
            '🛑': '[STOP]',
            '📐': '[SIZE]',
            '🎓': '[TRAIN]',
            '⏱️': '[TIMER]',
            '🏆': '[BEST]',
            '📚': '[LEARN]',
            'ℹ️': '[INFO]',
            '📂': '[FOLDER]',
            '🖼️': '[IMAGE]',
            '🔀': '[MERGE]',
            '🎈': '[INIT]'
        }
    
    def _detect_emoji_support(self):
        """Detect if the current environment supports emojis."""
        try:
            # Check if we're on Windows with limited encoding
            if sys.platform == 'win32':
                # Check console encoding
                console_encoding = sys.stdout.encoding if hasattr(sys.stdout, 'encoding') else None
                if console_encoding and 'cp' in console_encoding.lower():
                    # Windows console with code page encoding - no emoji support
                    return False
            
            # Try to encode a test emoji
            test_emoji = "✅"
            if hasattr(sys.stdout, 'encoding') and sys.stdout.encoding:
                test_emoji.encode(sys.stdout.encoding)
            else:
                test_emoji.encode('utf-8')
            
            return True
        except (UnicodeEncodeError, LookupError, AttributeError):
            return False
    
    def format(self, record):
        try:
            # Get the formatted message
            msg = super().format(record)
            
            # If emojis are not supported, replace them
            if not self.use_emojis:
                for emoji, fallback in self.emoji_fallbacks.items():
                    msg = msg.replace(emoji, fallback)
            
            return msg
            
        except Exception as e:
            # Ultimate fallback - return a safe error message
            try:
                # Try to at least get the basic message across
                level = record.levelname
                name = record.name
                basic_msg = str(record.msg)
                
                # Remove all non-ASCII characters as last resort
                safe_msg = ''.join(char if ord(char) < 128 else '?' for char in basic_msg)
                
                return f"[{level}] {name}: {safe_msg}"
            except:
                return "[LOGGING_ERROR] Failed to format log message"

class SafeFileHandler(logging.FileHandler):
    """File handler that properly handles UTF-8 encoding."""
    
    def __init__(self, filename, mode='a', encoding='utf-8', delay=False, errors='replace'):
        # Ensure directory exists
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        
        # Always use UTF-8 with error replacement for file logging
        super().__init__(filename, mode, encoding='utf-8', delay=delay, errors='replace')

class SafeStreamHandler(logging.StreamHandler):
    """Stream handler that safely handles Unicode on all platforms."""
    
    def __init__(self, stream=None, use_emojis=None):
        super().__init__(stream)
        self.use_emojis = use_emojis
        
        # Try to set UTF-8 mode on Windows
        if sys.platform == 'win32' and hasattr(self.stream, 'reconfigure'):
            try:
                self.stream.reconfigure(encoding='utf-8', errors='replace')
            except:
                pass
    
    def emit(self, record):
        try:
            msg = self.format(record)
            stream = self.stream
            
            # Write with proper encoding handling
            stream.write(msg + self.terminator)
            self.flush()
            
        except UnicodeEncodeError:
            try:
                # Fallback: Remove emojis and try again
                if hasattr(self.formatter, 'emoji_fallbacks'):
                    safe_msg = msg
                    for emoji, fallback in self.formatter.emoji_fallbacks.items():
                        safe_msg = safe_msg.replace(emoji, fallback)
                    
                    stream.write(safe_msg + self.terminator)
                    self.flush()
                else:
                    # Last resort: ASCII only
                    safe_msg = msg.encode('ascii', 'replace').decode('ascii')
                    stream.write(safe_msg + self.terminator)
                    self.flush()
            except:
                # Ultimate fallback
                pass
        except Exception:
            # Prevent any logging error from crashing the application
            self.handleError(record)

def setup_logging(log_level=logging.INFO, log_file=None, force_no_emoji=False):
    """
    Setup robust logging configuration for BIOIMAGIN system.
    
    Args:
        log_level: Logging level (default: INFO)
        log_file: Log file path (default: logs/bioimagin.log)
        force_no_emoji: Force disable emojis even if supported
    """
    
    # Default log file
    if log_file is None:
        log_file = 'logs/bioimagin.log'
    
    # Determine emoji usage
    use_emojis = not force_no_emoji
    
    # Create formatters
    if use_emojis:
        # Try to auto-detect
        console_formatter = SafeFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            use_emojis=None  # Auto-detect
        )
    else:
        # Force no emojis
        console_formatter = SafeFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            use_emojis=False
        )
    
    # File formatter always uses UTF-8, so can have emojis
    file_formatter = SafeFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        use_emojis=True
    )
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler with auto-detection
    console_handler = SafeStreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (always UTF-8)
    try:
        file_handler = SafeFileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    except Exception as e:
        print(f"Warning: Could not setup file logging: {e}")
    
    # Configure specific loggers to prevent propagation issues
    for logger_name in ['werkzeug', 'PIL', 'matplotlib']:
        specific_logger = logging.getLogger(logger_name)
        specific_logger.setLevel(logging.WARNING)
    
    # Test logging
    logger = logging.getLogger(__name__)
    if console_formatter.use_emojis:
        logger.info("✅ Logging system initialized with emoji support")
    else:
        logger.info("[OK] Logging system initialized (compatibility mode)")
    
    return root_logger

def get_safe_logger(name):
    """Get a logger that's guaranteed to work on all platforms."""
    return logging.getLogger(name)

def safe_log_message(message, logger, level=logging.INFO):
    """
    Safely log a message, handling any encoding issues.
    
    Args:
        message: Message to log (may contain emojis)
        logger: Logger instance
        level: Logging level
    """
    try:
        logger.log(level, message)
    except UnicodeEncodeError:
        # Fallback: try without emojis
        safe_msg = message
        emoji_fallbacks = SafeFormatter().emoji_fallbacks
        for emoji, fallback in emoji_fallbacks.items():
            safe_msg = safe_msg.replace(emoji, fallback)
        
        try:
            logger.log(level, safe_msg)
        except:
            # Last resort: ASCII only
            ascii_msg = message.encode('ascii', 'replace').decode('ascii')
            logger.log(level, ascii_msg)

# Initialize logging when module is imported
if __name__ != "__main__":
    # Only setup logging if not already configured
    if not logging.getLogger().handlers:
        setup_logging()