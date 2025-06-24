"""
Simple Logging Configuration (Emergency Fallback)
================================================

This is a simplified logging configuration that should work without issues.
"""

import logging
import os
from pathlib import Path

def setup_logging(log_level=logging.INFO):
    """Simple logging setup that works."""
    # Create logs directory
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    (logs_dir / "error").mkdir(exist_ok=True)
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),  # Console
            logging.FileHandler(logs_dir / "app.log"),  # File
        ]
    )
    
    # Add error file handler
    error_handler = logging.FileHandler(logs_dir / "error" / "error.log")
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(error_handler)
    
    logging.info("Simple logging initialized")

def get_logger(name):
    """Get a simple logger."""
    return logging.getLogger(name)

# For backward compatibility
def get_structured_logger(name):
    """Get a simple logger (backward compatibility).""" 
    return logging.getLogger(name)

class StructuredLogger:
    """Simple structured logger replacement."""
    def __init__(self, name):
        self.logger = logging.getLogger(name)
    
    def info(self, message, **kwargs):
        self.logger.info(message)
    
    def warning(self, message, **kwargs):
        self.logger.warning(message)
    
    def error(self, message, **kwargs):
        self.logger.error(message)
    
    def debug(self, message, **kwargs):
        self.logger.debug(message)
