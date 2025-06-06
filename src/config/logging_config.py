"""Logging configuration for the CV AI application."""

import logging
import logging.handlers
import os
from datetime import datetime
from pathlib import Path


def setup_logging(log_level=logging.INFO, log_to_file=True, log_to_console=True):
    """
    Set up logging configuration for the application.
    
    Args:
        log_level: Logging level (default: INFO)
        log_to_file: Whether to log to files (default: True)
        log_to_console: Whether to log to console (default: True)
    """
    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Create subdirectories for different log types
    (logs_dir / "debug").mkdir(exist_ok=True)
    (logs_dir / "error").mkdir(exist_ok=True)
    (logs_dir / "access").mkdir(exist_ok=True)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    )
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(simple_formatter)
        root_logger.addHandler(console_handler)
    
    if log_to_file:
        # Main application log (rotating)
        app_handler = logging.handlers.RotatingFileHandler(
            logs_dir / "app.log",
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        app_handler.setLevel(log_level)
        app_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(app_handler)
        
        # Error log (for ERROR and CRITICAL only)
        error_handler = logging.handlers.RotatingFileHandler(
            logs_dir / "error" / "error.log",
            maxBytes=5*1024*1024,  # 5MB
            backupCount=3
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(error_handler)
        
        # Debug log (for DEBUG level, separate file)
        debug_handler = logging.handlers.RotatingFileHandler(
            logs_dir / "debug" / "debug.log",
            maxBytes=20*1024*1024,  # 20MB
            backupCount=2
        )
        debug_handler.setLevel(logging.DEBUG)
        debug_handler.setFormatter(detailed_formatter)
        
        # Only add debug handler if log level is DEBUG
        if log_level <= logging.DEBUG:
            root_logger.addHandler(debug_handler)
    
    # Set specific logger levels for noisy libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    
    logging.info(f"Logging initialized - Level: {logging.getLevelName(log_level)}")
    logging.info(f"Log files location: {logs_dir.absolute()}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the given name.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def log_request(request_info: dict):
    """Log HTTP request information."""
    access_logger = logging.getLogger("access")
    access_logger.info(
        f"{request_info.get('method', 'UNKNOWN')} {request_info.get('path', '/')} - "
        f"Status: {request_info.get('status', 'UNKNOWN')} - "
        f"Duration: {request_info.get('duration', 0):.3f}s - "
        f"IP: {request_info.get('ip', 'UNKNOWN')}"
    )


def setup_test_logging(test_name: str, log_file_path: Path) -> logging.Logger:
    """Setup logging for test files with proper file handling.
    
    Args:
        test_name: Name of the test (used as logger name)
        log_file_path: Path to the log file (relative to project root)
        
    Returns:
        Configured logger instance
    """
    # Ensure the log directory exists
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(test_name)
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # File handler (overwrite mode for tests)
    file_handler = logging.FileHandler(log_file_path, mode="w")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Prevent propagation to avoid duplicate logs
    logger.propagate = False
    
    return logger