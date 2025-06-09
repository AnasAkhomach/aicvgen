"""Logging configuration for the CV AI application."""

import logging
import logging.handlers
import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class LLMCallLog:
    """Structured log entry for LLM API calls."""
    timestamp: str
    model: str
    prompt_type: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    duration_seconds: float
    success: bool
    error_message: Optional[str] = None
    rate_limit_hit: bool = False
    retry_count: int = 0
    session_id: Optional[str] = None
    user_id: Optional[str] = None


@dataclass
class RateLimitLog:
    """Structured log entry for rate limit tracking."""
    timestamp: str
    model: str
    requests_in_window: int
    tokens_in_window: int
    window_start: str
    window_end: str
    limit_exceeded: bool
    wait_time_seconds: float = 0.0


class StructuredLogger:
    """Logger with structured logging capabilities for LLM operations."""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.llm_logger = logging.getLogger(f"{name}.llm")
        self.rate_limit_logger = logging.getLogger(f"{name}.rate_limit")
    
    def log_llm_call(self, call_data: LLMCallLog):
        """Log an LLM API call with structured data."""
        log_entry = {
            "type": "llm_call",
            "data": asdict(call_data)
        }
        
        if call_data.success:
            self.llm_logger.info(json.dumps(log_entry))
        else:
            self.llm_logger.error(json.dumps(log_entry))
    
    def log_rate_limit(self, rate_data: RateLimitLog):
        """Log rate limit tracking information."""
        log_entry = {
            "type": "rate_limit",
            "data": asdict(rate_data)
        }
        
        if rate_data.limit_exceeded:
            self.rate_limit_logger.warning(json.dumps(log_entry))
        else:
            self.rate_limit_logger.info(json.dumps(log_entry))
    
    def info(self, message: str, **kwargs):
        """Log info message with optional structured data."""
        if kwargs:
            message = f"{message} | {json.dumps(kwargs)}"
        self.logger.info(message)
    
    def error(self, message: str, **kwargs):
        """Log error message with optional structured data."""
        if kwargs:
            message = f"{message} | {json.dumps(kwargs)}"
        self.logger.error(message)
    
    def warning(self, message: str, **kwargs):
        """Log warning message with optional structured data."""
        if kwargs:
            message = f"{message} | {json.dumps(kwargs)}"
        self.logger.warning(message)
    
    def debug(self, message: str, **kwargs):
        """Log debug message with optional structured data."""
        if kwargs:
            message = f"{message} | {json.dumps(kwargs)}"
        self.logger.debug(message)


def setup_logging(log_level=logging.INFO, log_to_file=True, log_to_console=True, config=None):
    """
    Set up logging configuration for the application.
    
    Args:
        log_level: Logging level (default: INFO)
        log_to_file: Whether to log to files (default: True)
        log_to_console: Whether to log to console (default: True)
        config: Optional AppConfig instance for environment-specific settings
    """
    # Import here to avoid circular imports
    try:
        from src.config.environment import config as app_config
        if config is None:
            config = app_config
    except ImportError:
        config = None
    
    # Use config values if available
    if config and hasattr(config, 'logging'):
        log_level = config.logging.get_log_level()
        log_to_file = config.logging.log_to_file
        log_to_console = config.logging.log_to_console
    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Create subdirectories for different log types
    (logs_dir / "debug").mkdir(exist_ok=True)
    (logs_dir / "error").mkdir(exist_ok=True)
    (logs_dir / "access").mkdir(exist_ok=True)
    (logs_dir / "llm").mkdir(exist_ok=True)
    (logs_dir / "rate_limit").mkdir(exist_ok=True)
    
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
        
        # Performance log (if enabled in config)
        if config and hasattr(config, 'logging') and config.logging.performance_logging:
            perf_handler = logging.handlers.RotatingFileHandler(
                logs_dir / "performance.log",
                maxBytes=10*1024*1024,  # 10MB
                backupCount=3
            )
            perf_handler.setLevel(logging.INFO)
            perf_handler.setFormatter(detailed_formatter)
            
            # Create performance logger
            perf_logger = logging.getLogger("performance")
            perf_logger.addHandler(perf_handler)
            perf_logger.setLevel(logging.INFO)
        
        # Security log for authentication and authorization events
        security_handler = logging.handlers.RotatingFileHandler(
            logs_dir / "security.log",
            maxBytes=5*1024*1024,  # 5MB
            backupCount=5
        )
        security_handler.setLevel(logging.WARNING)
        security_handler.setFormatter(detailed_formatter)
        
        # Create security logger
        security_logger = logging.getLogger("security")
        security_logger.addHandler(security_handler)
        security_logger.setLevel(logging.WARNING)
        
        # LLM API calls log (structured JSON)
        llm_handler = logging.handlers.RotatingFileHandler(
            logs_dir / "llm" / "llm_calls.log",
            maxBytes=50*1024*1024,  # 50MB
            backupCount=10
        )
        llm_handler.setLevel(logging.INFO)
        llm_formatter = logging.Formatter('%(asctime)s - %(message)s')
        llm_handler.setFormatter(llm_formatter)
        
        # Add LLM handler to all LLM loggers
        llm_logger = logging.getLogger('llm')
        llm_logger.addHandler(llm_handler)
        llm_logger.setLevel(logging.INFO)
        llm_logger.propagate = False  # Don't propagate to root logger
        
        # Rate limit tracking log
        rate_limit_handler = logging.handlers.RotatingFileHandler(
            logs_dir / "rate_limit" / "rate_limits.log",
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        rate_limit_handler.setLevel(logging.INFO)
        rate_limit_handler.setFormatter(llm_formatter)
        
        # Add rate limit handler
        rate_limit_logger = logging.getLogger('rate_limit')
        rate_limit_logger.addHandler(rate_limit_handler)
        rate_limit_logger.setLevel(logging.INFO)
        rate_limit_logger.propagate = False  # Don't propagate to root logger
    
    # Set specific logger levels for noisy libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    
    logging.info(f"Logging initialized - Level: {logging.getLevelName(log_level)}")
    logging.info(f"Log files location: {logs_dir.absolute()}")
    
    # Return the root logger
    return logging.getLogger()


def get_structured_logger(name: str) -> StructuredLogger:
    """Get a structured logger instance for LLM operations."""
    return StructuredLogger(name)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the given name.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def get_performance_logger() -> logging.Logger:
    """Get the performance logger for timing and metrics."""
    return logging.getLogger("performance")


def get_security_logger() -> logging.Logger:
    """Get the security logger for authentication and authorization events."""
    return logging.getLogger("security")


def log_function_performance(func_name: str, duration: float, **kwargs):
    """Log function performance metrics."""
    perf_logger = get_performance_logger()
    metrics = {
        "function": func_name,
        "duration_seconds": duration,
        "timestamp": datetime.now().isoformat(),
        **kwargs
    }
    perf_logger.info(f"PERFORMANCE: {json.dumps(metrics)}")


def log_security_event(event_type: str, details: Dict[str, Any] = None, severity: str = "info"):
    """
    Log a security-related event.
    
    Args:
        event_type: Type of security event (login, access_denied, etc.)
        details: Additional event details
        severity: Event severity (info, warning, error)
    """
    security_logger = get_security_logger()
    
    event_info = {
        "event_type": event_type,
        "severity": severity,
        "timestamp": datetime.now().isoformat(),
        "details": details or {}
    }
    
    message = f"Security Event: {event_type} | {json.dumps(event_info)}"
    
    if severity == "error":
        security_logger.error(message)
    elif severity == "warning":
        security_logger.warning(message)
    else:
        security_logger.info(message)


def log_error_with_context(logger_name: str, error: Exception, context: Dict[str, Any] = None, error_type: str = "application"):
    """
    Log an error with additional context information.
    
    Args:
        logger_name: Name of the logger to use
        error: The exception that occurred
        context: Additional context information
        error_type: Type of error (application, system, user, etc.)
    """
    logger = logging.getLogger(logger_name)
    
    error_info = {
        "error_type": error_type,
        "error_class": error.__class__.__name__,
        "error_message": str(error),
        "timestamp": datetime.now().isoformat(),
        "context": context or {}
    }
    
    if hasattr(error, '__traceback__') and error.__traceback__:
        import traceback
        error_info["traceback"] = traceback.format_exception(
            type(error), error, error.__traceback__
        )
    
    # Format the error message with structured data
    message = f"Error occurred: {error} | {json.dumps(error_info)}"
    logger.error(message)


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