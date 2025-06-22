"""Logging Configuration Module

This module provides comprehensive logging infrastructure for the aicvgen application,
including structured logging, security filtering, and specialized loggers for different
components (LLM calls, rate limiting, agent operations, etc.).

Key Components:
- SensitiveDataFilter: Filters sensitive data from logs
- StructuredLogger: Provides structured logging for agents and LLM operations
- setup_logging(): Comprehensive production logging setup
- setup_observability_logging(): Lightweight setup for testing/observability
"""

import logging
import logging.handlers
import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from pythonjsonlogger import jsonlogger
from pythonjsonlogger.jsonlogger import JsonFormatter


# Simple fallback redaction functions to avoid circular imports
def redact_sensitive_data(data):
    """Simple fallback redaction function."""
    return data


def redact_log_message(message):
    """Simple fallback redaction function."""
    return message


# ============================================================================
# SECURITY FILTERS
# ============================================================================


class SensitiveDataFilter(logging.Filter):
    """Filter to redact sensitive data from log records before they are formatted."""

    def filter(self, record):
        """Apply redaction to log record data.

        Args:
            record: LogRecord instance

        Returns:
            True to allow the record to be logged
        """
        # Redact sensitive data in record.args if present
        if hasattr(record, "args") and record.args:
            if isinstance(record.args, (tuple, list)):
                record.args = tuple(redact_sensitive_data(arg) for arg in record.args)
            elif isinstance(record.args, dict):
                record.args = redact_sensitive_data(record.args)

        # Redact sensitive data in record.__dict__ safely
        # We need to be very careful not to interfere with Python's internal logging attributes
        try:
            # Create a safe copy of attributes to modify
            safe_attrs = {}
            for key, value in record.__dict__.items():
                # Skip Python logging internal attributes that should not be modified
                if key in (
                    "name",
                    "msg",
                    "args",
                    "levelname",
                    "levelno",
                    "pathname",
                    "filename",
                    "module",
                    "lineno",
                    "funcName",
                    "created",
                    "msecs",
                    "relativeCreated",
                    "thread",
                    "threadName",
                    "processName",
                    "process",
                    "getMessage",
                    "exc_info",
                    "exc_text",
                    "stack_info",
                ):
                    continue

                # Only redact custom attributes that might contain sensitive data
                if isinstance(value, (str, dict, list, tuple)):
                    safe_attrs[key] = redact_sensitive_data(value)
                else:
                    safe_attrs[key] = value

            # Update the record with redacted custom attributes
            for key, value in safe_attrs.items():
                setattr(record, key, value)

        except (TypeError, AttributeError, KeyError):
            # If we can't safely process attributes, leave them as is
            pass

        # Redact the message itself if it contains sensitive patterns
        if hasattr(record, "msg") and isinstance(record.msg, str):
            record.msg = redact_log_message(record.msg)

        return True


# ============================================================================
# LOGGING SETUP FUNCTIONS
# ============================================================================


def setup_observability_logging():
    """Configure lightweight structured JSON logging for observability framework.

    This is a simplified logging setup specifically for observability and testing.
    For production use, prefer setup_logging() which provides comprehensive logging.
    """
    logger = logging.getLogger("aicvgen")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Create console handler
        log_handler = logging.StreamHandler()

        # Create JSON formatter with trace_id support
        formatter = jsonlogger.JsonFormatter(
            "%(asctime)s %(name)s %(levelname)s %(message)s %(trace_id)s %(module)s %(funcName)s %(lineno)d"
        )

        log_handler.setFormatter(formatter)

        # Add sensitive data filter
        log_handler.addFilter(SensitiveDataFilter())

        logger.addHandler(log_handler)

    return logger


# ============================================================================
# LOGGING DATA MODELS
# ============================================================================


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


# Import agent logging dataclasses from data models for clean architecture
from ..models.data_models import (
    AgentExecutionLog,
    AgentDecisionLog,
    AgentPerformanceLog,
)


# ============================================================================
# STRUCTURED LOGGER CLASS
# ============================================================================


class StructuredLogger:
    """Logger with structured logging capabilities for LLM operations and agent tracking."""

    def __init__(self, name: str):
        import logging

        self.logger = logging.getLogger(name)
        if self.logger is None:
            from .logging_config import setup_logging

            setup_logging()
            self.logger = logging.getLogger(name)
        self.llm_logger = logging.getLogger(f"{name}.llm")
        self.rate_limit_logger = logging.getLogger(f"{name}.rate_limit")
        self.agent_logger = logging.getLogger(f"{name}.agent")
        self.performance_logger = logging.getLogger(f"{name}.performance")

    def log_llm_call(self, call_data: LLMCallLog):
        """Log an LLM API call with structured data."""
        log_entry = {"type": "llm_call", "data": asdict(call_data)}

        # Redact sensitive data before logging
        safe_log_entry = redact_sensitive_data(log_entry)

        if call_data.success:
            self.llm_logger.info(json.dumps(safe_log_entry))
        else:
            self.llm_logger.error(json.dumps(safe_log_entry))

    def log_rate_limit(self, rate_data: RateLimitLog):
        """Log rate limit tracking information."""
        log_entry = {"type": "rate_limit", "data": asdict(rate_data)}

        if rate_data.limit_exceeded:
            self.rate_limit_logger.warning(json.dumps(log_entry))
        else:
            self.rate_limit_logger.info(json.dumps(log_entry))

    def log_agent_execution(self, execution_data: AgentExecutionLog):
        """Log agent execution with structured data."""
        log_entry = {"type": "agent_execution", "data": asdict(execution_data)}

        # Redact sensitive data before logging
        safe_log_entry = redact_sensitive_data(log_entry)

        if execution_data.execution_phase == "error":
            self.agent_logger.error(json.dumps(safe_log_entry))
        elif execution_data.execution_phase == "retry":
            self.agent_logger.warning(json.dumps(safe_log_entry))
        else:
            self.agent_logger.info(json.dumps(safe_log_entry))

    def log_agent_decision(self, decision_data: AgentDecisionLog):
        """Log agent decision with structured data."""
        log_entry = {"type": "agent_decision", "data": asdict(decision_data)}

        # Redact sensitive data before logging
        safe_log_entry = redact_sensitive_data(log_entry)
        self.agent_logger.info(json.dumps(safe_log_entry))

    def log_agent_performance(self, performance_data: AgentPerformanceLog):
        """Log agent performance metrics with structured data."""
        log_entry = {"type": "agent_performance", "data": asdict(performance_data)}

        self.performance_logger.info(json.dumps(log_entry))

    def info(self, message: str, extra=None, exc_info=None, **kwargs):
        """Log info message with optional structured data."""
        # Handle both extra parameter and kwargs
        log_data = {}
        if extra:
            # If extra is a dict, update log_data; otherwise, ignore non-dict extra
            if isinstance(extra, dict):
                log_data.update(extra)
        if kwargs:
            log_data.update(kwargs)

        # Prepare logging arguments
        log_kwargs = {}
        if exc_info is not None:
            log_kwargs["exc_info"] = exc_info

        if log_data:
            safe_log_data = redact_sensitive_data(log_data)
            log_kwargs["extra"] = safe_log_data
            self.logger.info(message, **log_kwargs)
        else:
            safe_message = redact_log_message(message)
            self.logger.info(safe_message, **log_kwargs)

    def error(self, message: str, extra=None, exc_info=None, **kwargs):
        """Log error message with optional structured data."""
        # Handle both extra parameter and kwargs
        log_data = {}
        if extra:
            log_data.update(extra)
        if kwargs:
            log_data.update(kwargs)

        # Prepare logging arguments
        log_kwargs = {}
        if exc_info is not None:
            log_kwargs["exc_info"] = exc_info

        if log_data:
            safe_log_data = redact_sensitive_data(log_data)
            log_kwargs["extra"] = safe_log_data
            self.logger.error(message, **log_kwargs)
        else:
            safe_message = redact_log_message(message)
            self.logger.error(safe_message, **log_kwargs)

    def warning(self, message: str, extra=None, exc_info=None, **kwargs):
        """Log warning message with optional structured data."""
        # Handle both extra parameter and kwargs
        log_data = {}
        if extra:
            log_data.update(extra)
        if kwargs:
            log_data.update(kwargs)

        # Prepare logging arguments
        log_kwargs = {}
        if exc_info is not None:
            log_kwargs["exc_info"] = exc_info

        if log_data:
            safe_log_data = redact_sensitive_data(log_data)
            log_kwargs["extra"] = safe_log_data
            self.logger.warning(message, **log_kwargs)
        else:
            safe_message = redact_log_message(message)
            self.logger.warning(safe_message, **log_kwargs)

    def debug(self, message: str, extra=None, exc_info=None, **kwargs):
        """Log debug message with optional structured data."""
        # Handle both extra parameter and kwargs
        log_data = {}
        if extra:
            log_data.update(extra)
        if kwargs:
            log_data.update(kwargs)

        # Prepare logging arguments
        log_kwargs = {}
        if exc_info is not None:
            log_kwargs["exc_info"] = exc_info

        if log_data:
            safe_log_data = redact_sensitive_data(log_data)
            log_kwargs["extra"] = safe_log_data
            self.logger.debug(message, **log_kwargs)
        else:
            safe_message = redact_log_message(message)
            self.logger.debug(safe_message, **log_kwargs)


def setup_logging(
    log_level=logging.INFO, log_to_file=True, log_to_console=True, config=None
):
    """
    Set up comprehensive logging configuration for the application.

    Args:
        log_level: Logging level (default: INFO)
        log_to_file: Whether to log to files (default: True)
        log_to_console: Whether to log to console (default: True)
        config: Optional AppConfig instance for environment-specific settings
    """
    # Import here to avoid circular imports
    try:
        from ..config.environment import config as app_config

        if config is None:
            config = app_config
    except ImportError:
        config = None

    # Use config values if available
    if config and hasattr(config, "logging"):
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
        "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
    )
    simple_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    json_formatter = JsonFormatter()

    # Create sensitive data filter
    sensitive_filter = SensitiveDataFilter()

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
        console_handler.addFilter(sensitive_filter)  # Add filter to console
        root_logger.addHandler(console_handler)

    if log_to_file:
        # Main application log (rotating) - Use JSON formatting for structured logs
        app_handler = logging.handlers.RotatingFileHandler(
            logs_dir / "app.log", maxBytes=10 * 1024 * 1024, backupCount=5  # 10MB
        )
        app_handler.setLevel(log_level)
        app_handler.setFormatter(
            json_formatter
        )  # Use JSON formatter for structured logging
        app_handler.addFilter(sensitive_filter)  # Add sensitive data filter
        root_logger.addHandler(app_handler)

        # Error log (for ERROR and CRITICAL only)
        error_handler = logging.handlers.RotatingFileHandler(
            logs_dir / "error" / "error.log",
            maxBytes=5 * 1024 * 1024,  # 5MB
            backupCount=3,
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(detailed_formatter)
        error_handler.addFilter(sensitive_filter)  # Add sensitive data filter
        root_logger.addHandler(error_handler)

        # Debug log (for DEBUG level, separate file)
        debug_handler = logging.handlers.RotatingFileHandler(
            logs_dir / "debug" / "debug.log",
            maxBytes=20 * 1024 * 1024,  # 20MB
            backupCount=2,
        )
        debug_handler.setLevel(logging.DEBUG)
        debug_handler.setFormatter(detailed_formatter)
        debug_handler.addFilter(sensitive_filter)  # Add sensitive data filter

        # Only add debug handler if log level is DEBUG
        if log_level <= logging.DEBUG:
            root_logger.addHandler(debug_handler)

        # Performance log (if enabled in config)
        if config and hasattr(config, "logging") and config.logging.performance_logging:
            perf_handler = logging.handlers.RotatingFileHandler(
                logs_dir / "performance.log",
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=3,
            )
            perf_handler.setLevel(logging.INFO)
            perf_handler.setFormatter(
                json_formatter
            )  # Use JSON formatter for structured performance logs
            perf_handler.addFilter(sensitive_filter)  # Add sensitive data filter

            # Create performance logger
            perf_logger = logging.getLogger("performance")
            perf_logger.addHandler(perf_handler)
            perf_logger.setLevel(logging.INFO)

        # Security log for authentication and authorization events - Use JSON formatting
        security_handler = logging.handlers.RotatingFileHandler(
            logs_dir / "security.log", maxBytes=10 * 1024 * 1024, backupCount=5  # 10MB
        )
        security_handler.setLevel(logging.INFO)
        security_handler.setFormatter(
            json_formatter
        )  # Use JSON formatter for structured security logs
        security_handler.addFilter(sensitive_filter)  # Add sensitive data filter

        # Create security logger
        security_logger = logging.getLogger("security")
        security_logger.addHandler(security_handler)
        security_logger.setLevel(logging.INFO)
        security_logger.propagate = False  # Don't propagate to root logger

        # LLM API calls log (structured JSON) - CRITICAL: Must filter sensitive data
        llm_handler = logging.handlers.RotatingFileHandler(
            logs_dir / "llm" / "llm_calls.log",
            maxBytes=50 * 1024 * 1024,  # 50MB
            backupCount=10,
        )
        llm_handler.setLevel(logging.INFO)
        llm_formatter = logging.Formatter("%(asctime)s - %(message)s")
        llm_handler.setFormatter(llm_formatter)
        llm_handler.addFilter(
            sensitive_filter
        )  # CRITICAL: Filter API keys and sensitive data

        # Add LLM handler to all LLM loggers
        llm_logger = logging.getLogger("llm")
        llm_logger.addHandler(llm_handler)
        llm_logger.setLevel(logging.INFO)
        llm_logger.propagate = False  # Don't propagate to root logger

        # Rate limit tracking log
        rate_limit_handler = logging.handlers.RotatingFileHandler(
            logs_dir / "rate_limit" / "rate_limits.log",
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
        )
        rate_limit_handler.setLevel(logging.INFO)
        rate_limit_handler.setFormatter(llm_formatter)
        rate_limit_handler.addFilter(sensitive_filter)  # Add sensitive data filter

        # Add rate limit handler
        rate_limit_logger = logging.getLogger("rate_limit")
        rate_limit_logger.addHandler(rate_limit_handler)
        rate_limit_logger.setLevel(logging.INFO)
        rate_limit_logger.propagate = False  # Don't propagate to root logger

    # Set specific logger levels for noisy libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    logging.info("Logging initialized - Level: %s", logging.getLevelName(log_level))
    logging.info("Log files location: %s", logs_dir.absolute())

    # Return the root logger
    return logging.getLogger()


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def get_structured_logger(name: str):
    """Get a structured logger instance for LLM operations. Always ensure logging is initialized."""
    import logging

    try:
        logger = logging.getLogger(name)
        # If no handlers, initialize logging (safe for repeated calls)
        if not logger.hasHandlers():
            from .logging_config import setup_logging

            setup_logging()
        return StructuredLogger(name)
    except Exception:
        # Fallback to stdlib logger
        logger = logging.getLogger(name)
        if not logger.hasHandlers():
            from .logging_config import setup_logging

            setup_logging()
        return logger


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
        **kwargs,
    }
    perf_logger.info("PERFORMANCE: %s", json.dumps(metrics))


def log_security_event(
    event_type: str, details: Dict[str, Any] = None, severity: str = "info"
):
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
        "details": details or {},
    }

    message = f"Security Event: {event_type} | {json.dumps(event_info)}"

    if severity == "error":
        security_logger.error(message)
    elif severity == "warning":
        security_logger.warning(message)
    else:
        security_logger.info(message)


def log_error_with_context(
    logger_name: str,
    error: Exception,
    context: Dict[str, Any] = None,
    error_type: str = "application",
):
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
        "context": context or {},
    }

    if hasattr(error, "__traceback__") and error.__traceback__:
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
        "%s %s - Status: %s - Duration: %.3fs - IP: %s",
        request_info.get("method", "UNKNOWN"),
        request_info.get("path", "/"),
        request_info.get("status", "UNKNOWN"),
        request_info.get("duration", 0),
        request_info.get("ip", "UNKNOWN"),
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
