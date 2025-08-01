"""Logging Configuration Module

This module provides a unified logging setup for the aicvgen application.
It supports environment-aware configuration to switch between simple text-based
logging for development and structured JSON logging for production.

The configuration is controlled by the `APP_ENV` environment variable.
- `APP_ENV=development` (default): Simple, human-readable console output.
- `APP_ENV=production`: Structured JSON logging for robust monitoring.
"""

import logging
import os
from pathlib import Path

from pythonjsonlogger import jsonlogger
from .settings import get_config

# from src.config.logging_config.settings import get_config

# Global flag to ensure setup_logging is only called once
_logging_initialized = False


def setup_logging(log_level=logging.DEBUG):
    """Configures logging based on the APP_ENV environment variable."""
    global _logging_initialized

    # Prevent multiple initializations
    if _logging_initialized:
        return

    APP_ENV = os.environ.get("APP_ENV", "development").lower()
    if APP_ENV == "production":
        _setup_production_logging(log_level)
    else:
        _setup_development_logging(log_level)

    _logging_initialized = True


def _setup_development_logging(log_level=logging.INFO):
    """Sets up simple, text-based logging for development in a robust, idempotent way."""
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Clear any existing handlers to prevent duplication
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    config = get_config()
    logs_dir = Path(config.logging.log_directory)
    error_logs_dir = logs_dir / "error"
    logs_dir.mkdir(exist_ok=True)
    error_logs_dir.mkdir(exist_ok=True)

    # Create a standard formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Create and add console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Create and add main log file handler
    main_log_path = logs_dir / config.logging.main_log_file
    main_file_handler = logging.FileHandler(main_log_path)
    main_file_handler.setFormatter(formatter)
    root_logger.addHandler(main_file_handler)

    # Create and add error log file handler
    error_log_path = error_logs_dir / config.logging.error_log_file
    error_file_handler = logging.FileHandler(error_log_path)
    error_file_handler.setLevel(logging.ERROR)
    error_file_handler.setFormatter(formatter)
    root_logger.addHandler(error_file_handler)

    root_logger.info(
        "Development logging initialized (log_level=%s)",
        logging.getLevelName(log_level),
    )


def _setup_production_logging(log_level=logging.INFO):
    """Sets up structured JSON logging for production with persistent file handlers."""
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # Remove any existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    try:
        # Get configuration and ensure log directories exist
        config = get_config()
        logs_dir = Path(config.logging.log_directory)
        error_logs_dir = logs_dir / "error"
        logs_dir.mkdir(exist_ok=True, parents=True)
        error_logs_dir.mkdir(exist_ok=True, parents=True)

        # Create a handler for structured JSON logging to stdout
        console_handler = logging.StreamHandler()
        json_formatter = jsonlogger.JsonFormatter(
            "%(asctime)s %(name)s %(levelname)s %(message)s %(filename)s %(lineno)d"
        )
        console_handler.setFormatter(json_formatter)
        logger.addHandler(console_handler)

        # Create standard formatter for file logs (human-readable)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # Create and add main log file handler
        main_log_path = logs_dir / config.logging.main_log_file
        main_file_handler = logging.FileHandler(main_log_path)
        main_file_handler.setFormatter(file_formatter)
        logger.addHandler(main_file_handler)

        # Create and add error log file handler
        error_log_path = error_logs_dir / config.logging.error_log_file
        error_file_handler = logging.FileHandler(error_log_path)
        error_file_handler.setLevel(logging.ERROR)
        error_file_handler.setFormatter(file_formatter)
        logger.addHandler(error_file_handler)

        logging.info("Production logging initialized with file persistence.")

    except (IOError, PermissionError) as e:
        # Graceful fallback to console-only logging
        console_handler = logging.StreamHandler()
        json_formatter = jsonlogger.JsonFormatter(
            "%(asctime)s %(name)s %(levelname)s %(message)s %(filename)s %(lineno)d"
        )
        console_handler.setFormatter(json_formatter)
        logger.addHandler(console_handler)

        logging.warning(
            "Failed to setup file logging (permissions or IO error): %s. "
            "Falling back to console-only logging.",
            str(e),
        )


def get_logger(name: str) -> logging.Logger:
    """Returns a logger instance."""
    return logging.getLogger(name)


# For backward compatibility with old code that might have used StructuredLogger
class StructuredLogger:
    """A compatibility wrapper to avoid breaking old code.

    This class mimics the old StructuredLogger but simply wraps the standard
    logging.Logger. It's intended as a temporary shim.
    """

    def __init__(self, name: str):
        self._logger = logging.getLogger(name)

    def info(self, message, **kwargs):
        # Filter out application-specific parameters that shouldn't be passed to logger
        filtered_kwargs = self._filter_logging_kwargs(kwargs)
        self._logger.info(message, extra=filtered_kwargs)

    def warning(self, message, **kwargs):
        # Filter out application-specific parameters that shouldn't be passed to logger
        filtered_kwargs = self._filter_logging_kwargs(kwargs)
        self._logger.warning(message, extra=filtered_kwargs)

    def error(self, message, *args, **kwargs):
        exc_info = kwargs.pop("exc_info", None)
        # Handle old-style string formatting
        if args:
            message = message % args
        # Filter out application-specific parameters that shouldn't be passed to logger
        filtered_kwargs = self._filter_logging_kwargs(kwargs)
        self._logger.error(message, extra=filtered_kwargs, exc_info=exc_info)

    def debug(self, message, **kwargs):
        # Filter out application-specific parameters that shouldn't be passed to logger
        filtered_kwargs = self._filter_logging_kwargs(kwargs)
        self._logger.debug(message, extra=filtered_kwargs)

    def _filter_logging_kwargs(self, kwargs):
        """Filter out application-specific parameters that shouldn't be passed to logger."""
        # Remove parameters that are not valid for logger.extra
        filtered = {
            k: v
            for k, v in kwargs.items()
            if k not in ["session_id", "trace_id", "user_id"]
        }
        return filtered

    def log_agent_decision(self, decision_log):
        """Log agent decision data."""
        # For compatibility, just log as info with agent decision context
        self._logger.info("Agent decision", extra={"decision_log": decision_log})


def get_structured_logger(name: str) -> "StructuredLogger":
    """Returns a structured logger instance for backward compatibility."""
    return StructuredLogger(name)


def log_error_with_context(logger, message, error=None):
    """Logs an error with additional context, including exception info."""
    if error:
        logger.error(f"{message}: {str(error)}", exc_info=True)
    else:
        logger.error(message, exc_info=True)
