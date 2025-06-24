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


def setup_logging(log_level=logging.INFO):
    """Configures logging based on the APP_ENV environment variable."""
    APP_ENV = os.environ.get("APP_ENV", "development").lower()
    if APP_ENV == "production":
        _setup_production_logging(log_level)
    else:
        _setup_development_logging(log_level)


def _setup_development_logging(log_level=logging.INFO):
    """Sets up simple, text-based logging for development."""
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    (logs_dir / "error").mkdir(exist_ok=True)

    # Basic configuration
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),  # To console
            logging.FileHandler(logs_dir / "app.log"),
        ],
    )

    # Specific handler for error logs
    error_handler = logging.FileHandler(logs_dir / "error" / "error.log")
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logging.getLogger().addHandler(error_handler)

    logging.info(
        "Development logging initialized (log_level=%s)",
        logging.getLevelName(log_level),
    )


def _setup_production_logging(log_level=logging.INFO):
    """Sets up structured JSON logging for production."""
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # Remove any existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create a handler for structured JSON logging to stdout
    log_handler = logging.StreamHandler()
    formatter = jsonlogger.JsonFormatter(
        "%(asctime)s %(name)s %(levelname)s %(message)s %(filename)s %(lineno)d"
    )
    log_handler.setFormatter(formatter)
    logger.addHandler(log_handler)

    logging.info("Production (JSON) logging initialized.")


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
        self._logger.info(message, extra=kwargs)

    def warning(self, message, **kwargs):
        self._logger.warning(message, extra=kwargs)

    def error(self, message, **kwargs):
        exc_info = kwargs.pop("exc_info", None)
        self._logger.error(message, extra=kwargs, exc_info=exc_info)

    def debug(self, message, **kwargs):
        self._logger.debug(message, extra=kwargs)


def get_structured_logger(name: str) -> "StructuredLogger":
    """Returns a structured logger instance for backward compatibility."""
    return StructuredLogger(name)


def log_error_with_context(logger, message, error=None):
    """Logs an error with additional context, including exception info."""
    if error:
        logger.error(f"{message}: {error}", exc_info=True)
    else:
        logger.error(message, exc_info=True)
