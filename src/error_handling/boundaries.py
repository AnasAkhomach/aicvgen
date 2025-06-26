#!/usr/bin/env python3
"""
Error Boundaries for Streamlit Components

This module provides error boundary decorators and context managers for handling
errors gracefully in Streamlit applications, preventing crashes and providing
user-friendly error messages.
"""

import functools
import traceback
from typing import Any, Callable, Optional, Dict
from contextlib import contextmanager
import streamlit as st
from datetime import datetime

from src.config.logging_config import get_logger, log_error_with_context
from src.error_handling.exceptions import (
    AicvgenError,
    NetworkError,
    OperationTimeoutError,
)
from src.error_handling.models import ErrorSeverity

logger = get_logger(__name__)

# A tuple of common, catchable exceptions to avoid capturing system-level exceptions.
CATCHABLE_EXCEPTIONS = (
    AicvgenError,
    ValueError,
    TypeError,
    KeyError,
    IOError,
    IndexError,
    AttributeError,
    ConnectionError,
)

# A tuple of exceptions that are considered transient and worth retrying.
RETRYABLE_EXCEPTIONS = (OperationTimeoutError, NetworkError, IOError, ConnectionError)


class StreamlitErrorBoundary:
    """Error boundary for Streamlit components with user-friendly error handling."""

    def __init__(
        self,
        component_name: str,
        show_error_details: bool = False,
        fallback_message: Optional[str] = None,
        severity: str = ErrorSeverity.MEDIUM,
    ):
        self.component_name = component_name
        self.show_error_details = show_error_details
        self.fallback_message = (
            fallback_message or f"An error occurred in {component_name}"
        )
        self.severity = severity

    def __call__(self, func: Callable) -> Callable:
        """Decorator to wrap functions with error boundary."""

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except CATCHABLE_EXCEPTIONS as e:
                self._handle_error(e, func.__name__, args, kwargs)
                return None

        return wrapper

    def _handle_error(
        self, error: Exception, func_name: str, args: tuple, kwargs: dict
    ):
        """Handle and display errors appropriately."""
        error_id = self._generate_error_id()

        context = {
            "component_name": self.component_name,
            "function_name": func_name,
            "error_id": error_id,
            "severity": self.severity,
            "args_count": len(args),
            "kwargs_keys": list(kwargs.keys()),
        }

        log_error_with_context("error_boundaries", error, context)

        self._display_error_message(error, error_id)

    def _generate_error_id(self) -> str:
        """Generate a unique error ID for tracking."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"ERR_{self.component_name}_{timestamp}"

    def _display_error_message(self, error: Exception, error_id: str):
        """Display appropriate error message to user."""
        if self.severity == ErrorSeverity.CRITICAL:
            st.error(f"🚨 Critical Error: {self.fallback_message}")
            st.error(f"Error ID: {error_id}")
            if self.show_error_details:
                with st.expander("Technical Details"):
                    st.code(str(error))
                    st.code(traceback.format_exc())
        elif self.severity == ErrorSeverity.HIGH:
            st.error(f"❌ {self.fallback_message}")
            st.info(f"Error ID: {error_id} - Please try again or contact support.")
            if self.show_error_details:
                with st.expander("Error Details"):
                    st.code(str(error))
        elif self.severity == ErrorSeverity.MEDIUM:
            st.warning(f"⚠️ {self.fallback_message}")
            st.info("Please try again. If the problem persists, refresh the page.")
            if self.show_error_details:
                st.code(str(error))
        else:  # LOW severity
            st.info(f"ℹ️ {self.fallback_message}")
            if self.show_error_details:
                st.code(str(error))


@contextmanager
def error_boundary(
    component_name: str,
    severity: str = ErrorSeverity.MEDIUM,
    show_details: bool = False,
    fallback_message: Optional[str] = None,
):
    """Context manager for error boundaries."""
    try:
        yield
    except CATCHABLE_EXCEPTIONS as e:
        boundary = StreamlitErrorBoundary(
            component_name=component_name,
            show_error_details=show_details,
            fallback_message=fallback_message,
            severity=severity,
        )
        boundary._handle_error(e, "context_manager", (), {})


def safe_streamlit_component(
    component_name: str,
    severity: str = ErrorSeverity.MEDIUM,
    show_details: bool = False,
    fallback_message: Optional[str] = None,
):
    """Decorator for making Streamlit components safe with error boundaries."""
    return StreamlitErrorBoundary(
        component_name=component_name,
        show_error_details=show_details,
        fallback_message=fallback_message,
        severity=severity,
    )


def handle_api_errors(func: Callable) -> Callable:
    """Decorator specifically for API-related functions."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ConnectionError as e:
            st.error(
                "🌐 Connection Error: Unable to connect to the service. Please check your internet connection."
            )
            logger.error("Connection error in %s: %s", func.__name__, str(e))
        except OperationTimeoutError as e:
            st.error("⏱️ Timeout Error: The request took too long. Please try again.")
            logger.error("Timeout error in %s: %s", func.__name__, str(e))
        except (ValueError, TypeError) as e:
            st.error(f"📝 Input Error: Invalid data provided. {e}")
            logger.error("Data validation error in %s: %s", func.__name__, str(e))
        except AicvgenError as e:
            st.error(f"❌ An application error occurred: {e}")
            log_error_with_context(
                "error_boundaries", e, {"function": func.__name__, "type": "api_error"}
            )
        return None

    return wrapper


def handle_file_operations(func: Callable) -> Callable:
    """Decorator specifically for file operation functions."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except FileNotFoundError as e:
            st.error(f"📁 File Not Found: The file could not be found. {e}")
            logger.error("File not found in %s: %s", func.__name__, str(e))
        except PermissionError as e:
            st.error(f"🔒 Permission Error: Unable to access the file. {e}")
            logger.error("Permission error in %s: %s", func.__name__, str(e))
        except (OSError, IOError) as e:
            st.error(
                f"💾 File System Error: A problem occurred with the file system. {e}"
            )
            logger.error("OS/IO error in %s: %s", func.__name__, str(e))
        except AicvgenError as e:
            st.error(f"❌ File Operation Error: An unexpected error occurred. {e}")
            log_error_with_context(
                "error_boundaries",
                e,
                {"function": func.__name__, "type": "file_error"},
            )
        return None

    return wrapper


def handle_data_processing(func: Callable) -> Callable:
    """Decorator specifically for data processing functions."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (KeyError, IndexError) as e:
            st.error(
                f"🔑 Data Error: Required data field is missing or out of bounds. {e}"
            )
            logger.error("Key/Index error in %s: %s", func.__name__, str(e))
        except (TypeError, ValueError) as e:
            st.error(f"🔧 Type/Value Error: Data format is incorrect or invalid. {e}")
            logger.error("Type/Value error in %s: %s", func.__name__, str(e))
        except AicvgenError as e:
            st.error(f"❌ Data Processing Error: An application error occurred. {e}")
            log_error_with_context(
                "error_boundaries",
                e,
                {"function": func.__name__, "type": "data_error"},
            )
        return None

    return wrapper


class ErrorRecovery:
    """Utility class for error recovery strategies."""

    @staticmethod
    def retry_with_backoff(
        func: Callable, max_retries: int = 3, backoff_factor: float = 1.0
    ):
        """Retry a function with exponential backoff for transient errors."""
        import time

        for attempt in range(max_retries):
            try:
                return func()
            except RETRYABLE_EXCEPTIONS as e:
                if attempt == max_retries - 1:
                    logger.error("Final attempt failed for %s.", func.__name__)
                    raise e

                wait_time = backoff_factor * (2**attempt)
                logger.warning(
                    "Attempt %d failed for %s, retrying in %.2fs: %s",
                    attempt + 1,
                    func.__name__,
                    wait_time,
                    str(e),
                )
                time.sleep(wait_time)

    @staticmethod
    def fallback_chain(*funcs: Callable):
        """Try functions in sequence until one succeeds."""
        last_error = None

        for func in funcs:
            try:
                return func()
            except CATCHABLE_EXCEPTIONS as e:
                last_error = e
                logger.warning(
                    "Function %s failed, trying next: %s", func.__name__, str(e)
                )
                continue

        if last_error:
            logger.error("Fallback chain exhausted. All functions failed.")
            raise last_error


def create_error_report(error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
    """Create a comprehensive error report for debugging."""
    return {
        "timestamp": datetime.now().isoformat(),
        "error_type": type(error).__name__,
        "error_message": str(error),
        "stack_trace": traceback.format_exc(),
        "context": context,
        "python_version": __import__("sys").version,
        "streamlit_version": st.__version__,
    }
