#!/usr/bin/env python3
"""
Error Boundaries for Streamlit Components

This module provides error boundary decorators and context managers for handling
errors gracefully in Streamlit applications, preventing crashes and providing
user-friendly error messages.
"""

import functools
import time
import traceback
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Callable, Dict, Optional

import streamlit as st
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception

from .exceptions import (CATCHABLE_EXCEPTIONS, AicvgenError, NetworkError, OperationTimeoutError)
from .models import ErrorSeverity
from ..config.logging_config import get_logger, log_error_with_context
from ..utils.retry_predicates import is_transient_error


logger = get_logger(__name__)

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

        logger = get_logger("error_boundaries")
        log_error_with_context(logger, f"Error in {self.component_name}.{func_name}", error)

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
            logger.error(f"Connection error in {func.__name__}: {str(e)}")
        except OperationTimeoutError as e:
            st.error("⏱️ Timeout Error: The request took too long. Please try again.")
            logger.error(f"Timeout error in {func.__name__}: {str(e)}")
        except (ValueError, TypeError) as e:
            st.error(f"📝 Input Error: Invalid data provided. {e}")
            logger.error(f"Data validation error in {func.__name__}: {str(e)}")
        except AicvgenError as e:
            st.error(f"❌ An application error occurred: {e}")
            log_error_with_context(
                logger, f"API error in {func.__name__}", e
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
            logger.error(f"File not found in {func.__name__}: {str(e)}")
        except PermissionError as e:
            st.error(f"🔒 Permission Error: Unable to access the file. {e}")
            logger.error(f"Permission error in {func.__name__}: {str(e)}")
        except (OSError, IOError) as e:
            st.error(
                f"💾 File System Error: A problem occurred with the file system. {e}"
            )
            logger.error(f"OS/IO error in {func.__name__}: {str(e)}")
        except AicvgenError as e:
            st.error(f"❌ File Operation Error: An unexpected error occurred. {e}")
            log_error_with_context(
                logger, f"File operation error in {func.__name__}", e
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
            logger.error(f"Key/Index error in {func.__name__}: {str(e)}")
        except (TypeError, ValueError) as e:
            st.error(f"🔧 Type/Value Error: Data format is incorrect or invalid. {e}")
            logger.error(f"Type/Value error in {func.__name__}: {str(e)}")
        except AicvgenError as e:
            st.error(f"❌ Data Processing Error: An application error occurred. {e}")
            log_error_with_context(
                logger, f"Data processing error in {func.__name__}", e
            )
        return None

    return wrapper


class ErrorRecovery:
    """Utility class for error recovery strategies."""

    @staticmethod
    def retry_with_backoff(
        func: Callable, max_retries: int = 3, backoff_factor: float = 1.0
    ):
        """Retry a function with exponential backoff for transient errors using tenacity."""
        
        # Create a tenacity retry decorator with custom predicate function
        retry_decorator = retry(
            stop=stop_after_attempt(max_retries),
            wait=wait_exponential(multiplier=backoff_factor, min=1, max=60),
            retry=retry_if_exception(is_transient_error),
            reraise=True
        )
        
        # Apply the decorator to the function and execute it
        decorated_func = retry_decorator(func)
        return decorated_func()

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
