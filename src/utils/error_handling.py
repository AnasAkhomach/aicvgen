"""Enhanced error handling utilities for AI CV Generator.

This module provides standardized error handling, propagation mechanisms,
and contextual error information for better debugging and user experience.
"""

import traceback
import logging
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from uuid import uuid4

# Import security utilities for safe error logging
try:
    from src.utils.security_utils import redact_sensitive_data, redact_log_message
except ImportError:
    def redact_sensitive_data(data):
        return data
    def redact_log_message(message):
        return message


class ErrorSeverity(str, Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(str, Enum):
    """Error categories for classification."""
    VALIDATION = "validation"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    RATE_LIMIT = "rate_limit"
    API_ERROR = "api_error"
    NETWORK = "network"
    TIMEOUT = "timeout"
    PARSING = "parsing"
    GENERATION = "generation"
    FILE_IO = "file_io"
    DATABASE = "database"
    CONFIGURATION = "configuration"
    UNKNOWN = "unknown"


@dataclass
class ErrorContext:
    """Contextual information for errors."""
    error_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    component: Optional[str] = None
    operation: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    additional_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StructuredError:
    """Structured error information."""
    message: str
    category: ErrorCategory
    severity: ErrorSeverity
    context: ErrorContext
    original_exception: Optional[Exception] = None
    stack_trace: Optional[str] = None
    user_message: Optional[str] = None
    recovery_suggestions: List[str] = field(default_factory=list)
    related_errors: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Post-initialization processing."""
        if self.original_exception and not self.stack_trace:
            self.stack_trace = traceback.format_exc()

        if not self.user_message:
            self.user_message = self._generate_user_message()

    def _generate_user_message(self) -> str:
        """Generate user-friendly error message."""
        category_messages = {
            ErrorCategory.VALIDATION: "Please check your input data and try again.",
            ErrorCategory.AUTHENTICATION: "Authentication failed. Please check your credentials.",
            ErrorCategory.AUTHORIZATION: "You don't have permission to perform this action.",
            ErrorCategory.RATE_LIMIT: "Too many requests. Please wait a moment and try again.",
            ErrorCategory.API_ERROR: "External service error. Please try again later.",
            ErrorCategory.NETWORK: "Network connection error. Please check your internet connection.",
            ErrorCategory.TIMEOUT: "Operation timed out. Please try again.",
            ErrorCategory.PARSING: "Failed to process the provided data. Please check the format.",
            ErrorCategory.GENERATION: "Content generation failed. Please try again.",
            ErrorCategory.FILE_IO: "File operation failed. Please check file permissions.",
            ErrorCategory.DATABASE: "Database error. Please try again later.",
            ErrorCategory.CONFIGURATION: "Configuration error. Please contact support.",
            ErrorCategory.UNKNOWN: "An unexpected error occurred. Please try again."
        }
        return category_messages.get(self.category, "An error occurred. Please try again.")

    def to_dict(self, include_sensitive: bool = False) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        data = {
            "error_id": self.context.error_id,
            "timestamp": self.context.timestamp.isoformat(),
            "message": self.message,
            "category": self.category.value,
            "severity": self.severity.value,
            "user_message": self.user_message,
            "recovery_suggestions": self.recovery_suggestions,
            "component": self.context.component,
            "operation": self.context.operation
        }

        if include_sensitive:
            data.update({
                "stack_trace": self.stack_trace,
                "user_id": self.context.user_id,
                "session_id": self.context.session_id,
                "request_id": self.context.request_id,
                "additional_data": self.context.additional_data,
                "related_errors": self.related_errors
            })

        return redact_sensitive_data(data) if not include_sensitive else data


class ErrorHandler:
    """Centralized error handling and logging."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self._error_history: List[StructuredError] = []

    def handle_error(
        self,
        exception: Union[Exception, str],
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[ErrorContext] = None,
        user_message: Optional[str] = None,
        recovery_suggestions: Optional[List[str]] = None
    ) -> StructuredError:
        """Handle and log an error with structured information."""

        if context is None:
            context = ErrorContext()

        if isinstance(exception, str):
            message = exception
            original_exception = None
        else:
            message = str(exception)
            original_exception = exception

        structured_error = StructuredError(
            message=message,
            category=category,
            severity=severity,
            context=context,
            original_exception=original_exception,
            user_message=user_message,
            recovery_suggestions=recovery_suggestions or []
        )

        # Log the error
        self._log_error(structured_error)

        # Store in history (keep last 100 errors)
        self._error_history.append(structured_error)
        if len(self._error_history) > 100:
            self._error_history.pop(0)

        return structured_error

    def _log_error(self, error: StructuredError):
        """Log structured error with appropriate level."""
        log_data = error.to_dict(include_sensitive=True)
        safe_log_data = redact_sensitive_data(log_data)

        log_message = f"Error {error.context.error_id}: {error.message}"

        if error.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(log_message, extra={"error_data": safe_log_data})
        elif error.severity == ErrorSeverity.HIGH:
            self.logger.error(log_message, extra={"error_data": safe_log_data})
        elif error.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(log_message, extra={"error_data": safe_log_data})
        else:
            self.logger.info(log_message, extra={"error_data": safe_log_data})

    def get_error_history(self, limit: int = 10) -> List[StructuredError]:
        """Get recent error history."""
        return self._error_history[-limit:]

    def get_error_by_id(self, error_id: str) -> Optional[StructuredError]:
        """Get error by ID."""
        for error in self._error_history:
            if error.context.error_id == error_id:
                return error
        return None

    def clear_history(self):
        """Clear error history."""
        self._error_history.clear()


# Global error handler instance
_global_error_handler = ErrorHandler()


def handle_error(
    exception: Union[Exception, str],
    category: ErrorCategory = ErrorCategory.UNKNOWN,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    context: Optional[ErrorContext] = None,
    user_message: Optional[str] = None,
    recovery_suggestions: Optional[List[str]] = None
) -> StructuredError:
    """Global error handling function."""
    return _global_error_handler.handle_error(
        exception=exception,
        category=category,
        severity=severity,
        context=context,
        user_message=user_message,
        recovery_suggestions=recovery_suggestions
    )


def get_error_handler() -> ErrorHandler:
    """Get the global error handler instance."""
    return _global_error_handler


class ErrorBoundary:
    """Context manager for error boundary handling."""

    def __init__(
        self,
        component: str,
        operation: str,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        user_message: Optional[str] = None,
        recovery_suggestions: Optional[List[str]] = None,
        reraise: bool = True
    ):
        self.component = component
        self.operation = operation
        self.category = category
        self.severity = severity
        self.user_message = user_message
        self.recovery_suggestions = recovery_suggestions
        self.reraise = reraise
        self.error: Optional[StructuredError] = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            context = ErrorContext(
                component=self.component,
                operation=self.operation
            )

            self.error = handle_error(
                exception=exc_val,
                category=self.category,
                severity=self.severity,
                context=context,
                user_message=self.user_message,
                recovery_suggestions=self.recovery_suggestions
            )

            if not self.reraise:
                return True  # Suppress the exception

        return False  # Let the exception propagate


def create_error_context(
    component: Optional[str] = None,
    operation: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    request_id: Optional[str] = None,
    **additional_data
) -> ErrorContext:
    """Create an error context with the provided information."""
    return ErrorContext(
        component=component,
        operation=operation,
        user_id=user_id,
        session_id=session_id,
        request_id=request_id,
        additional_data=additional_data
    )
