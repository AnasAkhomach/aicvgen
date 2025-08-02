"""Custom exception classes for the aicvgen application.

This module defines a hierarchy of custom exceptions to replace brittle string-based
error classification with explicit, type-based error handling. This establishes
a clear contract between error-producing and error-handling code.
"""

import traceback
from typing import Any, Dict, Optional

from .models import ErrorCategory, ErrorContext, ErrorSeverity, StructuredError


class AicvgenError(Exception):
    """Base class for all application-specific errors with enhanced context preservation."""

    def __init__(
        self,
        message: str,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[ErrorContext] = None,
        original_exception: Optional[Exception] = None,
        **kwargs,
    ):
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.context = context or ErrorContext()
        self.original_exception = original_exception
        self.additional_data = kwargs
        # Only capture stack trace if we're actually in an exception context
        current_trace = traceback.format_exc()
        self.stack_trace = (
            current_trace
            if current_trace != "NoneType: None\n" and "Traceback" in current_trace
            else None
        )

    def to_structured_error(self) -> StructuredError:
        """Convert to structured error format."""
        return StructuredError(
            message=self.message,
            category=self.category,
            severity=self.severity,
            context=self.context,
            original_exception=self.original_exception,
            stack_trace=self.stack_trace,
        )

    def with_context(self, **context_updates) -> "AicvgenError":
        """Create a copy with updated context."""
        if self.context:
            for key, value in context_updates.items():
                if hasattr(self.context, key):
                    setattr(self.context, key, value)
                else:
                    self.context.additional_data[key] = value
        return self


class WorkflowPreconditionError(ValueError, AicvgenError):
    """Raised when a condition for starting a workflow is not met (e.g., missing data)."""

    def __init__(self, message: str, missing_data: Optional[str] = None, **kwargs):
        # Initialize ValueError first
        ValueError.__init__(self, message)
        # Then initialize AicvgenError
        AicvgenError.__init__(
            self,
            message=message,
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.MEDIUM,
            **kwargs,
        )
        if missing_data:
            self.context.additional_data["missing_data"] = missing_data


class LLMResponseParsingError(ValueError, AicvgenError):
    """Raised when the response from an LLM cannot be parsed into the expected format."""

    def __init__(self, message: str, raw_response: str = "", **kwargs):
        full_message = f"{message}"
        if raw_response:
            full_message += f". Raw response snippet: {raw_response[:200]}..."

        # Initialize ValueError first
        ValueError.__init__(self, full_message)
        # Then initialize AicvgenError
        AicvgenError.__init__(
            self,
            message=full_message,
            category=ErrorCategory.PARSING,
            severity=ErrorSeverity.HIGH,
            **kwargs,
        )

        self.context.additional_data.update(
            {
                "raw_response": raw_response,
                "response_length": len(raw_response) if raw_response else 0,
            }
        )


class AgentExecutionError(AicvgenError):
    """Raised when an agent fails during its execution."""

    def __init__(self, agent_name: str, message: str, **kwargs):
        super().__init__(
            message=f"Agent '{agent_name}' failed: {message}",
            category=ErrorCategory.AGENT_LIFECYCLE,
            severity=ErrorSeverity.HIGH,
            **kwargs,
        )
        self.context.component = agent_name
        self.context.additional_data["agent_name"] = agent_name


class ConfigurationError(AicvgenError):
    """Raised for configuration-related issues."""

    def __init__(self, message: str, config_key: Optional[str] = None, **kwargs):
        super().__init__(
            message=message,
            category=ErrorCategory.CONFIGURATION,
            severity=ErrorSeverity.HIGH,
            **kwargs,
        )
        if config_key:
            self.context.additional_data["config_key"] = config_key


class ServiceInitializationError(AicvgenError):
    """Raised when a service fails to initialize properly."""

    def __init__(self, service_name: str, message: str, **kwargs):
        super().__init__(
            message=f"Service '{service_name}' initialization failed: {message}",
            category=ErrorCategory.CONFIGURATION,
            severity=ErrorSeverity.CRITICAL,
            **kwargs,
        )
        self.context.component = service_name
        self.context.additional_data["service_name"] = service_name


class StateManagerError(AicvgenError):
    """Raised for state management related issues."""

    def __init__(self, message: str, state_key: Optional[str] = None, **kwargs):
        super().__init__(
            message=message,
            category=ErrorCategory.DATABASE,
            severity=ErrorSeverity.HIGH,
            **kwargs,
        )
        if state_key:
            self.context.additional_data["state_key"] = state_key


class ValidationError(ValueError, AicvgenError):
    """Raised for data validation errors."""

    def __init__(self, message: str, field_name: Optional[str] = None, **kwargs):
        # Initialize ValueError first
        ValueError.__init__(self, message)
        # Then initialize AicvgenError
        AicvgenError.__init__(
            self,
            message=message,
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.MEDIUM,
            **kwargs,
        )
        if field_name:
            self.context.additional_data["field_name"] = field_name


class TemplateError(AicvgenError):
    """Raised for template-related errors."""

    def __init__(self, message: str, template_name: Optional[str] = None, **kwargs):
        super().__init__(
            message=message,
            category=ErrorCategory.GENERATION,
            severity=ErrorSeverity.MEDIUM,
            **kwargs,
        )
        if template_name:
            self.context.additional_data["template_name"] = template_name


class TemplateFormattingError(TemplateError):
    """Raised when formatting a template fails, e.g., due to missing keys."""

    def __init__(self, message: str, missing_keys: Optional[list] = None, **kwargs):
        super().__init__(message=message, **kwargs)
        self.severity = ErrorSeverity.HIGH
        if missing_keys:
            self.context.additional_data["missing_keys"] = missing_keys


class RateLimitError(AicvgenError):
    """Raised when rate limits are exceeded."""

    def __init__(self, message: str, retry_after: Optional[int] = None, **kwargs):
        super().__init__(
            message=message,
            category=ErrorCategory.RATE_LIMIT,
            severity=ErrorSeverity.MEDIUM,
            **kwargs,
        )
        if retry_after:
            self.context.additional_data["retry_after"] = retry_after


class NetworkError(AicvgenError):
    """Raised for network-related errors."""

    def __init__(self, message: str, status_code: Optional[int] = None, **kwargs):
        super().__init__(
            message=message,
            category=ErrorCategory.NETWORK,
            severity=ErrorSeverity.HIGH,
            **kwargs,
        )
        if status_code:
            self.context.additional_data["status_code"] = status_code


class OperationTimeoutError(AicvgenError):
    """Raised when operations timeout."""

    def __init__(
        self, message: str, timeout_duration: Optional[float] = None, **kwargs
    ):
        super().__init__(
            message=message,
            category=ErrorCategory.TIMEOUT,
            severity=ErrorSeverity.MEDIUM,
            **kwargs,
        )
        if timeout_duration:
            self.context.additional_data["timeout_duration"] = timeout_duration


class DataConversionError(AicvgenError):
    """Raised when data conversion fails."""

    def __init__(
        self,
        message: str,
        source_type: Optional[str] = None,
        target_type: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            message=message,
            category=ErrorCategory.PARSING,
            severity=ErrorSeverity.MEDIUM,
            **kwargs,
        )
        if source_type:
            self.context.additional_data["source_type"] = source_type
        if target_type:
            self.context.additional_data["target_type"] = target_type


class VectorStoreError(AicvgenError):
    """Raised when vector store operations fail."""

    def __init__(self, message: str, operation: Optional[str] = None, **kwargs):
        super().__init__(
            message=message,
            category=ErrorCategory.DATABASE,
            severity=ErrorSeverity.HIGH,
            **kwargs,
        )
        if operation:
            self.context.operation = operation


class DependencyError(AicvgenError):
    """Raised when a required dependency is missing or fails to load."""

    def __init__(self, message: str, dependency_name: Optional[str] = None, **kwargs):
        super().__init__(
            message=message,
            category=ErrorCategory.CONFIGURATION,
            severity=ErrorSeverity.CRITICAL,
            **kwargs,
        )
        if dependency_name:
            self.context.additional_data["dependency_name"] = dependency_name


class WorkflowError(AicvgenError):
    """Raised when there are issues with workflow execution or management."""

    def __init__(self, message: str, workflow_step: Optional[str] = None, **kwargs):
        super().__init__(
            message=message,
            category=ErrorCategory.AGENT_LIFECYCLE,
            severity=ErrorSeverity.HIGH,
            **kwargs,
        )
        if workflow_step:
            self.context.additional_data["workflow_step"] = workflow_step


# Centralized tuple of common, catchable exceptions to avoid capturing system-level exceptions.
# This provides a single source of truth for the set of exceptions that should be caught
# and handled gracefully throughout the application.
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
