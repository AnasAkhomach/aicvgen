"""Error handling module for the aicvgen application."""

from .exceptions import (
    AicvgenError,
    AgentExecutionError,
    ConfigurationError,
    DataConversionError,
    DependencyError,
    LLMResponseParsingError,
    NetworkError,
    OperationTimeoutError,
    RateLimitError,
    ServiceInitializationError,
    StateManagerError,
    TemplateError,
    TemplateFormattingError,
    ValidationError,
    VectorStoreError,
    WorkflowError,
    WorkflowPreconditionError,
)
from .boundaries import StreamlitErrorBoundary
from .models import (
    ErrorCategory,
    ErrorContext,
    ErrorSeverity,
    StructuredError,
)

__all__ = [
    # Exceptions
    "AicvgenError",
    "AgentExecutionError",
    "ConfigurationError",
    "DataConversionError",
    "DependencyError",
    "LLMResponseParsingError",
    "NetworkError",
    "OperationTimeoutError",
    "RateLimitError",
    "ServiceInitializationError",
    "StateManagerError",
    "TemplateError",
    "TemplateFormattingError",
    "ValidationError",
    "VectorStoreError",
    "WorkflowError",
    "WorkflowPreconditionError",
    # Boundaries
    "StreamlitErrorBoundary",
    # Models
    "ErrorCategory",
    "ErrorContext",
    "ErrorSeverity",
    "StructuredError",
]