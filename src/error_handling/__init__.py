"""Error handling module for the aicvgen application."""

from src.error_handling.exceptions import (
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
from src.error_handling.boundaries import StreamlitErrorBoundary
from src.error_handling.classification import is_retryable_error
from src.error_handling.models import (
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
    # Classification
    "is_retryable_error",
    # Models
    "ErrorCategory",
    "ErrorContext",
    "ErrorSeverity",
    "StructuredError",
]