"""Custom exception classes for the aicvgen application.

This module defines a hierarchy of custom exceptions to replace brittle string-based
error classification with explicit, type-based error handling. This establishes
a clear contract between error-producing and error-handling code.
"""


class AicvgenError(Exception):
    """Base class for all application-specific errors."""


class WorkflowPreconditionError(ValueError, AicvgenError):
    """Raised when a condition for starting a workflow is not met (e.g., missing data)."""


class LLMResponseParsingError(ValueError, AicvgenError):
    """Raised when the response from an LLM cannot be parsed into the expected format."""

    def __init__(self, message: str, raw_response: str = ""):
        self.raw_response = raw_response
        if raw_response:
            super().__init__(
                f"{message}. Raw response snippet: {raw_response[:200]}..."
            )
        else:
            super().__init__(message)


class AgentExecutionError(AicvgenError):
    """Raised when an agent fails during its execution."""

    def __init__(self, agent_name: str, message: str):
        self.agent_name = agent_name
        super().__init__(f"Agent '{agent_name}' failed: {message}")


class ConfigurationError(AicvgenError):
    """Raised for configuration-related issues."""


class ServiceInitializationError(AicvgenError):
    """Raised when a service fails to initialize properly."""


class StateManagerError(AicvgenError):
    """Raised for state management related issues."""


class ValidationError(ValueError, AicvgenError):
    """Raised for data validation errors."""


class TemplateError(AicvgenError):
    """Raised for template-related errors."""


class RateLimitError(AicvgenError):
    """Raised when rate limits are exceeded."""


class NetworkError(AicvgenError):
    """Raised for network-related errors."""


class OperationTimeoutError(AicvgenError):
    """Raised when operations timeout."""


class VectorStoreError(AicvgenError):
    """Raised when vector store operations fail."""
