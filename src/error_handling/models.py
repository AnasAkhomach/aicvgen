"""Core error models for the application."""

import traceback
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4


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
    AGENT_LIFECYCLE = "agent_lifecycle"
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
            ErrorCategory.UNKNOWN: "An unexpected error occurred. Please try again.",
        }
        return category_messages.get(self.category, "An unexpected error occurred.")
