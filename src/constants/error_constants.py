"""Error handling constants for centralized error management.

This module contains constants used across error handling operations
to eliminate hardcoded values and improve consistency.
"""

from typing import Final


class ErrorConstants:
    """Constants for error handling and exception management."""

    # Error codes
    ERROR_CODE_VALIDATION: Final[str] = "VALIDATION_ERROR"
    ERROR_CODE_NETWORK: Final[str] = "NETWORK_ERROR"
    ERROR_CODE_RATE_LIMIT: Final[str] = "RATE_LIMIT_ERROR"
    ERROR_CODE_TIMEOUT: Final[str] = "TIMEOUT_ERROR"
    ERROR_CODE_AUTHENTICATION: Final[str] = "AUTH_ERROR"
    ERROR_CODE_PERMISSION: Final[str] = "PERMISSION_ERROR"
    ERROR_CODE_NOT_FOUND: Final[str] = "NOT_FOUND_ERROR"
    ERROR_CODE_INTERNAL: Final[str] = "INTERNAL_ERROR"
    ERROR_CODE_CONFIGURATION: Final[str] = "CONFIG_ERROR"
    ERROR_CODE_DATA_CORRUPTION: Final[str] = "DATA_CORRUPTION_ERROR"

    # Error severity levels
    SEVERITY_CRITICAL: Final[str] = "CRITICAL"
    SEVERITY_HIGH: Final[str] = "HIGH"
    SEVERITY_MEDIUM: Final[str] = "MEDIUM"
    SEVERITY_LOW: Final[str] = "LOW"
    SEVERITY_INFO: Final[str] = "INFO"

    # Retry configuration
    DEFAULT_MAX_RETRIES: Final[int] = 3
    DEFAULT_RETRY_DELAY: Final[float] = 1.0
    MAX_RETRY_DELAY: Final[float] = 60.0
    RETRY_BACKOFF_MULTIPLIER: Final[float] = 2.0
    RETRY_JITTER_ENABLED: Final[bool] = True
    RETRY_JITTER_MAX_PERCENTAGE: Final[float] = 0.1

    # Timeout configuration
    DEFAULT_OPERATION_TIMEOUT: Final[int] = 30
    DEFAULT_NETWORK_TIMEOUT: Final[int] = 60
    DEFAULT_LLM_TIMEOUT: Final[int] = 120
    DEFAULT_FILE_OPERATION_TIMEOUT: Final[int] = 10

    # Error message templates
    MSG_VALIDATION_FAILED: Final[str] = "Validation failed for {field}: {reason}"
    MSG_NETWORK_ERROR: Final[str] = "Network error occurred: {details}"
    MSG_RATE_LIMIT_EXCEEDED: Final[
        str
    ] = "Rate limit exceeded for {resource}. Retry after {retry_after} seconds"
    MSG_TIMEOUT_ERROR: Final[
        str
    ] = "Operation timed out after {timeout} seconds: {operation}"
    MSG_AUTH_FAILED: Final[str] = "Authentication failed: {reason}"
    MSG_PERMISSION_DENIED: Final[str] = "Permission denied for operation: {operation}"
    MSG_RESOURCE_NOT_FOUND: Final[str] = "Resource not found: {resource}"
    MSG_INTERNAL_ERROR: Final[str] = "Internal error occurred: {details}"
    MSG_CONFIG_ERROR: Final[str] = "Configuration error: {setting} - {reason}"
    MSG_DATA_CORRUPTION: Final[str] = "Data corruption detected in {source}: {details}"

    # Recovery strategies
    RECOVERY_RETRY: Final[str] = "RETRY"
    RECOVERY_FALLBACK: Final[str] = "FALLBACK"
    RECOVERY_SKIP: Final[str] = "SKIP"
    RECOVERY_ABORT: Final[str] = "ABORT"
    RECOVERY_MANUAL: Final[str] = "MANUAL"

    # Error context keys
    CONTEXT_OPERATION: Final[str] = "operation"
    CONTEXT_RESOURCE: Final[str] = "resource"
    CONTEXT_USER_ID: Final[str] = "user_id"
    CONTEXT_SESSION_ID: Final[str] = "session_id"
    CONTEXT_REQUEST_ID: Final[str] = "request_id"
    CONTEXT_TIMESTAMP: Final[str] = "timestamp"
    CONTEXT_STACK_TRACE: Final[str] = "stack_trace"
    CONTEXT_ERROR_CODE: Final[str] = "error_code"
    CONTEXT_SEVERITY: Final[str] = "severity"

    # HTTP status codes
    HTTP_BAD_REQUEST: Final[int] = 400
    HTTP_UNAUTHORIZED: Final[int] = 401
    HTTP_FORBIDDEN: Final[int] = 403
    HTTP_NOT_FOUND: Final[int] = 404
    HTTP_TIMEOUT: Final[int] = 408
    HTTP_RATE_LIMITED: Final[int] = 429
    HTTP_INTERNAL_ERROR: Final[int] = 500
    HTTP_BAD_GATEWAY: Final[int] = 502
    HTTP_SERVICE_UNAVAILABLE: Final[int] = 503
    HTTP_GATEWAY_TIMEOUT: Final[int] = 504

    # Rate limit error patterns
    RATE_LIMIT_PATTERNS: Final[list] = [
        r"rate.?limit",
        r"too.?many.?requests",
        r"quota.?exceeded",
        r"throttled",
        r"429",
    ]

    # Network error patterns
    NETWORK_ERROR_PATTERNS: Final[list] = [
        r"connection.?error",
        r"network.?error",
        r"timeout",
        r"connection.?refused",
        r"dns.?resolution",
        r"ssl.?error",
    ]

    # Authentication error patterns
    AUTH_ERROR_PATTERNS: Final[list] = [
        r"unauthorized",
        r"authentication.?failed",
        r"invalid.?credentials",
        r"access.?denied",
        r"401",
    ]

    # Validation error patterns
    VALIDATION_ERROR_PATTERNS: Final[list] = [
        r"validation.?error",
        r"invalid.?input",
        r"schema.?error",
        r"format.?error",
        r"400",
    ]

    # Error logging configuration
    LOG_ERROR_DETAILS: Final[bool] = True
    LOG_STACK_TRACE: Final[bool] = True
    LOG_CONTEXT_DATA: Final[bool] = True
    MAX_ERROR_MESSAGE_LENGTH: Final[int] = 1000
    MAX_STACK_TRACE_LINES: Final[int] = 50

    # Error notification thresholds
    CRITICAL_ERROR_NOTIFICATION_THRESHOLD: Final[int] = 1
    HIGH_ERROR_NOTIFICATION_THRESHOLD: Final[int] = 5
    MEDIUM_ERROR_NOTIFICATION_THRESHOLD: Final[int] = 10

    # Error aggregation
    ERROR_AGGREGATION_WINDOW_MINUTES: Final[int] = 5
    MAX_SIMILAR_ERRORS_TO_TRACK: Final[int] = 100
    ERROR_SIMILARITY_THRESHOLD: Final[float] = 0.8

    # Circuit breaker configuration
    CIRCUIT_BREAKER_FAILURE_THRESHOLD: Final[int] = 5
    CIRCUIT_BREAKER_TIMEOUT_SECONDS: Final[int] = 60
    CIRCUIT_BREAKER_HALF_OPEN_MAX_ATTEMPTS: Final[int] = 3

    # Error recovery delay configuration (in seconds)
    RATE_LIMIT_RECOVERY_DELAY: Final[float] = 60.0
    API_ERROR_RECOVERY_DELAY: Final[float] = 2.0
    NETWORK_ERROR_RECOVERY_DELAY: Final[float] = 5.0
    TIMEOUT_ERROR_RECOVERY_DELAY: Final[float] = 10.0
    SYSTEM_ERROR_RECOVERY_DELAY: Final[float] = 30.0
    UNKNOWN_ERROR_RECOVERY_DELAY: Final[float] = 5.0

    # Error recovery max retries by type
    RATE_LIMIT_MAX_RETRIES: Final[int] = 5
    API_ERROR_MAX_RETRIES: Final[int] = 3
    NETWORK_ERROR_MAX_RETRIES: Final[int] = 4
    TIMEOUT_ERROR_MAX_RETRIES: Final[int] = 2
    VALIDATION_ERROR_MAX_RETRIES: Final[int] = 0
    PARSING_ERROR_MAX_RETRIES: Final[int] = 1
    CONTENT_ERROR_MAX_RETRIES: Final[int] = 1
    SYSTEM_ERROR_MAX_RETRIES: Final[int] = 2
    UNKNOWN_ERROR_MAX_RETRIES: Final[int] = 2

    # Error history configuration
    MAX_ERROR_HISTORY_SIZE: Final[int] = 100

    # Fallback values
    FALLBACK_ERROR_MESSAGE: Final[str] = "An unexpected error occurred"
    FALLBACK_ERROR_CODE: Final[str] = "UNKNOWN_ERROR"
    FALLBACK_RETRY_COUNT: Final[int] = 1
    FALLBACK_TIMEOUT: Final[int] = 30
