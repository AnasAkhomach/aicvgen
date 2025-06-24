"""Centralized error classification utilities."""

from src.error_handling.models import ErrorCategory
from src.utils.exceptions import RateLimitError, NetworkError, OperationTimeoutError


def is_retryable_error(exception: Exception) -> tuple[bool, str]:
    """
    Checks if an exception suggests a retry is warranted.

    Args:
        exception: The exception to classify.

    Returns:
        A tuple containing a boolean (True if retryable) and a string
        representing the error type.
    """
    if is_rate_limit_error(exception):
        return True, ErrorCategory.RATE_LIMIT.value
    if is_network_error(exception):
        return True, ErrorCategory.NETWORK.value
    if is_timeout_error(exception):
        return True, ErrorCategory.TIMEOUT.value

    # Default to not retryable
    return False, ErrorCategory.UNKNOWN.value


def is_rate_limit_error(exception: Exception) -> bool:
    """
    Checks if an exception is a rate-limit error.

    Args:
        exception: The exception to classify

    Returns:
        bool: True if this is a rate limit error, False otherwise
    """
    # Check for specific rate limit exception types
    if isinstance(exception, RateLimitError):
        return True

    # Check error message for rate limit keywords
    error_message = str(exception).lower()
    rate_limit_keywords = [
        "rate limit",
        "rate-limit",
        "too many requests",
        "quota exceeded",
        "quota_exceeded",
        "resource_exhausted",
        "resource exhausted",
        "429",
    ]

    return any(keyword in error_message for keyword in rate_limit_keywords)


def is_network_error(exception: Exception) -> bool:
    """
    Checks if an exception is a network-related error.

    Args:
        exception: The exception to classify

    Returns:
        bool: True if this is a network error, False otherwise
    """
    # Check for specific network exception types
    if isinstance(exception, NetworkError):
        return True

    # Check error message for network keywords
    error_message = str(exception).lower()
    network_keywords = [
        "connection",
        "network",
        "timeout",
        "dns",
        "unreachable",
        "host",
        "socket",
        "ssl",
        "certificate",
    ]

    return any(keyword in error_message for keyword in network_keywords)


def is_timeout_error(exception: Exception) -> bool:
    """
    Checks if an exception is a timeout error.

    Args:
        exception: The exception to classify

    Returns:
        bool: True if this is a timeout error, False otherwise
    """
    # Check for specific timeout exception types
    if isinstance(exception, OperationTimeoutError):
        return True

    # Check error message for timeout keywords
    error_message = str(exception).lower()
    return "timeout" in error_message
