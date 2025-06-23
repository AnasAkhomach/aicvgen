"""Centralized error classification utilities.

This module provides centralized logic for classifying exceptions and errors
across the application, particularly for rate limiting, network issues, and API errors.
"""

from ..utils.exceptions import RateLimitError, NetworkError, OperationTimeoutError


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
    timeout_keywords = [
        "timeout",
        "timed out",
        "time out",
        "deadline exceeded",
    ]

    return any(keyword in error_message for keyword in timeout_keywords)


def is_api_auth_error(exception: Exception) -> bool:
    """
    Checks if an exception is an API authentication/authorization error.

    Args:
        exception: The exception to classify

    Returns:
        bool: True if this is an API auth error, False otherwise
    """
    error_message = str(exception).lower()
    auth_keywords = [
        "api error",
        "invalid api key",
        "api key not found",
        "authentication failed",
        "unauthorized",
        "permission denied",
        "401",
        "403",
    ]

    return any(keyword in error_message for keyword in auth_keywords)


def is_retryable_error(exception: Exception) -> bool:
    """
    Checks if an exception is retryable (temporary failure).

    Args:
        exception: The exception to classify

    Returns:
        bool: True if this error is retryable, False otherwise
    """
    # Rate limit errors are retryable
    if is_rate_limit_error(exception):
        return True

    # Network errors are retryable
    if is_network_error(exception):
        return True

    # Timeout errors are retryable
    if is_timeout_error(exception):
        return True

    # Auth errors are NOT retryable
    if is_api_auth_error(exception):
        return False

    # Check for other non-retryable patterns
    error_message = str(exception).lower()
    non_retryable_keywords = [
        "invalid request",
        "malformed request",
        "bad request",
        "not found",
        "404",
        "validation error",
    ]

    if any(keyword in error_message for keyword in non_retryable_keywords):
        return False

    # Default to retryable for unknown errors
    return True


def get_retry_delay_for_error(exception: Exception, retry_count: int = 0) -> float:
    """
    Calculate appropriate retry delay based on error type and retry count.

    Args:
        exception: The exception that occurred
        retry_count: Number of retries already attempted

    Returns:
        float: Delay in seconds before next retry
    """
    if is_rate_limit_error(exception):
        # Rate limit errors - use exponential backoff with longer delays
        base_delay = (2**retry_count) * 5
        jitter = base_delay * 0.1 * (0.5 - abs(hash(str(exception)) % 100) / 100)
        return min(300, base_delay + jitter)  # Max 5 minutes

    elif is_network_error(exception) or is_timeout_error(exception):
        # Network/timeout errors - moderate delays
        return min(60, (retry_count + 1) * 2)  # Max 1 minute

    else:
        # Other errors - short delays
        return min(30, (retry_count + 1) * 1.5)  # Max 30 seconds
