"""Centralized error classification utilities."""

import re
from src.constants.error_constants import ErrorConstants
from src.error_handling.exceptions import (NetworkError, OperationTimeoutError, RateLimitError)
from src.error_handling.models import ErrorCategory


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

    # Check error message for rate limit patterns
    error_message = str(exception).lower()
    return any(
        re.search(pattern, error_message, re.IGNORECASE)
        for pattern in ErrorConstants.RATE_LIMIT_PATTERNS
    )


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

    # Check error message for network patterns
    error_message = str(exception).lower()
    return any(
        re.search(pattern, error_message, re.IGNORECASE)
        for pattern in ErrorConstants.NETWORK_ERROR_PATTERNS
    )


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

    # Check error message for timeout patterns
    error_message = str(exception).lower()
    return "timeout" in error_message
