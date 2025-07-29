"""Custom retry predicates for tenacity retry logic."""

from src.error_handling.exceptions import (
    NetworkError,
    OperationTimeoutError,
    RateLimitError,
)


def is_transient_error(exception: Exception) -> bool:
    """Custom predicate to determine if an error is transient and worth retrying.

    Uses explicit exception types instead of string matching for robust error classification.
    """
    # Direct exception type checks for known transient errors
    if isinstance(exception, (NetworkError, OperationTimeoutError, RateLimitError)):
        return True

    # Check for standard library transient exceptions
    if isinstance(exception, (ConnectionError, TimeoutError, IOError)):
        return True

    # Check for HTTP-related transient errors by exception attributes
    if hasattr(exception, "status_code"):
        transient_status_codes = {429, 500, 502, 503, 504}
        return exception.status_code in transient_status_codes

    # Check for Google API specific transient errors
    if hasattr(exception, "code"):
        # Google API error codes for transient errors
        transient_codes = {429, 500, 502, 503, 504}
        return exception.code in transient_codes

    return False
