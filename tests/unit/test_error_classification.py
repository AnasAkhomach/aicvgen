import pytest
from src.utils.error_classification import (
    is_rate_limit_error,
    is_network_error,
    is_api_auth_error,
    is_retryable_error,
    get_retry_delay_for_error,
)


class DummyRateLimitError(Exception):
    pass


class DummyNetworkError(Exception):
    pass


class DummyAuthError(Exception):
    pass


def test_is_rate_limit_error():
    assert is_rate_limit_error(DummyRateLimitError("rate limit exceeded"))
    assert is_rate_limit_error(Exception("Too many requests"))
    assert not is_rate_limit_error(Exception("other error"))


def test_is_network_error():
    assert is_network_error(DummyNetworkError("network timeout"))
    assert is_network_error(Exception("connection lost"))
    assert not is_network_error(Exception("other error"))


def test_is_api_auth_error():
    assert is_api_auth_error(DummyAuthError("invalid api key"))
    assert is_api_auth_error(Exception("unauthorized"))
    assert not is_api_auth_error(Exception("other error"))


def test_is_retryable_error():
    # Should retry for rate limit/network/timeout
    assert is_retryable_error(Exception("rate limit"))
    assert is_retryable_error(Exception("network error"))
    assert is_retryable_error(Exception("timeout"))
    # Should not retry for auth error
    assert not is_retryable_error(Exception("invalid api key"))
    # Should not retry for non-retryable patterns
    assert not is_retryable_error(Exception("bad request"))
    # Default to retryable for unknown errors
    assert is_retryable_error(Exception("unknown error"))


def test_get_retry_delay_for_error():
    # Rate limit error: exponential backoff
    delay1 = get_retry_delay_for_error(Exception("rate limit"), 0)
    delay2 = get_retry_delay_for_error(Exception("rate limit"), 2)
    assert delay2 > delay1
    # Network error: moderate delay
    delay3 = get_retry_delay_for_error(Exception("network error"), 1)
    assert delay3 > 0
    # Other error: short delay
    delay4 = get_retry_delay_for_error(Exception("other error"), 1)
    assert delay4 > 0
