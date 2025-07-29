"""Tests for the refactored retry logic using explicit exception types."""

from datetime import datetime
from unittest.mock import Mock, patch

import pytest
from tenacity import TryAgain

from src.error_handling.boundaries import ErrorRecovery
from src.error_handling.exceptions import (
    NetworkError,
    OperationTimeoutError,
    RateLimitError,
)
from src.models.data_models import ContentType
from src.services.error_recovery import ErrorContext, ErrorRecoveryService, ErrorType
from src.utils.retry_predicates import is_transient_error


class TestTransientErrorPredicate:
    """Test the custom predicate function for transient errors."""

    def test_is_transient_error_with_network_error(self):
        """Test that NetworkError is identified as transient."""
        error = NetworkError("Connection failed")
        assert is_transient_error(error) is True

    def test_is_transient_error_with_rate_limit_error(self):
        """Test that RateLimitError is identified as transient."""
        error = RateLimitError("Rate limit exceeded")
        assert is_transient_error(error) is True

    def test_is_transient_error_with_timeout_error(self):
        """Test that OperationTimeoutError is identified as transient."""
        error = OperationTimeoutError("Operation timed out")
        assert is_transient_error(error) is True

    def test_is_transient_error_with_connection_error(self):
        """Test that standard ConnectionError is identified as transient."""
        error = ConnectionError("Connection refused")
        assert is_transient_error(error) is True

    def test_is_transient_error_with_timeout_error_builtin(self):
        """Test that built-in TimeoutError is identified as transient."""
        error = TimeoutError("Request timed out")
        assert is_transient_error(error) is True

    def test_is_transient_error_with_io_error(self):
        """Test that IOError is identified as transient."""
        error = IOError("I/O operation failed")
        assert is_transient_error(error) is True

    def test_is_transient_error_with_http_status_code(self):
        """Test that exceptions with transient HTTP status codes are identified as transient."""
        # Mock exception with status_code attribute
        error = Exception("Server error")
        error.status_code = 503
        assert is_transient_error(error) is True

        error.status_code = 429
        assert is_transient_error(error) is True

        error.status_code = 500
        assert is_transient_error(error) is True

    def test_is_transient_error_with_google_api_code(self):
        """Test that exceptions with Google API error codes are identified as transient."""
        # Mock exception with code attribute
        error = Exception("API error")
        error.code = 503
        assert is_transient_error(error) is True

        error.code = 429
        assert is_transient_error(error) is True

    def test_is_transient_error_with_non_transient_error(self):
        """Test that non-transient errors are not identified as transient."""
        error = ValueError("Invalid value")
        assert is_transient_error(error) is False

        error = KeyError("Key not found")
        assert is_transient_error(error) is False

    def test_is_transient_error_with_non_transient_status_code(self):
        """Test that exceptions with non-transient HTTP status codes are not identified as transient."""
        error = Exception("Client error")
        error.status_code = 404
        assert is_transient_error(error) is False

        error.status_code = 400
        assert is_transient_error(error) is False


class TestExtractRetryAfter:
    """Test the refactored _extract_retry_after method."""

    def test_extract_retry_after_from_metadata(self):
        """Test extracting retry_after from error context metadata."""
        service = ErrorRecoveryService()

        error_context = ErrorContext(
            error_type=ErrorType.RATE_LIMIT,
            error_message="Rate limit exceeded",
            item_id="test_item",
            item_type=ContentType.QUALIFICATION,
            session_id="test_session",
            timestamp=datetime.now(),
            metadata={"retry_after": 30},
        )

        retry_after = service._extract_retry_after(error_context)
        assert retry_after == 30

    def test_extract_retry_after_from_exception_attribute(self):
        """Test extracting retry_after from original exception attribute."""
        service = ErrorRecoveryService()

        # Mock exception with retry_after attribute
        exception = RateLimitError("Rate limit exceeded")
        exception.retry_after = 45

        error_context = ErrorContext(
            error_type=ErrorType.RATE_LIMIT,
            error_message="Rate limit exceeded",
            item_id="test_item",
            item_type=ContentType.QUALIFICATION,
            session_id="test_session",
            timestamp=datetime.now(),
            original_exception=exception,
        )

        retry_after = service._extract_retry_after(error_context)
        assert retry_after == 45

    def test_extract_retry_after_from_exception_headers(self):
        """Test extracting retry_after from exception headers."""
        service = ErrorRecoveryService()

        # Mock exception with headers attribute
        exception = RateLimitError("Rate limit exceeded")
        exception.headers = {"Retry-After": "60"}

        error_context = ErrorContext(
            error_type=ErrorType.RATE_LIMIT,
            error_message="Rate limit exceeded",
            item_id="test_item",
            item_type=ContentType.QUALIFICATION,
            session_id="test_session",
            timestamp=datetime.now(),
            original_exception=exception,
        )

        retry_after = service._extract_retry_after(error_context)
        assert retry_after == 60

    def test_extract_retry_after_fallback_for_rate_limit(self):
        """Test fallback retry_after value for rate limit errors."""
        service = ErrorRecoveryService()

        error_context = ErrorContext(
            error_type=ErrorType.RATE_LIMIT,
            error_message="Rate limit exceeded",
            item_id="test_item",
            item_type=ContentType.QUALIFICATION,
            session_id="test_session",
            timestamp=datetime.now(),
        )

        retry_after = service._extract_retry_after(error_context)
        assert retry_after == 60  # Default fallback for rate limit

    def test_extract_retry_after_no_fallback_for_other_errors(self):
        """Test that non-rate-limit errors return None when no retry_after is found."""
        service = ErrorRecoveryService()

        error_context = ErrorContext(
            error_type=ErrorType.NETWORK_ERROR,
            error_message="Network error",
            item_id="test_item",
            item_type=ContentType.QUALIFICATION,
            session_id="test_session",
            timestamp=datetime.now(),
        )

        retry_after = service._extract_retry_after(error_context)
        assert retry_after is None


class TestRetryWithBackoff:
    """Test the refactored retry_with_backoff method using tenacity."""

    def test_retry_with_backoff_success_on_first_attempt(self):
        """Test that successful functions are executed without retry."""
        mock_func = Mock(return_value="success")

        result = ErrorRecovery.retry_with_backoff(mock_func)

        assert result == "success"
        assert mock_func.call_count == 1

    def test_retry_with_backoff_success_after_retries(self):
        """Test that functions succeed after transient errors."""
        mock_func = Mock()
        mock_func.side_effect = [
            NetworkError("Connection failed"),
            NetworkError("Connection failed"),
            "success",
        ]

        result = ErrorRecovery.retry_with_backoff(mock_func, max_retries=3)

        assert result == "success"
        assert mock_func.call_count == 3

    def test_retry_with_backoff_exhausts_retries(self):
        """Test that retry exhaustion raises the original exception."""
        mock_func = Mock()
        mock_func.side_effect = NetworkError("Persistent connection failure")

        with pytest.raises(NetworkError, match="Persistent connection failure"):
            ErrorRecovery.retry_with_backoff(mock_func, max_retries=2)

        assert mock_func.call_count == 2

    def test_retry_with_backoff_non_transient_error(self):
        """Test that non-transient errors are not retried."""
        mock_func = Mock()
        mock_func.side_effect = ValueError("Invalid input")

        with pytest.raises(ValueError, match="Invalid input"):
            ErrorRecovery.retry_with_backoff(mock_func, max_retries=3)

        assert mock_func.call_count == 1  # No retries for non-transient errors

    @patch("time.sleep")  # Mock sleep to speed up tests
    def test_retry_with_backoff_custom_parameters(self, mock_sleep):
        """Test retry with custom max_retries and backoff_factor."""
        mock_func = Mock()
        mock_func.side_effect = [
            OperationTimeoutError("Timeout"),
            OperationTimeoutError("Timeout"),
            OperationTimeoutError("Timeout"),
            OperationTimeoutError("Timeout"),
            "success",
        ]

        result = ErrorRecovery.retry_with_backoff(
            mock_func, max_retries=5, backoff_factor=2.0
        )

        assert result == "success"
        assert mock_func.call_count == 5
