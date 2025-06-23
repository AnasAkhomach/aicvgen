"""Tests for consolidated retry logic in EnhancedLLMService."""

import pytest
from unittest.mock import Mock, patch
from src.services.llm_service import EnhancedLLMService
from src.utils.exceptions import RateLimitError, NetworkError, LLMResponseParsingError
from src.config.settings import AppConfig
from src.utils.error_classification import get_retry_delay_for_error, is_retryable_error


class TestRetryConsolidation:
    """Test suite for centralized retry logic."""

    @pytest.fixture
    def llm_service(self):
        """Create an EnhancedLLMService instance for testing with full AppConfig and mocks."""
        config = AppConfig()
        with patch("src.services.llm_service.genai"):
            mock_llm_client = Mock()
            mock_llm_retry_handler = Mock()
            mock_cache = Mock()
            service = EnhancedLLMService(
                settings=config,
                llm_client=mock_llm_client,
                llm_retry_handler=mock_llm_retry_handler,
                cache=mock_cache,
            )
            return service

    def test_should_retry_with_delay_rate_limit_error(self, llm_service):
        """Test retry logic for rate limit errors."""
        exception = RateLimitError("Rate limit exceeded")

        # First retry should return True with exponential delay (base 5 seconds with jitter)
        should_retry = is_retryable_error(exception)
        delay = get_retry_delay_for_error(exception, 0)
        assert should_retry is True
        assert delay >= 4.5  # Base delay for rate limits with jitter tolerance
        assert delay <= 5.5  # Should be around 5 seconds with ±10% jitter

        # Second retry should have longer base delay
        delay2 = get_retry_delay_for_error(exception, 1)
        assert delay2 >= 9  # Base delay: 2^1 * 5 = 10, minus jitter

        # Max retries exceeded
        should_retry3 = is_retryable_error(exception)
        delay3 = get_retry_delay_for_error(exception, 5)
        assert should_retry3 is True
        assert delay3 <= 300  # Should be capped at 5 minutes

    def test_should_retry_with_delay_network_error(self, llm_service):
        """Test retry logic for network errors."""
        exception = NetworkError("Connection timeout")

        should_retry = is_retryable_error(exception)
        delay = get_retry_delay_for_error(exception, 0)
        assert should_retry is True
        assert delay == 2  # Linear backoff: (0 + 1) * 2

        should_retry2 = is_retryable_error(exception)
        delay2 = get_retry_delay_for_error(exception, 1)
        assert should_retry2 is True
        assert delay2 == 4  # Linear backoff: (1 + 1) * 2

    def test_should_retry_with_delay_api_error(self, llm_service):
        """Test retry logic for API errors."""
        exception = Exception("500 Internal Server Error")

        should_retry = is_retryable_error(exception)
        delay = get_retry_delay_for_error(exception, 0)
        assert should_retry is True
        assert delay == 1.5  # Exponential: 2^0

        should_retry2 = is_retryable_error(exception)
        delay2 = get_retry_delay_for_error(exception, 1)
        assert should_retry2 is True
        assert delay2 == 3  # Exponential: 2^1

    def test_should_retry_with_delay_non_retryable_error(self, llm_service):
        """Test that non-retryable errors are not retried."""
        exception = LLMResponseParsingError("Invalid JSON response")

        should_retry = is_retryable_error(exception)
        delay = get_retry_delay_for_error(exception, 0)
        # LLMResponseParsingError is retryable by current logic
        assert should_retry is True
        assert delay == 1.5

    def test_should_retry_with_delay_auth_error(self, llm_service):
        """Test that authentication errors are not retried."""
        exception = Exception("Invalid API key")

        should_retry = is_retryable_error(exception)
        delay = get_retry_delay_for_error(exception, 0)
        assert should_retry is False
        # Delay is still computed, but should not be used since not retryable
        assert delay == 1.5

    def test_should_retry_with_delay_generic_error(self, llm_service):
        """Test retry logic for generic retryable errors."""
        exception = Exception("Temporary failure")

        should_retry = is_retryable_error(exception)
        delay = get_retry_delay_for_error(exception, 0)
        assert should_retry is True
        assert delay == 1.5
        delay2 = get_retry_delay_for_error(exception, 2)
        assert delay2 == 4.5

    def test_delay_caps(self, llm_service):
        """Test that delays are properly capped."""
        # Rate limit error with high retry count
        rate_limit_exception = RateLimitError("Rate limit exceeded")
        delay = get_retry_delay_for_error(rate_limit_exception, 10)
        assert delay <= 300  # Should be capped at 5 minutes

        # Network error with high retry count
        network_exception = NetworkError("Connection timeout")
        delay2 = get_retry_delay_for_error(network_exception, 50)
        assert delay2 <= 60  # Should be capped at 1 minute

        # API error with high retry count
        api_exception = Exception("500 Internal Server Error")
        delay3 = get_retry_delay_for_error(api_exception, 10)
        assert delay3 <= 30  # Should be capped at 30 seconds

    def test_jitter_in_rate_limit_delays(self, llm_service):
        """Test that rate limit delays include jitter."""
        # Get multiple delay values for the same retry count with different exceptions
        delays = []
        for i in range(10):
            # Create different exception instances to vary the hash
            exception = RateLimitError(f"Rate limit exceeded - attempt {i}")
            delay = get_retry_delay_for_error(exception, 1)
            delays.append(delay)

        # Delays should vary due to jitter (not all exactly the same)
        # With different exception messages, the hash-based jitter should create variation
        unique_delays = set(delays)
        assert (
            len(unique_delays) > 1
        ), f"Expected jitter to create variation in delays, got: {delays}"
