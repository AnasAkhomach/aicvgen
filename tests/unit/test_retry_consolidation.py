"""Tests for consolidated retry logic in EnhancedLLMService."""

import pytest
from unittest.mock import Mock, patch
from src.services.llm_service import EnhancedLLMService
from src.utils.exceptions import RateLimitError, NetworkError, LLMResponseParsingError
from src.config.settings import LLMConfig


class TestRetryConsolidation:
    """Test suite for centralized retry logic."""

    @pytest.fixture
    def llm_service(self):
        """Create an LLMService instance for testing."""
        config = LLMConfig()
        with patch('src.services.llm_service.genai'):
            service = EnhancedLLMService(config)
            return service

    def test_should_retry_with_delay_rate_limit_error(self, llm_service):
        """Test retry logic for rate limit errors."""
        exception = RateLimitError("Rate limit exceeded")
        
        # First retry should return True with exponential delay (base 5 seconds with jitter)
        should_retry, delay = llm_service._should_retry_with_delay(exception, 0, 5)
        assert should_retry is True
        assert delay >= 4.5  # Base delay for rate limits with jitter tolerance
        assert delay <= 5.5  # Should be around 5 seconds with ±10% jitter
        
        # Second retry should have longer base delay
        should_retry2, delay2 = llm_service._should_retry_with_delay(exception, 1, 5)
        assert should_retry2 is True
        assert delay2 >= 9  # Base delay: 2^1 * 5 = 10, minus jitter
        
        # Max retries exceeded
        should_retry3, delay3 = llm_service._should_retry_with_delay(exception, 5, 5)
        assert should_retry3 is False
        assert delay3 == 0.0

    def test_should_retry_with_delay_network_error(self, llm_service):
        """Test retry logic for network errors."""
        exception = NetworkError("Connection timeout")
        
        should_retry, delay = llm_service._should_retry_with_delay(exception, 0, 3)
        assert should_retry is True
        assert delay == 2  # Linear backoff: (0 + 1) * 2
        
        should_retry2, delay2 = llm_service._should_retry_with_delay(exception, 1, 3)
        assert should_retry2 is True
        assert delay2 == 4  # Linear backoff: (1 + 1) * 2

    def test_should_retry_with_delay_api_error(self, llm_service):
        """Test retry logic for API errors."""
        exception = Exception("500 Internal Server Error")
        
        should_retry, delay = llm_service._should_retry_with_delay(exception, 0, 3)
        assert should_retry is True
        assert delay == 1  # Exponential: 2^0
        
        should_retry2, delay2 = llm_service._should_retry_with_delay(exception, 1, 3)
        assert should_retry2 is True
        assert delay2 == 2  # Exponential: 2^1

    def test_should_retry_with_delay_non_retryable_error(self, llm_service):
        """Test that non-retryable errors are not retried."""
        exception = LLMResponseParsingError("Invalid JSON response")
        
        should_retry, delay = llm_service._should_retry_with_delay(exception, 0, 3)
        assert should_retry is False
        assert delay == 0.0

    def test_should_retry_with_delay_auth_error(self, llm_service):
        """Test that authentication errors are not retried."""
        exception = Exception("Invalid API key")
        
        should_retry, delay = llm_service._should_retry_with_delay(exception, 0, 3)
        assert should_retry is False
        assert delay == 0.0

    def test_should_retry_with_delay_generic_error(self, llm_service):
        """Test retry logic for generic retryable errors."""
        exception = Exception("Temporary failure")
        
        should_retry, delay = llm_service._should_retry_with_delay(exception, 0, 3)
        assert should_retry is True
        assert delay == 0  # Linear: 0 * 1
        
        should_retry2, delay2 = llm_service._should_retry_with_delay(exception, 2, 3)
        assert should_retry2 is True
        assert delay2 == 2  # Linear: 2 * 1

    def test_delay_caps(self, llm_service):
        """Test that delays are properly capped."""
        # Rate limit error with high retry count
        rate_limit_exception = RateLimitError("Rate limit exceeded")
        should_retry, delay = llm_service._should_retry_with_delay(rate_limit_exception, 10, 15)
        assert should_retry is True
        assert delay <= 300  # Should be capped at 5 minutes
        
        # Network error with high retry count
        network_exception = NetworkError("Connection timeout")
        should_retry2, delay2 = llm_service._should_retry_with_delay(network_exception, 50, 60)
        assert should_retry2 is True
        assert delay2 <= 60  # Should be capped at 1 minute
        
        # API error with high retry count
        api_exception = Exception("500 Internal Server Error")
        should_retry3, delay3 = llm_service._should_retry_with_delay(api_exception, 10, 15)
        assert should_retry3 is True
        assert delay3 <= 30  # Should be capped at 30 seconds

    def test_jitter_in_rate_limit_delays(self, llm_service):
        """Test that rate limit delays include jitter."""
        # Get multiple delay values for the same retry count with different exceptions
        delays = []
        for i in range(10):
            # Create different exception instances to vary the hash
            exception = RateLimitError(f"Rate limit exceeded - attempt {i}")
            _, delay = llm_service._should_retry_with_delay(exception, 1, 5)
            delays.append(delay)
        
        # Delays should vary due to jitter (not all exactly the same)
        # With different exception messages, the hash-based jitter should create variation
        unique_delays = set(delays)
        assert len(unique_delays) > 1, f"Expected jitter to create variation in delays, got: {delays}"