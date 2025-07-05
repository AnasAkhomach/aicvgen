"""Tests for rate limiter service."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta

from src.services.rate_limiter import (
    RateLimiter,
    RateLimitConfig,
    RetryableRateLimiter,
    get_rate_limiter,
    reset_rate_limiter,
)
from src.error_handling.exceptions import RateLimitError, NetworkError
from src.models.llm_data_models import RateLimitState


class TestRateLimiter:
    """Test cases for RateLimiter class."""

    @pytest.fixture
    def rate_limiter(self):
        """Create a rate limiter instance for testing."""
        config = RateLimitConfig(
            requests_per_minute=10,
            tokens_per_minute=1000,
            max_retries=3,
            base_backoff_seconds=1.0,
            max_backoff_seconds=60.0,
        )
        return RateLimiter(config)

    @pytest.fixture
    def mock_logger(self, rate_limiter):
        """Mock logger for testing."""
        mock_logger = MagicMock()
        rate_limiter.logger = mock_logger
        return mock_logger

    def test_init(self, rate_limiter):
        """Test rate limiter initialization."""
        assert rate_limiter.config.requests_per_minute == 10
        assert rate_limiter.config.tokens_per_minute == 1000
        assert isinstance(rate_limiter.model_states, dict)
        assert len(rate_limiter.model_states) == 0

    def test_get_model_state_creates_new_state(self, rate_limiter):
        """Test that get_model_state creates new state for unknown model."""
        model = "test-model"
        state = rate_limiter.get_model_state(model)
        
        assert isinstance(state, RateLimitState)
        assert state.model == model
        assert state.requests_made == 0
        assert state.requests_limit == 10
        assert model in rate_limiter.model_states

    def test_get_model_state_returns_existing_state(self, rate_limiter):
        """Test that get_model_state returns existing state."""
        model = "test-model"
        state1 = rate_limiter.get_model_state(model)
        state2 = rate_limiter.get_model_state(model)
        
        assert state1 is state2

    def test_can_make_request(self, rate_limiter):
        """Test can_make_request method."""
        model = "test-model"
        
        # Should be able to make request initially
        assert rate_limiter.can_make_request(model) is True
        
        # Simulate hitting rate limit
        state = rate_limiter.get_model_state(model)
        state.requests_made = 10  # At limit
        
        assert rate_limiter.can_make_request(model) is False

    def test_get_wait_time_no_wait_needed(self, rate_limiter):
        """Test get_wait_time when no wait is needed."""
        model = "test-model"
        wait_time = rate_limiter.get_wait_time(model)
        
        assert wait_time == 0.0

    def test_get_wait_time_backoff_period(self, rate_limiter):
        """Test get_wait_time during backoff period."""
        model = "test-model"
        state = rate_limiter.get_model_state(model)
        
        # Set backoff until future time
        future_time = datetime.now() + timedelta(seconds=30)
        state.backoff_until = future_time
        
        wait_time = rate_limiter.get_wait_time(model)
        assert wait_time > 0
        assert wait_time <= 30

    @pytest.mark.asyncio
    async def test_async_context_manager_entry(self, rate_limiter):
        """Test async context manager __aenter__ method."""
        result = await rate_limiter.__aenter__()
        assert result is rate_limiter

    @pytest.mark.asyncio
    async def test_async_context_manager_exit_no_exception(self, rate_limiter, mock_logger):
        """Test async context manager __aexit__ method without exception."""
        result = await rate_limiter.__aexit__(None, None, None)
        assert result is False  # Should not suppress exceptions
        mock_logger.debug.assert_not_called()

    @pytest.mark.asyncio
    async def test_async_context_manager_exit_with_exception(self, rate_limiter, mock_logger):
        """Test async context manager __aexit__ method with exception."""
        exc_type = ValueError
        exc_val = ValueError("test error")
        exc_tb = None
        
        result = await rate_limiter.__aexit__(exc_type, exc_val, exc_tb)
        
        assert result is False  # Should not suppress exceptions
        mock_logger.debug.assert_called_once_with(
            "Rate limiter context exited with exception",
            exc_type="ValueError",
            exc_val="test error"
        )

    @pytest.mark.asyncio
    async def test_async_context_manager_usage(self, rate_limiter):
        """Test using rate limiter as async context manager."""
        async with rate_limiter as limiter:
            assert limiter is rate_limiter

    @pytest.mark.asyncio
    async def test_wait_if_needed_async_no_wait(self, rate_limiter, mock_logger):
        """Test wait_if_needed_async when no wait is needed."""
        model = "test-model"
        
        # Should complete immediately
        await rate_limiter.wait_if_needed_async(model)
        
        # Should not log any wait messages
        mock_logger.info.assert_not_called()

    @pytest.mark.asyncio
    async def test_wait_if_needed_async_with_wait(self, rate_limiter, mock_logger):
        """Test wait_if_needed_async when wait is needed."""
        model = "test-model"
        state = rate_limiter.get_model_state(model)
        
        # Force a wait condition
        state.requests_made = 10  # At limit
        
        with patch("asyncio.sleep") as mock_sleep:
            await rate_limiter.wait_if_needed_async(model)
            mock_sleep.assert_called_once()
            
        # Should log wait message
        mock_logger.info.assert_called()

    @pytest.mark.asyncio
    async def test_execute_with_rate_limit_async_success(self, rate_limiter):
        """Test execute_with_rate_limit_async with successful execution."""
        model = "test-model"
        
        async def mock_func(value):
            return {"result": value, "usage": {"total_tokens": 100}}
        
        result = await rate_limiter.execute_with_rate_limit_async(
            mock_func, model, 50, "test_value"
        )
        
        assert result["result"] == "test_value"
        
        # Check that request was recorded
        state = rate_limiter.get_model_state(model)
        assert state.requests_made == 1
        assert state.tokens_made == 100  # Should use actual tokens from result

    @pytest.mark.asyncio
    async def test_execute_with_rate_limit_async_with_error(self, rate_limiter):
        """Test execute_with_rate_limit_async with error."""
        model = "test-model"
        
        async def mock_func():
            raise ValueError("test error")
        
        with pytest.raises(NetworkError, match="API call failed"):
            await rate_limiter.execute_with_rate_limit_async(mock_func, model, 50)
        
        # Check that failed request was recorded
        state = rate_limiter.get_model_state(model)
        assert state.requests_made == 1

    def test_record_request_success(self, rate_limiter, mock_logger):
        """Test recording successful request."""
        model = "test-model"
        tokens_used = 100
        
        rate_limiter.record_request(model, tokens_used, success=True)
        
        state = rate_limiter.get_model_state(model)
        assert state.requests_made == 1
        assert state.tokens_made == tokens_used
        
        # Should log the request
        mock_logger.info.assert_called()

    def test_record_request_failure(self, rate_limiter):
        """Test recording failed request."""
        model = "test-model"
        tokens_used = 100
        
        rate_limiter.record_request(model, tokens_used, success=False)
        
        state = rate_limiter.get_model_state(model)
        assert state.requests_made == 1
        assert state.tokens_made == tokens_used


class TestRetryableRateLimiter:
    """Test cases for RetryableRateLimiter class."""

    @pytest.fixture
    def retryable_rate_limiter(self):
        """Create a retryable rate limiter instance for testing."""
        config = RateLimitConfig(
            requests_per_minute=10,
            tokens_per_minute=1000,
            max_retries=2,
        )
        return RetryableRateLimiter(config)

    def test_init(self, retryable_rate_limiter):
        """Test retryable rate limiter initialization."""
        assert isinstance(retryable_rate_limiter, RateLimiter)
        assert retryable_rate_limiter.retry_decorator is not None

    @pytest.mark.asyncio
    async def test_execute_with_retry_success(self, retryable_rate_limiter):
        """Test execute_with_retry with successful execution."""
        model = "test-model"
        
        async def mock_func(value):
            return {"result": value}
        
        result = await retryable_rate_limiter.execute_with_retry(
            mock_func, model, 50, "test_value"
        )
        
        assert result["result"] == "test_value"


class TestGlobalRateLimiter:
    """Test cases for global rate limiter functions."""

    def setup_method(self):
        """Reset global rate limiter before each test."""
        reset_rate_limiter()

    def test_get_rate_limiter_creates_instance(self):
        """Test that get_rate_limiter creates a new instance."""
        limiter = get_rate_limiter()
        assert isinstance(limiter, RetryableRateLimiter)

    def test_get_rate_limiter_returns_same_instance(self):
        """Test that get_rate_limiter returns the same instance."""
        limiter1 = get_rate_limiter()
        limiter2 = get_rate_limiter()
        assert limiter1 is limiter2

    def test_reset_rate_limiter(self):
        """Test that reset_rate_limiter clears the global instance."""
        limiter1 = get_rate_limiter()
        reset_rate_limiter()
        limiter2 = get_rate_limiter()
        assert limiter1 is not limiter2

    def test_get_rate_limiter_with_config(self):
        """Test get_rate_limiter with custom config."""
        config = RateLimitConfig(requests_per_minute=5)
        limiter = get_rate_limiter(config)
        assert limiter.config.requests_per_minute == 5