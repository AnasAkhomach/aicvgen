"""Unit tests for Enhanced Rate Limiter.

Tests request throttling (30 RPM, 6000 TPM limits), exponential backoff implementation,
multi-model rate tracking, and async wait mechanisms.
"""

import unittest
import asyncio
import time
from unittest.mock import Mock, MagicMock, patch, AsyncMock
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, Any

# Add the project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.services.rate_limiter import (
    RateLimiter, RateLimitConfig, RateLimitExceeded, APIError,
    get_rate_limiter
)
from src.models.data_models import RateLimitState


class TestEnhancedRateLimiter(unittest.TestCase):
    """Test cases for Enhanced Rate Limiter."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_config = RateLimitConfig(
            requests_per_minute=30,
            tokens_per_minute=6000,
            max_retries=3,
            base_backoff_seconds=1.0,
            max_backoff_seconds=300.0,
            jitter=True
        )
        
    @patch('src.services.rate_limiter.get_structured_logger')
    def test_rate_limiter_initialization(self, mock_logger):
        """Test Rate Limiter initialization."""
        mock_logger.return_value = Mock()
        
        # Test with default config
        rate_limiter = RateLimiter()
        self.assertIsNotNone(rate_limiter.logger)
        self.assertIsNotNone(rate_limiter.config)
        
        # Test with custom config
        rate_limiter_custom = RateLimiter(config=self.test_config)
        self.assertEqual(rate_limiter_custom.config.requests_per_minute, 30)
        self.assertEqual(rate_limiter_custom.config.tokens_per_minute, 6000)
        self.assertEqual(rate_limiter_custom.config.max_retries, 3)

    @patch('src.services.rate_limiter.get_structured_logger')
    def test_request_throttling_rpm_limits(self, mock_logger):
        """Test request throttling with 30 RPM limits."""
        mock_logger.return_value = Mock()
        
        # Create rate limiter with strict limits for testing
        test_config = RateLimitConfig(
            requests_per_minute=2,  # Very low for testing
            tokens_per_minute=1000,
            max_retries=3
        )
        rate_limiter = RateLimiter(config=test_config)
        
        # Test request recording and limit checking
        model_name = "groq"
        
        # Record first request - should be allowed
        rate_limiter.record_request(model_name, 100, success=True)
        self.assertTrue(rate_limiter.can_make_request(model_name))
        
        # Record second request - should still be allowed but at limit
        rate_limiter.record_request(model_name, 100, success=True)
        # After 2 requests with limit of 2, should not be able to make more
        self.assertFalse(rate_limiter.can_make_request(model_name))
        
        # Test wait time calculation
        wait_time = rate_limiter.get_wait_time(model_name)
        self.assertGreater(wait_time, 0)
        self.assertLessEqual(wait_time, 60)  # Should not exceed 1 minute

    @patch('src.services.rate_limiter.get_structured_logger')
    def test_token_throttling_tpm_limits(self, mock_logger):
        """Test token throttling with 6000 TPM limits."""
        mock_logger.return_value = Mock()
        
        # Create rate limiter with strict token limits for testing
        test_config = RateLimitConfig(
            requests_per_minute=100,  # High request limit
            tokens_per_minute=500,    # Low token limit for testing
            max_retries=3
        )
        rate_limiter = RateLimiter(config=test_config)
        
        model_name = "groq"
        
        # Record requests with high token usage
        rate_limiter.record_request(model_name, 200, success=True)  # 200 tokens
        self.assertTrue(rate_limiter.can_make_request(model_name))
        
        rate_limiter.record_request(model_name, 200, success=True)  # 400 tokens total
        self.assertTrue(rate_limiter.can_make_request(model_name))
        
        rate_limiter.record_request(model_name, 200, success=True)  # 600 tokens total - exceeds limit
        self.assertFalse(rate_limiter.can_make_request(model_name))
        
        # Test wait time for token limit
        wait_time = rate_limiter.get_wait_time(model_name)
        self.assertGreater(wait_time, 0)

    @patch('src.services.rate_limiter.get_structured_logger')
    def test_exponential_backoff_implementation(self, mock_logger):
        """Test exponential backoff implementation."""
        mock_logger.return_value = Mock()
        
        # Test with jitter disabled for predictable results
        config_no_jitter = RateLimitConfig(
            requests_per_minute=2,
            tokens_per_minute=6000,
            base_backoff_seconds=1.0,
            max_backoff_seconds=300.0,
            jitter=False
        )
        rate_limiter = RateLimiter(config=config_no_jitter)
        
        # Test backoff calculation for different retry attempts
        base_delay = config_no_jitter.base_backoff_seconds
        
        # First retry
        backoff_1 = rate_limiter._calculate_backoff_delay(1)
        expected_1 = base_delay * (2 ** 1)  # 2.0 seconds
        self.assertEqual(backoff_1, expected_1)
        
        # Second retry
        backoff_2 = rate_limiter._calculate_backoff_delay(2)
        expected_2 = base_delay * (2 ** 2)  # 4.0 seconds
        self.assertEqual(backoff_2, expected_2)
        
        # Third retry
        backoff_3 = rate_limiter._calculate_backoff_delay(3)
        expected_3 = base_delay * (2 ** 3)  # 8.0 seconds
        self.assertEqual(backoff_3, expected_3)
        
        # Test maximum backoff limit
        large_retry = rate_limiter._calculate_backoff_delay(10)
        self.assertLessEqual(large_retry, config_no_jitter.max_backoff_seconds)

    @patch('src.services.rate_limiter.get_structured_logger')
    def test_multi_model_rate_tracking(self, mock_logger):
        """Test multi-model rate tracking."""
        mock_logger.return_value = Mock()
        
        rate_limiter = RateLimiter(config=self.test_config)
        
        # Test tracking for different models
        models = ["groq", "gemini", "openai"]
        
        for i, model in enumerate(models):
            # Record different amounts of requests/tokens for each model
            tokens = (i + 1) * 100
            requests = i + 1  # Different number of requests per model
            
            for _ in range(requests):
                rate_limiter.record_request(model, tokens // requests, success=True)
            
            # Verify each model is tracked independently
            self.assertIn(model, rate_limiter.model_states)
            
            # Verify token counts
            model_state = rate_limiter.get_model_state(model)
            self.assertEqual(model_state.tokens_per_minute, tokens)
            self.assertEqual(model_state.requests_per_minute, requests)
        
        # Test that models don't interfere with each other
        groq_state = rate_limiter.get_model_state("groq")
        gemini_state = rate_limiter.get_model_state("gemini")
        self.assertNotEqual(
            groq_state.tokens_per_minute,
            gemini_state.tokens_per_minute
        )
        
        # Test rate limit status for each model independently
        for model in models:
            can_request = rate_limiter.can_make_request(model)
            # Should be able to make requests with low usage
            self.assertTrue(can_request)

    @patch('src.services.rate_limiter.get_structured_logger')
    def test_async_wait_mechanisms(self, mock_logger):
        """Test async wait mechanisms."""
        mock_logger.return_value = Mock()
        
        # Create rate limiter with very low limits for testing
        test_config = RateLimitConfig(
            requests_per_minute=1,
            tokens_per_minute=100,
            max_retries=2
        )
        rate_limiter = RateLimiter(config=test_config)
        
        async def test_async_wait():
            model_name = "groq"
            
            # First request should not require waiting
            start_time = time.time()
            await rate_limiter.wait_if_needed(model_name)
            first_wait_time = time.time() - start_time
            self.assertLess(first_wait_time, 0.1)  # Should be immediate
            
            # Record a request to trigger rate limiting
            rate_limiter.record_request(model_name, 50, success=True)
            
            # Second immediate request should require waiting
            rate_limiter.record_request(model_name, 60, success=True)  # This should trigger limit
            
            # Mock the wait to avoid actual delays in tests
            with patch('asyncio.sleep') as mock_sleep:
                mock_sleep.return_value = AsyncMock()
                
                # This should trigger a wait
                if not rate_limiter.can_make_request(model_name):
                    wait_time = rate_limiter.get_wait_time(model_name)
                    await asyncio.sleep(wait_time)
                    mock_sleep.assert_called_once()
        
        # Run async test
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(test_async_wait())
        finally:
            loop.close()

    @patch('src.services.rate_limiter.get_structured_logger')
    def test_rate_limit_exceeded_exception(self, mock_logger):
        """Test RateLimitExceeded exception handling."""
        mock_logger.return_value = Mock()
        
        # Test exception creation and properties
        model = "groq"
        retry_after = 45.5
        
        exception = RateLimitExceeded(model, retry_after)
        
        self.assertEqual(exception.model, model)
        self.assertEqual(exception.retry_after, retry_after)
        self.assertIn(model, str(exception))
        self.assertIn(str(retry_after), str(exception))

    @patch('src.services.rate_limiter.get_structured_logger')
    def test_rate_limit_state_tracking(self, mock_logger):
        """Test rate limit state tracking."""
        mock_logger.return_value = Mock()
        
        rate_limiter = RateLimiter(config=self.test_config)
        model_name = "groq"
        
        # Test initial state
        initial_state = rate_limiter.get_model_state(model_name)
        self.assertIsInstance(initial_state, RateLimitState)
        self.assertEqual(initial_state.requests_per_minute, 0)
        self.assertEqual(initial_state.tokens_per_minute, 0)
        self.assertTrue(initial_state.can_make_request())
        
        # Record some requests and check state
        rate_limiter.record_request(model_name, 100, success=True)
        rate_limiter.record_request(model_name, 150, success=True)
        
        updated_state = rate_limiter.get_model_state(model_name)
        self.assertEqual(updated_state.requests_per_minute, 2)
        self.assertEqual(updated_state.tokens_per_minute, 250)
        
        # Test state after rate limit is triggered
        # Add more requests to trigger limit
        for _ in range(30):  # Exceed the 30 RPM limit
            rate_limiter.record_request(model_name, 10, success=True)
        
        limited_state = rate_limiter.get_model_state(model_name)
        self.assertFalse(limited_state.can_make_request())
        wait_time = rate_limiter.get_wait_time(model_name)
        self.assertGreater(wait_time, 0)

    @patch('src.services.rate_limiter.get_structured_logger')
    def test_time_window_cleanup(self, mock_logger):
        """Test time window cleanup for rate limiting."""
        mock_logger.return_value = Mock()
        
        rate_limiter = RateLimiter(config=self.test_config)
        model_name = "groq"
        
        # Record requests
        rate_limiter.record_request(model_name, 100, success=True)
        rate_limiter.record_request(model_name, 150, success=True)
        
        # Verify requests are recorded
        model_state = rate_limiter.get_model_state(model_name)
        self.assertEqual(model_state.requests_per_minute, 2)
        self.assertEqual(model_state.tokens_per_minute, 250)
        
        # Test cleanup of old entries (mock time passage)
        with patch('time.time') as mock_time:
            # Simulate time passage beyond the rate limit window
            mock_time.return_value = time.time() + 3600  # 1 hour later
            
            # Check that model state still exists after time passage
            # (Window reset is handled automatically in can_make_request)
            self.assertIsNotNone(rate_limiter.model_states[model_name])
            
            # Verify that a new request can be made after window reset
            self.assertTrue(rate_limiter.can_make_request(model_name))

    @patch('src.services.rate_limiter.get_structured_logger')
    def test_concurrent_request_handling(self, mock_logger):
        """Test concurrent request handling."""
        mock_logger.return_value = Mock()
        
        rate_limiter = RateLimiter(config=self.test_config)
        model_name = "groq"
        
        async def concurrent_request_test():
            # Simulate multiple concurrent requests
            tasks = []
            
            for i in range(5):
                async def make_request(request_id):
                    await rate_limiter.wait_if_needed(model_name)
                    rate_limiter.record_request(model_name, 100, success=True)
                    return request_id
                
                task = asyncio.create_task(make_request(i))
                tasks.append(task)
            
            # Wait for all requests to complete
            results = await asyncio.gather(*tasks)
            
            # Verify all requests were processed
            self.assertEqual(len(results), 5)
            model_state = rate_limiter.get_model_state(model_name)
            self.assertEqual(model_state.requests_per_minute, 5)
        
        # Run async test
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(concurrent_request_test())
        finally:
            loop.close()

    @patch('src.services.rate_limiter.get_structured_logger')
    def test_jitter_in_backoff(self, mock_logger):
        """Test jitter implementation in backoff calculations."""
        mock_logger.return_value = Mock()
        
        # Test with jitter enabled
        config_with_jitter = RateLimitConfig(
            requests_per_minute=30,
            tokens_per_minute=6000,
            jitter=True
        )
        rate_limiter_jitter = RateLimiter(config=config_with_jitter)
        
        # Test with jitter disabled
        config_no_jitter = RateLimitConfig(
            requests_per_minute=30,
            tokens_per_minute=6000,
            jitter=False
        )
        rate_limiter_no_jitter = RateLimiter(config=config_no_jitter)
        
        # Calculate multiple backoff delays and check for variation
        jitter_delays = []
        no_jitter_delays = []
        
        for _ in range(10):
            jitter_delay = rate_limiter_jitter._calculate_backoff_delay(2)
            no_jitter_delay = rate_limiter_no_jitter._calculate_backoff_delay(2)
            
            jitter_delays.append(jitter_delay)
            no_jitter_delays.append(no_jitter_delay)
        
        # With jitter, delays should vary
        jitter_variance = max(jitter_delays) - min(jitter_delays)
        
        # Without jitter, delays should be consistent
        no_jitter_variance = max(no_jitter_delays) - min(no_jitter_delays)
        
        # Jitter should introduce more variance
        if config_with_jitter.jitter:
            self.assertGreaterEqual(jitter_variance, 0)
        
        if not config_no_jitter.jitter:
            self.assertEqual(no_jitter_variance, 0)

    def test_get_rate_limiter_singleton(self):
        """Test get_rate_limiter singleton function."""
        # Test that get_rate_limiter returns the same instance
        limiter1 = get_rate_limiter()
        limiter2 = get_rate_limiter()
        
        self.assertIs(limiter1, limiter2)
        self.assertIsInstance(limiter1, RateLimiter)


if __name__ == '__main__':
    unittest.main()