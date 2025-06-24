"""Rate limiting service for LLM API calls.

This module provides rate limiting functionality to prevent hitting API limits
and implements retry logic with exponential backoff.
"""

import asyncio
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Callable, Any, Awaitable
from dataclasses import dataclass, field
from collections import defaultdict

# Tenacity imports for retry logic
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from ..config.logging_config import get_structured_logger
from ..config.settings import LLMConfig
from ..models.data_models import RateLimitState, RateLimitLog
from ..utils.exceptions import RateLimitError, NetworkError


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""

    requests_per_minute: int = 30
    tokens_per_minute: int = 50000
    max_retries: int = 3
    base_backoff_seconds: float = 1.0
    max_backoff_seconds: float = 300.0
    jitter: bool = True


class RateLimiter:
    """Rate limiter for LLM API calls with retry logic."""

    def __init__(self, config: Optional[RateLimitConfig] = None):
        self.config = config or RateLimitConfig()
        self.model_states: Dict[str, RateLimitState] = {}
        self.logger = get_structured_logger("rate_limiter")
        self._locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    def get_model_state(self, model: str) -> RateLimitState:
        """Get or create rate limit state for a model."""
        if model not in self.model_states:
            self.model_states[model] = RateLimitState(
                model=model,
                requests_made=0,
                requests_limit=self.config.requests_per_minute,
            )
        return self.model_states[model]

    def can_make_request(self, model: str, estimated_tokens: int = 0) -> bool:
        """Check if a request can be made for the given model."""
        state = self.get_model_state(model)
        return state.can_make_request(estimated_tokens)

    def get_wait_time(self, model: str) -> float:
        """Get the time to wait before making a request."""
        state = self.get_model_state(model)
        now = datetime.now()

        # Check backoff period
        if state.backoff_until and now < state.backoff_until:
            return (state.backoff_until - now).total_seconds()

        # Check rate limit window
        window_elapsed = (now - state.window_start).total_seconds()
        if window_elapsed < 60:
            # Check if we're at the limit
            if (
                state.requests_per_minute >= self.config.requests_per_minute
                or state.tokens_per_minute >= self.config.tokens_per_minute
            ):
                return 60 - window_elapsed

        return 0.0

    def wait_if_needed(self, model: str, estimated_tokens: int = 0):
        """Wait if rate limit requires it (synchronous version)."""
        wait_time = self.get_wait_time(model)
        if wait_time > 0:
            self.logger.info(
                f"Rate limit wait required for model {model}",
                wait_time=wait_time,
                estimated_tokens=estimated_tokens,
            )

            # Log rate limit event
            state = self.get_model_state(model)
            rate_log = RateLimitLog(
                timestamp=datetime.now().isoformat(),
                model=model,
                requests_in_window=state.requests_per_minute,
                tokens_in_window=state.tokens_per_minute,
                window_start=state.window_start.isoformat(),
                window_end=(state.window_start + timedelta(minutes=1)).isoformat(),
                limit_exceeded=True,
                wait_time_seconds=wait_time,
            )
            self.logger.log_rate_limit(rate_log)

            time.sleep(wait_time)

    async def wait_if_needed_async(self, model: str, estimated_tokens: int = 0):
        """Wait if rate limit requires it (async version)."""
        wait_time = self.get_wait_time(model)
        if wait_time > 0:
            self.logger.info(
                f"Rate limit wait required for model {model}",
                wait_time=wait_time,
                estimated_tokens=estimated_tokens,
            )

            # Log rate limit event
            state = self.get_model_state(model)
            rate_log = RateLimitLog(
                timestamp=datetime.now().isoformat(),
                model=model,
                requests_in_window=state.requests_per_minute,
                tokens_in_window=state.tokens_per_minute,
                window_start=state.window_start.isoformat(),
                window_end=(state.window_start + timedelta(minutes=1)).isoformat(),
                limit_exceeded=True,
                wait_time_seconds=wait_time,
            )
            self.logger.log_rate_limit(rate_log)

            await asyncio.sleep(wait_time)

    def record_request(self, model: str, tokens_used: int, success: bool):
        """Record a request and update rate limit state."""
        state = self.get_model_state(model)
        state.record_request(tokens_used, success)

        # Log rate limit state
        rate_log = RateLimitLog(
            timestamp=datetime.now().isoformat(),
            model=model,
            requests_in_window=state.requests_per_minute,
            tokens_in_window=state.tokens_per_minute,
            window_start=state.window_start.isoformat(),
            window_end=(state.window_start + timedelta(minutes=1)).isoformat(),
            limit_exceeded=False,
        )
        self.logger.log_rate_limit(rate_log)

    def execute_with_rate_limit(
        self, func, model: str, estimated_tokens: int = 100, *args, **kwargs
    ):
        """Execute a function with rate limiting.

        Note: Retry logic has been moved to EnhancedLLMService for centralized handling.

        Args:
            func: Function to execute
            model: Model name for rate limiting
            estimated_tokens: Estimated token usage
            *args, **kwargs: Arguments to pass to the function

        Returns:
            Function result

        Raises:
            RateLimitError: If rate limit is exceeded
            NetworkError: If network error occurs
        """
        with self._locks[model]:
            self.wait_if_needed(model, estimated_tokens)

            start_time = time.time()
            try:
                result = func(*args, **kwargs)

                # Extract actual token usage if available
                actual_tokens = estimated_tokens
                if hasattr(result, "usage") and hasattr(result.usage, "total_tokens"):
                    actual_tokens = result.usage.total_tokens
                elif isinstance(result, dict) and "usage" in result:
                    actual_tokens = result["usage"].get(
                        "total_tokens", estimated_tokens
                    )

                self.record_request(model, actual_tokens, success=True)
                return result

            except Exception as e:
                # Record failed request
                self.record_request(model, estimated_tokens, success=False)

                # Check if it's a rate limit error
                if self._is_rate_limit_error(e):
                    retry_after = self._extract_retry_after(e)
                    raise RateLimitError(
                        f"Rate limit exceeded for model {model}. Retry after {retry_after} seconds."
                    )

                raise NetworkError(f"API call failed: {str(e)}") from e

    def _is_rate_limit_error(self, error: Exception) -> bool:
        """Check if an error is a rate limit error using centralized utility."""
        from ..utils.error_classification import is_rate_limit_error

        return is_rate_limit_error(error)

    def _extract_retry_after(self, error: Exception) -> float:
        """Extract retry-after time from error, or use default backoff."""
        # Try to extract from error message or headers
        # This is API-specific and would need to be customized
        error_str = str(error)

        # Look for retry-after in seconds
        import re

        retry_match = re.search(r"retry.after[:\s]+(\d+)", error_str, re.IGNORECASE)
        if retry_match:
            return float(retry_match.group(1))

        # Default exponential backoff
        state = self.get_model_state("unknown")
        return min(
            self.config.max_backoff_seconds,
            self.config.base_backoff_seconds * (2**state.consecutive_failures),
        )

    def _calculate_backoff_delay(self, retry_attempt: int) -> float:
        """Calculate backoff delay with optional jitter."""
        import random

        # Calculate base exponential backoff
        base_delay = self.config.base_backoff_seconds * (2**retry_attempt)
        delay = min(base_delay, self.config.max_backoff_seconds)

        # Add jitter if enabled (but keep within bounds)
        if self.config.jitter:
            # Add random jitter between 0% and 10% of the delay
            # But ensure we never exceed the maximum allowed
            max_jitter = min(delay * 0.1, self.config.max_backoff_seconds - delay)
            if max_jitter > 0:
                jitter_amount = max_jitter * random.random()
                delay += jitter_amount

        # Ensure we never exceed the maximum
        return min(delay, self.config.max_backoff_seconds)


class RetryableRateLimiter(RateLimiter):
    """Rate limiter with built-in retry logic using tenacity."""

    def __init__(self, config: Optional[RateLimitConfig] = None):
        super().__init__(config)
        self.retry_decorator = retry(
            stop=stop_after_attempt(self.config.max_retries),
            wait=wait_exponential(
                multiplier=self.config.base_backoff_seconds,
                max=self.config.max_backoff_seconds,
            ),
            retry=retry_if_exception_type((RateLimitError, NetworkError)),
            before_sleep=before_sleep_log(self.logger.logger, logging.WARNING),
        )

    async def execute_with_retry(
        self,
        func: Callable[..., Awaitable[Any]],
        model: str,
        estimated_tokens: int = 0,
        *args,
        **kwargs,
    ) -> Any:
        """Execute a function with rate limiting and automatic retries."""

        @self.retry_decorator
        async def _execute():
            return await self.execute_with_rate_limit(
                func, model, estimated_tokens, *args, **kwargs
            )

        return await _execute()


# Global rate limiter instance
_global_rate_limiter: Optional[RetryableRateLimiter] = None


def get_rate_limiter(config: Optional[RateLimitConfig] = None) -> RetryableRateLimiter:
    """Get the global rate limiter instance."""
    global _global_rate_limiter
    if _global_rate_limiter is None:
        _global_rate_limiter = RetryableRateLimiter(config)
    return _global_rate_limiter


def reset_rate_limiter():
    """Reset the global rate limiter (useful for testing)."""
    global _global_rate_limiter
    _global_rate_limiter = None


# Decorator for easy rate limiting
def rate_limited(model: str, estimated_tokens: int = 0):
    """Decorator to add rate limiting to async functions."""

    def decorator(func: Callable[..., Awaitable[Any]]):
        async def wrapper(*args, **kwargs):
            rate_limiter = get_rate_limiter()
            return await rate_limiter.execute_with_retry(
                func, model, estimated_tokens, *args, **kwargs
            )

        return wrapper

    return decorator


# Utility functions for common rate limiting scenarios
async def rate_limited_llm_call(
    llm_func: Callable[..., Awaitable[Any]],
    model: str,
    prompt: str,
    estimated_tokens: Optional[int] = None,
    **kwargs,
) -> Any:
    """Make a rate-limited LLM call with automatic token estimation."""
    if estimated_tokens is None:
        # Rough estimation: 1 token ≈ 4 characters
        estimated_tokens = len(prompt) // 4 + 500  # Add buffer for response

    rate_limiter = get_rate_limiter()
    return await rate_limiter.execute_with_retry(
        llm_func, model, estimated_tokens, prompt=prompt, **kwargs
    )


def get_rate_limit_status(model: str) -> Dict[str, Any]:
    """Get current rate limit status for a model."""
    rate_limiter = get_rate_limiter()
    state = rate_limiter.get_model_state(model)

    now = datetime.now()
    window_elapsed = (now - state.window_start).total_seconds()

    return {
        "model": model,
        "requests_in_window": state.requests_per_minute,
        "tokens_in_window": state.tokens_per_minute,
        "window_elapsed_seconds": window_elapsed,
        "can_make_request": state.can_make_request(),
        "wait_time_seconds": rate_limiter.get_wait_time(model),
        "consecutive_failures": state.consecutive_failures,
        "backoff_until": (
            state.backoff_until.isoformat() if state.backoff_until else None
        ),
        "last_request_time": state.last_request_time.isoformat(),
    }
