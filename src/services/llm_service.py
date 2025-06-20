print("DEBUG: llm_service.py module is being loaded")

import os
import time
import asyncio
import concurrent.futures
import threading
import hashlib
import functools
import json
import pickle
import traceback
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import OrderedDict

try:
    import google.generativeai as genai
except ImportError:
    genai = None

# Import tenacity for retry logic with exponential backoff
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from ..config.logging_config import get_structured_logger
from ..config.settings import get_config
from ..models.data_models import ContentType
from .rate_limiter import RateLimiter as EnhancedRateLimiter
from .error_recovery import get_error_recovery_service
from ..core.performance_optimizer import get_performance_optimizer
from ..core.async_optimizer import get_async_optimizer
from ..utils.exceptions import ConfigurationError, OperationTimeoutError

# is_transient_error function removed - using _should_retry_exception instead

logger = get_structured_logger("llm_service")

# Define retryable exceptions for LLM API calls
# These are common exceptions that indicate transient failures
RETRYABLE_EXCEPTIONS = (
    # Network and connection errors
    ConnectionError,
    OperationTimeoutError,
    OSError,  # Covers network-related OS errors
    # HTTP and API errors that are typically transient
    Exception,  # Temporary catch-all until we identify specific google.generativeai exceptions
)

# Non-retryable exceptions that indicate permanent failures
NON_RETRYABLE_EXCEPTIONS = (
    ValueError,  # Invalid input parameters
    TypeError,  # Type errors in our code
    KeyError,  # Missing configuration keys
    AttributeError,  # Missing attributes/methods
)


class AdvancedCache:
    """Advanced caching system with LRU eviction, TTL, and persistence."""

    def __init__(
        self,
        max_size: int = 1000,
        default_ttl_hours: int = 24,
        persist_file: Optional[str] = None,
    ):
        self.max_size = max_size
        self.default_ttl_hours = default_ttl_hours
        self.persist_file = persist_file
        self._cache: OrderedDict = OrderedDict()
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

        # Load persisted cache if available
        if persist_file:
            self._load_cache()

        logger.info(
            "Advanced cache initialized",
            max_size=max_size,
            default_ttl_hours=default_ttl_hours,
            persist_file=persist_file,
        )

    def _generate_cache_key(
        self, prompt: str, model: str, temperature: float = 0.7, **kwargs
    ) -> str:
        """Generate a comprehensive cache key."""
        key_data = {
            "prompt": prompt,
            "model": model,
            "temperature": temperature,
            **kwargs,
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_string.encode()).hexdigest()

    def _is_expired(self, entry: Dict[str, Any]) -> bool:
        """Check if cache entry is expired."""
        return datetime.now() > entry["expiry"]

    def _evict_expired(self):
        """Remove expired entries."""
        expired_keys = [
            key for key, entry in self._cache.items() if self._is_expired(entry)
        ]
        for key in expired_keys:
            del self._cache[key]
            self._evictions += 1

    def _evict_lru(self):
        """Evict least recently used entries if cache is full."""
        while len(self._cache) >= self.max_size:
            self._cache.popitem(last=False)  # Remove oldest (LRU)
            self._evictions += 1

    def get(self, prompt: str, model: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Get cached response."""
        cache_key = self._generate_cache_key(prompt, model, **kwargs)

        with self._lock:
            self._evict_expired()

            if cache_key in self._cache:
                entry = self._cache[cache_key]
                if not self._is_expired(entry):
                    # Move to end (mark as recently used)
                    self._cache.move_to_end(cache_key)
                    self._hits += 1
                    return entry["response"]
                else:
                    del self._cache[cache_key]
                    self._evictions += 1

            self._misses += 1
            return None

    def set(
        self,
        prompt: str,
        model: str,
        response: Dict[str, Any],
        ttl_hours: Optional[int] = None,
        **kwargs,
    ):
        """Cache response with TTL."""
        cache_key = self._generate_cache_key(prompt, model, **kwargs)
        ttl = ttl_hours or self.default_ttl_hours

        with self._lock:
            self._evict_expired()
            self._evict_lru()

            entry = {
                "response": response,
                "expiry": datetime.now() + timedelta(hours=ttl),
                "created_at": datetime.now(),
                "access_count": 1,
                "cache_key": cache_key,
            }

            self._cache[cache_key] = entry

            # Persist cache if configured
            if self.persist_file:
                self._save_cache()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0

            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate_percent": hit_rate,
                "evictions": self._evictions,
                "memory_usage_estimate_mb": self._estimate_memory_usage(),
            }

    def _estimate_memory_usage(self) -> float:
        """Estimate cache memory usage in MB."""
        try:
            # Rough estimation based on cache size
            return len(self._cache) * 0.1  # Assume ~100KB per entry
        except Exception:
            return 0.0

    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
            self._evictions = 0

            if self.persist_file:
                self._save_cache()

    def _save_cache(self):
        """Persist cache to file."""
        try:
            if self.persist_file:
                cache_data = {
                    "cache": dict(self._cache),
                    "stats": {
                        "hits": self._hits,
                        "misses": self._misses,
                        "evictions": self._evictions,
                    },
                    "saved_at": datetime.now().isoformat(),
                }

                with open(self.persist_file, "wb") as f:
                    pickle.dump(cache_data, f)
        except Exception as e:
            logger.warning("Failed to save cache", error=str(e))

    def _load_cache(self):
        """Load persisted cache from file."""
        try:
            if self.persist_file and os.path.exists(self.persist_file):
                with open(self.persist_file, "rb") as f:
                    cache_data = pickle.load(f)

                # Restore cache entries that haven't expired
                for key, entry in cache_data.get("cache", {}).items():
                    if not self._is_expired(entry):
                        self._cache[key] = entry

                # Restore stats
                stats = cache_data.get("stats", {})
                self._hits = stats.get("hits", 0)
                self._misses = stats.get("misses", 0)
                self._evictions = stats.get("evictions", 0)

                logger.info(
                    "Cache loaded from persistence",
                    entries_loaded=len(self._cache),
                    file=self.persist_file,
                )
        except Exception as e:
            logger.warning("Failed to load persisted cache", error=str(e))


# Global advanced cache instance
_ADVANCED_CACHE = None


def get_advanced_cache() -> AdvancedCache:
    """Get global advanced cache instance."""
    global _ADVANCED_CACHE
    if _ADVANCED_CACHE is None:
        settings = get_config()
        cache_file = (
            os.path.join(settings.data_dir, "llm_cache.pkl")
            if hasattr(settings, "data_dir")
            else None
        )
        _ADVANCED_CACHE = AdvancedCache(persist_file=cache_file)
    return _ADVANCED_CACHE


# --- Response Caching Mechanism ---
# Use LRU (Least Recently Used) cache for LLM responses.
# maxsize=128 means it will store up to 128 recent unique LLM calls.
# This is highly effective for regeneration requests where the prompt is identical.
def create_cache_key(prompt: str, model_name: str, content_type: str = "") -> str:
    """Creates a consistent hashable key for caching based on the prompt."""
    # Use a hash of the prompt to keep the key length manageable
    prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
    return f"{model_name}:{content_type}:{prompt_hash}"


# Global cache for LLM responses
@functools.lru_cache(maxsize=128)
# Legacy cache functions removed - now using AdvancedCache only


@dataclass
class LLMResponse:
    """Structured response from LLM calls."""

    content: str
    tokens_used: int = 0
    processing_time: float = 0.0
    model_used: str = ""
    success: bool = True
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class EnhancedLLMService:  # pylint: disable=too-many-instance-attributes
    """
    Enhanced LLM service with Phase 1 infrastructure integration.
    Provides structured logging, rate limiting, and error recovery.
    """

    def __init__(
        self,
        timeout: int = 30,
        rate_limiter: Optional[EnhancedRateLimiter] = None,
        error_recovery=None,
        user_api_key: Optional[str] = None,
    ):
        """
        Initialize the enhanced LLM service.

        Args:
            timeout: Maximum time in seconds to wait for LLM response
            rate_limiter: Optional rate limiter instance
            error_recovery: Optional error recovery service
            user_api_key: Optional user-provided API key (takes priority)
        """
        self.settings = get_config()
        self.timeout = timeout

        # Check if google-generativeai is available
        if genai is None:
            raise ImportError(
                "google-generativeai package is not installed. "
                "Please install it with: pip install google-generativeai"
            )

        # Simplified API key management with clear priority and fallback
        self.active_api_key = self._determine_active_api_key(user_api_key)
        self.fallback_api_key = self.settings.llm.gemini_api_key_fallback
        self.using_fallback = False
        self.using_user_key = bool(user_api_key)

        # Initialize the model with active API key
        try:
            genai.configure(api_key=self.active_api_key)
            self.model_name = self.settings.llm_settings.default_model
            self.llm = genai.GenerativeModel(self.model_name)
        except Exception as e:
            raise ValueError(f"Failed to initialize Gemini model: {str(e)}") from e

        # Enhanced services
        self.rate_limiter = rate_limiter or EnhancedRateLimiter()
        self.error_recovery = error_recovery or get_error_recovery_service()

        # Initialize optimizers
        self.performance_optimizer = get_performance_optimizer()
        self.async_optimizer = get_async_optimizer()

        # Initialize advanced caching
        self.cache = get_advanced_cache()

        # Performance tracking
        self.call_count = 0
        self.total_tokens = 0
        self.total_processing_time = 0.0

        # Cache performance tracking
        self.cache_hits = 0
        self.cache_misses = 0

        # Connection pooling for better performance
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=5, thread_name_prefix="llm_worker"
        )

        logger.info(
            "Enhanced LLM service initialized",
            model=self.model_name,
            timeout=timeout,
            using_user_key=self.using_user_key,
            using_fallback_key=self.using_fallback,
            has_fallback_key=bool(self.fallback_api_key),
            cache_enabled=True,
        )

    def _determine_active_api_key(self, user_api_key: Optional[str]) -> str:
        """
        Determine the active API key based on priority: user > primary > fallback.
        
        Args:
            user_api_key: Optional user-provided API key
            
        Returns:
            str: The API key to use
            
        Raises:
            ConfigurationError: If no valid API key is found
        """
        # Priority order: user-provided > primary > fallback
        if user_api_key:
            return user_api_key
        if self.settings.llm.gemini_api_key_primary:
            return self.settings.llm.gemini_api_key_primary
        if self.settings.llm.gemini_api_key_fallback:
            return self.settings.llm.gemini_api_key_fallback
        raise ConfigurationError(
            "CRITICAL: Gemini API key is not configured. "
            "Please set the GEMINI_API_KEY in your .env file or provide it in the UI. "
            "Application cannot start without a valid API key."
        )

    async def validate_api_key(self) -> bool:
        """
        Validate the current API key by making a lightweight API call.
        
        This method performs a simple, low-cost API call (listing models)
        to verify that the API key is valid and the service is accessible.
        
        Returns:
            bool: True if the API key is valid, False otherwise.
        """
        try:
            # Use the current event loop to run the synchronous API call
            loop = asyncio.get_event_loop()
            
            # Make a lightweight API call to validate the key
            # genai.list_models() is a simple call that requires authentication
            models = await loop.run_in_executor(
                None, 
                lambda: list(genai.list_models())
            )
            
            # If we get here without exception, the key is valid
            logger.info(
                "API key validation successful",
                extra={
                    "models_count": len(models),
                    "using_user_key": self.using_user_key,
                    "using_fallback": self.using_fallback
                }
            )
            return True
            
        except (ValueError, ConnectionError, OperationTimeoutError) as e:
            # Log the validation failure
            logger.warning(
                "API key validation failed",
                extra={
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "using_user_key": self.using_user_key,
                    "using_fallback": self.using_fallback
                }
            )
            return False

    def _switch_to_fallback_key(self) -> bool:
        """
        Switch to fallback API key when rate limits are encountered.
        
        Returns:
            bool: True if successfully switched to fallback, False otherwise
        """
        if not self.using_fallback and self.fallback_api_key:
            logger.warning(
                "Switching to fallback API key due to rate limit or error",
                current_key_type="primary" if not self.using_user_key else "user",
                fallback_available=True,
            )

            try:
                # Reconfigure with fallback key
                genai.configure(api_key=self.fallback_api_key)
                self.active_api_key = self.fallback_api_key
                self.using_fallback = True

                # Reinitialize the model with new key
                self.llm = genai.GenerativeModel(self.model_name)
            except (ValueError, ConnectionError) as e:
                logger.error(
                    "Failed to switch to fallback API key",
                    error=str(e),
                    error_type=type(e).__name__
                )
                return False

            logger.info("Successfully switched to fallback API key")
            return True
        else:
            logger.error(
                "Cannot switch to fallback key",
                already_using_fallback=self.using_fallback,
                fallback_available=bool(self.fallback_api_key),
            )
            return False

    def get_current_api_key_info(self) -> Dict[str, Any]:
        """
        Get information about the currently active API key.
        
        Returns:
            Dict containing API key status information
        """
        return {
            "using_user_key": self.using_user_key,
            "using_fallback": self.using_fallback,
            "has_fallback_available": bool(self.fallback_api_key),
            "key_source": (
                "user" if self.using_user_key 
                else "fallback" if self.using_fallback 
                else "primary"
            )
        }

    def _is_retryable_error(self, exception: Exception, retry_count: int = 0, max_retries: int = 5) -> Tuple[bool, float]:
        """
        Centralized retry logic with intelligent delay calculation and error-type-specific strategies.
        Returns (should_retry: bool, delay_seconds: float)
        """
        # Don't retry non-retryable exceptions
        if isinstance(exception, NON_RETRYABLE_EXCEPTIONS):
            logger.debug(
                f"Non-retryable exception detected: {type(exception).__name__}"
            )
            return False, 0.0

        # Check for specific error messages that indicate non-retryable failures
        error_msg = str(exception).lower()
        non_retryable_patterns = [
            "invalid api key",
            "api key not found",
            "authentication failed",
            "permission denied",
            "invalid request",
            "malformed request",
        ]
        if any(pattern in error_msg for pattern in non_retryable_patterns):
            logger.debug(f"Non-retryable error pattern detected: {error_msg}")
            return False, 0.0

        # Check if we've exceeded max retries
        if retry_count >= max_retries:
            return False, 0.0

        # Retry for retryable exceptions
        if isinstance(exception, RETRYABLE_EXCEPTIONS):
            # Rate limit errors - use longer delays
            if any(keyword in error_msg for keyword in ["rate limit", "quota", "429", "too many requests", "resource_exhausted"]):
                base_delay = (2 ** retry_count) * 5
                jitter = base_delay * 0.1 * (0.5 - abs(hash(str(exception)) % 100) / 100)
                delay = min(300, base_delay + jitter)
                return True, delay
            # Network/timeout errors - moderate delays
            elif any(keyword in error_msg for keyword in ["connection", "network", "timeout", "dns", "unreachable"]):
                delay = min(60, (retry_count + 1) * 2)
                return True, delay
            # API errors - short delays
            elif any(keyword in error_msg for keyword in ["api error", "server error", "500", "502", "503", "504"]):
                delay = min(30, 2 ** retry_count)
                return True, delay
            # Generic retryable errors - minimal delay
            else:
                delay = min(10, retry_count * 1)
                return True, delay
        # Default to not retrying unknown exceptions
        logger.debug(
            f"Unknown exception type, not retrying: {type(exception).__name__}"
        )
        return False, 0.0

    async def _generate_with_timeout(self, prompt: str, session_id: str = None, trace_id: Optional[str] = None) -> Any:
        """
        Generate content with timeout and consolidated retry logic.
        
        This method now handles both timeout and executor management internally,
        consolidating the async execution pattern.

        Args:
            prompt: Text prompt to send to the model
            session_id: Optional session ID for tracking
            trace_id: Optional trace ID for tracking

        Returns:
            Generated text response or raises TimeoutError
        """
        try:
            # Get the current event loop
            loop = asyncio.get_event_loop()
            
            # Create executor task and wrap with timeout
            executor_task = loop.run_in_executor(
                self.executor, 
                self._make_llm_api_call, 
                prompt
            )
            
            # Use asyncio.wait_for to handle timeout
            result = await asyncio.wait_for(
                executor_task, 
                timeout=self.timeout
            )

            # Log successful call
            logger.info(
                "LLM call completed successfully",
                session_id=session_id,
                model=self.model_name,
                prompt_length=len(prompt),
                response_length=len(result.text) if hasattr(result, "text") else 0,
            )

            return result

        except asyncio.TimeoutError:
            # Log timeout
            logger.error(
                "LLM request timed out",
                extra={
                    "trace_id": trace_id,
                    "session_id": session_id,
                    "timeout": self.timeout,
                    "prompt_length": len(prompt),
                },
            )

            raise OperationTimeoutError(
                "LLM request timed out after {} seconds".format(self.timeout)
            )
        except Exception as e:
            # Log other exceptions
            logger.error(
                "LLM request failed with exception",
                extra={
                    "trace_id": trace_id,
                    "session_id": session_id,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "prompt_length": len(prompt),
                },
            )
            # Re-raise the exception to be handled by the caller
            raise

    def _make_llm_api_call(self, prompt: str) -> Any:
        """
        Make the actual LLM API call.

        Args:
            prompt: Text prompt to send to the model

        Returns:
            Generated response from the LLM

        Raises:
            Various exceptions from the google-generativeai library
        """
        try:
            # Ensure LLM model is properly initialized
            if self.llm is None:
                raise ValueError("LLM model is not initialized. Service initialization may have failed.")
            
            response = self.llm.generate_content(prompt)

            # Clean up response text to handle encoding issues
            # Note: We don't modify the response object directly as it's read-only
            # The text cleaning will be handled in the calling method if needed

            logger.debug("LLM API call successful")
            return response
        except Exception as e:
            # Log the specific error for debugging
            logger.error(
                "LLM API call failed",
                error_type=type(e).__name__,
                error_message=str(e),
                prompt_length=len(prompt),
            )
            # Re-raise to let tenacity handle the retry logic
            raise

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        retry=lambda retry_state: (
            hasattr(retry_state.fn, '__self__') and
            hasattr(retry_state.fn.__self__, '_is_retryable_error') and
            retry_state.fn.__self__._is_retryable_error(
                retry_state.outcome.exception(),
                retry_state.attempt_number - 1,
                5
            )[0]
            if retry_state.outcome and retry_state.outcome.failed else False
        ),
        reraise=True
    )
    async def generate_content(
        self,
        prompt: str,
        content_type: ContentType = ContentType.QUALIFICATION,
        session_id: str = None,
        item_id: str = None,
        max_retries: int = 5,
        trace_id: Optional[str] = None,
    ) -> LLMResponse:

        """
        Generate content using the Gemini model with enhanced error handling and caching.

        Args:
            prompt: Text prompt to send to the model
            content_type: Type of content being generated
            session_id: Session ID for tracking
            item_id: Item ID for tracking
            max_retries: Maximum number of retries

        Returns:
            LLMResponse with generated content and metadata
        """

        start_time = time.time()

        # Create cache key for this request
        cache_key = create_cache_key(prompt, self.model_name, content_type.value)

        # Check multi-level cache first
        # Try performance optimizer cache first
        cached_response = self.performance_optimizer.cache.get(cache_key)
        if cached_response:
            self.cache_hits += 1
            logger.info(
                "Performance cache hit for LLM request",
                extra={
                    "trace_id": trace_id,
                    "session_id": session_id,
                    "item_id": item_id,
                    "content_type": content_type.value,
                    "cache_key": cache_key[:20] + "...",
                },
            )

            # Return cached response with updated metadata
            cached_response["metadata"]["cache_hit"] = True
            cached_response["metadata"]["session_id"] = session_id
            cached_response["metadata"]["item_id"] = item_id
            cached_response["processing_time"] = 0.001  # Minimal cache lookup time

            return LLMResponse(**cached_response)

        # Legacy local cache removed - using AdvancedCache only

        # Cache miss - proceed with LLM call
        self.cache_misses += 1
        logger.debug(
            "Cache miss for LLM request",
            extra={
                "trace_id": trace_id,
                "session_id": session_id,
                "item_id": item_id,
                "content_type": content_type.value,
            },
        )

        # Update call tracking
        self.call_count += 1

        try:
            # Apply rate limiting with centralized logic
            await self.rate_limiter.wait_if_needed_async(self.model_name)

            # Log the attempt
            logger.info(
                "Starting LLM generation",
                extra={
                    "trace_id": trace_id,
                    "session_id": session_id,
                    "item_id": item_id,
                    "content_type": content_type.value,
                    "prompt_length": len(prompt),
                    "max_retries": max_retries,
                },
            )

            # Generate content with timeout (temporarily removing optimization context)
            # Use run_in_executor to run the synchronous method in a thread pool
            try:
                response = await self._generate_with_timeout(prompt, session_id, trace_id)
            except Exception as timeout_error:
                logger.error(
                    "Error in _generate_with_timeout",
                    error_type=type(timeout_error).__name__,
                    error_message=str(timeout_error),
                    traceback=traceback.format_exc()
                )
                raise timeout_error

            processing_time = time.time() - start_time
            self.total_processing_time += processing_time

            if not hasattr(response, "text") or response.text is None:
                raise ValueError("LLM returned an empty or invalid response")

            # Estimate token usage (rough approximation)
            tokens_used = len(prompt.split()) + len(response.text.split())
            self.total_tokens += tokens_used

            # Create structured response
            llm_response = LLMResponse(
                content=response.text,
                tokens_used=tokens_used,
                processing_time=processing_time,
                model_used=self.model_name,
                success=True,
                metadata={
                    "session_id": session_id,
                    "item_id": item_id,
                    "content_type": content_type.value,
                    "timestamp": datetime.now().isoformat(),
                    "cache_hit": False,
                },
            )

            # Cache the successful response in both caches
            cache_data = {
                "content": response.text,
                "tokens_used": tokens_used,
                "processing_time": processing_time,
                "model_used": self.model_name,
                "success": True,
                "error_message": None,
                "metadata": {
                    "content_type": content_type.value,
                    "timestamp": datetime.now().isoformat(),
                    "cache_hit": False,
                },
            }
            # Store only in AdvancedCache
            self.performance_optimizer.cache.set(cache_key, cache_data, ttl_hours=2)

            # Log successful generation
            logger.info(
                "LLM generation completed successfully",
                session_id=session_id,
                item_id=item_id,
                processing_time=processing_time,
                tokens_used=tokens_used,
                response_length=len(response.text),
                cached=True,
            )

            return llm_response

        except ConfigurationError:
            # Do not retry on fatal config errors. Re-raise immediately.
            raise
        except Exception as e:
            processing_time = time.time() - start_time
            
            # Check if this is a rate limit error and try fallback key
            error_str = str(e).lower()
            is_rate_limit_error = any(
                keyword in error_str
                for keyword in [
                    "rate limit",
                    "quota",
                    "429",
                    "too many requests",
                    "resource_exhausted",
                ]
            )

            if is_rate_limit_error and not self.using_fallback:
                logger.warning(
                    "Rate limit detected, attempting to switch to fallback API key",
                    error=str(e),
                )

                if self._switch_to_fallback_key():
                    # Re-raise to let tenacity handle the retry
                    raise

            # Use error recovery service if available
            if self.error_recovery:
                try:
                    fallback_content = await self.error_recovery.get_fallback_content(
                        content_type, str(e)
                    )
                    if fallback_content:
                        logger.info(
                            "Using fallback content from error recovery service",
                            content_type=content_type.value,
                            error=str(e),
                        )
                        return LLMResponse(
                            content=fallback_content,
                            tokens_used=0,
                            processing_time=processing_time,
                            model_used=f"{self.model_name}_fallback",
                            success=True,
                            metadata={
                                "session_id": session_id,
                                "item_id": item_id,
                                "content_type": content_type.value,
                                "timestamp": datetime.now().isoformat(),
                                "fallback_used": True,
                            },
                        )
                except Exception as recovery_error:
                    logger.warning(
                        "Error recovery service failed",
                        error=str(recovery_error),
                    )

            # Re-raise to let tenacity handle retries
            raise

    def get_service_stats(self) -> Dict[str, Any]:
        """Get service performance statistics including cache metrics."""
        total_requests = self.cache_hits + self.cache_misses
        cache_hit_rate = (self.cache_hits / max(total_requests, 1)) * 100

        # Get advanced cache stats
        cache_stats = self.cache.get_stats()

        # Get optimizer stats
        optimizer_stats = self.performance_optimizer.get_comprehensive_stats()
        async_stats = self.async_optimizer.get_comprehensive_stats()

        return {
            "total_calls": self.call_count,
            "total_tokens": self.total_tokens,
            "total_processing_time": self.total_processing_time,
            "average_processing_time": self.total_processing_time
            / max(self.call_count, 1),
            "model_name": self.model_name,
            "rate_limiter_status": (
                self.rate_limiter.get_status()
                if hasattr(self.rate_limiter, "get_status")
                else None
            ),
            "cache_performance": {
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "total_requests": total_requests,
                "cache_hit_rate_percent": round(cache_hit_rate, 2),
                "cache_size": 0,  # Legacy cache removed
            },
            "advanced_cache": cache_stats,
            "performance_optimizer": optimizer_stats,
            "async_optimizer": async_stats,
        }

    def reset_stats(self):
        """Reset performance statistics including cache metrics."""
        self.call_count = 0
        self.total_tokens = 0
        self.total_processing_time = 0.0
        self.cache_hits = 0
        self.cache_misses = 0

        logger.info("LLM service statistics reset")

    def clear_cache(self):
        """Clear the LLM response cache."""
        self.cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        logger.info("LLM service cache cleared")

    def optimize_performance(self) -> Dict[str, Any]:
        """Run performance optimization."""
        # Clear expired cache entries
        cache_stats_before = self.cache.get_stats()
        self.cache._evict_expired()
        cache_stats_after = self.cache.get_stats()

        result = {
            "cache_optimization": {
                "entries_before": cache_stats_before["size"],
                "entries_after": cache_stats_after["size"],
                "entries_removed": cache_stats_before["size"]
                - cache_stats_after["size"],
            },
            "timestamp": datetime.now().isoformat(),
        }

        logger.info("Performance optimization completed", result=result)
        return result

    def __del__(self):
        """Cleanup resources on destruction."""
        try:
            if hasattr(self, "executor"):
                self.executor.shutdown(wait=False)
        except Exception:
            pass  # Ignore cleanup errors


# Use get_llm_service() from llm_service.py for new implementations


# Global service instance
_LLM_SERVICE_INSTANCE = None


def get_llm_service(user_api_key: Optional[str] = None) -> EnhancedLLMService:
    """Get the global LLM service instance (singleton)."""
    global _LLM_SERVICE_INSTANCE
    if _LLM_SERVICE_INSTANCE is None:
        # The constructor now raises ConfigurationError on failure.
        # This critical error should halt the application, so no try/except is needed.
        _LLM_SERVICE_INSTANCE = EnhancedLLMService(user_api_key=user_api_key)
        logger.info("LLM service singleton created.")
    # Update the API key if a new user key is provided on subsequent calls
    elif user_api_key and (_LLM_SERVICE_INSTANCE.active_api_key != user_api_key or not _LLM_SERVICE_INSTANCE.using_user_key):
        _LLM_SERVICE_INSTANCE = EnhancedLLM_SERVICE_INSTANCE(user_api_key=user_api_key)
        logger.info("LLM service re-initialized with new user-provided API key.")

    return _LLM_SERVICE_INSTANCE
