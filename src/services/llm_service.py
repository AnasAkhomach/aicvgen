print("DEBUG: llm_service.py module is being loaded")

import os
import time
import asyncio
import hashlib
import json
import pickle
import traceback
import threading
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
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
from ..models.data_models import ContentType, LLMResponse
from ..models.llm_service_models import (
    LLMCacheEntry,
    LLMApiKeyInfo,
    LLMServiceStats,
    LLMPerformanceOptimizationResult,
)
from .rate_limiter import RateLimiter
from .error_recovery import get_error_recovery_service
from ..core.performance_optimizer import get_performance_optimizer
from ..core.async_optimizer import get_async_optimizer
from ..utils.exceptions import ConfigurationError, OperationTimeoutError
from ..error_handling.classification import (
    is_rate_limit_error,
    is_network_error,
    is_timeout_error,
)
from .llm_client import LLMClient
from .llm_retry_handler import LLMRetryHandler

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

    def _is_expired(self, entry: LLMCacheEntry) -> bool:
        """Check if cache entry is expired."""
        return datetime.now() > entry.expiry

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
                    self._cache.move_to_end(cache_key)
                    self._hits += 1
                    return entry.response
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
            entry = LLMCacheEntry(
                response=response,
                expiry=datetime.now() + timedelta(hours=ttl),
                created_at=datetime.now(),
                access_count=1,
                cache_key=cache_key,
            )
            self._cache[cache_key] = entry
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


class EnhancedLLMService:  # pylint: disable=too-many-instance-attributes
    """
    Enhanced LLM service with Phase 2 infrastructure integration.
    Now composes LLMClient and LLMRetryHandler for SRP and testability.
    Strict DI: all dependencies must be injected.
    """

    def __init__(
        self,
        settings,
        llm_client: LLMClient,
        llm_retry_handler: LLMRetryHandler,
        cache,
        timeout: int = 30,
        rate_limiter: Optional[RateLimiter] = None,
        error_recovery=None,
        performance_optimizer=None,
        async_optimizer=None,
        user_api_key: Optional[str] = None,
    ):
        """
        Initialize the enhanced LLM service.

        Args:
            settings: Injected settings/config dependency
            llm_client: Injected LLMClient instance
            llm_retry_handler: Injected LLMRetryHandler instance
            cache: Injected cache instance
            timeout: Maximum time in seconds to wait for LLM response
            rate_limiter: Optional rate limiter instance
            error_recovery: Optional error recovery service
            performance_optimizer: Optional performance optimizer
            async_optimizer: Optional async optimizer
            user_api_key: Optional user-provided API key (takes priority)
        """
        self.settings = settings
        self.timeout = timeout
        self.llm_client = llm_client
        self.llm_retry_handler = llm_retry_handler
        self.cache = cache
        self.rate_limiter = rate_limiter
        self.error_recovery = error_recovery
        self.performance_optimizer = performance_optimizer
        self.async_optimizer = async_optimizer
        self.user_api_key = user_api_key

        # API key management
        self.active_api_key = self._determine_active_api_key(user_api_key)
        self.fallback_api_key = self.settings.llm.gemini_api_key_fallback
        self.using_fallback = False
        self.using_user_key = bool(user_api_key)
        self.model_name = self.settings.llm_settings.default_model

        # Performance tracking
        self.call_count = 0
        self.total_tokens = 0
        self.total_processing_time = 0.0
        self.cache_hits = 0
        self.cache_misses = 0
        logger.info(
            "Enhanced LLM service initialized",
            model=self.model_name,
            timeout=timeout,
            using_user_key=self.using_user_key,
            using_fallback_key=self.using_fallback,
            has_fallback_key=bool(self.fallback_api_key),
            cache_enabled=True,
        )

    async def ensure_api_key_valid(self):
        """
        Explicitly validate the API key. Raises ConfigurationError if invalid.
        Call this after construction in async context.
        """
        if not await self.validate_api_key():
            raise ConfigurationError(
                "Gemini API key validation failed. Please check your GEMINI_API_KEY or GEMINI_API_KEY_FALLBACK. Application cannot start without a valid key."
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
        import logging

        try:
            # Use the current event loop to run the synchronous API call
            loop = asyncio.get_event_loop()

            # Make a lightweight API call to validate the key
            # genai.list_models() is a simple call that requires authentication
            models = await loop.run_in_executor(None, lambda: list(genai.list_models()))

            # If we get here without exception, the key is valid
            logger.info(
                "API key validation successful",
                extra={
                    "models_count": len(models),
                    "using_user_key": self.using_user_key,
                    "using_fallback": self.using_fallback,
                },
            )
            return True
        except Exception as e:
            # Log the validation failure with full details
            logger.warning(
                "API key validation failed",
                extra={
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "using_user_key": self.using_user_key,
                    "using_fallback": self.using_fallback,
                },
            )
            logging.error(f"Gemini API key validation exception: {e}", exc_info=True)
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
                    error_type=type(e).__name__,
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

    def get_current_api_key_info(self) -> LLMApiKeyInfo:
        """
        Get information about the currently active API key.

        Returns:
            Dict containing API key status information
        """
        return LLMApiKeyInfo(
            using_user_key=self.using_user_key,
            using_fallback=self.using_fallback,
            has_fallback_available=bool(self.fallback_api_key),
            key_source=(
                "user"
                if self.using_user_key
                else ("fallback" if self.using_fallback else "none")
            ),
        )

    async def _call_llm_with_retry(
        self, prompt: str, session_id: str, trace_id: str
    ) -> Any:
        """Call LLM with retry logic using the composed retry handler."""
        self.call_count += 1
        logger.info(
            "Starting LLM generation",
            extra={
                "trace_id": trace_id,
                "session_id": session_id,
                "prompt_length": len(prompt),
            },
        )
        # Directly await the retry handler
        return await self.llm_retry_handler.generate_content(prompt)

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

        This method orchestrates the workflow: cache check -> rate limit -> retryable API call -> caching.

        Args:
            prompt: Text prompt to send to the model
            content_type: Type of content being generated
            session_id: Session ID for tracking
            item_id: Item ID for tracking
            max_retries: Maximum number of retries
            trace_id: Optional trace ID for tracking

        Returns:
            LLMResponse with generated content and metadata
        """
        start_time = time.time()

        # 1. Check cache
        cached_response = self._check_cache(
            prompt, content_type, session_id, item_id, trace_id
        )
        if cached_response:
            return cached_response

        # 2. Check rate limit
        await self._apply_rate_limiting()

        # 3. Call with retry logic
        try:
            # Await the retry handler with timeout
            response = await asyncio.wait_for(
                self._call_llm_with_retry(prompt, session_id, trace_id),
                timeout=self.timeout,
            )
            processing_time = time.time() - start_time

            # 4. Cache result and return
            llm_response = self._create_llm_response(
                response, processing_time, content_type, session_id, item_id
            )
            self._cache_response(prompt, content_type, llm_response)

            return llm_response

        except ConfigurationError:
            # Do not retry on fatal config errors. Re-raise immediately.
            raise
        except asyncio.TimeoutError:
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
                f"LLM request timed out after {self.timeout} seconds"
            )
        except Exception as e:
            # Handle errors and try fallback content
            return await self._handle_error_with_fallback(
                e, content_type, session_id, item_id, start_time
            )

    def _check_cache(
        self,
        prompt: str,
        content_type: ContentType,
        session_id: str,
        item_id: str,
        trace_id: str,
    ) -> Optional[LLMResponse]:
        """Check cache for existing response."""
        cache_key = create_cache_key(prompt, self.model_name, content_type.value)
        cached_response = (
            self.cache.get(cache_key, self.model_name) if self.cache else None
        )
        if cached_response:
            self.cache_hits += 1
            logger.info(
                "Cache hit for LLM request",
                extra={
                    "trace_id": trace_id,
                    "session_id": session_id,
                    "item_id": item_id,
                    "content_type": content_type.value,
                    "cache_key": cache_key[:20] + "...",
                },
            )
            cached_response["metadata"]["cache_hit"] = True
            cached_response["metadata"]["session_id"] = session_id
            cached_response["metadata"]["item_id"] = item_id
            cached_response["processing_time"] = 0.001
            return LLMResponse(**cached_response)
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
        return None

    async def _apply_rate_limiting(self) -> None:
        """Apply rate limiting with centralized logic."""
        if self.rate_limiter:
            await self.rate_limiter.wait_if_needed_async(self.model_name)

    def _create_llm_response(
        self,
        response: Any,
        processing_time: float,
        content_type: ContentType,
        session_id: str,
        item_id: str,
    ) -> LLMResponse:
        """Create structured LLM response from API response."""
        self.total_processing_time += processing_time

        if not hasattr(response, "text") or response.text is None:
            raise ValueError("LLM returned an empty or invalid response")

        # Estimate token usage (rough approximation)
        if hasattr(response, "text") and isinstance(response.text, str):
            tokens_used = len(response.text.split())
        else:
            tokens_used = 0
        self.total_tokens += tokens_used

        # Create structured response with defensive metadata
        safe_metadata = {
            "session_id": str(session_id) if session_id is not None else None,
            "item_id": str(item_id) if item_id is not None else None,
            "content_type": str(content_type.value) if content_type else None,
            "timestamp": datetime.now().isoformat(),
            "cache_hit": False,
        }

        llm_response = LLMResponse(
            content=response.text,
            tokens_used=tokens_used,
            processing_time=processing_time,
            model_used=self.model_name,
            success=True,
            metadata=safe_metadata,
        )  # Log successful generation
        logger.info(
            "LLM generation completed successfully",
            session_id=session_id,
            item_id=item_id,
            processing_time=processing_time,
            tokens_used=tokens_used,
            response_length=len(response.text),
        )

        return llm_response

    def _cache_response(
        self, prompt: str, content_type: ContentType, llm_response: LLMResponse
    ) -> None:
        """Cache the successful response."""
        cache_data = {
            "content": llm_response.content,
            "tokens_used": llm_response.tokens_used,
            "processing_time": llm_response.processing_time,
            "model_used": llm_response.model_used,
            "success": llm_response.success,
            "error_message": None,
            "metadata": {
                "content_type": content_type.value,
                "timestamp": datetime.now().isoformat(),
                "cache_hit": False,
            },
        }
        if self.cache:
            self.cache.set(prompt, self.model_name, cache_data, ttl_hours=2)

    async def _handle_error_with_fallback(
        self,
        error: Exception,
        content_type: ContentType,
        session_id: str,
        item_id: str,
        start_time: float,
    ) -> LLMResponse:
        """Handle errors and attempt fallback content."""
        processing_time = time.time() - start_time

        # Check if this is a rate limit error and try fallback key
        error_str = str(error).lower()
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
                error=str(error),
            )

            if self._switch_to_fallback_key():
                # Re-raise to let tenacity handle the retry
                raise error

        # Use error recovery service if available
        if self.error_recovery:
            try:
                fallback_content = await self.error_recovery.get_fallback_content(
                    content_type, str(error)
                )
                if fallback_content:
                    logger.info(
                        "Using fallback content from error recovery service",
                        content_type=content_type.value,
                        error=str(error),
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

        # If no fallback available, re-raise the original error
        logger.error(
            "LLM generation failed with no available fallback",
            content_type=content_type.value,
            error=str(error),
            processing_time=processing_time,
        )
        raise error

    def get_service_stats(self) -> LLMServiceStats:
        """Get service performance statistics including cache metrics."""
        total_requests = self.cache_hits + self.cache_misses
        cache_hit_rate = (
            self.cache_hits / max(total_requests, 1)
        ) * 100  # Get advanced cache stats
        cache_stats = self.cache.get_stats() if self.cache else {}

        # Get optimizer stats
        optimizer_stats = (
            self.performance_optimizer.get_comprehensive_stats()
            if self.performance_optimizer
            else {}
        )
        async_stats = (
            self.async_optimizer.get_comprehensive_stats()
            if self.async_optimizer
            else {}
        )

        return LLMServiceStats(
            total_calls=self.call_count,
            total_tokens=self.total_tokens,
            total_processing_time=self.total_processing_time,
            average_processing_time=self.total_processing_time
            / max(self.call_count, 1),
            model_name=self.model_name,
            rate_limiter_status=(
                self.rate_limiter.get_status()
                if hasattr(self.rate_limiter, "get_status")
                else None
            ),
            cache_stats=cache_stats,
            optimizer_stats=optimizer_stats,
            async_stats=async_stats,
        )

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
        if self.cache:
            self.cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        logger.info("LLM service cache cleared")

    def optimize_performance(self) -> LLMPerformanceOptimizationResult:
        """Run performance optimization."""
        # Clear expired cache entries (if cache is available)
        cache_stats_before = self.cache.get_stats() if self.cache else {"size": 0}
        if self.cache and hasattr(self.cache, "_evict_expired"):
            self.cache._evict_expired()
        cache_stats_after = self.cache.get_stats() if self.cache else {"size": 0}

        result = LLMPerformanceOptimizationResult(
            cache_optimization={
                "entries_before": cache_stats_before["size"],
                "entries_after": cache_stats_after["size"],
                "entries_removed": cache_stats_before["size"]
                - cache_stats_after["size"],
            },
            timestamp=datetime.now().isoformat(),
        )

        logger.info("Performance optimization completed", result=result.dict())
        return result

    def __del__(self):
        """Cleanup resources on destruction."""
        try:
            if hasattr(self, "executor"):
                self.executor.shutdown(wait=False)
        except Exception:
            pass  # Ignore cleanup errors

    async def generate(self, *args, **kwargs):
        """
        Backward-compatible wrapper for generate_content.
        Accepts arbitrary args/kwargs and passes them to generate_content.
        """
        return await self.generate_content(*args, **kwargs)


# get_llm_service is deprecated and removed. Use DI and instantiate EnhancedLLMService with explicit dependencies.
