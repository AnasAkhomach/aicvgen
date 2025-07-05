from datetime import datetime
from typing import Optional

from src.config.logging_config import get_structured_logger
from src.models.llm_data_models import LLMResponse
from src.models.llm_service_models import (LLMApiKeyInfo, LLMPerformanceOptimizationResult, LLMServiceStats)
from src.models.workflow_models import ContentType
from src.services.llm_api_key_manager import LLMApiKeyManager
from src.services.llm_caching_service import LLMCachingService
from src.services.llm_retry_service import LLMRetryService
from src.services.llm_service_interface import LLMServiceInterface
from src.services.rate_limiter import RateLimiter

logger = get_structured_logger("llm_service")


class EnhancedLLMService(LLMServiceInterface):  # pylint: disable=too-many-instance-attributes
    """
    Enhanced LLM service with decomposed responsibilities.
    Now uses composed services for caching, API key management, and retry logic.
    Strict DI: all dependencies must be injected.
    Fully asynchronous implementation.
    
    This implementation hides internal caching, retry, and rate limiting details
    behind a clean interface contract.
    """

    def __init__(
        self,
        settings,
        caching_service: LLMCachingService,
        api_key_manager: LLMApiKeyManager,
        retry_service: LLMRetryService,
        rate_limiter: Optional[RateLimiter] = None,
        performance_optimizer=None,
        async_optimizer=None,
    ):
        """
        Initialize the enhanced LLM service.

        Args:
            settings: Injected settings/config dependency
            caching_service: Injected LLMCachingService instance
            api_key_manager: Injected LLMApiKeyManager instance
            retry_service: Injected LLMRetryService instance
            rate_limiter: Optional rate limiter instance
            performance_optimizer: Optional performance optimizer
            async_optimizer: Optional async optimizer
        """
        self.settings = settings
        self.caching_service = caching_service
        self.api_key_manager = api_key_manager
        self.retry_service = retry_service
        self.rate_limiter = rate_limiter
        self.performance_optimizer = performance_optimizer
        self.async_optimizer = async_optimizer
        self._cache_initialized = False

        self.model_name = self.settings.llm_settings.default_model

        # Performance tracking
        self.call_count = 0
        self.total_tokens = 0
        self.total_processing_time = 0.0

        logger.info(
            "Enhanced LLM service initialized",
            model=self.model_name,
            api_key_info=self.api_key_manager.get_current_api_key_info().model_dump(),
        )

    async def _ensure_cache_initialized(self):
        """Initializes the cache if it hasn't been already."""
        if not self._cache_initialized:
            if self.caching_service and hasattr(self.caching_service, "initialize"):
                await self.caching_service.initialize()
            self._cache_initialized = True

    async def ensure_api_key_valid(self):
        """
        Explicitly validate the API key. Raises ConfigurationError if invalid.
        Call this after construction in async context.
        """
        await self.api_key_manager.ensure_api_key_valid()

    async def validate_api_key(self) -> bool:
        """
        Validate the current API key. Returns True if valid, False otherwise.
        This is intended for UI callbacks and does not raise.
        """
        try:
            await self.api_key_manager.ensure_api_key_valid()
            return True
        except Exception as e:
            logger.error("API key validation failed", error=str(e))
            return False

    def get_current_api_key_info(self) -> LLMApiKeyInfo:
        """Get information about the currently active API key."""
        return self.api_key_manager.get_current_api_key_info()

    async def generate_content(
        self,
        prompt: str,
        content_type: ContentType = None,
        session_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        item_id: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate content using the Gemini model with enhanced error handling and caching.

        This method orchestrates the workflow: cache check -> retry service -> caching.

        Args:
            prompt: Text prompt to send to the model
            content_type: Type of content being generated (optional, defaults to CV_ANALYSIS)
            session_id: Session identifier for tracking
            trace_id: Trace identifier for debugging
            item_id: Item identifier for tracking
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature for generation
            **kwargs: Additional LLM parameters

        Returns:
            LLMResponse with generated content and metadata
        """
        # Default content_type if not provided for backward compatibility
        if content_type is None:
            content_type = ContentType.CV_ANALYSIS
            logger.debug("No content_type provided, defaulting to CV_ANALYSIS")

        # Consolidate all parameters for consistent handling
        all_kwargs = {
            "session_id": session_id,
            "trace_id": trace_id,
            "item_id": item_id,
            "max_tokens": max_tokens,
            "temperature": temperature,
            **kwargs
        }

        # Remove None values to avoid passing them downstream
        all_kwargs = {k: v for k, v in all_kwargs.items() if v is not None}

        # 0. Ensure cache is initialized
        await self._ensure_cache_initialized()

        # 1. Check cache
        cached_response = await self.caching_service.check_cache(
            prompt, self.model_name, content_type, **all_kwargs
        )
        if cached_response:
            return cached_response

        # 2. Generate content with retry logic
        llm_response = await self.retry_service.generate_content_with_retry(
            prompt, content_type, **all_kwargs
        )

        # 3. Cache the response
        await self.caching_service.cache_response(
            prompt, self.model_name, content_type, llm_response, **all_kwargs
        )

        # 4. Update stats
        self.call_count += 1
        self.total_tokens += llm_response.tokens_used
        self.total_processing_time += llm_response.processing_time

        return llm_response

    async def _get_service_stats(self) -> LLMServiceStats:
        """
        Internal method to get service performance statistics including cache metrics.
        This is not part of the public interface to avoid exposing implementation details.
        """
        await self._ensure_cache_initialized()

        # Get cache stats from caching service
        cache_stats = await self.caching_service.get_cache_stats()

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
        """Reset performance statistics."""
        self.call_count = 0
        self.total_tokens = 0
        self.total_processing_time = 0.0
        logger.info("LLM service statistics reset")

    async def _clear_cache(self):
        """
        Internal method to clear the LLM response cache.
        This is not part of the public interface to avoid exposing implementation details.
        """
        await self._ensure_cache_initialized()
        if self.caching_service:
            await self.caching_service.clear()
        logger.info("LLM service cache cleared")

    async def _optimize_performance(self) -> LLMPerformanceOptimizationResult:
        """
        Internal method to run performance optimization.
        This is not part of the public interface to avoid exposing implementation details.
        """
        await self._ensure_cache_initialized()

        # Clear expired cache entries (if cache is available)
        cache_stats_before = (
            await self.caching_service.get_cache_stats()
            if self.caching_service
            else {"size": 0}
        )
        if self.caching_service and hasattr(
            self.caching_service, "evict_expired_entries"
        ):
            await self.caching_service.evict_expired_entries()
        cache_stats_after = (
            await self.caching_service.get_cache_stats()
            if self.caching_service
            else {"size": 0}
        )

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

    async def generate(self, prompt: str, **kwargs):
        """
        Backward-compatible wrapper for generate_content.
        Standardizes parameter handling for legacy callers.

        Args:
            prompt: Text prompt to send to the model
            **kwargs: Additional arguments including max_tokens, temperature, etc.

        Returns:
            LLMResponse with generated content and metadata
        """
        # Extract content_type from kwargs if provided, otherwise use default
        content_type = kwargs.pop('content_type', ContentType.CV_ANALYSIS)
        return await self.generate_content(prompt=prompt, content_type=content_type, **kwargs)
