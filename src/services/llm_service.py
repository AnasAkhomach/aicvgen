from typing import Optional
from datetime import datetime

from src.config.logging_config import get_structured_logger
from src.models.data_models import ContentType, LLMResponse
from src.models.llm_service_models import (
    LLMApiKeyInfo,
    LLMServiceStats,
    LLMPerformanceOptimizationResult,
)
from src.services.llm_caching_service import LLMCachingService
from src.services.llm_api_key_manager import LLMApiKeyManager
from src.services.llm_retry_service import LLMRetryService
from src.services.rate_limiter import RateLimiter

logger = get_structured_logger("llm_service")


class EnhancedLLMService:  # pylint: disable=too-many-instance-attributes
    """
    Enhanced LLM service with decomposed responsibilities.
    Now uses composed services for caching, API key management, and retry logic.
    Strict DI: all dependencies must be injected.
    Fully asynchronous implementation.
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

    def get_current_api_key_info(self) -> LLMApiKeyInfo:
        """Get information about the currently active API key."""
        return self.api_key_manager.get_current_api_key_info()

    async def generate_content(
        self, prompt: str, content_type: ContentType, **kwargs
    ) -> LLMResponse:
        """
        Generate content using the Gemini model with enhanced error handling and caching.

        This method orchestrates the workflow: cache check -> retry service -> caching.

        Args:
            prompt: Text prompt to send to the model
            content_type: Type of content being generated
            **kwargs: Additional arguments including session_id, item_id, trace_id

        Returns:
            LLMResponse with generated content and metadata
        """
        # 0. Ensure cache is initialized
        await self._ensure_cache_initialized()

        # 1. Check cache
        cached_response = await self.caching_service.check_cache(
            prompt, self.model_name, content_type, **kwargs
        )
        if cached_response:
            return cached_response

        # 2. Generate content with retry logic
        llm_response = await self.retry_service.generate_content_with_retry(
            prompt, content_type, **kwargs
        )

        # 3. Cache the response
        await self.caching_service.cache_response(
            prompt, self.model_name, content_type, llm_response, **kwargs
        )

        # 4. Update stats
        self.call_count += 1
        self.total_tokens += llm_response.tokens_used
        self.total_processing_time += llm_response.processing_time

        return llm_response

    async def get_service_stats(self) -> LLMServiceStats:
        """Get service performance statistics including cache metrics."""
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

    async def clear_cache(self):
        """Clear the LLM response cache."""
        await self._ensure_cache_initialized()
        if self.caching_service:
            await self.caching_service.clear()
        logger.info("LLM service cache cleared")

    async def optimize_performance(self) -> LLMPerformanceOptimizationResult:
        """Run performance optimization."""
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

    async def generate(self, *args, **kwargs):
        """
        Backward-compatible wrapper for generate_content.
        Accepts arbitrary args/kwargs and passes them to generate_content.
        """
        return await self.generate_content(*args, **kwargs)
