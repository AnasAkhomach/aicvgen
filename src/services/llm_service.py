from datetime import datetime
from typing import Optional, TYPE_CHECKING, Any

from langchain_core.language_models import BaseLanguageModel
from langchain_core.caches import InMemoryCache
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config.logging_config import get_structured_logger
from src.models.llm_data_models import LLMResponse
from src.models.llm_service_models import (
    LLMApiKeyInfo,
    LLMPerformanceOptimizationResult,
    LLMServiceStats,
)
from src.models.workflow_models import ContentType
from src.services.llm_api_key_manager import LLMApiKeyManager
from src.services.llm_service_interface import LLMServiceInterface

if TYPE_CHECKING:
    from src.services.llm.llm_client_interface import LLMClientInterface

logger = get_structured_logger("llm_service")


class EnhancedLLMService(
    LLMServiceInterface
):  # pylint: disable=too-many-instance-attributes
    """
    Modernized LLM service using native LangChain features.
    Eliminates custom caching and retry services in favor of:
    - LangChain's built-in caching (InMemoryCache)
    - Tenacity for retry logic
    - Native structured output via with_structured_output

    This implementation follows Phase 3 of the refactoring plan.
    """

    def __init__(
        self,
        settings,
        llm_client: "LLMClientInterface",
        api_key_manager: LLMApiKeyManager,
        cache: Optional[InMemoryCache] = None,
    ):
        """
        Initialize the modernized LLM service.

        Args:
            settings: Injected settings/config dependency
            llm_client: Injected LLM client interface
            api_key_manager: Injected API key manager
            cache: Optional LangChain cache instance
        """
        self.settings = settings
        self.llm_client = llm_client
        self.api_key_manager = api_key_manager
        self.cache = cache or InMemoryCache()

        self.model_name = self.settings.llm_settings.default_model

        # Initialize the LLM model for LCEL pattern
        self._llm_model = None

        # Performance tracking
        self.call_count = 0
        self.total_tokens = 0
        self.total_processing_time = 0.0

        logger.info(
            "Modernized LLM service initialized",
            model=self.model_name,
            api_key_info=self.api_key_manager.get_current_api_key_info().model_dump(),
        )

    async def _ensure_cache_initialized(self):
        """Cache is already initialized in constructor with LangChain's InMemoryCache."""
        # No initialization needed - LangChain cache is ready to use
        pass

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

    def get_llm(self) -> BaseLanguageModel:
        """Get the underlying LLM model for LCEL pattern usage."""
        if self._llm_model is None:
            # Use the LLM client interface to get the underlying model
            # This maintains abstraction while providing LCEL compatibility
            self._llm_model = self.llm_client.get_langchain_model(
                model=self.model_name,
                temperature=self.settings.llm_settings.temperature,
                max_tokens=self.settings.llm_settings.max_tokens,
            )
        return self._llm_model

    async def generate_content(
        self,
        prompt: str,
        content_type: ContentType = None,
        session_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        item_id: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        system_instruction: Optional[str] = None,
        **kwargs,
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
            "system_instruction": system_instruction,
            **kwargs,
        }

        # Remove None values to avoid passing them downstream
        all_kwargs = {k: v for k, v in all_kwargs.items() if v is not None}

        # Ensure cache is initialized
        await self._ensure_cache_initialized()

        # Generate content with native retry logic
        llm_response = await self._generate_with_retry(
            prompt, content_type, **all_kwargs
        )

        # Update stats
        self.call_count += 1
        self.total_tokens += llm_response.tokens_used
        self.total_processing_time += llm_response.processing_time

        return llm_response

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True,
    )
    async def _generate_with_retry(
        self, prompt: str, content_type: ContentType, **kwargs
    ) -> LLMResponse:
        """Generate content with Tenacity-based retry logic."""
        return await self.llm_client.generate_content(
            prompt=prompt, content_type=content_type, **kwargs
        )

    async def generate_structured_content(self, prompt: str, response_model, **kwargs):
        """Generate structured content using LangChain's with_structured_output.

        Args:
            prompt: Text prompt to send to the model
            response_model: Pydantic model class for structured output
            **kwargs: Additional arguments

        Returns:
            Instance of response_model with structured data
        """
        # Get the LangChain model and add structured output capability
        llm = self.get_llm()
        structured_llm = llm.with_structured_output(response_model)

        # Generate structured content with retry
        result = await self._generate_structured_with_retry(
            structured_llm, prompt, **kwargs
        )

        return result

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True,
    )
    async def _generate_structured_with_retry(
        self, structured_llm, prompt: str, **kwargs
    ):
        """Generate structured content with Tenacity-based retry logic."""
        return await structured_llm.ainvoke(prompt, **kwargs)

    async def _get_service_stats(self) -> LLMServiceStats:
        """
        Internal method to get service performance statistics.
        Simplified for modernized implementation without custom services.
        """
        await self._ensure_cache_initialized()

        return LLMServiceStats(
            total_calls=self.call_count,
            total_tokens=self.total_tokens,
            total_processing_time=self.total_processing_time,
            average_processing_time=self.total_processing_time
            / max(self.call_count, 1),
            model_name=self.model_name,
            rate_limiter_status=None,  # No custom rate limiter
            cache_stats={},  # LangChain cache stats not exposed
            optimizer_stats={},  # No custom optimizer
            async_stats={},  # No custom async optimizer
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
        if self.cache:
            self.cache.clear()
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
        content_type = kwargs.pop("content_type", ContentType.CV_ANALYSIS)
        return await self.generate_content(
            prompt=prompt, content_type=content_type, **kwargs
        )
