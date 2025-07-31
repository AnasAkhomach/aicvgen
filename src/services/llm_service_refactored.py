"""Refactored LLM Service using tenacity and LangChain native caching.

This module replaces custom retry and caching services with library-native equivalents:
- Uses tenacity @retry decorator for retry logic
- Uses LangChain's built-in caching (InMemoryCache/SQLiteCache)
- Maintains the same interface as the original EnhancedLLMService
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)
from langchain_core.globals import set_llm_cache
from langchain_core.caches import InMemoryCache
from langchain_community.cache import SQLiteCache

from src.config.logging_config import get_logger
from src.constants.llm_constants import LLMConstants
from src.error_handling.exceptions import (
    NetworkError,
    RateLimitError,
    OperationTimeoutError,
    ConfigurationError,
)
from src.models.workflow_models import ContentType
from src.models.llm_data_models import LLMResponse
from src.models.llm_service_models import (
    LLMApiKeyInfo,
    LLMServiceStats,
    LLMPerformanceOptimizationResult,
)
from src.services.llm.llm_client_interface import LLMClientInterface
from src.services.llm_api_key_manager import LLMApiKeyManager
from src.services.llm_service_interface import LLMServiceInterface

logger = get_logger(__name__)

T = TypeVar("T")


class RefactoredLLMService(LLMServiceInterface):
    """Refactored LLM service using tenacity and LangChain native caching.

    This service replaces the custom retry and caching services with:
    - tenacity @retry decorators for retry logic
    - LangChain's built-in caching mechanisms

    Maintains the same interface as EnhancedLLMService for drop-in replacement.
    """

    def __init__(
        self,
        settings,
        llm_client: LLMClientInterface,
        api_key_manager: LLMApiKeyManager,
        cache_type: str = "memory",  # "memory" or "sqlite"
        cache_database_path: Optional[str] = None,
        timeout: int = LLMConstants.DEFAULT_TIMEOUT,
        max_retries: int = 3,
        async_optimizer=None,
    ):
        """Initialize the refactored LLM service.

        Args:
            settings: Injected settings/config dependency
            llm_client: Injected LLM client interface
            api_key_manager: Injected LLMApiKeyManager instance
            cache_type: Type of cache to use ("memory" or "sqlite")
            cache_database_path: Path for SQLite cache (if using sqlite)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            async_optimizer: Optional async optimizer
        """
        self.settings = settings
        self.llm_client = llm_client
        self.api_key_manager = api_key_manager
        self.timeout = timeout
        self.max_retries = max_retries
        self.async_optimizer = async_optimizer
        self._cache_initialized = False

        self.model_name = self.settings.llm_settings.default_model
        self._llm_model = None

        # Initialize statistics
        self.call_count = 0
        self.total_tokens = 0
        self.total_processing_time = 0.0
        self.cache_hits = 0
        self.cache_misses = 0

        # Initialize LangChain cache
        self._initialize_cache(cache_type, cache_database_path)

        logger.info(
            "Refactored LLM service initialized",
            extra={
                "model_name": self.model_name,
                "cache_type": cache_type,
                "timeout": timeout,
                "max_retries": max_retries,
            },
        )

    def _initialize_cache(self, cache_type: str, database_path: Optional[str] = None):
        """Initialize LangChain cache.

        Args:
            cache_type: Type of cache ("memory" or "sqlite")
            database_path: Path for SQLite database (if using sqlite)
        """
        try:
            if cache_type.lower() == "sqlite":
                db_path = database_path or ".langchain_cache.db"
                cache = SQLiteCache(database_path=db_path)
                logger.info(f"Initialized SQLite cache at {db_path}")
            else:
                cache = InMemoryCache()
                logger.info("Initialized in-memory cache")

            set_llm_cache(cache)
            self._cache_initialized = True

        except Exception as e:
            logger.error(f"Failed to initialize cache: {e}")
            # Fall back to no caching
            self._cache_initialized = False

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(
            (NetworkError, RateLimitError, OperationTimeoutError)
        ),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async def _call_llm_with_retry(
        self, prompt: str, session_id: Optional[str] = None, **kwargs
    ) -> Any:
        """Call LLM with tenacity retry logic.

        Args:
            prompt: The prompt to send to the LLM
            session_id: Optional session identifier
            **kwargs: Additional arguments for the LLM call

        Returns:
            LLM response

        Raises:
            NetworkError: For network-related issues
            RateLimitError: For rate limiting issues
            OperationTimeoutError: For timeout issues
        """
        try:
            # Apply timeout
            response = await asyncio.wait_for(
                self.llm_client.generate_content(
                    prompt=prompt, session_id=session_id, **kwargs
                ),
                timeout=self.timeout,
            )
            return response

        except asyncio.TimeoutError as e:
            logger.warning(f"LLM call timed out after {self.timeout}s")
            raise OperationTimeoutError(
                f"LLM call timed out after {self.timeout}s"
            ) from e
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            # Convert known exceptions to our custom types for retry logic
            if "rate limit" in str(e).lower():
                raise RateLimitError(str(e)) from e
            elif "network" in str(e).lower() or "connection" in str(e).lower():
                raise NetworkError(str(e)) from e
            else:
                # Re-raise unknown exceptions without retry
                raise

    def _create_llm_response(
        self,
        raw_response: Any,
        processing_time: float,
        content_type: ContentType,
        session_id: Optional[str] = None,
        item_id: Optional[str] = None,
    ) -> LLMResponse:
        """Create structured LLM response.

        Args:
            raw_response: Raw response from LLM
            processing_time: Time taken to process the request
            content_type: Type of content being processed
            session_id: Optional session identifier
            item_id: Optional item identifier

        Returns:
            Structured LLM response
        """
        # Extract text content
        if hasattr(raw_response, "text"):
            text_content = raw_response.text
        elif hasattr(raw_response, "content"):
            text_content = raw_response.content
        else:
            text_content = str(raw_response)

        # Extract token usage with defensive checks
        tokens_used = 0
        if hasattr(raw_response, "tokens"):
            tokens_used = raw_response.tokens or 0
        elif hasattr(raw_response, "usage") and raw_response.usage:
            if isinstance(raw_response.usage, dict):
                tokens_used = raw_response.usage.get("total_tokens", 0)
            else:
                tokens_used = getattr(raw_response.usage, "total_tokens", 0)

        # Ensure processing time is valid
        if processing_time <= 0:
            processing_time = 0.1  # Estimated minimum processing time

        return LLMResponse(
            content=text_content,
            tokens_used=tokens_used,
            processing_time=processing_time,
            model_used=self.model_name,
            success=True,
            metadata={
                "content_type": content_type.value if content_type else None,
                "session_id": session_id,
                "item_id": item_id,
                "timestamp": time.time(),
                "cache_hit": False,
            },
        )

    async def generate_content(
        self,
        prompt: str,
        model_name: Optional[str] = None,
        content_type: ContentType = ContentType.CV_ANALYSIS,
        session_id: Optional[str] = None,
        item_id: Optional[str] = None,
        **kwargs,
    ) -> LLMResponse:
        """Generate content using the LLM with retry and caching.

        Args:
            prompt: The prompt to send to the LLM
            model_name: Optional model name override
            content_type: Type of content being generated
            session_id: Optional session identifier
            item_id: Optional item identifier
            **kwargs: Additional arguments for the LLM call

        Returns:
            LLM response with content and metadata
        """
        start_time = time.time()

        try:
            # Ensure API key is valid
            await self.api_key_manager.ensure_api_key_valid()

            # Call LLM with retry logic (caching is handled by LangChain internally)
            raw_response = await self._call_llm_with_retry(
                prompt=prompt,
                session_id=session_id,
                model_name=model_name or self.model_name,
                **kwargs,
            )

            processing_time = time.time() - start_time

            # Create structured response
            llm_response = self._create_llm_response(
                raw_response=raw_response,
                processing_time=processing_time,
                content_type=content_type,
                session_id=session_id,
                item_id=item_id,
            )

            # Update statistics
            self.call_count += 1
            self.total_tokens += llm_response.tokens_used
            self.total_processing_time += processing_time

            logger.info(
                "LLM content generated successfully",
                extra={
                    "content_type": content_type.value,
                    "tokens_used": llm_response.tokens_used,
                    "processing_time": processing_time,
                    "session_id": session_id,
                },
            )

            return llm_response

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(
                "Failed to generate LLM content",
                extra={
                    "error": str(e),
                    "content_type": content_type.value,
                    "processing_time": processing_time,
                    "session_id": session_id,
                },
            )
            raise

    async def generate(
        self, prompt: str, model_name: Optional[str] = None, **kwargs
    ) -> str:
        """Generate content and return just the text.

        Args:
            prompt: The prompt to send to the LLM
            model_name: Optional model name override
            **kwargs: Additional arguments for the LLM call

        Returns:
            Generated text content
        """
        response = await self.generate_content(
            prompt=prompt, model_name=model_name, **kwargs
        )
        return response.content

    async def generate_structured_content(
        self,
        prompt: str,
        response_model: Type[T],
        model_name: Optional[str] = None,
        **kwargs,
    ) -> T:
        """Generate structured content using Pydantic models.

        Args:
            prompt: The prompt to send to the LLM
            response_model: Pydantic model class for structured output
            model_name: Optional model name override
            **kwargs: Additional arguments for the LLM call

        Returns:
            Structured response as Pydantic model instance
        """
        # Use LangChain's with_structured_output if available
        if hasattr(self.llm_client, "with_structured_output"):
            structured_llm = self.llm_client.with_structured_output(response_model)
            return await structured_llm.ainvoke(prompt)
        else:
            # Fallback to regular generation and parsing
            response = await self.generate_content(
                prompt=prompt, model_name=model_name, **kwargs
            )
            # Attempt to parse JSON response into the model
            import json

            try:
                data = json.loads(response.content)
                return response_model(**data)
            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"Failed to parse structured response: {e}")
                raise ConfigurationError(f"Failed to parse structured response: {e}")

    async def validate_api_key(self, api_key: str) -> bool:
        """Validate an API key.

        Args:
            api_key: The API key to validate

        Returns:
            True if valid, False otherwise
        """
        return await self.api_key_manager.validate_api_key(api_key)

    async def get_current_api_key_info(self) -> LLMApiKeyInfo:
        """Get information about the current API key.

        Returns:
            API key information
        """
        return await self.api_key_manager.get_current_api_key_info()

    async def ensure_api_key_valid(self) -> bool:
        """Ensure the current API key is valid.

        Returns:
            True if valid, False otherwise
        """
        return await self.api_key_manager.ensure_api_key_valid()

    def get_stats(self) -> LLMServiceStats:
        """Get service statistics.

        Returns:
            Service statistics
        """
        return LLMServiceStats(
            total_calls=self.call_count,
            total_tokens=self.total_tokens,
            total_processing_time=self.total_processing_time,
            average_processing_time=(
                self.total_processing_time / self.call_count
                if self.call_count > 0
                else 0
            ),
            model_name=self.settings.llm_settings.default_model,
            rate_limiter_status=None,
            cache_stats={
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "hit_rate": self.cache_hits / (self.cache_hits + self.cache_misses)
                if (self.cache_hits + self.cache_misses) > 0
                else 0,
            },
            optimizer_stats={},
            async_stats={},
        )

    def reset_stats(self):
        """Reset service statistics."""
        self.call_count = 0
        self.total_tokens = 0
        self.total_processing_time = 0.0
        self.cache_hits = 0
        self.cache_misses = 0
        logger.info("LLM service statistics reset")

    # async def optimize_performance(self) -> LLMPerformanceOptimizationResult:
    #     """Optimize service performance.

    #     Returns:
    #         Performance optimization result
    #     """
    #     if self.performance_optimizer:
    #         return await self.performance_optimizer.optimize()
    #     else:
    #         # Return default optimization result
    #         return LLMPerformanceOptimizationResult(
    #             optimizations_applied=[],
    #             performance_improvement=0.0,
    #             recommendations=["Consider enabling performance optimizer"],
    #         )
