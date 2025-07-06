import asyncio
import time
from datetime import datetime
from typing import Any, Optional

from src.utils.import_fallbacks import get_google_exceptions
from src.config.logging_config import get_structured_logger
from src.constants.llm_constants import LLMConstants
from src.error_handling.classification import is_rate_limit_error
from src.error_handling.exceptions import (ConfigurationError, NetworkError, OperationTimeoutError, RateLimitError)
from src.models.llm_data_models import LLMResponse
from src.models.workflow_models import ContentType
from src.services.llm_api_key_manager import LLMApiKeyManager
from src.services.llm_retry_handler import LLMRetryHandler
from src.services.rate_limiter import RateLimiter
# Import Google API exceptions with standardized fallback handling
google_exceptions, _ = get_google_exceptions()



logger = get_structured_logger("llm_retry_service")


class LLMRetryService:
    """
    Handles retry logic, error handling, and fallback content generation.
    Manages all LLM request retry operations and error recovery.
    """

    def __init__(
        self,
        llm_retry_handler: LLMRetryHandler,
        api_key_manager: LLMApiKeyManager,
        rate_limiter: Optional[RateLimiter] = None,
        error_recovery=None,
        timeout: int = LLMConstants.DEFAULT_TIMEOUT,
        model_name: str = LLMConstants.DEFAULT_MODEL,
    ):
        """
        Initialize the retry service.

        Args:
            llm_retry_handler: Injected LLMRetryHandler instance
            api_key_manager: Injected LLMApiKeyManager instance
            rate_limiter: Optional rate limiter instance
            error_recovery: Optional error recovery service
            timeout: Maximum time in seconds to wait for LLM response
            model_name: Name of the LLM model being used
        """
        self.llm_retry_handler = llm_retry_handler
        self.api_key_manager = api_key_manager
        self.rate_limiter = rate_limiter
        self.error_recovery = error_recovery
        self.timeout = timeout
        self.model_name = model_name

        logger.info(
            "LLM retry service initialized",
            timeout=timeout,
            model_name=model_name,
            has_rate_limiter=bool(rate_limiter),
            has_error_recovery=bool(error_recovery),
        )

    async def apply_rate_limiting(self) -> None:
        """Apply rate limiting with centralized logic."""
        if self.rate_limiter:
            await self.rate_limiter.wait_if_needed_async(self.model_name)

    async def call_llm_with_retry(self, prompt: str, **kwargs) -> Any:
        """Call LLM via retry handler with timeout enforcement."""
        logger.info("Calling LLM via retry handler", **kwargs)
        try:
            # Enforce timeout on the entire retry operation
            return await asyncio.wait_for(
                self.llm_retry_handler.generate_content(prompt, **kwargs),
                timeout=self.timeout,
            )
        except asyncio.TimeoutError as e:
            logger.error(
                "LLM call timed out after %s seconds",
                self.timeout,
                prompt=prompt,
                **kwargs,
            )
            raise OperationTimeoutError(
                f"LLM operation timed out after {self.timeout} seconds"
            ) from e

    def create_llm_response(
        self,
        response,
        processing_time: float,
        content_type: ContentType,
        **kwargs,
    ) -> LLMResponse:
        """Create a structured LLMResponse object from raw LLM response."""
        session_id = kwargs.get("session_id")
        item_id = kwargs.get("item_id")

        # Safely get token usage
        tokens_used = getattr(response, "tokens", 0) or getattr(
            response, "usage", {}
        ).get("total_tokens", 0)

        # Defensive check for negative or excessively high token usage
        if tokens_used < LLMConstants.MIN_TOKEN_USAGE:
            logger.warning(
                "Negative token usage detected, defaulting to 0",
                response=response,
                processing_time=processing_time,
            )
            tokens_used = LLMConstants.MIN_TOKEN_USAGE
        elif tokens_used > LLMConstants.MAX_TOKEN_USAGE_THRESHOLD:
            logger.warning(
                "Excessive token usage detected, capping at threshold",
                response=response,
                processing_time=processing_time,
                tokens_used=tokens_used,
                threshold=LLMConstants.MAX_TOKEN_USAGE_THRESHOLD,
            )
            tokens_used = LLMConstants.MAX_TOKEN_USAGE_THRESHOLD

        # Estimate processing time based on response length (defensive)
        if processing_time is None or processing_time <= 0:
            processing_time = max(
                LLMConstants.MIN_PROCESSING_TIME,
                tokens_used / LLMConstants.TOKEN_PROCESSING_RATE
            )

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
        )

        logger.info(
            "LLM generation completed successfully",
            session_id=session_id,
            item_id=item_id,
            processing_time=processing_time,
            tokens_used=tokens_used,
            response_length=len(response.text),
        )

        return llm_response

    async def handle_error_with_fallback(
        self, error: Exception, content_type: ContentType, start_time: float, **kwargs
    ) -> LLMResponse:
        """Handle errors and attempt fallback content or API key switching."""
        processing_time = time.time() - start_time
        session_id = kwargs.get("session_id")
        item_id = kwargs.get("item_id")

        # Check if this is a rate limit error and try fallback key
        if is_rate_limit_error(error) and self.api_key_manager.has_fallback_available():
            logger.warning(
                "Rate limit detected, attempting to switch to fallback API key",
                error=str(error),
            )

            if await self.api_key_manager.switch_to_fallback_key():
                # After switching key, retry the operation by re-raising
                logger.info("Re-raising error to trigger retry with new key")
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
                    "Error recovery service failed, propagating original error",
                    recovery_error=str(recovery_error),
                    original_error=str(error),
                )
                # Re-raise the original error to maintain error propagation contract
                # Don't swallow exceptions that should be handled upstream
                raise error

        # If no fallback available, re-raise the original error
        logger.error(
            "LLM generation failed with no available fallback",
            content_type=content_type.value,
            error=str(error),
            processing_time=processing_time,
        )
        raise error

    async def generate_content_with_retry(
        self, prompt: str, content_type: ContentType, **kwargs
    ) -> LLMResponse:
        """
        Generate content with comprehensive retry and error handling.

        Args:
            prompt: Text prompt to send to the model
            content_type: Type of content being generated
            **kwargs: Additional arguments including session_id, item_id, trace_id

        Returns:
            LLMResponse with generated content and metadata

        Raises:
            ConfigurationError: For fatal configuration errors
            OperationTimeoutError: When operations timeout
            Various LLM-related exceptions: When all retries fail
        """
        start_time = time.time()

        # Apply rate limiting
        await self.apply_rate_limiting()

        try:
            # Call with retry logic and timeout
            response = await asyncio.wait_for(
                self.call_llm_with_retry(prompt=prompt, **kwargs),
                timeout=self.timeout,
            )
            processing_time = time.time() - start_time

            # Create and return structured response
            return self.create_llm_response(
                response, processing_time, content_type, **kwargs
            )

        except ConfigurationError as e:
            # Do not retry on fatal config errors. Re-raise immediately.
            raise e
        except (RateLimitError, NetworkError) as e:
            logger.warning(
                "A transient error occurred",
                error=str(e),
                trace_id=kwargs.get("trace_id"),
            )
            # Re-raise to be handled by tenacity if applicable
            raise e
        except asyncio.TimeoutError as e:
            logger.error(
                "LLM request timed out after %s seconds",
                self.timeout,
                trace_id=kwargs.get("trace_id"),
                session_id=kwargs.get("session_id"),
                prompt_length=len(prompt),
            )
            raise OperationTimeoutError(
                f"LLM request timed out after {self.timeout} seconds"
            ) from e
        except google_exceptions.GoogleAPICallError as e:
            # Handle errors and try fallback content
            return await self.handle_error_with_fallback(
                e, content_type, start_time=start_time, prompt=prompt, **kwargs
            )
