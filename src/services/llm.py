import os
import sys
import time
import asyncio
import concurrent.futures
import threading
from queue import Queue
import logging
import hashlib
import functools
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime

try:
    import google.generativeai as genai
except ImportError:
    genai = None

# Import tenacity for retry logic with exponential backoff
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)

from ..config.logging_config import get_structured_logger
from ..config.settings import get_config
from ..models.data_models import ContentType
from .rate_limiter import RateLimiter as EnhancedRateLimiter
from .error_recovery import get_error_recovery_service

logger = get_structured_logger("llm_service")

# Define retryable exceptions for LLM API calls
# NOTE: These exceptions are dependent on the google-generativeai library
# and may need to be updated if the library changes its exception hierarchy
RETRYABLE_EXCEPTIONS = (
    Exception,  # Catch-all for now, will be refined based on actual google.generativeai exceptions
    # Common network/service exceptions that should be retried:
    # - ResourceExhausted (rate limits)
    # - ServiceUnavailable (temporary service issues)
    # - InternalServerError (server errors)
    # - DeadlineExceeded (timeout errors)
    # - TimeoutError (connection timeouts)
    # - ConnectionError (network issues)
)


# --- Caching Mechanism ---
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
def _get_cached_response(cache_key: str) -> Optional[str]:
    """Internal cache lookup function."""
    # This function exists to provide a cacheable interface
    # The actual caching is handled by the decorator
    return None


# Cache storage for actual responses
_response_cache: Dict[str, Dict[str, Any]] = {}
_cache_lock = threading.Lock()


def get_cached_response(cache_key: str) -> Optional[Dict[str, Any]]:
    """Get cached response if available."""
    with _cache_lock:
        return _response_cache.get(cache_key)


def set_cached_response(cache_key: str, response_data: Dict[str, Any]):
    """Cache a response with LRU eviction."""
    with _cache_lock:
        # Simple LRU: if cache is full, remove oldest entry
        if len(_response_cache) >= 128:
            # Remove the first (oldest) item
            oldest_key = next(iter(_response_cache))
            del _response_cache[oldest_key]
            logger.debug(f"Cache evicted oldest entry: {oldest_key[:20]}...")
        
        _response_cache[cache_key] = response_data
        logger.debug(f"Response cached with key: {cache_key[:20]}...")


def clear_cache():
    """Clear the entire response cache."""
    with _cache_lock:
        _response_cache.clear()
        logger.info("LLM response cache cleared")


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


class EnhancedLLMService:
    """
    Enhanced LLM service with Phase 1 infrastructure integration.
    Provides structured logging, rate limiting, and error recovery.
    """

    def __init__(
        self, 
        timeout: int = 30, 
        rate_limiter: Optional[EnhancedRateLimiter] = None,
        error_recovery = None,
        user_api_key: Optional[str] = None
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
        
        # Configure API keys with user key priority and fallback support
        self.user_api_key = user_api_key
        self.primary_api_key = self.settings.llm.gemini_api_key_primary
        self.fallback_api_key = self.settings.llm.gemini_api_key_fallback
        
        # Prioritize user-provided key, then primary, then fallback
        if self.user_api_key:
            api_key = self.user_api_key
            self.using_user_key = True
        elif self.primary_api_key:
            api_key = self.primary_api_key
            self.using_user_key = False
        elif self.fallback_api_key:
            api_key = self.fallback_api_key
            self.using_user_key = False
        else:
            raise ValueError("No Gemini API key found. Please provide your API key or set GEMINI_API_KEY environment variable.")
        
        self.current_api_key = api_key

        # Initialize the model
        genai.configure(api_key=api_key)
        self.model_name = "gemini-2.0-flash"
        self.llm = genai.GenerativeModel(self.model_name)
        self.using_fallback = not bool(self.user_api_key or self.primary_api_key)
        
        # Enhanced services
        self.rate_limiter = rate_limiter or EnhancedRateLimiter()
        self.error_recovery = error_recovery or get_error_recovery_service()
        
        # Performance tracking
        self.call_count = 0
        self.total_tokens = 0
        self.total_processing_time = 0.0
        
        # Cache performance tracking
        self.cache_hits = 0
        self.cache_misses = 0
        
        logger.info(
            "Enhanced LLM service initialized",
            model=self.model_name,
            timeout=timeout,
            using_user_key=self.using_user_key,
            using_fallback_key=self.using_fallback
        )
    
    def _switch_to_fallback_key(self):
        """Switch to fallback API key when rate limits are encountered."""
        if not self.using_fallback and self.fallback_api_key:
            logger.warning(
                "Switching to fallback API key due to rate limit or error",
                previous_key_type="primary",
                fallback_available=True
            )
            
            # Reconfigure with fallback key
            genai.configure(api_key=self.fallback_api_key)
            self.current_api_key = self.fallback_api_key
            self.using_fallback = True
            
            # Reinitialize the model with new key
            self.llm = genai.GenerativeModel(self.model_name)
            
            logger.info("Successfully switched to fallback API key")
            return True
        else:
            logger.error(
                "Cannot switch to fallback key",
                already_using_fallback=self.using_fallback,
                fallback_available=bool(self.fallback_api_key)
            )
            return False

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
        reraise=True
    )
    def _make_llm_api_call(self, prompt: str) -> Any:
        """
        Make the actual LLM API call with retry logic using tenacity.
        
        This method is decorated with @retry to handle transient errors
        with exponential backoff. It will retry up to 3 times with
        exponentially increasing delays (4s, 8s, 10s max).
        
        Args:
            prompt: Text prompt to send to the model
            
        Returns:
            Generated response from the LLM
            
        Raises:
            Various exceptions from the google-generativeai library
        """
        return self.llm.generate_content(prompt)

    def _generate_with_timeout(self, prompt: str, session_id: str = None) -> Any:
        """
        Generate content with a timeout using ThreadPoolExecutor.
        Now uses the retry-enabled _make_llm_api_call method.

        Args:
            prompt: Text prompt to send to the model
            session_id: Optional session ID for tracking

        Returns:
            Generated text response or raises TimeoutError
        """
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(self._make_llm_api_call, prompt)
            try:
                result = future.result(timeout=self.timeout)
                
                # Log successful call
                logger.info(
                    "LLM call completed successfully",
                    session_id=session_id,
                    model=self.model_name,
                    prompt_length=len(prompt),
                    response_length=len(result.text) if hasattr(result, 'text') else 0
                )
                
                return result
                
            except concurrent.futures.TimeoutError:
                # Try to cancel if possible
                future.cancel()
                
                # Log timeout
                logger.error(
                    "LLM request timed out",
                    session_id=session_id,
                    timeout=self.timeout,
                    prompt_length=len(prompt)
                )
                
                raise TimeoutError(f"LLM request timed out after {self.timeout} seconds")

    async def generate_content(
        self, 
        prompt: str, 
        content_type: ContentType = ContentType.QUALIFICATION,
        session_id: str = None,
        item_id: str = None,
        max_retries: int = 3
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
        retry_count = 0
        
        # Create cache key for this request
        cache_key = create_cache_key(prompt, self.model_name, content_type.value)
        
        # Check cache first
        cached_response = get_cached_response(cache_key)
        if cached_response:
            self.cache_hits += 1
            logger.info(
                "Cache hit for LLM request",
                session_id=session_id,
                item_id=item_id,
                content_type=content_type.value,
                cache_key=cache_key[:20] + "..."
            )
            
            # Return cached response with updated metadata
            cached_response["metadata"]["cache_hit"] = True
            cached_response["metadata"]["session_id"] = session_id
            cached_response["metadata"]["item_id"] = item_id
            cached_response["processing_time"] = 0.001  # Minimal cache lookup time
            
            return LLMResponse(**cached_response)
        
        # Cache miss - proceed with LLM call
        self.cache_misses += 1
        logger.debug(
            "Cache miss for LLM request",
            session_id=session_id,
            item_id=item_id,
            content_type=content_type.value
        )
        
        # Update call tracking
        self.call_count += 1
        
        while retry_count <= max_retries:
            try:
                # Apply rate limiting
                await self.rate_limiter.wait_if_needed(self.model_name)
                
                # Log the attempt
                logger.info(
                    "Starting LLM generation",
                    session_id=session_id,
                    item_id=item_id,
                    content_type=content_type.value,
                    prompt_length=len(prompt),
                    retry_count=retry_count
                )

                # Generate content with timeout
                response = self._generate_with_timeout(prompt, session_id)
                
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
                        "retry_count": retry_count,
                        "timestamp": datetime.now().isoformat(),
                        "cache_hit": False
                    }
                )
                
                # Cache the successful response
                cache_data = {
                    "content": response.text,
                    "tokens_used": tokens_used,
                    "processing_time": processing_time,
                    "model_used": self.model_name,
                    "success": True,
                    "error_message": None,
                    "metadata": {
                        "content_type": content_type.value,
                        "retry_count": retry_count,
                        "timestamp": datetime.now().isoformat(),
                        "cache_hit": False
                    }
                }
                set_cached_response(cache_key, cache_data)
                
                # Log successful generation
                logger.info(
                    "LLM generation completed successfully",
                    session_id=session_id,
                    item_id=item_id,
                    processing_time=processing_time,
                    tokens_used=tokens_used,
                    response_length=len(response.text),
                    cached=True
                )
                
                return llm_response

            except Exception as e:
                processing_time = time.time() - start_time
                
                # Check if this is a rate limit error and try fallback key
                error_str = str(e).lower()
                is_rate_limit_error = any(keyword in error_str for keyword in [
                    "rate limit", "quota", "429", "too many requests", "resource_exhausted"
                ])
                
                if is_rate_limit_error and not self.using_fallback:
                    logger.warning(
                        "Rate limit detected, attempting to switch to fallback API key",
                        error=str(e),
                        retry_count=retry_count
                    )
                    
                    if self._switch_to_fallback_key():
                        # Retry with fallback key without incrementing retry count
                        logger.info("Retrying with fallback API key")
                        continue
                
                # Handle error with recovery service
                if self.error_recovery:
                    recovery_action = await self.error_recovery.handle_error(
                        e, item_id or "unknown", content_type, session_id, 
                        retry_count, {"prompt_length": len(prompt)}
                    )
                    
                    # Check if we should retry
                    if recovery_action.strategy.value == "retry" and retry_count < max_retries:
                        retry_count += 1
                        if recovery_action.delay_seconds > 0:
                            await asyncio.sleep(recovery_action.delay_seconds)
                        continue
                    
                    # Use fallback content if available
                    if recovery_action.fallback_content:
                        return LLMResponse(
                            content=recovery_action.fallback_content,
                            tokens_used=0,
                            processing_time=processing_time,
                            model_used=self.model_name,
                            success=False,
                            error_message=str(e),
                            metadata={
                                "session_id": session_id,
                                "item_id": item_id,
                                "content_type": content_type.value,
                                "retry_count": retry_count,
                                "fallback_used": True,
                                "timestamp": datetime.now().isoformat()
                            }
                        )
                
                # Log error
                logger.error(
                    "LLM generation failed",
                    session_id=session_id,
                    item_id=item_id,
                    error=str(e),
                    retry_count=retry_count,
                    processing_time=processing_time
                )
                
                # If we've exhausted retries, return error response
                if retry_count >= max_retries:
                    return LLMResponse(
                        content=f"Failed to generate content after {max_retries} retries: {str(e)}",
                        tokens_used=0,
                        processing_time=processing_time,
                        model_used=self.model_name,
                        success=False,
                        error_message=str(e),
                        metadata={
                            "session_id": session_id,
                            "item_id": item_id,
                            "content_type": content_type.value,
                            "retry_count": retry_count,
                            "timestamp": datetime.now().isoformat()
                        }
                    )
                
                retry_count += 1
    
    def get_service_stats(self) -> Dict[str, Any]:
        """Get service performance statistics including cache metrics."""
        total_requests = self.cache_hits + self.cache_misses
        cache_hit_rate = (self.cache_hits / max(total_requests, 1)) * 100
        
        return {
            "total_calls": self.call_count,
            "total_tokens": self.total_tokens,
            "total_processing_time": self.total_processing_time,
            "average_processing_time": self.total_processing_time / max(self.call_count, 1),
            "model_name": self.model_name,
            "rate_limiter_status": self.rate_limiter.get_status() if hasattr(self.rate_limiter, 'get_status') else None,
            "cache_performance": {
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "total_requests": total_requests,
                "cache_hit_rate_percent": round(cache_hit_rate, 2),
                "cache_size": len(_response_cache)
            }
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
        clear_cache()
        logger.info("LLM service cache cleared")


# Legacy compatibility
class LLM(EnhancedLLMService):
    """Legacy LLM class for backward compatibility."""
    
    def __init__(self, timeout=30, max_requests_per_minute=12):
        # Create a simple rate limiter for legacy compatibility
        from .rate_limiter import RateLimiter
        rate_limiter = RateLimiter()
        super().__init__(timeout=timeout, rate_limiter=rate_limiter)
    
    def generate_content(self, prompt: str) -> str:
        """Legacy method that returns string content directly."""
        import asyncio
        import concurrent.futures
        
        try:
            # Try to get the current event loop
            loop = asyncio.get_running_loop()
            # If we're in an async context, run in a thread pool
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(self._sync_generate_content, prompt)
                return future.result(timeout=self.timeout)
        except RuntimeError:
            # No event loop running, safe to create a new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                response = loop.run_until_complete(
                    super().generate_content(prompt)
                )
                return response.content
            finally:
                loop.close()
    
    def _sync_generate_content(self, prompt: str) -> str:
        """Helper method to run async generate_content in a new thread."""
        import asyncio
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            response = loop.run_until_complete(
                super().generate_content(prompt)
            )
            return response.content
        finally:
            loop.close()


# Global service instance
_llm_service_instance = None

def get_llm_service() -> EnhancedLLMService:
    """Get global LLM service instance."""
    global _llm_service_instance
    if _llm_service_instance is None:
        _llm_service_instance = EnhancedLLMService()
    return _llm_service_instance
