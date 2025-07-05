"""Abstract interface for LLM services."""

from abc import ABC, abstractmethod
from typing import Optional

from src.models.llm_data_models import LLMResponse
from src.models.llm_service_models import LLMApiKeyInfo
from src.models.workflow_models import ContentType


class LLMServiceInterface(ABC):
    """Abstract interface for LLM services that hides implementation details."""

    @abstractmethod
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
        """Generate content using the LLM.
        
        Args:
            prompt: Text prompt to send to the model
            content_type: Type of content being generated
            session_id: Session identifier for tracking
            trace_id: Trace identifier for debugging
            item_id: Item identifier for tracking
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature for generation
            **kwargs: Additional LLM parameters
            
        Returns:
            LLMResponse with generated content and metadata
        """
        raise NotImplementedError

    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Backward-compatible wrapper for generate_content.
        
        Args:
            prompt: Text prompt to send to the model
            **kwargs: Additional arguments
            
        Returns:
            LLMResponse with generated content and metadata
        """
        raise NotImplementedError

    @abstractmethod
    async def validate_api_key(self) -> bool:
        """Validate the current API key.
        
        Returns:
            True if valid, False otherwise
        """
        raise NotImplementedError

    @abstractmethod
    def get_current_api_key_info(self) -> LLMApiKeyInfo:
        """Get information about the currently active API key.
        
        Returns:
            LLMApiKeyInfo with current API key information
        """
        raise NotImplementedError

    @abstractmethod
    async def ensure_api_key_valid(self):
        """Ensure the API key is valid.
        
        Raises:
            ConfigurationError: If API key is invalid
        """
        raise NotImplementedError