"""Abstract interface for LLM clients."""

from abc import ABC, abstractmethod
from typing import Any, List, Optional


class LLMClientInterface(ABC):
    """Abstract interface for LLM clients that hides provider-specific implementation details.

    This interface ensures that the application can work with different LLM providers
    without being tightly coupled to any specific provider's library.
    """

    @abstractmethod
    async def generate_content(
        self, prompt: str, system_instruction: Optional[str] = None, **kwargs
    ) -> Any:
        """Generate content using the LLM provider's API.

        Args:
            prompt: Text prompt to send to the model
            system_instruction: Optional system instruction to guide model behavior
            **kwargs: Additional provider-specific parameters

        Returns:
            Provider-specific response object

        Raises:
            ValueError: If the client is not properly initialized
            Exception: Provider-specific errors
        """
        raise NotImplementedError

    @abstractmethod
    def get_model_name(self) -> str:
        """Get the name of the model being used.

        Returns:
            String identifier of the model
        """
        raise NotImplementedError

    @abstractmethod
    def is_initialized(self) -> bool:
        """Check if the client is properly initialized.

        Returns:
            True if the client is ready to use, False otherwise
        """
        raise NotImplementedError

    @abstractmethod
    async def list_models(self) -> List[Any]:
        """List available models for API key validation.

        Returns:
            List of available models from the provider

        Raises:
            Exception: Provider-specific errors
        """
        raise NotImplementedError

    @abstractmethod
    def reconfigure(self, api_key: str) -> None:
        """Reconfigure the client with a new API key.

        Args:
            api_key: New API key to use

        Raises:
            ValueError: If api_key is empty
        """
        raise NotImplementedError

    @abstractmethod
    def get_langchain_model(self, **kwargs) -> Any:
        """Get a LangChain-compatible model instance.

        Args:
            **kwargs: Model configuration parameters (temperature, max_tokens, etc.)

        Returns:
            LangChain-compatible model instance

        Raises:
            ValueError: If the client is not properly initialized
        """
        raise NotImplementedError
