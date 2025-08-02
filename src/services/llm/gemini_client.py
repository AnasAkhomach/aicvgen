"""Gemini-specific implementation of LLMClientInterface."""

import threading
from typing import Any, List, Optional

import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI

from .llm_client_interface import LLMClientInterface


class GeminiClient(LLMClientInterface):
    """Gemini-specific implementation of LLMClientInterface.

    This class handles direct API calls to Google's Gemini LLM provider.
    It is thread-safe and creates a new GenerativeModel instance
    for each thread to avoid 'Event loop is closed' errors.
    """

    def __init__(self, api_key: str, model_name: str):
        """Initialize the Gemini client with API key and model name.

        Args:
            api_key: Google Gemini API key
            model_name: Name of the Gemini model to use

        Raises:
            ValueError: If api_key or model_name is empty
        """
        if not api_key:
            raise ValueError("API key cannot be empty")
        if not model_name:
            raise ValueError("Model name cannot be empty")

        self._api_key = api_key
        self._model_name = model_name
        self._thread_local = threading.local()
        self._initialized = True

    def _get_thread_local_model(self) -> genai.GenerativeModel:
        """Get or create a thread-local GenerativeModel instance.

        Returns:
            Thread-local GenerativeModel instance

        Raises:
            ValueError: If the client is not initialized
        """
        if not self._initialized:
            raise ValueError("GeminiClient is not initialized")

        if not hasattr(self._thread_local, "model"):
            # Configure the API key for this thread
            genai.configure(api_key=self._api_key)

            # Create a new model instance for this thread
            self._thread_local.model = genai.GenerativeModel(self._model_name)

        return self._thread_local.model

    async def generate_content(
        self, prompt: str, system_instruction: Optional[str] = None, **kwargs
    ) -> Any:
        """Generate content using Gemini's API.

        Args:
            prompt: Text prompt to send to the model
            system_instruction: Optional system instruction to guide model behavior
            **kwargs: Additional Gemini-specific parameters

        Returns:
            Gemini response object

        Raises:
            ValueError: If the client is not initialized or prompt is empty
            Exception: Gemini API-specific errors
        """
        if not prompt:
            raise ValueError("Prompt cannot be empty")

        model = self._get_thread_local_model()

        # Prepare the content for the API call
        contents = [prompt]

        # If system instruction is provided, use the new API format
        if system_instruction:
            try:
                # Try the new API format first
                return await model.generate_content_async(
                    contents=contents, system_instruction=system_instruction, **kwargs
                )
            except TypeError:
                # Fallback to old format if new API not available
                # Prepend system instruction to the prompt
                modified_prompt = f"System: {system_instruction}\n\nUser: {prompt}"
                return await model.generate_content_async(
                    contents=[modified_prompt], **kwargs
                )
        else:
            return await model.generate_content_async(contents=contents, **kwargs)

    def get_model_name(self) -> str:
        """Get the name of the Gemini model being used.

        Returns:
            String identifier of the Gemini model
        """
        return self._model_name

    def is_initialized(self) -> bool:
        """Check if the Gemini client is properly initialized.

        Returns:
            True if the client is ready to use, False otherwise
        """
        return self._initialized and bool(self._api_key) and bool(self._model_name)

    async def list_models(self) -> List[Any]:
        """List available models for API key validation.

        Returns:
            List of available models from Gemini

        Raises:
            ValueError: If the client is not initialized
            Exception: Gemini API-specific errors
        """
        if not self._initialized:
            raise ValueError("GeminiClient is not initialized")

        # Configure the API key for this call
        genai.configure(api_key=self._api_key)

        # Use the genai module directly to list models
        models = []
        for model in genai.list_models():
            models.append(model)
        return models

    def reconfigure(self, api_key: str) -> None:
        """Reconfigure the client with a new API key.

        Args:
            api_key: New API key to use

        Raises:
            ValueError: If api_key is empty
        """
        if not api_key:
            raise ValueError("API key cannot be empty")

        self._api_key = api_key
        genai.configure(api_key=api_key)

        # Clear any existing thread-local models so they get recreated
        # with the new API key when accessed
        if hasattr(self._thread_local, "model"):
            delattr(self._thread_local, "model")

    def get_langchain_model(self, **kwargs) -> ChatGoogleGenerativeAI:
        """Get a LangChain-compatible ChatGoogleGenerativeAI model instance.

        Args:
            **kwargs: Model configuration parameters (temperature, max_tokens, etc.)

        Returns:
            ChatGoogleGenerativeAI instance configured with the provided parameters

        Raises:
            ValueError: If the client is not properly initialized
        """
        if not self.is_initialized():
            raise ValueError("GeminiClient is not initialized")

        # Extract common parameters with defaults
        model = kwargs.get("model", self._model_name)
        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_tokens", 1024)

        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=self._api_key,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
