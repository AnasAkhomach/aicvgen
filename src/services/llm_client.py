import threading
from typing import Any, List

import google.generativeai as genai


class LLMClient:
    """Handles the direct API call to the LLM provider (Gemini).
    
    This class is thread-safe and creates a new GenerativeModel instance
    for each thread to avoid 'Event loop is closed' errors.
    """

    def __init__(self, llm_model: genai.GenerativeModel):
        # Store the model configuration instead of the model instance
        self._api_key = None
        self._model_name = llm_model.model_name
        self._thread_local = threading.local()
        
        # Extract API key from the global configuration
        # This is a workaround since genai doesn't expose the configured API key
        try:
            # Try to get the API key from environment or global config
            import os
            self._api_key = os.getenv('GEMINI_API_KEY')
        except Exception:
            # Fallback: we'll configure when needed
            pass

    def _get_thread_local_model(self) -> genai.GenerativeModel:
        """Get or create a thread-local GenerativeModel instance."""
        if not hasattr(self._thread_local, 'model'):
            # Configure the API key for this thread
            if self._api_key:
                genai.configure(api_key=self._api_key)
            
            # Create a new model instance for this thread
            self._thread_local.model = genai.GenerativeModel(self._model_name)
        
        return self._thread_local.model

    async def generate_content(self, prompt: str, system_instruction: str = None, **kwargs) -> Any:
        """Directly call the LLM provider's API asynchronously."""
        model = self._get_thread_local_model()
        if model is None:
            raise ValueError("LLM model is not initialized.")
        
        # Prepare the content for the API call
        contents = [prompt]
        
        # If system instruction is provided, we need to use the new API format
        if system_instruction:
            # For the new Google GenAI SDK, we need to pass system_instruction as a parameter
            # Note: This requires updating to the new google-genai package
            try:
                # Try the new API format first
                return await model.generate_content_async(
                    contents=contents,
                    system_instruction=system_instruction,
                    **kwargs
                )
            except TypeError:
                # Fallback to old format if new API not available
                # Prepend system instruction to the prompt as a workaround
                enhanced_prompt = f"System: {system_instruction}\n\nUser: {prompt}"
                return await model.generate_content_async(enhanced_prompt)
        else:
            # Standard call without system instruction
            return await model.generate_content_async(prompt)

    async def list_models(self) -> List[Any]:
        """List available models for API key validation."""
        # Use the genai module directly to list models
        # This is a lightweight call that requires authentication
        models = []
        for model in genai.list_models():
            models.append(model)
        return models

    def reconfigure(self, api_key: str) -> None:
        """Reconfigure the client with a new API key."""
        self._api_key = api_key
        genai.configure(api_key=api_key)
        
        # Clear any existing thread-local models so they get recreated
        # with the new API key when accessed
        if hasattr(self._thread_local, 'model'):
            delattr(self._thread_local, 'model')
