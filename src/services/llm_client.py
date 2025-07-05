from typing import Any, List

import google.generativeai as genai


class LLMClient:
    """Handles the direct API call to the LLM provider (Gemini)."""

    def __init__(self, llm_model: genai.GenerativeModel):
        self.llm = llm_model

    async def generate_content(self, prompt: str, **kwargs) -> Any:
        """Directly call the LLM provider's API asynchronously."""
        if self.llm is None:
            raise ValueError("LLM model is not initialized.")
        # Note: kwargs like max_tokens, temperature are ignored as Gemini API
        # only accepts prompt parameter through generate_content_async
        return await self.llm.generate_content_async(prompt)

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
        genai.configure(api_key=api_key)
        # Note: The llm_model itself doesn't need to be recreated
        # as it will use the newly configured API key
