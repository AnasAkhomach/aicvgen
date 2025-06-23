from typing import Any
import google.generativeai as genai


class LLMClient:
    """Handles the direct API call to the LLM provider (Gemini)."""

    def __init__(self, llm_model: genai.GenerativeModel):
        self.llm = llm_model

    async def generate_content(self, prompt: str) -> Any:
        """Directly call the LLM provider's API asynchronously."""
        if self.llm is None:
            raise ValueError("LLM model is not initialized.")
        return await self.llm.generate_content_async(prompt)
