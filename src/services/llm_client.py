from typing import Any


class LLMClient:
    """Handles the direct API call to the LLM provider (Gemini)."""

    def __init__(self, llm):
        self.llm = llm

    def generate_content(self, prompt: str) -> Any:
        """Directly call the LLM provider's API."""
        if self.llm is None:
            raise ValueError("LLM model is not initialized.")
        return self.llm.generate_content(prompt)
