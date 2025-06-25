from typing import Any
from tenacity import retry, stop_after_attempt, wait_exponential
import asyncio

from ..error_handling.classification import is_retryable_error


class LLMRetryHandler:
    """Wraps an LLM client with retry logic using tenacity."""

    def __init__(self, llm_client):
        self.llm_client = llm_client

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        retry=lambda exc: is_retryable_error(exc)[0],
        reraise=True,
    )
    async def generate_content(self, prompt: str) -> Any:
        return await self.llm_client.generate_content(prompt)
