from typing import Any

from tenacity import retry, stop_after_attempt, wait_exponential

from src.constants.config_constants import ConfigConstants
from src.error_handling.classification import is_retryable_error


class LLMRetryHandler:
    """Wraps an LLM client with retry logic using tenacity."""

    def __init__(self, llm_client):
        self.llm_client = llm_client

    @retry(
        stop=stop_after_attempt(ConfigConstants.LLM_RETRY_MAX_ATTEMPTS),
        wait=wait_exponential(
            multiplier=ConfigConstants.LLM_RETRY_MULTIPLIER,
            min=ConfigConstants.LLM_RETRY_MIN_WAIT,
            max=ConfigConstants.LLM_RETRY_MAX_WAIT
        ),
        retry=lambda exc: is_retryable_error(exc)[0],
        reraise=True,
    )
    async def generate_content(self, prompt: str, **kwargs) -> Any:
        return await self.llm_client.generate_content(prompt, **kwargs)
