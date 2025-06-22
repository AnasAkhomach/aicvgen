from typing import Any
from tenacity import retry, stop_after_attempt, wait_exponential


class LLMRetryHandler:
    """Wraps an LLM client with retry logic using tenacity."""

    def __init__(self, llm_client, is_retryable_error):
        self.llm_client = llm_client
        self.is_retryable_error = is_retryable_error

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        reraise=True,
    )
    def generate_content(self, prompt: str) -> Any:
        try:
            return self.llm_client.generate_content(prompt)
        except Exception as e:
            should_retry, _ = self.is_retryable_error(e)
            if not should_retry:
                raise
            raise
