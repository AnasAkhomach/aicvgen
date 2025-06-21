import sys
import os
import asyncio
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.services.llm_service import AdvancedCache, EnhancedLLMService
from src.models.data_models import ContentType

import pytest


class DummyException(Exception):
    pass


class DummyRateLimit(Exception):
    pass


def test_advanced_cache_set_and_get():
    cache = AdvancedCache(max_size=10)
    prompt = "prompt"
    model = "model"
    response = {"result": 42}
    cache.set(prompt, model, response, ttl_hours=1)
    assert cache.get(prompt, model)["result"] == 42


def test_advanced_cache_expiry():
    cache = AdvancedCache(max_size=10)
    prompt = "prompt"
    model = "model"
    response = {"result": 42}
    cache.set(prompt, model, response, ttl_hours=-1)  # Expired
    assert cache.get(prompt, model) is None


def test_is_retryable_error_rate_limit():
    service = EnhancedLLMService()
    should_retry, delay = service._is_retryable_error(
        DummyRateLimit("rate limit"), 1, 5
    )
    assert should_retry
    assert delay > 0


def test_is_retryable_error_non_retryable():
    service = EnhancedLLMService()
    should_retry, delay = service._is_retryable_error(
        ValueError("invalid api key"), 1, 5
    )
    assert not should_retry
    assert delay == 0


def test_is_retryable_error_max_retries():
    service = EnhancedLLMService()
    should_retry, delay = service._is_retryable_error(
        DummyException("network error"), 5, 5
    )
    assert not should_retry
    assert delay == 0


def test_validate_api_key_uses_executor():
    service = EnhancedLLMService()

    async def run():
        loop = asyncio.get_running_loop()
        with patch.object(
            loop, "run_in_executor", wraps=loop.run_in_executor
        ) as mock_run:
            with patch("src.services.llm_service.genai") as mock_genai:
                mock_genai.list_models.return_value = ["model1", "model2"]
                # Patch self.executor to a MagicMock to check identity
                service.executor = MagicMock()
                await service.validate_api_key()
                # Check that run_in_executor was called with service.executor
                assert any(
                    call_args[0][0] is service.executor
                    for call_args in mock_run.call_args_list
                )

    asyncio.run(run())
