import pytest
import asyncio
from unittest.mock import patch, MagicMock

from src.services.llm_client import LLMClient
from src.services.llm_retry_handler import LLMRetryHandler
from src.services.llm_service import AdvancedCache, EnhancedLLMService
from src.models.data_models import ContentType


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


def make_settings_mock():
    mock = MagicMock()
    mock.llm.gemini_api_key_primary = "primary"
    mock.llm.gemini_api_key_fallback = "fallback"
    mock.llm_settings.default_model = "gemini-pro"
    return mock


# Patch all EnhancedLLMService instantiations to use settings mock
def test_is_retryable_error_rate_limit():
    service = EnhancedLLMService(settings=make_settings_mock())
    should_retry, delay = service._is_retryable_error(
        DummyRateLimit("rate limit"), 1, 5
    )
    assert should_retry
    assert delay > 0


def test_is_retryable_error_non_retryable():
    service = EnhancedLLMService(settings=make_settings_mock())
    should_retry, delay = service._is_retryable_error(
        ValueError("invalid api key"), 1, 5
    )
    assert not should_retry
    assert delay == 0


def test_is_retryable_error_max_retries():
    service = EnhancedLLMService(settings=make_settings_mock())
    should_retry, delay = service._is_retryable_error(
        DummyException("network error"), 5, 5
    )
    assert not should_retry
    assert delay == 0


def test_validate_api_key_uses_executor():
    service = EnhancedLLMService(settings=make_settings_mock())

    async def run():
        loop = asyncio.get_running_loop()
        with patch.object(
            loop, "run_in_executor", wraps=loop.run_in_executor
        ) as mock_run:
            with patch("src.services.llm_service.genai") as mock_genai:
                mock_genai.list_models.return_value = ["model1", "model2"]
                service.executor = MagicMock()
                await service.validate_api_key()
                assert any(
                    call_args[0][0] is service.executor
                    for call_args in mock_run.call_args_list
                )

    asyncio.run(run())


def test_llmclient_and_retryhandler():
    class DummyLLM:
        def __init__(self):
            self.called = False

        def generate_content(self, prompt):
            self.called = True
            if prompt == "fail":
                raise RuntimeError("retryable error")
            return type("Resp", (), {"text": f"Echo: {prompt}"})()

    def dummy_is_retryable_error(e, *_):
        return (True, 0.1)

    llm = DummyLLM()
    client = LLMClient(llm)
    retry_handler = LLMRetryHandler(client, dummy_is_retryable_error)
    # Should succeed
    resp = retry_handler.generate_content("hello")
    assert resp.text == "Echo: hello"
    # Should retry and eventually raise
    with pytest.raises(Exception):
        retry_handler.generate_content("fail")
