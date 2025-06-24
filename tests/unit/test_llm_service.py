import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pytest
from unittest.mock import AsyncMock, MagicMock
import asyncio
from src.services.llm_service import EnhancedLLMService
from src.utils.exceptions import OperationTimeoutError
from src.models.data_models import ContentType


@pytest.fixture
def mock_llm_retry_handler():
    handler = AsyncMock()
    handler.generate_content = AsyncMock()
    return handler


@pytest.fixture
def mock_llm_client():
    return MagicMock()


@pytest.fixture
def mock_settings():
    class Dummy:
        class LLMSettings:
            default_model = "test-model"

        class LLM:
            gemini_api_key_primary = "key"
            gemini_api_key_fallback = "fallback"

        llm_settings = LLMSettings()
        llm = LLM()

    return Dummy()


@pytest.fixture
def mock_cache():
    cache = MagicMock()
    cache.get.return_value = None
    return cache


@pytest.mark.asyncio
async def test_generate_content_success(
    mock_settings, mock_llm_client, mock_llm_retry_handler, mock_cache
):
    class DummyResponse:
        text = "LLM output"

    mock_llm_retry_handler.generate_content.return_value = DummyResponse()
    service = EnhancedLLMService(
        settings=mock_settings,
        llm_client=mock_llm_client,
        llm_retry_handler=mock_llm_retry_handler,
        cache=mock_cache,
        timeout=2,
    )
    result = await service.generate_content(
        "prompt", content_type=ContentType.QUALIFICATION
    )
    assert result is not None
    mock_llm_retry_handler.generate_content.assert_awaited_once()


@pytest.mark.asyncio
async def test_generate_content_timeout(
    mock_settings, mock_llm_client, mock_llm_retry_handler, mock_cache
):
    async def slow_generate_content(prompt):
        await asyncio.sleep(3)

    mock_llm_retry_handler.generate_content.side_effect = slow_generate_content
    service = EnhancedLLMService(
        settings=mock_settings,
        llm_client=mock_llm_client,
        llm_retry_handler=mock_llm_retry_handler,
        cache=mock_cache,
        timeout=1,
    )
    with pytest.raises(OperationTimeoutError):
        await service.generate_content("prompt", content_type=ContentType.QUALIFICATION)
    mock_llm_retry_handler.generate_content.assert_awaited_once()
