"""Tests for RefactoredLLMService using tenacity and LangChain caching."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Any

from langchain_core.caches import InMemoryCache
from langchain_community.cache import SQLiteCache
from tenacity import RetryError

from src.services.llm_service_refactored import RefactoredLLMService
from src.models.workflow_models import ContentType
from src.models.llm_data_models import LLMResponse
from src.models.llm_service_models import LLMApiKeyInfo, LLMServiceStats
from src.error_handling.exceptions import (
    NetworkError,
    RateLimitError,
    OperationTimeoutError,
    ConfigurationError,
)


class TestRefactoredLLMService:
    """Test suite for RefactoredLLMService."""

    @pytest.fixture
    def mock_settings(self):
        """Mock settings fixture."""
        settings = MagicMock()
        settings.llm_settings.default_model = "gemini-pro"
        return settings

    @pytest.fixture
    def mock_llm_client(self):
        """Mock LLM client fixture."""
        client = AsyncMock()
        client.generate_content = AsyncMock()
        return client

    @pytest.fixture
    def mock_api_key_manager(self):
        """Mock API key manager fixture."""
        manager = AsyncMock()
        manager.ensure_api_key_valid = AsyncMock(return_value=True)
        manager.validate_api_key = AsyncMock(return_value=True)
        manager.get_current_api_key_info = AsyncMock(
            return_value=LLMApiKeyInfo(
                using_user_key=False,
                using_fallback=False,
                has_fallback_available=True,
                key_source="primary",
            )
        )
        return manager

    @pytest.fixture
    def mock_response(self):
        """Mock LLM response fixture."""
        response = MagicMock()
        response.text = "Generated content"
        response.tokens = 50
        return response

    @pytest.fixture
    def sample_llm_response(self):
        """Create a sample LLM response."""
        return LLMResponse(
            content="Test response content",
            tokens_used=50,
            processing_time=1.5,
            model_used="test-model",
            success=True,
            metadata={
                "session_id": "123",
                "item_id": "456",
                "cache_hit": False,
                "content_type": ContentType.CV_ANALYSIS,
            },
        )

    @pytest.fixture
    def refactored_service(self, mock_settings, mock_llm_client, mock_api_key_manager):
        """RefactoredLLMService fixture."""
        with patch("src.services.llm_service_refactored.set_llm_cache"):
            service = RefactoredLLMService(
                settings=mock_settings,
                llm_client=mock_llm_client,
                api_key_manager=mock_api_key_manager,
                cache_type="memory",
                timeout=2,  # Short timeout for faster tests
                max_retries=3,
            )
        return service

    def test_initialization_with_memory_cache(
        self, mock_settings, mock_llm_client, mock_api_key_manager
    ):
        """Test service initialization with memory cache."""
        with patch(
            "src.services.llm_service_refactored.set_llm_cache"
        ) as mock_set_cache:
            service = RefactoredLLMService(
                settings=mock_settings,
                llm_client=mock_llm_client,
                api_key_manager=mock_api_key_manager,
                cache_type="memory",
            )

            assert service.model_name == "gemini-pro"
            assert service.timeout == 30  # default
            assert service.max_retries == 3  # default
            assert service._cache_initialized is True
            mock_set_cache.assert_called_once()

    def test_initialization_with_sqlite_cache(
        self, mock_settings, mock_llm_client, mock_api_key_manager
    ):
        """Test service initialization with SQLite cache."""
        with patch(
            "src.services.llm_service_refactored.set_llm_cache"
        ) as mock_set_cache:
            service = RefactoredLLMService(
                settings=mock_settings,
                llm_client=mock_llm_client,
                api_key_manager=mock_api_key_manager,
                cache_type="sqlite",
                cache_database_path="test.db",
            )

            assert service._cache_initialized is True
            mock_set_cache.assert_called_once()

    def test_cache_initialization_failure(
        self, mock_settings, mock_llm_client, mock_api_key_manager
    ):
        """Test graceful handling of cache initialization failure."""
        with patch(
            "src.services.llm_service_refactored.set_llm_cache",
            side_effect=Exception("Cache error"),
        ):
            service = RefactoredLLMService(
                settings=mock_settings,
                llm_client=mock_llm_client,
                api_key_manager=mock_api_key_manager,
                cache_type="memory",
            )

            assert service._cache_initialized is False

    @pytest.mark.asyncio
    async def test_successful_content_generation(
        self, refactored_service, mock_response
    ):
        """Test successful content generation."""
        refactored_service.llm_client.generate_content.return_value = mock_response

        result = await refactored_service.generate_content(
            prompt="Test prompt",
            content_type=ContentType.CV_ANALYSIS,
        )

        assert isinstance(result, LLMResponse)
        assert result.content == "Generated content"
        assert result.tokens_used == 50
        assert result.metadata.get("content_type") == ContentType.CV_ANALYSIS
        assert result.model_used == "gemini-pro"
        assert refactored_service.call_count == 1
        assert refactored_service.total_tokens == 50

    @pytest.mark.asyncio
    async def test_generate_simple_text(self, refactored_service, mock_response):
        """Test simple text generation."""
        refactored_service.llm_client.generate_content.return_value = mock_response

        result = await refactored_service.generate(
            prompt="Test prompt",
        )

        assert result == "Generated content"

    @pytest.mark.asyncio
    async def test_retry_on_rate_limit_error(self, refactored_service):
        """Test retry behavior on rate limit errors."""
        # First two calls fail with rate limit, third succeeds
        mock_response = MagicMock()
        mock_response.text = "Success after retry"
        mock_response.tokens = 25

        refactored_service.llm_client.generate_content.side_effect = [
            Exception("rate limit exceeded"),
            Exception("rate limit exceeded"),
            mock_response,
        ]

        result = await refactored_service.generate_content(
            prompt="Test prompt",
        )

        assert result.content == "Success after retry"
        assert refactored_service.llm_client.generate_content.call_count == 3

    @pytest.mark.asyncio
    async def test_retry_on_network_error(self, refactored_service):
        """Test retry behavior on network errors."""
        mock_response = MagicMock()
        mock_response.text = "Success after network retry"
        mock_response.tokens = 30

        refactored_service.llm_client.generate_content.side_effect = [
            Exception("network connection failed"),
            mock_response,
        ]

        result = await refactored_service.generate_content(
            prompt="Test prompt",
        )

        assert result.content == "Success after network retry"
        assert refactored_service.llm_client.generate_content.call_count == 2

    @pytest.mark.asyncio
    async def test_timeout_handling(self, refactored_service):
        """Test timeout handling."""

        # Mock a slow response that times out
        async def slow_response(*args, **kwargs):
            await asyncio.sleep(3)  # Longer than 2-second timeout
            return MagicMock()

        refactored_service.llm_client.generate_content.side_effect = slow_response

        with pytest.raises(OperationTimeoutError):
            await refactored_service.generate_content(
                prompt="Test prompt",
            )

    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self, refactored_service):
        """Test behavior when max retries are exceeded."""
        refactored_service.llm_client.generate_content.side_effect = Exception(
            "rate limit exceeded"
        )

        with pytest.raises(Exception):  # Should eventually raise the original exception
            await refactored_service.generate_content(
                prompt="Test prompt",
            )

        # Should have tried max_retries + 1 times (initial + retries)
        assert refactored_service.llm_client.generate_content.call_count >= 3

    @pytest.mark.asyncio
    async def test_non_retryable_error(self, refactored_service):
        """Test that non-retryable errors are not retried."""
        refactored_service.llm_client.generate_content.side_effect = ValueError(
            "Invalid input"
        )

        with pytest.raises(ValueError):
            await refactored_service.generate_content(
                prompt="Test prompt",
            )

        # Should only be called once (no retries for non-retryable errors)
        assert refactored_service.llm_client.generate_content.call_count == 1

    @pytest.mark.asyncio
    async def test_structured_content_generation_with_langchain(
        self, refactored_service
    ):
        """Test structured content generation using LangChain's with_structured_output."""
        from pydantic import BaseModel

        class TestModel(BaseModel):
            name: str
            value: int

        # Mock LangChain's structured output
        mock_structured_llm = AsyncMock()
        mock_structured_llm.ainvoke.return_value = TestModel(name="test", value=42)

        refactored_service.llm_client.with_structured_output = MagicMock(
            return_value=mock_structured_llm
        )

        result = await refactored_service.generate_structured_content(
            prompt="Generate structured data",
            response_model=TestModel,
        )

        assert isinstance(result, TestModel)
        assert result.name == "test"
        assert result.value == 42
        refactored_service.llm_client.with_structured_output.assert_called_once_with(
            TestModel
        )

    @pytest.mark.asyncio
    async def test_structured_content_generation_fallback(
        self, refactored_service, mock_response
    ):
        """Test structured content generation fallback when LangChain method not available."""
        from pydantic import BaseModel

        class TestModel(BaseModel):
            name: str
            value: int

        # Mock response with JSON content
        mock_response.text = '{"name": "fallback", "value": 123}'
        refactored_service.llm_client.generate_content.return_value = mock_response

        # Ensure with_structured_output is not available
        if hasattr(refactored_service.llm_client, "with_structured_output"):
            delattr(refactored_service.llm_client, "with_structured_output")

        result = await refactored_service.generate_structured_content(
            prompt="Generate structured data",
            response_model=TestModel,
        )

        assert isinstance(result, TestModel)
        assert result.name == "fallback"
        assert result.value == 123

    @pytest.mark.asyncio
    async def test_api_key_validation(self, refactored_service):
        """Test API key validation."""
        result = await refactored_service.validate_api_key("test-key")
        assert result is True
        refactored_service.api_key_manager.validate_api_key.assert_called_once_with(
            "test-key"
        )

    @pytest.mark.asyncio
    async def test_get_current_api_key_info(self, refactored_service):
        """Test getting current API key info."""
        result = await refactored_service.get_current_api_key_info()
        assert isinstance(result, LLMApiKeyInfo)
        assert result.key_source == "primary"
        assert result.using_user_key is False

    def test_get_stats(self, refactored_service):
        """Test getting service statistics."""
        # Set some test data
        refactored_service.call_count = 5
        refactored_service.total_tokens = 250
        refactored_service.total_processing_time = 10.0
        refactored_service.cache_hits = 2
        refactored_service.cache_misses = 3

        stats = refactored_service.get_stats()

        assert isinstance(stats, LLMServiceStats)
        assert stats.total_calls == 5
        assert stats.total_tokens == 250
        assert stats.total_processing_time == 10.0
        assert stats.cache_stats["cache_hits"] == 2
        assert stats.cache_stats["cache_misses"] == 3
        assert stats.average_processing_time == 2.0
        assert stats.model_name == "gemini-pro"

    def test_reset_stats(self, refactored_service):
        """Test resetting service statistics."""
        # Set some test data
        refactored_service.call_count = 5
        refactored_service.total_tokens = 250
        refactored_service.total_processing_time = 10.0

        refactored_service.reset_stats()

        assert refactored_service.call_count == 0
        assert refactored_service.total_tokens == 0
        assert refactored_service.total_processing_time == 0.0
        assert refactored_service.cache_hits == 0
        assert refactored_service.cache_misses == 0
