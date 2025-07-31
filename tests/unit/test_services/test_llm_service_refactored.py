import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.models.llm_data_models import LLMResponse
from src.models.llm_service_models import (
    LLMApiKeyInfo,
    LLMPerformanceOptimizationResult,
    LLMServiceStats,
)
from src.models.workflow_models import ContentType
from src.services.llm_service import EnhancedLLMService


class TestEnhancedLLMService:
    """Test cases for EnhancedLLMService (refactored version)."""

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings."""
        settings = MagicMock()
        settings.llm_settings.default_model = "test-model"
        return settings

    @pytest.fixture
    def mock_api_key_manager(self):
        """Create mock API key manager."""
        manager = AsyncMock()
        manager.ensure_api_key_valid = AsyncMock()
        manager.get_current_api_key_info = MagicMock(
            return_value=LLMApiKeyInfo(
                using_user_key=False,
                using_fallback=False,
                has_fallback_available=True,
                key_source="primary",
            )
        )
        return manager

    @pytest.fixture
    def mock_llm_client(self):
        """Create mock LLM client."""
        client = AsyncMock()
        client.generate_content = AsyncMock()
        return client

    @pytest.fixture
    def llm_service(
        self,
        mock_settings,
        mock_api_key_manager,
        mock_llm_client,
    ):
        """Create an EnhancedLLMService instance for testing."""
        return EnhancedLLMService(
            settings=mock_settings,
            llm_client=mock_llm_client,
            api_key_manager=mock_api_key_manager,
        )

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
                "content_type": "cv_analysis",
                "timestamp": datetime.now().isoformat(),
                "cache_hit": False,
            },
        )

    def test_initialization(self, llm_service, mock_settings):
        """Test service initialization."""
        assert llm_service.settings is mock_settings
        assert llm_service.model_name == "test-model"
        assert llm_service.call_count == 0
        assert llm_service.total_tokens == 0
        assert llm_service.total_processing_time == 0.0

    def test_initialization_with_optional_params(
        self,
        mock_settings,
        mock_api_key_manager,
        mock_llm_client,
    ):
        """Test initialization with optional cache parameter."""
        from langchain_core.caches import InMemoryCache

        cache = InMemoryCache()
        service = EnhancedLLMService(
            settings=mock_settings,
            llm_client=mock_llm_client,
            api_key_manager=mock_api_key_manager,
            cache=cache,
        )

        assert service.cache is cache

    # Cache initialization tests removed - caching is now handled internally

    @pytest.mark.asyncio
    async def test_ensure_api_key_valid(self, llm_service):
        """Test API key validation delegation."""
        await llm_service.ensure_api_key_valid()

        llm_service.api_key_manager.ensure_api_key_valid.assert_called_once()

    def test_get_current_api_key_info(self, llm_service):
        """Test API key info delegation."""
        result = llm_service.get_current_api_key_info()

        assert isinstance(result, LLMApiKeyInfo)
        llm_service.api_key_manager.get_current_api_key_info.assert_called()

    # Cache hit test removed - caching is now handled internally by LangChain

    # Cache miss test removed - caching is now handled internally by LangChain

    # Stats update test removed - will be replaced with simpler test

    # Note: Removed tests for get_service_stats, clear_cache, and optimize_performance
    # as these methods are now private implementation details and should not be
    # exposed through the public interface (CB-011 contract breach fix)

    def test_reset_stats(self, llm_service):
        """Test statistics reset."""
        # Set some data
        llm_service.call_count = 10
        llm_service.total_tokens = 500
        llm_service.total_processing_time = 25.0

        llm_service.reset_stats()

        assert llm_service.call_count == 0
        assert llm_service.total_tokens == 0
        assert llm_service.total_processing_time == 0.0

    # Note: Tests for clear_cache and optimize_performance removed as these
    # are now private implementation details (CB-011 contract breach fix)

    # Backward compatibility test removed - old services no longer exist

    # Error propagation test removed - old services no longer exist

    # CB004 test removed - old services no longer exist

    # Integration workflow test removed - old services no longer exist
