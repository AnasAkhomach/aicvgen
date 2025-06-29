import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from src.services.llm_service import EnhancedLLMService
from src.models.data_models import ContentType, LLMResponse
from src.models.llm_service_models import (
    LLMApiKeyInfo,
    LLMServiceStats,
    LLMPerformanceOptimizationResult,
)


class TestEnhancedLLMService:
    """Test cases for EnhancedLLMService (refactored version)."""

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings."""
        settings = MagicMock()
        settings.llm_settings.default_model = "test-model"
        return settings

    @pytest.fixture
    def mock_caching_service(self):
        """Create mock caching service."""
        service = AsyncMock()
        service.initialize = AsyncMock()
        service.check_cache = AsyncMock(return_value=None)
        service.cache_response = AsyncMock()
        service.get_cache_stats = AsyncMock(
            return_value={"size": 10, "hits": 5, "misses": 5, "hit_rate_percent": 50.0}
        )
        service.clear = AsyncMock()
        service.evict_expired_entries = AsyncMock()
        return service

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
    def mock_retry_service(self):
        """Create mock retry service."""
        service = AsyncMock()
        service.generate_content_with_retry = AsyncMock()
        return service

    @pytest.fixture
    def mock_rate_limiter(self):
        """Create mock rate limiter."""
        limiter = AsyncMock()
        limiter.get_status = MagicMock(return_value={"calls": 10, "limit": 100})
        return limiter

    @pytest.fixture
    def llm_service(
        self,
        mock_settings,
        mock_caching_service,
        mock_api_key_manager,
        mock_retry_service,
    ):
        """Create an EnhancedLLMService instance for testing."""
        return EnhancedLLMService(
            settings=mock_settings,
            caching_service=mock_caching_service,
            api_key_manager=mock_api_key_manager,
            retry_service=mock_retry_service,
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
        assert not llm_service._cache_initialized

    def test_initialization_with_optional_params(
        self,
        mock_settings,
        mock_caching_service,
        mock_api_key_manager,
        mock_retry_service,
        mock_rate_limiter,
    ):
        """Test initialization with optional parameters."""
        mock_performance_optimizer = MagicMock()
        mock_async_optimizer = MagicMock()

        service = EnhancedLLMService(
            settings=mock_settings,
            caching_service=mock_caching_service,
            api_key_manager=mock_api_key_manager,
            retry_service=mock_retry_service,
            rate_limiter=mock_rate_limiter,
            performance_optimizer=mock_performance_optimizer,
            async_optimizer=mock_async_optimizer,
        )

        assert service.rate_limiter is mock_rate_limiter
        assert service.performance_optimizer is mock_performance_optimizer
        assert service.async_optimizer is mock_async_optimizer

    @pytest.mark.asyncio
    async def test_ensure_cache_initialized(self, llm_service):
        """Test cache initialization."""
        assert not llm_service._cache_initialized

        await llm_service._ensure_cache_initialized()

        assert llm_service._cache_initialized
        llm_service.caching_service.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_ensure_cache_initialized_already_done(self, llm_service):
        """Test cache initialization when already initialized."""
        llm_service._cache_initialized = True

        await llm_service._ensure_cache_initialized()

        llm_service.caching_service.initialize.assert_not_called()

    @pytest.mark.asyncio
    async def test_ensure_api_key_valid(self, llm_service):
        """Test API key validation delegation."""
        await llm_service.ensure_api_key_valid()

        llm_service.api_key_manager.ensure_api_key_valid.assert_called_once()

    def test_get_current_api_key_info(self, llm_service):
        """Test API key info delegation."""
        result = llm_service.get_current_api_key_info()

        assert isinstance(result, LLMApiKeyInfo)
        llm_service.api_key_manager.get_current_api_key_info.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_content_cache_hit(self, llm_service, sample_llm_response):
        """Test content generation with cache hit."""
        # Configure cache to return a hit
        cached_response = sample_llm_response
        cached_response.metadata["cache_hit"] = True
        llm_service.caching_service.check_cache.return_value = cached_response

        result = await llm_service.generate_content(
            "test prompt", ContentType.CV_ANALYSIS, session_id="123"
        )

        assert result is cached_response
        assert result.metadata["cache_hit"] is True
        # Should not call retry service on cache hit
        llm_service.retry_service.generate_content_with_retry.assert_not_called()
        llm_service.caching_service.cache_response.assert_not_called()

    @pytest.mark.asyncio
    async def test_generate_content_cache_miss(self, llm_service, sample_llm_response):
        """Test content generation with cache miss."""
        # Configure cache to return miss
        llm_service.caching_service.check_cache.return_value = None
        llm_service.retry_service.generate_content_with_retry.return_value = (
            sample_llm_response
        )

        result = await llm_service.generate_content(
            "test prompt", ContentType.CV_ANALYSIS, session_id="123"
        )

        assert result is sample_llm_response

        # Verify the full workflow
        llm_service.caching_service.check_cache.assert_called_once_with(
            "test prompt", "test-model", ContentType.CV_ANALYSIS, session_id="123"
        )
        llm_service.retry_service.generate_content_with_retry.assert_called_once_with(
            "test prompt", ContentType.CV_ANALYSIS, session_id="123"
        )
        llm_service.caching_service.cache_response.assert_called_once_with(
            "test prompt",
            "test-model",
            ContentType.CV_ANALYSIS,
            sample_llm_response,
            session_id="123",
        )

    @pytest.mark.asyncio
    async def test_generate_content_stats_update(
        self, llm_service, sample_llm_response
    ):
        """Test that content generation updates service statistics."""
        llm_service.caching_service.check_cache.return_value = None
        llm_service.retry_service.generate_content_with_retry.return_value = (
            sample_llm_response
        )

        initial_call_count = llm_service.call_count
        initial_total_tokens = llm_service.total_tokens
        initial_processing_time = llm_service.total_processing_time

        await llm_service.generate_content("test prompt", ContentType.CV_ANALYSIS)

        assert llm_service.call_count == initial_call_count + 1
        assert (
            llm_service.total_tokens
            == initial_total_tokens + sample_llm_response.tokens_used
        )
        assert (
            llm_service.total_processing_time
            == initial_processing_time + sample_llm_response.processing_time
        )

    @pytest.mark.asyncio
    async def test_get_service_stats(self, llm_service):
        """Test service statistics collection."""
        # Set up some test data
        llm_service.call_count = 10
        llm_service.total_tokens = 500
        llm_service.total_processing_time = 25.0

        # Mock optimizer stats
        mock_optimizer = MagicMock()
        mock_optimizer.get_comprehensive_stats.return_value = {"optimization": "data"}
        llm_service.performance_optimizer = mock_optimizer

        mock_async_optimizer = MagicMock()
        mock_async_optimizer.get_comprehensive_stats.return_value = {"async": "data"}
        llm_service.async_optimizer = mock_async_optimizer

        # Mock rate limiter
        mock_rate_limiter = MagicMock()
        mock_rate_limiter.get_status.return_value = {"status": "ok"}
        llm_service.rate_limiter = mock_rate_limiter

        result = await llm_service.get_service_stats()

        assert isinstance(result, LLMServiceStats)
        assert result.total_calls == 10
        assert result.total_tokens == 500
        assert result.total_processing_time == 25.0
        assert result.average_processing_time == 2.5
        assert result.model_name == "test-model"
        assert result.rate_limiter_status == {"status": "ok"}
        assert result.cache_stats["hit_rate_percent"] == 50.0
        assert result.optimizer_stats == {"optimization": "data"}
        assert result.async_stats == {"async": "data"}

    @pytest.mark.asyncio
    async def test_get_service_stats_no_calls(self, llm_service):
        """Test service statistics when no calls have been made."""
        result = await llm_service.get_service_stats()

        assert result.total_calls == 0
        assert result.average_processing_time == 0.0

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

    @pytest.mark.asyncio
    async def test_clear_cache(self, llm_service):
        """Test cache clearing."""
        await llm_service.clear_cache()

        llm_service.caching_service.clear.assert_called_once()

    @pytest.mark.asyncio
    async def test_optimize_performance(self, llm_service):
        """Test performance optimization."""
        # Mock cache stats before and after
        llm_service.caching_service.get_cache_stats.side_effect = [
            {"size": 100},  # Before
            {"size": 80},  # After
        ]

        result = await llm_service.optimize_performance()

        assert isinstance(result, LLMPerformanceOptimizationResult)
        assert result.cache_optimization["entries_before"] == 100
        assert result.cache_optimization["entries_after"] == 80
        assert result.cache_optimization["entries_removed"] == 20

        llm_service.caching_service.evict_expired_entries.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_backward_compatibility(
        self, llm_service, sample_llm_response
    ):
        """Test backward-compatible generate method."""
        llm_service.caching_service.check_cache.return_value = None
        llm_service.retry_service.generate_content_with_retry.return_value = (
            sample_llm_response
        )

        # Test that generate() calls generate_content()
        result = await llm_service.generate(
            "test prompt", ContentType.CV_ANALYSIS, session_id="123"
        )

        assert result is sample_llm_response
        llm_service.retry_service.generate_content_with_retry.assert_called_once()

    @pytest.mark.asyncio
    async def test_error_propagation(self, llm_service):
        """Test that errors from underlying services are properly propagated."""
        from src.error_handling.exceptions import ConfigurationError

        llm_service.caching_service.check_cache.return_value = None
        llm_service.retry_service.generate_content_with_retry.side_effect = (
            ConfigurationError("Test error")
        )

        with pytest.raises(ConfigurationError, match="Test error"):
            await llm_service.generate_content("test prompt", ContentType.CV_ANALYSIS)

    @pytest.mark.asyncio
    async def test_integration_workflow(self, llm_service, sample_llm_response):
        """Test the complete integration workflow."""
        # Setup: cache miss, successful generation, successful caching
        llm_service.caching_service.check_cache.return_value = None
        llm_service.retry_service.generate_content_with_retry.return_value = (
            sample_llm_response
        )

        # Execute
        result = await llm_service.generate_content(
            "integration test prompt",
            ContentType.CV_ANALYSIS,
            session_id="integration_123",
            item_id="item_456",
        )

        # Verify complete workflow
        assert result is sample_llm_response

        # Verify cache was checked
        llm_service.caching_service.check_cache.assert_called_once_with(
            "integration test prompt",
            "test-model",
            ContentType.CV_ANALYSIS,
            session_id="integration_123",
            item_id="item_456",
        )

        # Verify retry service was called
        llm_service.retry_service.generate_content_with_retry.assert_called_once_with(
            "integration test prompt",
            ContentType.CV_ANALYSIS,
            session_id="integration_123",
            item_id="item_456",
        )

        # Verify response was cached
        llm_service.caching_service.cache_response.assert_called_once_with(
            "integration test prompt",
            "test-model",
            ContentType.CV_ANALYSIS,
            sample_llm_response,
            session_id="integration_123",
            item_id="item_456",
        )

        # Verify stats were updated
        assert llm_service.call_count == 1
        assert llm_service.total_tokens == sample_llm_response.tokens_used
        assert llm_service.total_processing_time == sample_llm_response.processing_time
