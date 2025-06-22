"""Unit tests for consolidated caching and retry logic (DP-01)."""

import sys
import os

# Ensure project root (containing src/) is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from src.services.llm_service import EnhancedLLMService
from src.models.data_models import ContentType
from src.models.data_models import LLMResponse
import concurrent.futures


class TestConsolidatedCaching:
    """Test consolidated caching functionality."""

    @pytest.fixture
    def llm_service(self):
        """Create an EnhancedLLMService instance for testing."""
        with patch("src.services.llm_service.genai.configure"):
            from src.config.settings import AppConfig
            from src.core.performance_optimizer import get_performance_optimizer

            mock_settings = AppConfig()
            # Provide a valid cache and performance optimizer
            perf_optimizer = get_performance_optimizer()
            service = EnhancedLLMService(
                settings=mock_settings,
                user_api_key="test_key",
                timeout=30,
                cache=perf_optimizer.cache,
                performance_optimizer=perf_optimizer,
            )
            return service

    def test_legacy_cache_functions_removed(self, llm_service):
        """Test that legacy cache functions are no longer available."""
        # Test that the service doesn't have legacy cache attributes
        import src.services.llm_service as llm_module

        # These functions should not exist in the module
        assert not hasattr(llm_module, "get_cached_response")
        assert not hasattr(llm_module, "set_cached_response")
        assert not hasattr(llm_module, "clear_cache")
        assert not hasattr(llm_module, "_response_cache")

    def test_is_transient_error_removed(self, llm_service):
        """Test that is_transient_error function is no longer available."""
        import src.services.llm_service as llm_module

        assert not hasattr(llm_module, "is_transient_error")

    @pytest.mark.asyncio
    async def test_generate_with_timeout_uses_asyncio_wait_for(self, llm_service):
        """Test that _generate_with_timeout properly uses asyncio.wait_for."""
        # Mock the LLM response
        mock_response = Mock()
        mock_response.text = "Test response"

        with patch.object(
            llm_service, "_make_llm_api_call", return_value=mock_response
        ) as mock_call:
            with patch("asyncio.wait_for") as mock_wait_for:
                mock_wait_for.return_value = mock_response

                result = await llm_service._generate_with_timeout(
                    "test prompt", session_id="test_session"
                )

                # Verify asyncio.wait_for was called
                mock_wait_for.assert_called_once()
                assert result == mock_response

    @pytest.mark.asyncio
    async def test_generate_with_timeout_handles_timeout(self, llm_service):
        """Test that _generate_with_timeout properly handles timeouts."""
        with patch.object(llm_service, "_make_llm_api_call") as mock_call:
            with patch(
                "asyncio.wait_for", side_effect=asyncio.TimeoutError()
            ) as mock_wait_for:
                from src.utils.exceptions import OperationTimeoutError

                with pytest.raises(
                    OperationTimeoutError, match="LLM request timed out"
                ):
                    await llm_service._generate_with_timeout(
                        "test prompt", session_id="test_session"
                    )

    def test_should_retry_exception_logic(self, llm_service):
        """Test the consolidated retry exception logic."""
        # Test retryable exceptions
        retryable_error = ConnectionError("Network error")
        should_retry, _ = llm_service._is_retryable_error(retryable_error)
        assert should_retry is True

        # Test non-retryable exceptions
        non_retryable_error = ValueError("Invalid API key")
        should_retry, _ = llm_service._is_retryable_error(non_retryable_error)
        assert should_retry is False

        # Test rate limit patterns
        rate_limit_error = Exception("Rate limit exceeded")
        should_retry, _ = llm_service._is_retryable_error(rate_limit_error)
        assert should_retry is True

    def test_should_retry_with_delay_logic(self, llm_service):
        """Test the consolidated retry with delay logic."""
        # Use the actual retry logic method (_is_retryable_error)
        error = ConnectionError("Network error")
        should_retry, delay = llm_service._is_retryable_error(error, 1, 3)
        assert should_retry is True
        assert delay > 0

        # Test max retries exceeded
        should_retry, delay = llm_service._is_retryable_error(error, 3, 3)
        assert should_retry is False
        assert delay == 0.0

        # Test non-retryable error
        non_retryable_error = ValueError("Invalid API key")
        should_retry, delay = llm_service._is_retryable_error(non_retryable_error, 1, 3)
        assert should_retry is False
        assert delay == 0.0

    def test_advanced_cache_integration(self, llm_service):
        """Test that AdvancedCache integration is properly set up."""
        # Verify that the service has the cache attribute
        assert hasattr(llm_service, "cache")
        assert llm_service.cache is not None

        # Verify that performance optimizer cache is available
        assert hasattr(llm_service, "performance_optimizer")
        assert hasattr(llm_service.performance_optimizer, "cache")

        # Verify cache tracking attributes exist
        assert hasattr(llm_service, "cache_hits")
        assert hasattr(llm_service, "cache_misses")
        assert llm_service.cache_hits == 0
        assert llm_service.cache_misses == 0
