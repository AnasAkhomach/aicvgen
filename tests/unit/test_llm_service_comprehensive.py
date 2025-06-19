"""Comprehensive tests for EnhancedLLMService.

Tests the retry consolidation implementation, error handling,
and service resilience patterns as specified in TASK_BLUEPRINT.md.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from google.api_core import exceptions as google_exceptions

from src.services.llm_service import EnhancedLLMService, LLMResponse
from src.utils.exceptions import ConfigurationError, RateLimitError, NetworkError


class TestEnhancedLLMServiceRetries:
    """Test retry behavior and resilience patterns."""

    @pytest.fixture
    def llm_service(self):
        """Create LLM service instance for testing."""
        with patch('src.services.llm_service.genai') as mock_genai:
            mock_model = Mock()
            mock_genai.GenerativeModel.return_value = mock_model
            service = EnhancedLLMService(api_key="test-key")
            service.llm = mock_model
            return service

    @pytest.mark.asyncio
    async def test_successful_generation_no_retries(self, llm_service):
        """Test successful content generation without retries."""
        # Mock successful response
        mock_response = Mock()
        mock_response.text = "Generated content"
        llm_service.llm.generate_content_async = AsyncMock(return_value=mock_response)
        
        result = await llm_service.generate_content(
            prompt="Test prompt",
            session_id="test-session",
            trace_id="test-trace"
        )
        
        assert result.success is True
        assert result.content == "Generated content"
        assert llm_service.llm.generate_content_async.call_count == 1

    @pytest.mark.asyncio
    async def test_transient_error_retries(self, llm_service):
        """Test that transient errors trigger retries."""
        # Mock to fail twice with rate limit, then succeed
        mock_response = Mock()
        mock_response.text = "Generated content"
        
        llm_service.llm.generate_content_async = AsyncMock(
            side_effect=[
                google_exceptions.ResourceExhausted("Rate limit exceeded"),
                google_exceptions.ServiceUnavailable("Service temporarily unavailable"),
                mock_response
            ]
        )
        
        result = await llm_service.generate_content(
            prompt="Test prompt",
            session_id="test-session",
            trace_id="test-trace"
        )
        
        assert result.success is True
        assert result.content == "Generated content"
        assert llm_service.llm.generate_content_async.call_count == 3

    @pytest.mark.asyncio
    async def test_fatal_error_no_retries(self, llm_service):
        """Test that fatal errors are not retried."""
        # Mock to raise a configuration error (fatal)
        llm_service.llm.generate_content_async = AsyncMock(
            side_effect=ConfigurationError("Invalid API key")
        )
        
        with pytest.raises(ConfigurationError, match="Invalid API key"):
            await llm_service.generate_content(
                prompt="Test prompt",
                session_id="test-session",
                trace_id="test-trace"
            )
        
        # Should only be called once (no retries)
        assert llm_service.llm.generate_content_async.call_count == 1

    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self, llm_service):
        """Test behavior when max retries are exceeded."""
        # Mock to always fail with transient error
        llm_service.llm.generate_content_async = AsyncMock(
            side_effect=google_exceptions.ResourceExhausted("Rate limit exceeded")
        )
        
        result = await llm_service.generate_content(
            prompt="Test prompt",
            session_id="test-session",
            trace_id="test-trace"
        )
        
        assert result.success is False
        assert "Rate limit exceeded" in result.error_message
        # Should be called 5 times (initial + 4 retries)
        assert llm_service.llm.generate_content_async.call_count == 5

    @pytest.mark.asyncio
    async def test_timeout_error_retries(self, llm_service):
        """Test that timeout errors are retried."""
        mock_response = Mock()
        mock_response.text = "Generated content"
        
        llm_service.llm.generate_content_async = AsyncMock(
            side_effect=[
                TimeoutError("Request timeout"),
                mock_response
            ]
        )
        
        result = await llm_service.generate_content(
            prompt="Test prompt",
            session_id="test-session",
            trace_id="test-trace"
        )
        
        assert result.success is True
        assert result.content == "Generated content"
        assert llm_service.llm.generate_content_async.call_count == 2


class TestEnhancedLLMServiceConfiguration:
    """Test service configuration and initialization."""

    def test_service_initialization_with_api_key(self):
        """Test successful service initialization with API key."""
        with patch('src.services.llm_service.genai') as mock_genai:
            service = EnhancedLLMService(api_key="test-key")
            assert service.api_key == "test-key"
            mock_genai.configure.assert_called_once_with(api_key="test-key")

    def test_service_initialization_without_api_key(self):
        """Test that service fails fast without API key."""
        with pytest.raises(ConfigurationError, match="GEMINI_API_KEY is required"):
            EnhancedLLMService(api_key=None)

    def test_service_initialization_empty_api_key(self):
        """Test that service fails fast with empty API key."""
        with pytest.raises(ConfigurationError, match="GEMINI_API_KEY is required"):
            EnhancedLLMService(api_key="")


class TestEnhancedLLMServiceErrorHandling:
    """Test error handling and recovery patterns."""

    @pytest.fixture
    def llm_service(self):
        """Create LLM service instance for testing."""
        with patch('src.services.llm_service.genai') as mock_genai:
            mock_model = Mock()
            mock_genai.GenerativeModel.return_value = mock_model
            service = EnhancedLLMService(api_key="test-key")
            service.llm = mock_model
            return service

    @pytest.mark.asyncio
    async def test_network_error_handling(self, llm_service):
        """Test handling of network errors."""
        llm_service.llm.generate_content_async = AsyncMock(
            side_effect=NetworkError("Network connection failed")
        )
        
        result = await llm_service.generate_content(
            prompt="Test prompt",
            session_id="test-session",
            trace_id="test-trace"
        )
        
        assert result.success is False
        assert "Network connection failed" in result.error_message

    @pytest.mark.asyncio
    async def test_rate_limit_error_handling(self, llm_service):
        """Test handling of rate limit errors."""
        llm_service.llm.generate_content_async = AsyncMock(
            side_effect=RateLimitError("Rate limit exceeded")
        )
        
        result = await llm_service.generate_content(
            prompt="Test prompt",
            session_id="test-session",
            trace_id="test-trace"
        )
        
        assert result.success is False
        assert "Rate limit exceeded" in result.error_message

    @pytest.mark.asyncio
    async def test_unexpected_error_handling(self, llm_service):
        """Test handling of unexpected errors."""
        llm_service.llm.generate_content_async = AsyncMock(
            side_effect=ValueError("Unexpected error")
        )
        
        result = await llm_service.generate_content(
            prompt="Test prompt",
            session_id="test-session",
            trace_id="test-trace"
        )
        
        assert result.success is False
        assert "Unexpected error" in result.error_message


class TestEnhancedLLMServiceMetrics:
    """Test service metrics and monitoring."""

    @pytest.fixture
    def llm_service(self):
        """Create LLM service instance for testing."""
        with patch('src.services.llm_service.genai') as mock_genai:
            mock_model = Mock()
            mock_genai.GenerativeModel.return_value = mock_model
            service = EnhancedLLMService(api_key="test-key")
            service.llm = mock_model
            return service

    @pytest.mark.asyncio
    async def test_metrics_collection_success(self, llm_service):
        """Test that metrics are collected for successful calls."""
        mock_response = Mock()
        mock_response.text = "Generated content"
        llm_service.llm.generate_content_async = AsyncMock(return_value=mock_response)
        
        result = await llm_service.generate_content(
            prompt="Test prompt",
            session_id="test-session",
            trace_id="test-trace"
        )
        
        assert result.success is True
        # Verify metrics are available
        metrics = llm_service.get_metrics()
        assert "total_requests" in metrics
        assert "successful_requests" in metrics

    @pytest.mark.asyncio
    async def test_metrics_collection_failure(self, llm_service):
        """Test that metrics are collected for failed calls."""
        llm_service.llm.generate_content_async = AsyncMock(
            side_effect=ValueError("Test error")
        )
        
        result = await llm_service.generate_content(
            prompt="Test prompt",
            session_id="test-session",
            trace_id="test-trace"
        )
        
        assert result.success is False
        # Verify metrics are available
        metrics = llm_service.get_metrics()
        assert "total_requests" in metrics
        assert "failed_requests" in metrics