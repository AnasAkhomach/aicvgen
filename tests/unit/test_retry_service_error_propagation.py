"""Test for CB-008 fix: Retry service error propagation contract compliance."""

import asyncio
import pytest
from unittest.mock import AsyncMock, Mock

from src.error_handling.exceptions import ConfigurationError, NetworkError, RateLimitError
from src.models.workflow_models import ContentType
from src.services.llm_retry_service import LLMRetryService
from src.services.llm_api_key_manager import LLMApiKeyManager
from src.services.llm_retry_handler import LLMRetryHandler


class TestCB008ErrorPropagationFix:
    """Test that the retry service properly propagates exceptions according to contract."""

    @pytest.fixture
    def mock_retry_handler(self):
        """Create a mock retry handler."""
        handler = Mock(spec=LLMRetryHandler)
        handler.generate_content = AsyncMock()
        return handler

    @pytest.fixture
    def mock_api_key_manager(self):
        """Create a mock API key manager."""
        manager = Mock(spec=LLMApiKeyManager)
        manager.has_fallback_available = Mock(return_value=False)
        return manager

    @pytest.fixture
    def mock_error_recovery(self):
        """Create a mock error recovery service."""
        recovery = Mock()
        recovery.get_fallback_content = AsyncMock()
        return recovery

    @pytest.fixture
    def retry_service(self, mock_retry_handler, mock_api_key_manager, mock_error_recovery):
        """Create a retry service with mocked dependencies."""
        return LLMRetryService(
            llm_retry_handler=mock_retry_handler,
            api_key_manager=mock_api_key_manager,
            error_recovery=mock_error_recovery,
            timeout=30,
            model_name="test-model"
        )

    @pytest.mark.asyncio
    async def test_configuration_error_propagated(self, retry_service, mock_retry_handler):
        """Test that ConfigurationError is properly propagated."""
        # Arrange
        config_error = ConfigurationError("Invalid configuration")
        mock_retry_handler.generate_content.side_effect = config_error

        # Act & Assert
        with pytest.raises(ConfigurationError, match="Invalid configuration"):
            await retry_service.generate_content_with_retry(
                "test prompt", ContentType.CV_ANALYSIS
            )

    @pytest.mark.asyncio
    async def test_rate_limit_error_propagated(self, retry_service, mock_retry_handler):
        """Test that RateLimitError is properly propagated."""
        # Arrange
        rate_limit_error = RateLimitError("Rate limit exceeded")
        mock_retry_handler.generate_content.side_effect = rate_limit_error

        # Act & Assert
        with pytest.raises(RateLimitError, match="Rate limit exceeded"):
            await retry_service.generate_content_with_retry(
                "test prompt", ContentType.CV_ANALYSIS
            )

    @pytest.mark.asyncio
    async def test_network_error_propagated(self, retry_service, mock_retry_handler):
        """Test that NetworkError is properly propagated."""
        # Arrange
        network_error = NetworkError("Network connection failed")
        mock_retry_handler.generate_content.side_effect = network_error

        # Act & Assert
        with pytest.raises(NetworkError, match="Network connection failed"):
            await retry_service.generate_content_with_retry(
                "test prompt", ContentType.CV_ANALYSIS
            )

    @pytest.mark.asyncio
    async def test_error_recovery_failure_propagates_original_error(self, retry_service, mock_error_recovery):
        """Test that when error recovery fails, the original error is propagated."""
        # Arrange
        original_error = ValueError("Original error that should be propagated")
        mock_error_recovery.get_fallback_content.side_effect = TypeError("Recovery failed")

        # Act & Assert
        with pytest.raises(ValueError, match="Original error that should be propagated"):
            await retry_service.handle_error_with_fallback(
                original_error, ContentType.CV_ANALYSIS, start_time=0.0
            )

    @pytest.mark.asyncio
    async def test_error_recovery_key_error_propagates_original(self, retry_service, mock_error_recovery):
        """Test that KeyError in recovery service propagates the original error."""
        # Arrange
        original_error = RuntimeError("Runtime error that should be propagated")
        mock_error_recovery.get_fallback_content.side_effect = KeyError("Missing key in recovery")

        # Act & Assert
        with pytest.raises(RuntimeError, match="Runtime error that should be propagated"):
            await retry_service.handle_error_with_fallback(
                original_error, ContentType.CV_ANALYSIS, start_time=0.0
            )

    @pytest.mark.asyncio
    async def test_error_recovery_value_error_propagates_original(self, retry_service, mock_error_recovery):
        """Test that ValueError in recovery service propagates the original error."""
        # Arrange
        original_error = ConnectionError("Connection error that should be propagated")
        mock_error_recovery.get_fallback_content.side_effect = ValueError("Invalid value in recovery")

        # Act & Assert
        with pytest.raises(ConnectionError, match="Connection error that should be propagated"):
            await retry_service.handle_error_with_fallback(
                original_error, ContentType.CV_ANALYSIS, start_time=0.0
            )

    @pytest.mark.asyncio
    async def test_successful_fallback_content_returns_response(self, retry_service, mock_error_recovery):
        """Test that successful fallback content returns a proper LLMResponse."""
        # Arrange
        original_error = RuntimeError("Some error")
        mock_error_recovery.get_fallback_content.return_value = "Fallback content"

        # Act
        response = await retry_service.handle_error_with_fallback(
            original_error, ContentType.CV_ANALYSIS, start_time=0.0
        )

        # Assert
        assert response.content == "Fallback content"
        assert response.success is True
        assert response.metadata["fallback_used"] is True
        assert "_fallback" in response.model_used