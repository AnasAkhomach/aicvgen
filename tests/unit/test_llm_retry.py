"""Unit tests for LLM service retry mechanism with tenacity."""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock
from tenacity import RetryError

from src.services.llm_service import EnhancedLLMService, RETRYABLE_EXCEPTIONS
from src.models.data_models import ContentType


class TestLLMRetryMechanism:
    """Test cases for the LLM service retry mechanism using tenacity."""

    @pytest.fixture
    def mock_rate_limiter(self):
        """Create a mock rate limiter."""
        mock_limiter = Mock()
        mock_limiter.acquire = Mock(return_value=True)
        mock_limiter.get_status = Mock(return_value={"requests_remaining": 10})
        return mock_limiter

    @pytest.fixture
    def llm_service(self, mock_rate_limiter):
        """Create an LLM service instance for testing."""
        with patch("src.services.llm.genai") as mock_genai:
            mock_model = Mock()
            mock_genai.GenerativeModel.return_value = mock_model

            service = EnhancedLLMService(timeout=30, rate_limiter=mock_rate_limiter)
            service.llm = mock_model
            return service

    def test_make_llm_api_call_success(self, llm_service):
        """Test successful API call without retries."""
        # Arrange
        mock_response = Mock()
        mock_response.text = "Generated content"
        llm_service.llm.generate_content.return_value = mock_response

        # Act
        result = llm_service._make_llm_api_call("test prompt")

        # Assert
        assert result == mock_response
        llm_service.llm.generate_content.assert_called_once_with("test prompt")

    def test_make_llm_api_call_retry_on_exception(self, llm_service):
        """Test that retryable exceptions trigger retries."""
        # Arrange
        mock_response = Mock()
        mock_response.text = "Generated content"

        # First two calls fail, third succeeds
        llm_service.llm.generate_content.side_effect = [
            Exception("Transient error"),
            Exception("Another transient error"),
            mock_response,
        ]

        # Act
        result = llm_service._make_llm_api_call("test prompt")

        # Assert
        assert result == mock_response
        assert llm_service.llm.generate_content.call_count == 3

    def test_make_llm_api_call_exhausts_retries(self, llm_service):
        """Test that method raises original exception after exhausting retries."""
        # Arrange
        llm_service.llm.generate_content.side_effect = Exception("Persistent error")

        # Act & Assert
        with pytest.raises(Exception, match="Persistent error"):
            llm_service._make_llm_api_call("test prompt")

        # Should have tried 3 times (initial + 2 retries)
        assert llm_service.llm.generate_content.call_count == 3

    def test_generate_with_timeout_integration(self, llm_service):
        """Test that _generate_with_timeout integrates with retry mechanism."""
        # Arrange
        mock_response = Mock()
        mock_response.text = "Generated content"

        with patch.object(
            llm_service, "_make_llm_api_call", return_value=mock_response
        ) as mock_api_call:
            # Act
            result = llm_service._generate_with_timeout("test prompt", "session123")

            # Assert
            assert result == mock_response
            mock_api_call.assert_called_once_with("test prompt")

    def test_retryable_exceptions_constant(self):
        """Test that RETRYABLE_EXCEPTIONS is properly defined."""
        # Assert
        assert isinstance(RETRYABLE_EXCEPTIONS, tuple)
        assert len(RETRYABLE_EXCEPTIONS) > 0
        assert Exception in RETRYABLE_EXCEPTIONS

    def test_retry_decorator_configuration(self, llm_service):
        """Test that the retry decorator is properly configured."""
        # Check that the _make_llm_api_call method has retry decorator
        method = llm_service._make_llm_api_call

        # The method should have retry attributes from tenacity
        assert hasattr(method, "retry")

        # Test that it fails after exhausting retries
        llm_service.llm.generate_content.side_effect = Exception("Persistent error")

        with pytest.raises(Exception, match="Persistent error"):
            llm_service._make_llm_api_call("test prompt")

        # Should have tried 3 times
        assert llm_service.llm.generate_content.call_count == 3

    def test_generate_with_timeout_uses_retry_method(self, llm_service):
        """Test that _generate_with_timeout uses the retry-enabled method."""
        # Arrange
        mock_response = Mock()
        mock_response.text = "Generated content"

        with patch.object(
            llm_service, "_make_llm_api_call", return_value=mock_response
        ) as mock_api_call:
            # Act
            result = llm_service._generate_with_timeout("test prompt", "session123")

            # Assert
            assert result == mock_response
            mock_api_call.assert_called_once_with("test prompt")
