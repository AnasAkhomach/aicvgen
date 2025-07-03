import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from src.services.llm_retry_service import LLMRetryService
from src.models.data_models import ContentType, LLMResponse
from src.error_handling.exceptions import (
    OperationTimeoutError,
    RateLimitError,
    NetworkError,
    ConfigurationError,
)


class TestLLMRetryService:
    """Test cases for LLMRetryService."""

    @pytest.fixture
    def mock_retry_handler(self):
        """Create mock retry handler."""
        handler = AsyncMock()
        handler.generate_content = AsyncMock()
        return handler

    @pytest.fixture
    def mock_api_key_manager(self):
        """Create mock API key manager."""
        manager = AsyncMock()
        manager.has_fallback_available = MagicMock(return_value=True)
        manager.switch_to_fallback_key = AsyncMock(return_value=True)
        return manager

    @pytest.fixture
    def mock_rate_limiter(self):
        """Create mock rate limiter."""
        limiter = AsyncMock()
        limiter.wait_if_needed_async = AsyncMock()
        return limiter

    @pytest.fixture
    def mock_error_recovery(self):
        """Create mock error recovery service."""
        recovery = AsyncMock()
        recovery.get_fallback_content = AsyncMock(return_value="Fallback content")
        return recovery

    @pytest.fixture
    def retry_service(self, mock_retry_handler, mock_api_key_manager):
        """Create an LLMRetryService instance for testing."""
        return LLMRetryService(
            llm_retry_handler=mock_retry_handler,
            api_key_manager=mock_api_key_manager,
            timeout=30,
            model_name="test-model",
        )

    @pytest.fixture
    def mock_llm_response(self):
        """Create a mock LLM response."""
        response = MagicMock()
        response.text = "Test response content"
        response.tokens = 50
        response.usage = {"total_tokens": 50}
        return response

    def test_initialization(self, retry_service):
        """Test service initialization."""
        assert retry_service.timeout == 30
        assert retry_service.model_name == "test-model"
        assert retry_service.rate_limiter is None
        assert retry_service.error_recovery is None

    def test_initialization_with_optional_params(
        self,
        mock_retry_handler,
        mock_api_key_manager,
        mock_rate_limiter,
        mock_error_recovery,
    ):
        """Test initialization with optional parameters."""
        service = LLMRetryService(
            llm_retry_handler=mock_retry_handler,
            api_key_manager=mock_api_key_manager,
            rate_limiter=mock_rate_limiter,
            error_recovery=mock_error_recovery,
            timeout=60,
            model_name="custom-model",
        )

        assert service.timeout == 60
        assert service.model_name == "custom-model"
        assert service.rate_limiter is mock_rate_limiter
        assert service.error_recovery is mock_error_recovery

    @pytest.mark.asyncio
    async def test_apply_rate_limiting_with_limiter(
        self, retry_service, mock_rate_limiter
    ):
        """Test rate limiting application when rate limiter is present."""
        retry_service.rate_limiter = mock_rate_limiter

        await retry_service.apply_rate_limiting()

        mock_rate_limiter.wait_if_needed_async.assert_called_once_with("test-model")

    @pytest.mark.asyncio
    async def test_apply_rate_limiting_without_limiter(self, retry_service):
        """Test rate limiting application when no rate limiter is present."""
        # Should not raise exception
        await retry_service.apply_rate_limiting()

    @pytest.mark.asyncio
    async def test_call_llm_with_retry_success(self, retry_service, mock_llm_response):
        """Test successful LLM call with retry handler."""
        retry_service.llm_retry_handler.generate_content.return_value = (
            mock_llm_response
        )

        result = await retry_service.call_llm_with_retry(
            "test prompt", session_id="123"
        )

        assert result is mock_llm_response
        retry_service.llm_retry_handler.generate_content.assert_called_once_with(
            "test prompt", session_id="123"
        )

    

    def test_create_llm_response_basic(self, retry_service, mock_llm_response):
        """Test basic LLM response creation."""
        content_type = ContentType.CV_ANALYSIS
        processing_time = 1.5

        result = retry_service.create_llm_response(
            mock_llm_response,
            processing_time,
            content_type,
            session_id="123",
            item_id="456",
        )

        assert isinstance(result, LLMResponse)
        assert result.content == "Test response content"
        assert result.tokens_used == 50
        assert result.processing_time == 1.5
        assert result.model_used == "test-model"
        assert result.success is True
        assert result.metadata["session_id"] == "123"
        assert result.metadata["item_id"] == "456"
        assert result.metadata["content_type"] == "cv_analysis"

    def test_create_llm_response_negative_tokens(self, retry_service):
        """Test LLM response creation with negative token count."""
        mock_response = MagicMock()
        mock_response.text = "Test content"
        mock_response.tokens = -10  # Negative tokens

        result = retry_service.create_llm_response(
            mock_response, 1.0, ContentType.CV_ANALYSIS
        )

        assert result.tokens_used == 0  # Should be capped at 0

    def test_create_llm_response_excessive_tokens(self, retry_service):
        """Test LLM response creation with excessive token count."""
        mock_response = MagicMock()
        mock_response.text = "Test content"
        mock_response.tokens = 15000  # Excessive tokens

        result = retry_service.create_llm_response(
            mock_response, 1.0, ContentType.CV_ANALYSIS
        )

        assert result.tokens_used == 10000  # Should be capped at 10000

    def test_create_llm_response_invalid_processing_time(
        self, retry_service, mock_llm_response
    ):
        """Test LLM response creation with invalid processing time."""
        result = retry_service.create_llm_response(
            mock_llm_response, -1.0, ContentType.CV_ANALYSIS  # Invalid processing time
        )

        assert result.processing_time > 0  # Should be estimated

    @pytest.mark.asyncio
    async def test_handle_error_with_fallback_rate_limit(self, retry_service):
        """Test error handling with rate limit error and successful fallback switch."""
        from src.error_handling.classification import is_rate_limit_error

        error = RateLimitError("Rate limit exceeded")

        with patch(
            "src.error_handling.classification.is_rate_limit_error", return_value=True
        ):
            with pytest.raises(
                RateLimitError
            ):  # Should re-raise after successful switch
                await retry_service.handle_error_with_fallback(
                    error, ContentType.CV_ANALYSIS, 1.0, session_id="123"
                )

        retry_service.api_key_manager.switch_to_fallback_key.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_error_with_fallback_no_fallback_available(
        self, retry_service
    ):
        """Test error handling when no fallback is available."""
        retry_service.api_key_manager.has_fallback_available.return_value = False
        error = RateLimitError("Rate limit exceeded")

        with patch(
            "src.error_handling.classification.is_rate_limit_error", return_value=True
        ):
            with pytest.raises(RateLimitError):
                await retry_service.handle_error_with_fallback(
                    error, ContentType.CV_ANALYSIS, 1.0
                )

    @pytest.mark.asyncio
    async def test_handle_error_with_fallback_content(
        self, retry_service, mock_error_recovery
    ):
        """Test error handling with fallback content generation."""
        retry_service.error_recovery = mock_error_recovery
        retry_service.api_key_manager.has_fallback_available.return_value = False

        error = NetworkError("Network error")

        result = await retry_service.handle_error_with_fallback(
            error, ContentType.CV_ANALYSIS, 1.0, session_id="123"
        )

        assert isinstance(result, LLMResponse)
        assert result.content == "Fallback content"
        assert result.model_used == "test-model_fallback"
        assert result.metadata["fallback_used"] is True

    @pytest.mark.asyncio
    async def test_handle_error_with_fallback_recovery_failure(
        self, retry_service, mock_error_recovery
    ):
        """Test error handling when fallback content generation fails."""
        retry_service.error_recovery = mock_error_recovery
        retry_service.api_key_manager.has_fallback_available.return_value = False
        mock_error_recovery.get_fallback_content.side_effect = ValueError(
            "Recovery failed"
        )

        error = NetworkError("Network error")

        with pytest.raises(NetworkError):
            await retry_service.handle_error_with_fallback(
                error, ContentType.CV_ANALYSIS, 1.0
            )

    @pytest.mark.asyncio
    async def test_generate_content_with_retry_success(
        self, retry_service, mock_llm_response
    ):
        """Test successful content generation with retry."""
        retry_service.llm_retry_handler.generate_content.return_value = (
            mock_llm_response
        )

        result = await retry_service.generate_content_with_retry(
            "test prompt", ContentType.CV_ANALYSIS, session_id="123"
        )

        assert isinstance(result, LLMResponse)
        assert result.content == "Test response content"

    @pytest.mark.asyncio
    async def test_generate_content_with_retry_configuration_error(self, retry_service):
        """Test content generation with configuration error (should not retry)."""
        retry_service.llm_retry_handler.generate_content.side_effect = (
            ConfigurationError("Config error")
        )

        with pytest.raises(ConfigurationError):
            await retry_service.generate_content_with_retry(
                "test prompt", ContentType.CV_ANALYSIS
            )

    @pytest.mark.asyncio
    async def test_generate_content_with_retry_rate_limit_error(self, retry_service):
        """Test content generation with rate limit error (should be re-raised)."""
        retry_service.llm_retry_handler.generate_content.side_effect = RateLimitError(
            "Rate limit"
        )

        with pytest.raises(RateLimitError):
            await retry_service.generate_content_with_retry(
                "test prompt", ContentType.CV_ANALYSIS
            )

    @pytest.mark.asyncio
    async def test_generate_content_with_retry_timeout_error(self, retry_service):
        """Test content generation with timeout error."""
        retry_service.llm_retry_handler.generate_content.side_effect = (
            asyncio.TimeoutError()
        )
        retry_service.timeout = 0.1

        with pytest.raises(OperationTimeoutError, match="LLM operation timed out after 0.1 seconds"):
            await retry_service.generate_content_with_retry(
                "test prompt", ContentType.CV_ANALYSIS
            )

    @pytest.mark.asyncio
    async def test_generate_content_with_retry_google_api_error(
        self, retry_service, mock_error_recovery
    ):
        """Test content generation with Google API error and fallback."""
        from google.api_core import exceptions as google_exceptions

        retry_service.error_recovery = mock_error_recovery
        retry_service.llm_retry_handler.generate_content.side_effect = (
            google_exceptions.GoogleAPICallError("API error")
        )

        result = await retry_service.generate_content_with_retry(
            "test prompt", ContentType.CV_ANALYSIS
        )

        assert result.content == "Fallback content"
        assert result.metadata["fallback_used"] is True
