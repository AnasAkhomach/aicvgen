#!/usr/bin/env python3
"""Unit tests for LLM Service API key validation functionality."""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from src.services.llm_service import EnhancedLLMService
from src.utils.exceptions import ConfigurationError


class TestEnhancedLLMServiceAPIValidation:
    """Test cases for API key validation in EnhancedLLMService."""

    @pytest.fixture
    def mock_genai_configure(self):
        """Mock genai.configure to prevent actual API calls during testing."""
        with patch("src.services.llm_service.genai.configure") as mock_configure:
            yield mock_configure

    @pytest.fixture
    def mock_genai_model(self):
        """Mock genai.GenerativeModel to prevent actual model initialization."""
        with patch("src.services.llm_service.genai.GenerativeModel") as mock_model:
            yield mock_model

    @pytest.fixture
    def llm_service(self, mock_genai_configure, mock_genai_model):
        """Create an EnhancedLLMService instance for testing."""
        return EnhancedLLMService(user_api_key="test-api-key")

    @pytest.mark.asyncio
    async def test_validate_api_key_success(self, llm_service):
        """Test successful API key validation."""
        # Mock genai.list_models to return a successful response
        mock_models = [Mock(name="gemini-pro"), Mock(name="gemini-pro-vision")]

        with patch(
            "src.services.llm_service.genai.list_models", return_value=mock_models
        ):
            result = await llm_service.validate_api_key()

        assert result is True

    @pytest.mark.asyncio
    async def test_validate_api_key_authentication_failure(self, llm_service):
        """Test API key validation with authentication error."""
        # Mock genai.list_models to raise an authentication error
        with patch(
            "src.services.llm_service.genai.list_models",
            side_effect=Exception("Invalid API key"),
        ):
            result = await llm_service.validate_api_key()

        assert result is False

    @pytest.mark.asyncio
    async def test_validate_api_key_network_error(self, llm_service):
        """Test API key validation with network error."""
        # Mock genai.list_models to raise a network error
        with patch(
            "src.services.llm_service.genai.list_models",
            side_effect=ConnectionError("Network unreachable"),
        ):
            result = await llm_service.validate_api_key()

        assert result is False

    @pytest.mark.asyncio
    async def test_validate_api_key_generic_exception(self, llm_service):
        """Test API key validation with generic exception."""
        # Mock genai.list_models to raise a generic exception
        with patch(
            "src.services.llm_service.genai.list_models",
            side_effect=RuntimeError("Unexpected error"),
        ):
            result = await llm_service.validate_api_key()

        assert result is False

    @pytest.mark.asyncio
    async def test_validate_api_key_empty_models_list(self, llm_service):
        """Test API key validation with empty models list (still valid)."""
        # Mock genai.list_models to return an empty list
        with patch("src.services.llm_service.genai.list_models", return_value=[]):
            result = await llm_service.validate_api_key()

        # Even with empty list, if no exception is raised, the key is considered valid
        assert result is True

    @pytest.mark.asyncio
    async def test_validate_api_key_logs_success(self, llm_service):
        """Test that successful validation logs appropriate message."""
        mock_models = [Mock(name="gemini-pro")]

        with patch(
            "src.services.llm_service.genai.list_models", return_value=mock_models
        ):
            with patch("src.services.llm_service.logger.info") as mock_log_info:
                result = await llm_service.validate_api_key()

                # Verify the method returned True and logged success
                assert result is True
                mock_log_info.assert_called_once_with(
                    "API key validation successful",
                    extra={
                        "models_count": 1,
                        "using_user_key": True,
                        "using_fallback": False,
                    },
                )

    @pytest.mark.asyncio
    async def test_validate_api_key_logs_failure(self, llm_service):
        """Test that failed validation logs appropriate warning."""
        with patch(
            "src.services.llm_service.genai.list_models",
            side_effect=Exception("Invalid API key"),
        ):
            with patch("src.services.llm_service.logger.warning") as mock_log_warning:
                result = await llm_service.validate_api_key()

                # Verify the method returned False and logged failure
                assert result is False
                mock_log_warning.assert_called_once_with(
                    "API key validation failed",
                    extra={
                        "error_type": "Exception",
                        "error_message": "Invalid API key",
                        "using_user_key": True,
                        "using_fallback": False,
                    },
                )

    def test_service_initialization_without_api_key_fails_fast(
        self, mock_genai_configure, mock_genai_model
    ):
        """Test that service initialization fails fast without API key."""
        # Mock settings to return None for all API keys
        with patch("src.services.llm_service.get_config") as mock_config:
            mock_settings = Mock()
            mock_settings.llm.gemini_api_key_primary = None
            mock_settings.llm.gemini_api_key_fallback = None
            mock_config.return_value = mock_settings

            # Should raise ConfigurationError immediately
            with pytest.raises(
                ConfigurationError, match="CRITICAL: Gemini API key is not configured"
            ):
                EnhancedLLMService(user_api_key=None)

    @pytest.mark.asyncio
    async def test_validate_api_key_uses_executor(self, llm_service):
        """Test that validate_api_key properly uses executor for async execution."""
        mock_models = [Mock(name="gemini-pro")]

        with patch(
            "src.services.llm_service.genai.list_models", return_value=mock_models
        ) as mock_list_models:
            with patch("asyncio.get_event_loop") as mock_get_loop:
                mock_loop = Mock()
                mock_executor_result = asyncio.Future()
                mock_executor_result.set_result(mock_models)
                mock_loop.run_in_executor.return_value = mock_executor_result
                mock_get_loop.return_value = mock_loop

                result = await llm_service.validate_api_key()

                # Verify that run_in_executor was called
                mock_loop.run_in_executor.assert_called_once()
                assert result is True
