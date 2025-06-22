"""Unit tests for LLM service configuration error handling (constructor-based DI only)."""

import os
import pytest
from unittest.mock import patch

from src.services.llm_service import EnhancedLLMService
from src.utils.exceptions import ConfigurationError


class TestLLMServiceConfiguration:
    """Test LLM service configuration and error handling (DI only)."""

    def test_enhanced_llm_service_raises_configuration_error_no_api_key(self):
        """Test that EnhancedLLMService raises ConfigurationError when no API key is available."""
        # Mock environment variables to ensure no API key is available
        with patch.dict(os.environ, {}, clear=True):
            # Mock the settings to return None for all API keys
            with patch("src.services.llm_service.get_config") as mock_config:
                mock_settings = mock_config.return_value
                mock_settings.llm.gemini_api_key_primary = None
                mock_settings.llm.gemini_api_key_fallback = None

                # Test that ConfigurationError is raised with no user API key
                with pytest.raises(ConfigurationError) as exc_info:
                    EnhancedLLMService(user_api_key=None)

                # Verify the error message is descriptive and actionable
                assert "CRITICAL: Gemini API key is not configured" in str(
                    exc_info.value
                )
                assert "Please set the GEMINI_API_KEY in your .env file" in str(
                    exc_info.value
                )
                assert "Application cannot start without a valid API key" in str(
                    exc_info.value
                )

    def test_enhanced_llm_service_succeeds_with_user_api_key(self):
        """Test that EnhancedLLMService initializes successfully with user-provided API key."""
        # Mock environment variables to ensure no API key is available
        with patch.dict(os.environ, {}, clear=True):
            # Mock the settings to return None for all API keys
            with patch("src.services.llm_service.get_config") as mock_config:
                mock_settings = mock_config.return_value
                mock_settings.llm.gemini_api_key_primary = None
                mock_settings.llm.gemini_api_key_fallback = None
                mock_settings.llm_settings.default_model = "gemini-pro"

                # Mock genai to prevent actual API calls
                with patch("src.services.llm_service.genai") as mock_genai:
                    mock_genai.GenerativeModel.return_value = object()

                    # Test that service initializes successfully with user API key
                    service = EnhancedLLMService(user_api_key="test-api-key")
                    assert service.active_api_key == "test-api-key"
                    assert service.using_user_key is True
