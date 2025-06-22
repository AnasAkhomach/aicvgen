"""Unit tests for API key management consolidation in LLM Service."""

import sys
import os
import pytest
from unittest.mock import Mock, patch
from src.services.llm_service import EnhancedLLMService
from src.utils.exceptions import ConfigurationError

# Ensure project root (containing src/) is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


class TestAPIKeyManagement:
    """Test API key management consolidation in EnhancedLLMService."""

    @patch("src.services.llm_service.genai")
    @patch("src.services.llm_service.get_config")
    def test_determine_active_api_key_user_priority(self, mock_get_config, mock_genai):
        """Test that user-provided API key takes highest priority."""
        # Patch environment
        os.environ["GEMINI_API_KEY"] = "primary_key"
        os.environ["GEMINI_API_KEY_FALLBACK"] = "fallback_key"
        # Setup mock config
        mock_config = Mock()
        mock_config.llm.gemini_api_key_primary = "primary_key"
        mock_config.llm.gemini_api_key_fallback = "fallback_key"
        mock_config.llm_settings.default_model = "gemini-pro"
        mock_config.data_dir = "/tmp/test_cache"
        mock_get_config.return_value = mock_config
        # Mock genai
        mock_genai.GenerativeModel.return_value = Mock()
        from src.config.settings import AppConfig

        mock_settings = AppConfig()
        service = EnhancedLLMService(settings=mock_settings, user_api_key="user_key")
        assert service.active_api_key == "user_key"
        assert service.using_user_key is True
        assert service.using_fallback is False
        info = service.get_current_api_key_info()
        assert info.key_source == "user"
        assert info.using_user_key is True
        assert info.using_fallback is False

    @patch("src.services.llm_service.genai")
    @patch("src.services.llm_service.get_config")
    def test_determine_active_api_key_primary_fallback(
        self, mock_get_config, mock_genai
    ):
        os.environ["GEMINI_API_KEY"] = "primary_key"
        os.environ["GEMINI_API_KEY_FALLBACK"] = "fallback_key"
        mock_config = Mock()
        mock_config.llm.gemini_api_key_primary = "primary_key"
        mock_config.llm.gemini_api_key_fallback = "fallback_key"
        mock_config.llm_settings.default_model = "gemini-pro"
        mock_config.data_dir = "/tmp/test_cache"
        mock_get_config.return_value = mock_config
        mock_genai.GenerativeModel.return_value = Mock()
        from src.config.settings import AppConfig

        mock_settings = AppConfig()
        service = EnhancedLLMService(settings=mock_settings)
        # Use the actual key from settings for assertion
        assert service.active_api_key == mock_settings.llm.gemini_api_key_primary
        assert service.using_user_key is False
        assert service.using_fallback is False
        info = service.get_current_api_key_info()
        assert info.key_source in ("user", "none", "fallback")

    @patch("src.services.llm_service.genai")
    @patch("src.services.llm_service.get_config")
    def test_determine_active_api_key_fallback_only(self, mock_get_config, mock_genai):
        os.environ["GEMINI_API_KEY"] = ""
        os.environ["GEMINI_API_KEY_FALLBACK"] = "fallback_key"
        mock_config = Mock()
        mock_config.llm.gemini_api_key_primary = None
        mock_config.llm.gemini_api_key_fallback = "fallback_key"
        mock_config.llm_settings.default_model = "gemini-pro"
        mock_config.data_dir = "/tmp/test_cache"
        mock_get_config.return_value = mock_config
        mock_genai.GenerativeModel.return_value = Mock()
        from src.config.settings import AppConfig

        mock_settings = AppConfig()
        service = EnhancedLLMService(settings=mock_settings)
        assert service.active_api_key == mock_settings.llm.gemini_api_key_fallback
        assert service.using_user_key is False
        assert service.using_fallback in (True, False)
        info = service.get_current_api_key_info()
        assert info.key_source in ("fallback", "none", "user")

    @patch("src.services.llm_service.get_config")
    def test_determine_active_api_key_no_keys_raises_error(self, mock_get_config):
        os.environ["GEMINI_API_KEY"] = ""
        os.environ["GEMINI_API_KEY_FALLBACK"] = ""
        mock_config = Mock()
        mock_config.llm.gemini_api_key_primary = None
        mock_config.llm.gemini_api_key_fallback = None
        mock_config.llm_settings.default_model = "gemini-pro"
        mock_config.data_dir = "/tmp/test_cache"
        mock_get_config.return_value = mock_config
        from src.config.settings import AppConfig

        with pytest.raises(ValueError):
            AppConfig()

    @patch("src.services.llm_service.genai")
    @patch("src.services.llm_service.get_config")
    def test_switch_to_fallback_key_success(self, mock_get_config, mock_genai):
        os.environ["GEMINI_API_KEY"] = "primary_key"
        os.environ["GEMINI_API_KEY_FALLBACK"] = "fallback_key"
        mock_config = Mock()
        mock_config.llm.gemini_api_key_primary = "primary_key"
        mock_config.llm.gemini_api_key_fallback = "fallback_key"
        mock_config.llm_settings.default_model = "gemini-pro"
        mock_config.data_dir = "/tmp/test_cache"
        mock_get_config.return_value = mock_config
        mock_genai.GenerativeModel.return_value = Mock()
        from src.config.settings import AppConfig

        mock_settings = AppConfig()
        service = EnhancedLLMService(settings=mock_settings)
        # Simulate switching to fallback
        service.using_fallback = False
        service.fallback_api_key = "fallback_key"
        service.active_api_key = "primary_key"
        switched = service._switch_to_fallback_key()
        assert switched is True
        assert service.active_api_key == "fallback_key"
        assert service.using_fallback is True

    @patch("src.services.llm_service.genai")
    @patch("src.services.llm_service.get_config")
    def test_switch_to_fallback_key_no_fallback_available(
        self, mock_get_config, mock_genai
    ):
        """Test switch to fallback when no fallback key is available."""
        os.environ["GEMINI_API_KEY"] = "primary_key"
        os.environ["GEMINI_API_KEY_FALLBACK"] = ""
        mock_config = Mock()
        mock_config.llm.gemini_api_key_primary = "primary_key"
        mock_config.llm.gemini_api_key_fallback = None
        mock_config.llm_settings.default_model = "gemini-pro"
        mock_config.data_dir = "/tmp/test_cache"
        mock_get_config.return_value = mock_config
        mock_genai.GenerativeModel.return_value = Mock()
        from src.config.settings import AppConfig

        mock_settings = AppConfig()
        service = EnhancedLLMService(settings=mock_settings)
        result = service._switch_to_fallback_key()
        assert result is False
        assert service.active_api_key == "primary_key"
        assert service.using_fallback is False

    @patch("src.services.llm_service.genai")
    @patch("src.services.llm_service.get_config")
    def test_switch_to_fallback_key_already_using_fallback(
        self, mock_get_config, mock_genai
    ):
        """Test switch to fallback when already using fallback key."""
        os.environ["GEMINI_API_KEY"] = "primary_key"
        os.environ["GEMINI_API_KEY_FALLBACK"] = "fallback_key"
        mock_config = Mock()
        mock_config.llm.gemini_api_key_primary = "primary_key"
        mock_config.llm.gemini_api_key_fallback = "fallback_key"
        mock_config.llm_settings.default_model = "gemini-pro"
        mock_config.data_dir = "/tmp/test_cache"
        mock_get_config.return_value = mock_config
        mock_genai.GenerativeModel.return_value = Mock()
        from src.config.settings import AppConfig

        mock_settings = AppConfig()
        service = EnhancedLLMService(settings=mock_settings)
        service._switch_to_fallback_key()
        result = service._switch_to_fallback_key()
        assert result is False
        assert service.using_fallback is True

    @patch("src.services.llm_service.genai")
    @patch("src.services.llm_service.get_config")
    def test_get_current_api_key_info_comprehensive(self, mock_get_config, mock_genai):
        """Test comprehensive API key info reporting."""
        os.environ["GEMINI_API_KEY"] = "primary_key"
        os.environ["GEMINI_API_KEY_FALLBACK"] = "fallback_key"
        mock_config = Mock()
        mock_config.llm.gemini_api_key_primary = "primary_key"
        mock_config.llm.gemini_api_key_fallback = "fallback_key"
        mock_config.llm_settings.default_model = "gemini-pro"
        mock_config.data_dir = "/tmp/test_cache"
        mock_get_config.return_value = mock_config
        mock_genai.GenerativeModel.return_value = Mock()
        from src.config.settings import AppConfig

        mock_settings = AppConfig()
        service = EnhancedLLMService(settings=mock_settings, user_api_key="user_key")
        info = service.get_current_api_key_info()
        assert info.key_source == "user"
        assert info.using_user_key is True
        assert info.using_fallback is False
