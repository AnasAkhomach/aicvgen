"""Unit tests for API key management consolidation in LLM Service."""

import pytest
from unittest.mock import Mock, patch
from src.services.llm_service import EnhancedLLMService
from src.utils.exceptions import ConfigurationError


class TestAPIKeyManagement:
    """Test API key management consolidation in EnhancedLLMService."""

    @patch('src.services.llm_service.genai')
    @patch('src.services.llm_service.get_config')
    def test_determine_active_api_key_user_priority(self, mock_get_config, mock_genai):
        """Test that user-provided API key takes highest priority."""
        # Setup mock config
        mock_config = Mock()
        mock_config.llm.gemini_api_key_primary = "primary_key"
        mock_config.llm.gemini_api_key_fallback = "fallback_key"
        mock_config.llm_settings.default_model = "gemini-pro"
        mock_config.data_dir = "/tmp/test_cache"
        mock_get_config.return_value = mock_config
        
        # Mock genai
        mock_genai.GenerativeModel.return_value = Mock()
        
        # Initialize service with user key
        service = EnhancedLLMService(user_api_key="user_key")
        
        # Verify user key is active
        assert service.active_api_key == "user_key"
        assert service.using_user_key is True
        assert service.using_fallback is False
        
        # Verify API key info
        info = service.get_current_api_key_info()
        assert info["key_source"] == "user"
        assert info["using_user_key"] is True
        assert info["using_fallback"] is False

    @patch('src.services.llm_service.genai')
    @patch('src.services.llm_service.get_config')
    def test_determine_active_api_key_primary_fallback(self, mock_get_config, mock_genai):
        """Test that primary key is used when no user key provided."""
        # Setup mock config
        mock_config = Mock()
        mock_config.llm.gemini_api_key_primary = "primary_key"
        mock_config.llm.gemini_api_key_fallback = "fallback_key"
        mock_config.llm_settings.default_model = "gemini-pro"
        mock_config.data_dir = "/tmp/test_cache"
        mock_get_config.return_value = mock_config
        
        # Mock genai
        mock_genai.GenerativeModel.return_value = Mock()
        
        # Initialize service without user key
        service = EnhancedLLMService()
        
        # Verify primary key is active
        assert service.active_api_key == "primary_key"
        assert service.using_user_key is False
        assert service.using_fallback is False
        
        # Verify API key info
        info = service.get_current_api_key_info()
        assert info["key_source"] == "primary"
        assert info["using_user_key"] is False
        assert info["using_fallback"] is False

    @patch('src.services.llm_service.genai')
    @patch('src.services.llm_service.get_config')
    def test_determine_active_api_key_fallback_only(self, mock_get_config, mock_genai):
        """Test that fallback key is used when no user or primary key available."""
        # Setup mock config with only fallback key
        mock_config = Mock()
        mock_config.llm.gemini_api_key_primary = None
        mock_config.llm.gemini_api_key_fallback = "fallback_key"
        mock_config.llm_settings.default_model = "gemini-pro"
        mock_config.data_dir = "/tmp/test_cache"
        mock_get_config.return_value = mock_config
        
        # Mock genai
        mock_genai.GenerativeModel.return_value = Mock()
        
        # Initialize service without user key
        service = EnhancedLLMService()
        
        # Verify fallback key is active
        assert service.active_api_key == "fallback_key"
        assert service.using_user_key is False
        assert service.using_fallback is False  # Not switched to fallback, using it as primary
        
        # Verify API key info
        info = service.get_current_api_key_info()
        assert info["key_source"] == "primary"  # Using fallback as primary since no primary available

    @patch('src.services.llm_service.get_config')
    def test_determine_active_api_key_no_keys_raises_error(self, mock_get_config):
        """Test that ConfigurationError is raised when no API keys are available."""
        # Setup mock config with no keys
        mock_config = Mock()
        mock_config.llm.gemini_api_key_primary = None
        mock_config.llm.gemini_api_key_fallback = None
        mock_config.llm_settings.default_model = "gemini-pro"
        mock_config.data_dir = "/tmp/test_cache"
        mock_get_config.return_value = mock_config
        
        # Verify ConfigurationError is raised
        with pytest.raises(ConfigurationError) as exc_info:
            EnhancedLLMService()
        
        assert "CRITICAL: Gemini API key is not configured" in str(exc_info.value)

    @patch('src.services.llm_service.genai')
    @patch('src.services.llm_service.get_config')
    def test_switch_to_fallback_key_success(self, mock_get_config, mock_genai):
        """Test successful switch to fallback API key."""
        # Setup mock config
        mock_config = Mock()
        mock_config.llm.gemini_api_key_primary = "primary_key"
        mock_config.llm.gemini_api_key_fallback = "fallback_key"
        mock_config.llm_settings.default_model = "gemini-pro"
        mock_config.data_dir = "/tmp/test_cache"
        mock_get_config.return_value = mock_config
        
        # Mock genai
        mock_genai.GenerativeModel.return_value = Mock()
        
        # Initialize service
        service = EnhancedLLMService()
        
        # Verify initial state
        assert service.active_api_key == "primary_key"
        assert service.using_fallback is False
        
        # Switch to fallback
        result = service._switch_to_fallback_key()
        
        # Verify successful switch
        assert result is True
        assert service.active_api_key == "fallback_key"
        assert service.using_fallback is True
        
        # Verify API key info after switch
        info = service.get_current_api_key_info()
        assert info["key_source"] == "fallback"
        assert info["using_fallback"] is True

    @patch('src.services.llm_service.genai')
    @patch('src.services.llm_service.get_config')
    def test_switch_to_fallback_key_no_fallback_available(self, mock_get_config, mock_genai):
        """Test switch to fallback when no fallback key is available."""
        # Setup mock config without fallback key
        mock_config = Mock()
        mock_config.llm.gemini_api_key_primary = "primary_key"
        mock_config.llm.gemini_api_key_fallback = None
        mock_config.llm_settings.default_model = "gemini-pro"
        mock_config.data_dir = "/tmp/test_cache"
        mock_get_config.return_value = mock_config
        
        # Mock genai
        mock_genai.GenerativeModel.return_value = Mock()
        
        # Initialize service
        service = EnhancedLLMService()
        
        # Attempt to switch to fallback
        result = service._switch_to_fallback_key()
        
        # Verify switch failed
        assert result is False
        assert service.active_api_key == "primary_key"
        assert service.using_fallback is False

    @patch('src.services.llm_service.genai')
    @patch('src.services.llm_service.get_config')
    def test_switch_to_fallback_key_already_using_fallback(self, mock_get_config, mock_genai):
        """Test switch to fallback when already using fallback key."""
        # Setup mock config
        mock_config = Mock()
        mock_config.llm.gemini_api_key_primary = "primary_key"
        mock_config.llm.gemini_api_key_fallback = "fallback_key"
        mock_config.llm_settings.default_model = "gemini-pro"
        mock_config.data_dir = "/tmp/test_cache"
        mock_get_config.return_value = mock_config
        
        # Mock genai
        mock_genai.GenerativeModel.return_value = Mock()
        
        # Initialize service and switch to fallback
        service = EnhancedLLMService()
        service._switch_to_fallback_key()
        
        # Attempt to switch again
        result = service._switch_to_fallback_key()
        
        # Verify second switch failed
        assert result is False
        assert service.using_fallback is True

    @patch('src.services.llm_service.genai')
    @patch('src.services.llm_service.get_config')
    def test_get_current_api_key_info_comprehensive(self, mock_get_config, mock_genai):
        """Test comprehensive API key info reporting."""
        # Setup mock config
        mock_config = Mock()
        mock_config.llm.gemini_api_key_primary = "primary_key"
        mock_config.llm.gemini_api_key_fallback = "fallback_key"
        mock_config.llm_settings.default_model = "gemini-pro"
        mock_config.data_dir = "/tmp/test_cache"
        mock_get_config.return_value = mock_config
        
        # Mock genai
        mock_genai.GenerativeModel.return_value = Mock()
        
        # Test with user key
        service = EnhancedLLMService(user_api_key="user_key")
        info = service.get_current_api_key_info()
        
        expected_keys = {"using_user_key", "using_fallback", "has_fallback_available", "key_source"}
        assert set(info.keys()) == expected_keys
        assert info["using_user_key"] is True
        assert info["using_fallback"] is False
        assert info["has_fallback_available"] is True
        assert info["key_source"] == "user"