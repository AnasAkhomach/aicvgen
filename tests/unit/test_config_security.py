"""Unit tests for configuration security validation.

This module tests that the application properly validates environment variables
and raises appropriate errors when required API keys are missing.
"""

import os
import pytest
from unittest.mock import patch
from src.config.settings import LLMConfig


class TestConfigSecurity:
    """Test cases for configuration security validation."""

    def test_llm_config_raises_error_when_no_api_keys_set(self):
        """Test that LLMConfig raises ValueError when no API keys are provided."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove any existing API key environment variables
            if 'GEMINI_API_KEY' in os.environ:
                del os.environ['GEMINI_API_KEY']
            if 'GEMINI_API_KEY_FALLBACK' in os.environ:
                del os.environ['GEMINI_API_KEY_FALLBACK']
            
            with pytest.raises(ValueError) as exc_info:
                LLMConfig()
            
            assert "At least one Gemini API key is required" in str(exc_info.value)
            assert "GEMINI_API_KEY" in str(exc_info.value)
            assert "GEMINI_API_KEY_FALLBACK" in str(exc_info.value)

    def test_llm_config_succeeds_with_primary_key_only(self):
        """Test that LLMConfig succeeds when primary API key is provided."""
        with patch.dict(os.environ, {'GEMINI_API_KEY': 'test_primary_key'}, clear=True):
            config = LLMConfig()
            assert config.gemini_api_key_primary == 'test_primary_key'
            assert config.gemini_api_key_fallback == ''

    def test_llm_config_succeeds_with_fallback_key_only(self):
        """Test that LLMConfig succeeds when fallback API key is provided."""
        with patch.dict(os.environ, {'GEMINI_API_KEY_FALLBACK': 'test_fallback_key'}, clear=True):
            config = LLMConfig()
            assert config.gemini_api_key_primary == ''
            assert config.gemini_api_key_fallback == 'test_fallback_key'

    def test_llm_config_succeeds_with_both_keys(self):
        """Test that LLMConfig succeeds when both API keys are provided."""
        with patch.dict(os.environ, {
            'GEMINI_API_KEY': 'test_primary_key',
            'GEMINI_API_KEY_FALLBACK': 'test_fallback_key'
        }, clear=True):
            config = LLMConfig()
            assert config.gemini_api_key_primary == 'test_primary_key'
            assert config.gemini_api_key_fallback == 'test_fallback_key'

    def test_llm_config_empty_string_keys_raise_error(self):
        """Test that empty string API keys are treated as missing."""
        with patch.dict(os.environ, {
            'GEMINI_API_KEY': '',
            'GEMINI_API_KEY_FALLBACK': ''
        }, clear=True):
            with pytest.raises(ValueError) as exc_info:
                LLMConfig()
            
            assert "At least one Gemini API key is required" in str(exc_info.value)

    def test_security_warning_in_error_message(self):
        """Test that the error message includes security warning about hardcoding keys."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError) as exc_info:
                LLMConfig()
            
            error_message = str(exc_info.value)
            assert "CRITICAL SECURITY NOTE" in error_message
            assert "Never hardcode API keys" in error_message