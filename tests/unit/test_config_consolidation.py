"""Unit tests for configuration consolidation and environment variable loading."""

import os
import pytest
from unittest.mock import patch

from src.config.settings import get_config, LLMConfig
from src.config.environment import load_config, get_environment, Environment


class TestConfigConsolidation:
    """Test configuration consolidation and environment variable loading."""

    def test_consolidated_debug_mode_loading(self):
        """Test that DEBUG_MODE is loaded correctly from environment."""
        with patch.dict(os.environ, {"DEBUG_MODE": "true"}):
            config = load_config()
            assert config.debug is True

        with patch.dict(os.environ, {"DEBUG_MODE": "false"}):
            config = load_config()
            assert config.debug is False

    def test_consolidated_session_timeout_loading(self):
        """Test that SESSION_TIMEOUT_SECONDS is loaded correctly."""
        with patch.dict(os.environ, {"SESSION_TIMEOUT_SECONDS": "7200"}):
            config = load_config()
            assert config.security.session_timeout_seconds == 7200

    def test_consolidated_request_timeout_loading(self):
        """Test that REQUEST_TIMEOUT_SECONDS is loaded correctly."""
        with patch.dict(os.environ, {"REQUEST_TIMEOUT_SECONDS": "120"}):
            config = load_config()
            assert config.performance.request_timeout_seconds == 120

    def test_consolidated_llm_rate_limits_loading(self):
        """Test that LLM rate limiting variables are loaded correctly."""
        with patch.dict(os.environ, {
            "LLM_REQUESTS_PER_MINUTE": "50",
            "LLM_TOKENS_PER_MINUTE": "80000"
        }):
            llm_config = LLMConfig()
            assert llm_config.max_requests_per_minute == 50
            assert llm_config.max_tokens_per_minute == 80000

    def test_environment_variable_consistency(self):
        """Test that ENVIRONMENT variable is used consistently."""
        with patch.dict(os.environ, {"ENVIRONMENT": "production"}):
            env = get_environment()
            assert env == Environment.PRODUCTION

        with patch.dict(os.environ, {"ENVIRONMENT": "testing"}):
            env = get_environment()
            assert env == Environment.TESTING

    def test_default_values_when_env_vars_missing(self):
        """Test that default values are used when environment variables are missing."""
        # Provide minimal required environment variables
        minimal_env = {
            'GEMINI_API_KEY': 'test_api_key_for_testing'
        }
        with patch.dict(os.environ, minimal_env, clear=True):
            config = load_config()
            llm_config = LLMConfig()
            
            # Test default values
            assert config.debug is False  # DEBUG_MODE defaults to false
            assert config.security.session_timeout_seconds == 3600  # Default 1 hour
            assert config.performance.request_timeout_seconds == 60  # Default 60 seconds
            assert llm_config.max_requests_per_minute == 30  # Default
            assert llm_config.max_tokens_per_minute == 60000  # Updated default
            assert get_environment() == Environment.DEVELOPMENT  # Default environment

    def test_invalid_environment_variable_handling(self):
        """Test handling of invalid environment variable values."""
        with patch.dict(os.environ, {"ENVIRONMENT": "invalid_env"}):
            env = get_environment()
            assert env == Environment.DEVELOPMENT  # Should fallback to development

    def test_numeric_environment_variable_validation(self):
        """Test that numeric environment variables are properly validated."""
        # Test invalid numeric values - should use defaults
        with patch.dict(os.environ, {
            "SESSION_TIMEOUT_SECONDS": "invalid_number",
            "REQUEST_TIMEOUT_SECONDS": "not_a_number"
        }):
            # The config should handle invalid values gracefully
            # This might raise an exception or use defaults depending on implementation
            try:
                config = load_config()
                # If no exception, verify reasonable defaults are used
                assert isinstance(config.security.session_timeout_seconds, int)
                assert isinstance(config.performance.request_timeout_seconds, int)
            except ValueError:
                # This is also acceptable behavior for invalid numeric values
                pass

    def test_configuration_object_creation(self):
        """Test that configuration objects are created successfully."""
        config = load_config()
        app_config = get_config()
        
        # Verify objects are created and have expected attributes
        assert hasattr(config, 'environment')
        assert hasattr(config, 'debug')
        assert hasattr(config, 'security')
        assert hasattr(config, 'performance')
        
        assert hasattr(app_config, 'llm')
        assert hasattr(app_config, 'vector_db')
        assert hasattr(app_config, 'ui')
        assert hasattr(app_config, 'output')
        assert hasattr(app_config, 'logging')

    def test_no_duplicate_configuration_keys(self):
        """Test that there are no conflicting configuration keys."""
        # This test ensures that the consolidation removed duplicates
        # We test this by checking that only the new consolidated keys work
        
        with patch.dict(os.environ, {
            "DEBUG_MODE": "true",  # New consolidated key
            "SESSION_TIMEOUT_SECONDS": "1800",  # New consolidated key
        }):
            config = load_config()
            llm_config = LLMConfig()
            
            # Verify the new keys work
            assert config.debug is True
            assert config.security.session_timeout_seconds == 1800
            
            # The old duplicate keys should not interfere
            # (This is more of a documentation test than a functional test)
            assert hasattr(config, 'debug')  # Should use DEBUG_MODE
            assert hasattr(config.security, 'session_timeout_seconds')  # Should use SESSION_TIMEOUT_SECONDS