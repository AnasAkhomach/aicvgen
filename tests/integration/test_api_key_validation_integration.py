"""Integration tests for API key validation workflow."""

import pytest
import streamlit as st
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import asyncio

# Import the modules we're testing
from src.frontend.callbacks import handle_api_key_validation
from src.services.llm_service import EnhancedLLMService
from src.utils.exceptions import ConfigurationError


class TestAPIKeyValidationIntegration:
    """Integration tests for the complete API key validation workflow."""

    def setup_method(self):
        """Set up test environment before each test."""
        # Clear session state
        if hasattr(st, 'session_state'):
            for key in list(st.session_state.keys()):
                del st.session_state[key]

    @patch('src.frontend.callbacks.get_llm_service')
    @patch('src.frontend.callbacks.st.spinner')
    @patch('src.frontend.callbacks.st.success')
    @patch('src.frontend.callbacks.st.rerun')
    def test_successful_api_key_validation_workflow(self, mock_rerun, mock_success, mock_spinner, mock_get_llm_service):
        """Test the complete workflow for successful API key validation."""
        # Setup
        st.session_state.user_gemini_api_key = "valid-api-key"
        
        # Mock LLM service
        mock_llm_service = Mock(spec=EnhancedLLMService)
        mock_llm_service.validate_api_key = AsyncMock(return_value=True)
        mock_get_llm_service.return_value = mock_llm_service
        
        # Mock spinner context manager
        mock_spinner.return_value.__enter__ = Mock()
        mock_spinner.return_value.__exit__ = Mock()
        
        # Execute
        handle_api_key_validation()
        
        # Verify
        mock_get_llm_service.assert_called_once_with(user_api_key="valid-api-key")
        assert st.session_state.api_key_validated is True
        assert st.session_state.get("api_key_validation_failed", False) is False
        mock_success.assert_called_once_with("✅ API key is valid and ready to use!")
        mock_rerun.assert_called_once()

    @patch('src.frontend.callbacks.get_llm_service')
    @patch('src.frontend.callbacks.st.spinner')
    @patch('src.frontend.callbacks.st.error')
    @patch('src.frontend.callbacks.st.rerun')
    def test_failed_api_key_validation_workflow(self, mock_rerun, mock_error, mock_spinner, mock_get_llm_service):
        """Test the complete workflow for failed API key validation."""
        # Setup
        st.session_state.user_gemini_api_key = "invalid-api-key"
        
        # Mock LLM service
        mock_llm_service = Mock(spec=EnhancedLLMService)
        mock_llm_service.validate_api_key = AsyncMock(return_value=False)
        mock_get_llm_service.return_value = mock_llm_service
        
        # Mock spinner context manager
        mock_spinner.return_value.__enter__ = Mock()
        mock_spinner.return_value.__exit__ = Mock()
        
        # Execute
        handle_api_key_validation()
        
        # Verify
        mock_get_llm_service.assert_called_once_with(user_api_key="invalid-api-key")
        assert st.session_state.get("api_key_validated", True) is False
        assert st.session_state.api_key_validation_failed is True
        mock_error.assert_called_once_with("❌ API key validation failed. Please check your key.")
        mock_rerun.assert_called_once()

    @patch('src.frontend.callbacks.st.error')
    def test_validation_without_api_key(self, mock_error):
        """Test validation attempt without providing an API key."""
        # Setup - no API key in session state
        
        # Execute
        handle_api_key_validation()
        
        # Verify
        mock_error.assert_called_once_with("Please enter an API key first")
        assert st.session_state.get("api_key_validated", False) is False
        assert st.session_state.get("api_key_validation_failed", False) is False

    @patch('src.frontend.callbacks.get_llm_service')
    @patch('src.frontend.callbacks.st.error')
    @patch('src.frontend.callbacks.st.rerun')
    def test_configuration_error_handling(self, mock_rerun, mock_error, mock_get_llm_service):
        """Test handling of configuration errors during validation."""
        # Setup
        st.session_state.user_gemini_api_key = "test-key"
        
        # Mock configuration error
        mock_get_llm_service.side_effect = ConfigurationError("Missing configuration")
        
        # Call the function
        handle_api_key_validation()
        
        # Verify error handling
        mock_error.assert_called_once_with("❌ Configuration error: Missing configuration")
        mock_rerun.assert_called_once()

    @patch('src.frontend.callbacks.st.rerun')
    @patch('src.frontend.callbacks.st.error')
    @patch('src.frontend.callbacks.get_llm_service')
    def test_generic_exception_handling(self, mock_get_llm_service, mock_error, mock_rerun):
        """Test handling of generic exceptions during validation."""
        # Setup
        st.session_state.user_gemini_api_key = "test-key"
        
        # Mock get_llm_service to raise a generic exception
        mock_get_llm_service.side_effect = Exception("Network error")
        
        # Call the function
        handle_api_key_validation()
        
        # Verify error handling
        mock_error.assert_called_once_with("❌ Validation failed: Network error")
        mock_rerun.assert_called_once()

    def test_session_state_reset_on_validation_start(self):
        """Test that validation states are properly reset when validation starts."""
        # Setup - set some initial states
        st.session_state.api_key_validated = True
        st.session_state.api_key_validation_failed = True
        st.session_state.user_gemini_api_key = "test-key"
        
        with patch('src.frontend.callbacks.get_llm_service') as mock_get_llm_service:
            with patch('src.frontend.callbacks.st.spinner') as mock_spinner:
                with patch('src.frontend.callbacks.st.rerun'):
                    # Mock LLM service
                    mock_llm_service = Mock(spec=EnhancedLLMService)
                    mock_llm_service.validate_api_key = AsyncMock(return_value=True)
                    mock_get_llm_service.return_value = mock_llm_service
                    
                    # Mock spinner context manager
                    mock_spinner.return_value.__enter__ = Mock()
                    mock_spinner.return_value.__exit__ = Mock()
                    
                    # Execute
                    handle_api_key_validation()
        
        # Verify that states were reset and then set correctly
        assert st.session_state.api_key_validated is True
        assert st.session_state.api_key_validation_failed is False