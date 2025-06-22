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
    """Integration tests for the complete API key validation workflow (DI only)."""

    def setup_method(self):
        """Set up test environment before each test."""
        # Clear session state
        if hasattr(st, "session_state"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]

    def test_successful_api_key_validation_workflow(self):
        """Test the complete workflow for successful API key validation."""
        st.session_state.user_gemini_api_key = "valid-api-key"

        # Mock LLM service
        mock_llm_service = Mock(spec=EnhancedLLMService)
        mock_llm_service.validate_api_key = AsyncMock(return_value=True)

        # Mock spinner context manager
        with patch("src.frontend.callbacks.st.spinner") as mock_spinner, patch(
            "src.frontend.callbacks.st.success"
        ) as mock_success, patch("src.frontend.callbacks.st.rerun") as mock_rerun:
            mock_spinner.return_value.__enter__ = Mock()
            mock_spinner.return_value.__exit__ = Mock()

            # Execute
            handle_api_key_validation(llm_service=mock_llm_service)

            # Verify
            assert st.session_state.api_key_validated is True
            assert st.session_state.get("api_key_validation_failed", False) is False
            mock_success.assert_called_once_with(
                "✅ API key is valid and ready to use!"
            )
            mock_rerun.assert_called_once()

    def test_failed_api_key_validation_workflow(self):
        """Test the complete workflow for failed API key validation."""
        st.session_state.user_gemini_api_key = "invalid-api-key"

        # Mock LLM service
        mock_llm_service = Mock(spec=EnhancedLLMService)
        mock_llm_service.validate_api_key = AsyncMock(return_value=False)

        # Mock spinner context manager
        with patch("src.frontend.callbacks.st.spinner") as mock_spinner, patch(
            "src.frontend.callbacks.st.error"
        ) as mock_error, patch("src.frontend.callbacks.st.rerun") as mock_rerun:
            mock_spinner.return_value.__enter__ = Mock()
            mock_spinner.return_value.__exit__ = Mock()

            # Execute
            handle_api_key_validation(llm_service=mock_llm_service)

            # Verify
            assert st.session_state.get("api_key_validated", True) is False
            assert st.session_state.api_key_validation_failed is True
            mock_error.assert_called_once_with(
                "❌ API key validation failed. Please check your key."
            )
            mock_rerun.assert_called_once()

    def test_validation_without_api_key(self):
        """Test validation attempt without providing an API key."""
        # Setup - no API key in session state

        with patch("src.frontend.callbacks.st.error") as mock_error:
            handle_api_key_validation(llm_service=Mock())
            mock_error.assert_called_once_with("Please enter an API key first")
            assert st.session_state.get("api_key_validated", False) is False
            assert st.session_state.get("api_key_validation_failed", False) is False

    def test_configuration_error_handling(self):
        """Test handling of configuration errors during validation."""
        st.session_state.user_gemini_api_key = "test-key"

        # Mock LLM service to raise ConfigurationError
        mock_llm_service = Mock(spec=EnhancedLLMService)
        mock_llm_service.validate_api_key = AsyncMock(
            side_effect=ConfigurationError("Missing configuration")
        )

        with patch("streamlit.error") as mock_error, patch(
            "src.frontend.callbacks.st.rerun"
        ) as mock_rerun, patch("src.frontend.callbacks.st.spinner") as mock_spinner:
            mock_spinner.return_value.__enter__ = Mock()
            mock_spinner.return_value.__exit__ = Mock()
            handle_api_key_validation(llm_service=mock_llm_service)
            mock_error.assert_called_once_with(
                "❌ Configuration error: Missing configuration"
            )
            mock_rerun.assert_called_once()

    def test_generic_exception_handling(self):
        """Test handling of generic exceptions during validation."""
        st.session_state.user_gemini_api_key = "test-key"

        # Mock LLM service to raise generic Exception
        mock_llm_service = Mock(spec=EnhancedLLMService)
        mock_llm_service.validate_api_key = AsyncMock(
            side_effect=Exception("Network error")
        )

        with patch("streamlit.error") as mock_error, patch(
            "src.frontend.callbacks.st.rerun"
        ) as mock_rerun, patch("src.frontend.callbacks.st.spinner") as mock_spinner:
            mock_spinner.return_value.__enter__ = Mock()
            mock_spinner.return_value.__exit__ = Mock()
            handle_api_key_validation(llm_service=mock_llm_service)
            mock_error.assert_called_once_with("❌ Validation failed: Network error")
            mock_rerun.assert_called_once()

    def test_session_state_reset_on_validation_start(self):
        """Test that validation states are properly reset when validation starts."""
        # Setup - set some initial states
        st.session_state.api_key_validated = True
        st.session_state.api_key_validation_failed = True
        st.session_state.user_gemini_api_key = "test-key"

        mock_llm_service = Mock(spec=EnhancedLLMService)
        mock_llm_service.validate_api_key = AsyncMock(return_value=True)
        with patch("src.frontend.callbacks.st.spinner") as mock_spinner, patch(
            "src.frontend.callbacks.st.rerun"
        ) as mock_rerun, patch("src.frontend.callbacks.st.success") as mock_success:
            mock_spinner.return_value.__enter__ = Mock()
            mock_spinner.return_value.__exit__ = Mock()
            handle_api_key_validation(llm_service=mock_llm_service)
            assert st.session_state.api_key_validated is False
            assert st.session_state.api_key_validation_failed is False
