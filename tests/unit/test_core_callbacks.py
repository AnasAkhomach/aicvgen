"""Unit tests for core callbacks.

Tests the callback functions that handle CV generation workflow.
"""

import pytest
import streamlit as st
from unittest.mock import Mock, patch, MagicMock
from src.core.callbacks import handle_cv_generation


class TestCoreCallbacks:
    """Test cases for core callback functions."""

    @pytest.fixture
    def mock_streamlit(self):
        """Mock Streamlit session state."""
        with patch("streamlit.session_state") as mock_session_state:
            mock_session_state.get.return_value = None
            mock_session_state.__setitem__ = Mock()
            mock_session_state.__getitem__ = Mock()
            yield mock_session_state

    @patch("streamlit.error")
    def test_handle_cv_generation_no_api_key(self, mock_error, mock_streamlit):
        """Test CV generation when no API key is provided."""
        mock_streamlit.get.return_value = None  # No API key

        with patch("streamlit.session_state", mock_streamlit):
            handle_cv_generation()

        mock_error.assert_called_with("Please provide a valid Gemini API key")
        # Verify processing state is not set when no API key
        mock_streamlit.__setitem__.assert_not_called()

    @patch("streamlit.error")
    def test_handle_cv_generation_empty_api_key(self, mock_error, mock_streamlit):
        """Test CV generation when API key is empty."""
        mock_streamlit.get.side_effect = lambda key, default=None: {
            "gemini_api_key": ""
        }.get(key, default)

        with patch("streamlit.session_state", mock_streamlit):
            handle_cv_generation()

        mock_error.assert_called_with("Please provide a valid Gemini API key")

    @patch("streamlit.error")
    def test_handle_cv_generation_whitespace_api_key(self, mock_error, mock_streamlit):
        """Test CV generation when API key is only whitespace."""
        mock_streamlit.get.side_effect = lambda key, default=None: {
            "gemini_api_key": "   "
        }.get(key, default)

        with patch("streamlit.session_state", mock_streamlit):
            handle_cv_generation()

        mock_error.assert_called_with("Please provide a valid Gemini API key")

    def test_handle_cv_generation_valid_api_key_no_inputs(self, mock_streamlit):
        """Test CV generation with valid API key but no job description or CV."""
        mock_streamlit.get.side_effect = lambda key, default=None: {
            "gemini_api_key": "valid_api_key_123",
            "job_description_input": "",
            "cv_text_input": "",
        }.get(key, default)

        with patch("streamlit.session_state", mock_streamlit):
            handle_cv_generation()

        # Should still proceed and store inputs in session state
        expected_calls = [
            ("job_description_input", ""),
            ("cv_text_input", ""),
            ("run_workflow", True),
        ]

        for key, value in expected_calls:
            mock_streamlit.__setitem__.assert_any_call(key, value)

    def test_handle_cv_generation_with_job_description(self, mock_streamlit):
        """Test CV generation with job description input."""
        job_description = "Software Engineer position at Tech Corp"

        mock_streamlit.get.side_effect = lambda key, default=None: {
            "gemini_api_key": "valid_api_key_123",
            "job_description_input": job_description,
            "cv_text_input": "",
        }.get(key, default)

        with patch("streamlit.session_state", mock_streamlit):
            handle_cv_generation()

        # Verify job description is stored
        mock_streamlit.__setitem__.assert_any_call(
            "job_description_input", job_description
        )
        mock_streamlit.__setitem__.assert_any_call("run_workflow", True)

    def test_handle_cv_generation_with_cv_text(self, mock_streamlit):
        """Test CV generation with CV text input."""
        cv_text = "John Doe\nSoftware Developer\n5 years experience"

        mock_streamlit.get.side_effect = lambda key, default=None: {
            "gemini_api_key": "valid_api_key_123",
            "job_description_input": "",
            "cv_text_input": cv_text,
        }.get(key, default)

        with patch("streamlit.session_state", mock_streamlit):
            handle_cv_generation()

        # Verify CV text is stored
        mock_streamlit.__setitem__.assert_any_call("cv_text_input", cv_text)
        mock_streamlit.__setitem__.assert_any_call("run_workflow", True)

    def test_handle_cv_generation_with_both_inputs(self, mock_streamlit):
        """Test CV generation with both job description and CV text."""
        job_description = "Software Engineer position"
        cv_text = "John Doe\nDeveloper"

        mock_streamlit.get.side_effect = lambda key, default=None: {
            "gemini_api_key": "valid_api_key_123",
            "job_description_input": job_description,
            "cv_text_input": cv_text,
        }.get(key, default)

        with patch("streamlit.session_state", mock_streamlit):
            handle_cv_generation()

        # Verify both inputs are stored
        mock_streamlit.__setitem__.assert_any_call(
            "job_description_input", job_description
        )
        mock_streamlit.__setitem__.assert_any_call("cv_text_input", cv_text)
        mock_streamlit.__setitem__.assert_any_call("run_workflow", True)

    def test_handle_cv_generation_workflow_flag_setting(self, mock_streamlit):
        """Test that workflow flag is properly set."""
        mock_streamlit.get.side_effect = lambda key, default=None: {
            "gemini_api_key": "valid_api_key_123",
            "job_description_input": "Test job",
            "cv_text_input": "Test CV",
        }.get(key, default)

        with patch("streamlit.session_state", mock_streamlit):
            handle_cv_generation()

        # Verify workflow flag is set to True
        mock_streamlit.__setitem__.assert_any_call("run_workflow", True)

    def test_handle_cv_generation_input_validation_sequence(self, mock_streamlit):
        """Test the sequence of input validation and storage."""
        mock_streamlit.get.side_effect = lambda key, default=None: {
            "gemini_api_key": "valid_api_key_123",
            "job_description_input": "Test job description",
            "cv_text_input": "Test CV text",
        }.get(key, default)

        with patch("streamlit.session_state", mock_streamlit):
            handle_cv_generation()

        # Verify the order and presence of all expected calls
        all_calls = mock_streamlit.__setitem__.call_args_list
        call_keys = [call[0][0] for call in all_calls]

        # All three keys should be set
        assert "job_description_input" in call_keys
        assert "cv_text_input" in call_keys
        assert "run_workflow" in call_keys

    def test_handle_cv_generation_api_key_validation_priority(self, mock_streamlit):
        """Test that API key validation happens before input processing."""
        mock_streamlit.get.side_effect = lambda key, default=None: {
            "gemini_api_key": "",  # Invalid API key
            "job_description_input": "Test job",
            "cv_text_input": "Test CV",
        }.get(key, default)

        with patch("streamlit.session_state", mock_streamlit), patch(
            "streamlit.error"
        ) as mock_error:
            handle_cv_generation()

        # Should show error and not process inputs
        mock_error.assert_called_with("Please provide a valid Gemini API key")
        # Should not set any session state variables
        mock_streamlit.__setitem__.assert_not_called()

    def test_handle_cv_generation_multiple_calls(self, mock_streamlit):
        """Test multiple calls to handle_cv_generation."""
        mock_streamlit.get.side_effect = lambda key, default=None: {
            "gemini_api_key": "valid_api_key_123",
            "job_description_input": "First job",
            "cv_text_input": "First CV",
        }.get(key, default)

        with patch("streamlit.session_state", mock_streamlit):
            # First call
            handle_cv_generation()
            first_call_count = mock_streamlit.__setitem__.call_count

            # Reset mock for second call
            mock_streamlit.reset_mock()
            mock_streamlit.get.side_effect = lambda key, default=None: {
                "gemini_api_key": "valid_api_key_123",
                "job_description_input": "Second job",
                "cv_text_input": "Second CV",
            }.get(key, default)

            # Second call
            handle_cv_generation()
            second_call_count = mock_streamlit.__setitem__.call_count

        # Both calls should set the same number of variables
        assert first_call_count == second_call_count
        assert (
            second_call_count == 3
        )  # job_description_input, cv_text_input, run_workflow

    def test_handle_cv_generation_preserves_other_session_state(self, mock_streamlit):
        """Test that function doesn't interfere with other session state variables."""
        mock_streamlit.get.side_effect = lambda key, default=None: {
            "gemini_api_key": "valid_api_key_123",
            "job_description_input": "Test job",
            "cv_text_input": "Test CV",
            "other_variable": "should_not_change",
        }.get(key, default)

        with patch("streamlit.session_state", mock_streamlit):
            handle_cv_generation()

        # Verify only expected variables are set
        set_keys = [call[0][0] for call in mock_streamlit.__setitem__.call_args_list]
        expected_keys = {"job_description_input", "cv_text_input", "run_workflow"}

        assert set(set_keys) == expected_keys
        assert "other_variable" not in set_keys

    def test_handle_cv_generation_api_key_strip_whitespace(self, mock_streamlit):
        """Test that API key whitespace is properly handled."""
        mock_streamlit.get.side_effect = lambda key, default=None: {
            "gemini_api_key": "  valid_api_key_123  ",  # With whitespace
            "job_description_input": "Test job",
            "cv_text_input": "Test CV",
        }.get(key, default)

        with patch("streamlit.session_state", mock_streamlit):
            handle_cv_generation()

        # Should proceed normally (whitespace should be stripped)
        mock_streamlit.__setitem__.assert_any_call("run_workflow", True)
