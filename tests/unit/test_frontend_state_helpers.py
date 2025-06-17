"""Unit tests for frontend state helpers.

Tests the state management functions for the frontend.
"""

import pytest
import streamlit as st
from unittest.mock import Mock, patch, MagicMock
from src.frontend.state_helpers import initialize_session_state


class TestFrontendStateHelpers:
    """Test cases for frontend state helper functions."""

    @pytest.fixture
    def mock_streamlit(self):
        """Mock Streamlit session state."""
        with patch("streamlit.session_state") as mock_session_state:
            # Mock the session state as a dictionary-like object
            mock_session_state.__contains__ = Mock(return_value=False)
            mock_session_state.__setitem__ = Mock()
            mock_session_state.__getitem__ = Mock()
            yield mock_session_state

    def test_initialize_session_state_all_new(self, mock_streamlit):
        """Test initializing session state when all variables are new."""
        # Mock that no variables exist yet
        mock_streamlit.__contains__.return_value = False

        with patch("streamlit.session_state", mock_streamlit):
            initialize_session_state()

        # Verify all expected variables were set
        expected_calls = [
            ("agent_state", None),
            ("run_workflow", False),
            ("processing", False),
            ("session_tokens", 0),
            ("session_token_limit", 100000),
            ("manual_stop", False),
            ("session_id", None),
            ("job_description_input", ""),
            ("cv_text_input", ""),
            ("start_from_scratch_input", False),
        ]

        for key, value in expected_calls:
            mock_streamlit.__setitem__.assert_any_call(key, value)

        # Verify session_id was set to a string (UUID)
        session_id_calls = [
            call
            for call in mock_streamlit.__setitem__.call_args_list
            if call[0][0] == "session_id"
        ]
        assert len(session_id_calls) == 1
        session_id_value = session_id_calls[0][0][1]
        assert isinstance(session_id_value, str)
        assert len(session_id_value) > 0

    def test_initialize_session_state_some_exist(self, mock_streamlit):
        """Test initializing session state when some variables already exist."""
        # Mock that some variables already exist
        existing_vars = {"agent_state", "processing", "session_tokens"}
        mock_streamlit.__contains__.side_effect = lambda key: key in existing_vars

        with patch("streamlit.session_state", mock_streamlit):
            initialize_session_state()

        # Verify only non-existing variables were set
        set_calls = [call[0][0] for call in mock_streamlit.__setitem__.call_args_list]

        # These should NOT be set (they already exist)
        for existing_var in existing_vars:
            assert existing_var not in set_calls

        # These should be set (they don't exist)
        expected_new_vars = {
            "run_workflow",
            "session_token_limit",
            "manual_stop",
            "session_id",
            "job_description_input",
            "cv_text_input",
            "start_from_scratch_input",
        }
        for new_var in expected_new_vars:
            assert new_var in set_calls

    def test_initialize_session_state_all_exist(self, mock_streamlit):
        """Test initializing session state when all variables already exist."""
        # Mock that all variables already exist
        mock_streamlit.__contains__.return_value = True

        with patch("streamlit.session_state", mock_streamlit):
            initialize_session_state()

        # Verify no variables were set (they all already exist)
        mock_streamlit.__setitem__.assert_not_called()

    def test_session_id_generation(self, mock_streamlit):
        """Test that session_id is properly generated."""
        mock_streamlit.__contains__.return_value = False

        with patch("streamlit.session_state", mock_streamlit):
            initialize_session_state()

        # Find the session_id call
        session_id_calls = [
            call
            for call in mock_streamlit.__setitem__.call_args_list
            if call[0][0] == "session_id"
        ]
        assert len(session_id_calls) == 1

        session_id_value = session_id_calls[0][0][1]

        # Verify it's a valid UUID string format
        assert isinstance(session_id_value, str)
        assert len(session_id_value) == 36  # Standard UUID length
        assert session_id_value.count("-") == 4  # Standard UUID format

    def test_default_values(self, mock_streamlit):
        """Test that default values are set correctly."""
        mock_streamlit.__contains__.return_value = False

        with patch("streamlit.session_state", mock_streamlit):
            initialize_session_state()

        # Verify specific default values
        expected_defaults = {
            "agent_state": None,
            "run_workflow": False,
            "processing": False,
            "session_tokens": 0,
            "session_token_limit": 100000,
            "manual_stop": False,
            "job_description_input": "",
            "cv_text_input": "",
            "start_from_scratch_input": False,
        }

        for key, expected_value in expected_defaults.items():
            mock_streamlit.__setitem__.assert_any_call(key, expected_value)

    def test_initialize_session_state_idempotent(self, mock_streamlit):
        """Test that calling initialize_session_state multiple times is safe."""
        # First call - no variables exist
        mock_streamlit.__contains__.return_value = False

        with patch("streamlit.session_state", mock_streamlit):
            initialize_session_state()

        first_call_count = mock_streamlit.__setitem__.call_count

        # Reset mock and simulate second call - all variables now exist
        mock_streamlit.reset_mock()
        mock_streamlit.__contains__.return_value = True

        with patch("streamlit.session_state", mock_streamlit):
            initialize_session_state()

        # Second call should not set any variables
        assert mock_streamlit.__setitem__.call_count == 0

    def test_boolean_defaults(self, mock_streamlit):
        """Test that boolean variables are set to correct default values."""
        mock_streamlit.__contains__.return_value = False

        with patch("streamlit.session_state", mock_streamlit):
            initialize_session_state()

        # Check boolean defaults
        boolean_vars = {
            "run_workflow": False,
            "processing": False,
            "manual_stop": False,
            "start_from_scratch_input": False,
        }

        for var_name, expected_value in boolean_vars.items():
            mock_streamlit.__setitem__.assert_any_call(var_name, expected_value)

    def test_numeric_defaults(self, mock_streamlit):
        """Test that numeric variables are set to correct default values."""
        mock_streamlit.__contains__.return_value = False

        with patch("streamlit.session_state", mock_streamlit):
            initialize_session_state()

        # Check numeric defaults
        numeric_vars = {"session_tokens": 0, "session_token_limit": 100000}

        for var_name, expected_value in numeric_vars.items():
            mock_streamlit.__setitem__.assert_any_call(var_name, expected_value)

    def test_string_defaults(self, mock_streamlit):
        """Test that string variables are set to correct default values."""
        mock_streamlit.__contains__.return_value = False

        with patch("streamlit.session_state", mock_streamlit):
            initialize_session_state()

        # Check string defaults
        string_vars = {"job_description_input": "", "cv_text_input": ""}

        for var_name, expected_value in string_vars.items():
            mock_streamlit.__setitem__.assert_any_call(var_name, expected_value)

    @patch("uuid.uuid4")
    def test_session_id_uniqueness(self, mock_uuid, mock_streamlit):
        """Test that session_id uses UUID generation."""
        mock_uuid.return_value.hex = "test_uuid_hex_value"
        mock_streamlit.__contains__.return_value = False

        with patch("streamlit.session_state", mock_streamlit):
            initialize_session_state()

        # Verify UUID was called
        mock_uuid.assert_called_once()

        # Verify session_id was set to the UUID hex value
        mock_streamlit.__setitem__.assert_any_call("session_id", "test_uuid_hex_value")
