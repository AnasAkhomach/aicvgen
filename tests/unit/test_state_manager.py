"""Unit tests for StateManager class.

These tests verify the StateManager's state initialization, manipulation,
and property access methods without requiring a running Streamlit app.
"""

from unittest.mock import Mock, patch
from src.core.state_manager import StateManager


class TestStateManager:
    """Test suite for StateManager class."""

    @patch("streamlit.session_state", new_callable=lambda: {})
    def test_initialize_state_sets_defaults(self, mock_session_state):
        """Test that _initialize_state sets all default values correctly."""
        # Act
        StateManager()

        # Assert
        expected_defaults = {
            "agent_state": None,
            "user_gemini_api_key": "",
            "uploaded_cv_file": None,
            "uploaded_job_file": None,
            "cv_text": None,
            "job_description_text": None,
            "is_processing": False,
            "just_finished": False,
            "workflow_error": None,
            "current_tab": 0,
            "session_id": None,
        }

        for key, expected_value in expected_defaults.items():
            assert mock_session_state[key] == expected_value

    @patch("streamlit.session_state", new_callable=lambda: {})
    def test_initialize_state_preserves_existing_values(self, mock_session_state):
        """Test that existing session state values are not overwritten."""
        # Arrange
        mock_session_state["user_gemini_api_key"] = "existing_key"
        mock_session_state["cv_text"] = "existing_cv"

        # Act
        StateManager()

        # Assert
        assert mock_session_state["user_gemini_api_key"] == "existing_key"
        assert mock_session_state["cv_text"] == "existing_cv"

    @patch("streamlit.session_state", new_callable=lambda: {})
    def test_get_returns_correct_value(self, mock_session_state):
        """Test that get method returns the correct value."""
        # Arrange
        mock_session_state["test_key"] = "test_value"
        state_manager = StateManager()

        # Act & Assert
        assert state_manager.get("test_key") == "test_value"

    @patch("streamlit.session_state", new_callable=lambda: {})
    def test_get_returns_default_for_missing_key(self, _):
        """Test that get method returns default value for missing keys."""
        # Arrange
        state_manager = StateManager()

        # Act & Assert
        assert state_manager.get("missing_key", "default") == "default"
        assert state_manager.get("missing_key") is None

    @patch("streamlit.session_state", new_callable=lambda: {})
    def test_set_updates_session_state(self, mock_session_state):
        """Test that set method updates session state correctly."""
        # Arrange
        state_manager = StateManager()

        # Act
        state_manager.set("test_key", "test_value")

        # Assert
        assert mock_session_state["test_key"] == "test_value"

    @patch("streamlit.session_state", new_callable=lambda: {})
    def test_clear_key_removes_existing_key(self, mock_session_state):
        """Test that clear_key removes existing keys."""
        # Arrange
        mock_session_state["test_key"] = "test_value"
        state_manager = StateManager()

        # Act
        state_manager.clear_key("test_key")

        # Assert
        assert "test_key" not in mock_session_state

    @patch("streamlit.session_state", new_callable=lambda: {})
    def test_clear_key_handles_missing_key(self, _):
        """Test that clear_key handles missing keys gracefully."""
        # Arrange
        state_manager = StateManager()

        # Act & Assert (should not raise exception)
        state_manager.clear_key("missing_key")

    @patch("streamlit.session_state", new_callable=lambda: {})
    def test_reset_processing_state(self, _):
        """Test that reset_processing_state resets all processing flags."""
        # Arrange
        state_manager = StateManager()
        state_manager.set("is_processing", True)
        state_manager.set("just_finished", True)
        state_manager.set("workflow_error", "Some error")

        # Act
        state_manager.reset_processing_state()

        # Assert
        assert state_manager.get("is_processing") is False
        assert state_manager.get("just_finished") is False
        assert state_manager.get("workflow_error") is None

    @patch("streamlit.session_state", new_callable=lambda: {})
    def test_agent_state_property_getter(self, mock_session_state):
        """Test agent_state property getter."""
        # Arrange
        mock_agent_state = Mock()
        mock_session_state["agent_state"] = mock_agent_state
        state_manager = StateManager()

        # Act & Assert
        assert state_manager.agent_state == mock_agent_state

    @patch("streamlit.session_state", new_callable=lambda: {})
    def test_agent_state_property_setter(self, mock_session_state):
        """Test agent_state property setter."""
        # Arrange
        mock_agent_state = Mock()
        state_manager = StateManager()

        # Act
        state_manager.agent_state = mock_agent_state

        # Assert
        assert mock_session_state["agent_state"] == mock_agent_state

    @patch("streamlit.session_state", new_callable=lambda: {})
    def test_user_gemini_api_key_property(self, mock_session_state):
        """Test user_gemini_api_key property getter and setter."""
        # Arrange
        state_manager = StateManager()

        # Act
        state_manager.user_gemini_api_key = "test_key"

        # Assert
        assert state_manager.user_gemini_api_key == "test_key"
        assert mock_session_state["user_gemini_api_key"] == "test_key"

    @patch("streamlit.session_state", new_callable=lambda: {})
    def test_cv_text_property(self, mock_session_state):
        """Test cv_text property getter and setter."""
        # Arrange
        state_manager = StateManager()

        # Act
        state_manager.cv_text = "test cv content"

        # Assert
        assert state_manager.cv_text == "test cv content"
        assert mock_session_state["cv_text"] == "test cv content"

    @patch("streamlit.session_state", new_callable=lambda: {})
    def test_job_description_text_property(self, mock_session_state):
        """Test job_description_text property getter and setter."""
        # Arrange
        state_manager = StateManager()

        # Act
        state_manager.job_description_text = "test job description"

        # Assert
        assert state_manager.job_description_text == "test job description"
        assert mock_session_state["job_description_text"] == "test job description"

    @patch("streamlit.session_state", new_callable=lambda: {})
    def test_is_processing_property(self, mock_session_state):
        """Test is_processing property getter and setter."""
        # Arrange
        state_manager = StateManager()

        # Act
        state_manager.is_processing = True

        # Assert
        assert state_manager.is_processing is True
        assert mock_session_state["is_processing"] is True

    @patch("streamlit.session_state", new_callable=lambda: {})
    def test_just_finished_property(self, mock_session_state):
        """Test just_finished property getter and setter."""
        # Arrange
        state_manager = StateManager()

        # Act
        state_manager.just_finished = True

        # Assert
        assert state_manager.just_finished is True
        assert mock_session_state["just_finished"] is True

    @patch("streamlit.session_state", new_callable=lambda: {})
    def test_workflow_error_property(self, mock_session_state):
        """Test workflow_error property getter and setter."""
        # Arrange
        state_manager = StateManager()

        # Act
        state_manager.workflow_error = "Test error"

        # Assert
        assert state_manager.workflow_error == "Test error"
        assert mock_session_state["workflow_error"] == "Test error"

    @patch("streamlit.session_state", new_callable=lambda: {})
    def test_has_required_data_with_both_texts(self, _):
        """Test has_required_data returns True when both CV and job description are present."""
        # Arrange
        state_manager = StateManager()
        state_manager.cv_text = "cv content"
        state_manager.job_description_text = "job description"

        # Act & Assert
        assert state_manager.has_required_data() is True

    @patch("streamlit.session_state", new_callable=lambda: {})
    def test_has_required_data_with_missing_cv(self, _):
        """Test has_required_data returns False when CV text is missing."""
        # Arrange
        state_manager = StateManager()
        state_manager.job_description_text = "job description"

        # Act & Assert
        assert state_manager.has_required_data() is False

    @patch("streamlit.session_state", new_callable=lambda: {})
    def test_has_required_data_with_missing_job(self, _):
        """Test has_required_data returns False when job description is missing."""
        # Arrange
        state_manager = StateManager()
        state_manager.cv_text = "cv content"

        # Act & Assert
        assert state_manager.has_required_data() is False

    @patch("streamlit.session_state", new_callable=lambda: {})
    def test_get_state_summary(self, _):
        """Test get_state_summary returns correct summary."""
        # Arrange
        mock_agent_state = Mock()
        state_manager = StateManager()
        state_manager.agent_state = mock_agent_state
        state_manager.cv_text = "cv content"
        state_manager.job_description_text = "job description"
        state_manager.is_processing = True
        state_manager.just_finished = False
        state_manager.workflow_error = None
        state_manager.set("session_id", "test_session")

        # Act
        summary = state_manager.get_state_summary()

        # Assert
        expected_summary = {
            "has_agent_state": True,
            "has_cv_text": True,
            "has_job_description": True,
            "is_processing": True,
            "just_finished": False,
            "has_workflow_error": False,
            "session_id": "test_session",
        }
        assert summary == expected_summary
