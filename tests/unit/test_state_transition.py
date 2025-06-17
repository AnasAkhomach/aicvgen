"""Unit tests for UI-to-backend state transition functionality."""

import pytest
from unittest.mock import patch, MagicMock
import streamlit as st
from src.core.main import create_agent_state_from_ui
from src.orchestration.state import AgentState
from src.models.data_models import JobDescriptionData, StructuredCV


class TestCreateAgentStateFromUI:
    """Test cases for the create_agent_state_from_ui function."""

    @patch("streamlit.session_state")
    def test_create_agent_state_with_valid_inputs(self, mock_session_state):
        """Test creating agent state with valid job description and CV text."""
        # Arrange
        mock_session_state.get.side_effect = lambda key, default=None: {
            "job_description": "Software Engineer position requiring Python skills",
            "cv_text": "John Doe\nSoftware Developer\nPython, JavaScript",
        }.get(key, default)

        # Act
        result = create_agent_state_from_ui()

        # Assert
        assert isinstance(result, AgentState)
        assert isinstance(result.job_description_data, JobDescriptionData)
        assert isinstance(result.structured_cv, StructuredCV)
        assert (
            result.job_description_data.raw_text
            == "Software Engineer position requiring Python skills"
        )
        assert (
            result.structured_cv.raw_text
            == "John Doe\nSoftware Developer\nPython, JavaScript"
        )
        assert result.current_section_key == ""
        assert result.current_item_id == ""
        assert result.items_to_process_queue == []
        assert result.error_messages == []
        assert result.user_feedback is None

    @patch("streamlit.session_state")
    def test_create_agent_state_with_empty_inputs(self, mock_session_state):
        """Test creating agent state with empty inputs."""
        # Arrange
        mock_session_state.get.side_effect = lambda key, default=None: {
            "job_description": "",
            "cv_text": "",
        }.get(key, default)

        # Act
        result = create_agent_state_from_ui()

        # Assert
        assert isinstance(result, AgentState)
        assert result.job_description_data.raw_text == ""
        assert result.structured_cv.raw_text == ""

    @patch("streamlit.session_state")
    def test_create_agent_state_with_missing_keys(self, mock_session_state):
        """Test creating agent state when session state keys are missing."""
        # Arrange
        mock_session_state.get.side_effect = lambda key, default=None: {
            # Missing both keys
        }.get(key, default)

        # Act
        result = create_agent_state_from_ui()

        # Assert
        assert isinstance(result, AgentState)
        assert result.job_description_data.raw_text is None
        assert result.structured_cv.raw_text is None

    @patch("streamlit.session_state")
    def test_create_agent_state_default_values(self, mock_session_state):
        """Test that default values are properly set in the agent state."""
        # Arrange
        mock_session_state.get.side_effect = lambda key, default=None: {
            "job_description": "Test job",
            "cv_text": "Test CV",
        }.get(key, default)

        # Act
        result = create_agent_state_from_ui()

        # Assert
        # Test all default values are properly initialized
        assert result.current_section_key == ""
        assert result.current_item_id == ""
        assert result.items_to_process_queue == []
        assert result.error_messages == []
        assert result.user_feedback is None
        assert result.generated_content == {}
        assert result.qa_results == {}
        assert result.research_results == {}
        assert result.final_cv_data is None

    @patch("streamlit.session_state")
    def test_create_agent_state_data_types(self, mock_session_state):
        """Test that the correct data types are created."""
        # Arrange
        test_job_desc = "Senior Python Developer"
        test_cv_text = "Jane Smith\nSenior Developer\n5 years experience"

        mock_session_state.get.side_effect = lambda key, default=None: {
            "job_description": test_job_desc,
            "cv_text": test_cv_text,
        }.get(key, default)

        # Act
        result = create_agent_state_from_ui()

        # Assert
        assert isinstance(result.job_description_data, JobDescriptionData)
        assert isinstance(result.structured_cv, StructuredCV)
        assert isinstance(result.current_section_key, str)
        assert isinstance(result.current_item_id, str)
        assert isinstance(result.items_to_process_queue, list)
        assert isinstance(result.error_messages, list)
        assert isinstance(result.generated_content, dict)
        assert isinstance(result.qa_results, dict)
        assert isinstance(result.research_results, dict)

    @patch("streamlit.session_state")
    def test_create_agent_state_with_whitespace_inputs(self, mock_session_state):
        """Test creating agent state with whitespace-only inputs."""
        # Arrange
        mock_session_state.get.side_effect = lambda key, default=None: {
            "job_description": "   \n\t   ",
            "cv_text": "\n   \t\n   ",
        }.get(key, default)

        # Act
        result = create_agent_state_from_ui()

        # Assert
        assert isinstance(result, AgentState)
        assert result.job_description_data.raw_text == "   \n\t   "
        assert result.structured_cv.raw_text == "\n   \t\n   "

    @patch("streamlit.session_state")
    def test_create_agent_state_preserves_input_formatting(self, mock_session_state):
        """Test that input formatting is preserved in the agent state."""
        # Arrange
        job_desc_with_formatting = """Software Engineer Position

        Requirements:
        - Python 3.8+
        - Django/Flask
        - PostgreSQL

        Nice to have:
        - Docker
        - AWS"""

        cv_with_formatting = """John Doe
        Software Developer

        Experience:
        • 5 years Python development
        • 3 years Django
        • 2 years AWS

        Education:
        • BS Computer Science"""

        mock_session_state.get.side_effect = lambda key, default=None: {
            "job_description": job_desc_with_formatting,
            "cv_text": cv_with_formatting,
        }.get(key, default)

        # Act
        result = create_agent_state_from_ui()

        # Assert
        assert result.job_description_data.raw_text == job_desc_with_formatting
        assert result.structured_cv.raw_text == cv_with_formatting
        # Verify specific formatting elements are preserved
        assert (
            "\n        \n        Requirements:" in result.job_description_data.raw_text
        )
        assert "• 5 years Python development" in result.structured_cv.raw_text
