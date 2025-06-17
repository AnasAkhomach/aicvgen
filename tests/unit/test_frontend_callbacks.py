"""Unit tests for frontend callbacks.

Tests the callback functions that handle user interactions in the frontend.
"""

import pytest
import streamlit as st
from unittest.mock import Mock, patch, MagicMock
from src.frontend.callbacks import handle_user_action
from src.orchestration.state import AgentState, UserFeedback
from src.models.data_models import StructuredCV, Section, Item, ItemStatus


class TestFrontendCallbacks:
    """Test cases for frontend callback functions."""

    @pytest.fixture
    def mock_streamlit(self):
        """Mock Streamlit session state."""
        with patch("streamlit.session_state") as mock_session_state:
            mock_session_state.get.return_value = None
            yield mock_session_state

    @pytest.fixture
    def sample_agent_state(self):
        """Create a sample agent state for testing."""
        item1 = Item(
            id="test_item_1", content="Test experience item", status=ItemStatus.PENDING
        )

        section1 = Section(name="Professional Experience", items=[item1])

        structured_cv = StructuredCV(sections=[section1], metadata={"test": "data"})

        return AgentState(
            structured_cv=structured_cv,
            current_step="review",
            processing_status="completed",
        )

    @patch("streamlit.success")
    @patch("streamlit.rerun")
    def test_handle_user_action_accept(
        self, mock_rerun, mock_success, mock_streamlit, sample_agent_state
    ):
        """Test handling accept action."""
        mock_streamlit.get.return_value = sample_agent_state

        with patch("streamlit.session_state", mock_streamlit):
            handle_user_action("accept", "test_item_1")

        # Verify that agent state was updated
        assert mock_streamlit.agent_state.user_feedback is not None
        assert mock_streamlit.agent_state.user_feedback.feedback_type == "accept"
        assert mock_streamlit.agent_state.user_feedback.data["item_id"] == "test_item_1"

        # Verify workflow flag was set
        assert mock_streamlit.run_workflow is True

        # Verify UI feedback
        mock_success.assert_called_with("✅ Item accepted")
        mock_rerun.assert_called_once()

    @patch("streamlit.info")
    @patch("streamlit.rerun")
    def test_handle_user_action_regenerate(
        self, mock_rerun, mock_info, mock_streamlit, sample_agent_state
    ):
        """Test handling regenerate action."""
        mock_streamlit.get.return_value = sample_agent_state

        with patch("streamlit.session_state", mock_streamlit):
            handle_user_action("regenerate", "test_item_1")

        # Verify that agent state was updated
        assert mock_streamlit.agent_state.user_feedback is not None
        assert mock_streamlit.agent_state.user_feedback.feedback_type == "regenerate"
        assert mock_streamlit.agent_state.user_feedback.data["item_id"] == "test_item_1"

        # Verify workflow flag was set
        assert mock_streamlit.run_workflow is True

        # Verify UI feedback
        mock_info.assert_called_with("🔄 Regenerating item...")
        mock_rerun.assert_called_once()

    @patch("streamlit.error")
    def test_handle_user_action_no_agent_state(self, mock_error, mock_streamlit):
        """Test handling action when no agent state exists."""
        mock_streamlit.get.return_value = None

        with patch("streamlit.session_state", mock_streamlit):
            handle_user_action("accept", "test_item_1")

        mock_error.assert_called_with("No agent state found")

    @patch("streamlit.error")
    def test_handle_user_action_unknown_action(
        self, mock_error, mock_streamlit, sample_agent_state
    ):
        """Test handling unknown action type."""
        mock_streamlit.get.return_value = sample_agent_state

        with patch("streamlit.session_state", mock_streamlit):
            handle_user_action("unknown_action", "test_item_1")

        mock_error.assert_called_with("Unknown action: unknown_action")

    def test_user_feedback_object_creation(self, mock_streamlit, sample_agent_state):
        """Test that UserFeedback object is created correctly."""
        mock_streamlit.get.return_value = sample_agent_state

        with patch("streamlit.session_state", mock_streamlit), patch(
            "streamlit.success"
        ), patch("streamlit.rerun"):
            handle_user_action("accept", "test_item_1")

        user_feedback = mock_streamlit.agent_state.user_feedback
        assert isinstance(user_feedback, UserFeedback)
        assert user_feedback.feedback_type == "accept"
        assert user_feedback.data == {"item_id": "test_item_1"}

    def test_workflow_flag_setting(self, mock_streamlit, sample_agent_state):
        """Test that workflow flag is properly set."""
        mock_streamlit.get.return_value = sample_agent_state
        mock_streamlit.run_workflow = False  # Initially false

        with patch("streamlit.session_state", mock_streamlit), patch(
            "streamlit.success"
        ), patch("streamlit.rerun"):
            handle_user_action("regenerate", "test_item_1")

        assert mock_streamlit.run_workflow is True

    @patch("streamlit.success")
    @patch("streamlit.rerun")
    def test_multiple_actions_sequence(
        self, mock_rerun, mock_success, mock_streamlit, sample_agent_state
    ):
        """Test handling multiple actions in sequence."""
        mock_streamlit.get.return_value = sample_agent_state

        with patch("streamlit.session_state", mock_streamlit):
            # First action
            handle_user_action("accept", "item_1")
            first_feedback = mock_streamlit.agent_state.user_feedback

            # Second action (should overwrite the first)
            handle_user_action("regenerate", "item_2")
            second_feedback = mock_streamlit.agent_state.user_feedback

        # Verify that the second feedback overwrote the first
        assert second_feedback.feedback_type == "regenerate"
        assert second_feedback.data["item_id"] == "item_2"
        assert second_feedback != first_feedback

        # Verify both actions triggered rerun
        assert mock_rerun.call_count == 2

    def test_agent_state_persistence(self, mock_streamlit, sample_agent_state):
        """Test that agent state is properly updated and persisted."""
        original_step = sample_agent_state.current_step
        original_status = sample_agent_state.processing_status

        mock_streamlit.get.return_value = sample_agent_state

        with patch("streamlit.session_state", mock_streamlit), patch(
            "streamlit.success"
        ), patch("streamlit.rerun"):
            handle_user_action("accept", "test_item_1")

        # Verify that other state properties are preserved
        updated_state = mock_streamlit.agent_state
        assert updated_state.current_step == original_step
        assert updated_state.processing_status == original_status
        assert updated_state.user_feedback is not None
