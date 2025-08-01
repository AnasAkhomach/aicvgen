"""Tests for status-driven UI rendering in UIManager."""

import pytest
from unittest.mock import Mock, patch, MagicMock, mock_open
import streamlit as st

from src.frontend.ui_manager import UIManager
from src.core.state_manager import StateManager
from src.orchestration.state import AgentState


class TestUIManagerStatusDriven:
    """Test class for status-driven UI rendering functionality."""

    @pytest.fixture
    def mock_state_manager(self):
        """Create a mock StateManager."""
        return Mock(spec=StateManager)

    @pytest.fixture
    def ui_manager(self, mock_state_manager):
        """Create UIManager instance with mocked dependencies."""
        with patch("src.frontend.ui_manager.st.set_page_config"):
            return UIManager(mock_state_manager)

    @pytest.fixture
    def mock_agent_state(self):
        """Create a mock AgentState."""
        state = {
            "workflow_status": "PROCESSING",
            "ui_display_data": {},
            "final_output_path": None,
            "error_messages": [],
        }
        return state

    @patch("src.frontend.ui_manager.st")
    def test_render_status_driven_ui_processing(
        self, mock_st, ui_manager, mock_agent_state
    ):
        """Test render_status_driven_ui for PROCESSING status."""
        mock_agent_state["workflow_status"] = "PROCESSING"

        with patch.object(ui_manager, "render_processing_indicator") as mock_render:
            ui_manager.render_status_driven_ui("PROCESSING", mock_agent_state)

            mock_render.assert_called_once()

    def test_render_status_driven_ui_awaiting_feedback(
        self, ui_manager, mock_agent_state
    ):
        """Test render_status_driven_ui for AWAITING_FEEDBACK status."""
        mock_agent_state["workflow_status"] = "AWAITING_FEEDBACK"
        mock_agent_state["ui_display_data"] = {
            "generated_summary": "Test summary content",
            "skills_analysis": {"technical": ["Python", "AI"]},
        }

        with patch.object(ui_manager, "_render_awaiting_feedback_ui") as mock_render:
            ui_manager.render_status_driven_ui("AWAITING_FEEDBACK", mock_agent_state)

            mock_render.assert_called_once_with(mock_agent_state)

    def test_render_status_driven_ui_completed(self, ui_manager, mock_agent_state):
        """Test render_status_driven_ui for COMPLETED status."""
        mock_agent_state["workflow_status"] = "COMPLETED"
        mock_agent_state["final_output_path"] = "/path/to/cv.pdf"

        with patch.object(ui_manager, "_render_completed_ui") as mock_render:
            ui_manager.render_status_driven_ui("COMPLETED", mock_agent_state)

            mock_render.assert_called_once_with(mock_agent_state)

    def test_render_status_driven_ui_error(self, ui_manager, mock_agent_state):
        """Test render_status_driven_ui for ERROR status."""
        mock_agent_state["workflow_status"] = "ERROR"
        mock_agent_state["error_messages"] = ["Test error message"]

        with patch.object(ui_manager, "_render_error_ui") as mock_render:
            ui_manager.render_status_driven_ui("ERROR", mock_agent_state)

            mock_render.assert_called_once_with(mock_agent_state)

    @patch("src.frontend.ui_manager.st")
    def test_render_awaiting_feedback_ui(self, mock_st, ui_manager, mock_agent_state):
        """Test _render_awaiting_feedback_ui method."""
        mock_agent_state["generated_content"] = "Test generated content"

        # Mock streamlit components
        mock_st.info = Mock()
        mock_st.subheader = Mock()
        mock_st.markdown = Mock()

        # Mock columns context manager
        mock_col1 = Mock()
        mock_col2 = Mock()
        mock_col1.__enter__ = Mock(return_value=mock_col1)
        mock_col1.__exit__ = Mock(return_value=None)
        mock_col2.__enter__ = Mock(return_value=mock_col2)
        mock_col2.__exit__ = Mock(return_value=None)
        mock_st.columns.return_value = [mock_col1, mock_col2]
        mock_st.button.return_value = False

        ui_manager._render_awaiting_feedback_ui(mock_agent_state)

        # Verify UI components were called
        mock_st.info.assert_called_with(
            "Please review the generated content and provide your feedback."
        )
        mock_st.subheader.assert_called_with("🔍 Review Generated Content")
        assert mock_st.markdown.called

    @patch("src.frontend.ui_manager.st")
    def test_render_completed_ui_with_file(self, mock_st, ui_manager, mock_agent_state):
        """Test _render_completed_ui method when file is available."""
        mock_agent_state["final_output_path"] = "/path/to/cv.pdf"
        mock_agent_state["pdf_content"] = b"fake pdf content"

        # Mock streamlit components
        mock_st.success = Mock()
        mock_st.markdown = Mock()
        mock_st.download_button = Mock()

        ui_manager._render_completed_ui(mock_agent_state)

        # Verify UI components were called
        mock_st.success.assert_called_with("🎉 CV Generation Completed Successfully!")
        mock_st.download_button.assert_called_with(
            label="📄 Download PDF",
            data=b"fake pdf content",
            file_name="optimized_cv.pdf",
            mime="application/pdf",
            type="primary",
        )

    @patch("src.frontend.ui_manager.st")
    def test_render_error_ui(self, mock_st, ui_manager, mock_agent_state):
        """Test _render_error_ui method."""
        mock_agent_state["error_messages"] = ["Error 1", "Error 2"]

        # Mock streamlit components
        mock_st.error = Mock()
        mock_st.code = Mock()
        mock_st.button.return_value = False
        mock_st.expander = Mock()

        # Mock expander context manager
        mock_expander = Mock()
        mock_expander.__enter__ = Mock(return_value=mock_expander)
        mock_expander.__exit__ = Mock(return_value=None)
        mock_st.expander.return_value = mock_expander

        ui_manager._render_error_ui(mock_agent_state)

        # Verify UI components were called
        mock_st.error.assert_called_with("❌ An error occurred during CV generation")
        mock_st.expander.assert_called_with("Error Details", expanded=False)
        mock_st.button.assert_called()

    @patch("src.frontend.ui_manager.st")
    def test_handle_approve_action(self, mock_st, ui_manager, mock_agent_state):
        """Test _handle_approve_action method."""
        # Setup mocks
        mock_st.success = Mock()
        mock_st.rerun = Mock()

        # Mock state manager
        mock_state_manager = Mock()
        mock_state_manager.get_workflow_session_id.return_value = "test_session_id"
        ui_manager.state = mock_state_manager

        # Mock facade
        mock_facade = Mock()
        mock_facade.submit_user_feedback.return_value = True
        ui_manager.facade = mock_facade

        ui_manager._handle_approve_action(mock_agent_state)

        # Verify facade was called
        mock_facade.submit_user_feedback.assert_called()
        mock_st.success.assert_called_with("Feedback submitted successfully!")

    @patch("src.frontend.ui_manager.st")
    def test_handle_regenerate_action(self, mock_st, ui_manager, mock_agent_state):
        """Test _handle_regenerate_action method."""
        # Setup mocks
        mock_st.success = Mock()
        mock_st.rerun = Mock()

        # Mock state manager
        mock_state_manager = Mock()
        mock_state_manager.get_workflow_session_id.return_value = "test_session_id"
        ui_manager.state = mock_state_manager

        # Mock facade
        mock_facade = Mock()
        mock_facade.submit_user_feedback.return_value = True
        ui_manager.facade = mock_facade

        ui_manager._handle_regenerate_action(mock_agent_state)

        # Verify facade was called
        mock_facade.submit_user_feedback.assert_called()
        mock_st.success.assert_called_with("Regeneration request submitted!")

    @patch("src.frontend.ui_manager.st")
    def test_handle_restart_workflow(self, mock_st, ui_manager):
        """Test _handle_restart_workflow method."""
        # Setup mocks
        mock_st.success = Mock()
        mock_st.rerun = Mock()

        mock_state_manager = Mock()
        ui_manager.state = mock_state_manager

        ui_manager._handle_restart_workflow()

        # Verify state was cleared
        mock_state_manager.clear_workflow_state.assert_called()
        mock_st.success.assert_called_with("Workflow restarted successfully!")

    @patch("src.frontend.ui_manager.st")
    def test_render_status_driven_ui_unknown_status(
        self, mock_st, ui_manager, mock_agent_state
    ):
        """Test render_status_driven_ui for unknown status."""
        mock_agent_state["workflow_status"] = "UNKNOWN"

        ui_manager.render_status_driven_ui("UNKNOWN", mock_agent_state)

        # Verify info message was displayed
        mock_st.info.assert_called_with("Workflow status: UNKNOWN")
