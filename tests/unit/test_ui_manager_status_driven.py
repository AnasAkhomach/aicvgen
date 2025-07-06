"""Tests for status-driven UI rendering in UIManager."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import streamlit as st

from src.ui.ui_manager import UIManager
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
        with patch('src.ui.ui_manager.st.set_page_config'):
            return UIManager(mock_state_manager)

    @pytest.fixture
    def mock_agent_state(self):
        """Create a mock AgentState."""
        state = Mock(spec=AgentState)
        state.workflow_status = "PROCESSING"
        state.ui_display_data = {}
        state.final_output_path = None
        state.error_messages = []
        return state

    @patch('src.ui.ui_manager.st.session_state')
    def test_render_status_driven_ui_no_session_id(self, mock_session_state, ui_manager):
        """Test render_status_driven_ui when no workflow_session_id exists."""
        mock_session_state.get.return_value = None
        
        # Should return early without error
        ui_manager.render_status_driven_ui()
        
        mock_session_state.get.assert_called_once_with("workflow_session_id")

    @patch('src.ui.ui_manager.get_container')
    @patch('src.ui.ui_manager.st.session_state')
    def test_render_status_driven_ui_awaiting_feedback(self, mock_session_state, mock_get_container, ui_manager, mock_agent_state):
        """Test render_status_driven_ui for AWAITING_FEEDBACK status."""
        # Setup mocks
        mock_session_state.get.return_value = "test_session_id"
        mock_agent_state.workflow_status = "AWAITING_FEEDBACK"
        mock_agent_state.ui_display_data = {
            "generated_summary": "Test summary content",
            "skills_analysis": {"technical": ["Python", "AI"]}
        }
        
        mock_container = Mock()
        mock_workflow_manager = Mock()
        mock_workflow_manager.get_workflow_status.return_value = mock_agent_state
        mock_container.workflow_manager.return_value = mock_workflow_manager
        mock_get_container.return_value = mock_container
        
        with patch.object(ui_manager, '_render_awaiting_feedback_ui') as mock_render:
            ui_manager.render_status_driven_ui()
            
            mock_render.assert_called_once_with(mock_agent_state)

    @patch('src.ui.ui_manager.get_container')
    @patch('src.ui.ui_manager.st.session_state')
    def test_render_status_driven_ui_completed(self, mock_session_state, mock_get_container, ui_manager, mock_agent_state):
        """Test render_status_driven_ui for COMPLETED status."""
        # Setup mocks
        mock_session_state.get.return_value = "test_session_id"
        mock_agent_state.workflow_status = "COMPLETED"
        mock_agent_state.final_output_path = "/path/to/cv.pdf"
        
        mock_container = Mock()
        mock_workflow_manager = Mock()
        mock_workflow_manager.get_workflow_status.return_value = mock_agent_state
        mock_container.workflow_manager.return_value = mock_workflow_manager
        mock_get_container.return_value = mock_container
        
        with patch.object(ui_manager, '_render_completed_ui') as mock_render:
            ui_manager.render_status_driven_ui()
            
            mock_render.assert_called_once_with(mock_agent_state)

    @patch('src.ui.ui_manager.get_container')
    @patch('src.ui.ui_manager.st.session_state')
    def test_render_status_driven_ui_error(self, mock_session_state, mock_get_container, ui_manager, mock_agent_state):
        """Test render_status_driven_ui for ERROR status."""
        # Setup mocks
        mock_session_state.get.return_value = "test_session_id"
        mock_agent_state.workflow_status = "ERROR"
        mock_agent_state.error_messages = ["Test error message"]
        
        mock_container = Mock()
        mock_workflow_manager = Mock()
        mock_workflow_manager.get_workflow_status.return_value = mock_agent_state
        mock_container.workflow_manager.return_value = mock_workflow_manager
        mock_get_container.return_value = mock_container
        
        with patch.object(ui_manager, '_render_error_ui') as mock_render:
            ui_manager.render_status_driven_ui()
            
            mock_render.assert_called_once_with(mock_agent_state)

    @patch('src.ui.ui_manager.st')
    def test_render_awaiting_feedback_ui(self, mock_st, ui_manager, mock_agent_state):
        """Test _render_awaiting_feedback_ui method."""
        mock_agent_state.ui_display_data = {
            "summary": "Test summary",
            "skills": {"technical": ["Python"]}
        }
        
        # Mock streamlit components
        mock_st.info = Mock()
        mock_st.subheader = Mock()
        mock_st.markdown = Mock()
        mock_st.json = Mock()
        
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
        mock_st.info.assert_called_once_with("🔄 Workflow is awaiting your feedback")
        mock_st.subheader.assert_called_once_with("Review Generated Content")
        assert mock_st.markdown.call_count >= 2  # Called for each key
        mock_st.json.assert_called_once()  # Called for dict value

    @patch('src.ui.ui_manager.st')
    @patch('os.path.exists')
    @patch('builtins.open')
    def test_render_completed_ui_with_file(self, mock_open, mock_exists, mock_st, ui_manager, mock_agent_state):
        """Test _render_completed_ui method when PDF file exists."""
        mock_agent_state.final_output_path = "/path/to/cv.pdf"
        mock_exists.return_value = True
        
        # Mock file content
        mock_file = Mock()
        mock_file.read.return_value = b"PDF content"
        mock_open.return_value.__enter__.return_value = mock_file
        
        # Mock streamlit components
        mock_st.success = Mock()
        mock_st.subheader = Mock()
        mock_st.download_button = Mock()
        
        ui_manager._render_completed_ui(mock_agent_state)
        
        # Verify UI components were called
        mock_st.success.assert_called_once_with("✅ CV Generation Completed!")
        mock_st.subheader.assert_called_once_with("Download Your CV")
        mock_st.download_button.assert_called_once()

    @patch('src.ui.ui_manager.st')
    def test_render_error_ui(self, mock_st, ui_manager, mock_agent_state):
        """Test _render_error_ui method."""
        mock_agent_state.error_messages = ["Error 1", "Error 2"]
        
        # Mock streamlit components
        mock_st.error = Mock()
        mock_st.subheader = Mock()
        mock_st.button.return_value = False
        
        ui_manager._render_error_ui(mock_agent_state)
        
        # Verify UI components were called
        mock_st.error.assert_called()  # Called multiple times
        mock_st.subheader.assert_called_once_with("Error Details")
        mock_st.button.assert_called_once()

    @patch('src.ui.ui_manager.get_container')
    @patch('src.ui.ui_manager.st')
    def test_handle_approve_action(self, mock_st, mock_get_container, ui_manager, mock_agent_state):
        """Test _handle_approve_action method."""
        # Setup mocks
        mock_st.session_state.get.return_value = "test_session_id"
        mock_st.success = Mock()
        mock_st.rerun = Mock()
        
        mock_container = Mock()
        mock_workflow_manager = Mock()
        mock_container.workflow_manager.return_value = mock_workflow_manager
        mock_get_container.return_value = mock_container
        
        ui_manager._handle_approve_action(mock_agent_state)
        
        # Verify workflow manager was called
        mock_workflow_manager.send_feedback.assert_called_once_with(
            session_id="test_session_id",
            feedback_type="approve",
            feedback_data={"action": "approve"}
        )
        mock_st.success.assert_called_once()
        mock_st.rerun.assert_called_once()

    @patch('src.ui.ui_manager.get_container')
    @patch('src.ui.ui_manager.st')
    def test_handle_regenerate_action(self, mock_st, mock_get_container, ui_manager, mock_agent_state):
        """Test _handle_regenerate_action method."""
        # Setup mocks
        mock_st.session_state.get.return_value = "test_session_id"
        mock_st.success = Mock()
        mock_st.rerun = Mock()
        
        mock_container = Mock()
        mock_workflow_manager = Mock()
        mock_container.workflow_manager.return_value = mock_workflow_manager
        mock_get_container.return_value = mock_container
        
        ui_manager._handle_regenerate_action(mock_agent_state)
        
        # Verify workflow manager was called
        mock_workflow_manager.send_feedback.assert_called_once_with(
            session_id="test_session_id",
            feedback_type="regenerate",
            feedback_data={"action": "regenerate"}
        )
        mock_st.success.assert_called_once()
        mock_st.rerun.assert_called_once()

    @patch('src.ui.ui_manager.st')
    def test_handle_restart_workflow(self, mock_st, ui_manager):
        """Test _handle_restart_workflow method."""
        # Setup mocks
        mock_st.session_state.workflow_session_id = "test_session_id"
        mock_st.success = Mock()
        mock_st.rerun = Mock()
        
        mock_state_manager = Mock()
        ui_manager.state = mock_state_manager
        
        ui_manager._handle_restart_workflow()
        
        # Verify session state was cleared and state was reset
        assert mock_st.session_state.workflow_session_id is None
        mock_state_manager.reset_processing_state.assert_called_once()
        mock_st.success.assert_called_once()
        mock_st.rerun.assert_called_once()

    @patch('src.ui.ui_manager.get_container')
    @patch('src.ui.ui_manager.st.session_state')
    @patch('src.ui.ui_manager.logger')
    def test_render_status_driven_ui_exception_handling(self, mock_logger, mock_session_state, mock_get_container, ui_manager):
        """Test exception handling in render_status_driven_ui."""
        # Setup mocks to raise exception
        mock_session_state.get.return_value = "test_session_id"
        mock_get_container.side_effect = Exception("Test exception")
        
        # Should not raise exception, just log error
        ui_manager.render_status_driven_ui()
        
        # Verify error was logged
        mock_logger.error.assert_called_once()