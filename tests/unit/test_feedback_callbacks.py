"""Unit tests for UI feedback callbacks with async execution.

These tests verify that the feedback callbacks correctly implement the requirements:
- Approve button calls workflow_manager.send_feedback(session_id, "approve")
- Regenerate button calls workflow_manager.send_feedback(session_id, "regenerate")
- After send_feedback, calls asyncio.run(workflow_manager.trigger_workflow_step(session_id))
"""

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest
import streamlit as st

from src.frontend.callbacks import handle_user_action
from src.models.workflow_models import UserAction, UserFeedback
from src.orchestration.state import AgentState


class TestFeedbackCallbacks:
    """Test class for feedback callback functionality."""

    @pytest.fixture
    def mock_agent_state(self):
        """Create a mock AgentState."""
        return Mock(spec=AgentState)

    @pytest.fixture
    def mock_workflow_manager(self):
        """Create a mock WorkflowManager."""
        mock_manager = Mock()
        mock_manager.send_feedback.return_value = True
        mock_manager.get_workflow_status.return_value = Mock(spec=AgentState)
        mock_manager.trigger_workflow_step = AsyncMock(return_value=Mock(spec=AgentState))
        return mock_manager

    @pytest.fixture
    def mock_container(self, mock_workflow_manager):
        """Create a mock container."""
        mock_container = Mock()
        mock_container.workflow_manager.return_value = mock_workflow_manager
        return mock_container

    @patch('src.frontend.callbacks.st')
    @patch('src.frontend.callbacks.get_container')
    @patch('src.frontend.callbacks.logger')
    def test_handle_user_action_approve(
        self, mock_logger, mock_get_container, mock_st, mock_container, mock_workflow_manager, mock_agent_state
    ):
        """Test that approve action calls send_feedback with 'approve' and triggers workflow step."""
        # Setup
        session_id = "test-session-123"
        item_id = "test-item-456"
        
        mock_st.session_state.get.side_effect = lambda key, default=None: {
            "agent_state": mock_agent_state,
            "workflow_session_id": session_id
        }.get(key, default)
        
        mock_get_container.return_value = mock_container
        
        # Execute
        handle_user_action("accept", item_id)
        
        # Verify send_feedback was called with correct parameters
        mock_workflow_manager.send_feedback.assert_called_once()
        call_args = mock_workflow_manager.send_feedback.call_args
        assert call_args[0][0] == session_id  # session_id
        
        feedback = call_args[0][1]  # UserFeedback object
        assert isinstance(feedback, UserFeedback)
        assert feedback.action == UserAction.APPROVE
        assert feedback.item_id == item_id
        assert "approved" in feedback.feedback_text.lower()
        
        # Verify get_workflow_status was called
        mock_workflow_manager.get_workflow_status.assert_called_once_with(session_id)
        
        # Verify trigger_workflow_step was called
        mock_workflow_manager.trigger_workflow_step.assert_called_once()
        trigger_call_args = mock_workflow_manager.trigger_workflow_step.call_args
        assert trigger_call_args[0][0] == session_id
        
        # Verify success message
        mock_st.success.assert_called()
        success_calls = [call for call in mock_st.success.call_args_list]
        assert any("approved" in str(call).lower() for call in success_calls)

    @patch('src.frontend.callbacks.st')
    @patch('src.frontend.callbacks.get_container')
    @patch('src.frontend.callbacks.logger')
    def test_handle_user_action_regenerate(
        self, mock_logger, mock_get_container, mock_st, mock_container, mock_workflow_manager, mock_agent_state
    ):
        """Test that regenerate action calls send_feedback with 'regenerate' and triggers workflow step."""
        # Setup
        session_id = "test-session-123"
        item_id = "test-item-456"
        
        mock_st.session_state.get.side_effect = lambda key, default=None: {
            "agent_state": mock_agent_state,
            "workflow_session_id": session_id
        }.get(key, default)
        
        mock_get_container.return_value = mock_container
        
        # Execute
        handle_user_action("regenerate", item_id)
        
        # Verify send_feedback was called with correct parameters
        mock_workflow_manager.send_feedback.assert_called_once()
        call_args = mock_workflow_manager.send_feedback.call_args
        assert call_args[0][0] == session_id  # session_id
        
        feedback = call_args[0][1]  # UserFeedback object
        assert isinstance(feedback, UserFeedback)
        assert feedback.action == UserAction.REGENERATE
        assert feedback.item_id == item_id
        assert "regenerate" in feedback.feedback_text.lower()
        
        # Verify get_workflow_status was called
        mock_workflow_manager.get_workflow_status.assert_called_once_with(session_id)
        
        # Verify trigger_workflow_step was called
        mock_workflow_manager.trigger_workflow_step.assert_called_once()
        trigger_call_args = mock_workflow_manager.trigger_workflow_step.call_args
        assert trigger_call_args[0][0] == session_id
        
        # Verify info message
        mock_st.info.assert_called()
        info_calls = [call for call in mock_st.info.call_args_list]
        assert any("regenerat" in str(call).lower() for call in info_calls)

    @patch('src.frontend.callbacks.st')
    @patch('src.frontend.callbacks.get_container')
    @patch('src.frontend.callbacks.logger')
    def test_handle_user_action_no_agent_state(
        self, mock_logger, mock_get_container, mock_st, mock_container, mock_workflow_manager
    ):
        """Test that function handles missing agent state gracefully."""
        # Setup - no agent state
        mock_st.session_state.get.side_effect = lambda key, default=None: {
            "agent_state": None,
            "workflow_session_id": "test-session"
        }.get(key, default)
        
        # Execute
        handle_user_action("accept", "test-item")
        
        # Verify error message and early return
        mock_st.error.assert_called_with("No agent state found")
        mock_workflow_manager.send_feedback.assert_not_called()
        mock_workflow_manager.trigger_workflow_step.assert_not_called()

    @patch('src.frontend.callbacks.st')
    @patch('src.frontend.callbacks.get_container')
    @patch('src.frontend.callbacks.logger')
    def test_handle_user_action_no_workflow_session(
        self, mock_logger, mock_get_container, mock_st, mock_container, mock_workflow_manager, mock_agent_state
    ):
        """Test that function handles missing workflow session gracefully."""
        # Setup - no workflow session
        mock_st.session_state.get.side_effect = lambda key, default=None: {
            "agent_state": mock_agent_state,
            "workflow_session_id": None
        }.get(key, default)
        
        # Execute
        handle_user_action("accept", "test-item")
        
        # Verify error message and early return
        mock_st.error.assert_called_with("No workflow session found")
        mock_workflow_manager.send_feedback.assert_not_called()
        mock_workflow_manager.trigger_workflow_step.assert_not_called()

    @patch('src.frontend.callbacks.st')
    @patch('src.frontend.callbacks.get_container')
    @patch('src.frontend.callbacks.logger')
    def test_handle_user_action_send_feedback_fails(
        self, mock_logger, mock_get_container, mock_st, mock_container, mock_workflow_manager, mock_agent_state
    ):
        """Test that function handles send_feedback failure gracefully."""
        # Setup
        session_id = "test-session-123"
        item_id = "test-item-456"
        
        mock_st.session_state.get.side_effect = lambda key, default=None: {
            "agent_state": mock_agent_state,
            "workflow_session_id": session_id
        }.get(key, default)
        
        mock_get_container.return_value = mock_container
        mock_workflow_manager.send_feedback.return_value = False  # Simulate failure
        
        # Execute
        handle_user_action("accept", item_id)
        
        # Verify error handling
        mock_st.error.assert_called_with("Failed to send feedback to workflow")
        mock_workflow_manager.trigger_workflow_step.assert_not_called()

    @patch('src.frontend.callbacks.st')
    @patch('src.frontend.callbacks.get_container')
    @patch('src.frontend.callbacks.logger')
    def test_handle_user_action_async_execution_fails(
        self, mock_logger, mock_get_container, mock_st, mock_container, mock_workflow_manager, mock_agent_state
    ):
        """Test that function handles async execution failure gracefully."""
        # Setup
        session_id = "test-session-123"
        item_id = "test-item-456"
        
        mock_st.session_state.get.side_effect = lambda key, default=None: {
            "agent_state": mock_agent_state,
            "workflow_session_id": session_id
        }.get(key, default)
        
        mock_get_container.return_value = mock_container
        mock_workflow_manager.trigger_workflow_step.side_effect = Exception("Async execution failed")
        
        # Execute
        handle_user_action("accept", item_id)
        
        # Verify error handling
        mock_st.error.assert_called()
        error_calls = [str(call) for call in mock_st.error.call_args_list]
        assert any("Failed to resume workflow" in call for call in error_calls)
        
        # Verify logger was called
        mock_logger.error.assert_called()

    @patch('src.frontend.callbacks.st')
    @patch('src.frontend.callbacks.get_container')
    @patch('src.frontend.callbacks.logger')
    def test_handle_user_action_unknown_action(
        self, mock_logger, mock_get_container, mock_st, mock_container, mock_workflow_manager, mock_agent_state
    ):
        """Test that function handles unknown actions gracefully."""
        # Setup
        session_id = "test-session-123"
        item_id = "test-item-456"
        
        mock_st.session_state.get.side_effect = lambda key, default=None: {
            "agent_state": mock_agent_state,
            "workflow_session_id": session_id
        }.get(key, default)
        
        # Execute with unknown action
        handle_user_action("unknown_action", item_id)
        
        # Verify error message and early return
        mock_st.error.assert_called_with("Unknown action: unknown_action")
        mock_workflow_manager.send_feedback.assert_not_called()
        mock_workflow_manager.trigger_workflow_step.assert_not_called()