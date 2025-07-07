"""Unit tests for UI feedback callbacks with persistent async event loop.

These tests verify that the feedback callbacks correctly implement the requirements:
- Approve button calls workflow_manager.send_feedback(session_id, "approve")
- Regenerate button calls workflow_manager.send_feedback(session_id, "regenerate")
- After send_feedback, uses asyncio.run_coroutine_threadsafe to submit tasks to persistent event loop
"""

from unittest.mock import Mock, patch

import pytest
import streamlit as st

from src.frontend.callbacks import handle_user_action


class TestFeedbackCallbacks:
    """Test class for feedback callback functionality."""

    @patch('src.frontend.callbacks._get_or_create_workflow_controller')
    @patch('src.frontend.callbacks.st')
    @patch('src.frontend.callbacks.logger')
    def test_handle_user_action_approve(
        self, mock_logger, mock_st, mock_get_controller
    ):
        """Test that approve action uses WorkflowController."""
        # Setup
        session_id = "test-session-123"
        item_id = "test-item-456"
        
        mock_st.session_state.get.side_effect = lambda key, default=None: {
            "agent_state": Mock(),
            "workflow_session_id": session_id
        }.get(key, default)
        
        # Mock WorkflowController
        mock_controller = Mock()
        mock_controller.submit_user_feedback.return_value = True
        mock_get_controller.return_value = mock_controller
        
        # Execute
        handle_user_action("accept", item_id)
        
        # Verify submit_user_feedback was called with correct parameters
        mock_controller.submit_user_feedback.assert_called_once_with(
            action="accept",
            item_id=item_id,
            workflow_session_id=session_id
        )
        
        # Verify success message was shown
        mock_st.success.assert_called_once_with("Action 'accept' processed successfully")

    @patch('src.frontend.callbacks._get_or_create_workflow_controller')
    @patch('src.frontend.callbacks.st')
    @patch('src.frontend.callbacks.logger')
    def test_handle_user_action_regenerate(
        self, mock_logger, mock_st, mock_get_controller
    ):
        """Test that regenerate action uses WorkflowController."""
        # Setup
        session_id = "test-session-123"
        item_id = "test-item-456"
        
        mock_st.session_state.get.side_effect = lambda key, default=None: {
            "agent_state": Mock(),
            "workflow_session_id": session_id
        }.get(key, default)
        
        # Mock WorkflowController
        mock_controller = Mock()
        mock_controller.submit_user_feedback.return_value = True
        mock_get_controller.return_value = mock_controller
        
        # Execute
        handle_user_action("regenerate", item_id)
        
        # Verify submit_user_feedback was called with correct parameters
        mock_controller.submit_user_feedback.assert_called_once_with(
            action="regenerate",
            item_id=item_id,
            workflow_session_id=session_id
        )
        
        # Verify success message was shown
        mock_st.success.assert_called_once_with("Action 'regenerate' processed successfully")

    @patch('src.frontend.callbacks.st')
    def test_handle_user_action_no_agent_state(self, mock_st):
        """Test handle_user_action when agent state is missing."""
        # Mock session state without agent_state
        mock_st.session_state.get.side_effect = lambda key, default=None: {
        }.get(key, default)
        
        # Call the function
        handle_user_action('approve', 'item_123')
        
        # Verify session state was checked for agent_state
        mock_st.session_state.get.assert_called_with('agent_state')
        
        # Verify error message was shown
        mock_st.error.assert_called_once_with("No agent state found")

    @patch('src.frontend.callbacks._get_or_create_workflow_controller')
    @patch('src.frontend.callbacks.st')
    @patch('src.frontend.callbacks.logger')
    def test_handle_user_action_no_workflow_session(
        self, mock_logger, mock_st, mock_get_controller
    ):
        """Test that function handles missing workflow session gracefully."""
        # Setup - agent_state exists but no workflow session
        mock_st.session_state.get.side_effect = lambda key, default=None: {
            "agent_state": Mock(),
            "workflow_session_id": None
        }.get(key, default)
        
        # Execute
        handle_user_action("accept", "test-item")
        
        # Verify error message and early return
        mock_st.error.assert_called_with("No workflow session found")
        mock_get_controller.assert_not_called()

    @patch('src.frontend.callbacks.logger')
    @patch('src.frontend.callbacks._get_or_create_workflow_controller')
    @patch('src.frontend.callbacks.st')
    def test_handle_user_action_controller_error(
        self, mock_st, mock_get_controller, mock_logger
    ):
        """Test handle_user_action when WorkflowController raises an exception."""
        # Setup mocks
        mock_controller = Mock()
        mock_get_controller.return_value = mock_controller
        test_exception = Exception("Controller error")
        mock_controller.submit_user_feedback.side_effect = test_exception
        
        # Mock session state
        mock_st.session_state.get.side_effect = lambda key, default=None: {
            'agent_state': Mock(),
            'workflow_session_id': 'test_session_789'
        }.get(key, default)
        
        # Call the function
        handle_user_action('approve', 'item_789')
        
        # Verify error was logged (should be the unexpected error case)
        mock_logger.error.assert_called_with(
            "Unexpected error handling user action: Controller error",
            exc_info=True
        )
        
        # Verify error message was displayed
        mock_st.error.assert_called_with("An unexpected error occurred while handling approve action")