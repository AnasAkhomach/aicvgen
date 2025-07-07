# tests/unit/test_workflow_controller.py
import asyncio
import threading
import unittest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from concurrent.futures import Future

import pytest
import streamlit as st

from src.frontend.workflow_controller import WorkflowController
from src.orchestration.state import AgentState, UserFeedback
from src.models.data_models import UserAction
from src.models.cv_models import StructuredCV


class TestWorkflowController(unittest.TestCase):
    """Test cases for WorkflowController class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_workflow_manager = Mock()
        self.mock_workflow_manager._background_loop = None
        
        # Create a valid AgentState for testing
        from src.models.cv_models import StructuredCV
        structured_cv = StructuredCV.create_empty(cv_text="Sample CV text")
        self.test_agent_state = AgentState(
            structured_cv=structured_cv,
            cv_text="Sample CV text"
        )
        
        # Mock the background thread creation
        self.thread_patcher = patch('threading.Thread')
        self.mock_thread_class = self.thread_patcher.start()
        self.mock_thread = Mock()
        self.mock_thread_class.return_value = self.mock_thread
        
        # Mock asyncio.new_event_loop
        self.loop_patcher = patch('asyncio.new_event_loop')
        self.mock_loop_class = self.loop_patcher.start()
        self.mock_loop = Mock()
        self.mock_loop_class.return_value = self.mock_loop
        
        # Mock asyncio.set_event_loop
        self.set_loop_patcher = patch('asyncio.set_event_loop')
        self.mock_set_loop = self.set_loop_patcher.start()
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.thread_patcher.stop()
        self.loop_patcher.stop()
        self.set_loop_patcher.stop()
    
    def test_init_creates_background_thread(self):
        """Test that WorkflowController initialization creates a background thread."""
        controller = WorkflowController(self.mock_workflow_manager)
        
        # Verify thread was created and started
        self.mock_thread_class.assert_called_once()
        self.mock_thread.start.assert_called_once()
        
        # Verify controller state
        self.assertEqual(controller.workflow_manager, self.mock_workflow_manager)
        self.assertTrue(controller._is_running)
    
    def test_background_thread_setup(self):
        """Test that the background thread sets up the event loop correctly."""
        controller = WorkflowController(self.mock_workflow_manager)
        
        # Get the target function that was passed to Thread
        thread_call_args = self.mock_thread_class.call_args
        target_function = thread_call_args[1]['target']
        
        # Execute the target function
        target_function()
        
        # Verify event loop setup
        self.mock_loop_class.assert_called_once()
        self.mock_set_loop.assert_called_once_with(self.mock_loop)
        self.assertEqual(controller._background_loop, self.mock_loop)
        self.assertEqual(self.mock_workflow_manager._background_loop, self.mock_loop)
        self.mock_loop.run_forever.assert_called_once()
    
    @patch('streamlit.session_state')
    @patch('uuid.uuid4')
    def test_start_generation(self, mock_uuid, mock_session_state):
        """Test start_generation method."""
        # Setup
        mock_uuid.return_value = Mock()
        mock_uuid.return_value.__str__ = Mock(return_value='test-trace-id')
        
        mock_session_state.is_processing = False
        mock_session_state.workflow_error = None
        mock_session_state.just_finished = False
        
        controller = WorkflowController(self.mock_workflow_manager)
        controller._background_loop = self.mock_loop
        controller._is_running = True
        
        # Create test data
        initial_state = self.test_agent_state
        workflow_session_id = 'test-session-id'
        
        # Mock asyncio.run_coroutine_threadsafe
        with patch('asyncio.run_coroutine_threadsafe') as mock_run_coroutine:
            mock_future = Mock(spec=Future)
            mock_run_coroutine.return_value = mock_future
            
            # Execute
            controller.start_generation(initial_state, workflow_session_id)
            
            # Verify
            self.assertTrue(mock_session_state.is_processing)
            self.assertIsNone(mock_session_state.workflow_error)
            self.assertFalse(mock_session_state.just_finished)
            self.assertEqual(initial_state.trace_id, 'test-trace-id')
            
            # Verify coroutine was submitted to background loop
            mock_run_coroutine.assert_called_once()
            call_args = mock_run_coroutine.call_args
            self.assertEqual(call_args[0][1], self.mock_loop)  # Second arg should be the loop
    
    @patch('streamlit.session_state')
    def test_start_generation_not_ready(self, mock_session_state):
        """Test start_generation when controller is not ready."""
        controller = WorkflowController(self.mock_workflow_manager)
        controller._is_running = False
        controller._background_loop = None
        
        initial_state = self.test_agent_state
        workflow_session_id = 'test-session-id'
        
        with patch('streamlit.error') as mock_error:
            controller.start_generation(initial_state, workflow_session_id)
            mock_error.assert_called_once_with("Workflow controller not ready")
    
    def test_submit_user_feedback_accept(self):
        """Test submit_user_feedback with accept action."""
        controller = WorkflowController(self.mock_workflow_manager)
        controller._background_loop = self.mock_loop
        controller._is_running = True
        
        # Mock workflow manager methods
        self.mock_workflow_manager.send_feedback.return_value = True
        self.mock_workflow_manager.get_workflow_status.return_value = self.test_agent_state
        
        # Mock asyncio.run_coroutine_threadsafe
        with patch('asyncio.run_coroutine_threadsafe') as mock_run_coroutine:
            mock_future = Mock(spec=Future)
            mock_result_state = self.test_agent_state
            mock_future.result.return_value = mock_result_state
            mock_run_coroutine.return_value = mock_future
            
            with patch('streamlit.session_state') as mock_session_state:
                with patch('streamlit.success') as mock_success:
                    # Execute
                    result = controller.submit_user_feedback(
                        action='accept',
                        item_id='test-item',
                        workflow_session_id='test-session'
                    )
                    
                    # Verify
                    self.assertTrue(result)
                    self.mock_workflow_manager.send_feedback.assert_called_once()
                    self.mock_workflow_manager.get_workflow_status.assert_called_once_with('test-session')
                    mock_run_coroutine.assert_called_once()
                    mock_future.result.assert_called_once_with(timeout=30)
                    mock_session_state.agent_state = mock_result_state
                    mock_success.assert_called()
    
    def test_submit_user_feedback_regenerate(self):
        """Test submit_user_feedback with regenerate action."""
        controller = WorkflowController(self.mock_workflow_manager)
        controller._background_loop = self.mock_loop
        controller._is_running = True
        
        # Mock workflow manager methods
        self.mock_workflow_manager.send_feedback.return_value = True
        self.mock_workflow_manager.get_workflow_status.return_value = self.test_agent_state
        
        # Mock asyncio.run_coroutine_threadsafe
        with patch('asyncio.run_coroutine_threadsafe') as mock_run_coroutine:
            mock_future = Mock(spec=Future)
            mock_result_state = self.test_agent_state
            mock_future.result.return_value = mock_result_state
            mock_run_coroutine.return_value = mock_future
            
            with patch('streamlit.session_state') as mock_session_state:
                with patch('streamlit.info') as mock_info:
                    # Execute
                    result = controller.submit_user_feedback(
                        action='regenerate',
                        item_id='test-item',
                        workflow_session_id='test-session'
                    )
                    
                    # Verify
                    self.assertTrue(result)
                    mock_info.assert_called_with("🔄 Regenerating item...")
    
    def test_submit_user_feedback_unknown_action(self):
        """Test submit_user_feedback with unknown action."""
        controller = WorkflowController(self.mock_workflow_manager)
        controller._background_loop = self.mock_loop
        controller._is_running = True
        
        with patch('streamlit.error') as mock_error:
            result = controller.submit_user_feedback(
                action='unknown',
                item_id='test-item',
                workflow_session_id='test-session'
            )
            
            self.assertFalse(result)
            mock_error.assert_called_with("Unknown action: unknown")
    
    def test_submit_user_feedback_not_ready(self):
        """Test submit_user_feedback when controller is not ready."""
        controller = WorkflowController(self.mock_workflow_manager)
        controller._is_running = False
        controller._background_loop = None
        
        with patch('streamlit.error') as mock_error:
            result = controller.submit_user_feedback(
                action='accept',
                item_id='test-item',
                workflow_session_id='test-session'
            )
            
            self.assertFalse(result)
            mock_error.assert_called_with("Workflow controller not ready")
    
    def test_is_ready(self):
        """Test is_ready method."""
        controller = WorkflowController(self.mock_workflow_manager)
        
        # Test not ready state
        controller._is_running = False
        controller._background_loop = None
        self.assertFalse(controller.is_ready())
        
        # Test ready state
        controller._is_running = True
        controller._background_loop = self.mock_loop
        self.assertTrue(controller.is_ready())
    
    @patch('asyncio.all_tasks')
    def test_shutdown_graceful(self, mock_all_tasks):
        """Test graceful shutdown method with task cancellation."""
        controller = WorkflowController(self.mock_workflow_manager)
        controller._background_loop = self.mock_loop
        controller._background_thread = self.mock_thread
        controller._is_running = True
        
        # Mock tasks
        mock_task1 = Mock()
        mock_task1.done.return_value = False
        mock_task2 = Mock()
        mock_task2.done.return_value = True  # Already done, should not be cancelled
        mock_all_tasks.return_value = [mock_task1, mock_task2]
        
        # Mock thread.is_alive()
        self.mock_thread.is_alive.return_value = True
        
        # Execute
        controller.shutdown()
        
        # Verify call_soon_threadsafe was called
        self.mock_loop.call_soon_threadsafe.assert_called_once()
        
        # Verify thread cleanup
        self.mock_thread.join.assert_called_once_with(timeout=10)
        self.assertFalse(controller._is_running)
        self.assertIsNone(controller._background_loop)
        self.assertIsNone(controller._background_thread)
    
    def test_shutdown_already_shutdown(self):
        """Test shutdown when controller is already shutdown."""
        controller = WorkflowController(self.mock_workflow_manager)
        controller._is_running = False
        controller._background_loop = None
        
        # Execute
        controller.shutdown()
        
        # Verify no operations were performed
        self.mock_loop.call_soon_threadsafe.assert_not_called()
    
    @patch('asyncio.all_tasks')
    def test_shutdown_task_cancellation_error(self, mock_all_tasks):
        """Test shutdown handles errors during task cancellation gracefully."""
        controller = WorkflowController(self.mock_workflow_manager)
        controller._background_loop = self.mock_loop
        controller._background_thread = self.mock_thread
        controller._is_running = True
        
        # Mock tasks that will raise an error
        mock_all_tasks.side_effect = Exception("Task enumeration failed")
        
        # Mock thread.is_alive()
        self.mock_thread.is_alive.return_value = True
        
        # Execute
        controller.shutdown()
        
        # Verify call_soon_threadsafe was called
        self.mock_loop.call_soon_threadsafe.assert_called_once()
        
        # Verify cleanup still happened
        self.assertFalse(controller._is_running)
        self.assertIsNone(controller._background_loop)
        self.assertIsNone(controller._background_thread)
    
    def test_shutdown_thread_timeout(self):
        """Test shutdown when background thread doesn't exit within timeout."""
        controller = WorkflowController(self.mock_workflow_manager)
        controller._background_loop = self.mock_loop
        controller._background_thread = self.mock_thread
        controller._is_running = True
        
        # Mock thread that doesn't exit within timeout
        self.mock_thread.is_alive.return_value = True
        
        with patch('asyncio.all_tasks', return_value=[]):
            # Execute
            controller.shutdown()
            
            # Verify thread.join was called with timeout
            self.mock_thread.join.assert_called_once_with(timeout=10)
            
            # Verify cleanup still happened despite timeout
            self.assertFalse(controller._is_running)
            self.assertIsNone(controller._background_loop)
            self.assertIsNone(controller._background_thread)
    
    def test_shutdown_exception_handling(self):
        """Test shutdown handles general exceptions gracefully."""
        controller = WorkflowController(self.mock_workflow_manager)
        controller._background_loop = self.mock_loop
        controller._background_thread = self.mock_thread
        controller._is_running = True
        
        # Mock call_soon_threadsafe to raise an exception
        self.mock_loop.call_soon_threadsafe.side_effect = Exception("Thread communication failed")
        
        # Execute
        controller.shutdown()
        
        # Verify cleanup still happened despite the exception
        self.assertFalse(controller._is_running)
        self.assertIsNone(controller._background_loop)
        self.assertIsNone(controller._background_thread)
    
    def test_del_method_triggers_shutdown(self):
        """Test that __del__ method triggers shutdown when controller is running."""
        controller = WorkflowController(self.mock_workflow_manager)
        controller._background_loop = self.mock_loop
        controller._background_thread = self.mock_thread
        controller._is_running = True
        
        # Mock the shutdown method
        with patch.object(controller, 'shutdown') as mock_shutdown:
            # Trigger __del__
            controller.__del__()
            
            # Verify shutdown was called
            mock_shutdown.assert_called_once()
    
    def test_del_method_not_running(self):
        """Test that __del__ method doesn't call shutdown when controller is not running."""
        controller = WorkflowController(self.mock_workflow_manager)
        controller._is_running = False
        
        # Mock the shutdown method
        with patch.object(controller, 'shutdown') as mock_shutdown:
            # Trigger __del__
            controller.__del__()
            
            # Verify shutdown was not called
            mock_shutdown.assert_not_called()
    
    def test_del_method_exception_handling(self):
        """Test that __del__ method handles exceptions gracefully."""
        controller = WorkflowController(self.mock_workflow_manager)
        controller._is_running = True
        
        # Mock shutdown to raise an exception
        with patch.object(controller, 'shutdown', side_effect=Exception("Shutdown failed")):
            with patch('builtins.print') as mock_print:
                # Trigger __del__ - should not raise exception
                controller.__del__()
                
                # Verify error was printed (since logger might not be available)
                mock_print.assert_called_once()
                self.assertIn("Error in WorkflowController.__del__", mock_print.call_args[0][0])


if __name__ == '__main__':
    unittest.main()