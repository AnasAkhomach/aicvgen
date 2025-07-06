"""Tests for WorkflowManager integration in callbacks module."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import streamlit as st
from src.frontend.callbacks import start_cv_generation
from src.orchestration.state import AgentState


class MockSessionState(dict):
    """Mock session state that supports both dict and attribute access."""
    def __getattr__(self, key):
        return self.get(key)
    
    def __setattr__(self, key, value):
        self[key] = value


class TestWorkflowManagerIntegration:
    """Test WorkflowManager integration in callbacks."""

    @patch('src.frontend.callbacks.get_container')
    @patch('src.frontend.callbacks._start_workflow_manager_thread')
    @patch('src.frontend.callbacks.create_initial_agent_state')
    def test_start_cv_generation_uses_workflow_manager(
        self, 
        mock_create_state, 
        mock_start_thread, 
        mock_get_container
    ):
        """Test that start_cv_generation uses WorkflowManager correctly."""
        # Setup mocks
        mock_workflow_manager = Mock()
        mock_workflow_manager.create_new_workflow.return_value = "test-session-123"
        
        mock_container = Mock()
        mock_container.workflow_manager.return_value = mock_workflow_manager
        mock_get_container.return_value = mock_container
        
        mock_agent_state = Mock(spec=AgentState)
        mock_create_state.return_value = mock_agent_state
        
        # Setup Streamlit session state
        mock_session_state = MockSessionState({
            'job_description_input': "Test job description",
            'cv_text_input': "Test CV content",
            'start_from_scratch_input': False
        })
        
        with patch.object(st, 'session_state', mock_session_state):
            # Call the function
            start_cv_generation()
            
            # Verify WorkflowManager.create_new_workflow was called
            mock_workflow_manager.create_new_workflow.assert_called_once_with(
                cv_text="Test CV content",
                jd_text="Test job description"
            )
            
            # Verify session_id was stored
            assert mock_session_state.get('workflow_session_id') == "test-session-123"
            
            # Verify agent state was created and stored
            mock_create_state.assert_called_once_with(
                job_description_raw="Test job description",
                cv_text="Test CV content",
                start_from_scratch=False
            )
            assert mock_session_state.get('agent_state') == mock_agent_state
            
            # Verify workflow manager thread was started
            mock_start_thread.assert_called_once_with(
                mock_agent_state, 
                "test-session-123", 
                mock_workflow_manager
            )

    @patch('src.frontend.callbacks.get_container')
    def test_workflow_manager_container_access(self, mock_get_container):
        """Test that WorkflowManager is correctly accessed from container."""
        mock_workflow_manager = Mock()
        mock_container = Mock()
        mock_container.workflow_manager.return_value = mock_workflow_manager
        mock_get_container.return_value = mock_container
        
        mock_session_state = MockSessionState({
            'job_description_input': "Test job",
            'cv_text_input': "Test CV",
            'start_from_scratch_input': False
        })
        
        with patch.object(st, 'session_state', mock_session_state):
            
            with patch('src.frontend.callbacks._start_workflow_manager_thread'):
                with patch('src.frontend.callbacks.create_initial_agent_state'):
                    start_cv_generation()
            
            # Verify container was accessed
            mock_get_container.assert_called_once()
            mock_container.workflow_manager.assert_called_once()

    def test_session_state_workflow_session_id_storage(self):
        """Test that workflow_session_id is properly stored in session state."""
        with patch('src.frontend.callbacks.get_container') as mock_get_container:
            mock_workflow_manager = Mock()
            mock_workflow_manager.create_new_workflow.return_value = "unique-session-456"
            
            mock_container = Mock()
            mock_container.workflow_manager.return_value = mock_workflow_manager
            mock_get_container.return_value = mock_container
            
            mock_session_state = MockSessionState({
                'job_description_input': "Job desc",
                'cv_text_input': "CV text",
                'start_from_scratch_input': True
            })
            
            with patch.object(st, 'session_state', mock_session_state):
                
                with patch('src.frontend.callbacks._start_workflow_manager_thread'):
                    with patch('src.frontend.callbacks.create_initial_agent_state'):
                        start_cv_generation()
                
                # Verify the session ID is stored correctly
                assert 'workflow_session_id' in mock_session_state
                assert mock_session_state.get('workflow_session_id') == "unique-session-456"