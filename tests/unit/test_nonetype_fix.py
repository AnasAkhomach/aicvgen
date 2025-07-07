"""Test for NoneType error fixes in cv_workflow_graph.py and app.py."""
import pytest
from unittest.mock import Mock, patch
from src.orchestration.cv_workflow_graph import CVWorkflowGraph
from src.orchestration.state import AgentState
from src.utils.state_utils import create_initial_agent_state


class TestNoneTypeFixes:
    """Test cases for NoneType error fixes."""

    @pytest.fixture
    def workflow_graph(self):
        """Create a workflow graph instance for testing."""
        return CVWorkflowGraph(session_id="test-session")

    @pytest.fixture
    def base_agent_state(self):
        """Create a base agent state for testing."""
        return create_initial_agent_state(
            cv_text="Test CV",
            job_description_raw="Test JD"
        )

    def test_save_state_to_file_with_none_state(self, workflow_graph):
        """Test that _save_state_to_file handles None state gracefully."""
        # This should not raise an exception
        workflow_graph._save_state_to_file(None)
        # No assertion needed - we just want to ensure no exception is raised

    def test_save_state_to_file_with_invalid_state(self, workflow_graph):
        """Test that _save_state_to_file handles invalid state types gracefully."""
        # This should not raise an exception
        workflow_graph._save_state_to_file("invalid_state")
        # No assertion needed - we just want to ensure no exception is raised

    def test_save_state_to_file_with_valid_state(self, workflow_graph, base_agent_state):
        """Test that _save_state_to_file works correctly with valid state."""
        with patch('builtins.open'), patch('src.orchestration.cv_workflow_graph.get_config') as mock_config:
            mock_config.return_value.paths.project_root = Mock()
            mock_config.return_value.paths.project_root.__truediv__ = Mock(return_value=Mock())
            
            # This should not raise an exception
            workflow_graph._save_state_to_file(base_agent_state)

    def test_trigger_workflow_step_with_empty_node_result(self, workflow_graph, base_agent_state):
        """Test that trigger_workflow_step handles empty node results correctly."""
        # Mock a node that returns an empty dictionary
        def mock_node(state):
            return {}

        # Mock the workflow steps
        workflow_graph.workflow_steps = [("test_node", mock_node)]
        
        with patch.object(workflow_graph, '_save_state_to_file') as mock_save:
            result = workflow_graph.trigger_workflow_step(base_agent_state)
            
            # Should return a valid AgentState, not None
            assert result is not None
            assert isinstance(result, AgentState)
            
            # Should have called save_state_to_file
            mock_save.assert_called()

    def test_trigger_workflow_step_with_model_copy_returning_none(self, workflow_graph, base_agent_state):
        """Test that trigger_workflow_step handles model_copy returning None."""
        # Mock a node that returns a dictionary
        def mock_node(state):
            return {"test_key": "test_value"}

        # Mock the workflow steps
        workflow_graph.workflow_steps = [("test_node", mock_node)]
        
        # Mock model_copy to return None
        with patch.object(base_agent_state, 'model_copy', return_value=None) as mock_model_copy:
            with patch.object(workflow_graph, '_save_state_to_file') as mock_save:
                result = workflow_graph.trigger_workflow_step(base_agent_state)
                
                # Should return the original state when model_copy returns None
                assert result is not None
                assert result == base_agent_state
                
                # Should have called model_copy
                mock_model_copy.assert_called_once_with(update={"test_key": "test_value"})
                
                # Should have called save_state_to_file with original state
                mock_save.assert_called_with(base_agent_state)

    @patch('src.core.application_startup.get_container')
    def test_on_start_generation_with_none_initial_state(self, mock_get_container):
        """Test that on_start_generation handles None initial state gracefully."""
        from app import on_start_generation
        import streamlit as st
        
        # Mock the workflow manager to return None for get_workflow_status
        mock_manager = Mock()
        mock_manager.create_new_workflow.return_value = "test-session"
        mock_manager.get_workflow_status.return_value = None
        
        mock_container = Mock()
        mock_container.workflow_manager.return_value = mock_manager
        mock_get_container.return_value = mock_container
        
        # Mock streamlit session state
        with patch.object(st, 'session_state', {'cv_text_input': 'test cv', 'job_description_input': 'test jd'}):
            with patch.object(st, 'error') as mock_error:
                # This should not raise an exception
                on_start_generation()
                
                # Should have called st.error with appropriate message
                mock_error.assert_called_with("Failed to initialize workflow. Please try again.")
                
                # Should not have called trigger_workflow_step
                mock_manager.trigger_workflow_step.assert_not_called()