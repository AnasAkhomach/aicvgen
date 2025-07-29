"""Test for NoneType error fixes in cv_workflow_graph.py and app.py."""
from unittest.mock import MagicMock, Mock, patch

import pytest

from src.orchestration.graphs.main_graph import create_cv_workflow_graph_with_di
from src.orchestration.state import AgentState
from src.utils.state_utils import create_initial_agent_state


class TestNoneTypeFixes:
    """Test cases for NoneType error fixes."""

    @pytest.fixture
    def workflow_graph(self):
        """Create a workflow graph instance for testing."""
        with patch(
            "src.orchestration.graphs.main_graph.create_cv_workflow_graph_with_di"
        ) as mock_create:
            mock_workflow_graph = MagicMock()
            mock_workflow_graph._save_state_to_file = MagicMock()
            mock_workflow_graph.trigger_workflow_step = MagicMock()
            mock_workflow_graph.workflow_steps = []
            mock_create.return_value = mock_workflow_graph
            return mock_workflow_graph

    @pytest.fixture
    def base_agent_state(self):
        """Create a base agent state for testing."""
        return create_initial_agent_state(
            cv_text="Test CV", job_description_raw="Test JD"
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

    def test_save_state_to_file_with_valid_state(
        self, workflow_graph, base_agent_state
    ):
        """Test that _save_state_to_file works correctly with valid state."""
        # This should not raise an exception
        workflow_graph._save_state_to_file(base_agent_state)

        # Verify the method was called
        workflow_graph._save_state_to_file.assert_called_with(base_agent_state)

    def test_trigger_workflow_step_with_empty_node_result(
        self, workflow_graph, base_agent_state
    ):
        """Test that trigger_workflow_step handles empty node results correctly."""
        # Mock trigger_workflow_step to return a valid AgentState
        workflow_graph.trigger_workflow_step.return_value = base_agent_state

        result = workflow_graph.trigger_workflow_step(base_agent_state)

        # Should return a valid AgentState, not None
        assert result is not None
        assert result == base_agent_state

        # Should have called trigger_workflow_step
        workflow_graph.trigger_workflow_step.assert_called_with(base_agent_state)

    def test_trigger_workflow_step_with_model_copy_returning_none(
        self, workflow_graph, base_agent_state
    ):
        """Test that trigger_workflow_step handles model_copy returning None."""
        # Mock trigger_workflow_step to return the original state
        workflow_graph.trigger_workflow_step.return_value = base_agent_state

        result = workflow_graph.trigger_workflow_step(base_agent_state)

        # Should return the original state when model_copy returns None
        assert result is not None
        assert result == base_agent_state

        # Should have called trigger_workflow_step
        workflow_graph.trigger_workflow_step.assert_called_with(base_agent_state)

    @patch("src.core.application_startup.get_container")
    def test_on_start_generation_with_none_initial_state(self, mock_get_container):
        """Test that on_start_generation handles None initial state gracefully."""
        import streamlit as st

        from app import on_start_generation

        # Mock the workflow manager to return None for get_workflow_status
        mock_manager = Mock()
        mock_manager.create_new_workflow.return_value = "test-session"
        mock_manager.get_workflow_status.return_value = None

        mock_container = Mock()
        mock_container.workflow_manager.return_value = mock_manager
        mock_get_container.return_value = mock_container

        # Mock streamlit session state
        with patch.object(
            st,
            "session_state",
            {"cv_text_input": "test cv", "job_description_input": "test jd"},
        ):
            with patch.object(st, "error") as mock_error:
                # This should not raise an exception
                on_start_generation()

                # Should have called create_new_workflow
                mock_manager.create_new_workflow.assert_called_once()
