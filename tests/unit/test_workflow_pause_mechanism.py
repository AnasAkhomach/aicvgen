"""Test cases for the workflow pause mechanism fix.

This module tests that the WorkflowManager correctly pauses the workflow
when the graph signals it's ready for user feedback.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from src.core.workflow_manager import WorkflowManager
from src.orchestration.cv_workflow_graph import CVWorkflowGraph
from src.orchestration.state import AgentState
from src.models.cv_models import StructuredCV


class TestWorkflowPauseMechanism:
    """Test cases for workflow pause mechanism."""

    @pytest.fixture
    def mock_cv_workflow_graph(self):
        """Create a mock CVWorkflowGraph."""
        mock_graph = MagicMock(spec=CVWorkflowGraph)
        mock_graph.trigger_workflow_step = AsyncMock()
        return mock_graph

    @pytest.fixture
    def workflow_manager(self, mock_cv_workflow_graph, tmp_path):
        """Create a WorkflowManager instance with mocked dependencies."""
        manager = WorkflowManager(mock_cv_workflow_graph)
        # Override sessions directory to use temp path
        manager.sessions_dir = tmp_path / "sessions"
        manager.sessions_dir.mkdir(parents=True, exist_ok=True)
        return manager

    @pytest.fixture
    def sample_agent_state(self):
        """Create a sample AgentState for testing."""
        return AgentState(
            session_id="test-session-123",
            structured_cv=StructuredCV(),
            cv_text="Sample CV text",
            workflow_status="PROCESSING"
        )

    @pytest.mark.asyncio
    async def test_workflow_pauses_on_awaiting_feedback(
        self, workflow_manager, mock_cv_workflow_graph, sample_agent_state
    ):
        """Test that workflow pauses when status becomes AWAITING_FEEDBACK."""
        # Setup: Create a session file
        session_id = "test-session-123"
        session_file = workflow_manager.sessions_dir / f"{session_id}.json"
        
        with open(session_file, 'w', encoding='utf-8') as f:
            f.write(sample_agent_state.model_dump_json(indent=2))
        
        # Mock the CVWorkflowGraph to return state with AWAITING_FEEDBACK status
        paused_state = sample_agent_state.set_workflow_status("AWAITING_FEEDBACK")
        mock_cv_workflow_graph.trigger_workflow_step.return_value = paused_state
        
        # Execute
        result = await workflow_manager.trigger_workflow_step(
            session_id, sample_agent_state
        )
        
        # Verify
        assert result.workflow_status == "AWAITING_FEEDBACK"
        mock_cv_workflow_graph.trigger_workflow_step.assert_called_once_with(sample_agent_state)
        
        # Verify state was saved to file
        with open(session_file, 'r', encoding='utf-8') as f:
            saved_data = json.load(f)
            assert saved_data["workflow_status"] == "AWAITING_FEEDBACK"

    @pytest.mark.asyncio
    async def test_workflow_continues_on_processing_status(
        self, workflow_manager, mock_cv_workflow_graph, sample_agent_state
    ):
        """Test that workflow continues when status remains PROCESSING."""
        # Setup: Create a session file
        session_id = "test-session-123"
        session_file = workflow_manager.sessions_dir / f"{session_id}.json"
        
        with open(session_file, 'w', encoding='utf-8') as f:
            f.write(sample_agent_state.model_dump_json(indent=2))
        
        # Mock the CVWorkflowGraph to return state with PROCESSING status
        processing_state = sample_agent_state.set_workflow_status("PROCESSING")
        mock_cv_workflow_graph.trigger_workflow_step.return_value = processing_state
        
        # Execute
        result = await workflow_manager.trigger_workflow_step(
            session_id, sample_agent_state
        )
        
        # Verify
        assert result.workflow_status == "PROCESSING"
        mock_cv_workflow_graph.trigger_workflow_step.assert_called_once_with(sample_agent_state)
        
        # Verify state was saved to file
        with open(session_file, 'r', encoding='utf-8') as f:
            saved_data = json.load(f)
            assert saved_data["workflow_status"] == "PROCESSING"

    @pytest.mark.asyncio
    async def test_workflow_handles_completed_status(
        self, workflow_manager, mock_cv_workflow_graph, sample_agent_state
    ):
        """Test that workflow handles COMPLETED status correctly."""
        # Setup: Create a session file
        session_id = "test-session-123"
        session_file = workflow_manager.sessions_dir / f"{session_id}.json"
        
        with open(session_file, 'w', encoding='utf-8') as f:
            f.write(sample_agent_state.model_dump_json(indent=2))
        
        # Mock the CVWorkflowGraph to return state with COMPLETED status
        completed_state = sample_agent_state.set_workflow_status("COMPLETED")
        mock_cv_workflow_graph.trigger_workflow_step.return_value = completed_state
        
        # Execute
        result = await workflow_manager.trigger_workflow_step(
            session_id, sample_agent_state
        )
        
        # Verify
        assert result.workflow_status == "COMPLETED"
        mock_cv_workflow_graph.trigger_workflow_step.assert_called_once_with(sample_agent_state)

    @pytest.mark.asyncio
    async def test_workflow_handles_error_status(
        self, workflow_manager, mock_cv_workflow_graph, sample_agent_state
    ):
        """Test that workflow handles ERROR status correctly."""
        # Setup: Create a session file
        session_id = "test-session-123"
        session_file = workflow_manager.sessions_dir / f"{session_id}.json"
        
        with open(session_file, 'w', encoding='utf-8') as f:
            f.write(sample_agent_state.model_dump_json(indent=2))
        
        # Mock the CVWorkflowGraph to return state with ERROR status
        error_state = sample_agent_state.set_workflow_status("ERROR")
        error_state = error_state.model_copy(
            update={"error_messages": ["Test error message"]}
        )
        mock_cv_workflow_graph.trigger_workflow_step.return_value = error_state
        
        # Execute
        result = await workflow_manager.trigger_workflow_step(
            session_id, sample_agent_state
        )
        
        # Verify
        assert result.workflow_status == "ERROR"
        assert "Test error message" in result.error_messages
        mock_cv_workflow_graph.trigger_workflow_step.assert_called_once_with(sample_agent_state)

    @pytest.mark.asyncio
    async def test_workflow_raises_error_for_nonexistent_session(
        self, workflow_manager, sample_agent_state
    ):
        """Test that workflow raises ValueError for non-existent session."""
        session_id = "nonexistent-session"
        
        with pytest.raises(ValueError, match="No active workflow found"):
            await workflow_manager.trigger_workflow_step(
                session_id, sample_agent_state
            )

    def test_save_state_helper_method(
        self, workflow_manager, sample_agent_state
    ):
        """Test the _save_state helper method."""
        session_id = "test-session-123"
        
        # Execute
        workflow_manager._save_state(session_id, sample_agent_state)
        
        # Verify file was created and contains correct data
        session_file = workflow_manager.sessions_dir / f"{session_id}.json"
        assert session_file.exists()
        
        with open(session_file, 'r', encoding='utf-8') as f:
            saved_data = json.load(f)
            assert saved_data["session_id"] == session_id
            assert saved_data["cv_text"] == "Sample CV text"
            assert saved_data["workflow_status"] == "PROCESSING"