"""Test cases for the workflow pause mechanism fix.

This module tests that the WorkflowManager correctly pauses the workflow
when the graph signals it's ready for user feedback.
"""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.agent_base import AgentBase
from src.core.workflow_manager import WorkflowManager
from src.models.cv_models import StructuredCV
from src.orchestration.graphs.main_graph import create_cv_workflow_graph_with_di
from src.orchestration.state import GlobalState


class TestWorkflowPauseMechanism:
    """Test cases for workflow pause mechanism."""

    @pytest.fixture
    def mock_container(self):
        """Create mock dependency injection container."""
        container = MagicMock()

        # Create a mock agent that can be reused
        mock_agent = MagicMock(spec=AgentBase)
        mock_agent.run_as_node = AsyncMock()

        # Mock all agent methods that the container is expected to have
        container.job_description_parser_agent.return_value = mock_agent
        container.research_agent.return_value = mock_agent
        container.cv_analyzer_agent.return_value = mock_agent
        container.key_qualifications_writer_agent.return_value = mock_agent
        container.professional_experience_writer_agent.return_value = mock_agent
        container.projects_writer_agent.return_value = mock_agent
        container.executive_summary_writer_agent.return_value = mock_agent
        container.qa_agent.return_value = mock_agent
        container.formatter_agent.return_value = mock_agent

        # Mock cv_template_loader_service
        mock_cv_template_service = MagicMock()
        container.cv_template_loader_service.return_value = mock_cv_template_service

        return container

    @pytest.fixture
    def mock_workflow_graph(self, mock_container):
        """Create a mock workflow graph."""
        session_id = "test-session"

        with patch(
            "src.orchestration.graphs.main_graph.build_main_workflow_graph"
        ) as mock_build:
            mock_graph = MagicMock()
            mock_build.return_value = mock_graph
            mock_compiled_graph = MagicMock()
            mock_graph.compile.return_value = mock_compiled_graph

            # Mock the ainvoke method that WorkflowManager uses
            mock_compiled_graph.ainvoke = AsyncMock()

            return create_cv_workflow_graph_with_di(mock_container)

    @pytest.fixture
    def workflow_manager(self, mock_container, tmp_path):
        """Create a WorkflowManager instance with mocked dependencies."""
        manager = WorkflowManager(mock_container)
        # Override sessions directory to use temp path
        manager.sessions_dir = tmp_path / "sessions"
        manager.sessions_dir.mkdir(parents=True, exist_ok=True)
        return manager

    @pytest.fixture
    def sample_global_state(self):
        """Create a sample GlobalState for testing."""
        return GlobalState(
            session_id="test-session-123",
            trace_id="test-trace",
            structured_cv=StructuredCV(),
            cv_text="Sample CV text",
            job_description_data=None,
            current_section_key=None,
            current_section_index=None,
            items_to_process_queue=[],
            current_item_id=None,
            current_content_type=None,
            is_initial_generation=True,
            content_generation_queue=[],
            user_feedback=None,
            research_findings=None,
            cv_analysis_results=None,
            quality_check_results=None,
            generated_key_qualifications=None,
            final_output_path=None,
            error_messages=[],
            node_execution_metadata={},
            workflow_status="PROCESSING",
            ui_display_data={},
            automated_mode=False,
        )

    @pytest.mark.asyncio
    async def test_workflow_pauses_on_awaiting_feedback(
        self, workflow_manager, mock_workflow_graph, sample_global_state
    ):
        """Test that workflow pauses when status becomes AWAITING_FEEDBACK."""
        # Setup: Create a session file
        session_id = "test-session-123"
        session_file = workflow_manager.sessions_dir / f"{session_id}.json"

        # Convert GlobalState to JSON manually since it's a TypedDict
        state_dict = dict(sample_global_state)
        with open(session_file, "w", encoding="utf-8") as f:
            json.dump(state_dict, f, indent=2, default=str)

        # Mock the workflow graph to return state with AWAITING_FEEDBACK status
        paused_state = sample_global_state.copy()
        paused_state["workflow_status"] = "AWAITING_FEEDBACK"

        with patch.object(
            workflow_manager, "_get_workflow_graph", return_value=mock_workflow_graph
        ):
            mock_workflow_graph.ainvoke.return_value = paused_state

            # Execute
            result = await workflow_manager.trigger_workflow_step(
                session_id, sample_global_state
            )

            # Verify
            assert result["workflow_status"] == "AWAITING_FEEDBACK"
            mock_workflow_graph.ainvoke.assert_called_once_with(sample_global_state)

            # Verify state was saved to file
            with open(session_file, "r", encoding="utf-8") as f:
                saved_data = json.load(f)
                assert saved_data["workflow_status"] == "AWAITING_FEEDBACK"

    @pytest.mark.asyncio
    async def test_workflow_continues_on_processing_status(
        self, workflow_manager, mock_workflow_graph, sample_global_state
    ):
        """Test that workflow continues when status remains PROCESSING."""
        # Setup: Create a session file
        session_id = "test-session-123"
        session_file = workflow_manager.sessions_dir / f"{session_id}.json"

        # Convert GlobalState to JSON manually since it's a TypedDict
        state_dict = dict(sample_global_state)
        with open(session_file, "w", encoding="utf-8") as f:
            json.dump(state_dict, f, indent=2, default=str)

        # Mock the workflow graph to return state with PROCESSING status
        processing_state = sample_global_state.copy()
        processing_state["workflow_status"] = "PROCESSING"

        with patch.object(
            workflow_manager, "_get_workflow_graph", return_value=mock_workflow_graph
        ):
            mock_workflow_graph.ainvoke.return_value = processing_state

            # Execute
            result = await workflow_manager.trigger_workflow_step(
                session_id, sample_global_state
            )

            # Verify
            assert result["workflow_status"] == "PROCESSING"
            mock_workflow_graph.ainvoke.assert_called_once_with(sample_global_state)

            # Verify state was saved to file
            with open(session_file, "r", encoding="utf-8") as f:
                saved_data = json.load(f)
                assert saved_data["workflow_status"] == "PROCESSING"

    @pytest.mark.asyncio
    async def test_workflow_handles_completed_status(
        self, workflow_manager, mock_workflow_graph, sample_global_state
    ):
        """Test that workflow handles COMPLETED status correctly."""
        # Setup: Create a session file
        session_id = "test-session-123"
        session_file = workflow_manager.sessions_dir / f"{session_id}.json"

        # Convert GlobalState to JSON manually since it's a TypedDict
        state_dict = dict(sample_global_state)
        with open(session_file, "w", encoding="utf-8") as f:
            json.dump(state_dict, f, indent=2, default=str)

        # Mock the workflow graph to return state with COMPLETED status
        completed_state = sample_global_state.copy()
        completed_state["workflow_status"] = "COMPLETED"

        with patch.object(
            workflow_manager, "_get_workflow_graph", return_value=mock_workflow_graph
        ):
            mock_workflow_graph.ainvoke.return_value = completed_state

            # Execute
            result = await workflow_manager.trigger_workflow_step(
                session_id, sample_global_state
            )

            # Verify
            assert result["workflow_status"] == "COMPLETED"
            mock_workflow_graph.ainvoke.assert_called_once_with(sample_global_state)

    @pytest.mark.asyncio
    async def test_workflow_handles_error_status(
        self, workflow_manager, mock_workflow_graph, sample_global_state
    ):
        """Test that workflow handles ERROR status correctly."""
        # Setup: Create a session file
        session_id = "test-session-123"
        session_file = workflow_manager.sessions_dir / f"{session_id}.json"

        # Convert GlobalState to JSON manually since it's a TypedDict
        state_dict = dict(sample_global_state)
        with open(session_file, "w", encoding="utf-8") as f:
            json.dump(state_dict, f, indent=2, default=str)

        # Mock the workflow graph to return state with ERROR status
        error_state = sample_global_state.copy()
        error_state["workflow_status"] = "ERROR"
        error_state["error_messages"] = ["Test error message"]

        with patch.object(
            workflow_manager, "_get_workflow_graph", return_value=mock_workflow_graph
        ):
            mock_workflow_graph.ainvoke.return_value = error_state

            # Execute
            result = await workflow_manager.trigger_workflow_step(
                session_id, sample_global_state
            )

            # Verify
            assert result["workflow_status"] == "ERROR"
            assert "Test error message" in result["error_messages"]
            mock_workflow_graph.ainvoke.assert_called_once_with(sample_global_state)

    @pytest.mark.asyncio
    async def test_workflow_raises_error_for_nonexistent_session(
        self, workflow_manager, sample_global_state
    ):
        """Test that workflow raises ValueError for non-existent session."""
        session_id = "nonexistent-session"

        with pytest.raises(ValueError, match="No active workflow found"):
            await workflow_manager.trigger_workflow_step(
                session_id, sample_global_state
            )

    def test_save_state_helper_method(self, workflow_manager, sample_global_state):
        """Test the _save_state helper method."""
        session_id = "test-session-123"

        # Execute
        workflow_manager._save_state(session_id, sample_global_state)

        # Verify file was created and contains correct data
        session_file = workflow_manager.sessions_dir / f"{session_id}.json"
        assert session_file.exists()

        with open(session_file, "r", encoding="utf-8") as f:
            saved_data = json.load(f)
            assert saved_data["session_id"] == session_id
            assert saved_data["cv_text"] == "Sample CV text"
            assert saved_data["workflow_status"] == "PROCESSING"
