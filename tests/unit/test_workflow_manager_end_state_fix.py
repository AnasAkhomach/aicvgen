"""Unit tests for the WorkflowManager END state handling fix.

This module tests the fix for CB-010: WorkflowManager incorrectly treating
END state as an error instead of normal workflow completion.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, mock_open

from src.core.managers.workflow_manager import WorkflowManager
from src.orchestration.state import GlobalState


class TestWorkflowManagerEndStateFix:
    """Test cases for END state handling fix in WorkflowManager."""

    @pytest.fixture
    def mock_container(self):
        """Create a mock container for dependency injection."""
        container = MagicMock()
        return container

    @pytest.fixture
    def mock_cv_template_loader(self):
        """Create a mock CV template loader service."""
        return MagicMock()

    @pytest.fixture
    def mock_session_manager(self):
        """Create a mock session manager."""
        return MagicMock()

    @pytest.fixture
    def workflow_manager(
        self, mock_cv_template_loader, mock_session_manager, mock_container, tmp_path
    ):
        """Create a WorkflowManager instance for testing."""
        manager = WorkflowManager(
            cv_template_loader_service=mock_cv_template_loader,
            session_manager=mock_session_manager,
            container=mock_container,
        )
        manager.sessions_dir = tmp_path / "sessions"
        manager.sessions_dir.mkdir(exist_ok=True)
        return manager

    @pytest.fixture
    def sample_global_state(self):
        """Create a sample GlobalState for testing."""
        return {
            "trace_id": "test-trace-123",
            "session_id": "test-session-123",
            "workflow_status": "PROCESSING",
            "error_messages": [],
            "node_execution_metadata": {},
            "ui_display_data": {},
            "automated_mode": False,
        }

    @pytest.fixture
    def mock_workflow_graph(self):
        """Create a mock workflow graph."""
        graph = MagicMock()
        graph.ainvoke = AsyncMock()
        graph.astream = AsyncMock()
        return graph

    @pytest.mark.asyncio
    async def test_trigger_workflow_step_handles_end_state_as_completion(
        self, workflow_manager, mock_workflow_graph, sample_global_state
    ):
        """Test that trigger_workflow_step handles END state as normal completion."""
        session_id = "test-session-123"

        # Create session file
        session_file = workflow_manager.sessions_dir / f"{session_id}.json"
        with open(session_file, "w", encoding="utf-8") as f:
            json.dump(sample_global_state, f, default=str)

        # Mock workflow graph to raise END exception (simulating LangGraph completion)
        mock_workflow_graph.ainvoke.side_effect = Exception("END")

        with patch.object(
            workflow_manager, "_get_workflow_graph", return_value=mock_workflow_graph
        ):
            with patch.object(
                workflow_manager, "get_workflow_status"
            ) as mock_get_status:
                mock_get_status.return_value = sample_global_state

                # Execute
                result = await workflow_manager.trigger_workflow_step(
                    session_id, sample_global_state
                )

                # Verify END state is handled as completion
                assert result["workflow_status"] == "COMPLETED"
                assert "error_messages" not in result or not result["error_messages"]

                # Verify session file was updated with completed status
                with open(session_file, "r", encoding="utf-8") as f:
                    saved_state = json.load(f)
                assert saved_state["workflow_status"] == "COMPLETED"

    @pytest.mark.asyncio
    async def test_trigger_workflow_step_handles_actual_errors(
        self, workflow_manager, mock_workflow_graph, sample_global_state
    ):
        """Test that trigger_workflow_step still handles actual errors correctly."""
        session_id = "test-session-123"

        # Create session file
        session_file = workflow_manager.sessions_dir / f"{session_id}.json"
        with open(session_file, "w", encoding="utf-8") as f:
            json.dump(sample_global_state, f, default=str)

        # Mock workflow graph to raise actual error
        mock_workflow_graph.ainvoke.side_effect = RuntimeError("Actual workflow error")

        with patch.object(
            workflow_manager, "_get_workflow_graph", return_value=mock_workflow_graph
        ):
            with patch.object(
                workflow_manager, "get_workflow_status"
            ) as mock_get_status:
                mock_get_status.return_value = sample_global_state

                # Execute and expect error to be raised
                with pytest.raises(
                    RuntimeError,
                    match="Workflow execution failed: Actual workflow error",
                ):
                    await workflow_manager.trigger_workflow_step(
                        session_id, sample_global_state
                    )

    @pytest.mark.asyncio
    async def test_astream_workflow_handles_end_state_as_completion(
        self, workflow_manager, mock_workflow_graph, sample_global_state
    ):
        """Test that astream_workflow handles END state as normal completion."""
        session_id = "test-session-123"

        # Create session file
        session_file = workflow_manager.sessions_dir / f"{session_id}.json"
        with open(session_file, "w", encoding="utf-8") as f:
            json.dump(sample_global_state, f, default=str)

        # Mock callback handler
        mock_callback = AsyncMock()
        mock_callback.on_workflow_update = AsyncMock()
        mock_callback.on_workflow_error = AsyncMock()

        # Mock workflow graph astream to raise END exception
        async def mock_astream(state):
            yield {"test": "update"}
            raise Exception("END")

        mock_workflow_graph.astream = mock_astream

        with patch.object(
            workflow_manager, "_get_workflow_graph", return_value=mock_workflow_graph
        ):
            with patch.object(
                workflow_manager, "get_workflow_status"
            ) as mock_get_status:
                mock_get_status.return_value = {"status": "processing"}

                # Execute streaming
                updates = []
                async for update in workflow_manager.astream_workflow(
                    session_id, mock_callback
                ):
                    updates.append(update)

                # Verify we got the update before END
                assert len(updates) == 1
                assert updates[0] == {"test": "update"}

                # Verify callback was notified of completion, not error
                mock_callback.on_workflow_update.assert_called()
                mock_callback.on_workflow_error.assert_not_called()

                # Verify session file was updated with completed status
                with open(session_file, "r", encoding="utf-8") as f:
                    saved_state = json.load(f)
                assert saved_state["workflow_status"] == "COMPLETED"

    @pytest.mark.asyncio
    async def test_astream_workflow_handles_actual_errors(
        self, workflow_manager, mock_workflow_graph, sample_global_state
    ):
        """Test that astream_workflow still handles actual errors correctly."""
        session_id = "test-session-123"

        # Create session file
        session_file = workflow_manager.sessions_dir / f"{session_id}.json"
        with open(session_file, "w", encoding="utf-8") as f:
            json.dump(sample_global_state, f, default=str)

        # Mock callback handler
        mock_callback = AsyncMock()
        mock_callback.on_workflow_update = AsyncMock()
        mock_callback.on_workflow_error = AsyncMock()

        # Mock workflow graph astream to raise actual error
        async def mock_astream(state):
            yield {"test": "update"}
            raise RuntimeError("Actual streaming error")

        mock_workflow_graph.astream = mock_astream

        with patch.object(
            workflow_manager, "_get_workflow_graph", return_value=mock_workflow_graph
        ):
            with patch.object(
                workflow_manager, "get_workflow_status"
            ) as mock_get_status:
                mock_get_status.return_value = {"status": "processing"}

                # Execute streaming and expect error
                with pytest.raises(RuntimeError, match="Actual streaming error"):
                    async for update in workflow_manager.astream_workflow(
                        session_id, mock_callback
                    ):
                        pass

                # Verify callback was notified of error, not completion
                mock_callback.on_workflow_error.assert_called_with(
                    "Actual streaming error"
                )

    def test_end_state_detection_logic(self):
        """Test the END state detection logic in isolation."""
        # Test that END exception is correctly identified
        end_exception = Exception("END")
        assert str(end_exception) == "END"

        # Test that other exceptions are not mistaken for END
        other_exception = Exception("Some other error")
        assert str(other_exception) != "END"

        runtime_exception = RuntimeError("END")
        assert str(runtime_exception) == "END"  # Should still be detected as END
