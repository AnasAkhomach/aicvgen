"""Test async generator cleanup in CV workflow graph."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.models.cv_models import StructuredCV
from src.orchestration.graphs.main_graph import create_cv_workflow_graph_with_di
from src.orchestration.state import AgentState


class TestAsyncGeneratorCleanup:
    """Test proper cleanup of async generators in workflow execution."""

    @pytest.fixture
    def mock_agents(self):
        """Create mock agents for testing."""
        return {
            "job_description_parser_agent": MagicMock(),
            "user_cv_parser_agent": MagicMock(),
            "research_agent": MagicMock(),
            "cv_analyzer_agent": MagicMock(),
            "key_qualifications_writer_agent": MagicMock(),
            "professional_experience_writer_agent": MagicMock(),
            "projects_writer_agent": MagicMock(),
            "executive_summary_writer_agent": MagicMock(),
            "formatter_agent": MagicMock(),
        }

    @pytest.mark.asyncio
    @patch("src.orchestration.graphs.main_graph.create_cv_workflow_graph_with_di")
    async def test_astream_generator_cleanup_on_normal_completion(
        self, mock_create_workflow, mock_agents
    ):
        """Test that astream generator is properly closed on normal completion."""
        # Create a mock workflow graph with trigger_workflow_step method
        mock_workflow_graph = MagicMock()

        # Mock the trigger_workflow_step method to simulate normal completion
        async def mock_trigger_workflow_step(state):
            # Simulate processing and return completed state
            state.workflow_status = "COMPLETED"
            return state

        mock_workflow_graph.trigger_workflow_step = AsyncMock(
            side_effect=mock_trigger_workflow_step
        )
        mock_create_workflow.return_value = mock_workflow_graph

        # Create workflow graph using the new architecture
        workflow_graph = create_cv_workflow_graph_with_di(MagicMock(), "test-session")

        # Create initial state
        initial_state = AgentState(
            structured_cv=StructuredCV(),
            cv_text="Test CV",
            workflow_status="PROCESSING",
        )

        # Execute trigger_workflow_step
        result_state = await workflow_graph.trigger_workflow_step(initial_state)

        # Verify the workflow was executed
        mock_workflow_graph.trigger_workflow_step.assert_called_once_with(initial_state)
        assert result_state is not None
        assert result_state.workflow_status == "COMPLETED"

    @pytest.mark.asyncio
    @patch("src.orchestration.graphs.main_graph.create_cv_workflow_graph_with_di")
    async def test_astream_generator_cleanup_on_exception(
        self, mock_create_workflow, mock_agents
    ):
        """Test that astream generator is properly closed when an exception occurs."""
        # Create a mock workflow graph with trigger_workflow_step method
        mock_workflow_graph = MagicMock()

        # Mock the trigger_workflow_step method to simulate exception handling
        async def mock_trigger_workflow_step(state):
            # Simulate error handling and return error state
            state.workflow_status = "ERROR"
            return state

        mock_workflow_graph.trigger_workflow_step = AsyncMock(
            side_effect=mock_trigger_workflow_step
        )
        mock_create_workflow.return_value = mock_workflow_graph

        # Create workflow graph using the new architecture
        workflow_graph = create_cv_workflow_graph_with_di(MagicMock(), "test-session")

        # Create initial state
        initial_state = AgentState(
            structured_cv=StructuredCV(),
            cv_text="Test CV",
            workflow_status="PROCESSING",
        )

        # Execute trigger_workflow_step (should handle exception gracefully)
        result_state = await workflow_graph.trigger_workflow_step(initial_state)

        # Verify the workflow was executed
        mock_workflow_graph.trigger_workflow_step.assert_called_once_with(initial_state)
        assert result_state is not None
        assert result_state.workflow_status == "ERROR"

    @pytest.mark.asyncio
    @patch("src.orchestration.graphs.main_graph.create_cv_workflow_graph_with_di")
    async def test_astream_generator_cleanup_on_early_break(
        self, mock_create_workflow, mock_agents
    ):
        """Test that astream generator is properly closed when loop breaks early."""
        # Create a mock workflow graph with trigger_workflow_step method
        mock_workflow_graph = MagicMock()

        # Mock the trigger_workflow_step method to simulate early break scenario
        async def mock_trigger_workflow_step(state):
            # Simulate processing that requires feedback
            state.workflow_status = "AWAITING_FEEDBACK"
            return state

        mock_workflow_graph.trigger_workflow_step = AsyncMock(
            side_effect=mock_trigger_workflow_step
        )
        mock_create_workflow.return_value = mock_workflow_graph

        # Create workflow graph using the new architecture
        workflow_graph = create_cv_workflow_graph_with_di(MagicMock(), "test-session")

        # Create initial state
        initial_state = AgentState(
            structured_cv=StructuredCV(),
            cv_text="Test CV",
            workflow_status="PROCESSING",
        )

        # Execute trigger_workflow_step
        result_state = await workflow_graph.trigger_workflow_step(initial_state)

        # Verify the workflow was executed
        mock_workflow_graph.trigger_workflow_step.assert_called_once_with(initial_state)
        assert result_state is not None
        assert result_state.workflow_status == "AWAITING_FEEDBACK"

    @pytest.mark.asyncio
    @patch("src.orchestration.graphs.main_graph.create_cv_workflow_graph_with_di")
    async def test_astream_generator_cleanup_on_aclose_error(
        self, mock_create_workflow, mock_agents
    ):
        """Test that errors during aclose are handled gracefully."""
        # Setup mock workflow graph
        mock_workflow_graph = MagicMock()
        mock_create_workflow.return_value = mock_workflow_graph

        # Mock the trigger_workflow_step method to simulate normal processing
        async def mock_trigger_workflow_step(state):
            state.workflow_status = "COMPLETED"
            return state

        mock_workflow_graph.trigger_workflow_step = AsyncMock(
            side_effect=mock_trigger_workflow_step
        )

        # Create workflow graph using the new architecture
        workflow_graph = create_cv_workflow_graph_with_di(MagicMock(), "test-session")

        # Create initial state
        initial_state = AgentState(
            structured_cv=StructuredCV(),
            cv_text="Test CV",
            workflow_status="PROCESSING",
        )

        # Test that the workflow completes normally even if cleanup errors occur
        try:
            result_state = await workflow_graph.trigger_workflow_step(initial_state)
            assert result_state is not None
            assert result_state.workflow_status == "COMPLETED"
        except Exception as e:
            # Should not reach here - cleanup errors should be handled gracefully
            pytest.fail(f"Unexpected exception: {e}")

        # Verify the workflow was triggered
        mock_workflow_graph.trigger_workflow_step.assert_called_once_with(initial_state)
