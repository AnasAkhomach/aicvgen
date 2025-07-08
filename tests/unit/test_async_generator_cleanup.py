"""Test async generator cleanup in CV workflow graph."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.orchestration.cv_workflow_graph import CVWorkflowGraph
from src.orchestration.state import AgentState
from src.models.cv_models import StructuredCV


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
    @patch.object(CVWorkflowGraph, '_build_graph')
    @patch('src.orchestration.cv_workflow_graph.get_config')
    @patch('builtins.open', new_callable=MagicMock)
    async def test_astream_generator_cleanup_on_normal_completion(
        self, mock_open, mock_get_config, mock_build_graph, mock_agents
    ):
        """Test that astream generator is properly closed on normal completion."""
        # Mock the graph building
        mock_workflow = MagicMock()
        mock_build_graph.return_value = mock_workflow
        mock_app = MagicMock()
        mock_workflow.compile.return_value = mock_app
        
        # Create a mock async generator with aclose method
        mock_stream = AsyncMock()
        mock_stream.aclose = AsyncMock()
        
        # Mock astream to return our controlled generator
        async def mock_astream_generator():
            yield {"test_node": {"workflow_status": "COMPLETED"}}
        
        mock_stream.__aiter__ = lambda self: mock_astream_generator()
        mock_app.astream.return_value = mock_stream
        
        # Create workflow graph
        workflow_graph = CVWorkflowGraph(session_id="test-session", **mock_agents)
        
        # Create initial state
        initial_state = AgentState(
            structured_cv=StructuredCV(),
            cv_text="Test CV",
            workflow_status="PROCESSING"
        )
        
        # Execute trigger_workflow_step
        result_state = await workflow_graph.trigger_workflow_step(initial_state)
        
        # Verify aclose was called
        mock_stream.aclose.assert_called_once()
        assert result_state is not None
        assert result_state.workflow_status == "COMPLETED"

    @pytest.mark.asyncio
    @patch.object(CVWorkflowGraph, '_build_graph')
    @patch('src.orchestration.cv_workflow_graph.get_config')
    @patch('builtins.open', new_callable=MagicMock)
    async def test_astream_generator_cleanup_on_exception(
        self, mock_open, mock_get_config, mock_build_graph, mock_agents
    ):
        """Test that astream generator is properly closed when an exception occurs."""
        # Mock the graph building
        mock_workflow = MagicMock()
        mock_build_graph.return_value = mock_workflow
        mock_app = MagicMock()
        mock_workflow.compile.return_value = mock_app
        
        # Create a mock async generator with aclose method
        mock_stream = AsyncMock()
        mock_stream.aclose = AsyncMock()
        
        # Mock astream to raise an exception during iteration
        async def mock_astream_generator_with_error():
            yield {"test_node": {"workflow_status": "PROCESSING"}}
            raise Exception("Test exception during streaming")
        
        mock_stream.__aiter__ = lambda self: mock_astream_generator_with_error()
        mock_app.astream.return_value = mock_stream
        
        # Create workflow graph
        workflow_graph = CVWorkflowGraph(session_id="test-session", **mock_agents)
        
        # Create initial state
        initial_state = AgentState(
            structured_cv=StructuredCV(),
            cv_text="Test CV",
            workflow_status="PROCESSING"
        )
        
        # Execute trigger_workflow_step (should handle exception gracefully)
        result_state = await workflow_graph.trigger_workflow_step(initial_state)
        
        # Verify aclose was called even when exception occurred
        mock_stream.aclose.assert_called_once()
        assert result_state is not None
        assert result_state.workflow_status == "ERROR"

    @pytest.mark.asyncio
    @patch.object(CVWorkflowGraph, '_build_graph')
    @patch('src.orchestration.cv_workflow_graph.get_config')
    @patch('builtins.open', new_callable=MagicMock)
    async def test_astream_generator_cleanup_on_early_break(
        self, mock_open, mock_get_config, mock_build_graph, mock_agents
    ):
        """Test that astream generator is properly closed when loop breaks early."""
        # Mock the graph building
        mock_workflow = MagicMock()
        mock_build_graph.return_value = mock_workflow
        mock_app = MagicMock()
        mock_workflow.compile.return_value = mock_app
        
        # Create a mock async generator with aclose method
        mock_stream = AsyncMock()
        mock_stream.aclose = AsyncMock()
        
        # Mock astream to return steps that trigger early break
        async def mock_astream_generator_with_feedback():
            yield {"test_node": {"workflow_status": "AWAITING_FEEDBACK"}}
            # This should not be reached due to early break
            yield {"test_node": {"workflow_status": "PROCESSING"}}
        
        mock_stream.__aiter__ = lambda self: mock_astream_generator_with_feedback()
        mock_app.astream.return_value = mock_stream
        
        # Create workflow graph
        workflow_graph = CVWorkflowGraph(session_id="test-session", **mock_agents)
        
        # Create initial state
        initial_state = AgentState(
            structured_cv=StructuredCV(),
            cv_text="Test CV",
            workflow_status="PROCESSING"
        )
        
        # Execute trigger_workflow_step
        result_state = await workflow_graph.trigger_workflow_step(initial_state)
        
        # Verify aclose was called even when loop broke early
        mock_stream.aclose.assert_called_once()
        assert result_state is not None
        assert result_state.workflow_status == "AWAITING_FEEDBACK"

    @pytest.mark.asyncio
    @patch.object(CVWorkflowGraph, '_build_graph')
    @patch('src.orchestration.cv_workflow_graph.get_config')
    @patch('builtins.open', new_callable=MagicMock)
    async def test_astream_generator_cleanup_handles_aclose_error(
        self, mock_open, mock_get_config, mock_build_graph, mock_agents
    ):
        """Test that errors during aclose are handled gracefully."""
        # Mock the graph building
        mock_workflow = MagicMock()
        mock_build_graph.return_value = mock_workflow
        mock_app = MagicMock()
        mock_workflow.compile.return_value = mock_app
        
        # Create a mock async generator with aclose method that raises an error
        mock_stream = AsyncMock()
        mock_stream.aclose = AsyncMock(side_effect=Exception("Error during aclose"))
        
        # Mock astream to return a simple completion
        async def mock_astream_generator():
            yield {"test_node": {"workflow_status": "COMPLETED"}}
        
        mock_stream.__aiter__ = lambda self: mock_astream_generator()
        mock_app.astream.return_value = mock_stream
        
        # Create workflow graph
        workflow_graph = CVWorkflowGraph(session_id="test-session", **mock_agents)
        
        # Create initial state
        initial_state = AgentState(
            structured_cv=StructuredCV(),
            cv_text="Test CV",
            workflow_status="PROCESSING"
        )
        
        # Execute trigger_workflow_step (should not raise exception despite aclose error)
        result_state = await workflow_graph.trigger_workflow_step(initial_state)
        
        # Verify aclose was attempted
        mock_stream.aclose.assert_called_once()
        assert result_state is not None
        assert result_state.workflow_status == "COMPLETED"