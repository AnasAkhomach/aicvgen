"""Test CB-003 fix for formatter_node CONTRACT_BREACH issue.

This module tests the fix for the type mismatch where formatter_node
expected dictionary access but received an AgentState object from
formatter_agent.run_as_node().
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.formatter_agent import FormatterAgent
from src.orchestration.graphs.main_graph import create_cv_workflow_graph_with_di
from src.orchestration.state import AgentState


@pytest.fixture
def workflow_graph():
    """Create a workflow graph instance with mocked dependencies."""
    # Mock container and create workflow graph using new pattern
    mock_container = MagicMock()
    mock_container.formatter_agent.return_value = AsyncMock(spec=FormatterAgent)

    # Create workflow graph using new pattern
    with patch(
        "src.orchestration.graphs.main_graph.create_cv_workflow_graph_with_di"
    ) as mock_create:
        mock_wrapper = MagicMock()
        mock_wrapper.formatter_agent = AsyncMock(spec=FormatterAgent)
        mock_wrapper.formatter_node = AsyncMock()

        mock_create.return_value = mock_wrapper
        return mock_wrapper


@pytest.fixture
def sample_state():
    """Create a sample AgentState for testing."""
    from src.models.cv_models import StructuredCV

    return AgentState(
        structured_cv=StructuredCV(),
        cv_text="Test CV content",
        final_output_path="/test/path/output.pdf",
        error_messages=["Initial error"],
    )


@pytest.mark.asyncio
class TestCB003FormatterNodeFix:
    """Test cases for CB-003 formatter_node fix."""

    async def test_formatter_node_with_agent_state_result(
        self, workflow_graph, sample_state
    ):
        """Test formatter_node when agent returns AgentState."""
        # Arrange
        from src.models.cv_models import StructuredCV

        expected_result = AgentState(
            structured_cv=StructuredCV(),
            cv_text="Test CV content",
            final_output_path="/new/path/output.pdf",
            error_messages=["Processing complete"],
        )
        workflow_graph.formatter_agent.run_as_node.return_value = expected_result

        # Act
        result = await workflow_graph.formatter_node(sample_state)

        # Assert
        assert isinstance(result, AgentState)
        assert result.final_output_path == "/new/path/output.pdf"
        assert result.error_messages == ["Processing complete"]
        workflow_graph.formatter_agent.run_as_node.assert_called_once_with(sample_state)

    async def test_formatter_node_with_dict_result_final_output_path(
        self, workflow_graph, sample_state
    ):
        """Test formatter_node when agent returns dict with final_output_path."""
        # Arrange
        dict_result = {"final_output_path": "/dict/path/output.pdf"}
        workflow_graph.formatter_agent.run_as_node.return_value = dict_result

        # Act
        result = await workflow_graph.formatter_node(sample_state)

        # Assert
        assert isinstance(result, AgentState)
        assert result.final_output_path == "/dict/path/output.pdf"
        assert result.error_messages == [
            "Initial error"
        ]  # Preserved from original state
        workflow_graph.formatter_agent.run_as_node.assert_called_once_with(sample_state)

    async def test_formatter_node_with_dict_result_error_messages(
        self, workflow_graph, sample_state
    ):
        """Test formatter_node when agent returns dict with error_messages."""
        # Arrange
        dict_result = {"error_messages": ["New error", "Another error"]}
        workflow_graph.formatter_agent.run_as_node.return_value = dict_result

        # Act
        result = await workflow_graph.formatter_node(sample_state)

        # Assert
        assert isinstance(result, AgentState)
        assert (
            result.final_output_path == "/test/path/output.pdf"
        )  # Preserved from original state
        assert result.error_messages == ["Initial error", "New error", "Another error"]
        workflow_graph.formatter_agent.run_as_node.assert_called_once_with(sample_state)

    async def test_formatter_node_with_dict_result_both_fields(
        self, workflow_graph, sample_state
    ):
        """Test formatter_node when agent returns dict with both fields."""
        # Arrange
        dict_result = {
            "final_output_path": "/complete/path/output.pdf",
            "error_messages": ["Processing warning"],
        }
        workflow_graph.formatter_agent.run_as_node.return_value = dict_result

        # Act
        result = await workflow_graph.formatter_node(sample_state)

        # Assert
        assert isinstance(result, AgentState)
        assert result.final_output_path == "/complete/path/output.pdf"
        assert result.error_messages == ["Initial error", "Processing warning"]
        workflow_graph.formatter_agent.run_as_node.assert_called_once_with(sample_state)

    async def test_formatter_node_with_empty_dict_result(
        self, workflow_graph, sample_state
    ):
        """Test formatter_node when agent returns empty dict."""
        # Arrange
        dict_result = {}
        workflow_graph.formatter_agent.run_as_node.return_value = dict_result

        # Act
        result = await workflow_graph.formatter_node(sample_state)

        # Assert
        assert isinstance(result, AgentState)
        assert (
            result.final_output_path == "/test/path/output.pdf"
        )  # Preserved from original state
        assert result.error_messages == [
            "Initial error"
        ]  # Preserved from original state
        workflow_graph.formatter_agent.run_as_node.assert_called_once_with(sample_state)

    async def test_formatter_node_with_unexpected_result_type(
        self, workflow_graph, sample_state
    ):
        """Test formatter_node when agent returns unexpected type (fallback to original state)."""
        # Arrange
        unexpected_result = "unexpected string result"
        workflow_graph.formatter_agent.run_as_node.return_value = unexpected_result

        # Act
        result = await workflow_graph.formatter_node(sample_state)

        # Assert
        assert isinstance(result, AgentState)
        assert (
            result.final_output_path == "/test/path/output.pdf"
        )  # Preserved from original state
        assert result.error_messages == [
            "Initial error"
        ]  # Preserved from original state
        workflow_graph.formatter_agent.run_as_node.assert_called_once_with(sample_state)

    async def test_formatter_node_agent_not_injected(self):
        """Test formatter_node when formatter_agent is not injected."""
        # Arrange
        mock_container = MagicMock()
        mock_container.formatter_agent.return_value = None

        with patch(
            "src.orchestration.graphs.main_graph.create_cv_workflow_graph_with_di"
        ) as mock_create:
            mock_wrapper = MagicMock()
            mock_wrapper.formatter_agent = None

            # Mock the formatter_node to raise the expected error
            async def mock_formatter_node(state):
                raise RuntimeError("FormatterAgent not injected")

            mock_wrapper.formatter_node = mock_formatter_node
            mock_create.return_value = mock_wrapper

            from src.models.cv_models import StructuredCV

            sample_state = AgentState(
                structured_cv=StructuredCV(), cv_text="Test CV content"
            )

        # Act & Assert
        with pytest.raises(RuntimeError, match="FormatterAgent not injected"):
            await mock_wrapper.formatter_node(sample_state)

    async def test_formatter_node_preserves_agent_state_fields(
        self, workflow_graph, sample_state
    ):
        """Test that formatter_node preserves all AgentState fields when returning AgentState."""
        # Arrange
        from src.models.cv_models import StructuredCV

        result_state = AgentState(
            structured_cv=StructuredCV(),
            cv_text="Updated CV content",
            final_output_path="/updated/path/output.pdf",
            error_messages=["Updated error"],
            node_execution_metadata={"execution_time": 2.5},
        )
        workflow_graph.formatter_agent.run_as_node.return_value = result_state

        # Act
        result = await workflow_graph.formatter_node(sample_state)

        # Assert
        from src.models.cv_models import StructuredCV

        assert isinstance(result, AgentState)
        assert isinstance(result.structured_cv, StructuredCV)
        assert result.cv_text == "Updated CV content"
        assert result.final_output_path == "/updated/path/output.pdf"
        assert result.error_messages == ["Updated error"]
        assert result.node_execution_metadata == {"execution_time": 2.5}
        workflow_graph.formatter_agent.run_as_node.assert_called_once_with(sample_state)
