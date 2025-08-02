"""Integration tests for refactored content nodes.

Tests the simplified thin wrapper pattern where nodes directly call agent.run_as_node(state).
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from src.orchestration.nodes.content_nodes import (
    key_qualifications_writer_node,
    key_qualifications_updater_node,
    professional_experience_writer_node,
    professional_experience_updater_node,
    projects_writer_node,
    projects_updater_node,
    executive_summary_writer_node,
    executive_summary_updater_node,
    qa_node,
)


class TestContentNodesThinWrappers:
    """Test content nodes as thin wrappers calling agent.run_as_node."""

    @pytest.fixture
    def mock_state(self):
        """Create a mock GlobalState."""
        return {
            "structured_cv": MagicMock(),
            "parsed_jd": MagicMock(),
            "current_item_id": "test_id",
            "research_data": {"test": "data"},
            "session_id": "test_session",
            "error_messages": [],
        }

    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent with run_as_node method."""
        agent = AsyncMock()
        agent.run_as_node = AsyncMock(
            return_value={"updated": True, "last_executed_node": "TEST_NODE"}
        )
        return agent

    @pytest.mark.asyncio
    async def test_key_qualifications_writer_node(self, mock_state, mock_agent):
        """Test key qualifications writer node calls agent.run_as_node."""
        result = await key_qualifications_writer_node(mock_state, agent=mock_agent)

        # Verify agent.run_as_node was called with state
        mock_agent.run_as_node.assert_called_once_with(mock_state)

        # Verify result is what agent returned
        assert result == {"updated": True, "last_executed_node": "TEST_NODE"}

    @pytest.mark.asyncio
    async def test_key_qualifications_updater_node(self, mock_state, mock_agent):
        """Test key qualifications updater node calls agent.run_as_node."""
        result = await key_qualifications_updater_node(mock_state, agent=mock_agent)

        # Verify agent.run_as_node was called with state
        mock_agent.run_as_node.assert_called_once_with(mock_state)

        # Verify result is what agent returned
        assert result == {"updated": True, "last_executed_node": "TEST_NODE"}

    @pytest.mark.asyncio
    async def test_professional_experience_writer_node(self, mock_state, mock_agent):
        """Test professional experience writer node calls agent.run_as_node."""
        result = await professional_experience_writer_node(mock_state, agent=mock_agent)

        # Verify agent.run_as_node was called with state
        mock_agent.run_as_node.assert_called_once_with(mock_state)

        # Verify result is what agent returned
        assert result == {"updated": True, "last_executed_node": "TEST_NODE"}

    @pytest.mark.asyncio
    async def test_professional_experience_updater_node(self, mock_state, mock_agent):
        """Test professional experience updater node calls agent.run_as_node."""
        result = await professional_experience_updater_node(
            mock_state, agent=mock_agent
        )

        # Verify agent.run_as_node was called with state
        mock_agent.run_as_node.assert_called_once_with(mock_state)

        # Verify result is what agent returned
        assert result == {"updated": True, "last_executed_node": "TEST_NODE"}

    @pytest.mark.asyncio
    async def test_projects_writer_node(self, mock_state, mock_agent):
        """Test projects writer node calls agent.run_as_node."""
        result = await projects_writer_node(mock_state, agent=mock_agent)

        # Verify agent.run_as_node was called with state
        mock_agent.run_as_node.assert_called_once_with(mock_state)

        # Verify result is what agent returned
        assert result == {"updated": True, "last_executed_node": "TEST_NODE"}

    @pytest.mark.asyncio
    async def test_projects_updater_node(self, mock_state, mock_agent):
        """Test projects updater node calls agent.run_as_node."""
        result = await projects_updater_node(mock_state, agent=mock_agent)

        # Verify agent.run_as_node was called with state
        mock_agent.run_as_node.assert_called_once_with(mock_state)

        # Verify result is what agent returned
        assert result == {"updated": True, "last_executed_node": "TEST_NODE"}

    @pytest.mark.asyncio
    async def test_executive_summary_writer_node(self, mock_state, mock_agent):
        """Test executive summary writer node calls agent.run_as_node."""
        result = await executive_summary_writer_node(mock_state, agent=mock_agent)

        # Verify agent.run_as_node was called with state
        mock_agent.run_as_node.assert_called_once_with(mock_state)

        # Verify result is what agent returned
        assert result == {"updated": True, "last_executed_node": "TEST_NODE"}

    @pytest.mark.asyncio
    async def test_executive_summary_updater_node(self, mock_state, mock_agent):
        """Test executive summary updater node calls agent.run_as_node."""
        result = await executive_summary_updater_node(mock_state, agent=mock_agent)

        # Verify agent.run_as_node was called with state
        mock_agent.run_as_node.assert_called_once_with(mock_state)

        # Verify result is what agent returned
        assert result == {"updated": True, "last_executed_node": "TEST_NODE"}

    @pytest.mark.asyncio
    async def test_qa_node(self, mock_state, mock_agent):
        """Test QA node calls agent.run_as_node."""
        mock_agent.run_as_node.return_value = {
            "qa_results": {"score": 0.9},
            "last_executed_node": "QA",
        }

        result = await qa_node(mock_state, agent=mock_agent)

        # Verify agent.run_as_node was called with state
        mock_agent.run_as_node.assert_called_once_with(mock_state)

        # Verify result is what agent returned
        assert result == {"qa_results": {"score": 0.9}, "last_executed_node": "QA"}


class TestContentNodesErrorHandling:
    """Test error handling in refactored content nodes."""

    @pytest.fixture
    def mock_state(self):
        """Create a mock GlobalState."""
        return {
            "structured_cv": MagicMock(),
            "parsed_jd": MagicMock(),
            "current_item_id": "test_id",
            "research_data": {"test": "data"},
            "session_id": "test_session",
            "error_messages": [],
        }

    @pytest.fixture
    def mock_failing_agent(self):
        """Create a mock agent that raises an exception."""
        agent = AsyncMock()
        agent.run_as_node = AsyncMock(side_effect=Exception("Agent execution failed"))
        return agent

    @pytest.mark.asyncio
    async def test_node_propagates_agent_errors(self, mock_state, mock_failing_agent):
        """Test that nodes propagate agent errors without additional handling."""
        # Since nodes are now thin wrappers, they should propagate exceptions
        # Error handling is centralized in the agents themselves

        with pytest.raises(Exception, match="Agent execution failed"):
            await key_qualifications_writer_node(mock_state, agent=mock_failing_agent)

        # Verify agent.run_as_node was called
        mock_failing_agent.run_as_node.assert_called_once_with(mock_state)

    @pytest.mark.asyncio
    async def test_all_nodes_propagate_errors(self, mock_state, mock_failing_agent):
        """Test that all content nodes propagate agent errors consistently."""
        nodes_to_test = [
            key_qualifications_writer_node,
            key_qualifications_updater_node,
            professional_experience_writer_node,
            professional_experience_updater_node,
            projects_writer_node,
            projects_updater_node,
            executive_summary_writer_node,
            executive_summary_updater_node,
            qa_node,
        ]

        for node_func in nodes_to_test:
            # Reset the mock for each test
            mock_failing_agent.run_as_node.reset_mock()

            with pytest.raises(Exception, match="Agent execution failed"):
                await node_func(mock_state, agent=mock_failing_agent)

            # Verify agent.run_as_node was called for each node
            mock_failing_agent.run_as_node.assert_called_once_with(mock_state)


class TestContentNodesCompliance:
    """Test that content nodes comply with the thin wrapper pattern."""

    def test_nodes_are_simple_wrappers(self):
        """Test that all content nodes are simple wrappers with minimal logic."""
        import inspect
        from src.orchestration.nodes.content_nodes import (
            key_qualifications_writer_node,
            key_qualifications_updater_node,
            professional_experience_writer_node,
            professional_experience_updater_node,
            projects_writer_node,
            projects_updater_node,
            executive_summary_writer_node,
            executive_summary_updater_node,
            qa_node,
        )

        nodes_to_check = [
            key_qualifications_writer_node,
            key_qualifications_updater_node,
            professional_experience_writer_node,
            professional_experience_updater_node,
            projects_writer_node,
            projects_updater_node,
            executive_summary_writer_node,
            executive_summary_updater_node,
            qa_node,
        ]

        for node_func in nodes_to_check:
            # Get the source code of the function
            source_lines = inspect.getsource(node_func).split("\n")

            # Filter out empty lines, comments, and docstrings
            code_lines = []
            in_docstring = False
            for line in source_lines:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                if '"""' in stripped:
                    in_docstring = not in_docstring
                    continue
                if in_docstring:
                    continue
                if stripped.startswith("async def") or stripped.startswith("def"):
                    continue
                code_lines.append(stripped)

            # Each node should have only one line of actual code: return await agent.run_as_node(state)
            assert (
                len(code_lines) == 1
            ), f"Node {node_func.__name__} has {len(code_lines)} lines of code, expected 1"
            assert (
                "return await agent.run_as_node(state)" in code_lines[0]
            ), f"Node {node_func.__name__} doesn't follow thin wrapper pattern"
