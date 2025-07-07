"""Test for CV parser node fix to handle empty sections correctly."""

import pytest
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

from src.orchestration.state import AgentState
from src.models.cv_models import StructuredCV, Section, Item, ItemStatus
from src.orchestration.cv_workflow_graph import CVWorkflowGraph


class TestCVParserNodeFix:
    """Test cases for CV parser node fix."""

    @pytest.fixture
    def workflow_graph(self):
        """Create a CVWorkflowGraph instance for testing."""
        return CVWorkflowGraph(session_id="test-session")

    @pytest.fixture
    def empty_structured_cv(self):
        """Create a StructuredCV with empty sections (like from create_empty)."""
        return StructuredCV.create_empty()

    @pytest.fixture
    def populated_structured_cv(self):
        """Create a StructuredCV with items in sections."""
        item = Item(content="Test content", status=ItemStatus.INITIAL)
        section = Section(
            name="Test Section",
            content_type="DYNAMIC",
            order=0,
            status=ItemStatus.INITIAL,
            items=[item]
        )
        return StructuredCV(sections=[section])

    def test_has_meaningful_cv_content_with_none(self, workflow_graph):
        """Test _has_meaningful_cv_content returns False for None."""
        assert not workflow_graph._has_meaningful_cv_content(None)

    def test_has_meaningful_cv_content_with_empty_sections(self, workflow_graph, empty_structured_cv):
        """Test _has_meaningful_cv_content returns False for empty sections."""
        assert not workflow_graph._has_meaningful_cv_content(empty_structured_cv)

    def test_has_meaningful_cv_content_with_populated_sections(self, workflow_graph, populated_structured_cv):
        """Test _has_meaningful_cv_content returns True for sections with items."""
        assert workflow_graph._has_meaningful_cv_content(populated_structured_cv)

    def test_has_meaningful_cv_content_with_no_sections(self, workflow_graph):
        """Test _has_meaningful_cv_content returns False for CV with no sections."""
        cv = StructuredCV(sections=[])
        assert not workflow_graph._has_meaningful_cv_content(cv)

    @pytest.mark.asyncio
    async def test_cv_parser_node_skips_with_populated_cv(self, workflow_graph, populated_structured_cv):
        """Test that cv_parser_node skips parsing when CV has meaningful content."""
        # Arrange
        state = AgentState(
            session_id="test-session",
            structured_cv=populated_structured_cv,
            cv_text="Sample CV text"
        )
        
        # Act
        result = await workflow_graph.cv_parser_node(state)
        
        # Assert
        assert result == {}  # Should return empty dict when skipping

    @pytest.mark.asyncio
    async def test_cv_parser_node_processes_with_empty_cv(self, workflow_graph, empty_structured_cv):
        """Test that cv_parser_node processes when CV has empty sections."""
        # Arrange
        mock_parser_agent = AsyncMock()
        mock_parser_agent.run_as_node.return_value = {
            "structured_cv": empty_structured_cv  # This would be populated by real parser
        }
        workflow_graph.user_cv_parser_agent = mock_parser_agent
        
        state = AgentState(
            session_id="test-session",
            structured_cv=empty_structured_cv,
            cv_text="Sample CV text"
        )
        
        # Act
        result = await workflow_graph.cv_parser_node(state)
        
        # Assert
        mock_parser_agent.run_as_node.assert_called_once_with(state)
        assert "structured_cv" in result

    @pytest.mark.asyncio
    async def test_cv_parser_node_processes_with_none_cv(self, workflow_graph):
        """Test that cv_parser_node processes when CV is None."""
        # Arrange
        mock_parser_agent = AsyncMock()
        populated_cv = StructuredCV(sections=[
            Section(
                name="Test Section",
                content_type="DYNAMIC",
                order=0,
                status=ItemStatus.INITIAL,
                items=[Item(content="Test content", status=ItemStatus.INITIAL)]
            )
        ])
        mock_parser_agent.run_as_node.return_value = {
            "structured_cv": populated_cv
        }
        workflow_graph.user_cv_parser_agent = mock_parser_agent
        
        state = AgentState(
            session_id="test-session",
            structured_cv=None,
            cv_text="Sample CV text"
        )
        
        # Act
        result = await workflow_graph.cv_parser_node(state)
        
        # Assert
        mock_parser_agent.run_as_node.assert_called_once_with(state)
        assert "structured_cv" in result
        assert result["structured_cv"] == populated_cv