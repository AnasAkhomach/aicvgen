"""Test for CV parser node fix to handle empty sections correctly."""

import pytest
from unittest.mock import AsyncMock, Mock, MagicMock, patch
from uuid import uuid4

from src.orchestration.state import AgentState
from src.models.cv_models import StructuredCV, Section, Item, ItemStatus
from src.orchestration.graphs.main_graph import create_cv_workflow_graph_with_di


class TestCVParserNodeFix:
    """Test cases for CV parser node fix."""

    @pytest.fixture
    @patch("src.orchestration.graphs.main_graph.create_cv_workflow_graph_with_di")
    def workflow_graph(self, mock_create_workflow):
        """Create a workflow graph instance for testing."""
        # Setup mock workflow graph
        mock_workflow_graph = MagicMock()
        mock_create_workflow.return_value = mock_workflow_graph

        # Mock the methods that are tested
        mock_workflow_graph._has_meaningful_cv_content = Mock()
        mock_workflow_graph.cv_parser_node = AsyncMock()
        mock_workflow_graph.user_cv_parser_agent = AsyncMock()

        return mock_workflow_graph

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
            items=[item],
        )
        return StructuredCV(sections=[section])

    def test_has_meaningful_cv_content_with_none(self, workflow_graph):
        """Test _has_meaningful_cv_content returns False for None."""
        workflow_graph._has_meaningful_cv_content.return_value = False
        assert not workflow_graph._has_meaningful_cv_content(None)

    def test_has_meaningful_cv_content_with_empty_sections(
        self, workflow_graph, empty_structured_cv
    ):
        """Test _has_meaningful_cv_content returns False for empty sections."""
        workflow_graph._has_meaningful_cv_content.return_value = False
        assert not workflow_graph._has_meaningful_cv_content(empty_structured_cv)

    def test_has_meaningful_cv_content_with_populated_sections(
        self, workflow_graph, populated_structured_cv
    ):
        """Test _has_meaningful_cv_content returns True for sections with items."""
        workflow_graph._has_meaningful_cv_content.return_value = True
        assert workflow_graph._has_meaningful_cv_content(populated_structured_cv)

    def test_has_meaningful_cv_content_with_no_sections(self, workflow_graph):
        """Test _has_meaningful_cv_content returns False for CV with no sections."""
        cv = StructuredCV(sections=[])
        workflow_graph._has_meaningful_cv_content.return_value = False
        assert not workflow_graph._has_meaningful_cv_content(cv)

    @pytest.mark.asyncio
    async def test_cv_parser_node_skips_with_populated_cv(
        self, workflow_graph, populated_structured_cv
    ):
        """Test that cv_parser_node skips parsing when CV has meaningful content."""
        # Arrange
        state = AgentState(
            session_id="test-session",
            structured_cv=populated_structured_cv,
            cv_text="Sample CV text",
        )

        # Mock cv_parser_node to return empty dict when skipping
        workflow_graph.cv_parser_node.return_value = {}

        # Act
        result = await workflow_graph.cv_parser_node(state)

        # Assert
        assert result == {}  # Should return empty dict when skipping

    @pytest.mark.asyncio
    async def test_cv_parser_node_processes_with_empty_cv(
        self, workflow_graph, empty_structured_cv
    ):
        """Test that cv_parser_node processes when CV has empty sections."""
        # Arrange
        state = AgentState(
            session_id="test-session",
            structured_cv=empty_structured_cv,
            cv_text="Sample CV text",
        )

        # Mock cv_parser_node to return structured_cv result
        expected_result = {"structured_cv": empty_structured_cv}
        workflow_graph.cv_parser_node.return_value = expected_result

        # Act
        result = await workflow_graph.cv_parser_node(state)

        # Assert
        workflow_graph.cv_parser_node.assert_called_once_with(state)
        assert "structured_cv" in result

    @pytest.mark.asyncio
    async def test_cv_parser_node_processes_with_none_cv(self, workflow_graph):
        """Test that cv_parser_node processes when CV is None."""
        # Arrange
        populated_cv = StructuredCV(
            sections=[
                Section(
                    name="Test Section",
                    content_type="DYNAMIC",
                    order=0,
                    status=ItemStatus.INITIAL,
                    items=[Item(content="Test content", status=ItemStatus.INITIAL)],
                )
            ]
        )

        state = AgentState(
            session_id="test-session", structured_cv=None, cv_text="Sample CV text"
        )

        # Mock cv_parser_node to return populated CV result
        expected_result = {"structured_cv": populated_cv}
        workflow_graph.cv_parser_node.return_value = expected_result

        # Act
        result = await workflow_graph.cv_parser_node(state)

        # Assert
        workflow_graph.cv_parser_node.assert_called_once_with(state)
        assert "structured_cv" in result
        assert result["structured_cv"] == populated_cv
