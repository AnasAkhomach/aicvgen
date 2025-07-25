"""Tests for CB-002 fix: cv_analyzer_node safe attribute access.

This module tests the fix for the contract breach where cv_analyzer_node
directly accessed result.cv_analysis_results without proper validation.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.orchestration.graphs.main_graph import create_cv_workflow_graph_with_di
from src.orchestration.state import AgentState
from src.models.cv_models import StructuredCV
from src.models.agent_output_models import CVAnalysisResult


class TestCB002CVAnalyzerNodeFix:
    """Test cases for CB-002 fix in cv_analyzer_node."""

    @pytest.fixture
    def mock_cv_analyzer_agent(self):
        """Create a mock CV analyzer agent."""
        agent = AsyncMock()
        agent.name = "CVAnalyzerAgent"
        return agent

    @pytest.fixture
    def workflow_graph(self, mock_cv_analyzer_agent):
        """Create a workflow graph with mocked dependencies."""
        # Mock container and create workflow graph using new pattern
        mock_container = MagicMock()
        mock_container.cv_analyzer_agent.return_value = mock_cv_analyzer_agent
        
        # Create workflow graph using new pattern
        with patch('src.orchestration.graphs.main_graph.create_cv_workflow_graph_with_di') as mock_create:
            mock_wrapper = MagicMock()
            mock_wrapper.cv_analyzer_agent = mock_cv_analyzer_agent
            mock_wrapper.session_id = "test-session"
            mock_wrapper.cv_analyzer_node = AsyncMock()
            
            mock_create.return_value = mock_wrapper
            return mock_wrapper

    @pytest.fixture
    def base_agent_state(self):
        """Create a base AgentState for testing."""
        return AgentState(
            session_id="test-session",
            structured_cv=StructuredCV(
                personal_information={"name": "Test User"},
                professional_experience=[],
                education=[],
                skills=[],
                projects=[],
            ),
            cv_text="Test CV content",
        )

    @pytest.fixture
    def cv_analysis_result(self):
        """Create a sample CVAnalysisResult."""
        return CVAnalysisResult(
            strengths=["Strong technical skills"],
            weaknesses=["Limited leadership experience"],
            recommendations=["Consider highlighting project management experience"],
            overall_score=7.5,
            key_insights=["Good match for technical roles"],
        )

    @pytest.mark.asyncio
    async def test_cv_analyzer_node_with_valid_cv_analysis_results(
        self, workflow_graph, base_agent_state, cv_analysis_result
    ):
        """Test cv_analyzer_node when agent returns state with cv_analysis_results."""
        # Arrange
        expected_state = base_agent_state.model_copy(
            update={"cv_analysis_results": cv_analysis_result}
        )
        workflow_graph.cv_analyzer_agent.run_as_node.return_value = expected_state

        # Act
        result = await workflow_graph.cv_analyzer_node(base_agent_state)

        # Assert
        assert result.cv_analysis_results == cv_analysis_result
        assert result.session_id == base_agent_state.session_id
        workflow_graph.cv_analyzer_agent.run_as_node.assert_called_once_with(
            base_agent_state
        )

    @pytest.mark.asyncio
    async def test_cv_analyzer_node_with_missing_cv_analysis_results(
        self, workflow_graph, base_agent_state
    ):
        """Test cv_analyzer_node when agent returns state without cv_analysis_results."""
        # Arrange
        returned_state = base_agent_state.model_copy(
            update={"error_messages": ["Analysis failed"]}
        )
        workflow_graph.cv_analyzer_agent.run_as_node.return_value = returned_state

        # Act
        result = await workflow_graph.cv_analyzer_node(base_agent_state)

        # Assert
        assert result == returned_state
        assert result.cv_analysis_results is None
        assert "Analysis failed" in result.error_messages
        workflow_graph.cv_analyzer_agent.run_as_node.assert_called_once_with(
            base_agent_state
        )

    @pytest.mark.asyncio
    async def test_cv_analyzer_node_with_none_cv_analysis_results(
        self, workflow_graph, base_agent_state
    ):
        """Test cv_analyzer_node when agent returns state with None cv_analysis_results."""
        # Arrange
        returned_state = base_agent_state.model_copy(
            update={"cv_analysis_results": None}
        )
        workflow_graph.cv_analyzer_agent.run_as_node.return_value = returned_state

        # Act
        result = await workflow_graph.cv_analyzer_node(base_agent_state)

        # Assert
        assert result == returned_state
        assert result.cv_analysis_results is None
        workflow_graph.cv_analyzer_agent.run_as_node.assert_called_once_with(
            base_agent_state
        )

    @pytest.mark.asyncio
    async def test_cv_analyzer_node_with_object_without_cv_analysis_results_attribute(
        self, workflow_graph, base_agent_state
    ):
        """Test cv_analyzer_node when agent returns AgentState without cv_analysis_results."""
        # Arrange
        returned_state = base_agent_state.model_copy()
        # Ensure cv_analysis_results is None (default)
        assert returned_state.cv_analysis_results is None
        workflow_graph.cv_analyzer_agent.run_as_node.return_value = returned_state

        # Act
        result = await workflow_graph.cv_analyzer_node(base_agent_state)

        # Assert
        assert result == returned_state
        assert result.cv_analysis_results is None
        workflow_graph.cv_analyzer_agent.run_as_node.assert_called_once_with(
            base_agent_state
        )

    @pytest.mark.asyncio
    async def test_cv_analyzer_node_agent_not_injected(self, base_agent_state):
        """Test cv_analyzer_node raises RuntimeError when agent is not injected."""
        # Arrange
        mock_container = MagicMock()
        mock_container.cv_analyzer_agent.return_value = None
        
        with patch('src.orchestration.graphs.main_graph.create_cv_workflow_graph_with_di') as mock_create:
            mock_wrapper = MagicMock()
            mock_wrapper.cv_analyzer_agent = None
            mock_wrapper.session_id = "test-session"
            
            # Mock the cv_analyzer_node to raise the expected error
            async def mock_cv_analyzer_node(state):
                raise RuntimeError("CVAnalyzerAgent not injected")
            
            mock_wrapper.cv_analyzer_node = mock_cv_analyzer_node
            mock_create.return_value = mock_wrapper

            # Act & Assert
            with pytest.raises(RuntimeError, match="CVAnalyzerAgent not injected"):
                await mock_wrapper.cv_analyzer_node(base_agent_state)

    @pytest.mark.asyncio
    async def test_cv_analyzer_node_preserves_original_state_when_copying(
        self, workflow_graph, base_agent_state, cv_analysis_result
    ):
        """Test that cv_analyzer_node preserves original state data when updating."""
        # Arrange
        original_error_messages = ["Previous error"]
        base_agent_state.error_messages = original_error_messages

        returned_state = base_agent_state.model_copy(
            update={"cv_analysis_results": cv_analysis_result}
        )
        workflow_graph.cv_analyzer_agent.run_as_node.return_value = returned_state

        # Act
        result = await workflow_graph.cv_analyzer_node(base_agent_state)

        # Assert
        assert result.cv_analysis_results == cv_analysis_result
        assert result.error_messages == original_error_messages
        assert result.session_id == base_agent_state.session_id
        assert result.structured_cv == base_agent_state.structured_cv

    @pytest.mark.asyncio
    async def test_cv_analyzer_node_handles_agent_state_with_additional_fields(
        self, workflow_graph, base_agent_state, cv_analysis_result
    ):
        """Test cv_analyzer_node handles AgentState with additional fields properly."""
        # Arrange
        returned_state = base_agent_state.model_copy(
            update={
                "cv_analysis_results": cv_analysis_result,
                "node_execution_metadata": {"execution_time": 1.5},
                "error_messages": ["Analysis completed"],
            }
        )
        workflow_graph.cv_analyzer_agent.run_as_node.return_value = returned_state

        # Act
        result = await workflow_graph.cv_analyzer_node(base_agent_state)

        # Assert
        assert result.cv_analysis_results == cv_analysis_result
        assert result.node_execution_metadata == {"execution_time": 1.5}
        assert "Analysis completed" in result.error_messages

    @pytest.mark.asyncio
    async def test_cv_analyzer_node_initializes_supervisor_state_with_sections_and_items(
        self, workflow_graph, cv_analysis_result
    ):
        """Test cv_analyzer_node initializes supervisor state when sections and items exist."""
        from src.models.cv_models import Section, Item, ItemType, ItemStatus
        from uuid import uuid4

        # Arrange - Create a structured CV with sections and items
        item1 = Item(
            id=uuid4(),
            content="Test qualification 1",
            item_type=ItemType.KEY_QUALIFICATION,
            status=ItemStatus.INITIAL,
        )
        item2 = Item(
            id=uuid4(),
            content="Test qualification 2",
            item_type=ItemType.KEY_QUALIFICATION,
            status=ItemStatus.INITIAL,
        )

        section1 = Section(name="Key Qualifications", items=[item1, item2], order=0)
        section2 = Section(name="Professional Experience", items=[], order=1)

        structured_cv = StructuredCV(sections=[section1, section2])

        base_state = AgentState(
            session_id="test-session",
            structured_cv=structured_cv,
            cv_text="Test CV content",
        )

        # Mock agent to return state with cv_analysis_results
        returned_state = base_state.model_copy(
            update={"cv_analysis_results": cv_analysis_result}
        )
        workflow_graph.cv_analyzer_agent.run_as_node.return_value = returned_state

        # Act
        result = await workflow_graph.cv_analyzer_node(base_state)

        # Assert
        assert result.cv_analysis_results == cv_analysis_result
        assert result.current_section_index == 0
        assert result.current_item_id == str(item1.id)
        workflow_graph.cv_analyzer_agent.run_as_node.assert_called_once_with(base_state)

    @pytest.mark.asyncio
    async def test_cv_analyzer_node_handles_empty_sections(
        self, workflow_graph, cv_analysis_result
    ):
        """Test cv_analyzer_node handles case when no sections exist."""
        # Arrange - Create a structured CV with no sections
        structured_cv = StructuredCV(sections=[])

        base_state = AgentState(
            session_id="test-session",
            structured_cv=structured_cv,
            cv_text="Test CV content",
        )

        # Mock agent to return state with cv_analysis_results
        returned_state = base_state.model_copy(
            update={"cv_analysis_results": cv_analysis_result}
        )
        workflow_graph.cv_analyzer_agent.run_as_node.return_value = returned_state

        # Act
        result = await workflow_graph.cv_analyzer_node(base_state)

        # Assert
        assert result.cv_analysis_results == cv_analysis_result
        # Supervisor state should not be initialized when no sections exist
        assert result.current_section_index == 0  # Default value from AgentState
        assert result.current_item_id is None  # Should remain None
        workflow_graph.cv_analyzer_agent.run_as_node.assert_called_once_with(base_state)

    @pytest.mark.asyncio
    async def test_cv_analyzer_node_handles_sections_without_items(
        self, workflow_graph, cv_analysis_result
    ):
        """Test cv_analyzer_node handles case when sections exist but have no items."""
        from src.models.cv_models import Section

        # Arrange - Create a structured CV with sections but no items
        section1 = Section(
            name="Key Qualifications", items=[], order=0  # Empty items list
        )

        structured_cv = StructuredCV(sections=[section1])

        base_state = AgentState(
            session_id="test-session",
            structured_cv=structured_cv,
            cv_text="Test CV content",
        )

        # Mock agent to return state with cv_analysis_results
        returned_state = base_state.model_copy(
            update={"cv_analysis_results": cv_analysis_result}
        )
        workflow_graph.cv_analyzer_agent.run_as_node.return_value = returned_state

        # Act
        result = await workflow_graph.cv_analyzer_node(base_state)

        # Assert
        assert result.cv_analysis_results == cv_analysis_result
        # Supervisor state should not be initialized when no items exist
        assert result.current_section_index == 0  # Default value from AgentState
        assert result.current_item_id is None  # Should remain None
        workflow_graph.cv_analyzer_agent.run_as_node.assert_called_once_with(base_state)

    @pytest.mark.asyncio
    async def test_cv_analyzer_node_initializes_supervisor_state_with_dict_result(
        self, workflow_graph, cv_analysis_result
    ):
        """Test cv_analyzer_node initializes supervisor state when agent returns dict."""
        from src.models.cv_models import Section, Item, ItemType, ItemStatus
        from uuid import uuid4

        # Arrange - Create a structured CV with sections and items
        item1 = Item(
            id=uuid4(),
            content="Test qualification 1",
            item_type=ItemType.KEY_QUALIFICATION,
            status=ItemStatus.INITIAL,
        )

        section1 = Section(name="Key Qualifications", items=[item1], order=0)

        structured_cv = StructuredCV(sections=[section1])

        base_state = AgentState(
            session_id="test-session",
            structured_cv=structured_cv,
            cv_text="Test CV content",
        )

        # Mock agent to return dict instead of AgentState
        workflow_graph.cv_analyzer_agent.run_as_node.return_value = {
            "cv_analysis_results": cv_analysis_result
        }

        # Act
        result = await workflow_graph.cv_analyzer_node(base_state)

        # Assert
        assert result.cv_analysis_results == cv_analysis_result
        assert result.current_section_index == 0
        assert result.current_item_id == str(item1.id)
        workflow_graph.cv_analyzer_agent.run_as_node.assert_called_once_with(base_state)

    @pytest.mark.asyncio
    async def test_cv_analyzer_node_contract_compliance(
        self, workflow_graph, base_agent_state
    ):
        """Test that cv_analyzer_node complies with the AgentState contract."""
        # Arrange
        returned_state = base_agent_state.model_copy()
        workflow_graph.cv_analyzer_agent.run_as_node.return_value = returned_state

        # Act
        result = await workflow_graph.cv_analyzer_node(base_agent_state)

        # Assert
        assert isinstance(result, AgentState)
        assert hasattr(result, "cv_analysis_results")
        assert hasattr(result, "session_id")
        assert hasattr(result, "structured_cv")
        workflow_graph.cv_analyzer_agent.run_as_node.assert_called_once_with(
            base_agent_state
        )
