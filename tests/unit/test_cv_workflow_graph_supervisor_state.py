"""Unit tests for the _initialize_supervisor_state method in CVWorkflowGraph.

This test module specifically tests the centralized supervisor state initialization
method to ensure it correctly sets current_section_index and current_item_id
based on the structured_cv content.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from src.orchestration.cv_workflow_graph import CVWorkflowGraph
from src.orchestration.state import AgentState
from src.models.cv_models import StructuredCV, Section, Item
from src.models.data_models import JobDescriptionData


class TestCVWorkflowGraphSupervisorState:
    """Test cases for the _initialize_supervisor_state method."""

    @pytest.fixture
    def mock_agents(self):
        """Create mock agents for testing."""
        return {
            'job_description_parser_agent': AsyncMock(),
            'user_cv_parser_agent': AsyncMock(),
            'research_agent': AsyncMock(),
            'cv_analyzer_agent': AsyncMock(),
            'key_qualifications_writer_agent': AsyncMock(),
            'professional_experience_writer_agent': AsyncMock(),
            'projects_writer_agent': AsyncMock(),
            'executive_summary_writer_agent': AsyncMock(),
            'qa_agent': AsyncMock(),
            'formatter_agent': AsyncMock(),
        }

    @pytest.fixture
    def workflow_graph(self, mock_agents):
        """Create CVWorkflowGraph instance with mock agents."""
        return CVWorkflowGraph(
            session_id="test-session",
            **mock_agents
        )

    @pytest.fixture
    def structured_cv_with_items(self):
        """Create a structured CV with sections and items."""
        item1 = Item(content="Sample content 1")
        item2 = Item(content="Sample content 2")
        item3 = Item(content="Sample content 3")
        
        # Set IDs manually for testing
        item1.id = "item-1"
        item2.id = "item-2"
        item3.id = "item-3"
        
        section1 = Section(
            name="Key Qualifications",
            items=[item1, item2]
        )
        section2 = Section(
            name="Professional Experience",
            items=[item3]
        )
        
        return StructuredCV(sections=[section1, section2])

    @pytest.fixture
    def empty_structured_cv(self):
        """Create an empty structured CV."""
        return StructuredCV(sections=[])

    @pytest.fixture
    def structured_cv_empty_sections(self):
        """Create a structured CV with sections but no items."""
        section1 = Section(name="Empty Section 1", items=[])
        section2 = Section(name="Empty Section 2", items=[])
        
        return StructuredCV(sections=[section1, section2])

    def test_initialize_supervisor_state_with_valid_cv(self, workflow_graph, structured_cv_with_items):
        """Test _initialize_supervisor_state with a valid structured CV."""
        # Create state with structured CV
        state = AgentState(
            cv_text="Sample CV text",
            structured_cv=structured_cv_with_items,
            node_execution_metadata={}
        )
        
        # Call the method
        updated_state = workflow_graph._initialize_supervisor_state(state)
        
        # Verify supervisor state is correctly initialized
        assert updated_state.current_section_index == 0
        assert updated_state.current_item_id == "item-1"
        
        # Verify original state is not modified (immutability)
        assert state.current_section_index is None
        assert state.current_item_id is None

    def test_initialize_supervisor_state_with_none_cv(self, workflow_graph):
        """Test _initialize_supervisor_state with None structured_cv."""
        # Create state without structured CV
        state = AgentState(
            cv_text="Sample CV text",
            structured_cv=None,
            node_execution_metadata={}
        )
        
        # Call the method
        updated_state = workflow_graph._initialize_supervisor_state(state)
        
        # Verify supervisor state remains None
        assert updated_state.current_section_index is None
        assert updated_state.current_item_id is None

    def test_initialize_supervisor_state_with_empty_cv(self, workflow_graph, empty_structured_cv):
        """Test _initialize_supervisor_state with empty structured CV."""
        # Create state with empty structured CV
        state = AgentState(
            cv_text="Sample CV text",
            structured_cv=empty_structured_cv,
            node_execution_metadata={}
        )
        
        # Call the method
        updated_state = workflow_graph._initialize_supervisor_state(state)
        
        # Verify supervisor state remains None due to empty sections
        assert updated_state.current_section_index is None
        assert updated_state.current_item_id is None

    def test_initialize_supervisor_state_with_empty_sections(self, workflow_graph, structured_cv_empty_sections):
        """Test _initialize_supervisor_state with sections that have no items."""
        # Create state with structured CV that has sections but no items
        state = AgentState(
            cv_text="Sample CV text",
            structured_cv=structured_cv_empty_sections,
            node_execution_metadata={}
        )
        
        # Call the method
        updated_state = workflow_graph._initialize_supervisor_state(state)
        
        # Verify supervisor state remains None due to no items
        assert updated_state.current_section_index is None
        assert updated_state.current_item_id is None

    def test_initialize_supervisor_state_finds_first_section_with_items(self, workflow_graph):
        """Test that _initialize_supervisor_state finds the first section with items."""
        # Create structured CV with first section empty, second section with items
        item1 = Item(content="Content in second section")
        item1.id = "item-second-1"
        
        empty_section = Section(name="Empty Section", items=[])
        section_with_items = Section(name="Section With Items", items=[item1])
        
        cv = StructuredCV(sections=[empty_section, section_with_items])
        
        state = AgentState(
            cv_text="Sample CV text",
            structured_cv=cv,
            node_execution_metadata={}
        )
        
        # Call the method
        updated_state = workflow_graph._initialize_supervisor_state(state)
        
        # Verify supervisor state points to the first section with items (index 1)
        assert updated_state.current_section_index == 1
        assert updated_state.current_item_id == "item-second-1"

    def test_initialize_supervisor_state_preserves_other_state_fields(self, workflow_graph, structured_cv_with_items):
        """Test that _initialize_supervisor_state preserves all other state fields."""
        # Create state with various fields
        original_state = AgentState(
            cv_text="Sample CV text",
            job_description_data=JobDescriptionData(
                raw_text="Sample job description",
                parsed_requirements=["Python", "Machine Learning"]
            ),
            structured_cv=structured_cv_with_items,
            node_execution_metadata={"test_key": "test_value"},
            workflow_status="PROCESSING",
            error_messages=["Some error"]
        )
        
        # Call the method
        updated_state = workflow_graph._initialize_supervisor_state(original_state)
        
        # Verify supervisor state is set
        assert updated_state.current_section_index == 0
        assert updated_state.current_item_id == "item-1"
        
        # Verify all other fields are preserved
        assert updated_state.cv_text == original_state.cv_text
        assert updated_state.job_description_data == original_state.job_description_data
        assert updated_state.structured_cv == original_state.structured_cv
        assert updated_state.node_execution_metadata == original_state.node_execution_metadata
        assert updated_state.workflow_status == original_state.workflow_status
        assert updated_state.error_messages == original_state.error_messages

    def test_initialize_supervisor_state_with_single_item(self, workflow_graph):
        """Test _initialize_supervisor_state with a single item in a single section."""
        # Create structured CV with single section and single item
        item = Item(content="Single item content")
        item.id = "single-item-id"
        
        section = Section(name="Single Section", items=[item])
        cv = StructuredCV(sections=[section])
        
        state = AgentState(
            cv_text="Sample CV text",
            structured_cv=cv,
            node_execution_metadata={}
        )
        
        # Call the method
        updated_state = workflow_graph._initialize_supervisor_state(state)
        
        # Verify supervisor state is correctly set
        assert updated_state.current_section_index == 0
        assert updated_state.current_item_id == "single-item-id"