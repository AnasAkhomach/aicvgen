"""Unit tests for the supervisor state initialization in the modular workflow graph.

This test module specifically tests the centralized supervisor state initialization
method to ensure it correctly sets current_section_index and current_item_id
based on the structured_cv content using the new modular pattern.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.orchestration.graphs.main_graph import create_cv_workflow_graph_with_di
from src.orchestration.nodes.workflow_nodes import initialize_supervisor_node
from src.orchestration.state import GlobalState
from src.models.cv_models import StructuredCV, Section, Item
from src.models.data_models import JobDescriptionData
from src.agents.agent_base import AgentBase


class TestWorkflowGraphSupervisorState:
    """Test cases for the supervisor state initialization in the modular workflow graph."""

    @pytest.fixture
    def mock_container(self):
        """Create mock dependency injection container."""
        container = MagicMock()
        
        # Mock agents
        agent_types = [
            "job_description_parser_agent",
            "user_cv_parser_agent", 
            "research_agent",
            "cv_analyzer_agent",
            "key_qualifications_writer_agent",
            "professional_experience_writer_agent",
            "projects_writer_agent",
            "executive_summary_writer_agent",
            "qa_agent",
            "formatter_agent",
        ]
        
        for agent_type in agent_types:
            mock_agent = MagicMock(spec=AgentBase)
            mock_agent.run_as_node = AsyncMock()
            container.get.return_value = mock_agent
            
        return container

    @pytest.fixture
    def workflow_wrapper(self, mock_container):
        """Create WorkflowGraphWrapper instance with mocked dependencies."""
        session_id = "test-session"
        
        with patch('src.orchestration.graphs.main_graph.build_main_workflow_graph') as mock_build:
            mock_graph = MagicMock()
            mock_build.return_value = mock_graph
            mock_compiled_graph = MagicMock()
            mock_graph.compile.return_value = mock_compiled_graph
            
            wrapper = create_cv_workflow_graph_with_di(mock_container)
            return wrapper

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

    @pytest.mark.asyncio
    async def test_initialize_supervisor_state_with_valid_cv(self, workflow_wrapper, structured_cv_with_items):
        """Test initialize_supervisor_node with a valid structured CV."""
        # Create state with structured CV
        state = GlobalState(
            cv_text="Sample CV text",
            structured_cv=structured_cv_with_items,
            session_id="test-session",
            trace_id="test-trace",
            items_to_process_queue=[],
            content_generation_queue=[],
            is_initial_generation=True,
            current_section_key=None,
            current_section_index=None,
            current_item_id=None,
            current_content_type=None,
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
            automated_mode=False
        )
        
        # Call the function directly
        updated_state = await initialize_supervisor_node(state)
        
        # Verify supervisor state is correctly initialized
        assert updated_state["current_section_index"] == 0
        assert updated_state["current_item_id"] == "item-1"
        
        # Verify original state is not modified (immutability)
        assert state["current_section_index"] is None
        assert state["current_item_id"] is None

    @pytest.mark.asyncio
    async def test_initialize_supervisor_state_with_none_cv(self, workflow_wrapper):
        """Test initialize_supervisor_node with None structured_cv."""
        # Create state without structured CV
        state = GlobalState(
            cv_text="Sample CV text",
            structured_cv=None,
            session_id="test-session",
            trace_id="test-trace",
            items_to_process_queue=[],
            content_generation_queue=[],
            is_initial_generation=True,
            current_section_key=None,
            current_section_index=None,
            current_item_id=None,
            current_content_type=None,
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
            automated_mode=False
        )
        
        # Call the function directly
        updated_state = await initialize_supervisor_node(state)
        
        # Verify supervisor state remains None
        assert updated_state["current_section_index"] is None
        assert updated_state["current_item_id"] is None

    @pytest.mark.asyncio
    async def test_initialize_supervisor_state_with_empty_cv(self, workflow_wrapper, empty_structured_cv):
        """Test initialize_supervisor_node with empty structured CV."""
        # Create state with empty structured CV
        state = GlobalState(
            cv_text="Sample CV text",
            structured_cv=empty_structured_cv,
            session_id="test-session",
            trace_id="test-trace",
            items_to_process_queue=[],
            content_generation_queue=[],
            is_initial_generation=True,
            current_section_key=None,
            current_section_index=None,
            current_item_id=None,
            current_content_type=None,
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
            automated_mode=False
        )
        
        # Call the function directly
        updated_state = await initialize_supervisor_node(state)
        
        # Verify supervisor state remains None due to empty sections
        assert updated_state["current_section_index"] is None
        assert updated_state["current_item_id"] is None

    @pytest.mark.asyncio
    async def test_initialize_supervisor_state_with_empty_sections(self, workflow_wrapper, structured_cv_empty_sections):
        """Test initialize_supervisor_node with sections that have no items."""
        # Create state with structured CV that has sections but no items
        state = GlobalState(
            cv_text="Sample CV text",
            structured_cv=structured_cv_empty_sections,
            session_id="test-session",
            trace_id="test-trace",
            items_to_process_queue=[],
            content_generation_queue=[],
            is_initial_generation=True,
            current_section_key=None,
            current_section_index=None,
            current_item_id=None,
            current_content_type=None,
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
            automated_mode=False
        )
        
        # Call the function directly
        updated_state = await initialize_supervisor_node(state)
        
        # Verify supervisor state remains None due to no items
        assert updated_state["current_section_index"] is None
        assert updated_state["current_item_id"] is None

    @pytest.mark.asyncio
    async def test_initialize_supervisor_state_finds_first_section_with_items(self, workflow_wrapper):
        """Test that initialize_supervisor_node finds the first section with items."""
        # Create structured CV with first section empty, second section with items
        item1 = Item(content="Content in second section")
        item1.id = "item-second-1"
        
        empty_section = Section(name="Key Qualifications", items=[])
        section_with_items = Section(name="Professional Experience", items=[item1])
        
        cv = StructuredCV(sections=[empty_section, section_with_items])
        
        state = GlobalState(
            cv_text="Sample CV text",
            structured_cv=cv,
            session_id="test-session",
            trace_id="test-trace",
            items_to_process_queue=[],
            content_generation_queue=[],
            is_initial_generation=True,
            current_section_key=None,
            current_section_index=None,
            current_item_id=None,
            current_content_type=None,
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
            automated_mode=False
        )
        
        # Call the function directly
        updated_state = await initialize_supervisor_node(state)
        
        # Verify supervisor state points to the first section with items (index 1)
        assert updated_state["current_section_index"] == 1
        assert updated_state["current_item_id"] == "item-second-1"

    @pytest.mark.asyncio
    async def test_initialize_supervisor_state_preserves_other_state_fields(self, workflow_wrapper, structured_cv_with_items):
        """Test that initialize_supervisor_node preserves all other state fields."""
        # Create state with various fields
        original_state = GlobalState(
            cv_text="Sample CV text",
            job_description_data=JobDescriptionData(
                raw_text="Sample job description",
                parsed_requirements=["Python", "Machine Learning"]
            ),
            structured_cv=structured_cv_with_items,
            session_id="test-session",
            trace_id="test-trace",
            items_to_process_queue=[],
            content_generation_queue=[],
            is_initial_generation=True,
            current_section_key=None,
            current_section_index=None,
            current_item_id=None,
            current_content_type=None,
            user_feedback=None,
            research_findings=None,
            cv_analysis_results=None,
            quality_check_results=None,
            generated_key_qualifications=None,
            final_output_path=None,
            error_messages=["Some error"],
            node_execution_metadata={"test_key": "test_value"},
            workflow_status="PROCESSING",
            ui_display_data={},
            automated_mode=False
        )
        
        # Call the function directly
        updated_state = await initialize_supervisor_node(original_state)
        
        # Verify supervisor state is set
        assert updated_state["current_section_index"] == 0
        assert updated_state["current_item_id"] == "item-1"
        
        # Verify all other fields are preserved
        assert updated_state["cv_text"] == original_state["cv_text"]
        assert updated_state["job_description_data"] == original_state["job_description_data"]
        assert updated_state["structured_cv"] == original_state["structured_cv"]
        assert updated_state["node_execution_metadata"] == original_state["node_execution_metadata"]
        assert updated_state["workflow_status"] == original_state["workflow_status"]
        assert updated_state["error_messages"] == original_state["error_messages"]

    @pytest.mark.asyncio
    async def test_initialize_supervisor_state_with_single_item(self, workflow_wrapper):
        """Test initialize_supervisor_node with a single item in a single section."""
        # Create structured CV with single section and single item
        item = Item(content="Single item content")
        item.id = "single-item-id"
        
        section = Section(name="Key Qualifications", items=[item])
        cv = StructuredCV(sections=[section])
        
        state = GlobalState(
            cv_text="Sample CV text",
            structured_cv=cv,
            session_id="test-session",
            trace_id="test-trace",
            items_to_process_queue=[],
            content_generation_queue=[],
            is_initial_generation=True,
            current_section_key=None,
            current_section_index=None,
            current_item_id=None,
            current_content_type=None,
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
            automated_mode=False
        )
        
        # Call the function directly
        updated_state = await initialize_supervisor_node(state)
        
        # Verify supervisor state is correctly set
        assert updated_state["current_section_index"] == 0
        assert updated_state["current_item_id"] == "single-item-id"