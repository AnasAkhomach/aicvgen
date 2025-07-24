import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import uuid

from src.orchestration.cv_workflow_graph import CVWorkflowGraph
from src.agents.agent_base import AgentBase
from src.orchestration.state import GlobalState
from src.models.cv_models import StructuredCV, Section, Item, ItemType, ItemStatus
from src.models.workflow_models import UserFeedback, UserAction


class TestCVWorkflowGraph:
    @pytest.fixture
    def mock_agents(self):
        """Create mock agents for testing."""
        agents = {}
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
            # Create a valid GlobalState for the mock return value
            mock_state = GlobalState(
            session_id="test-session",
            trace_id="test-trace",
            structured_cv=StructuredCV(),
            cv_text="Sample CV text",
            job_description_data=None,
            current_section_key=None,
            current_section_index=None,
            items_to_process_queue=[],
            current_item_id=None,
            current_content_type=None,
            is_initial_generation=True,
            content_generation_queue=[],
            user_feedback=None,
            research_findings=None,
            quality_check_results=None,
            cv_analysis_results=None,
            generated_key_qualifications=None,
            final_output_path=None,
            error_messages=[],
            node_execution_metadata={},
            workflow_status="PROCESSING",
            ui_display_data={},
            automated_mode=False
        )
            mock_agent.run_as_node = AsyncMock(return_value=mock_state)
            agents[agent_type] = mock_agent
            
        return agents
    
    @patch.object(CVWorkflowGraph, '_build_graph')
    def test_initialization_with_injected_agents(self, mock_build_graph, mock_agents):
        """Test that CVWorkflowGraph initializes correctly with injected agents."""
        # Mock the graph building to avoid LangGraph issues in tests
        mock_workflow = MagicMock()
        mock_build_graph.return_value = mock_workflow
        mock_workflow.compile.return_value = MagicMock()
        
        session_id = str(uuid.uuid4())
        
        # Initialize with all agents
        graph = CVWorkflowGraph(
            session_id=session_id,
            job_description_parser_agent=mock_agents["job_description_parser_agent"],
            user_cv_parser_agent=mock_agents["user_cv_parser_agent"],
            research_agent=mock_agents["research_agent"],
            cv_analyzer_agent=mock_agents["cv_analyzer_agent"],
            key_qualifications_writer_agent=mock_agents["key_qualifications_writer_agent"],
            professional_experience_writer_agent=mock_agents["professional_experience_writer_agent"],
            projects_writer_agent=mock_agents["projects_writer_agent"],
            executive_summary_writer_agent=mock_agents["executive_summary_writer_agent"],
            qa_agent=mock_agents["qa_agent"],
            formatter_agent=mock_agents["formatter_agent"],
        )
        
        # Verify that all agents are correctly assigned
        assert graph.session_id == session_id
        assert graph.job_description_parser_agent == mock_agents["job_description_parser_agent"]
        assert graph.user_cv_parser_agent == mock_agents["user_cv_parser_agent"]
        assert graph.research_agent == mock_agents["research_agent"]
        assert graph.cv_analyzer_agent == mock_agents["cv_analyzer_agent"]
        assert graph.key_qualifications_writer_agent == mock_agents["key_qualifications_writer_agent"]
        assert graph.professional_experience_writer_agent == mock_agents["professional_experience_writer_agent"]
        assert graph.projects_writer_agent == mock_agents["projects_writer_agent"]
        assert graph.executive_summary_writer_agent == mock_agents["executive_summary_writer_agent"]
        assert graph.qa_agent == mock_agents["qa_agent"]
        assert graph.formatter_agent == mock_agents["formatter_agent"]
        
        # Verify that the workflow building was attempted
        mock_build_graph.assert_called_once()
    
    @pytest.mark.asyncio
    @patch.object(CVWorkflowGraph, '_build_graph')
    async def test_jd_parser_node_with_injected_agent(self, mock_build_graph, mock_agents):
        """Test that jd_parser_node uses the injected agent."""
        # Mock the graph building to avoid LangGraph issues in tests
        mock_workflow = MagicMock()
        mock_build_graph.return_value = mock_workflow
        mock_workflow.compile.return_value = MagicMock()
        
        session_id = str(uuid.uuid4())
        graph = CVWorkflowGraph(
            session_id=session_id,
            job_description_parser_agent=mock_agents["job_description_parser_agent"],
        )
        
        # Create a valid GlobalState with required fields
        state = GlobalState(
            session_id="test-session",
            trace_id="test-trace",
            structured_cv=StructuredCV(),
            cv_text="Sample CV text",
            job_description_data=None,
            current_section_key=None,
            current_section_index=None,
            items_to_process_queue=[],
            current_item_id=None,
            current_content_type=None,
            is_initial_generation=True,
            content_generation_queue=[],
            user_feedback=None,
            research_findings=None,
            quality_check_results=None,
            cv_analysis_results=None,
            generated_key_qualifications=None,
            final_output_path=None,
            error_messages=[],
            node_execution_metadata={},
            workflow_status="PROCESSING",
            ui_display_data={},
            automated_mode=False
        )
        
        # Mock the run_as_node method to return a dict with job_description_data
        from src.models.cv_models import JobDescriptionData
        mock_jd_data = JobDescriptionData(
            raw_text="Software Engineer at Test Company. Test description. Requirements: Python, Testing. Responsibilities: Develop software.",
            job_title="Software Engineer",
            company_name="Test Company",
            main_job_description_raw="Test description",
            responsibilities=["Develop software"],
            skills=["Python", "Git"]
        )
        mock_agents["job_description_parser_agent"].run_as_node.return_value = {
            "job_description_data": mock_jd_data
        }
        
        result = await graph.jd_parser_node(state)
        
        # Verify that the injected agent's run_as_node method was called
        mock_agents["job_description_parser_agent"].run_as_node.assert_called_once_with(state)
        # Verify the result is a dict with job_description_data
        assert isinstance(result, dict)
        assert "job_description_data" in result
        assert result["job_description_data"] == mock_jd_data
    
    @pytest.mark.asyncio
    @patch.object(CVWorkflowGraph, '_build_graph')
    async def test_jd_parser_node_without_injected_agent(self, mock_build_graph):
        """Test that jd_parser_node handles missing agent gracefully."""
        # Mock the graph building to avoid LangGraph issues in tests
        mock_workflow = MagicMock()
        mock_build_graph.return_value = mock_workflow
        mock_workflow.compile.return_value = MagicMock()
        
        session_id = str(uuid.uuid4())
        graph = CVWorkflowGraph(session_id=session_id)
        
        # Create a valid GlobalState with required fields
        state = GlobalState(
            session_id="test-session",
            trace_id="test-trace",
            structured_cv=StructuredCV(),
            cv_text="Sample CV text",
            job_description_data=None,
            current_section_key=None,
            current_section_index=None,
            items_to_process_queue=[],
            current_item_id=None,
            current_content_type=None,
            is_initial_generation=True,
            content_generation_queue=[],
            user_feedback=None,
            research_findings=None,
            quality_check_results=None,
            cv_analysis_results=None,
            generated_key_qualifications=None,
            final_output_path=None,
            error_messages=[],
            node_execution_metadata={},
            workflow_status="PROCESSING",
            ui_display_data={},
            automated_mode=False
        )
        
        # Test that the node handles missing agent gracefully by returning error messages
        result = await graph.jd_parser_node(state)
        
        # Verify that an error message was returned in the dict
        assert isinstance(result, dict)
        assert "error_messages" in result
        assert len(result["error_messages"]) == 1
        assert "JobDescriptionParserAgent failed: JobDescriptionParserAgent not injected" in result["error_messages"][0]
    
    @pytest.mark.asyncio
    @patch.object(CVWorkflowGraph, '_build_graph')
    async def test_initialize_supervisor_node(self, mock_build_graph, mock_agents):
        """Test that initialize_supervisor_node correctly initializes state."""
        # Mock the graph building to avoid LangGraph issues in tests
        mock_workflow = MagicMock()
        mock_build_graph.return_value = mock_workflow
        mock_workflow.compile.return_value = MagicMock()
        
        session_id = str(uuid.uuid4())
        graph = CVWorkflowGraph(session_id=session_id)
        
        # Test with structured_cv containing sections
        from src.models.cv_models import StructuredCV, Section, Item, ItemType
        
        # Create test sections with items
        from uuid import uuid4
        item1_id = uuid4()
        item2_id = uuid4()
        
        item1 = Item(
            id=item1_id,
            item_type=ItemType.BULLET_POINT,
            content="Test experience 1",
            status=ItemStatus.PENDING
        )
        item2 = Item(
            id=item2_id, 
            item_type=ItemType.BULLET_POINT,
            content="Test experience 2",
            status=ItemStatus.PENDING
        )
        
        section1_id = uuid4()
        section2_id = uuid4()
        
        section1 = Section(
            id=section1_id,
            name="Professional Experience",
            items=[item1, item2]
        )
        section2 = Section(
            id=section2_id,
            name="Projects", 
            items=[]
        )
        
        structured_cv = StructuredCV(
            sections=[section1, section2]
        )
        
        state = GlobalState(
            session_id="test-session",
            trace_id="test-trace",
            structured_cv=structured_cv,
            cv_text="Sample CV text",
            job_description_data=None,
            current_section_key=None,
            current_section_index=None,
            items_to_process_queue=[],
            current_item_id=None,
            current_content_type=None,
            is_initial_generation=True,
            content_generation_queue=[],
            user_feedback=None,
            research_findings=None,
            quality_check_results=None,
            cv_analysis_results=None,
            generated_key_qualifications=None,
            final_output_path=None,
            error_messages=[],
            node_execution_metadata={},
            workflow_status="PROCESSING",
            ui_display_data={},
            automated_mode=False
        )
        
        result = await graph.initialize_supervisor_node(state)
        
        # Verify the result is a dict with correct initialization
        assert isinstance(result, dict)
        assert "current_section_index" in result
        assert "current_item_id" in result
        assert result["current_section_index"] == 0
        assert result["current_item_id"] == str(item1_id)
        
        # Test with current_section_index already set
        state_with_index = GlobalState(
            session_id="test-session",
            trace_id="test-trace",
            structured_cv=structured_cv,
            cv_text="Sample CV text",
            job_description_data=None,
            current_section_key=None,
            current_section_index=1,
            items_to_process_queue=[],
            current_item_id=None,
            current_content_type=None,
            is_initial_generation=True,
            content_generation_queue=[],
            user_feedback=None,
            research_findings=None,
            quality_check_results=None,
            cv_analysis_results=None,
            generated_key_qualifications=None,
            final_output_path=None,
            error_messages=[],
            node_execution_metadata={},
            workflow_status="PROCESSING",
            ui_display_data={},
            automated_mode=False
        )
        
        result2 = await graph.initialize_supervisor_node(state_with_index)
        
        # Should keep existing index but still calculate item_id from first section with items
        assert result2["current_section_index"] == 1
        assert result2["current_item_id"] == str(item1_id)  # Still finds first item from section1
        
        # Test with empty structured_cv
        empty_state = GlobalState(
            session_id="test-session",
            trace_id="test-trace",
            structured_cv=StructuredCV(sections=[]),
            cv_text="Sample CV text",
            job_description_data=None,
            current_section_key=None,
            current_section_index=None,
            items_to_process_queue=[],
            current_item_id=None,
            current_content_type=None,
            is_initial_generation=True,
            content_generation_queue=[],
            user_feedback=None,
            research_findings=None,
            quality_check_results=None,
            cv_analysis_results=None,
            generated_key_qualifications=None,
            final_output_path=None,
            error_messages=[],
            node_execution_metadata={},
            workflow_status="PROCESSING",
            ui_display_data={},
            automated_mode=False
        )
        
        result3 = await graph.initialize_supervisor_node(empty_state)
        
        # Should default to 0 and None
        assert result3["current_section_index"] == 0
        assert result3["current_item_id"] is None

    @pytest.mark.asyncio
    @patch.object(CVWorkflowGraph, '_build_graph')
    @patch('src.orchestration.cv_workflow_graph.get_config')
    @patch('builtins.open', new_callable=MagicMock)
    async def test_trigger_workflow_step_success(self, mock_open, mock_get_config, mock_build_graph, mock_agents):
        """Test successful trigger_workflow_step execution with streaming."""
        # Mock the graph building
        mock_workflow = MagicMock()
        mock_build_graph.return_value = mock_workflow
        mock_app = MagicMock()
        mock_workflow.compile.return_value = mock_app
        
        # Mock astream to return multiple steps
        step1_state = GlobalState(
            session_id="test-session",
            trace_id="test-trace",
            structured_cv=StructuredCV(),
            cv_text="Sample CV text",
            job_description_data=None,
            current_section_key=None,
            current_section_index=None,
            items_to_process_queue=[],
            current_item_id=None,
            current_content_type=None,
            is_initial_generation=True,
            content_generation_queue=[],
            user_feedback=None,
            research_findings=None,
            quality_check_results=None,
            cv_analysis_results=None,
            generated_key_qualifications=None,
            final_output_path=None,
            error_messages=[],
            node_execution_metadata={},
            workflow_status="PROCESSING",
            ui_display_data={},
            automated_mode=False
        )
        step2_state = GlobalState(
            session_id="test-session",
            trace_id="test-trace",
            structured_cv=StructuredCV(),
            cv_text="Sample CV text",
            job_description_data=None,
            current_section_key=None,
            current_section_index=None,
            items_to_process_queue=[],
            current_item_id=None,
            current_content_type=None,
            is_initial_generation=True,
            content_generation_queue=[],
            user_feedback=None,
            research_findings=None,
            quality_check_results=None,
            cv_analysis_results=None,
            generated_key_qualifications=None,
            final_output_path=None,
            error_messages=[],
            node_execution_metadata={},
            workflow_status="COMPLETED",
            ui_display_data={},
            automated_mode=False
        )
        
        async def mock_astream(state, config=None):
            yield {"step1": step1_state}
            yield {"step2": step2_state}
        
        mock_app.astream = mock_astream
        
        # Mock config and file operations
        mock_config = MagicMock()
        mock_sessions_dir = MagicMock()
        mock_config.paths.project_root = MagicMock()
        mock_config.paths.project_root.__truediv__ = MagicMock(return_value=mock_sessions_dir)
        mock_sessions_dir.__truediv__ = MagicMock(return_value=mock_sessions_dir)
        mock_sessions_dir.mkdir = MagicMock()
        mock_get_config.return_value = mock_config
        
        mock_file_handle = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file_handle
        
        session_id = str(uuid.uuid4())
        graph = CVWorkflowGraph(session_id=session_id)
        
        initial_state = GlobalState(
            session_id="test-session",
            trace_id="test-trace",
            structured_cv=StructuredCV(),
            cv_text="Sample CV text",
            job_description_data=None,
            current_section_key=None,
            current_section_index=None,
            items_to_process_queue=[],
            current_item_id=None,
            current_content_type=None,
            is_initial_generation=True,
            content_generation_queue=[],
            user_feedback=None,
            research_findings=None,
            quality_check_results=None,
            cv_analysis_results=None,
            generated_key_qualifications=None,
            final_output_path=None,
            error_messages=[],
            node_execution_metadata={},
            workflow_status="AWAITING_FEEDBACK",
            ui_display_data={},
            automated_mode=False
        )
        
        result = await graph.trigger_workflow_step(initial_state)
        
        # Verify state was saved to file (called twice for each step)
        assert mock_file_handle.write.call_count == 2
        
        # Verify final state
        assert result.workflow_status == "COMPLETED"
        assert result.ui_display_data == {}
    
    @pytest.mark.asyncio
    @patch.object(CVWorkflowGraph, '_build_graph')
    @patch('src.orchestration.cv_workflow_graph.get_config')
    @patch('builtins.open', new_callable=MagicMock)
    async def test_trigger_workflow_step_pauses_on_feedback(self, mock_open, mock_get_config, mock_build_graph, mock_agents):
        """Test that trigger_workflow_step pauses when workflow_status is AWAITING_FEEDBACK."""
        # Mock the graph building
        mock_workflow = MagicMock()
        mock_build_graph.return_value = mock_workflow
        mock_app = MagicMock()
        mock_workflow.compile.return_value = mock_app
        
        # Mock astream to return a step that requires feedback
        feedback_state = GlobalState(
            session_id="test-session",
            trace_id="test-trace",
            structured_cv=StructuredCV(),
            cv_text="Sample CV text",
            job_description_data=None,
            current_section_key=None,
            current_section_index=None,
            items_to_process_queue=[],
            current_item_id=None,
            current_content_type=None,
            is_initial_generation=True,
            content_generation_queue=[],
            user_feedback=None,
            research_findings=None,
            quality_check_results=None,
            cv_analysis_results=None,
            generated_key_qualifications=None,
            final_output_path=None,
            error_messages=[],
            node_execution_metadata={},
            workflow_status="AWAITING_FEEDBACK",
            ui_display_data={"section": "key_qualifications", "requires_feedback": True},
            automated_mode=False
        )
        
        async def mock_astream(state, config=None):
            yield {"feedback_step": feedback_state}
        
        mock_app.astream = mock_astream
        
        # Mock config and file operations
        mock_config = MagicMock()
        mock_sessions_dir = MagicMock()
        mock_config.paths.project_root = MagicMock()
        mock_config.paths.project_root.__truediv__ = MagicMock(return_value=mock_sessions_dir)
        mock_sessions_dir.__truediv__ = MagicMock(return_value=mock_sessions_dir)
        mock_sessions_dir.mkdir = MagicMock()
        mock_get_config.return_value = mock_config
        
        mock_file_handle = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file_handle
        
        session_id = str(uuid.uuid4())
        graph = CVWorkflowGraph(session_id=session_id)
        
        initial_state = GlobalState(
            session_id="test-session",
            trace_id="test-trace",
            structured_cv=StructuredCV(),
            cv_text="Sample CV text",
            job_description_data=None,
            current_section_key=None,
            current_section_index=None,
            items_to_process_queue=[],
            current_item_id=None,
            current_content_type=None,
            is_initial_generation=True,
            content_generation_queue=[],
            user_feedback=None,
            research_findings=None,
            quality_check_results=None,
            cv_analysis_results=None,
            generated_key_qualifications=None,
            final_output_path=None,
            error_messages=[],
            node_execution_metadata={},
            workflow_status="PROCESSING",
            ui_display_data={},
            automated_mode=False
        )
        
        result = await graph.trigger_workflow_step(initial_state)
        
        # Verify the workflow paused at feedback step
        assert result.workflow_status == "AWAITING_FEEDBACK"
        assert result.ui_display_data["requires_feedback"] is True
        
        # Verify state was saved
        mock_file_handle.write.assert_called_once()
    
    @pytest.mark.asyncio
    @patch.object(CVWorkflowGraph, '_build_graph')
    @patch('src.orchestration.cv_workflow_graph.get_config')
    @patch('builtins.open', new_callable=MagicMock)
    async def test_trigger_workflow_step_state_propagation(self, mock_open, mock_get_config, mock_build_graph, mock_agents):
        """Test that trigger_workflow_step correctly propagates state using model_copy."""
        # Mock the graph building
        mock_workflow = MagicMock()
        mock_build_graph.return_value = mock_workflow
        mock_app = MagicMock()
        mock_workflow.compile.return_value = mock_app
        
        # Create initial state with specific values
        initial_cv_text = "Initial CV text"
        initial_session_id = "initial-session"
        
        # Mock astream to return state updates that should be propagated
        step1_updates = {
            "cv_text": "Updated CV text from step 1",
            "workflow_status": "PROCESSING",
            "current_section_index": 1
        }
        step2_updates = {
            "research_findings": {"key_skills": ["Python", "Testing"]},
            "workflow_status": "COMPLETED"
        }
        
        async def mock_astream(state, config=None):
            yield {"step1": step1_updates}
            yield {"step2": step2_updates}
        
        mock_app.astream = mock_astream
        
        # Mock config and file operations
        mock_config = MagicMock()
        mock_sessions_dir = MagicMock()
        mock_config.paths.project_root = MagicMock()
        mock_config.paths.project_root.__truediv__ = MagicMock(return_value=mock_sessions_dir)
        mock_sessions_dir.__truediv__ = MagicMock(return_value=mock_sessions_dir)
        mock_sessions_dir.mkdir = MagicMock()
        mock_get_config.return_value = mock_config
        
        mock_file_handle = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file_handle
        
        session_id = str(uuid.uuid4())
        graph = CVWorkflowGraph(session_id=session_id)
        
        initial_state = GlobalState(
            session_id=initial_session_id,
            trace_id="test-trace",
            structured_cv=StructuredCV(),
            cv_text=initial_cv_text,
            job_description_data=None,
            current_section_key=None,
            current_section_index=0,
            items_to_process_queue=[],
            current_item_id=None,
            current_content_type=None,
            is_initial_generation=True,
            content_generation_queue=[],
            user_feedback=None,
            research_findings=None,
            quality_check_results=None,
            cv_analysis_results=None,
            generated_key_qualifications=None,
            final_output_path=None,
            error_messages=[],
            node_execution_metadata={},
            workflow_status="AWAITING_FEEDBACK",
            ui_display_data={},
            automated_mode=False
        )
        
        result = await graph.trigger_workflow_step(initial_state)
        
        # Verify that state was correctly propagated using model_copy
        # The cv_text should be updated from step1
        assert result.cv_text == "Updated CV text from step 1"
        # The research_findings should be updated from step2
        assert result.research_findings == {"key_skills": ["Python", "Testing"]}
        # The workflow_status should be the final value from step2
        assert result.workflow_status == "COMPLETED"
        # The current_section_index should be updated from step1
        assert result.current_section_index == 1
        # The session_id should remain unchanged (not in updates)
        assert result.session_id == initial_session_id
        # The ui_display_data should be reset to empty dict
        assert result.ui_display_data == {}
        
        # Verify state was saved to file (called twice for each step)
        assert mock_file_handle.write.call_count == 2

    @pytest.mark.asyncio
    @patch.object(CVWorkflowGraph, '_build_graph')
    @patch('src.orchestration.cv_workflow_graph.get_config')
    @patch('builtins.open', new_callable=MagicMock)
    async def test_trigger_workflow_step_handles_error(self, mock_open, mock_get_config, mock_build_graph, mock_agents):
        """Test that trigger_workflow_step handles errors gracefully."""
        # Mock the graph building
        mock_workflow = MagicMock()
        mock_build_graph.return_value = mock_workflow
        mock_app = MagicMock()
        mock_workflow.compile.return_value = mock_app
        
        # Mock astream to raise an exception
        async def mock_astream_error(state, config=None):
            raise Exception("Test error")
            yield  # This line will never be reached but makes it a generator
        
        mock_app.astream = mock_astream_error
        
        # Mock config and file operations
        mock_config = MagicMock()
        mock_sessions_dir = MagicMock()
        mock_config.paths.project_root = MagicMock()
        mock_config.paths.project_root.__truediv__ = MagicMock(return_value=mock_sessions_dir)
        mock_sessions_dir.__truediv__ = MagicMock(return_value=mock_sessions_dir)
        mock_sessions_dir.mkdir = MagicMock()
        mock_get_config.return_value = mock_config
        
        mock_file_handle = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file_handle
        
        session_id = str(uuid.uuid4())
        graph = CVWorkflowGraph(session_id=session_id)
        
        initial_state = GlobalState(
            session_id="test-session",
            trace_id="test-trace",
            structured_cv=StructuredCV(),
            cv_text="Sample CV text",
            job_description_data=None,
            current_section_key=None,
            current_section_index=None,
            items_to_process_queue=[],
            current_item_id=None,
            current_content_type=None,
            is_initial_generation=True,
            content_generation_queue=[],
            user_feedback=None,
            research_findings=None,
            quality_check_results=None,
            cv_analysis_results=None,
            generated_key_qualifications=None,
            final_output_path=None,
            error_messages=[],
            node_execution_metadata={},
            workflow_status="PROCESSING",
            ui_display_data={},
            automated_mode=False
        )
        
        result = await graph.trigger_workflow_step(initial_state)
        
        # Verify error was handled
        assert result.workflow_status == "ERROR"
        assert len(result.error_messages) > 0
        assert "Test error" in result.error_messages[-1]
        
        # Verify error state was saved
        mock_file_handle.write.assert_called_once()
    
    @pytest.mark.asyncio
    @patch.object(CVWorkflowGraph, '_build_graph')
    @patch('src.orchestration.cv_workflow_graph.get_config')
    @patch('builtins.open', new_callable=MagicMock)
    async def test_save_state_to_file(self, mock_open, mock_get_config, mock_build_graph):
        """Test the _save_state_to_file helper method."""
        # Mock the graph building
        mock_workflow = MagicMock()
        mock_build_graph.return_value = mock_workflow
        mock_workflow.compile.return_value = MagicMock()
        
        # Mock config and file operations
        mock_config = MagicMock()
        mock_project_root = MagicMock()
        mock_instance_dir = MagicMock()
        mock_sessions_dir = MagicMock()
        mock_session_file = MagicMock()
        
        mock_config.paths.project_root = mock_project_root
        # Mock the path chain: project_root / "instance" / "sessions"
        mock_project_root.__truediv__.return_value = mock_instance_dir
        mock_instance_dir.__truediv__.return_value = mock_sessions_dir
        mock_sessions_dir.__truediv__.return_value = mock_session_file
        mock_sessions_dir.mkdir = MagicMock()
        mock_get_config.return_value = mock_config
        
        mock_file_handle = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file_handle
        
        session_id = str(uuid.uuid4())
        graph = CVWorkflowGraph(session_id=session_id)
        
        state = GlobalState(
            session_id="test-session",
            trace_id="test-trace",
            structured_cv=StructuredCV(),
            cv_text="Sample CV text",
            job_description_data=None,
            current_section_key=None,
            current_section_index=None,
            items_to_process_queue=[],
            current_item_id=None,
            current_content_type=None,
            is_initial_generation=True,
            content_generation_queue=[],
            user_feedback=None,
            research_findings=None,
            quality_check_results=None,
            cv_analysis_results=None,
            generated_key_qualifications=None,
            final_output_path=None,
            error_messages=[],
            node_execution_metadata={},
            workflow_status="PROCESSING",
            ui_display_data={},
            automated_mode=False
        )
        
        # Call the private method
        graph._save_state_to_file(state)
        
        # Verify directory creation
        mock_sessions_dir.mkdir.assert_called_once_with(parents=True, exist_ok=True)
        
        # Verify file was opened and written to
        mock_open.assert_called_once_with(mock_session_file, 'w', encoding='utf-8')
        mock_file_handle.write.assert_called_once()
        
        # Verify JSON content was written
        written_content = mock_file_handle.write.call_args[0][0]
        assert '"workflow_status": "PROCESSING"' in written_content
        assert '"cv_text": "Sample CV text"' in written_content

    @pytest.mark.asyncio
    @patch.object(CVWorkflowGraph, '_build_graph')
    async def test_handle_feedback_node_with_valid_feedback(self, mock_build_graph):
        """Test handle_feedback_node with valid user feedback containing item_id."""
        # Mock the graph building
        mock_workflow = MagicMock()
        mock_build_graph.return_value = mock_workflow
        mock_workflow.compile.return_value = MagicMock()
        
        session_id = str(uuid.uuid4())
        graph = CVWorkflowGraph(session_id=session_id)
        
        # Create valid user feedback with item_id
        valid_feedback = UserFeedback(
            action=UserAction.APPROVE,
            item_id="test-item-123"
        )
        
        state = GlobalState(
            session_id="test-session",
            trace_id="test-trace",
            structured_cv=StructuredCV(),
            cv_text="Sample CV text",
            job_description_data=None,
            current_section_key=None,
            current_section_index=None,
            items_to_process_queue=[],
            current_item_id="test-item-123",
            current_content_type=None,
            is_initial_generation=True,
            content_generation_queue=[],
            user_feedback=valid_feedback,
            research_findings=None,
            quality_check_results=None,
            cv_analysis_results=None,
            generated_key_qualifications=None,
            final_output_path=None,
            error_messages=[],
            node_execution_metadata={},
            workflow_status="PROCESSING",
            ui_display_data={},
            automated_mode=False
        )
        
        result = await graph.handle_feedback_node(state)
        
        # Verify that the node processes valid feedback without errors
        assert result.get("error_messages", []) == []
        assert result["workflow_status"] == "PROCESSING"
        assert result.get("user_feedback") is None  # Feedback should be cleared after processing

    @pytest.mark.asyncio
    @patch.object(CVWorkflowGraph, '_build_graph')
    async def test_handle_feedback_node_with_missing_item_id(self, mock_build_graph):
        """Test handle_feedback_node validation when user_feedback has no item_id."""
        # Mock the graph building
        mock_workflow = MagicMock()
        mock_build_graph.return_value = mock_workflow
        mock_workflow.compile.return_value = MagicMock()
        
        session_id = str(uuid.uuid4())
        graph = CVWorkflowGraph(session_id=session_id)
        
        # Create user feedback with empty item_id (this should trigger validation error)
        invalid_feedback = UserFeedback(
            action=UserAction.APPROVE,
            item_id=""  # Empty string should trigger validation
        )
        
        state = GlobalState(
            session_id="test-session",
            trace_id="test-trace",
            structured_cv=StructuredCV(),
            cv_text="Sample CV text",
            job_description_data=None,
            current_section_key=None,
            current_section_index=None,
            items_to_process_queue=[],
            current_item_id=None,
            current_content_type=None,
            is_initial_generation=True,
            content_generation_queue=[],
            user_feedback=invalid_feedback,
            research_findings=None,
            quality_check_results=None,
            cv_analysis_results=None,
            generated_key_qualifications=None,
            final_output_path=None,
            error_messages=[],
            node_execution_metadata={},
            workflow_status="PROCESSING",
            ui_display_data={},
            automated_mode=False
        )
        
        # Verify that error is returned for strict validation
        result = await graph.handle_feedback_node(state)
        assert "error_messages" in result
        assert len(result["error_messages"]) == 1
        assert "Handle feedback node failed: Feedback received with no item_id" in result["error_messages"][0]

    @pytest.mark.asyncio
    @patch.object(CVWorkflowGraph, '_build_graph')
    async def test_handle_feedback_node_with_none_item_id(self, mock_build_graph):
        """Test handle_feedback_node validation when user_feedback has None item_id."""
        # Mock the graph building
        mock_workflow = MagicMock()
        mock_build_graph.return_value = mock_workflow
        mock_workflow.compile.return_value = MagicMock()
        
        session_id = str(uuid.uuid4())
        graph = CVWorkflowGraph(session_id=session_id)
        
        # Create a mock UserFeedback with None item_id
        # We need to bypass Pydantic validation for this test case
        invalid_feedback = MagicMock(spec=UserFeedback)
        invalid_feedback.action = UserAction.APPROVE
        invalid_feedback.item_id = None  # This should trigger validation
        
        state = GlobalState(
            session_id="test-session",
            trace_id="test-trace",
            structured_cv=StructuredCV(),
            cv_text="Sample CV text",
            job_description_data=None,
            current_section_key=None,
            current_section_index=None,
            items_to_process_queue=[],
            current_item_id=None,
            current_content_type=None,
            is_initial_generation=True,
            content_generation_queue=[],
            user_feedback=invalid_feedback,
            research_findings=None,
            quality_check_results=None,
            cv_analysis_results=None,
            generated_key_qualifications=None,
            final_output_path=None,
            error_messages=[],
            node_execution_metadata={},
            workflow_status="PROCESSING",
            ui_display_data={},
            automated_mode=False
        )
        
        # Verify that error is returned for strict validation
        result = await graph.handle_feedback_node(state)
        assert "error_messages" in result
        assert len(result["error_messages"]) == 1
        assert "Handle feedback node failed: Feedback received with no item_id" in result["error_messages"][0]

    @pytest.mark.asyncio
    @patch.object(CVWorkflowGraph, '_build_graph')
    async def test_handle_feedback_node_without_feedback(self, mock_build_graph):
        """Test handle_feedback_node when no user feedback is present."""
        # Mock the graph building
        mock_workflow = MagicMock()
        mock_build_graph.return_value = mock_workflow
        mock_workflow.compile.return_value = MagicMock()
        
        session_id = str(uuid.uuid4())
        graph = CVWorkflowGraph(session_id=session_id)
        
        state = GlobalState(
            session_id="test-session",
            trace_id="test-trace",
            structured_cv=StructuredCV(),
            cv_text="Sample CV text",
            job_description_data=None,
            current_section_key=None,
            current_section_index=0,
            items_to_process_queue=[],
            current_item_id=None,
            current_content_type=None,
            is_initial_generation=True,
            content_generation_queue=[],
            user_feedback=None,
            research_findings=None,
            quality_check_results=None,
            cv_analysis_results=None,
            generated_key_qualifications=None,
            final_output_path=None,
            error_messages=[],
            node_execution_metadata={},
            workflow_status="PROCESSING",
            ui_display_data={},
            automated_mode=False
        )
        
        result = await graph.handle_feedback_node(state)
        
        # Verify that the node sets status to awaiting feedback
        assert result["workflow_status"] == "AWAITING_FEEDBACK"
        assert "requires_feedback" in result["ui_display_data"]
        assert result["ui_display_data"]["requires_feedback"] is True
        assert result.get("error_messages", []) == []  # No errors when no feedback present

    @pytest.mark.asyncio
    @patch.object(CVWorkflowGraph, '_build_graph')
    async def test_handle_feedback_node_raises_value_error_for_invalid_item_id(self, mock_build_graph):
        """Test that handle_feedback_node raises ValueError for invalid item_id."""
        # Mock the graph building
        mock_workflow = MagicMock()
        mock_build_graph.return_value = mock_workflow
        mock_workflow.compile.return_value = MagicMock()
        
        session_id = str(uuid.uuid4())
        graph = CVWorkflowGraph(session_id=session_id)
        
        # Create user feedback with whitespace-only item_id
        invalid_feedback = UserFeedback(
            action=UserAction.REGENERATE,
            item_id="   "  # Whitespace-only should trigger validation
        )
        
        state = GlobalState(
            session_id=session_id,
            trace_id=str(uuid.uuid4()),
            structured_cv=StructuredCV(),
            cv_text="Sample CV text",
            job_description_data=None,
            current_section_key=None,
            current_section_index=0,
            items_to_process_queue=[],
            current_item_id=None,
            current_content_type=None,
            is_initial_generation=True,
            content_generation_queue=[],
            user_feedback=invalid_feedback,
            research_findings=None,
            quality_check_results=None,
            cv_analysis_results=None,
            generated_key_qualifications=None,
            final_output_path=None,
            error_messages=[],
            node_execution_metadata={},
            workflow_status="RUNNING",
            ui_display_data={},
            automated_mode=False
        )
        
        # Verify that error is returned for strict validation
        result = await graph.handle_feedback_node(state)
        assert "error_messages" in result
        assert len(result["error_messages"]) == 1
        assert "Handle feedback node failed: Feedback received with no item_id" in result["error_messages"][0]

    @pytest.mark.asyncio
    @patch.object(CVWorkflowGraph, '_build_graph')
    async def test_supervisor_node_with_valid_current_item_id(self, mock_build_graph):
        """Test supervisor_node with valid current_item_id."""
        # Mock the graph building
        mock_workflow = MagicMock()
        mock_build_graph.return_value = mock_workflow
        mock_workflow.compile.return_value = MagicMock()
        
        session_id = str(uuid.uuid4())
        graph = CVWorkflowGraph(session_id=session_id)
        
        state = GlobalState(
            session_id=session_id,
            trace_id=str(uuid.uuid4()),
            structured_cv=StructuredCV(),
            cv_text="Sample CV text",
            job_description_data=None,
            current_section_key=None,
            current_section_index=0,
            items_to_process_queue=[],
            current_item_id="valid-item-123",
            current_content_type=None,
            is_initial_generation=True,
            content_generation_queue=[],
            user_feedback=None,
            research_findings=None,
            quality_check_results=None,
            cv_analysis_results=None,
            generated_key_qualifications=None,
            final_output_path=None,
            error_messages=[],
            node_execution_metadata={},
            workflow_status="RUNNING",
            ui_display_data={},
            automated_mode=False
        )
        
        result = await graph.supervisor_node(state)
        
        # Verify that the node processes valid state without errors
        assert result.get("error_messages", []) == []
        # supervisor_node doesn't always return workflow_status for valid states
        assert "next_node" in result["node_execution_metadata"]

    @pytest.mark.asyncio
    @patch.object(CVWorkflowGraph, '_build_graph')
    async def test_supervisor_node_with_missing_current_item_id(self, mock_build_graph):
        """Test supervisor_node validation when current_item_id is missing."""
        # Mock the graph building
        mock_workflow = MagicMock()
        mock_build_graph.return_value = mock_workflow
        mock_workflow.compile.return_value = MagicMock()
        
        session_id = str(uuid.uuid4())
        graph = CVWorkflowGraph(session_id=session_id)
        
        state = GlobalState(
            session_id=session_id,
            trace_id=str(uuid.uuid4()),
            structured_cv=StructuredCV(),
            cv_text="Sample CV text",
            job_description_data=None,
            current_section_key=None,
            current_section_index=0,
            items_to_process_queue=[],
            current_item_id=None,
            current_content_type=None,
            is_initial_generation=True,
            content_generation_queue=[],
            user_feedback=None,
            research_findings=None,
            quality_check_results=None,
            cv_analysis_results=None,
            generated_key_qualifications=None,
            final_output_path=None,
            error_messages=[],
            node_execution_metadata={},
            workflow_status="RUNNING",
            ui_display_data={},
            automated_mode=False
        )
        
        result = await graph.supervisor_node(state)
        
        # Verify that validation error was triggered
        assert "error_messages" in result
        assert len(result["error_messages"]) == 1
        assert "Supervisor node: Invalid or missing current_item_id" in result["error_messages"][0]
        assert result["workflow_status"] == "ERROR"
        assert result["node_execution_metadata"]["next_node"] == "error_handler"

    @pytest.mark.asyncio
    @patch.object(CVWorkflowGraph, '_build_graph')
    async def test_initialize_supervisor_node_returns_state_dict(self, mock_build_graph):
        """Test that initialize_supervisor_node returns state dict format."""
        # Mock the graph building
        mock_workflow = MagicMock()
        mock_build_graph.return_value = mock_workflow
        mock_workflow.compile.return_value = MagicMock()
        
        session_id = str(uuid.uuid4())
        graph = CVWorkflowGraph(session_id=session_id)
        
        # Create a structured CV with sections and items
        test_item = Item(
            content="Test content",
            item_type=ItemType.BULLET_POINT,
            status=ItemStatus.PENDING
        )
        test_section = Section(
            name="Key Qualifications",
            items=[test_item]
        )
        structured_cv = StructuredCV(sections=[test_section])
        
        state = GlobalState(
            session_id=session_id,
            trace_id=str(uuid.uuid4()),
            structured_cv=structured_cv,
            cv_text="Sample CV text",
            job_description_data=None,
            current_section_key=None,
            current_section_index=0,
            items_to_process_queue=[],
            current_item_id=None,
            current_content_type=None,
            is_initial_generation=True,
            content_generation_queue=[],
            user_feedback=None,
            research_findings=None,
            quality_check_results=None,
            cv_analysis_results=None,
            generated_key_qualifications=None,
            final_output_path=None,
            error_messages=[],
            node_execution_metadata={},
            workflow_status="RUNNING",
            ui_display_data={},
            automated_mode=False
        )
        
        result = await graph.initialize_supervisor_node(state)
        
        # Verify the result is a dict with supervisor state
        assert isinstance(result, dict)
        assert "current_section_index" in result
        assert "current_item_id" in result
        assert result["current_section_index"] == 0
        assert result["current_item_id"] == str(test_item.id)

    @pytest.mark.asyncio
    @patch.object(CVWorkflowGraph, '_build_graph')
    async def test_cv_analyzer_node_returns_analysis_results(self, mock_build_graph, mock_agents):
        """Test that cv_analyzer_node returns cv_analysis_results in dict format."""
        # Mock the graph building
        mock_workflow = MagicMock()
        mock_build_graph.return_value = mock_workflow
        mock_workflow.compile.return_value = MagicMock()
        
        session_id = str(uuid.uuid4())
        graph = CVWorkflowGraph(
            session_id=session_id,
            cv_analyzer_agent=mock_agents["cv_analyzer_agent"]
        )
        
        # Create a structured CV with sections and items
        test_item = Item(
            content="Test content",
            item_type=ItemType.BULLET_POINT,
            status=ItemStatus.PENDING
        )
        test_section = Section(
            name="Experience",
            items=[test_item]
        )
        structured_cv = StructuredCV(sections=[test_section])
        
        # Mock the cv_analyzer_agent to return a dict with analysis results
        mock_analysis_results = {"analysis": "mock analysis"}
        mock_agents["cv_analyzer_agent"].run_as_node.return_value = {
            "cv_analysis_results": mock_analysis_results
        }
        
        state = GlobalState(
            session_id=session_id,
            trace_id=str(uuid.uuid4()),
            structured_cv=structured_cv,
            cv_text="Sample CV text",
            job_description_data=None,
            current_section_key=None,
            current_section_index=0,
            items_to_process_queue=[],
            current_item_id=str(test_item.id),
            current_content_type=None,
            is_initial_generation=True,
            content_generation_queue=[],
            user_feedback=None,
            research_findings=None,
            quality_check_results=None,
            cv_analysis_results=None,
            generated_key_qualifications=None,
            final_output_path=None,
            error_messages=[],
            node_execution_metadata={},
            workflow_status="RUNNING",
            ui_display_data={},
            automated_mode=False
        )
        
        result = await graph.cv_analyzer_node(state)
        
        # Verify the result is a dict with cv_analysis_results
        assert isinstance(result, dict)
        assert "cv_analysis_results" in result
        assert result["cv_analysis_results"] == mock_analysis_results
        
        # Verify the agent was called
        mock_agents["cv_analyzer_agent"].run_as_node.assert_called_once_with(state)

    @pytest.mark.asyncio
    @patch.object(CVWorkflowGraph, '_build_graph')
    async def test_supervisor_node_with_empty_current_item_id(self, mock_build_graph):
        """Test supervisor_node validation when current_item_id is empty string."""
        # Mock the graph building
        mock_workflow = MagicMock()
        mock_build_graph.return_value = mock_workflow
        mock_workflow.compile.return_value = MagicMock()
        
        session_id = str(uuid.uuid4())
        graph = CVWorkflowGraph(session_id=session_id)
        
        state = GlobalState(
            session_id=session_id,
            trace_id=str(uuid.uuid4()),
            structured_cv=StructuredCV(),
            cv_text="Sample CV text",
            job_description_data=None,
            current_section_key=None,
            current_section_index=0,
            items_to_process_queue=[],
            current_item_id="",
            current_content_type=None,
            is_initial_generation=True,
            content_generation_queue=[],
            user_feedback=None,
            research_findings=None,
            quality_check_results=None,
            cv_analysis_results=None,
            generated_key_qualifications=None,
            final_output_path=None,
            error_messages=[],
            node_execution_metadata={},
            workflow_status="RUNNING",
            ui_display_data={},
            automated_mode=False
        )
        
        result = await graph.supervisor_node(state)
        
        # Verify that validation error was triggered
        assert "error_messages" in result
        assert len(result["error_messages"]) == 1
        assert "Supervisor node: Invalid or missing current_item_id" in result["error_messages"][0]
        assert result["workflow_status"] == "ERROR"
        assert result["node_execution_metadata"]["next_node"] == "error_handler"

    @pytest.mark.asyncio
    @patch.object(CVWorkflowGraph, '_build_graph')
    async def test_supervisor_node_with_whitespace_current_item_id(self, mock_build_graph):
        """Test supervisor_node validation when current_item_id is whitespace-only."""
        # Mock the graph building
        mock_workflow = MagicMock()
        mock_build_graph.return_value = mock_workflow
        mock_workflow.compile.return_value = MagicMock()
        
        session_id = str(uuid.uuid4())
        graph = CVWorkflowGraph(session_id=session_id)
        
        state = GlobalState(
            session_id=session_id,
            trace_id=str(uuid.uuid4()),
            structured_cv=StructuredCV(),
            cv_text="Sample CV text",
            job_description_data=None,
            current_section_key=None,
            current_section_index=0,
            items_to_process_queue=[],
            current_item_id="   \t\n  ",
            current_content_type=None,
            is_initial_generation=True,
            content_generation_queue=[],
            user_feedback=None,
            research_findings=None,
            quality_check_results=None,
            cv_analysis_results=None,
            generated_key_qualifications=None,
            final_output_path=None,
            error_messages=[],
            node_execution_metadata={},
            workflow_status="RUNNING",
            ui_display_data={},
            automated_mode=False
        )
        
        result = await graph.supervisor_node(state)
        
        # Verify that validation error was triggered
        assert "error_messages" in result
        assert len(result["error_messages"]) == 1
        assert "Supervisor node: Invalid or missing current_item_id" in result["error_messages"][0]
        assert result["workflow_status"] == "ERROR"
        assert result["node_execution_metadata"]["next_node"] == "error_handler"