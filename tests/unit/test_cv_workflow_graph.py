import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import uuid

from src.orchestration.cv_workflow_graph import CVWorkflowGraph
from src.agents.agent_base import AgentBase
from src.orchestration.state import AgentState
from src.models.cv_models import StructuredCV


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
            # Create a valid AgentState for the mock return value
            mock_state = AgentState(
                structured_cv=StructuredCV(),
                cv_text="Sample CV text"
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
        
        # Create a valid AgentState with required fields
        state = AgentState(
            structured_cv=StructuredCV(),
            cv_text="Sample CV text"
        )
        
        # Mock the run_as_node method to return the same state
        mock_agents["job_description_parser_agent"].run_as_node.return_value = state
        
        result = await graph.jd_parser_node(state)
        
        # Verify that the injected agent's run_as_node method was called
        mock_agents["job_description_parser_agent"].run_as_node.assert_called_once_with(state)
        assert result == state
    
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
        
        # Create a valid AgentState with required fields
        state = AgentState(
            structured_cv=StructuredCV(),
            cv_text="Sample CV text"
        )
        
        # Test that the node handles missing agent gracefully by adding error message
        result = await graph.jd_parser_node(state)
        
        # Verify that an error message was added to the state
        assert len(result.error_messages) == 1
        assert "JobDescriptionParserAgent failed: JobDescriptionParserAgent not injected" in result.error_messages[0]
    
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
        step1_state = AgentState(
            structured_cv=StructuredCV(),
            cv_text="Sample CV text",
            workflow_status="PROCESSING"
        )
        step2_state = AgentState(
            structured_cv=StructuredCV(),
            cv_text="Sample CV text", 
            workflow_status="COMPLETED"
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
        
        initial_state = AgentState(
            structured_cv=StructuredCV(),
            cv_text="Sample CV text",
            workflow_status="AWAITING_FEEDBACK"
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
        feedback_state = AgentState(
            structured_cv=StructuredCV(),
            cv_text="Sample CV text",
            workflow_status="AWAITING_FEEDBACK",
            ui_display_data={"section": "key_qualifications", "requires_feedback": True}
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
        
        initial_state = AgentState(
            structured_cv=StructuredCV(),
            cv_text="Sample CV text",
            workflow_status="PROCESSING"
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
        
        initial_state = AgentState(
            structured_cv=StructuredCV(),
            cv_text="Sample CV text",
            workflow_status="PROCESSING"
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
        
        state = AgentState(
            structured_cv=StructuredCV(),
            cv_text="Sample CV text",
            workflow_status="PROCESSING"
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