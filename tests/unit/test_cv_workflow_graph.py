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