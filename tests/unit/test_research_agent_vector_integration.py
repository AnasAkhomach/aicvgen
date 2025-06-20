"""Test ResearchAgent integration with consolidated VectorStoreService."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from src.agents.research_agent import ResearchAgent
from src.models.data_models import StructuredCV, JobDescriptionData
from src.agents.agent_base import AgentExecutionContext


class TestResearchAgentVectorIntegration:
    """Test ResearchAgent integration with VectorStoreService."""

    @pytest.fixture
    def mock_llm_service(self):
        """Mock LLM service for testing."""
        mock_llm = Mock()
        mock_llm.generate_content.return_value = Mock(
            content="{\"company_info\": {\"name\": \"Test Company\"}}"
        )
        return mock_llm

    @pytest.fixture
    def mock_vector_store_service(self):
        """Mock vector store service for testing."""
        mock_service = Mock()
        mock_service.add_item.return_value = "test_id_123"
        mock_service.search.return_value = [
            {
                "content": "Test CV content",
                "metadata": {"section": "experience"},
                "id": "test_id_123",
                "distance": 0.1
            }
        ]
        return mock_service

    @pytest.fixture
    def sample_structured_cv(self):
        """Sample structured CV for testing."""
        return StructuredCV(
            metadata={"original_cv_text": "Sample CV text"},
            sections=[]
        )

    @pytest.fixture
    def sample_job_description(self):
        """Sample job description for testing."""
        return JobDescriptionData(
            raw_text="Software Engineer position at Test Company",
            title="Software Engineer",
            company="Test Company",
            requirements=["Python", "Testing"]
        )

    @patch('src.agents.research_agent.get_vector_store_service')
    @patch('src.agents.research_agent.get_llm_service')
    def test_research_agent_uses_vector_store_service(
        self, mock_get_llm, mock_get_vector_store, mock_llm_service, mock_vector_store_service
    ):
        """Test that ResearchAgent correctly uses the consolidated VectorStoreService."""
        # Setup mocks
        mock_get_llm.return_value = mock_llm_service
        mock_get_vector_store.return_value = mock_vector_store_service

        # Create ResearchAgent
        agent = ResearchAgent(
            name="TestResearchAgent",
            description="Test agent for vector store integration"
        )

        # Verify vector store service is used
        assert agent.vector_db == mock_vector_store_service
        mock_get_vector_store.assert_called_once()

    @patch('src.agents.research_agent.get_vector_store_service')
    @patch('src.agents.research_agent.get_llm_service')
    def test_research_agent_vector_operations(
        self, mock_get_llm, mock_get_vector_store, 
        mock_llm_service, mock_vector_store_service,
        sample_structured_cv, sample_job_description
    ):
        """Test that ResearchAgent can perform vector operations."""
        # Setup mocks
        mock_get_llm.return_value = mock_llm_service
        mock_get_vector_store.return_value = mock_vector_store_service

        # Create ResearchAgent
        agent = ResearchAgent(
            name="TestResearchAgent",
            description="Test agent for vector operations"
        )

        # Create test context
        context = AgentExecutionContext(
            session_id="test_session",
            item_id="test_item",
            metadata={"test": "data"}
        )

        # Test input data
        input_data = {
            "structured_cv": sample_structured_cv,
            "job_description_data": sample_job_description
        }

        # Mock the async run method to return a successful result
        from unittest.mock import AsyncMock
        mock_result = Mock()
        mock_result.success = True
        mock_result.output_data = {"research_findings": "Test research findings"}
        agent.run_async = AsyncMock(return_value=mock_result)

        # Test that the agent has vector store access
        assert agent.vector_db == mock_vector_store_service
        
        # Test vector store operations directly
        agent.vector_db.search("test query")
        mock_vector_store_service.search.assert_called_with("test query")

    @patch('src.agents.research_agent.get_vector_store_service')
    @patch('src.agents.research_agent.get_llm_service')
    def test_research_agent_vector_store_initialization(
        self, mock_get_llm, mock_get_vector_store, mock_llm_service, mock_vector_store_service
    ):
        """Test that ResearchAgent properly initializes with vector store service."""
        # Setup mocks
        mock_get_llm.return_value = mock_llm_service
        mock_get_vector_store.return_value = mock_vector_store_service

        # Create ResearchAgent without passing vector_db parameter
        agent = ResearchAgent(
            name="TestResearchAgent",
            description="Test agent initialization"
        )

        # Verify that the agent has the vector store service
        assert hasattr(agent, 'vector_db')
        assert agent.vector_db is not None
        assert agent.vector_db == mock_vector_store_service

        # Verify get_vector_store_service was called during initialization
        mock_get_vector_store.assert_called_once()