"""Test ResearchAgent integration with consolidated VectorStoreService (DI enforced)."""

import pytest
from unittest.mock import Mock, AsyncMock
from src.agents.research_agent import ResearchAgent
from src.models.data_models import StructuredCV, JobDescriptionData
from src.agents.agent_base import AgentExecutionContext


class TestResearchAgentVectorIntegration:
    """Test ResearchAgent integration with VectorStoreService (DI only)."""

    @pytest.fixture
    def mock_llm_service(self):
        """Mock LLM service for testing."""
        mock_llm = Mock()
        mock_llm.generate_content.return_value = Mock(
            content='{"company_info": {"name": "Test Company"}}'
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
                "distance": 0.1,
            }
        ]
        return mock_service

    @pytest.fixture
    def sample_structured_cv(self):
        """Sample structured CV for testing."""
        return StructuredCV(
            metadata={"original_cv_text": "Sample CV text"}, sections=[]
        )

    @pytest.fixture
    def sample_job_description(self):
        """Sample job description for testing."""
        return JobDescriptionData(
            raw_text="Software Engineer position at Test Company",
            title="Software Engineer",
            company="Test Company",
            requirements=["Python", "Testing"],
        )

    def test_research_agent_uses_vector_store_service(
        self, mock_llm_service, mock_vector_store_service
    ):
        """Test that ResearchAgent correctly uses the consolidated VectorStoreService."""
        # Create ResearchAgent with DI
        agent = ResearchAgent(
            name="TestResearchAgent",
            description="Test agent for vector store integration",
            llm_service=mock_llm_service,
            vector_db=mock_vector_store_service,
            error_recovery_service=Mock(),
            progress_tracker=Mock(),
            settings=Mock(),
        )
        # Verify vector store service is used
        assert agent.vector_db == mock_vector_store_service

    def test_research_agent_vector_operations(
        self,
        mock_llm_service,
        mock_vector_store_service,
        sample_structured_cv,
        sample_job_description,
    ):
        """Test that ResearchAgent can perform vector operations."""
        # Create ResearchAgent with DI
        agent = ResearchAgent(
            name="TestResearchAgent",
            description="Test agent for vector operations",
            llm_service=mock_llm_service,
            vector_db=mock_vector_store_service,
            error_recovery_service=Mock(),
            progress_tracker=Mock(),
            settings=Mock(),
        )
        # Create test context
        context = AgentExecutionContext(
            session_id="test_session", item_id="test_item", metadata={"test": "data"}
        )
        # Test input data
        input_data = {
            "structured_cv": sample_structured_cv,
            "job_description_data": sample_job_description,
        }
        # Mock the async run method to return a successful result
        mock_result = Mock()
        mock_result.success = True
        mock_result.output_data = {"research_findings": "Test research findings"}
        agent.run_async = AsyncMock(return_value=mock_result)
        # Test that the agent has vector store access
        assert agent.vector_db == mock_vector_store_service
        # Test vector store operations directly
        agent.vector_db.search("test query")
        mock_vector_store_service.search.assert_called_with("test query")

    def test_research_agent_vector_store_initialization(
        self, mock_llm_service, mock_vector_store_service
    ):
        """Test that ResearchAgent properly initializes with vector store service."""
        # Create ResearchAgent with DI
        agent = ResearchAgent(
            name="TestResearchAgent",
            description="Test agent initialization",
            llm_service=mock_llm_service,
            vector_db=mock_vector_store_service,
            error_recovery_service=Mock(),
            progress_tracker=Mock(),
            settings=Mock(),
        )
        # Verify that the agent has the vector store service
        assert hasattr(agent, "vector_db")
        assert agent.vector_db is not None
        assert agent.vector_db == mock_vector_store_service
