"""Test to verify that the defensive programming code in ResearchAgent is reachable."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from src.agents.research_agent import ResearchAgent
from src.models.data_models import (
    JobDescriptionData,
    StructuredCV,
    PersonalInfo,
    Education,
)
from src.models.cv_models import Item, Section
from src.config.settings import AgentSettings
from src.templates.content_templates import ContentTemplateManager


class TestResearchAgentDefensiveProgramming:
    """Test class for ResearchAgent defensive programming."""

    @pytest.fixture
    def mock_llm_service(self):
        """Mock LLM service."""
        mock = AsyncMock()
        # Mock the generate_content method that returns an LLMResponse object
        mock_response = MagicMock()
        mock_response.content = "Mock research response"
        mock.generate_content.return_value = mock_response
        return mock

    @pytest.fixture
    def mock_vector_store_service(self):
        """Mock vector store service."""
        return MagicMock()

    @pytest.fixture
    def mock_settings(self):
        """Mock agent settings."""
        return AgentSettings(max_tokens=1000, temperature=0.7, timeout=30)

    @pytest.fixture
    def mock_template_manager(self):
        """Mock template manager."""
        return MagicMock(spec=ContentTemplateManager)

    @pytest.fixture
    def research_agent(
        self,
        mock_llm_service,
        mock_vector_store_service,
        mock_settings,
        mock_template_manager,
    ):
        """Create a ResearchAgent instance for testing."""
        return ResearchAgent(
            llm_service=mock_llm_service,
            vector_store_service=mock_vector_store_service,
            settings=mock_settings,
            template_manager=mock_template_manager,
            session_id="test-session",
        )

    @pytest.fixture
    def structured_cv(self):
        """Create a mock StructuredCV."""
        return StructuredCV(sections=[])

    def test_defensive_programming_with_dict(self, research_agent, caplog):
        """Test that defensive programming converts dict to JobDescriptionData."""
        # Create a dictionary instead of JobDescriptionData object
        job_desc_dict = {
            "raw_text": "Software Engineer position at Tech Corp",
            "job_title": "Software Engineer",
            "company_name": "Tech Corp",
            "skills": ["Python", "Django"],
            "responsibilities": ["Code development", "Testing"],
        }

        # Call the method with a dictionary
        result = research_agent._create_research_prompt(job_desc_dict)

        # Verify the result contains expected content
        assert "Software Engineer" in result
        assert "Tech Corp" in result

        # Verify that the warning was logged (defensive programming was triggered)
        assert (
            "Converted dict to JobDescriptionData model in ResearchAgent" in caplog.text
        )

    def test_defensive_programming_with_invalid_dict(self, research_agent):
        """Test that defensive programming raises TypeError for invalid dict."""
        # Create an invalid dictionary that cannot be converted to JobDescriptionData
        invalid_dict = {"invalid_field": "value"}

        # Expect TypeError to be raised
        with pytest.raises(
            TypeError,
            match="Expected JobDescriptionData model, but received a dict that could not be validated",
        ):
            research_agent._create_research_prompt(invalid_dict)

    def test_defensive_programming_with_wrong_type(self, research_agent):
        """Test that defensive programming raises TypeError for wrong type."""
        # Pass a string instead of JobDescriptionData or dict
        wrong_type = "not a dict or JobDescriptionData"

        # Expect TypeError to be raised
        with pytest.raises(TypeError, match="Expected JobDescriptionData model, got"):
            research_agent._create_research_prompt(wrong_type)

    def test_normal_operation_with_pydantic_model(self, research_agent):
        """Test normal operation with proper JobDescriptionData object."""
        # Create a proper JobDescriptionData object
        job_desc_data = JobDescriptionData(
            raw_text="Software Engineer position at Tech Corp",
            job_title="Software Engineer",
            company_name="Tech Corp",
            skills=["Python", "Django"],
            responsibilities=["Code development", "Testing"],
        )

        # Call the method with proper object
        result = research_agent._create_research_prompt(job_desc_data)

        # Verify the result contains expected content
        assert "Software Engineer" in result
        assert "Tech Corp" in result
