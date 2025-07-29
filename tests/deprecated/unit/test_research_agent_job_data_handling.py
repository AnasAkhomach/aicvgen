"""Test for ResearchAgent fix to handle JobDescriptionData properly."""

from unittest.mock import Mock, patch

import pytest

from src.agents.research_agent import ResearchAgent
from src.models.agent_input_models import ResearchAgentInput
from src.models.agent_models import AgentResult
from src.models.cv_models import JobDescriptionData, MetadataModel, StructuredCV
from src.orchestration.state import AgentState


class TestResearchAgentFix:
    """Test cases for ResearchAgent fix."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_llm_service = Mock()
        self.mock_vector_store_service = Mock()
        self.mock_settings = Mock()
        self.mock_template_manager = Mock()
        # Configure template manager to have empty templates (fallback to default prompt)
        self.mock_template_manager.templates = {}
        self.research_agent = ResearchAgent(
            llm_service=self.mock_llm_service,
            vector_store_service=self.mock_vector_store_service,
            settings=self.mock_settings,
            template_manager=self.mock_template_manager,
            session_id="test-session",
        )

    def test_create_research_prompt_with_pydantic_model(self):
        """Test _create_research_prompt with proper JobDescriptionData model."""
        # Create test data
        job_desc_data = JobDescriptionData(
            raw_text="Test job description",
            job_title="Software Engineer",
            company_name="Test Company",
            main_job_description_raw="Test description",
            skills=["Python", "Django"],
            experience_level="Mid-level",
            responsibilities=["Develop software"],
            industry_terms=["Agile"],
            company_values=["Innovation"],
        )

        # Test the method
        prompt = self.research_agent._create_research_prompt(job_desc_data)

        # Verify the prompt contains expected information
        assert "Software Engineer" in prompt
        assert "Test Company" in prompt
        assert "Analyze the following job description" in prompt

    def test_create_research_prompt_with_dict_conversion(self):
        """Test _create_research_prompt with dict that gets converted to JobDescriptionData."""
        # Create test data as dict (simulating the bug scenario)
        job_desc_dict = {
            "raw_text": "Test job description",
            "job_title": "Data Scientist",
            "company_name": "AI Corp",
            "main_job_description_raw": "Test description",
            "skills": ["Python", "ML"],
            "experience_level": "Senior",
            "responsibilities": ["Build models"],
            "industry_terms": ["Machine Learning"],
            "company_values": ["Data-driven"],
        }

        # Test the method with dict input
        with patch("src.agents.research_agent.logger") as mock_logger:
            prompt = self.research_agent._create_research_prompt(job_desc_dict)

            # Verify warning was logged (may be called multiple times due to template manager)
            mock_logger.warning.assert_any_call(
                "Converted dict to JobDescriptionData model in ResearchAgent"
            )

        # Verify the prompt contains expected information
        assert "Data Scientist" in prompt
        assert "AI Corp" in prompt
        assert "Analyze the following job description" in prompt

    def test_create_research_prompt_with_invalid_dict(self):
        """Test _create_research_prompt with invalid dict that cannot be converted."""
        # Create invalid dict (missing required fields)
        invalid_dict = {
            "job_title": "Engineer",
            # Missing required fields like raw_text, main_job_description_raw, etc.
        }

        # Test the method with invalid dict input
        with pytest.raises(
            TypeError,
            match="Expected JobDescriptionData model, but received a dict that could not be validated",
        ):
            self.research_agent._create_research_prompt(invalid_dict)

    def test_create_research_prompt_with_invalid_type(self):
        """Test _create_research_prompt with completely invalid type."""
        # Test the method with invalid type
        with pytest.raises(TypeError, match="Expected JobDescriptionData model, got"):
            self.research_agent._create_research_prompt("invalid_string")

    @pytest.mark.asyncio
    async def test_run_as_node_integration(self):
        """Integration test to verify the agent works with AgentState containing dict."""
        # Create minimal StructuredCV for required field
        structured_cv = StructuredCV(sections=[], metadata={"version": "1.0"})

        # Create AgentState with dict job_description_data (simulating the bug)
        job_desc_dict = {
            "raw_text": "Test job description",
            "job_title": "Full Stack Developer",
            "company_name": "Tech Startup",
            "main_job_description_raw": "Test description",
            "skills": ["React", "Node.js"],
            "experience_level": "Mid-level",
            "responsibilities": ["Build web apps"],
            "industry_terms": ["SaaS"],
            "company_values": ["Innovation"],
        }

        state = AgentState(
            structured_cv=structured_cv,
            cv_text="Test CV content",
            job_description_data=job_desc_dict,  # This is a dict, not a Pydantic model
            error_messages=[],
        )

        # Mock the LLM service to return a simple response
        mock_response = Mock()
        mock_response.content = "Mock research findings"
        self.mock_llm_service.generate_content.return_value = mock_response

        # Run the agent
        with patch("src.agents.research_agent.logger"):
            result = await self.research_agent.run_as_node(state)

        # Verify the agent handled the dict input gracefully
        assert isinstance(result, AgentState)
        # The agent should not crash and should return a valid state
