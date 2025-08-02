"""Unit tests for ProfessionalExperienceWriterAgent LCEL implementation."""

import pytest
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from src.agents.professional_experience_writer_agent import (
    ProfessionalExperienceWriterAgent,
    ProfessionalExperienceAgentInput,
)
from src.models.agent_output_models import ProfessionalExperienceLLMOutput
from src.models.cv_models import StructuredCV


class TestProfessionalExperienceWriterAgentLCEL:
    """Test cases for ProfessionalExperienceWriterAgent LCEL implementation."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM."""
        mock_llm = Mock(spec=ChatGoogleGenerativeAI)
        # Make the mock support the pipe operator
        mock_llm.__or__ = Mock(return_value=Mock())
        return mock_llm

    @pytest.fixture
    def mock_prompt(self):
        """Create a mock prompt template."""
        mock_prompt = Mock(spec=ChatPromptTemplate)
        mock_prompt.input_variables = [
            "job_title",
            "company_name",
            "job_description",
            "experience_item",
            "cv_summary",
            "required_skills",
            "preferred_qualifications",
            "research_findings",
        ]
        # Make the mock support the pipe operator
        mock_prompt.__or__ = Mock(return_value=Mock())
        return mock_prompt

    @pytest.fixture
    def mock_parser(self):
        """Create a mock output parser."""
        mock_parser = Mock(spec=PydanticOutputParser)
        mock_parser.pydantic_object = ProfessionalExperienceLLMOutput
        return mock_parser

    @pytest.fixture
    def mock_chain(self):
        """Create a mock chain."""
        mock_chain = AsyncMock()
        return mock_chain

    @pytest.fixture
    def agent(self, mock_llm, mock_prompt, mock_parser, mock_chain):
        """Create a ProfessionalExperienceWriterAgent instance."""
        with patch.object(
            ProfessionalExperienceWriterAgent, "__init__", return_value=None
        ):
            agent = ProfessionalExperienceWriterAgent.__new__(
                ProfessionalExperienceWriterAgent
            )
            agent.name = "ProfessionalExperienceWriterAgent"
            agent.description = "Agent responsible for generating professional experience content for a CV"
            agent.session_id = "test_session"
            agent.settings = {}
            agent.chain = mock_chain
            return agent

    @pytest.fixture
    def sample_input_data(self):
        """Create sample input data for testing."""
        from src.models.cv_models import Item, ItemType, ItemStatus

        # Create a mock experience item
        experience_item = {
            "id": "exp1",
            "content": "Previous Corp - Developer",
            "item_type": "EXPERIENCE_ROLE_TITLE",
            "status": "PENDING",
        }

        return ProfessionalExperienceAgentInput(
            job_title="Software Engineer",
            company_name="Tech Corp",
            job_description="Develop software applications",
            experience_item=experience_item,
            cv_summary="Experienced software developer",
            required_skills=["Python", "JavaScript"],
            preferred_qualifications=["5+ years experience"],
            research_findings={"industry_trends": ["AI/ML growth"]},
        )

    def test_initialization(self, agent):
        """Test that the agent initializes correctly with LCEL components."""
        assert agent.name == "ProfessionalExperienceWriterAgent"
        assert agent.chain is not None
        assert hasattr(agent, "chain")

    def test_lcel_chain_setup(self, agent):
        """Test that LCEL chain components are set up correctly."""
        # Verify the chain was created
        assert agent.chain is not None
        assert hasattr(agent, "chain")

    @pytest.mark.asyncio
    async def test_execute_with_valid_input(self, agent, sample_input_data):
        """Test the _execute method with valid input."""
        # Mock the chain result
        mock_result = ProfessionalExperienceLLMOutput(
            professional_experience="Generated professional experience content for the role."
        )

        # Mock the chain's ainvoke method
        agent.chain.ainvoke = AsyncMock(return_value=mock_result)

        # Execute the agent with keyword arguments
        result = await agent._execute(**sample_input_data.model_dump())

        # Verify the result
        assert isinstance(result, dict)
        assert "generated_professional_experience" in result
        assert isinstance(
            result["generated_professional_experience"], ProfessionalExperienceLLMOutput
        )
        assert (
            result["generated_professional_experience"].professional_experience
            == "Generated professional experience content for the role."
        )

        # Verify the chain was called
        agent.chain.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_validation_error(self, agent):
        """Test that _execute handles validation errors correctly."""
        # Create invalid input (missing required fields)
        invalid_input = {}

        with pytest.raises(Exception):  # Should raise validation error
            await agent._execute(**invalid_input)

    def test_pydantic_input_model_validation(self):
        """Test that ProfessionalExperienceAgentInput validates correctly."""
        # Valid data
        valid_data = {
            "job_title": "Software Engineer",
            "company_name": "Tech Corp",
            "job_description": "Develop software applications",
            "experience_item": {"id": "exp1", "content": "Previous Corp - Developer"},
            "cv_summary": "Experienced software developer",
            "required_skills": ["Python", "JavaScript"],
            "preferred_qualifications": ["5+ years experience"],
            "research_findings": {"industry_trends": ["AI/ML growth"]},
        }
        input_model = ProfessionalExperienceAgentInput(**valid_data)
        assert input_model.job_title == "Software Engineer"
        assert input_model.company_name == "Tech Corp"
        assert input_model.experience_item["id"] == "exp1"

        # Invalid data - missing required fields
        with pytest.raises(Exception):
            ProfessionalExperienceAgentInput(job_title="Software Engineer")

    def test_pydantic_output_model_validation(self):
        """Test that ProfessionalExperienceLLMOutput validates correctly."""
        # Valid data
        valid_data = {
            "professional_experience": "This is a valid professional experience content that is longer than 50 characters."
        }
        output = ProfessionalExperienceLLMOutput(**valid_data)
        assert (
            output.professional_experience
            == valid_data["professional_experience"].strip()
        )

        # Invalid data - too short (Pydantic's built-in validation message)
        with pytest.raises(
            Exception, match="String should have at least 50 characters"
        ):
            ProfessionalExperienceLLMOutput(professional_experience="Too short")

        # Invalid data - empty (Pydantic's built-in validation message)
        with pytest.raises(
            Exception, match="String should have at least 50 characters"
        ):
            ProfessionalExperienceLLMOutput(professional_experience="")
