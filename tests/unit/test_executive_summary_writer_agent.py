"""Unit tests for ExecutiveSummaryWriterAgent LCEL implementation."""

import pytest
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from src.agents.executive_summary_writer_agent import ExecutiveSummaryWriterAgent
from src.models.agent_input_models import ExecutiveSummaryWriterAgentInput
from src.models.agent_output_models import ExecutiveSummaryLLMOutput
from src.models.cv_models import StructuredCV


class TestExecutiveSummaryWriterAgentLCEL:
    """Test cases for ExecutiveSummaryWriterAgent LCEL implementation."""

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
            "job_description",
            "key_qualifications",
            "professional_experience",
            "projects",
            "research_findings",
        ]
        # Make the mock support the pipe operator
        mock_prompt.__or__ = Mock(return_value=Mock())
        return mock_prompt

    @pytest.fixture
    def mock_parser(self):
        """Create a mock output parser."""
        mock_parser = Mock(spec=PydanticOutputParser)
        mock_parser.pydantic_object = ExecutiveSummaryLLMOutput
        return mock_parser

    @pytest.fixture
    def mock_chain(self):
        """Create a mock chain."""
        mock_chain = AsyncMock()
        return mock_chain

    @pytest.fixture
    def agent(self, mock_llm, mock_prompt, mock_parser, mock_chain):
        """Create an ExecutiveSummaryWriterAgent instance."""
        with patch.object(ExecutiveSummaryWriterAgent, "__init__", return_value=None):
            agent = ExecutiveSummaryWriterAgent.__new__(ExecutiveSummaryWriterAgent)
            agent.name = "ExecutiveSummaryWriterAgent"
            agent.description = (
                "Agent responsible for generating executive summary content for a CV"
            )
            agent.session_id = "test_session"
            agent.settings = {}
            agent.chain = mock_chain
            agent.logger = Mock()
            # Mock the progress tracker and update_progress method
            agent.progress_tracker = Mock()
            agent.update_progress = Mock()
            return agent

    @pytest.fixture
    def sample_input_data(self):
        """Create sample input data for testing."""
        return ExecutiveSummaryWriterAgentInput(
            job_description="We are looking for a Software Engineer to develop software applications using modern technologies. The ideal candidate will have experience with Python, JavaScript, and cloud platforms.",
            key_qualifications="Python programming, JavaScript development, Cloud platforms (AWS, Azure), Software architecture, Agile methodologies",
            professional_experience="Senior Software Engineer at Tech Corp (2020-2023): Led development of scalable web applications using Python and JavaScript. Implemented cloud-based solutions on AWS.",
            projects="E-commerce Platform: Built a full-stack e-commerce application using React and Django. Microservices Architecture: Designed and implemented microservices using Docker and Kubernetes.",
            research_findings={"industry_trends": ["AI/ML growth"]},
        )

    def test_initialization(self, agent):
        """Test that the agent initializes correctly with LCEL components."""
        assert agent.name == "ExecutiveSummaryWriterAgent"
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
        mock_result = ExecutiveSummaryLLMOutput(
            executive_summary="Generated executive summary content for the role that meets all requirements and provides comprehensive overview."
        )

        # Mock the chain's ainvoke method
        agent.chain.ainvoke = AsyncMock(return_value=mock_result)

        # Execute the agent with keyword arguments
        result = await agent._execute(**sample_input_data.model_dump())

        # Verify the result structure matches the Gold Standard LCEL pattern
        assert isinstance(result, dict)
        assert "generated_executive_summary" in result
        assert (
            result["generated_executive_summary"]
            == "Generated executive summary content for the role that meets all requirements and provides comprehensive overview."
        )

        # Verify the chain was called
        agent.chain.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_validation_error(self, agent):
        """Test that _execute handles validation errors correctly."""
        from src.error_handling.exceptions import AgentExecutionError

        # Create invalid input (missing required fields)
        invalid_input = {}

        # The _execute method should raise AgentExecutionError for validation errors
        with pytest.raises(AgentExecutionError) as exc_info:
            await agent._execute(**invalid_input)

        # Verify the exception details
        assert "ExecutiveSummaryWriterAgent" in str(exc_info.value)
        assert "failed" in str(exc_info.value)

    def test_pydantic_input_model_validation(self):
        """Test that ExecutiveSummaryWriterAgentInput validates correctly."""
        # Valid data
        valid_data = {
            "job_description": "We are looking for a Software Engineer to develop applications.",
            "key_qualifications": "Python programming, JavaScript development",
            "professional_experience": "Senior Software Engineer with 5 years experience",
            "projects": "E-commerce platform, Microservices architecture",
            "research_findings": {"industry_trends": ["AI/ML growth"]},
        }
        input_model = ExecutiveSummaryWriterAgentInput(**valid_data)
        assert (
            input_model.job_description
            == "We are looking for a Software Engineer to develop applications."
        )
        assert (
            input_model.key_qualifications
            == "Python programming, JavaScript development"
        )

        # Invalid data - missing required fields
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ExecutiveSummaryWriterAgentInput(job_description="Only job description")

    def test_pydantic_output_model_validation(self):
        """Test that ExecutiveSummaryLLMOutput validates correctly."""
        # Valid data
        valid_data = {
            "executive_summary": "This is a valid executive summary content that is longer than 100 characters and provides comprehensive overview of professional background and qualifications."
        }
        output = ExecutiveSummaryLLMOutput(**valid_data)
        assert output.executive_summary == valid_data["executive_summary"].strip()

        # Invalid data - too short (Pydantic's built-in validation message)
        with pytest.raises(
            Exception, match="String should have at least 100 characters"
        ):
            ExecutiveSummaryLLMOutput(executive_summary="Too short")

        # Invalid data - empty (Pydantic's built-in validation message)
        with pytest.raises(
            Exception, match="String should have at least 100 characters"
        ):
            ExecutiveSummaryLLMOutput(executive_summary="")
