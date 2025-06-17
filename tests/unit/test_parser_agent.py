#!/usr/bin/env python3
"""
Unit tests for ParserAgent.

Tests the job description parsing and CV parsing functionality,
including the run_as_node method for LangGraph integration.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any

from src.agents.parser_agent import ParserAgent
from src.agents.agent_base import AgentExecutionContext, AgentResult
from src.models.data_models import StructuredCV, Section, Item, ItemStatus
from src.orchestration.state import AgentState
from src.services.llm_service import get_llm_service


class TestParserAgent:
    """Test cases for ParserAgent."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM instance."""
        llm = Mock()
        llm.generate_content = AsyncMock()
        return llm

    @pytest.fixture
    def parser_agent(self, mock_llm):
        """Create a ParserAgent instance for testing."""
        return ParserAgent(
            name="TestParser", description="Test parser agent", llm_service=mock_llm
        )

    @pytest.fixture
    def execution_context(self):
        """Create a test execution context."""
        return AgentExecutionContext(
            session_id="test_session_123",
            item_id="test_item",
            metadata={"test": "data"},
        )

    @pytest.fixture
    def sample_job_description(self):
        """Sample job description for testing."""
        return """
        Senior Software Engineer

        We are seeking a Senior Software Engineer with 5+ years of experience
        in Python, React, and AWS. The ideal candidate will have:

        - Strong programming skills in Python and JavaScript
        - Experience with React and modern web frameworks
        - Knowledge of AWS services and cloud architecture
        - Bachelor's degree in Computer Science or related field
        - Excellent communication and teamwork skills

        Responsibilities:
        - Design and develop scalable web applications
        - Collaborate with cross-functional teams
        - Mentor junior developers
        """

    @pytest.fixture
    def sample_agent_state(self, sample_job_description):
        """Create a sample agent state for testing."""
        from src.models.data_models import JobDescriptionData

        return AgentState(
            job_description_data=JobDescriptionData(raw_text=sample_job_description),
            structured_cv=StructuredCV(),
        )

    def test_parser_agent_initialization(self, mock_llm):
        """Test ParserAgent initialization."""
        agent = ParserAgent(
            name="TestParser", description="Test description", llm_service=mock_llm
        )

        assert agent.name == "TestParser"
        assert agent.description == "Test description"
        assert agent.llm == mock_llm
        assert hasattr(agent, "input_schema")
        assert hasattr(agent, "output_schema")

    @pytest.mark.asyncio
    @patch("src.agents.parser_agent.ParserAgent._parse_job_description_with_llm")
    async def test_parse_job_description_success(
        self, mock_parse_llm, parser_agent, sample_job_description
    ):
        """Test successful job description parsing."""
        # Mock the LLM parsing result
        mock_parse_llm.return_value = {
            "job_title": "Senior Software Engineer",
            "required_skills": ["Python", "React", "AWS"],
            "experience_level": "5+ years",
            "education_requirements": ["Bachelor's degree in Computer Science"],
            "responsibilities": [
                "Design and develop scalable web applications",
                "Collaborate with cross-functional teams",
            ],
        }

        result = await parser_agent.parse_job_description(sample_job_description)

        assert result["job_title"] == "Senior Software Engineer"
        assert "Python" in result["required_skills"]
        assert "React" in result["required_skills"]
        assert "AWS" in result["required_skills"]
        assert result["experience_level"] == "5+ years"
        mock_parse_llm.assert_called_once_with(sample_job_description)

    @pytest.mark.asyncio
    @patch("src.agents.parser_agent.ParserAgent._parse_job_description_with_llm")
    async def test_parse_job_description_llm_error(
        self, mock_parse_llm, parser_agent, sample_job_description
    ):
        """Test job description parsing with LLM error."""
        # Mock LLM error
        mock_parse_llm.side_effect = Exception("LLM service error")

        with pytest.raises(Exception, match="LLM service error"):
            await parser_agent.parse_job_description(sample_job_description)

    @pytest.mark.asyncio
    async def test_run_as_node_success(self, parser_agent, sample_agent_state):
        """Test successful run_as_node execution."""
        # Mock the parse_job_description method
        with patch.object(parser_agent, "parse_job_description") as mock_parse:
            mock_parse.return_value = {
                "job_title": "Senior Software Engineer",
                "required_skills": ["Python", "React"],
                "experience_level": "5+ years",
            }

            result_state = await parser_agent.run_as_node(sample_agent_state)

            assert "job_description_data" in result_state
            assert (
                result_state["job_description_data"]["job_title"]
                == "Senior Software Engineer"
            )
            assert "Python" in result_state["job_description_data"]["required_skills"]
            assert (
                "error_messages" not in result_state
                or len(result_state["error_messages"]) == 0
            )
            mock_parse.assert_called_once_with(
                sample_agent_state.job_description_data.raw_text
            )

    @pytest.mark.asyncio
    async def test_run_as_node_missing_job_description(self, parser_agent):
        """Test run_as_node with missing job description."""
        from src.models.data_models import JobDescriptionData

        state = AgentState(job_description_data=None, structured_cv=StructuredCV())

        result_state = await parser_agent.run_as_node(state)

        # The method should create an empty JobDescriptionData when none exists
        assert "job_description_data" in result_state
        assert isinstance(result_state["job_description_data"], JobDescriptionData)
        assert result_state["job_description_data"].raw_text == ""

    @pytest.mark.asyncio
    async def test_run_as_node_empty_job_description(self, parser_agent):
        """Test run_as_node with empty job description."""
        from src.models.data_models import JobDescriptionData

        state = AgentState(
            job_description_data=JobDescriptionData(raw_text="   "),
            structured_cv=StructuredCV(),
        )

        result_state = await parser_agent.run_as_node(state)

        # The method should handle empty text gracefully
        assert (
            "job_description_data" in result_state or "error_messages" in result_state
        )

    @pytest.mark.asyncio
    async def test_run_as_node_parsing_error(self, parser_agent, sample_agent_state):
        """Test run_as_node with parsing error."""
        # Mock the parse_job_description method to raise an exception
        with patch.object(parser_agent, "parse_job_description") as mock_parse:
            mock_parse.side_effect = Exception("Parsing failed")

            result_state = await parser_agent.run_as_node(sample_agent_state)

            assert "error_messages" in result_state
            assert len(result_state["error_messages"]) >= 1
            assert (
                "ParserAgent Error: Parsing failed" in result_state["error_messages"][0]
            )

    @pytest.mark.asyncio
    @patch("src.agents.parser_agent.ParserAgent._extract_sections")
    async def test_parse_cv_success(self, mock_extract_sections, parser_agent):
        """Test successful CV parsing."""
        # Mock CV content
        cv_content = "John Doe\nSoftware Engineer\nExperience: 5 years Python"

        # Mock the section extraction
        mock_extract_sections.return_value = [
            Section(
                name="Personal Information",
                items=[Item(content="John Doe", status=ItemStatus.INITIAL)],
            ),
            Section(
                name="Professional Experience",
                items=[
                    Item(content="5 years Python experience", status=ItemStatus.INITIAL)
                ],
            ),
        ]

        result = await parser_agent.parse_cv(cv_content)

        assert isinstance(result, StructuredCV)
        assert len(result.sections) == 2
        assert result.sections[0].name == "Personal Information"
        assert result.sections[1].name == "Professional Experience"
        mock_extract_sections.assert_called_once_with(cv_content)

    @pytest.mark.asyncio
    async def test_parse_cv_empty_content(self, parser_agent):
        """Test CV parsing with empty content."""
        with pytest.raises(ValueError, match="CV content cannot be empty"):
            await parser_agent.parse_cv("")

    @pytest.mark.asyncio
    async def test_parse_cv_none_content(self, parser_agent):
        """Test CV parsing with None content."""
        with pytest.raises(ValueError, match="CV content cannot be empty"):
            await parser_agent.parse_cv(None)

    def test_extract_skills_from_text(self, parser_agent):
        """Test skill extraction from text."""
        text = "Experienced in Python, JavaScript, React, AWS, and Docker"

        skills = parser_agent._extract_skills_from_text(text)

        assert "Python" in skills
        assert "JavaScript" in skills
        assert "React" in skills
        assert "AWS" in skills
        assert "Docker" in skills

    def test_extract_skills_from_empty_text(self, parser_agent):
        """Test skill extraction from empty text."""
        skills = parser_agent._extract_skills_from_text("")
        assert skills == []

    def test_extract_skills_from_none_text(self, parser_agent):
        """Test skill extraction from None text."""
        skills = parser_agent._extract_skills_from_text(None)
        assert skills == []

    @patch("src.agents.parser_agent.logger")
    @pytest.mark.asyncio
    async def test_run_as_node_logs_execution(
        self, mock_logger, parser_agent, sample_agent_state
    ):
        """Test that run_as_node logs execution properly."""
        with patch.object(parser_agent, "parse_job_description") as mock_parse:
            mock_parse.return_value = {"job_title": "Test"}

            await parser_agent.run_as_node(sample_agent_state)

            # Verify logging calls
            mock_logger.info.assert_called()
            assert any(
                "Starting job description parsing" in str(call)
                for call in mock_logger.info.call_args_list
            )

    @pytest.mark.asyncio
    async def test_run_as_node_basic(self, parser_agent):
        """Test basic run_as_node functionality with consolidated logic."""
        # Create a mock state
        from src.orchestration.state import AgentState
        from src.models.data_models import JobDescriptionData, StructuredCV

        job_data = JobDescriptionData(raw_text="Test job description")
        state = AgentState(job_description_data=job_data, structured_cv=StructuredCV())

        # Mock the parse_job_description method
        with patch.object(parser_agent, "parse_job_description") as mock_parse:
            mock_parse.return_value = job_data

            result = await parser_agent.run_as_node(state)

            # Verify the result structure
            assert "structured_cv" in result
            assert "job_description_data" in result
            mock_parse.assert_called_once_with("Test job description")

            # Verify job data is returned as JobDescriptionData object
            job_result = result["job_description_data"]
            assert isinstance(job_result, JobDescriptionData)
            assert hasattr(job_result, "raw_text")
            assert hasattr(job_result, "skills")
            assert hasattr(job_result, "experience_level")

    @pytest.mark.asyncio
    async def test_run_as_node_with_cv_text(self, parser_agent):
        """Test run_as_node with CV text processing."""
        from src.orchestration.state import AgentState
        from src.models.data_models import JobDescriptionData, StructuredCV

        job_data = JobDescriptionData(raw_text="Test job description")
        # Create structured_cv with metadata containing original_cv_text
        structured_cv = StructuredCV(
            metadata={"original_cv_text": "Sample CV text content"}
        )
        state = AgentState(job_description_data=job_data, structured_cv=structured_cv)

        with patch.object(
            parser_agent, "parse_job_description"
        ) as mock_parse_job, patch.object(
            parser_agent, "parse_cv_text"
        ) as mock_parse_cv:

            mock_parse_job.return_value = job_data
            mock_structured_cv = StructuredCV()
            mock_parse_cv.return_value = mock_structured_cv

            result = await parser_agent.run_as_node(state)

            # Verify both parsing methods were called
            mock_parse_job.assert_called_once_with("Test job description")
            mock_parse_cv.assert_called_once_with(
                "Sample CV text content", result["job_description_data"]
            )

            # Verify CV was processed
            assert "structured_cv" in result
            assert result["structured_cv"] == mock_structured_cv

    @pytest.mark.asyncio
    async def test_run_as_node_start_from_scratch(self, parser_agent):
        """Test run_as_node with start from scratch scenario."""
        from src.orchestration.state import AgentState
        from src.models.data_models import JobDescriptionData, StructuredCV

        job_data = JobDescriptionData(raw_text="Test job description")
        # Create structured_cv with metadata indicating start_from_scratch
        structured_cv = StructuredCV(metadata={"start_from_scratch": True})
        state = AgentState(job_description_data=job_data, structured_cv=structured_cv)

        with patch.object(
            parser_agent, "parse_job_description"
        ) as mock_parse_job, patch.object(
            parser_agent, "create_empty_cv_structure"
        ) as mock_create_cv:

            mock_parse_job.return_value = job_data
            mock_empty_cv = StructuredCV()
            mock_create_cv.return_value = mock_empty_cv

            result = await parser_agent.run_as_node(state)

            # Verify empty CV structure was created
            mock_create_cv.assert_called_once_with(result["job_description_data"])
            assert result["structured_cv"] == mock_empty_cv

    @pytest.mark.asyncio
    async def test_run_as_node_error_handling(self, parser_agent):
        """Test run_as_node error handling."""
        from src.orchestration.state import AgentState
        from src.models.data_models import JobDescriptionData, StructuredCV

        job_data = JobDescriptionData(raw_text="Test job description")
        state = AgentState(job_description_data=job_data, structured_cv=StructuredCV())

        with patch.object(parser_agent, "parse_job_description") as mock_parse:
            mock_parse.side_effect = Exception("Test error")

            result = await parser_agent.run_as_node(state)

            # Verify error handling
            assert "error_messages" in result
            assert any(
                "ParserAgent Error: Test error" in msg
                for msg in result["error_messages"]
            )

    def test_confidence_score_calculation(self, parser_agent):
        """Test confidence score calculation."""
        # Test with complete job data
        complete_data = {
            "job_title": "Senior Engineer",
            "required_skills": ["Python", "React"],
            "experience_level": "5+ years",
            "education_requirements": ["Bachelor's degree"],
        }

        confidence = parser_agent.get_confidence_score(complete_data)
        assert confidence > 0.8  # Should be high for complete data

        # Test with incomplete job data
        incomplete_data = {"job_title": "Engineer"}

        confidence = parser_agent.get_confidence_score(incomplete_data)
        assert confidence < 0.5  # Should be lower for incomplete data
