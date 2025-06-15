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
from src.models.workflow_models import AgentState
from src.utils.llm_service import LLM


class TestParserAgent:
    """Test cases for ParserAgent."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM instance."""
        llm = Mock(spec=LLM)
        llm.generate_content = AsyncMock()
        return llm

    @pytest.fixture
    def parser_agent(self, mock_llm):
        """Create a ParserAgent instance for testing."""
        return ParserAgent(
            name="TestParser",
            description="Test parser agent",
            llm=mock_llm
        )

    @pytest.fixture
    def execution_context(self):
        """Create a test execution context."""
        return AgentExecutionContext(
            session_id="test_session_123",
            item_id="test_item",
            metadata={"test": "data"}
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
        return AgentState(
            job_description=sample_job_description,
            structured_cv=StructuredCV(),
            error_messages=[]
        )

    def test_parser_agent_initialization(self, mock_llm):
        """Test ParserAgent initialization."""
        agent = ParserAgent(
            name="TestParser",
            description="Test description",
            llm=mock_llm
        )
        
        assert agent.name == "TestParser"
        assert agent.description == "Test description"
        assert agent.llm == mock_llm
        assert hasattr(agent, 'input_schema')
        assert hasattr(agent, 'output_schema')

    @patch('src.agents.parser_agent.ParserAgent._parse_job_description_with_llm')
    async def test_parse_job_description_success(self, mock_parse_llm, parser_agent, sample_job_description):
        """Test successful job description parsing."""
        # Mock the LLM parsing result
        mock_parse_llm.return_value = {
            "job_title": "Senior Software Engineer",
            "required_skills": ["Python", "React", "AWS"],
            "experience_level": "5+ years",
            "education_requirements": ["Bachelor's degree in Computer Science"],
            "responsibilities": [
                "Design and develop scalable web applications",
                "Collaborate with cross-functional teams"
            ]
        }
        
        result = await parser_agent.parse_job_description(sample_job_description)
        
        assert result["job_title"] == "Senior Software Engineer"
        assert "Python" in result["required_skills"]
        assert "React" in result["required_skills"]
        assert "AWS" in result["required_skills"]
        assert result["experience_level"] == "5+ years"
        mock_parse_llm.assert_called_once_with(sample_job_description)

    @patch('src.agents.parser_agent.ParserAgent._parse_job_description_with_llm')
    async def test_parse_job_description_llm_error(self, mock_parse_llm, parser_agent, sample_job_description):
        """Test job description parsing with LLM error."""
        # Mock LLM error
        mock_parse_llm.side_effect = Exception("LLM service error")
        
        with pytest.raises(Exception, match="LLM service error"):
            await parser_agent.parse_job_description(sample_job_description)

    async def test_run_as_node_success(self, parser_agent, sample_agent_state):
        """Test successful run_as_node execution."""
        # Mock the parse_job_description method
        with patch.object(parser_agent, 'parse_job_description') as mock_parse:
            mock_parse.return_value = {
                "job_title": "Senior Software Engineer",
                "required_skills": ["Python", "React"],
                "experience_level": "5+ years"
            }
            
            result_state = await parser_agent.run_as_node(sample_agent_state)
            
            assert result_state.job_data is not None
            assert result_state.job_data["job_title"] == "Senior Software Engineer"
            assert "Python" in result_state.job_data["required_skills"]
            assert len(result_state.error_messages) == 0
            mock_parse.assert_called_once_with(sample_agent_state.job_description)

    async def test_run_as_node_missing_job_description(self, parser_agent):
        """Test run_as_node with missing job description."""
        state = AgentState(
            job_description=None,
            structured_cv=StructuredCV(),
            error_messages=[]
        )
        
        result_state = await parser_agent.run_as_node(state)
        
        assert len(result_state.error_messages) == 1
        assert "No job description provided" in result_state.error_messages[0]
        assert result_state.job_data is None

    async def test_run_as_node_empty_job_description(self, parser_agent):
        """Test run_as_node with empty job description."""
        state = AgentState(
            job_description="   ",
            structured_cv=StructuredCV(),
            error_messages=[]
        )
        
        result_state = await parser_agent.run_as_node(state)
        
        assert len(result_state.error_messages) == 1
        assert "No job description provided" in result_state.error_messages[0]
        assert result_state.job_data is None

    async def test_run_as_node_parsing_error(self, parser_agent, sample_agent_state):
        """Test run_as_node with parsing error."""
        # Mock the parse_job_description method to raise an exception
        with patch.object(parser_agent, 'parse_job_description') as mock_parse:
            mock_parse.side_effect = Exception("Parsing failed")
            
            result_state = await parser_agent.run_as_node(sample_agent_state)
            
            assert len(result_state.error_messages) == 1
            assert "Error parsing job description: Parsing failed" in result_state.error_messages[0]
            assert result_state.job_data is None

    @patch('src.agents.parser_agent.ParserAgent._extract_cv_sections')
    async def test_parse_cv_success(self, mock_extract_sections, parser_agent):
        """Test successful CV parsing."""
        # Mock CV content
        cv_content = "John Doe\nSoftware Engineer\nExperience: 5 years Python"
        
        # Mock the section extraction
        mock_extract_sections.return_value = [
            Section(
                name="Personal Information",
                items=[
                    Item(content="John Doe", status=ItemStatus.INITIAL)
                ]
            ),
            Section(
                name="Professional Experience",
                items=[
                    Item(content="5 years Python experience", status=ItemStatus.INITIAL)
                ]
            )
        ]
        
        result = await parser_agent.parse_cv(cv_content)
        
        assert isinstance(result, StructuredCV)
        assert len(result.sections) == 2
        assert result.sections[0].name == "Personal Information"
        assert result.sections[1].name == "Professional Experience"
        mock_extract_sections.assert_called_once_with(cv_content)

    async def test_parse_cv_empty_content(self, parser_agent):
        """Test CV parsing with empty content."""
        with pytest.raises(ValueError, match="CV content cannot be empty"):
            await parser_agent.parse_cv("")

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

    @patch('src.agents.parser_agent.logger')
    async def test_run_as_node_logs_execution(self, mock_logger, parser_agent, sample_agent_state):
        """Test that run_as_node logs execution properly."""
        with patch.object(parser_agent, 'parse_job_description') as mock_parse:
            mock_parse.return_value = {"job_title": "Test"}
            
            await parser_agent.run_as_node(sample_agent_state)
            
            # Verify logging calls
            mock_logger.info.assert_called()
            assert any("Starting job description parsing" in str(call) for call in mock_logger.info.call_args_list)

    def test_confidence_score_calculation(self, parser_agent):
        """Test confidence score calculation."""
        # Test with complete job data
        complete_data = {
            "job_title": "Senior Engineer",
            "required_skills": ["Python", "React"],
            "experience_level": "5+ years",
            "education_requirements": ["Bachelor's degree"]
        }
        
        confidence = parser_agent.get_confidence_score(complete_data)
        assert confidence > 0.8  # Should be high for complete data
        
        # Test with incomplete job data
        incomplete_data = {
            "job_title": "Engineer"
        }
        
        confidence = parser_agent.get_confidence_score(incomplete_data)
        assert confidence < 0.5  # Should be lower for incomplete data