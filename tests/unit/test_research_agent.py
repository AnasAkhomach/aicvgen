#!/usr/bin/env python3
"""
Unit tests for ResearchAgent.

Tests the research functionality for gathering relevant information
to enhance CV content generation.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any, List

from src.agents.research_agent import ResearchAgent
from src.agents.agent_base import AgentExecutionContext, AgentResult
from src.models.data_models import StructuredCV, Section, Item, ItemStatus
from src.models.workflow_models import AgentState
from src.utils.llm_service import LLM


class TestResearchAgent:
    """Test cases for ResearchAgent."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM instance."""
        llm = Mock(spec=LLM)
        llm.generate_content = AsyncMock()
        return llm

    @pytest.fixture
    def research_agent(self, mock_llm):
        """Create a ResearchAgent instance for testing."""
        return ResearchAgent(
            name="TestResearcher",
            description="Test research agent",
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
    def sample_job_data(self):
        """Sample job data for testing."""
        return {
            "job_title": "Senior Software Engineer",
            "required_skills": ["Python", "React", "AWS", "Docker"],
            "experience_level": "5+ years",
            "company_info": "Tech startup focused on cloud solutions",
            "responsibilities": [
                "Design and develop scalable web applications",
                "Lead technical architecture decisions",
                "Mentor junior developers"
            ]
        }

    @pytest.fixture
    def sample_structured_cv(self):
        """Create a sample structured CV for testing."""
        cv = StructuredCV()
        cv.sections = [
            Section(
                name="Professional Experience",
                items=[
                    Item(content="Software Developer at ABC Corp", status=ItemStatus.INITIAL),
                    Item(content="3 years Python development", status=ItemStatus.INITIAL)
                ]
            ),
            Section(
                name="Skills",
                items=[
                    Item(content="Python, JavaScript", status=ItemStatus.INITIAL),
                    Item(content="Web development", status=ItemStatus.INITIAL)
                ]
            )
        ]
        return cv

    @pytest.fixture
    def sample_agent_state(self, sample_job_data, sample_structured_cv):
        """Create a sample agent state for testing."""
        return AgentState(
            job_description="Senior Software Engineer position",
            job_data=sample_job_data,
            structured_cv=sample_structured_cv,
            error_messages=[]
        )

    def test_research_agent_initialization(self, mock_llm):
        """Test ResearchAgent initialization."""
        agent = ResearchAgent(
            name="TestResearcher",
            description="Test description",
            llm=mock_llm
        )
        
        assert agent.name == "TestResearcher"
        assert agent.description == "Test description"
        assert agent.llm == mock_llm
        assert hasattr(agent, 'input_schema')
        assert hasattr(agent, 'output_schema')

    async def test_run_async_success(self, research_agent, execution_context):
        """Test successful research execution."""
        input_data = {
            "job_data": {
                "job_title": "Senior Software Engineer",
                "required_skills": ["Python", "React"]
            },
            "structured_cv": StructuredCV()
        }
        
        # Mock the research methods
        with patch.object(research_agent, '_research_industry_trends') as mock_trends, \
             patch.object(research_agent, '_research_skill_requirements') as mock_skills, \
             patch.object(research_agent, '_research_company_culture') as mock_culture:
            
            mock_trends.return_value = ["Cloud computing growth", "AI/ML adoption"]
            mock_skills.return_value = ["Python expertise", "React proficiency"]
            mock_culture.return_value = ["Innovation-focused", "Collaborative environment"]
            
            result = await research_agent.run_async(input_data, execution_context)
            
            assert result.success is True
            assert "industry_trends" in result.output_data
            assert "skill_insights" in result.output_data
            assert "company_culture" in result.output_data
            assert result.confidence_score > 0.0

    async def test_run_async_missing_job_data(self, research_agent, execution_context):
        """Test research execution with missing job data."""
        input_data = {
            "structured_cv": StructuredCV()
        }
        
        result = await research_agent.run_async(input_data, execution_context)
        
        assert result.success is False
        assert "Missing required job_data" in result.error_message
        assert result.confidence_score == 0.0

    async def test_run_async_missing_cv(self, research_agent, execution_context):
        """Test research execution with missing CV data."""
        input_data = {
            "job_data": {
                "job_title": "Engineer",
                "required_skills": ["Python"]
            }
        }
        
        result = await research_agent.run_async(input_data, execution_context)
        
        assert result.success is False
        assert "Missing required structured_cv" in result.error_message
        assert result.confidence_score == 0.0

    async def test_run_as_node_success(self, research_agent, sample_agent_state):
        """Test successful run_as_node execution."""
        # Mock the run_async method
        mock_result = AgentResult(
            success=True,
            output_data={
                "industry_trends": ["Cloud growth", "AI adoption"],
                "skill_insights": ["Python in demand", "React popular"],
                "company_culture": ["Innovation-focused"]
            },
            confidence_score=0.85
        )
        
        with patch.object(research_agent, 'run_async', return_value=mock_result):
            result_state = await research_agent.run_as_node(sample_agent_state)
            
            assert result_state.research_findings is not None
            assert "industry_trends" in result_state.research_findings
            assert "skill_insights" in result_state.research_findings
            assert "company_culture" in result_state.research_findings
            assert len(result_state.error_messages) == 0

    async def test_run_as_node_missing_job_data(self, research_agent):
        """Test run_as_node with missing job data."""
        state = AgentState(
            job_description="Test job",
            job_data=None,
            structured_cv=StructuredCV(),
            error_messages=[]
        )
        
        result_state = await research_agent.run_as_node(state)
        
        assert len(result_state.error_messages) == 1
        assert "Missing job_data" in result_state.error_messages[0]
        assert result_state.research_findings is None

    async def test_run_as_node_missing_cv(self, research_agent, sample_job_data):
        """Test run_as_node with missing CV data."""
        state = AgentState(
            job_description="Test job",
            job_data=sample_job_data,
            structured_cv=None,
            error_messages=[]
        )
        
        result_state = await research_agent.run_as_node(state)
        
        assert len(result_state.error_messages) == 1
        assert "Missing structured_cv" in result_state.error_messages[0]
        assert result_state.research_findings is None

    async def test_run_as_node_execution_error(self, research_agent, sample_agent_state):
        """Test run_as_node with execution error."""
        # Mock the run_async method to raise an exception
        with patch.object(research_agent, 'run_async') as mock_run:
            mock_run.side_effect = Exception("Research failed")
            
            result_state = await research_agent.run_as_node(sample_agent_state)
            
            assert len(result_state.error_messages) == 1
            assert "Error during research: Research failed" in result_state.error_messages[0]
            assert result_state.research_findings is None

    @patch('src.agents.research_agent.ResearchAgent.llm')
    async def test_research_industry_trends(self, mock_llm, research_agent):
        """Test industry trends research."""
        mock_llm.generate_content.return_value = """
        Current industry trends:
        1. Cloud computing adoption increasing
        2. AI/ML integration in software development
        3. Remote work becoming standard
        4. DevOps practices widespread
        """
        
        job_data = {
            "job_title": "Software Engineer",
            "required_skills": ["Python", "AWS"]
        }
        
        trends = await research_agent._research_industry_trends(job_data)
        
        assert isinstance(trends, list)
        assert len(trends) > 0
        assert any("cloud" in trend.lower() for trend in trends)
        mock_llm.generate_content.assert_called_once()

    @patch('src.agents.research_agent.ResearchAgent.llm')
    async def test_research_skill_requirements(self, mock_llm, research_agent):
        """Test skill requirements research."""
        mock_llm.generate_content.return_value = """
        Key skill insights:
        1. Python is highly demanded in backend development
        2. React skills are essential for frontend roles
        3. AWS knowledge is crucial for cloud positions
        4. Docker containerization is increasingly important
        """
        
        job_data = {
            "required_skills": ["Python", "React", "AWS"],
            "job_title": "Full Stack Developer"
        }
        
        insights = await research_agent._research_skill_requirements(job_data)
        
        assert isinstance(insights, list)
        assert len(insights) > 0
        assert any("python" in insight.lower() for insight in insights)
        mock_llm.generate_content.assert_called_once()

    @patch('src.agents.research_agent.ResearchAgent.llm')
    async def test_research_company_culture(self, mock_llm, research_agent):
        """Test company culture research."""
        mock_llm.generate_content.return_value = """
        Company culture insights:
        1. Innovation-driven environment
        2. Collaborative team structure
        3. Emphasis on continuous learning
        4. Work-life balance priority
        """
        
        job_data = {
            "company_info": "Tech startup focused on AI solutions",
            "job_title": "Senior Engineer"
        }
        
        culture = await research_agent._research_company_culture(job_data)
        
        assert isinstance(culture, list)
        assert len(culture) > 0
        assert any("innovation" in item.lower() for item in culture)
        mock_llm.generate_content.assert_called_once()

    async def test_research_llm_error_handling(self, research_agent):
        """Test error handling when LLM fails."""
        job_data = {"job_title": "Engineer"}
        
        # Mock LLM to raise an exception
        with patch.object(research_agent, 'llm') as mock_llm:
            mock_llm.generate_content.side_effect = Exception("LLM service error")
            
            trends = await research_agent._research_industry_trends(job_data)
            
            # Should return empty list on error
            assert trends == []

    def test_confidence_score_calculation(self, research_agent):
        """Test confidence score calculation."""
        # Test with comprehensive research data
        comprehensive_data = {
            "industry_trends": ["Trend 1", "Trend 2", "Trend 3"],
            "skill_insights": ["Insight 1", "Insight 2"],
            "company_culture": ["Culture 1", "Culture 2"]
        }
        
        confidence = research_agent.get_confidence_score(comprehensive_data)
        assert confidence > 0.7  # Should be high for comprehensive data
        
        # Test with minimal research data
        minimal_data = {
            "industry_trends": ["Trend 1"]
        }
        
        confidence = research_agent.get_confidence_score(minimal_data)
        assert confidence < 0.5  # Should be lower for minimal data

    @patch('src.agents.research_agent.logger')
    async def test_run_as_node_logs_execution(self, mock_logger, research_agent, sample_agent_state):
        """Test that run_as_node logs execution properly."""
        mock_result = AgentResult(
            success=True,
            output_data={"industry_trends": []},
            confidence_score=0.5
        )
        
        with patch.object(research_agent, 'run_async', return_value=mock_result):
            await research_agent.run_as_node(sample_agent_state)
            
            # Verify logging calls
            mock_logger.info.assert_called()
            assert any("Starting research" in str(call) for call in mock_logger.info.call_args_list)

    async def test_extract_research_insights(self, research_agent):
        """Test extraction of research insights from CV and job data."""
        cv = StructuredCV()
        cv.sections = [
            Section(
                name="Skills",
                items=[Item(content="Python, JavaScript, React", status=ItemStatus.INITIAL)]
            )
        ]
        
        job_data = {
            "required_skills": ["Python", "React", "AWS"],
            "job_title": "Full Stack Developer"
        }
        
        insights = research_agent._extract_research_insights(cv, job_data)
        
        assert "skill_gaps" in insights
        assert "skill_matches" in insights
        assert "AWS" in insights["skill_gaps"]  # Missing skill
        assert "Python" in insights["skill_matches"]  # Matching skill

    def test_parse_llm_response_list(self, research_agent):
        """Test parsing of LLM response into list format."""
        response = """
        Key insights:
        1. First insight here
        2. Second insight here
        3. Third insight here
        """
        
        parsed = research_agent._parse_llm_response_to_list(response)
        
        assert isinstance(parsed, list)
        assert len(parsed) == 3
        assert "First insight here" in parsed[0]
        assert "Second insight here" in parsed[1]
        assert "Third insight here" in parsed[2]

    def test_parse_llm_response_empty(self, research_agent):
        """Test parsing of empty LLM response."""
        parsed = research_agent._parse_llm_response_to_list("")
        assert parsed == []
        
        parsed = research_agent._parse_llm_response_to_list(None)
        assert parsed == []