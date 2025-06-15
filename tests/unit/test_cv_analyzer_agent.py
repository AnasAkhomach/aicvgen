#!/usr/bin/env python3
"""
Unit tests for CVAnalyzerAgent.

Tests the CV analysis functionality including content analysis,
optimization recommendations, and LangGraph integration.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any, List

from src.agents.cv_analyzer_agent import CVAnalyzerAgent
from src.agents.agent_base import AgentExecutionContext, AgentResult
from src.models.data_models import StructuredCV, Section, Item, ItemStatus
from src.models.workflow_models import AgentState
from src.services.llm_service import LLMService


class TestCVAnalyzerAgent:
    """Test cases for CVAnalyzerAgent."""

    @pytest.fixture
    def mock_llm_service(self):
        """Create a mock LLM service."""
        mock_service = Mock(spec=LLMService)
        mock_service.generate_async = AsyncMock()
        return mock_service

    @pytest.fixture
    def cv_analyzer_agent(self, mock_llm_service):
        """Create a CVAnalyzerAgent instance for testing."""
        return CVAnalyzerAgent(
            name="TestCVAnalyzer",
            description="Test CV analyzer agent",
            llm_service=mock_llm_service
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
    def sample_structured_cv(self):
        """Create a sample structured CV for testing."""
        cv = StructuredCV()
        cv.sections = [
            Section(
                name="Personal Information",
                items=[
                    Item(content="John Doe", status=ItemStatus.GENERATED),
                    Item(content="john.doe@email.com", status=ItemStatus.GENERATED),
                    Item(content="+1-555-0123", status=ItemStatus.GENERATED)
                ]
            ),
            Section(
                name="Professional Experience",
                items=[
                    Item(content="Software Engineer at TechCorp (2020-2024)", status=ItemStatus.GENERATED),
                    Item(content="Developed web applications using Python and Django", status=ItemStatus.GENERATED),
                    Item(content="Led team of 3 developers", status=ItemStatus.GENERATED)
                ]
            ),
            Section(
                name="Technical Skills",
                items=[
                    Item(content="Python, JavaScript, React, Django", status=ItemStatus.GENERATED),
                    Item(content="AWS, Docker, Kubernetes", status=ItemStatus.GENERATED)
                ]
            ),
            Section(
                name="Education",
                items=[
                    Item(content="BS Computer Science, MIT (2016-2020)", status=ItemStatus.GENERATED)
                ]
            )
        ]
        return cv

    @pytest.fixture
    def sample_job_data(self):
        """Create sample job data for testing."""
        return {
            "title": "Senior Software Engineer",
            "company": "TechCorp",
            "requirements": [
                "5+ years Python experience",
                "Experience with web frameworks",
                "Leadership experience",
                "Cloud platform experience"
            ],
            "responsibilities": [
                "Lead development team",
                "Design scalable systems",
                "Mentor junior developers",
                "Implement best practices"
            ],
            "preferred_skills": [
                "Python", "Django", "React", "AWS", "Docker", "Kubernetes", "PostgreSQL"
            ]
        }

    @pytest.fixture
    def sample_agent_state(self, sample_structured_cv, sample_job_data):
        """Create a sample agent state for testing."""
        return AgentState(
            job_description="Senior Software Engineer position",
            job_data=sample_job_data,
            structured_cv=sample_structured_cv,
            error_messages=[]
        )

    def test_cv_analyzer_initialization(self, mock_llm_service):
        """Test CVAnalyzerAgent initialization."""
        agent = CVAnalyzerAgent(
            name="TestCVAnalyzer",
            description="Test description",
            llm_service=mock_llm_service
        )
        
        assert agent.name == "TestCVAnalyzer"
        assert agent.description == "Test description"
        assert agent.llm_service == mock_llm_service
        assert hasattr(agent, 'input_schema')
        assert hasattr(agent, 'output_schema')

    def test_cv_analyzer_initialization_without_llm(self):
        """Test CVAnalyzerAgent initialization without LLM service."""
        agent = CVAnalyzerAgent(
            name="TestCVAnalyzer",
            description="Test description"
        )
        
        assert agent.name == "TestCVAnalyzer"
        assert agent.llm_service is not None  # Should create default LLM service

    async def test_run_async_success(self, cv_analyzer_agent, sample_structured_cv, sample_job_data, execution_context):
        """Test successful run_async execution."""
        content_data = {
            "structured_cv": sample_structured_cv,
            "job_data": sample_job_data
        }
        
        # Mock the analysis methods
        with patch.object(cv_analyzer_agent, '_analyze_content_relevance') as mock_relevance, \
             patch.object(cv_analyzer_agent, '_analyze_skill_alignment') as mock_skills, \
             patch.object(cv_analyzer_agent, '_analyze_experience_match') as mock_experience, \
             patch.object(cv_analyzer_agent, '_generate_recommendations') as mock_recommendations:
            
            mock_relevance.return_value = {"score": 0.85, "details": "Good content relevance"}
            mock_skills.return_value = {"score": 0.78, "missing_skills": ["PostgreSQL"]}
            mock_experience.return_value = {"score": 0.82, "gaps": []}
            mock_recommendations.return_value = ["Add PostgreSQL to skills", "Quantify achievements"]
            
            result = await cv_analyzer_agent.run_async(content_data, execution_context)
            
            assert isinstance(result, AgentResult)
            assert "analysis_results" in result.output
            assert "overall_score" in result.output["analysis_results"]
            assert "recommendations" in result.output["analysis_results"]
            assert result.confidence_score > 0
            
            mock_relevance.assert_called_once()
            mock_skills.assert_called_once()
            mock_experience.assert_called_once()
            mock_recommendations.assert_called_once()

    async def test_run_async_missing_structured_cv(self, cv_analyzer_agent, execution_context):
        """Test run_async with missing structured CV."""
        content_data = {
            "job_data": {"title": "Engineer"}
        }
        
        with pytest.raises(ValueError, match="Missing required structured_cv"):
            await cv_analyzer_agent.run_async(content_data, execution_context)

    async def test_run_async_missing_job_data(self, cv_analyzer_agent, sample_structured_cv, execution_context):
        """Test run_async with missing job data."""
        content_data = {
            "structured_cv": sample_structured_cv
        }
        
        with pytest.raises(ValueError, match="Missing required job_data"):
            await cv_analyzer_agent.run_async(content_data, execution_context)

    async def test_run_as_node_success(self, cv_analyzer_agent, sample_agent_state):
        """Test successful run_as_node execution."""
        # Mock the run_async method
        with patch.object(cv_analyzer_agent, 'run_async') as mock_run_async:
            mock_result = AgentResult(
                output={
                    "analysis_results": {
                        "overall_score": 0.82,
                        "content_relevance": {"score": 0.85},
                        "skill_alignment": {"score": 0.78},
                        "experience_match": {"score": 0.82},
                        "recommendations": ["Add more quantified achievements"]
                    }
                },
                confidence_score=0.85
            )
            mock_run_async.return_value = mock_result
            
            result_state = await cv_analyzer_agent.run_as_node(sample_agent_state)
            
            assert result_state.analysis_results is not None
            assert result_state.analysis_results["overall_score"] == 0.82
            assert len(result_state.error_messages) == 0
            mock_run_async.assert_called_once()

    async def test_run_as_node_missing_structured_cv(self, cv_analyzer_agent, sample_job_data):
        """Test run_as_node with missing structured CV."""
        state = AgentState(
            job_description="Test job",
            job_data=sample_job_data,
            structured_cv=None,
            error_messages=[]
        )
        
        result_state = await cv_analyzer_agent.run_as_node(state)
        
        assert len(result_state.error_messages) == 1
        assert "Missing structured_cv" in result_state.error_messages[0]
        assert result_state.analysis_results is None

    async def test_run_as_node_missing_job_data(self, cv_analyzer_agent, sample_structured_cv):
        """Test run_as_node with missing job data."""
        state = AgentState(
            job_description="Test job",
            job_data=None,
            structured_cv=sample_structured_cv,
            error_messages=[]
        )
        
        result_state = await cv_analyzer_agent.run_as_node(state)
        
        assert len(result_state.error_messages) == 1
        assert "Missing job_data" in result_state.error_messages[0]
        assert result_state.analysis_results is None

    async def test_run_as_node_analysis_error(self, cv_analyzer_agent, sample_agent_state):
        """Test run_as_node with analysis error."""
        # Mock the run_async method to raise an exception
        with patch.object(cv_analyzer_agent, 'run_async') as mock_run_async:
            mock_run_async.side_effect = Exception("Analysis failed")
            
            result_state = await cv_analyzer_agent.run_as_node(sample_agent_state)
            
            assert len(result_state.error_messages) == 1
            assert "Error during CV analysis: Analysis failed" in result_state.error_messages[0]
            assert result_state.analysis_results is None

    async def test_analyze_content_relevance_high_score(self, cv_analyzer_agent, sample_structured_cv, sample_job_data, mock_llm_service):
        """Test content relevance analysis with high score."""
        # Mock LLM response for high relevance
        mock_llm_service.generate_async.return_value = "Score: 0.85\nThe CV content is highly relevant to the job requirements. The candidate's experience in Python development and team leadership aligns well with the position."
        
        result = await cv_analyzer_agent._analyze_content_relevance(
            sample_structured_cv,
            sample_job_data
        )
        
        assert "score" in result
        assert result["score"] >= 0.8
        assert "details" in result
        mock_llm_service.generate_async.assert_called_once()

    async def test_analyze_content_relevance_low_score(self, cv_analyzer_agent, sample_structured_cv, sample_job_data, mock_llm_service):
        """Test content relevance analysis with low score."""
        # Mock LLM response for low relevance
        mock_llm_service.generate_async.return_value = "Score: 0.45\nThe CV content has limited relevance to the job requirements. Missing key technologies and experience areas."
        
        result = await cv_analyzer_agent._analyze_content_relevance(
            sample_structured_cv,
            sample_job_data
        )
        
        assert "score" in result
        assert result["score"] < 0.5
        assert "details" in result
        mock_llm_service.generate_async.assert_called_once()

    async def test_analyze_skill_alignment_good_match(self, cv_analyzer_agent, sample_structured_cv, sample_job_data):
        """Test skill alignment analysis with good match."""
        result = await cv_analyzer_agent._analyze_skill_alignment(
            sample_structured_cv,
            sample_job_data
        )
        
        assert "score" in result
        assert "matched_skills" in result
        assert "missing_skills" in result
        assert result["score"] > 0.5  # Should have decent match
        assert "Python" in result["matched_skills"]

    async def test_analyze_skill_alignment_poor_match(self, cv_analyzer_agent, sample_job_data):
        """Test skill alignment analysis with poor match."""
        # Create CV with no matching skills
        poor_cv = StructuredCV()
        poor_cv.sections = [
            Section(
                name="Technical Skills",
                items=[
                    Item(content="COBOL, FORTRAN, Assembly", status=ItemStatus.GENERATED)
                ]
            )
        ]
        
        result = await cv_analyzer_agent._analyze_skill_alignment(
            poor_cv,
            sample_job_data
        )
        
        assert "score" in result
        assert result["score"] < 0.3  # Should have poor match
        assert len(result["missing_skills"]) > 0

    async def test_analyze_experience_match_good_match(self, cv_analyzer_agent, sample_structured_cv, sample_job_data, mock_llm_service):
        """Test experience match analysis with good match."""
        # Mock LLM response for good experience match
        mock_llm_service.generate_async.return_value = "Score: 0.82\nThe candidate's experience aligns well with the job requirements. 4+ years of relevant experience in software development and team leadership."
        
        result = await cv_analyzer_agent._analyze_experience_match(
            sample_structured_cv,
            sample_job_data
        )
        
        assert "score" in result
        assert result["score"] > 0.8
        assert "analysis" in result
        mock_llm_service.generate_async.assert_called_once()

    async def test_analyze_experience_match_poor_match(self, cv_analyzer_agent, sample_structured_cv, sample_job_data, mock_llm_service):
        """Test experience match analysis with poor match."""
        # Mock LLM response for poor experience match
        mock_llm_service.generate_async.return_value = "Score: 0.35\nThe candidate's experience has limited alignment with job requirements. Lacks senior-level experience and leadership background."
        
        result = await cv_analyzer_agent._analyze_experience_match(
            sample_structured_cv,
            sample_job_data
        )
        
        assert "score" in result
        assert result["score"] < 0.4
        assert "analysis" in result
        mock_llm_service.generate_async.assert_called_once()

    async def test_generate_recommendations_comprehensive(self, cv_analyzer_agent, mock_llm_service):
        """Test comprehensive recommendation generation."""
        analysis_results = {
            "content_relevance": {"score": 0.75, "details": "Good relevance"},
            "skill_alignment": {"score": 0.68, "missing_skills": ["PostgreSQL", "Redis"]},
            "experience_match": {"score": 0.72, "gaps": ["Leadership experience"]}
        }
        
        # Mock LLM response with recommendations
        mock_llm_service.generate_async.return_value = "1. Add PostgreSQL and Redis to technical skills section\n2. Quantify achievements with specific metrics\n3. Highlight leadership experience more prominently\n4. Include relevant certifications\n5. Optimize keywords for ATS systems"
        
        recommendations = await cv_analyzer_agent._generate_recommendations(
            analysis_results,
            {"title": "Senior Software Engineer"}
        )
        
        assert isinstance(recommendations, list)
        assert len(recommendations) >= 3
        assert any("PostgreSQL" in rec for rec in recommendations)
        assert any("leadership" in rec.lower() for rec in recommendations)
        mock_llm_service.generate_async.assert_called_once()

    async def test_generate_recommendations_minimal(self, cv_analyzer_agent, mock_llm_service):
        """Test recommendation generation with minimal issues."""
        analysis_results = {
            "content_relevance": {"score": 0.92, "details": "Excellent relevance"},
            "skill_alignment": {"score": 0.89, "missing_skills": []},
            "experience_match": {"score": 0.91, "gaps": []}
        }
        
        # Mock LLM response with minimal recommendations
        mock_llm_service.generate_async.return_value = "1. Consider adding more quantified achievements\n2. Update contact information format"
        
        recommendations = await cv_analyzer_agent._generate_recommendations(
            analysis_results,
            {"title": "Software Engineer"}
        )
        
        assert isinstance(recommendations, list)
        assert len(recommendations) >= 1
        mock_llm_service.generate_async.assert_called_once()

    def test_extract_cv_skills(self, cv_analyzer_agent, sample_structured_cv):
        """Test extracting skills from CV."""
        skills = cv_analyzer_agent._extract_cv_skills(sample_structured_cv)
        
        assert isinstance(skills, list)
        assert "Python" in skills
        assert "JavaScript" in skills
        assert "React" in skills
        assert "Django" in skills
        assert "AWS" in skills

    def test_extract_cv_skills_empty_cv(self, cv_analyzer_agent):
        """Test extracting skills from empty CV."""
        empty_cv = StructuredCV()
        
        skills = cv_analyzer_agent._extract_cv_skills(empty_cv)
        
        assert isinstance(skills, list)
        assert len(skills) == 0

    def test_extract_cv_skills_no_skills_section(self, cv_analyzer_agent):
        """Test extracting skills from CV without skills section."""
        cv = StructuredCV()
        cv.sections = [
            Section(
                name="Education",
                items=[Item(content="BS Computer Science", status=ItemStatus.GENERATED)]
            )
        ]
        
        skills = cv_analyzer_agent._extract_cv_skills(cv)
        
        assert isinstance(skills, list)
        assert len(skills) == 0

    def test_calculate_skill_match_score(self, cv_analyzer_agent):
        """Test skill match score calculation."""
        cv_skills = ["Python", "JavaScript", "React", "Django"]
        job_skills = ["Python", "Django", "PostgreSQL", "Redis"]
        
        score, matched, missing = cv_analyzer_agent._calculate_skill_match_score(
            cv_skills,
            job_skills
        )
        
        assert isinstance(score, float)
        assert 0 <= score <= 1
        assert score == 0.5  # 2 out of 4 job skills matched
        assert "Python" in matched
        assert "Django" in matched
        assert "PostgreSQL" in missing
        assert "Redis" in missing

    def test_calculate_skill_match_score_perfect_match(self, cv_analyzer_agent):
        """Test skill match score with perfect match."""
        cv_skills = ["Python", "Django", "PostgreSQL", "Redis", "Docker"]
        job_skills = ["Python", "Django", "PostgreSQL", "Redis"]
        
        score, matched, missing = cv_analyzer_agent._calculate_skill_match_score(
            cv_skills,
            job_skills
        )
        
        assert score == 1.0  # Perfect match
        assert len(matched) == 4
        assert len(missing) == 0

    def test_calculate_skill_match_score_no_match(self, cv_analyzer_agent):
        """Test skill match score with no match."""
        cv_skills = ["COBOL", "FORTRAN"]
        job_skills = ["Python", "Django", "React"]
        
        score, matched, missing = cv_analyzer_agent._calculate_skill_match_score(
            cv_skills,
            job_skills
        )
        
        assert score == 0.0  # No match
        assert len(matched) == 0
        assert len(missing) == 3

    def test_parse_score_from_response(self, cv_analyzer_agent):
        """Test parsing score from LLM response."""
        response = "Score: 0.85\nThe analysis shows good alignment..."
        
        score = cv_analyzer_agent._parse_score_from_response(response)
        
        assert score == 0.85

    def test_parse_score_from_response_alternative_format(self, cv_analyzer_agent):
        """Test parsing score from alternative response format."""
        response = "Analysis Score: 0.72\nDetailed analysis follows..."
        
        score = cv_analyzer_agent._parse_score_from_response(response)
        
        assert score == 0.72

    def test_parse_score_from_response_no_score(self, cv_analyzer_agent):
        """Test parsing score when no score is found."""
        response = "The analysis shows good alignment but no specific score."
        
        score = cv_analyzer_agent._parse_score_from_response(response)
        
        assert score == 0.5  # Default score

    def test_calculate_overall_score(self, cv_analyzer_agent):
        """Test overall score calculation."""
        analysis_results = {
            "content_relevance": {"score": 0.85},
            "skill_alignment": {"score": 0.78},
            "experience_match": {"score": 0.82}
        }
        
        overall_score = cv_analyzer_agent._calculate_overall_score(analysis_results)
        
        assert isinstance(overall_score, float)
        assert 0 <= overall_score <= 1
        # Should be weighted average, approximately 0.82
        assert 0.80 <= overall_score <= 0.85

    def test_calculate_overall_score_missing_components(self, cv_analyzer_agent):
        """Test overall score calculation with missing components."""
        analysis_results = {
            "content_relevance": {"score": 0.85},
            "skill_alignment": {"score": 0.78}
            # Missing experience_match
        }
        
        overall_score = cv_analyzer_agent._calculate_overall_score(analysis_results)
        
        assert isinstance(overall_score, float)
        assert 0 <= overall_score <= 1
        # Should handle missing components gracefully

    def test_confidence_score_calculation(self, cv_analyzer_agent):
        """Test confidence score calculation."""
        # Test with high-quality analysis
        high_quality_output = {
            "analysis_results": {
                "overall_score": 0.85,
                "content_relevance": {"score": 0.88},
                "skill_alignment": {"score": 0.82},
                "experience_match": {"score": 0.85},
                "recommendations": ["rec1", "rec2", "rec3"]
            }
        }
        
        confidence = cv_analyzer_agent.get_confidence_score(high_quality_output)
        assert confidence > 0.8  # Should be high for comprehensive analysis
        
        # Test with minimal analysis
        minimal_output = {
            "analysis_results": {
                "overall_score": 0.45
            }
        }
        
        confidence = cv_analyzer_agent.get_confidence_score(minimal_output)
        assert confidence < 0.7  # Should be lower for minimal analysis

    def test_build_analysis_prompt(self, cv_analyzer_agent, sample_structured_cv, sample_job_data):
        """Test building analysis prompt."""
        prompt = cv_analyzer_agent._build_analysis_prompt(
            sample_structured_cv,
            sample_job_data,
            "content_relevance"
        )
        
        assert isinstance(prompt, str)
        assert "content_relevance" in prompt.lower()
        assert "Senior Software Engineer" in prompt
        assert "Python" in prompt

    def test_extract_experience_years(self, cv_analyzer_agent, sample_structured_cv):
        """Test extracting years of experience from CV."""
        years = cv_analyzer_agent._extract_experience_years(sample_structured_cv)
        
        assert isinstance(years, int)
        assert years >= 0
        # Should extract approximately 4 years from 2020-2024
        assert 3 <= years <= 5

    def test_extract_experience_years_no_experience(self, cv_analyzer_agent):
        """Test extracting years from CV with no experience section."""
        cv = StructuredCV()
        cv.sections = [
            Section(
                name="Education",
                items=[Item(content="BS Computer Science", status=ItemStatus.GENERATED)]
            )
        ]
        
        years = cv_analyzer_agent._extract_experience_years(cv)
        
        assert years == 0

    @patch('src.agents.cv_analyzer_agent.logger')
    async def test_run_as_node_logs_execution(self, mock_logger, cv_analyzer_agent, sample_agent_state):
        """Test that run_as_node logs execution properly."""
        with patch.object(cv_analyzer_agent, 'run_async') as mock_run_async:
            mock_result = AgentResult(
                output={"analysis_results": {"overall_score": 0.8}},
                confidence_score=0.8
            )
            mock_run_async.return_value = mock_result
            
            await cv_analyzer_agent.run_as_node(sample_agent_state)
            
            # Verify logging calls
            mock_logger.info.assert_called()
            assert any("Starting CV analysis" in str(call) for call in mock_logger.info.call_args_list)

    def test_normalize_skill_name(self, cv_analyzer_agent):
        """Test skill name normalization."""
        # Test various skill formats
        assert cv_analyzer_agent._normalize_skill_name("Python") == "python"
        assert cv_analyzer_agent._normalize_skill_name("JavaScript") == "javascript"
        assert cv_analyzer_agent._normalize_skill_name("Node.js") == "node.js"
        assert cv_analyzer_agent._normalize_skill_name("React.js") == "react.js"
        assert cv_analyzer_agent._normalize_skill_name("C++") == "c++"

    def test_extract_skills_from_text(self, cv_analyzer_agent):
        """Test extracting skills from text content."""
        text = "Proficient in Python, JavaScript, React, Django, and AWS cloud services"
        
        skills = cv_analyzer_agent._extract_skills_from_text(text)
        
        assert isinstance(skills, list)
        assert "Python" in skills
        assert "JavaScript" in skills
        assert "React" in skills
        assert "Django" in skills
        assert "AWS" in skills

    def test_extract_skills_from_text_complex(self, cv_analyzer_agent):
        """Test extracting skills from complex text."""
        text = "Technologies: Python (5+ years), JavaScript/TypeScript, React.js, Node.js, Django/Flask, PostgreSQL, MongoDB, AWS (EC2, S3, Lambda), Docker, Kubernetes"
        
        skills = cv_analyzer_agent._extract_skills_from_text(text)
        
        assert isinstance(skills, list)
        assert len(skills) > 5
        assert "Python" in skills
        assert "TypeScript" in skills or "JavaScript" in skills
        assert "PostgreSQL" in skills