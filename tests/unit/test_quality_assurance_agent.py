#!/usr/bin/env python3
"""
Unit tests for QualityAssuranceAgent.

Tests the quality assurance functionality for validating
generated CV content against job requirements.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any, List

from src.agents.quality_assurance_agent import QualityAssuranceAgent
from src.agents.agent_base import AgentExecutionContext, AgentResult
from src.models.data_models import StructuredCV, Section, Item, ItemStatus
from src.models.workflow_models import AgentState
from src.utils.llm_service import LLM


class TestQualityAssuranceAgent:
    """Test cases for QualityAssuranceAgent."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM instance."""
        llm = Mock(spec=LLM)
        llm.generate_content = AsyncMock()
        return llm

    @pytest.fixture
    def qa_agent(self, mock_llm):
        """Create a QualityAssuranceAgent instance for testing."""
        return QualityAssuranceAgent(
            name="TestQA", description="Test quality assurance agent", llm=mock_llm
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
    def sample_job_data(self):
        """Sample job data for testing."""
        return {
            "job_title": "Senior Software Engineer",
            "required_skills": ["Python", "React", "AWS", "Docker"],
            "experience_level": "5+ years",
            "education_requirements": ["Bachelor's degree in Computer Science"],
            "responsibilities": [
                "Design and develop scalable web applications",
                "Lead technical architecture decisions",
                "Mentor junior developers",
            ],
        }

    @pytest.fixture
    def sample_structured_cv(self):
        """Create a sample structured CV for testing."""
        cv = StructuredCV()
        cv.sections = [
            Section(
                name="Professional Experience",
                items=[
                    Item(
                        content="Senior Software Developer at XYZ Corp (2019-2024)",
                        status=ItemStatus.GENERATED,
                    ),
                    Item(
                        content="Led development of scalable web applications using Python and React",
                        status=ItemStatus.GENERATED,
                    ),
                    Item(
                        content="Mentored team of 3 junior developers",
                        status=ItemStatus.GENERATED,
                    ),
                ],
            ),
            Section(
                name="Technical Skills",
                items=[
                    Item(
                        content="Programming Languages: Python, JavaScript, TypeScript",
                        status=ItemStatus.GENERATED,
                    ),
                    Item(
                        content="Frameworks: React, Django, Flask",
                        status=ItemStatus.GENERATED,
                    ),
                    Item(
                        content="Cloud Platforms: AWS, Docker, Kubernetes",
                        status=ItemStatus.GENERATED,
                    ),
                ],
            ),
            Section(
                name="Education",
                items=[
                    Item(
                        content="Bachelor of Science in Computer Science, ABC University (2015-2019)",
                        status=ItemStatus.GENERATED,
                    )
                ],
            ),
        ]
        return cv

    @pytest.fixture
    def sample_agent_state(self, sample_job_data, sample_structured_cv):
        """Create a sample agent state for testing."""
        return AgentState(
            job_description="Senior Software Engineer position",
            job_data=sample_job_data,
            structured_cv=sample_structured_cv,
            error_messages=[],
        )

    def test_qa_agent_initialization(self, mock_llm):
        """Test QualityAssuranceAgent initialization."""
        agent = QualityAssuranceAgent(
            name="TestQA", description="Test description", llm=mock_llm
        )

        assert agent.name == "TestQA"
        assert agent.description == "Test description"
        assert agent.llm == mock_llm
        assert hasattr(agent, "input_schema")
        assert hasattr(agent, "output_schema")

    def test_qa_agent_initialization_without_llm(self):
        """Test QualityAssuranceAgent initialization without LLM."""
        agent = QualityAssuranceAgent(name="TestQA", description="Test description")

        assert agent.name == "TestQA"
        assert agent.description == "Test description"
        assert agent.llm is None

    async def test_run_async_success(
        self, qa_agent, execution_context, sample_structured_cv, sample_job_data
    ):
        """Test successful quality assurance execution."""
        input_data = {
            "structured_cv": sample_structured_cv,
            "job_description_data": sample_job_data,
        }

        # Mock the quality check methods
        with patch.object(
            qa_agent, "_check_content_relevance"
        ) as mock_relevance, patch.object(
            qa_agent, "_check_skill_alignment"
        ) as mock_skills, patch.object(
            qa_agent, "_check_experience_match"
        ) as mock_experience, patch.object(
            qa_agent, "_check_formatting_quality"
        ) as mock_formatting:

            mock_relevance.return_value = {"score": 0.85, "issues": []}
            mock_skills.return_value = {
                "score": 0.90,
                "missing_skills": [],
                "matched_skills": ["Python", "React"],
            }
            mock_experience.return_value = {"score": 0.80, "issues": []}
            mock_formatting.return_value = {"score": 0.95, "issues": []}

            result = await qa_agent.run_async(input_data, execution_context)

            assert result.success is True
            assert "overall_score" in result.output_data
            assert "content_relevance" in result.output_data
            assert "skill_alignment" in result.output_data
            assert "experience_match" in result.output_data
            assert "formatting_quality" in result.output_data
            assert result.confidence_score > 0.0

    async def test_run_async_missing_cv(
        self, qa_agent, execution_context, sample_job_data
    ):
        """Test quality assurance execution with missing CV."""
        input_data = {"job_description_data": sample_job_data}

        result = await qa_agent.run_async(input_data, execution_context)

        assert result.success is False
        assert "Missing required structured_cv" in result.error_message
        assert result.confidence_score == 0.0

    async def test_run_async_missing_job_data(
        self, qa_agent, execution_context, sample_structured_cv
    ):
        """Test quality assurance execution with missing job data."""
        input_data = {"structured_cv": sample_structured_cv}

        result = await qa_agent.run_async(input_data, execution_context)

        assert result.success is False
        assert "Missing required job_description_data" in result.error_message
        assert result.confidence_score == 0.0

    async def test_run_as_node_success(self, qa_agent, sample_agent_state):
        """Test successful run_as_node execution."""
        # Mock the run_async method
        mock_result = AgentResult(
            success=True,
            output_data={
                "overall_score": 0.85,
                "content_relevance": {"score": 0.85, "issues": []},
                "skill_alignment": {"score": 0.90, "missing_skills": []},
                "experience_match": {"score": 0.80, "issues": []},
                "formatting_quality": {"score": 0.95, "issues": []},
                "recommendations": ["Consider adding more specific metrics"],
            },
            confidence_score=0.85,
        )

        with patch.object(qa_agent, "run_async", return_value=mock_result):
            result_state = await qa_agent.run_as_node(sample_agent_state)

            assert result_state.quality_check_results is not None
            assert "overall_score" in result_state.quality_check_results
            assert result_state.updated_structured_cv is not None
            assert len(result_state.error_messages) == 0

    async def test_run_as_node_missing_cv(self, qa_agent, sample_job_data):
        """Test run_as_node with missing CV."""
        state = AgentState(
            job_description="Test job",
            job_data=sample_job_data,
            structured_cv=None,
            error_messages=[],
        )

        result_state = await qa_agent.run_as_node(state)

        assert len(result_state.error_messages) == 1
        assert "Missing cv" in result_state.error_messages[0]
        assert result_state.quality_check_results is None

    async def test_run_as_node_missing_job_data(self, qa_agent, sample_structured_cv):
        """Test run_as_node with missing job data."""
        state = AgentState(
            job_description="Test job",
            job_data=None,
            structured_cv=sample_structured_cv,
            error_messages=[],
        )

        result_state = await qa_agent.run_as_node(state)

        assert len(result_state.error_messages) == 1
        assert "Missing job_data" in result_state.error_messages[0]
        assert result_state.quality_check_results is None

    async def test_run_as_node_execution_error(self, qa_agent, sample_agent_state):
        """Test run_as_node with execution error."""
        # Mock the run_async method to raise an exception
        with patch.object(qa_agent, "run_async") as mock_run:
            mock_run.side_effect = Exception("QA check failed")

            result_state = await qa_agent.run_as_node(sample_agent_state)

            assert len(result_state.error_messages) == 1
            assert (
                "Error during quality assurance: QA check failed"
                in result_state.error_messages[0]
            )
            assert result_state.quality_check_results is None

    def test_check_content_relevance_high_score(
        self, qa_agent, sample_structured_cv, sample_job_data
    ):
        """Test content relevance check with high relevance."""
        result = qa_agent._check_content_relevance(
            sample_structured_cv, sample_job_data
        )

        assert "score" in result
        assert "issues" in result
        assert isinstance(result["score"], float)
        assert isinstance(result["issues"], list)
        assert 0.0 <= result["score"] <= 1.0

    def test_check_skill_alignment_perfect_match(
        self, qa_agent, sample_structured_cv, sample_job_data
    ):
        """Test skill alignment check with good skill match."""
        result = qa_agent._check_skill_alignment(sample_structured_cv, sample_job_data)

        assert "score" in result
        assert "missing_skills" in result
        assert "matched_skills" in result
        assert isinstance(result["score"], float)
        assert isinstance(result["missing_skills"], list)
        assert isinstance(result["matched_skills"], list)
        assert 0.0 <= result["score"] <= 1.0

    def test_check_skill_alignment_missing_skills(self, qa_agent, sample_job_data):
        """Test skill alignment check with missing skills."""
        # Create CV with limited skills
        cv = StructuredCV()
        cv.sections = [
            Section(
                name="Technical Skills",
                items=[
                    Item(
                        content="Programming Languages: Python",
                        status=ItemStatus.GENERATED,
                    )
                ],
            )
        ]

        result = qa_agent._check_skill_alignment(cv, sample_job_data)

        assert result["score"] < 1.0  # Should be less than perfect
        assert len(result["missing_skills"]) > 0  # Should have missing skills
        assert "React" in result["missing_skills"]  # React should be missing

    def test_check_experience_match(
        self, qa_agent, sample_structured_cv, sample_job_data
    ):
        """Test experience level matching."""
        result = qa_agent._check_experience_match(sample_structured_cv, sample_job_data)

        assert "score" in result
        assert "issues" in result
        assert isinstance(result["score"], float)
        assert isinstance(result["issues"], list)
        assert 0.0 <= result["score"] <= 1.0

    def test_check_formatting_quality(self, qa_agent, sample_structured_cv):
        """Test formatting quality check."""
        result = qa_agent._check_formatting_quality(sample_structured_cv)

        assert "score" in result
        assert "issues" in result
        assert isinstance(result["score"], float)
        assert isinstance(result["issues"], list)
        assert 0.0 <= result["score"] <= 1.0

    def test_check_formatting_quality_poor_formatting(self, qa_agent):
        """Test formatting quality check with poor formatting."""
        # Create CV with poor formatting
        cv = StructuredCV()
        cv.sections = [
            Section(
                name="experience",  # lowercase section name
                items=[
                    Item(
                        content="worked at company", status=ItemStatus.GENERATED
                    ),  # poor capitalization
                    Item(
                        content="did stuff", status=ItemStatus.GENERATED
                    ),  # vague content
                ],
            )
        ]

        result = qa_agent._check_formatting_quality(cv)

        assert result["score"] < 0.8  # Should be lower for poor formatting
        assert len(result["issues"]) > 0  # Should have formatting issues

    def test_extract_skills_from_cv(self, qa_agent, sample_structured_cv):
        """Test skill extraction from CV."""
        skills = qa_agent._extract_skills_from_cv(sample_structured_cv)

        assert isinstance(skills, list)
        assert len(skills) > 0
        assert "Python" in skills
        assert "JavaScript" in skills
        assert "React" in skills

    def test_extract_skills_from_empty_cv(self, qa_agent):
        """Test skill extraction from empty CV."""
        cv = StructuredCV()
        skills = qa_agent._extract_skills_from_cv(cv)

        assert isinstance(skills, list)
        assert len(skills) == 0

    def test_calculate_overall_score(self, qa_agent):
        """Test overall score calculation."""
        scores = {
            "content_relevance": {"score": 0.8},
            "skill_alignment": {"score": 0.9},
            "experience_match": {"score": 0.7},
            "formatting_quality": {"score": 0.95},
        }

        overall_score = qa_agent._calculate_overall_score(scores)

        assert isinstance(overall_score, float)
        assert 0.0 <= overall_score <= 1.0
        assert overall_score > 0.8  # Should be high for good scores

    def test_calculate_overall_score_poor_scores(self, qa_agent):
        """Test overall score calculation with poor individual scores."""
        scores = {
            "content_relevance": {"score": 0.3},
            "skill_alignment": {"score": 0.4},
            "experience_match": {"score": 0.2},
            "formatting_quality": {"score": 0.5},
        }

        overall_score = qa_agent._calculate_overall_score(scores)

        assert isinstance(overall_score, float)
        assert 0.0 <= overall_score <= 1.0
        assert overall_score < 0.5  # Should be low for poor scores

    def test_generate_recommendations(self, qa_agent):
        """Test recommendation generation."""
        quality_results = {
            "content_relevance": {
                "score": 0.6,
                "issues": ["Content not specific enough"],
            },
            "skill_alignment": {
                "score": 0.7,
                "missing_skills": ["Docker", "Kubernetes"],
            },
            "experience_match": {"score": 0.8, "issues": []},
            "formatting_quality": {"score": 0.5, "issues": ["Inconsistent formatting"]},
        }

        recommendations = qa_agent._generate_recommendations(quality_results)

        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        assert any("specific" in rec.lower() for rec in recommendations)
        assert any("docker" in rec.lower() for rec in recommendations)
        assert any("format" in rec.lower() for rec in recommendations)

    def test_confidence_score_calculation(self, qa_agent):
        """Test confidence score calculation."""
        # Test with high quality results
        high_quality_data = {
            "overall_score": 0.9,
            "content_relevance": {"score": 0.9},
            "skill_alignment": {"score": 0.95},
            "experience_match": {"score": 0.85},
            "formatting_quality": {"score": 0.9},
        }

        confidence = qa_agent.get_confidence_score(high_quality_data)
        assert confidence > 0.8  # Should be high for high quality

        # Test with low quality results
        low_quality_data = {
            "overall_score": 0.3,
            "content_relevance": {"score": 0.3},
            "skill_alignment": {"score": 0.2},
            "experience_match": {"score": 0.4},
            "formatting_quality": {"score": 0.3},
        }

        confidence = qa_agent.get_confidence_score(low_quality_data)
        assert confidence < 0.5  # Should be lower for low quality

    @patch("src.agents.quality_assurance_agent.logger")
    async def test_run_as_node_logs_execution(
        self, mock_logger, qa_agent, sample_agent_state
    ):
        """Test that run_as_node logs execution properly."""
        mock_result = AgentResult(
            success=True, output_data={"overall_score": 0.8}, confidence_score=0.8
        )

        with patch.object(qa_agent, "run_async", return_value=mock_result):
            await qa_agent.run_as_node(sample_agent_state)

            # Verify logging calls
            mock_logger.info.assert_called()
            assert any(
                "Starting quality assurance" in str(call)
                for call in mock_logger.info.call_args_list
            )

    def test_validate_cv_structure(self, qa_agent, sample_structured_cv):
        """Test CV structure validation."""
        issues = qa_agent._validate_cv_structure(sample_structured_cv)

        assert isinstance(issues, list)
        # Well-structured CV should have minimal issues
        assert len(issues) == 0 or all("minor" in issue.lower() for issue in issues)

    def test_validate_cv_structure_missing_sections(self, qa_agent):
        """Test CV structure validation with missing sections."""
        # Create CV with minimal sections
        cv = StructuredCV()
        cv.sections = [
            Section(
                name="Skills",
                items=[Item(content="Python", status=ItemStatus.GENERATED)],
            )
        ]

        issues = qa_agent._validate_cv_structure(cv)

        assert isinstance(issues, list)
        assert len(issues) > 0  # Should have issues for missing sections
        assert any("experience" in issue.lower() for issue in issues)
