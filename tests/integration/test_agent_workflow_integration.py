#!/usr/bin/env python3
"""
Integration tests for the complete agent workflow pipeline.
Tests the interaction between multiple agents in sequence.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, Mock, patch
from pathlib import Path
import tempfile
from uuid import uuid4

from src.orchestration.cv_workflow_graph import (
    parser_node,
    research_node,
    content_writer_node,
    qa_node,
    formatter_node,
)
from src.orchestration.state import AgentState
from src.models.data_models import (
    StructuredCV,
    CVSection,
    CVItem,
    JobDescriptionData,
    Section,
    Item,
    ItemStatus,
    ItemType,
)
from src.agents.parser_agent import ParserAgent
from src.agents.research_agent import ResearchAgent
from src.agents.enhanced_content_writer import EnhancedContentWriterAgent
from src.agents.quality_assurance_agent import QualityAssuranceAgent
from src.agents.formatter_agent import FormatterAgent


class TestAgentWorkflowIntegration:
    """Integration tests for agent workflow sequences."""

    @pytest.fixture
    def sample_job_description(self):
        """Sample job description for testing."""
        return """
        Senior Software Engineer - AI/ML

        We are seeking a Senior Software Engineer with expertise in:
        - Python programming and machine learning frameworks
        - Experience with TensorFlow, PyTorch, or similar ML libraries
        - Strong background in data structures and algorithms
        - Experience with cloud platforms (AWS, GCP, Azure)
        - Knowledge of containerization (Docker, Kubernetes)
        - 5+ years of software development experience

        Responsibilities:
        - Design and implement ML models for production systems
        - Collaborate with data scientists and product teams
        - Optimize model performance and scalability
        - Maintain and improve existing ML infrastructure
        """

    @pytest.fixture
    def sample_cv_data(self):
        """Sample CV data for testing."""
        return StructuredCV(
            id="test-cv-integration",
            metadata={
                "name": "John Doe",
                "email": "john.doe@example.com",
                "phone": "+1-555-0123",
                "linkedin": "https://linkedin.com/in/johndoe",
            },
            sections=[
                CVSection(
                    name="Key Qualifications",
                    items=[
                        CVItem(content="Python Programming"),
                        CVItem(content="Machine Learning"),
                        CVItem(content="Data Analysis"),
                    ],
                ),
                CVSection(
                    name="Professional Experience",
                    items=[
                        CVItem(content="Software Engineer at TechCorp (2020-2023)"),
                        CVItem(
                            content="Developed ML models for recommendation systems"
                        ),
                        CVItem(
                            content="Implemented data pipelines processing 1M+ records"
                        ),
                    ],
                ),
                CVSection(
                    name="Education",
                    items=[
                        CVItem(content="BS Computer Science, University of Technology")
                    ],
                ),
            ],
        )

    @pytest.fixture
    def initial_state(self, sample_job_description, sample_cv_data):
        """Initial agent state for testing."""
        return AgentState(
            job_description_data=JobDescriptionData(raw_text=sample_job_description),
            structured_cv=sample_cv_data,
            error_messages=[],
            processing_queue=[],
            research_data={},
            content_data={},
            quality_scores={},
            output_data={},
        )

    @pytest.mark.asyncio
    async def test_parser_to_research_workflow(self, initial_state):
        """Test parser -> research agent workflow integration."""
        # Mock LLM responses
        mock_llm_response = """
        Industry Trends:
        - AI/ML adoption increasing across industries
        - Focus on production-ready ML systems
        - Emphasis on MLOps and model deployment

        Key Skills:
        - Python, TensorFlow, PyTorch
        - Cloud platforms (AWS, GCP)
        - Containerization technologies
        """

        with patch(
            "src.agents.parser_agent.ParserAgent.run_as_node"
        ) as mock_parser, patch(
            "src.agents.research_agent.ResearchAgent.run_as_node"
        ) as mock_research:

            # Configure parser mock
            mock_parser.return_value = {
                **initial_state,
                "job_description_data": JobDescriptionData(
                    raw_text=initial_state["job_description_data"].raw_text,
                    parsed_skills=["Python", "Machine Learning", "TensorFlow"],
                    parsed_requirements=["5+ years experience", "ML frameworks"],
                ),
            }

            # Configure research mock
            mock_research.return_value = {
                **initial_state,
                "research_data": {
                    "industry_trends": mock_llm_response,
                    "skill_requirements": "Python, ML frameworks, cloud platforms",
                    "company_culture": "Innovation-focused, collaborative environment",
                },
            }

            # Execute parser node
            parser_result = await parser_node(initial_state)

            # Execute research node with parser output
            research_result = await research_node(parser_result)

            # Verify parser was called
            mock_parser.assert_called_once()

            # Verify research was called with parser output
            mock_research.assert_called_once()

            # Verify research data was populated
            assert "research_data" in research_result
            assert "industry_trends" in research_result["research_data"]
            assert "skill_requirements" in research_result["research_data"]

    @pytest.mark.asyncio
    async def test_content_writer_to_qa_workflow(self, initial_state):
        """Test content writer -> QA agent workflow integration."""
        # Add research data to state
        state_with_research = {
            **initial_state,
            "research_data": {
                "industry_trends": "AI/ML growth in enterprise",
                "skill_requirements": "Python, TensorFlow, cloud platforms",
                "company_culture": "Innovation-focused environment",
            },
            "processing_queue": ["item_1", "item_2"],
        }

        with patch(
            "src.agents.enhanced_content_writer.EnhancedContentWriterAgent.run_as_node"
        ) as mock_writer, patch(
            "src.agents.quality_assurance_agent.QualityAssuranceAgent.run_as_node"
        ) as mock_qa:

            # Configure content writer mock
            mock_writer.return_value = {
                **state_with_research,
                "content_data": {
                    "enhanced_items": {
                        "item_1": "Enhanced content for item 1",
                        "item_2": "Enhanced content for item 2",
                    }
                },
                "processing_queue": [],  # Items processed
            }

            # Configure QA mock
            mock_qa.return_value = {
                **state_with_research,
                "quality_scores": {
                    "overall_score": 85,
                    "content_relevance": 90,
                    "skill_alignment": 80,
                    "experience_match": 85,
                },
                "qa_recommendations": [
                    "Consider adding more specific metrics",
                    "Highlight cloud platform experience",
                ],
            }

            # Execute content writer node
            writer_result = await content_writer_node(state_with_research)

            # Execute QA node with writer output
            qa_result = await qa_node(writer_result)

            # Verify content writer was called
            mock_writer.assert_called_once()

            # Verify QA was called with writer output
            mock_qa.assert_called_once()

            # Verify quality scores were generated
            assert "quality_scores" in qa_result
            assert qa_result["quality_scores"]["overall_score"] == 85
            assert "qa_recommendations" in qa_result

    @pytest.mark.asyncio
    async def test_qa_to_formatter_workflow(self, initial_state):
        """Test QA -> formatter agent workflow integration."""
        # State after QA processing
        state_after_qa = {
            **initial_state,
            "content_data": {
                "enhanced_items": {
                    "item_1": "Enhanced professional experience",
                    "item_2": "Enhanced technical skills",
                }
            },
            "quality_scores": {
                "overall_score": 88,
                "content_relevance": 90,
                "skill_alignment": 85,
            },
        }

        with patch(
            "src.agents.formatter_agent.FormatterAgent.run_as_node"
        ) as mock_formatter:

            # Configure formatter mock
            mock_formatter.return_value = {
                **state_after_qa,
                "output_data": {
                    "pdf_path": "/tmp/formatted_cv.pdf",
                    "html_content": "<html>Formatted CV content</html>",
                    "format_type": "professional",
                },
            }

            # Execute formatter node
            formatter_result = await formatter_node(state_after_qa)

            # Verify formatter was called
            mock_formatter.assert_called_once()

            # Verify output data was generated
            assert "output_data" in formatter_result
            assert "pdf_path" in formatter_result["output_data"]
            assert "html_content" in formatter_result["output_data"]

    @pytest.mark.asyncio
    async def test_complete_workflow_integration(self, initial_state):
        """Test complete workflow from parser to formatter."""
        with patch(
            "src.agents.parser_agent.ParserAgent.run_as_node"
        ) as mock_parser, patch(
            "src.agents.research_agent.ResearchAgent.run_as_node"
        ) as mock_research, patch(
            "src.agents.enhanced_content_writer.EnhancedContentWriterAgent.run_as_node"
        ) as mock_writer, patch(
            "src.agents.quality_assurance_agent.QualityAssuranceAgent.run_as_node"
        ) as mock_qa, patch(
            "src.agents.formatter_agent.FormatterAgent.run_as_node"
        ) as mock_formatter:

            # Configure all mocks to pass state through with additions
            mock_parser.return_value = {
                **initial_state,
                "job_description_data": JobDescriptionData(
                    raw_text=initial_state["job_description_data"].raw_text,
                    parsed_skills=["Python", "ML", "TensorFlow"],
                ),
            }

            mock_research.return_value = {
                **mock_parser.return_value,
                "research_data": {"industry_trends": "AI growth"},
            }

            mock_writer.return_value = {
                **mock_research.return_value,
                "content_data": {"enhanced_items": {"item_1": "Enhanced content"}},
            }

            mock_qa.return_value = {
                **mock_writer.return_value,
                "quality_scores": {"overall_score": 85},
            }

            mock_formatter.return_value = {
                **mock_qa.return_value,
                "output_data": {"pdf_path": "/tmp/cv.pdf"},
            }

            # Execute complete workflow
            result = initial_state
            result = await parser_node(result)
            result = await research_node(result)
            result = await content_writer_node(result)
            result = await qa_node(result)
            result = await formatter_node(result)

            # Verify all agents were called in sequence
            mock_parser.assert_called_once()
            mock_research.assert_called_once()
            mock_writer.assert_called_once()
            mock_qa.assert_called_once()
            mock_formatter.assert_called_once()

            # Verify final state contains all expected data
            assert "job_description_data" in result
            assert "research_data" in result
            assert "content_data" in result
            assert "quality_scores" in result
            assert "output_data" in result

            # Verify final AgentState fields are populated and not None
            assert hasattr(result, "job_description_data")
            assert result.job_description_data is not None
            assert hasattr(result, "structured_cv")
            assert result.structured_cv is not None
            assert hasattr(result, "research_findings")
            assert result.research_findings is not None
            assert hasattr(result, "quality_check_results")
            assert result.quality_check_results is not None
            assert hasattr(result, "final_output_path")
            assert result.final_output_path is not None

    @pytest.mark.asyncio
    async def test_workflow_error_propagation(self, initial_state):
        """Test error handling across workflow nodes."""
        with patch(
            "src.agents.parser_agent.ParserAgent.run_as_node"
        ) as mock_parser, patch(
            "src.agents.research_agent.ResearchAgent.run_as_node"
        ) as mock_research:

            # Configure parser to succeed
            mock_parser.return_value = {
                **initial_state,
                "job_description_data": JobDescriptionData(
                    raw_text=initial_state["job_description_data"].raw_text
                ),
            }

            # Configure research to add error
            mock_research.return_value = {
                **mock_parser.return_value,
                "error_messages": ["Research agent failed to connect to LLM service"],
            }

            # Execute workflow with error
            result = initial_state
            result = await parser_node(result)
            result = await research_node(result)

            # Verify error was propagated
            assert len(result["error_messages"]) > 0
            assert "Research agent failed" in result["error_messages"][0]

    def test_state_consistency_across_nodes(self, initial_state):
        """Test that state structure remains consistent across workflow nodes."""
        # Verify initial state has required keys
        required_keys = [
            "job_description_data",
            "structured_cv",
            "error_messages",
            "processing_queue",
            "research_data",
            "content_data",
            "quality_scores",
            "output_data",
        ]

        for key in required_keys:
            assert key in initial_state, f"Missing required key: {key}"

        # Verify data types
        assert isinstance(initial_state["error_messages"], list)
        assert isinstance(initial_state["processing_queue"], list)
        assert isinstance(initial_state["research_data"], dict)
        assert isinstance(initial_state["content_data"], dict)
        assert isinstance(initial_state["quality_scores"], dict)
        assert isinstance(initial_state["output_data"], dict)

    @pytest.mark.asyncio
    async def test_workflow_performance_tracking(self, initial_state):
        """Test that workflow nodes track performance metrics."""
        with patch("src.agents.parser_agent.ParserAgent.run_as_node") as mock_parser:

            # Configure parser with timing simulation
            async def slow_parser(state):
                await asyncio.sleep(0.1)  # Simulate processing time
                return {
                    **state,
                    "performance_metrics": {
                        "parser_duration": 0.1,
                        "parser_timestamp": asyncio.get_event_loop().time(),
                    },
                }

            mock_parser.side_effect = slow_parser

            # Execute parser node and measure time
            start_time = asyncio.get_event_loop().time()
            result = await parser_node(initial_state)
            end_time = asyncio.get_event_loop().time()

            # Verify timing
            assert end_time - start_time >= 0.1
            mock_parser.assert_called_once()
