#!/usr/bin/env python3
"""
Integration tests for LLM service interactions across different agents.
Tests the integration between agents and the LLM service with various scenarios.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from typing import Dict, Any
import json

from src.services.llm import EnhancedLLMService
from src.agents.parser_agent import ParserAgent
from src.agents.research_agent import ResearchAgent
from src.agents.enhanced_content_writer import EnhancedContentWriterAgent
from src.agents.quality_assurance_agent import QualityAssuranceAgent
from src.models.data_models import (
    StructuredCV, CVSection, CVItem, JobDescriptionData,
    Section, Item, ItemStatus, ItemType
)
from src.orchestration.state import AgentState


class TestLLMServiceIntegration:
    """Integration tests for LLM service interactions."""

    @pytest.fixture
    def mock_llm_service(self):
        """Create a mock LLM service for testing."""
        service = AsyncMock(spec=EnhancedLLMService)
        
        # Configure default responses
        service.generate_content_async.return_value = "Mock LLM response"
        service.generate_structured_content_async.return_value = {
            "skills": ["Python", "Machine Learning", "Data Analysis"],
            "requirements": ["5+ years experience", "Bachelor's degree"]
        }
        
        return service

    @pytest.fixture
    def sample_job_description(self):
        """Sample job description for testing."""
        return """
        Data Scientist Position
        
        We are looking for an experienced Data Scientist with:
        - Strong Python programming skills
        - Experience with machine learning frameworks (scikit-learn, TensorFlow)
        - Knowledge of statistical analysis and data visualization
        - Experience with SQL and database management
        - 3+ years of experience in data science role
        
        Responsibilities:
        - Develop predictive models for business insights
        - Analyze large datasets to identify trends
        - Collaborate with engineering teams on model deployment
        - Present findings to stakeholders
        """

    @pytest.fixture
    def sample_cv_data(self):
        """Sample CV data for testing."""
        return StructuredCV(
            id="llm-integration-test",
            metadata={
                "name": "Alice Johnson",
                "email": "alice.johnson@example.com",
                "phone": "+1-555-0789"
            },
            sections=[
                CVSection(
                    name="Key Qualifications",
                    items=[
                        CVItem(id="skill_1", content="Python Programming"),
                        CVItem(id="skill_2", content="Data Analysis"),
                        CVItem(id="skill_3", content="Statistical Modeling")
                    ]
                ),
                CVSection(
                    name="Professional Experience",
                    items=[
                        CVItem(id="exp_1", content="Data Analyst at DataCorp (2021-2023)"),
                        CVItem(id="exp_2", content="Built predictive models for customer segmentation")
                    ]
                )
            ]
        )

    @pytest.mark.asyncio
    async def test_parser_agent_llm_integration(self, mock_llm_service, sample_job_description):
        """Test ParserAgent integration with LLM service."""
        # Configure LLM response for job parsing
        mock_llm_service.generate_structured_content_async.return_value = {
            "skills": ["Python", "Machine Learning", "SQL", "TensorFlow"],
            "requirements": ["3+ years experience", "Data science background"],
            "responsibilities": ["Develop models", "Analyze datasets"]
        }
        
        # Create parser agent with mock LLM
        parser = ParserAgent(llm_service=mock_llm_service)
        
        # Test job description parsing
        result = await parser.parse_job_description(sample_job_description)
        
        # Verify LLM was called
        mock_llm_service.generate_structured_content_async.assert_called_once()
        
        # Verify call arguments contain job description
        call_args = mock_llm_service.generate_structured_content_async.call_args
        assert sample_job_description in call_args[0][0]  # First positional arg (prompt)
        
        # Verify result structure
        assert "skills" in result
        assert "requirements" in result
        assert len(result["skills"]) > 0

    @pytest.mark.asyncio
    async def test_research_agent_llm_integration(self, mock_llm_service, sample_job_description, sample_cv_data):
        """Test ResearchAgent integration with LLM service."""
        # Configure different LLM responses for different research types
        def mock_llm_response(prompt, **kwargs):
            if "industry trends" in prompt.lower():
                return "AI and ML adoption is growing rapidly in enterprise environments"
            elif "skill requirements" in prompt.lower():
                return "Python, TensorFlow, SQL, and cloud platforms are in high demand"
            elif "company culture" in prompt.lower():
                return "Data-driven decision making and collaborative environment"
            else:
                return "General research response"
        
        mock_llm_service.generate_content_async.side_effect = mock_llm_response
        
        # Create research agent with mock LLM
        research_agent = ResearchAgent(llm_service=mock_llm_service)
        
        # Create agent state
        state = AgentState(
            job_description_data=JobDescriptionData(raw_text=sample_job_description),
            structured_cv=sample_cv_data,
            error_messages=[],
            processing_queue=[],
            research_data={},
            content_data={},
            quality_scores={},
            output_data={}
        )
        
        # Test research execution
        result = await research_agent.run_async(state)
        
        # Verify LLM was called multiple times for different research areas
        assert mock_llm_service.generate_content_async.call_count >= 3
        
        # Verify research data was populated
        assert "research_data" in result
        research_data = result["research_data"]
        assert "industry_trends" in research_data
        assert "skill_requirements" in research_data
        assert "company_culture" in research_data

    @pytest.mark.asyncio
    async def test_content_writer_llm_integration(self, mock_llm_service, sample_cv_data):
        """Test EnhancedContentWriterAgent integration with LLM service."""
        # Configure LLM response for content enhancement
        mock_llm_service.generate_content_async.return_value = """
        Enhanced content with specific achievements:
        - Developed machine learning models that improved prediction accuracy by 25%
        - Analyzed datasets containing over 1 million records using Python and SQL
        - Collaborated with cross-functional teams to deploy models in production
        """
        
        # Create content writer with mock LLM
        content_writer = EnhancedContentWriterAgent(llm_service=mock_llm_service)
        
        # Create state with research data
        state = AgentState(
            job_description_data=JobDescriptionData(raw_text="Data Scientist role"),
            structured_cv=sample_cv_data,
            error_messages=[],
            processing_queue=["exp_1"],  # Process first experience item
            research_data={
                "industry_trends": "AI/ML growth",
                "skill_requirements": "Python, ML frameworks"
            },
            content_data={},
            quality_scores={},
            output_data={}
        )
        
        # Test content enhancement
        result = await content_writer.run_async(state)
        
        # Verify LLM was called for content enhancement
        mock_llm_service.generate_content_async.assert_called()
        
        # Verify enhanced content was generated
        assert "content_data" in result
        assert "enhanced_items" in result["content_data"]

    @pytest.mark.asyncio
    async def test_quality_assurance_llm_integration(self, mock_llm_service, sample_cv_data):
        """Test QualityAssuranceAgent integration with LLM service."""
        # Configure LLM responses for quality checks
        def mock_qa_response(prompt, **kwargs):
            if "content relevance" in prompt.lower():
                return "Score: 85. The content is highly relevant to the job requirements."
            elif "skill alignment" in prompt.lower():
                return "Score: 90. Skills align well with job requirements."
            elif "experience match" in prompt.lower():
                return "Score: 80. Experience matches job level expectations."
            else:
                return "Score: 85. Good overall quality."
        
        mock_llm_service.generate_content_async.side_effect = mock_qa_response
        
        # Create QA agent with mock LLM
        qa_agent = QualityAssuranceAgent(llm_service=mock_llm_service)
        
        # Create state with enhanced content
        state = AgentState(
            job_description_data=JobDescriptionData(raw_text="Data Scientist role"),
            structured_cv=sample_cv_data,
            error_messages=[],
            processing_queue=[],
            research_data={},
            content_data={
                "enhanced_items": {
                    "exp_1": "Enhanced experience content",
                    "skill_1": "Enhanced skill content"
                }
            },
            quality_scores={},
            output_data={}
        )
        
        # Test quality assessment
        result = await qa_agent.run_async(state)
        
        # Verify LLM was called for quality checks
        assert mock_llm_service.generate_content_async.call_count >= 3
        
        # Verify quality scores were generated
        assert "quality_scores" in result
        quality_scores = result["quality_scores"]
        assert "overall_score" in quality_scores
        assert quality_scores["overall_score"] > 0

    @pytest.mark.asyncio
    async def test_llm_error_handling_integration(self, mock_llm_service, sample_job_description):
        """Test LLM service error handling across agents."""
        # Configure LLM to raise an exception
        mock_llm_service.generate_content_async.side_effect = Exception("LLM service unavailable")
        
        # Create parser agent with failing LLM
        parser = ParserAgent(llm_service=mock_llm_service)
        
        # Test error handling
        with pytest.raises(Exception) as exc_info:
            await parser.parse_job_description(sample_job_description)
        
        assert "LLM service unavailable" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_llm_retry_mechanism_integration(self, mock_llm_service, sample_job_description):
        """Test LLM service retry mechanism integration."""
        # Configure LLM to fail first call, succeed on second
        call_count = 0
        
        def mock_llm_with_retry(prompt, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Temporary LLM failure")
            return {"skills": ["Python"], "requirements": ["Experience"]}
        
        mock_llm_service.generate_structured_content_async.side_effect = mock_llm_with_retry
        
        # Create parser with retry-enabled LLM
        parser = ParserAgent(llm_service=mock_llm_service)
        
        # Mock the retry mechanism in the parser
        with patch.object(parser, '_retry_llm_call') as mock_retry:
            mock_retry.return_value = {"skills": ["Python"], "requirements": ["Experience"]}
            
            result = await parser.parse_job_description(sample_job_description)
            
            # Verify retry was attempted
            mock_retry.assert_called_once()
            assert "skills" in result

    @pytest.mark.asyncio
    async def test_llm_response_validation_integration(self, mock_llm_service, sample_job_description):
        """Test LLM response validation across agents."""
        # Configure LLM to return invalid JSON
        mock_llm_service.generate_structured_content_async.return_value = "Invalid JSON response"
        
        # Create parser agent
        parser = ParserAgent(llm_service=mock_llm_service)
        
        # Test response validation
        with pytest.raises((ValueError, TypeError, json.JSONDecodeError)):
            await parser.parse_job_description(sample_job_description)

    @pytest.mark.asyncio
    async def test_llm_prompt_construction_integration(self, mock_llm_service, sample_job_description, sample_cv_data):
        """Test LLM prompt construction across different agents."""
        # Create research agent
        research_agent = ResearchAgent(llm_service=mock_llm_service)
        
        # Create state
        state = AgentState(
            job_description_data=JobDescriptionData(raw_text=sample_job_description),
            structured_cv=sample_cv_data,
            error_messages=[],
            processing_queue=[],
            research_data={},
            content_data={},
            quality_scores={},
            output_data={}
        )
        
        # Execute research
        await research_agent.run_async(state)
        
        # Verify LLM was called with properly constructed prompts
        assert mock_llm_service.generate_content_async.called
        
        # Check that prompts contain relevant context
        call_args_list = mock_llm_service.generate_content_async.call_args_list
        for call_args in call_args_list:
            prompt = call_args[0][0]  # First positional argument
            assert isinstance(prompt, str)
            assert len(prompt) > 0
            # Verify prompt contains job description context
            assert any(keyword in prompt.lower() for keyword in ["data scientist", "python", "machine learning"])

    @pytest.mark.asyncio
    async def test_llm_concurrent_calls_integration(self, mock_llm_service, sample_cv_data):
        """Test concurrent LLM calls from multiple agents."""
        # Configure LLM with delay to simulate concurrent calls
        async def delayed_response(prompt, **kwargs):
            await asyncio.sleep(0.1)
            return "Delayed LLM response"
        
        mock_llm_service.generate_content_async.side_effect = delayed_response
        
        # Create multiple agents
        research_agent = ResearchAgent(llm_service=mock_llm_service)
        content_writer = EnhancedContentWriterAgent(llm_service=mock_llm_service)
        
        # Create states
        research_state = AgentState(
            job_description_data=JobDescriptionData(raw_text="Job description"),
            structured_cv=sample_cv_data,
            error_messages=[],
            processing_queue=[],
            research_data={},
            content_data={},
            quality_scores={},
            output_data={}
        )
        
        content_state = AgentState(
            job_description_data=JobDescriptionData(raw_text="Job description"),
            structured_cv=sample_cv_data,
            error_messages=[],
            processing_queue=["skill_1"],
            research_data={"industry_trends": "AI growth"},
            content_data={},
            quality_scores={},
            output_data={}
        )
        
        # Execute agents concurrently
        start_time = asyncio.get_event_loop().time()
        results = await asyncio.gather(
            research_agent.run_async(research_state),
            content_writer.run_async(content_state),
            return_exceptions=True
        )
        end_time = asyncio.get_event_loop().time()
        
        # Verify both agents completed
        assert len(results) == 2
        assert all(not isinstance(result, Exception) for result in results)
        
        # Verify concurrent execution (should be faster than sequential)
        assert end_time - start_time < 0.3  # Less than 2 * 0.1 + overhead

    def test_llm_service_configuration_integration(self, mock_llm_service):
        """Test LLM service configuration across agents."""
        # Test that agents can be created with different LLM configurations
        agents = [
            ParserAgent(llm_service=mock_llm_service),
            ResearchAgent(llm_service=mock_llm_service),
            EnhancedContentWriterAgent(llm_service=mock_llm_service),
            QualityAssuranceAgent(llm_service=mock_llm_service)
        ]
        
        # Verify all agents have the same LLM service instance
        for agent in agents:
            assert agent.llm_service is mock_llm_service
        
        # Verify agents can work without LLM service (fallback mode)
        fallback_agents = [
            ParserAgent(),
            ResearchAgent(),
            EnhancedContentWriterAgent(),
            QualityAssuranceAgent()
        ]
        
        for agent in fallback_agents:
            assert agent.llm_service is None or hasattr(agent.llm_service, 'generate_content_async')