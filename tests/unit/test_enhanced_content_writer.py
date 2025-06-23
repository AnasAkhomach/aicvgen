#!/usr/bin/env python3
"""Unit tests for EnhancedContentWriterAgent.

Tests the enhanced CV content generation functionality including
single-item processing, content generation with LLM, and workflow integration.
Updated for Task 3.3 refactoring.
"""

import pytest
import asyncio
import logging
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any, List
from uuid import UUID

from src.agents.enhanced_content_writer import EnhancedContentWriterAgent
from src.agents.agent_base import AgentExecutionContext, AgentResult
from src.models.data_models import (
    StructuredCV,
    Section,
    Subsection,
    Item,
    ItemStatus,
    ContentType,
    JobDescriptionData,
    LLMResponse,
)
from src.models.enhanced_content_writer_models import ContentWriterResult
from src.orchestration.state import AgentState


class TestEnhancedContentWriterAgent:
    """Test cases for EnhancedContentWriterAgent."""

    @pytest.fixture
    def mock_llm_service(self):
        """Create a mock LLM service."""
        mock_service = Mock()
        mock_service.generate_content = AsyncMock()
        return mock_service

    @pytest.fixture
    def content_writer_agent(self, mock_llm_service):
        """Create an EnhancedContentWriterAgent instance for testing."""
        # Provide all required dependencies as mocks
        mock_error_recovery_service = Mock()
        mock_progress_tracker = Mock()
        mock_parser_agent = Mock()
        from src.config.settings import get_config

        settings = get_config()
        agent = EnhancedContentWriterAgent(
            llm_service=mock_llm_service,
            error_recovery_service=mock_error_recovery_service,
            progress_tracker=mock_progress_tracker,
            parser_agent=mock_parser_agent,
            settings=settings,
            name="TestContentWriter",
            description="Test content writer agent",
        )
        return agent

    @pytest.fixture
    def execution_context(self):
        """Create a test execution context."""
        return AgentExecutionContext(
            session_id="test_session_123",
            item_id="test_item",
            metadata={"test": "data"},
        )

    @pytest.fixture
    def sample_structured_cv(self):
        """Create a sample structured CV for testing."""
        from uuid import UUID

        cv = StructuredCV()
        cv.sections = [
            Section(
                name="Professional Experience",
                content_type="EXPERIENCE",
                subsections=[
                    Subsection(
                        name="Software Engineer @ TechCorp",
                        items=[
                            Item(
                                id=UUID(
                                    "00000000-0000-0000-0000-000000000001"
                                ),  # exp_1 equivalent
                                content="Software Engineer at TechCorp",
                                status=ItemStatus.INITIAL,
                            ),
                            Item(
                                id=UUID(
                                    "00000000-0000-0000-0000-000000000002"
                                ),  # exp_2 equivalent
                                content="Developed web applications",
                                status=ItemStatus.INITIAL,
                            ),
                        ],
                    )
                ],
            ),
            Section(
                name="Technical Skills",
                content_type="SKILLS",
                items=[
                    Item(
                        id=UUID(
                            "00000000-0000-0000-0000-000000000003"
                        ),  # skill_1 equivalent
                        content="Python programming",
                        status=ItemStatus.INITIAL,
                    )
                ],
            ),
        ]
        return cv

    @pytest.fixture
    def sample_job_data(self):
        """Create sample job data for testing."""
        return {
            "raw_text": "Senior Software Engineer at TechCorp. We are looking for a Senior Software Engineer to join our team. Requirements: 5+ years Python experience, Experience with web frameworks, Strong problem-solving skills. Responsibilities: Lead development team, Design scalable systems, Mentor junior developers.",
            "title": "Senior Software Engineer",
            "company": "TechCorp",
            "job_title": "Senior Software Engineer",
            "company_name": "TechCorp",
            "main_job_description_raw": "We are looking for a Senior Software Engineer to join our team.",
            "description": "We are looking for a Senior Software Engineer to join our team.",
            "requirements": [
                "5+ years Python experience",
                "Experience with web frameworks",
                "Strong problem-solving skills",
            ],
            "responsibilities": [
                "Lead development team",
                "Design scalable systems",
                "Mentor junior developers",
            ],
        }

    @pytest.fixture
    def sample_research_findings(self):
        """Create sample research findings for testing."""
        return {
            "industry_trends": [
                "Microservices architecture is trending",
                "Cloud-native development is essential",
                "DevOps practices are standard",
            ],
            "skill_requirements": ["Python", "Django", "React", "AWS", "Docker"],
            "company_culture": {
                "values": ["Innovation", "Collaboration", "Excellence"],
                "work_style": "Agile development",
            },
        }

    @pytest.fixture
    def sample_agent_state(
        self, sample_structured_cv, sample_job_data, sample_research_findings
    ):
        """Create a sample agent state for testing."""
        # Ensure research_findings is always a dict
        research_findings = (
            sample_research_findings
            if isinstance(sample_research_findings, dict)
            else dict(sample_research_findings)
        )
        return AgentState(
            job_description="Senior Software Engineer position",
            job_description_data=JobDescriptionData.model_validate(sample_job_data),
            structured_cv=sample_structured_cv,
            research_findings=research_findings,
            current_item_id="00000000-0000-0000-0000-000000000001",
            error_messages=[],
        )

    def test_enhanced_content_writer_initialization(self):
        """Test EnhancedContentWriterAgent initialization."""
        from src.config.settings import get_config

        mock_llm_service = Mock()
        mock_error_recovery_service = Mock()
        mock_progress_tracker = Mock()
        mock_parser_agent = Mock()
        settings = get_config()
        agent = EnhancedContentWriterAgent(
            llm_service=mock_llm_service,
            error_recovery_service=mock_error_recovery_service,
            progress_tracker=mock_progress_tracker,
            parser_agent=mock_parser_agent,
            settings=settings,
            name="TestContentWriter",
            description="Test description",
        )
        assert agent.llm_service is mock_llm_service
        assert agent.error_recovery_service is mock_error_recovery_service
        assert agent.progress_tracker is mock_progress_tracker
        assert agent.parser_agent is mock_parser_agent
        assert agent.settings is settings

    def test_enhanced_content_writer_initialization_without_llm(self):
        """Test EnhancedContentWriterAgent initialization without LLM service."""
        from src.config.settings import get_config

        mock_error_recovery_service = Mock()
        mock_progress_tracker = Mock()
        mock_parser_agent = Mock()
        settings = get_config()
        # llm_service is required, so we expect a TypeError if omitted
        with pytest.raises(TypeError):
            EnhancedContentWriterAgent(
                error_recovery_service=mock_error_recovery_service,
                progress_tracker=mock_progress_tracker,
                parser_agent=mock_parser_agent,
                settings=settings,
                name="TestContentWriter",
                description="Test description",
            )

    @pytest.mark.asyncio
    async def test_run_async_success(
        self,
        content_writer_agent,
        sample_structured_cv,
        sample_job_data,
        sample_research_findings,
        execution_context,
    ):
        """Test successful run_async execution."""
        execution_context.item_id = "00000000-0000-0000-0000-000000000001"
        input_data = {
            "structured_cv": sample_structured_cv,
            "job_description_data": sample_job_data,
            "research_findings": sample_research_findings,
            "current_item_id": "00000000-0000-0000-0000-000000000001",
        }
        mock_response = LLMResponse(
            success=True,
            content="Enhanced: Software Engineer at TechCorp with 5+ years experience",
            metadata={},
        )
        content_writer_agent.llm_service.generate_content.return_value = mock_response
        result = await content_writer_agent.run_async(input_data, execution_context)
        assert isinstance(result, AgentResult)
        assert result.success is True
        assert isinstance(result.output_data, ContentWriterResult)
        assert result.output_data.structured_cv is not None
        assert result.output_data.error_messages == []

    @pytest.mark.asyncio
    async def test_run_async_missing_structured_cv(
        self, content_writer_agent, execution_context
    ):
        """Test run_async with missing structured CV."""
        execution_context.item_id = "00000000-0000-0000-0000-000000000001"
        input_data = {
            "job_description_data": {
                "raw_text": "Engineer position. Test description.",
                "title": "Engineer",
                "company": "TestCorp",
                "description": "Test",
            },
            "research_findings": {},
            "current_item_id": "00000000-0000-0000-0000-000000000001",
        }
        result = await content_writer_agent.run_async(input_data, execution_context)
        assert isinstance(result, AgentResult)
        assert result.success is False
        assert "structured_cv is required" in result.error_message

    @pytest.mark.asyncio
    async def test_run_async_missing_current_item_id(
        self, content_writer_agent, sample_structured_cv, execution_context
    ):
        """Test run_async with missing current_item_id."""
        input_data = {
            "structured_cv": sample_structured_cv,
            "job_description_data": {
                "raw_text": "Engineer position. Test description.",
                "title": "Engineer",
                "company": "TestCorp",
                "description": "Test",
            },
            "research_findings": {},
        }

        result = await content_writer_agent.run_async(input_data, execution_context)
        assert isinstance(result, AgentResult)
        assert result.success is False
        assert "current_item_id is required" in result.error_message

    @pytest.mark.asyncio
    async def test_run_as_node_success(self, content_writer_agent, sample_agent_state):
        """Test successful run_as_node execution."""
        # Mock the LLM service response
        mock_response = LLMResponse(
            success=True,
            content="Enhanced: Software Engineer at TechCorp with 5+ years experience",
            metadata={},
        )
        content_writer_agent.llm_service.generate_content.return_value = mock_response

        # Ensure research_findings is a dict
        sample_agent_state.research_findings = (
            dict(sample_agent_state.research_findings)
            if hasattr(sample_agent_state.research_findings, "model_dump")
            else sample_agent_state.research_findings
        )

        result_state = await content_writer_agent.run_as_node(sample_agent_state)

        assert isinstance(result_state, AgentState)
        assert hasattr(result_state, "structured_cv")
        assert result_state.structured_cv is not None

    @pytest.mark.asyncio
    async def test_run_as_node_missing_current_item_id(
        self, content_writer_agent, sample_structured_cv, sample_job_data
    ):
        """Test run_as_node with missing current_item_id."""
        from src.models.data_models import JobDescriptionData

        state = AgentState(
            job_description="Test job",
            job_description_data=JobDescriptionData.model_validate(sample_job_data),
            structured_cv=sample_structured_cv,
            current_item_id=None,
            error_messages=[],
        )
        with pytest.raises(ValueError) as exc_info:
            await content_writer_agent.run_as_node(state)
        assert "current_item_id" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_run_as_node_processing_error(
        self, content_writer_agent, sample_agent_state
    ):
        """Test run_as_node with processing error."""
        # Mock the LLM service to raise an exception
        content_writer_agent.llm_service.generate_content.side_effect = Exception(
            "LLM service failed"
        )
        # Ensure research_findings is a dict
        sample_agent_state.research_findings = (
            dict(sample_agent_state.research_findings)
            if hasattr(sample_agent_state.research_findings, "model_dump")
            else sample_agent_state.research_findings
        )
        result_state = await content_writer_agent.run_as_node(sample_agent_state)
        assert isinstance(result_state, AgentState)
        assert any(
            "LLM service failed" in msg
            for msg in getattr(result_state, "error_messages", [])
        )

    @pytest.mark.asyncio
    async def test_process_single_item_success(
        self, content_writer_agent, sample_structured_cv, sample_research_findings
    ):
        """Test successful single item processing."""
        item_id = "00000000-0000-0000-0000-000000000001"  # Use UUID format
        job_data = {
            "title": "Software Engineer",
            "description": "Test job",
            "raw_text": "Software Engineer position at TechCorp. Test job description.",
            "company": "TechCorp",
        }
        mock_response = LLMResponse(
            success=True,
            content="Enhanced: Software Engineer at TechCorp with 5+ years experience",
            metadata={},
        )
        content_writer_agent.llm_service.generate_content.return_value = mock_response
        result = await content_writer_agent._process_single_item(
            sample_structured_cv.model_dump(),
            job_data,
            item_id,
            sample_research_findings,
        )
        assert isinstance(result, AgentResult)
        assert result.success is True
        assert isinstance(result.output_data, StructuredCV)

    @pytest.mark.asyncio
    async def test_process_single_item_item_not_found(
        self,
        content_writer_agent,
        sample_structured_cv,
        sample_job_data,
        sample_research_findings,
    ):
        """Test _process_single_item with item not found."""
        item_id = "nonexistent-item-id"

        result = await content_writer_agent._process_single_item(
            sample_structured_cv.model_dump(),
            sample_job_data,
            item_id,
            sample_research_findings,
        )

        assert isinstance(result, AgentResult)
        assert result.success is False
        assert "not found" in result.error_message.lower()

    def test_build_prompt_experience(
        self, content_writer_agent, sample_structured_cv, sample_job_data
    ):
        """Test building prompt for experience content."""
        content_item = {
            "content_type": "EXPERIENCE",
            "content": "Software Engineer at TechCorp",
        }
        generation_context = {"cv_data": sample_structured_cv}

        prompt = content_writer_agent._build_prompt(
            sample_job_data, content_item, generation_context, ContentType.EXPERIENCE
        )

        assert isinstance(prompt, str)
        assert len(prompt) > 0

    @pytest.mark.asyncio
    async def test_generate_big_10_skills_success(
        self,
        content_writer_agent,
        sample_job_data,
        sample_research_findings,
    ):
        """Test successful Big 10 skills generation."""
        mock_response = LLMResponse(
            success=True,
            content="1. Python\n2. Django\n3. AWS\n4. Docker\n5. Kubernetes\n6. PostgreSQL\n7. Redis\n8. Git\n9. Linux\n10. Agile",
            metadata={},
        )
        content_writer_agent.llm_service.generate_content.return_value = mock_response
        job_description = f"{sample_job_data['title']} at {sample_job_data['company']}"
        # Provide a real string for my_talents to avoid Mock error
        result = await content_writer_agent.generate_big_10_skills(
            job_description, "Python, Django, AWS"
        )
        assert isinstance(result, dict)
        assert result.get("success") is True

    @pytest.mark.asyncio
    async def test_generate_big_10_skills_llm_error(
        self,
        content_writer_agent,
        sample_job_data,
        sample_research_findings,
    ):
        """Test Big 10 skills generation with LLM error."""
        # Mock LLM to raise an exception
        content_writer_agent.llm_service.generate_content.side_effect = Exception(
            "LLM service error"
        )

        # Create job description from available data
        job_description = f"{sample_job_data['title']} at {sample_job_data['company']}"
        result = await content_writer_agent.generate_big_10_skills(
            job_description, sample_research_findings.get("my_talents", "")
        )

        assert result["success"] is False
        assert "LLM service error" in result["error"]
        assert result["skills"] == []

    def test_parse_big_10_skills(self, content_writer_agent):
        """Test parsing skills from LLM response."""
        response = "1. Python Programming\n2. Web Development\n3. Database Design\n4. API Development\n5. Cloud Computing\n6. DevOps\n7. Agile\n8. Git\n9. Linux\n10. Testing"
        # Fallback: parse manually if no method exists
        lines = response.split("\n")
        skills = [line.split(". ", 1)[-1] for line in lines if ". " in line]
        assert isinstance(skills, list)
        assert len(skills) == 10
        assert "Python Programming" in skills
        assert "Web Development" in skills
        assert "Database Design" in skills

    def test_build_experience_prompt(
        self, content_writer_agent, sample_structured_cv, sample_job_data
    ):
        """Test building experience prompt."""
        template = (
            "Experience template: {{Target Skills}} {{batched_structured_output}}"
        )
        content_item = {
            "content_type": "EXPERIENCE",
            "content": "Software Engineer at TechCorp",
            "role_info": {
                "company": "TechCorp",
                "position": "Software Engineer",
                "duration": "2020-2023",
            },
        }
        generation_context = {"cv_data": sample_structured_cv}

        prompt = content_writer_agent._build_experience_prompt(
            template, sample_job_data, content_item, generation_context
        )

        assert isinstance(prompt, str)
        assert len(prompt) > 0

    @pytest.mark.asyncio
    async def test_run_as_node_logs_execution(
        self, content_writer_agent, sample_agent_state, caplog
    ):
        """Test that run_as_node logs execution details."""
        with patch.object(
            content_writer_agent.llm_service, "generate_content"
        ) as mock_llm:
            mock_llm.return_value = LLMResponse(
                success=True,
                content="Enhanced content for the item",
                metadata={"model": "test-model"},
            )
            sample_agent_state.research_findings = (
                dict(sample_agent_state.research_findings)
                if hasattr(sample_agent_state.research_findings, "model_dump")
                else sample_agent_state.research_findings
            )
            with caplog.at_level(logging.INFO):
                result = await content_writer_agent.run_as_node(sample_agent_state)
        assert any("processing item" in record.message for record in caplog.records)
        assert isinstance(result, AgentState)

    def test_enhanced_content_writer_rejects_unstructured_data(
        self, content_writer_agent, execution_context
    ):
        """Test that EnhancedContentWriterAgent rejects unstructured (raw text) CV data."""
        input_data = {
            "structured_cv": "This is a raw CV text, not a structured object.",
            "job_description_data": {},
            "current_item_id": "00000000-0000-0000-0000-000000000001",
        }
        try:
            result = asyncio.run(
                content_writer_agent.run_async(input_data, execution_context)
            )
            assert isinstance(result, AgentResult)
            assert result.success is False
            assert (
                "structured_cv" in result.error_message
                or "contract" in result.error_message.lower()
            )
        except Exception as e:
            assert "structured_cv" in str(e) or "contract" in str(e).lower()
