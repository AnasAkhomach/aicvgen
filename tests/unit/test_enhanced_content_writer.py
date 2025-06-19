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
from src.models.data_models import StructuredCV, Section, Subsection, Item, ItemStatus, ContentType, JobDescriptionData
from src.orchestration.state import AgentState
from src.services.llm_service import LLMResponse


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
        agent = EnhancedContentWriterAgent(
            name="TestContentWriter",
            description="Test content writer agent",
        )
        # Inject the mock LLM service
        agent.llm_service = mock_llm_service
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
                                id=UUID("00000000-0000-0000-0000-000000000001"),  # exp_1 equivalent
                                content="Software Engineer at TechCorp",
                                status=ItemStatus.INITIAL,
                            ),
                            Item(
                                id=UUID("00000000-0000-0000-0000-000000000002"),  # exp_2 equivalent
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
                        id=UUID("00000000-0000-0000-0000-000000000003"),  # skill_1 equivalent
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
        return AgentState(
            job_description="Senior Software Engineer position",
            job_description_data=JobDescriptionData.model_validate(sample_job_data),
            structured_cv=sample_structured_cv,
            research_findings=sample_research_findings,
            current_item_id="00000000-0000-0000-0000-000000000001",
            error_messages=[],
        )

    def test_enhanced_content_writer_initialization(self, mock_llm_service):
        """Test EnhancedContentWriterAgent initialization."""
        agent = EnhancedContentWriterAgent(
            name="TestContentWriter",
            description="Test description",
        )
        # Inject the mock LLM service after initialization
        agent.llm_service = mock_llm_service

        assert agent.name == "TestContentWriter"
        assert agent.description == "Test description"
        assert agent.llm_service == mock_llm_service
        assert hasattr(agent, "input_schema")
        assert hasattr(agent, "output_schema")

    def test_enhanced_content_writer_initialization_without_llm(self):
        """Test EnhancedContentWriterAgent initialization without LLM service."""
        agent = EnhancedContentWriterAgent(
            name="TestContentWriter", description="Test description"
        )

        assert agent.name == "TestContentWriter"
        assert agent.llm_service is not None  # Should create default LLM service

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
        # Set up the execution context with current_item_id
        execution_context.item_id = "00000000-0000-0000-0000-000000000001"
        
        # Create input data matching the current implementation
        input_data = {
            "structured_cv": sample_structured_cv,
            "job_description_data": JobDescriptionData.model_validate(sample_job_data),
            "research_findings": sample_research_findings,
            "current_item_id": "00000000-0000-0000-0000-000000000001",
        }

        # Mock the LLM service response
        mock_response = LLMResponse(
            success=True,
            content="Enhanced: Software Engineer at TechCorp with 5+ years experience",
            metadata={}
        )
        content_writer_agent.llm_service.generate_content.return_value = mock_response

        result = await content_writer_agent.run_async(input_data, execution_context)

        assert isinstance(result, AgentResult)
        assert result.success is True
        assert "structured_cv" in result.output_data

    @pytest.mark.asyncio
    async def test_run_async_missing_structured_cv(
        self, content_writer_agent, execution_context
    ):
        """Test run_async with missing structured CV."""
        execution_context.item_id = "00000000-0000-0000-0000-000000000001"
        input_data = {
            "job_description_data": JobDescriptionData(raw_text="Engineer position. Test description.", title="Engineer", company="TestCorp", description="Test"),
            "research_findings": {},
            "current_item_id": "00000000-0000-0000-0000-000000000001"
        }

        result = await content_writer_agent.run_async(input_data, execution_context)
        assert isinstance(result, AgentResult)
        assert result.success is False
        assert "structured_cv" in result.error_message

    @pytest.mark.asyncio
    async def test_run_async_missing_current_item_id(
        self, content_writer_agent, sample_structured_cv, execution_context
    ):
        """Test run_async with missing current_item_id."""
        input_data = {
            "structured_cv": sample_structured_cv,
            "job_description_data": JobDescriptionData(raw_text="Engineer position. Test description.", title="Engineer", company="TestCorp", description="Test"),
            "research_findings": {}
        }

        result = await content_writer_agent.run_async(input_data, execution_context)
        assert isinstance(result, AgentResult)
        assert result.success is False
        assert "current_item_id" in result.error_message

    @pytest.mark.asyncio
    async def test_run_as_node_success(self, content_writer_agent, sample_agent_state):
        """Test successful run_as_node execution."""
        # Mock the LLM service response
        mock_response = LLMResponse(
            success=True,
            content="Enhanced: Software Engineer at TechCorp with 5+ years experience",
            metadata={}
        )
        content_writer_agent.llm_service.generate_content.return_value = mock_response

        result_dict = await content_writer_agent.run_as_node(sample_agent_state)

        assert isinstance(result_dict, dict)
        assert "structured_cv" in result_dict
        assert "error_messages" not in result_dict

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

        result_dict = await content_writer_agent.run_as_node(state)

        assert isinstance(result_dict, dict)
        assert "error_messages" in result_dict
        assert len(result_dict["error_messages"]) >= 1
        assert any("current_item_id" in error for error in result_dict["error_messages"])

    @pytest.mark.asyncio
    async def test_run_as_node_processing_error(
        self, content_writer_agent, sample_agent_state
    ):
        """Test run_as_node with processing error."""
        # Mock the LLM service to raise an exception
        content_writer_agent.llm_service.generate_content.side_effect = Exception("LLM service failed")

        result_dict = await content_writer_agent.run_as_node(sample_agent_state)

        assert isinstance(result_dict, dict)
        assert "error_messages" in result_dict
        assert any("LLM service failed" in msg for msg in result_dict["error_messages"])

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
            "company": "TechCorp"
        }

        # Mock the LLM service response
        mock_response = LLMResponse(
            success=True,
            content="Enhanced: Software Engineer at TechCorp with 5+ years experience",
            metadata={}
        )
        content_writer_agent.llm_service.generate_content.return_value = mock_response

        result = await content_writer_agent._process_single_item(
            sample_structured_cv.model_dump(), job_data, item_id, sample_research_findings
        )

        assert isinstance(result, AgentResult)
        assert result.success is True
        assert "structured_cv" in result.output_data
        
        # Verify the item was updated
        updated_cv_data = result.output_data["structured_cv"]
        updated_cv = StructuredCV.model_validate(updated_cv_data)
        processed_item, _, _ = updated_cv.find_item_by_id(item_id)
        
        assert processed_item is not None
        assert processed_item.status == ItemStatus.GENERATED
        assert "Enhanced:" in processed_item.content

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
            sample_structured_cv.model_dump(), sample_job_data, item_id, sample_research_findings
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
            "content": "Software Engineer at TechCorp"
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
        # Mock the LLM service response
        mock_response = LLMResponse(
            success=True,
            content="1. Python\n2. Django\n3. AWS\n4. Docker\n5. Kubernetes\n6. PostgreSQL\n7. Redis\n8. Git\n9. Linux\n10. Agile",
            metadata={}
        )
        content_writer_agent.llm_service.generate_content.return_value = mock_response

        # Create job description from available data
        job_description = f"{sample_job_data['title']} at {sample_job_data['company']}"
        result = await content_writer_agent.generate_big_10_skills(
            job_description, sample_research_findings.get("my_talents", "")
        )

        assert result["success"] is True
        assert isinstance(result["skills"], list)
        assert len(result["skills"]) == 10
        # Check that we got some skills
        assert all(isinstance(skill, str) and len(skill) > 0 for skill in result["skills"])
        content_writer_agent.llm_service.generate_content.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_big_10_skills_llm_error(
        self,
        content_writer_agent,
        sample_job_data,
        sample_research_findings,
    ):
        """Test Big 10 skills generation with LLM error."""
        # Mock LLM to raise an exception
        content_writer_agent.llm_service.generate_content.side_effect = Exception("LLM service error")

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

        skills = content_writer_agent._parse_big_10_skills(response)

        assert isinstance(skills, list)
        assert len(skills) == 10
        assert "Python Programming" in skills
        assert "Web Development" in skills
        assert "Database Design" in skills

    def test_build_experience_prompt(
        self, content_writer_agent, sample_structured_cv, sample_job_data
    ):
        """Test building experience prompt."""
        template = "Experience template: {{Target Skills}} {{batched_structured_output}}"
        content_item = {
            "content_type": "EXPERIENCE",
            "content": "Software Engineer at TechCorp",
            "role_info": {
                "company": "TechCorp",
                "position": "Software Engineer",
                "duration": "2020-2023"
            }
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
        # Mock the LLM service to return success
        with patch.object(content_writer_agent.llm_service, "generate_content") as mock_llm:
            mock_llm.return_value = LLMResponse(
                success=True,
                content="Enhanced content for the item",
                metadata={"model": "test-model"}
            )
            
            with caplog.at_level(logging.INFO):
                result = await content_writer_agent.run_as_node(sample_agent_state)

        # Check that execution was logged
        assert any("processing item" in record.message for record in caplog.records)
        assert isinstance(result, dict)
        assert "structured_cv" in result
