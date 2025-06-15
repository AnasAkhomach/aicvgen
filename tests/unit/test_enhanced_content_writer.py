#!/usr/bin/env python3
"""
Unit tests for EnhancedContentWriterAgent.

Tests the enhanced CV content generation functionality including
item processing, skill generation, and LangGraph integration.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any, List

from src.agents.enhanced_content_writer import EnhancedContentWriterAgent
from src.agents.agent_base import AgentExecutionContext, AgentResult
from src.models.data_models import StructuredCV, Section, Item, ItemStatus
from src.models.workflow_models import AgentState
from src.services.llm_service import LLMService


class TestEnhancedContentWriterAgent:
    """Test cases for EnhancedContentWriterAgent."""

    @pytest.fixture
    def mock_llm_service(self):
        """Create a mock LLM service."""
        mock_service = Mock(spec=LLMService)
        mock_service.generate_async = AsyncMock()
        return mock_service

    @pytest.fixture
    def content_writer_agent(self, mock_llm_service):
        """Create an EnhancedContentWriterAgent instance for testing."""
        return EnhancedContentWriterAgent(
            name="TestContentWriter",
            description="Test content writer agent",
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
                name="Professional Experience",
                items=[
                    Item(
                        id="exp_1",
                        content="Software Engineer at TechCorp",
                        status=ItemStatus.PENDING
                    ),
                    Item(
                        id="exp_2",
                        content="Developed web applications",
                        status=ItemStatus.PENDING
                    )
                ]
            ),
            Section(
                name="Technical Skills",
                items=[
                    Item(
                        id="skill_1",
                        content="Python programming",
                        status=ItemStatus.PENDING
                    )
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
                "Strong problem-solving skills"
            ],
            "responsibilities": [
                "Lead development team",
                "Design scalable systems",
                "Mentor junior developers"
            ]
        }

    @pytest.fixture
    def sample_research_findings(self):
        """Create sample research findings for testing."""
        return {
            "industry_trends": [
                "Microservices architecture is trending",
                "Cloud-native development is essential",
                "DevOps practices are standard"
            ],
            "skill_requirements": [
                "Python", "Django", "React", "AWS", "Docker"
            ],
            "company_culture": {
                "values": ["Innovation", "Collaboration", "Excellence"],
                "work_style": "Agile development"
            }
        }

    @pytest.fixture
    def sample_agent_state(self, sample_structured_cv, sample_job_data, sample_research_findings):
        """Create a sample agent state for testing."""
        return AgentState(
            job_description="Senior Software Engineer position",
            job_data=sample_job_data,
            structured_cv=sample_structured_cv,
            research_findings=sample_research_findings,
            current_item_id="exp_1",
            error_messages=[]
        )

    def test_enhanced_content_writer_initialization(self, mock_llm_service):
        """Test EnhancedContentWriterAgent initialization."""
        agent = EnhancedContentWriterAgent(
            name="TestContentWriter",
            description="Test description",
            llm_service=mock_llm_service
        )
        
        assert agent.name == "TestContentWriter"
        assert agent.description == "Test description"
        assert agent.llm_service == mock_llm_service
        assert hasattr(agent, 'input_schema')
        assert hasattr(agent, 'output_schema')

    def test_enhanced_content_writer_initialization_without_llm(self):
        """Test EnhancedContentWriterAgent initialization without LLM service."""
        agent = EnhancedContentWriterAgent(
            name="TestContentWriter",
            description="Test description"
        )
        
        assert agent.name == "TestContentWriter"
        assert agent.llm_service is not None  # Should create default LLM service

    async def test_run_async_success(self, content_writer_agent, sample_structured_cv, sample_job_data, sample_research_findings, execution_context):
        """Test successful run_async execution."""
        content_data = {
            "structured_cv": sample_structured_cv,
            "job_data": sample_job_data,
            "research_findings": sample_research_findings
        }
        
        # Mock the content enhancement methods
        with patch.object(content_writer_agent, '_enhance_experience_items') as mock_enhance_exp, \
             patch.object(content_writer_agent, '_enhance_skills_items') as mock_enhance_skills, \
             patch.object(content_writer_agent, '_enhance_education_items') as mock_enhance_edu:
            
            mock_enhance_exp.return_value = None  # Modifies CV in place
            mock_enhance_skills.return_value = None
            mock_enhance_edu.return_value = None
            
            result = await content_writer_agent.run_async(content_data, execution_context)
            
            assert isinstance(result, AgentResult)
            assert "structured_cv" in result.output
            assert result.confidence_score > 0
            mock_enhance_exp.assert_called_once()
            mock_enhance_skills.assert_called_once()
            mock_enhance_edu.assert_called_once()

    async def test_run_async_missing_structured_cv(self, content_writer_agent, execution_context):
        """Test run_async with missing structured CV."""
        content_data = {
            "job_data": {"title": "Engineer"},
            "research_findings": {}
        }
        
        with pytest.raises(ValueError, match="Missing required structured_cv"):
            await content_writer_agent.run_async(content_data, execution_context)

    async def test_run_async_missing_job_data(self, content_writer_agent, sample_structured_cv, execution_context):
        """Test run_async with missing job data."""
        content_data = {
            "structured_cv": sample_structured_cv,
            "research_findings": {}
        }
        
        with pytest.raises(ValueError, match="Missing required job_data"):
            await content_writer_agent.run_async(content_data, execution_context)

    async def test_run_as_node_success(self, content_writer_agent, sample_agent_state):
        """Test successful run_as_node execution."""
        # Mock the _process_single_item method
        with patch.object(content_writer_agent, '_process_single_item') as mock_process:
            mock_enhanced_cv = sample_agent_state.structured_cv
            mock_process.return_value = mock_enhanced_cv
            
            result_state = await content_writer_agent.run_as_node(sample_agent_state)
            
            assert result_state.structured_cv is not None
            assert len(result_state.error_messages) == 0
            mock_process.assert_called_once()

    async def test_run_as_node_missing_current_item_id(self, content_writer_agent, sample_structured_cv, sample_job_data):
        """Test run_as_node with missing current_item_id."""
        state = AgentState(
            job_description="Test job",
            job_data=sample_job_data,
            structured_cv=sample_structured_cv,
            current_item_id=None,
            error_messages=[]
        )
        
        result_state = await content_writer_agent.run_as_node(state)
        
        assert len(result_state.error_messages) == 1
        assert "Missing current_item_id" in result_state.error_messages[0]

    async def test_run_as_node_processing_error(self, content_writer_agent, sample_agent_state):
        """Test run_as_node with processing error."""
        # Mock the _process_single_item method to raise an exception
        with patch.object(content_writer_agent, '_process_single_item') as mock_process:
            mock_process.side_effect = Exception("Processing failed")
            
            result_state = await content_writer_agent.run_as_node(sample_agent_state)
            
            assert len(result_state.error_messages) == 1
            assert "Error processing item" in result_state.error_messages[0]
            assert "Processing failed" in result_state.error_messages[0]

    async def test_process_single_item_success(self, content_writer_agent, sample_structured_cv, sample_research_findings):
        """Test successful single item processing."""
        item_id = "exp_1"
        
        # Mock the enhancement method
        with patch.object(content_writer_agent, '_enhance_single_item') as mock_enhance:
            mock_enhance.return_value = "Enhanced: Software Engineer at TechCorp with 5+ years experience"
            
            result_cv = await content_writer_agent._process_single_item(
                sample_structured_cv,
                item_id,
                sample_research_findings
            )
            
            # Find the processed item
            processed_item = None
            for section in result_cv.sections:
                for item in section.items:
                    if item.id == item_id:
                        processed_item = item
                        break
            
            assert processed_item is not None
            assert processed_item.status == ItemStatus.GENERATED
            assert "Enhanced:" in processed_item.content
            mock_enhance.assert_called_once()

    async def test_process_single_item_item_not_found(self, content_writer_agent, sample_structured_cv, sample_research_findings):
        """Test single item processing with non-existent item ID."""
        item_id = "nonexistent_item"
        
        with pytest.raises(ValueError, match="Item with ID .* not found"):
            await content_writer_agent._process_single_item(
                sample_structured_cv,
                item_id,
                sample_research_findings
            )

    async def test_enhance_single_item_experience(self, content_writer_agent, mock_llm_service):
        """Test enhancing a single experience item."""
        item = Item(
            id="exp_1",
            content="Software Engineer at TechCorp",
            status=ItemStatus.PENDING
        )
        
        research_findings = {
            "skill_requirements": ["Python", "Django", "AWS"],
            "industry_trends": ["Microservices", "Cloud-native"]
        }
        
        # Mock LLM response
        mock_llm_service.generate_async.return_value = "Enhanced: Senior Software Engineer at TechCorp, leading development of scalable microservices using Python and Django, deployed on AWS cloud infrastructure"
        
        enhanced_content = await content_writer_agent._enhance_single_item(
            item,
            "Professional Experience",
            research_findings
        )
        
        assert "Enhanced:" in enhanced_content
        assert "Python" in enhanced_content or "Django" in enhanced_content
        mock_llm_service.generate_async.assert_called_once()

    async def test_enhance_single_item_skills(self, content_writer_agent, mock_llm_service):
        """Test enhancing a single skills item."""
        item = Item(
            id="skill_1",
            content="Python programming",
            status=ItemStatus.PENDING
        )
        
        research_findings = {
            "skill_requirements": ["Python", "Django", "Flask", "FastAPI"],
            "industry_trends": ["API development", "Microservices"]
        }
        
        # Mock LLM response
        mock_llm_service.generate_async.return_value = "Advanced Python programming with expertise in Django, Flask, and FastAPI frameworks for building scalable web applications and microservices"
        
        enhanced_content = await content_writer_agent._enhance_single_item(
            item,
            "Technical Skills",
            research_findings
        )
        
        assert "Python" in enhanced_content
        assert "Django" in enhanced_content or "Flask" in enhanced_content
        mock_llm_service.generate_async.assert_called_once()

    async def test_enhance_experience_items(self, content_writer_agent, sample_structured_cv, sample_research_findings):
        """Test enhancing all experience items."""
        # Mock the single item enhancement
        with patch.object(content_writer_agent, '_enhance_single_item') as mock_enhance:
            mock_enhance.return_value = "Enhanced experience item"
            
            await content_writer_agent._enhance_experience_items(
                sample_structured_cv,
                sample_research_findings
            )
            
            # Should be called for each experience item
            assert mock_enhance.call_count == 2  # Two experience items in fixture

    async def test_enhance_skills_items(self, content_writer_agent, sample_structured_cv, sample_research_findings):
        """Test enhancing all skills items."""
        # Mock the single item enhancement
        with patch.object(content_writer_agent, '_enhance_single_item') as mock_enhance:
            mock_enhance.return_value = "Enhanced skill item"
            
            await content_writer_agent._enhance_skills_items(
                sample_structured_cv,
                sample_research_findings
            )
            
            # Should be called for each skill item
            assert mock_enhance.call_count == 1  # One skill item in fixture

    async def test_enhance_education_items(self, content_writer_agent, sample_structured_cv, sample_research_findings):
        """Test enhancing all education items."""
        # Add education section to CV
        education_section = Section(
            name="Education",
            items=[
                Item(
                    id="edu_1",
                    content="Computer Science Degree",
                    status=ItemStatus.PENDING
                )
            ]
        )
        sample_structured_cv.sections.append(education_section)
        
        # Mock the single item enhancement
        with patch.object(content_writer_agent, '_enhance_single_item') as mock_enhance:
            mock_enhance.return_value = "Enhanced education item"
            
            await content_writer_agent._enhance_education_items(
                sample_structured_cv,
                sample_research_findings
            )
            
            # Should be called for each education item
            assert mock_enhance.call_count == 1

    async def test_generate_big_10_skills_success(self, content_writer_agent, mock_llm_service, sample_job_data, sample_research_findings):
        """Test successful Big 10 skills generation."""
        # Mock prompt loading
        with patch('src.agents.enhanced_content_writer.load_prompt') as mock_load_prompt:
            mock_load_prompt.return_value = "Generate top 10 skills for: {job_title}"
            
            # Mock LLM response
            mock_llm_service.generate_async.return_value = "1. Python Programming\n2. Web Development\n3. Database Design\n4. API Development\n5. Cloud Computing\n6. DevOps\n7. Agile Methodology\n8. Problem Solving\n9. Team Leadership\n10. Code Review"
            
            skills = await content_writer_agent.generate_big_10_skills(
                sample_job_data,
                sample_research_findings
            )
            
            assert isinstance(skills, list)
            assert len(skills) == 10
            assert "Python Programming" in skills
            assert "Web Development" in skills
            mock_llm_service.generate_async.assert_called_once()

    async def test_generate_big_10_skills_llm_error(self, content_writer_agent, mock_llm_service, sample_job_data, sample_research_findings):
        """Test Big 10 skills generation with LLM error."""
        # Mock prompt loading
        with patch('src.agents.enhanced_content_writer.load_prompt') as mock_load_prompt:
            mock_load_prompt.return_value = "Generate skills prompt"
            
            # Mock LLM to raise an exception
            mock_llm_service.generate_async.side_effect = Exception("LLM service error")
            
            with pytest.raises(Exception, match="LLM service error"):
                await content_writer_agent.generate_big_10_skills(
                    sample_job_data,
                    sample_research_findings
                )

    def test_parse_skills_from_response(self, content_writer_agent):
        """Test parsing skills from LLM response."""
        response = "1. Python Programming\n2. Web Development\n3. Database Design\n4. API Development\n5. Cloud Computing"
        
        skills = content_writer_agent._parse_skills_from_response(response)
        
        assert isinstance(skills, list)
        assert len(skills) == 5
        assert "Python Programming" in skills
        assert "Web Development" in skills
        assert "Database Design" in skills

    def test_parse_skills_from_response_bullet_points(self, content_writer_agent):
        """Test parsing skills from response with bullet points."""
        response = "• Python Programming\n• Web Development\n• Database Design"
        
        skills = content_writer_agent._parse_skills_from_response(response)
        
        assert isinstance(skills, list)
        assert len(skills) == 3
        assert "Python Programming" in skills

    def test_parse_skills_from_response_mixed_format(self, content_writer_agent):
        """Test parsing skills from response with mixed format."""
        response = "Top skills:\n1. Python Programming\n- Web Development\n• Database Design\n2. API Development"
        
        skills = content_writer_agent._parse_skills_from_response(response)
        
        assert isinstance(skills, list)
        assert len(skills) >= 3
        assert "Python Programming" in skills
        assert "Web Development" in skills

    def test_build_enhancement_prompt(self, content_writer_agent):
        """Test building enhancement prompt."""
        item = Item(content="Software Engineer", status=ItemStatus.PENDING)
        section_name = "Professional Experience"
        research_findings = {
            "skill_requirements": ["Python", "Django"],
            "industry_trends": ["Microservices"]
        }
        
        prompt = content_writer_agent._build_enhancement_prompt(
            item,
            section_name,
            research_findings
        )
        
        assert isinstance(prompt, str)
        assert "Software Engineer" in prompt
        assert "Professional Experience" in prompt
        assert "Python" in prompt or "Django" in prompt

    def test_extract_relevant_research(self, content_writer_agent):
        """Test extracting relevant research for section."""
        research_findings = {
            "skill_requirements": ["Python", "Django", "AWS"],
            "industry_trends": ["Microservices", "Cloud-native"],
            "company_culture": {"values": ["Innovation"]}
        }
        
        # Test for experience section
        relevant = content_writer_agent._extract_relevant_research(
            "Professional Experience",
            research_findings
        )
        
        assert "skill_requirements" in relevant
        assert "industry_trends" in relevant
        
        # Test for skills section
        relevant = content_writer_agent._extract_relevant_research(
            "Technical Skills",
            research_findings
        )
        
        assert "skill_requirements" in relevant

    def test_confidence_score_calculation(self, content_writer_agent):
        """Test confidence score calculation."""
        # Test with successful enhancement
        successful_output = {
            "structured_cv": StructuredCV(),
            "items_enhanced": 5,
            "enhancement_quality": "high"
        }
        
        confidence = content_writer_agent.get_confidence_score(successful_output)
        assert confidence > 0.7  # Should be high for successful enhancement
        
        # Test with minimal enhancement
        minimal_output = {
            "structured_cv": StructuredCV(),
            "items_enhanced": 1
        }
        
        confidence = content_writer_agent.get_confidence_score(minimal_output)
        assert confidence < 0.8  # Should be lower for minimal enhancement

    def test_find_item_by_id(self, content_writer_agent, sample_structured_cv):
        """Test finding item by ID."""
        item, section = content_writer_agent._find_item_by_id(sample_structured_cv, "exp_1")
        
        assert item is not None
        assert item.id == "exp_1"
        assert section is not None
        assert section.name == "Professional Experience"

    def test_find_item_by_id_not_found(self, content_writer_agent, sample_structured_cv):
        """Test finding non-existent item by ID."""
        item, section = content_writer_agent._find_item_by_id(sample_structured_cv, "nonexistent")
        
        assert item is None
        assert section is None

    def test_get_section_items_by_name(self, content_writer_agent, sample_structured_cv):
        """Test getting section items by name."""
        items = content_writer_agent._get_section_items_by_name(
            sample_structured_cv,
            "Professional Experience"
        )
        
        assert isinstance(items, list)
        assert len(items) == 2
        assert all(item.status == ItemStatus.PENDING for item in items)

    def test_get_section_items_by_name_not_found(self, content_writer_agent, sample_structured_cv):
        """Test getting items from non-existent section."""
        items = content_writer_agent._get_section_items_by_name(
            sample_structured_cv,
            "Nonexistent Section"
        )
        
        assert isinstance(items, list)
        assert len(items) == 0

    @patch('src.agents.enhanced_content_writer.logger')
    async def test_run_as_node_logs_execution(self, mock_logger, content_writer_agent, sample_agent_state):
        """Test that run_as_node logs execution properly."""
        with patch.object(content_writer_agent, '_process_single_item') as mock_process:
            mock_process.return_value = sample_agent_state.structured_cv
            
            await content_writer_agent.run_as_node(sample_agent_state)
            
            # Verify logging calls
            mock_logger.info.assert_called()
            assert any("Processing item" in str(call) for call in mock_logger.info.call_args_list)

    def test_validate_enhancement_quality(self, content_writer_agent):
        """Test enhancement quality validation."""
        original_content = "Software Engineer"
        enhanced_content = "Senior Software Engineer with 5+ years of experience in Python development, specializing in web applications and microservices architecture"
        
        is_valid = content_writer_agent._validate_enhancement_quality(
            original_content,
            enhanced_content
        )
        
        assert is_valid is True
        
        # Test with poor enhancement (too similar)
        poor_enhancement = "Software Engineer at company"
        
        is_valid = content_writer_agent._validate_enhancement_quality(
            original_content,
            poor_enhancement
        )
        
        assert is_valid is False

    def test_calculate_enhancement_score(self, content_writer_agent):
        """Test enhancement score calculation."""
        original = "Python developer"
        enhanced = "Senior Python developer with expertise in Django, Flask, and FastAPI frameworks, experienced in building scalable web applications"
        
        score = content_writer_agent._calculate_enhancement_score(original, enhanced)
        
        assert isinstance(score, float)
        assert 0 <= score <= 1
        assert score > 0.5  # Should be high for good enhancement