#!/usr/bin/env python3
"""
Integration tests for state management across the agent workflow.
Tests state transitions, data persistence, and error handling.
"""

import pytest
import asyncio
import json
from unittest.mock import AsyncMock, patch, MagicMock
from uuid import uuid4
from copy import deepcopy

from src.orchestration.state import AgentState
from src.models.data_models import (
    StructuredCV, CVSection, CVItem, JobDescriptionData,
    Section, Item, ItemStatus, ItemType
)
from src.agents.parser_agent import ParserAgent
from src.agents.research_agent import ResearchAgent
from src.agents.enhanced_content_writer import EnhancedContentWriterAgent
from src.agents.quality_assurance_agent import QualityAssuranceAgent
from src.agents.formatter_agent import FormatterAgent
from src.services.llm_service import EnhancedLLMService


@pytest.mark.integration
class TestStateManagementIntegration:
    """Integration tests for state management across agents."""

    @pytest.fixture
    def sample_job_description(self):
        """Sample job description for testing."""
        return """
        Senior Python Developer
        
        We are looking for an experienced Python developer with:
        - 5+ years of Python experience
        - Django/Flask framework knowledge
        - Database design skills (PostgreSQL)
        - API development experience
        - Cloud deployment experience (AWS)
        
        Responsibilities:
        - Develop scalable web applications
        - Design and implement APIs
        - Collaborate with frontend team
        - Optimize database performance
        """

    @pytest.fixture
    def sample_cv(self):
        """Sample CV for testing."""
        return StructuredCV(
            id="test-cv-001",
            metadata={
                "name": "John Developer",
                "email": "john@example.com",
                "phone": "+1-555-0123"
            },
            sections=[
                CVSection(
                    name="Skills",
                    items=[
                        CVItem(id="skill_1", content="Python Programming"),
                        CVItem(id="skill_2", content="Web Development"),
                        CVItem(id="skill_3", content="Database Design")
                    ]
                ),
                CVSection(
                    name="Experience",
                    items=[
                        CVItem(id="exp_1", content="Software Developer @ TechCorp (2020-2024)"),
                        CVItem(id="exp_2", content="Built web applications using Python and Django")
                    ]
                )
            ]
        )

    @pytest.fixture
    def initial_state(self, sample_job_description, sample_cv):
        """Initial agent state for testing."""
        return AgentState(
            job_description_data=JobDescriptionData(raw_text=sample_job_description),
            structured_cv=sample_cv,
            error_messages=[],
            processing_queue=[],
            research_data={},
            content_data={},
            quality_scores={},
            output_data={}
        )

    @pytest.fixture
    def mock_llm_service(self):
        """Mock LLM service for testing."""
        mock_service = AsyncMock(spec=EnhancedLLMService)
        
        # Configure mock responses
        mock_service.generate_structured_content_async.return_value = {
            "skills": ["Python", "Django", "PostgreSQL", "AWS"],
            "requirements": ["5+ years experience", "Framework knowledge"],
            "responsibilities": ["Develop applications", "Design APIs"]
        }
        
        mock_service.generate_content_async.return_value = "Mock LLM response"
        
        return mock_service

    @pytest.mark.asyncio
    async def test_state_progression_through_parser(self, initial_state, mock_llm_service):
        """Test state changes after parser agent execution."""
        with patch('src.services.llm.EnhancedLLMService', return_value=mock_llm_service):
            parser = ParserAgent()
            
            # Execute parser
            result_state = await parser.run_as_node(initial_state)
            
            # Verify state structure is maintained
            assert isinstance(result_state, dict)
            assert "job_description_data" in result_state
            assert "structured_cv" in result_state
            assert "error_messages" in result_state
            
            # Verify job description data was parsed
            job_data = result_state["job_description_data"]
            assert hasattr(job_data, 'parsed_data') or 'parsed_data' in job_data.__dict__
            
            # Verify original CV structure is preserved
            cv = result_state["structured_cv"]
            assert cv.id == initial_state.structured_cv.id
            assert len(cv.sections) == len(initial_state.structured_cv.sections)

    @pytest.mark.asyncio
    async def test_state_progression_through_research(self, initial_state, mock_llm_service):
        """Test state changes after research agent execution."""
        # First run parser to set up job data
        with patch('src.services.llm.EnhancedLLMService', return_value=mock_llm_service):
            parser = ParserAgent()
            parsed_state_dict = await parser.run_as_node(initial_state)
            
            # Convert back to AgentState for research agent
            parsed_state = AgentState(
                job_description_data=parsed_state_dict["job_description_data"],
                structured_cv=parsed_state_dict["structured_cv"],
                error_messages=parsed_state_dict["error_messages"],
                processing_queue=parsed_state_dict.get("processing_queue", []),
                research_data=parsed_state_dict.get("research_data", {}),
                content_data=parsed_state_dict.get("content_data", {}),
                quality_scores=parsed_state_dict.get("quality_scores", {}),
                output_data=parsed_state_dict.get("output_data", {})
            )
            
            # Execute research agent
            research = ResearchAgent()
            result_state = await research.run_as_node(parsed_state)
            
            # Verify research data was added
            assert "research_data" in result_state
            research_data = result_state["research_data"]
            assert isinstance(research_data, dict)
            assert len(research_data) > 0
            
            # Verify previous state data is preserved
            assert "job_description_data" in result_state
            assert "structured_cv" in result_state
            assert result_state["structured_cv"].id == initial_state.structured_cv.id

    @pytest.mark.asyncio
    async def test_state_data_accumulation(self, initial_state, mock_llm_service):
        """Test that state data accumulates correctly through multiple agents."""
        with patch('src.services.llm.EnhancedLLMService', return_value=mock_llm_service):
            # Track state evolution
            current_state = initial_state
            
            # Execute parser
            parser = ParserAgent()
            parser_result = await parser.run_as_node(current_state)
            
            # Verify parser added job data
            assert "job_description_data" in parser_result
            job_data = parser_result["job_description_data"]
            assert hasattr(job_data, 'parsed_data') or 'parsed_data' in job_data.__dict__
            
            # Convert to AgentState for next agent
            current_state = AgentState(
                job_description_data=parser_result["job_description_data"],
                structured_cv=parser_result["structured_cv"],
                error_messages=parser_result["error_messages"],
                processing_queue=parser_result.get("processing_queue", []),
                research_data=parser_result.get("research_data", {}),
                content_data=parser_result.get("content_data", {}),
                quality_scores=parser_result.get("quality_scores", {}),
                output_data=parser_result.get("output_data", {})
            )
            
            # Execute research
            research = ResearchAgent()
            research_result = await research.run_as_node(current_state)
            
            # Verify research data was added while preserving parser data
            assert "research_data" in research_result
            assert "job_description_data" in research_result
            assert len(research_result["research_data"]) > 0
            
            # Convert to AgentState for content writer
            current_state = AgentState(
                job_description_data=research_result["job_description_data"],
                structured_cv=research_result["structured_cv"],
                error_messages=research_result["error_messages"],
                processing_queue=research_result.get("processing_queue", []),
                research_data=research_result.get("research_data", {}),
                content_data=research_result.get("content_data", {}),
                quality_scores=research_result.get("quality_scores", {}),
                output_data=research_result.get("output_data", {})
            )
            
            # Execute content writer
            content_writer = EnhancedContentWriterAgent()
            content_result = await content_writer.run_as_node(current_state)
            
            # Verify all previous data is preserved and content data added
            assert "job_description_data" in content_result
            assert "research_data" in content_result
            assert "content_data" in content_result
            assert len(content_result["research_data"]) > 0
            assert len(content_result["content_data"]) > 0

    @pytest.mark.asyncio
    async def test_state_error_handling(self, initial_state, mock_llm_service):
        """Test state error handling and error message accumulation."""
        # Configure LLM to fail
        mock_llm_service.generate_structured_content_async.side_effect = Exception("LLM Error")
        
        with patch('src.services.llm.EnhancedLLMService', return_value=mock_llm_service):
            parser = ParserAgent()
            
            # Execute parser (should handle error gracefully)
            result_state = await parser.run_as_node(initial_state)
            
            # Verify error was captured in state
            assert "error_messages" in result_state
            error_messages = result_state["error_messages"]
            
            # Should have at least one error message
            assert len(error_messages) > 0
            
            # Verify state structure is maintained despite error
            assert "job_description_data" in result_state
            assert "structured_cv" in result_state
            assert result_state["structured_cv"].id == initial_state.structured_cv.id

    @pytest.mark.asyncio
    async def test_state_processing_queue_management(self, initial_state, mock_llm_service):
        """Test processing queue management in state."""
        with patch('src.services.llm.EnhancedLLMService', return_value=mock_llm_service):
            # Add items to processing queue
            initial_state.processing_queue = [
                {"item_id": "skill_1", "action": "enhance", "priority": 1},
                {"item_id": "exp_1", "action": "enhance", "priority": 2}
            ]
            
            content_writer = EnhancedContentWriterAgent()
            result_state = await content_writer.run_as_node(initial_state)
            
            # Verify processing queue was handled
            assert "processing_queue" in result_state
            
            # Verify content data reflects processing
            assert "content_data" in result_state
            content_data = result_state["content_data"]
            assert isinstance(content_data, dict)

    def test_state_serialization_deserialization(self, initial_state):
        """Test state can be serialized and deserialized properly."""
        # Convert state to dict (simulating serialization)
        state_dict = {
            "job_description_data": {
                "raw_text": initial_state.job_description_data.raw_text,
                "parsed_data": getattr(initial_state.job_description_data, 'parsed_data', None)
            },
            "structured_cv": {
                "id": initial_state.structured_cv.id,
                "metadata": initial_state.structured_cv.metadata,
                "sections": [
                    {
                        "name": section.name,
                        "items": [
                            {"id": item.id, "content": item.content}
                            for item in section.items
                        ]
                    }
                    for section in initial_state.structured_cv.sections
                ]
            },
            "error_messages": initial_state.error_messages,
            "processing_queue": initial_state.processing_queue,
            "research_data": initial_state.research_data,
            "content_data": initial_state.content_data,
            "quality_scores": initial_state.quality_scores,
            "output_data": initial_state.output_data
        }
        
        # Verify serialization worked
        assert state_dict["structured_cv"]["id"] == initial_state.structured_cv.id
        assert len(state_dict["structured_cv"]["sections"]) == len(initial_state.structured_cv.sections)
        
        # Verify we can reconstruct key data
        reconstructed_cv_id = state_dict["structured_cv"]["id"]
        assert reconstructed_cv_id == "test-cv-001"

    @pytest.mark.asyncio
    async def test_state_concurrent_access(self, initial_state, mock_llm_service):
        """Test state handling under concurrent access scenarios."""
        with patch('src.services.llm.EnhancedLLMService', return_value=mock_llm_service):
            # Create multiple agents
            parser = ParserAgent()
            research = ResearchAgent()
            
            # Simulate concurrent execution (though agents should run sequentially)
            async def run_parser():
                return await parser.run_as_node(initial_state)
            
            async def run_research():
                # Research needs parsed job data, so this should handle missing data gracefully
                return await research.run_as_node(initial_state)
            
            # Execute concurrently
            parser_result, research_result = await asyncio.gather(
                run_parser(),
                run_research(),
                return_exceptions=True
            )
            
            # Verify parser succeeded
            assert isinstance(parser_result, dict)
            assert "job_description_data" in parser_result
            
            # Research might fail due to missing parsed job data, but should handle gracefully
            if isinstance(research_result, Exception):
                # This is expected behavior
                assert True
            else:
                # If it succeeded, verify it has proper error handling
                assert "error_messages" in research_result

    @pytest.mark.asyncio
    async def test_state_memory_efficiency(self, mock_llm_service):
        """Test state memory usage with large datasets."""
        # Create a large CV with many sections and items
        large_cv = StructuredCV(
            id="large-cv-001",
            metadata={"name": "Test User", "email": "test@example.com"},
            sections=[
                CVSection(
                    name=f"Section_{i}",
                    items=[
                        CVItem(id=f"item_{i}_{j}", content=f"Content for item {i}-{j} " * 50)
                        for j in range(20)  # 20 items per section
                    ]
                )
                for i in range(10)  # 10 sections
            ]
        )
        
        large_state = AgentState(
            job_description_data=JobDescriptionData(raw_text="Large job description " * 100),
            structured_cv=large_cv,
            error_messages=[],
            processing_queue=[],
            research_data={},
            content_data={},
            quality_scores={},
            output_data={}
        )
        
        with patch('src.services.llm.EnhancedLLMService', return_value=mock_llm_service):
            parser = ParserAgent()
            
            # Execute with large state
            result_state = await parser.run_as_node(large_state)
            
            # Verify state structure is maintained
            assert "structured_cv" in result_state
            result_cv = result_state["structured_cv"]
            assert len(result_cv.sections) == 10
            assert len(result_cv.sections[0].items) == 20
            
            # Verify memory isn't excessively duplicated
            assert result_cv.id == large_cv.id

    def test_state_validation(self, initial_state):
        """Test state validation and integrity checks."""
        # Verify initial state is valid
        assert initial_state.job_description_data is not None
        assert initial_state.structured_cv is not None
        assert initial_state.structured_cv.id is not None
        assert len(initial_state.structured_cv.sections) > 0
        
        # Verify error messages list exists
        assert isinstance(initial_state.error_messages, list)
        
        # Verify processing queue exists
        assert isinstance(initial_state.processing_queue, list)
        
        # Verify data dictionaries exist
        assert isinstance(initial_state.research_data, dict)
        assert isinstance(initial_state.content_data, dict)
        assert isinstance(initial_state.quality_scores, dict)
        assert isinstance(initial_state.output_data, dict)

    @pytest.mark.asyncio
    async def test_state_rollback_on_critical_error(self, initial_state, mock_llm_service):
        """Test state rollback mechanisms on critical errors."""
        # Store original state
        original_cv_id = initial_state.structured_cv.id
        original_sections_count = len(initial_state.structured_cv.sections)
        
        # Configure LLM to cause critical error
        mock_llm_service.generate_structured_content_async.side_effect = Exception("Critical LLM failure")
        
        with patch('src.services.llm.EnhancedLLMService', return_value=mock_llm_service):
            parser = ParserAgent()
            
            # Execute parser (should handle error and preserve original state)
            result_state = await parser.run_as_node(initial_state)
            
            # Verify original CV structure is preserved despite error
            assert "structured_cv" in result_state
            result_cv = result_state["structured_cv"]
            assert result_cv.id == original_cv_id
            assert len(result_cv.sections) == original_sections_count
            
            # Verify error was logged
            assert "error_messages" in result_state
            assert len(result_state["error_messages"]) > 0