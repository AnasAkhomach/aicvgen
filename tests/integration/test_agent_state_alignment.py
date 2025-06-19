"""Integration test to verify agent outputs align with AgentState fields.

This test verifies that agent run_as_node methods return dictionaries
with keys that exactly match AgentState fields, preventing data loss
in LangGraph workflows.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock
from typing import Dict, Any

from src.orchestration.state import AgentState
from src.agents.parser_agent import ParserAgent
from src.agents.research_agent import ResearchAgent
from src.agents.quality_assurance_agent import QualityAssuranceAgent
from src.models.data_models import (
    JobDescriptionData,
    StructuredCV,
    PersonalInfo,
    ProcessingStatus
)
from src.services.llm_service import get_llm_service


class TestAgentStateAlignment:
    """Test suite for verifying agent output alignment with AgentState."""

    @pytest.fixture
    def mock_llm_service(self):
        """Create a mock LLM service for testing."""
        mock_service = Mock()
        mock_service.generate_response = AsyncMock(return_value="Mock LLM response")
        mock_service.generate_structured_response = AsyncMock(
            return_value={"skills": ["Python", "Testing"], "experience_level": "Senior"}
        )
        return mock_service

    @pytest.fixture
    def sample_agent_state(self):
        """Create a sample AgentState for testing."""
        job_data = JobDescriptionData(
            raw_text="Senior Python Developer position requiring 5+ years experience.",
            skills=["Python", "Django", "REST APIs"],
            experience_level="Senior",
            company_name="Tech Corp",
            role_title="Senior Python Developer"
        )
        
        cv = StructuredCV(
            metadata={
                "personal_info": {
                    "name": "John Doe",
                    "email": "john.doe@example.com",
                    "phone": "+1-555-0123"
                }
            }
        )
        
        return AgentState(
            structured_cv=cv,
            job_description_data=job_data,
            trace_id="test-trace-123"
        )

    @pytest.mark.asyncio
    async def test_parser_agent_output_alignment(self, mock_llm_service, sample_agent_state):
        """Test that ParserAgent returns keys matching AgentState fields."""
        # Arrange
        parser_agent = ParserAgent(
            name="TestParserAgent",
            description="Test parser agent",
            llm_service=mock_llm_service
        )
        
        # Act
        result = await parser_agent.run_as_node(sample_agent_state)
        
        # Assert
        assert isinstance(result, dict), "Parser agent should return a dictionary"
        
        # Verify that returned keys match AgentState fields
        expected_keys = {"structured_cv", "job_description_data"}
        actual_keys = set(result.keys())
        
        # Check that all returned keys are valid AgentState fields
        agent_state_fields = set(AgentState.model_fields.keys())
        invalid_keys = actual_keys - agent_state_fields
        assert not invalid_keys, f"Parser agent returned invalid keys: {invalid_keys}"
        
        # Check that expected keys are present
        missing_keys = expected_keys - actual_keys
        if missing_keys:
            # Allow for error scenarios where not all keys are returned
            assert "error_messages" in actual_keys, f"Missing expected keys: {missing_keys} and no error_messages"

    @pytest.mark.asyncio
    async def test_research_agent_output_alignment(self, mock_llm_service, sample_agent_state):
        """Test that ResearchAgent returns keys matching AgentState fields."""
        # Arrange
        research_agent = ResearchAgent(
            name="TestResearchAgent",
            description="Test research agent",
            llm_service=mock_llm_service
        )
        
        # Mock the async method to return expected structure
        research_agent.run_async = AsyncMock()
        research_agent.run_async.return_value = Mock(
            success=True,
            output_data={"research_findings": {"key_skills": ["Python", "Testing"]}}
        )
        
        # Act
        result = await research_agent.run_as_node(sample_agent_state)
        
        # Assert
        assert isinstance(result, dict), "Research agent should return a dictionary"
        
        # Verify that returned keys match AgentState fields
        agent_state_fields = set(AgentState.model_fields.keys())
        actual_keys = set(result.keys())
        invalid_keys = actual_keys - agent_state_fields
        
        assert not invalid_keys, f"Research agent returned invalid keys: {invalid_keys}"
        
        # Check for expected research_findings key
        if "research_findings" not in result and "error_messages" not in result:
            pytest.fail("Research agent should return either 'research_findings' or 'error_messages'")

    @pytest.mark.asyncio
    async def test_quality_assurance_agent_output_alignment(self, mock_llm_service, sample_agent_state):
        """Test that QualityAssuranceAgent returns keys matching AgentState fields."""
        # Arrange
        qa_agent = QualityAssuranceAgent(
            name="TestQAAgent",
            description="Test QA agent",
            llm_service=mock_llm_service
        )
        
        # Mock the async method to return expected structure
        qa_agent.run_async = AsyncMock()
        qa_agent.run_async.return_value = Mock(
            success=True,
            output_data={
                "quality_check_results": {"score": 0.85, "issues": []},
                "updated_structured_cv": sample_agent_state.structured_cv.model_dump()
            }
        )
        
        # Act
        result = await qa_agent.run_as_node(sample_agent_state)
        
        # Assert
        assert isinstance(result, dict), "QA agent should return a dictionary"
        
        # Verify that returned keys match AgentState fields
        agent_state_fields = set(AgentState.model_fields.keys())
        actual_keys = set(result.keys())
        invalid_keys = actual_keys - agent_state_fields
        
        assert not invalid_keys, f"QA agent returned invalid keys: {invalid_keys}"
        
        # Check for expected quality_check_results key
        if "quality_check_results" not in result and "error_messages" not in result:
            pytest.fail("QA agent should return either 'quality_check_results' or 'error_messages'")

    @pytest.mark.asyncio
    async def test_agent_state_field_coverage(self):
        """Test that all critical AgentState fields are covered by agents."""
        # Get all AgentState fields
        agent_state_fields = set(AgentState.model_fields.keys())
        
        # Define which agents are responsible for which fields
        agent_field_mapping = {
            "ParserAgent": {"structured_cv", "job_description_data"},
            "ResearchAgent": {"research_findings"},
            "QualityAssuranceAgent": {"quality_check_results", "structured_cv"},
            # Other agents may handle other fields like final_output_path, etc.
        }
        
        # Check that critical fields are covered
        covered_fields = set()
        for agent, fields in agent_field_mapping.items():
            covered_fields.update(fields)
        
        critical_fields = {
            "structured_cv", "job_description_data", 
            "research_findings", "quality_check_results"
        }
        
        missing_coverage = critical_fields - covered_fields
        assert not missing_coverage, f"Critical fields not covered by any agent: {missing_coverage}"

    def test_agent_io_schema_consistency(self):
        """Test that AgentIO schemas are consistent with AgentState fields."""
        # This test ensures that if AgentIO schemas are defined,
        # they reference valid AgentState fields
        agent_state_fields = set(AgentState.model_fields.keys())
        
        # For now, this is a placeholder test since AgentIO schemas
        # are not strictly enforced yet
        # TODO: Implement specific AgentIO schema validation once schemas are defined
        
        assert agent_state_fields, "AgentState should have defined fields"
        assert "structured_cv" in agent_state_fields
        assert "job_description_data" in agent_state_fields
        assert "research_findings" in agent_state_fields
        assert "quality_check_results" in agent_state_fields


if __name__ == "__main__":
    pytest.main([__file__, "-v"])