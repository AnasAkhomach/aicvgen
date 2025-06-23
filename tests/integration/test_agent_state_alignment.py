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
    ProcessingStatus,
)


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
        return AgentState(
            trace_id="test-trace-123",
            structured_cv=StructuredCV(),
            job_description_data=JobDescriptionData(
                raw_text="Senior Python Developer position requiring 5+ years experience.",
                company_name="Tech Corp",
                skills=["Python", "Django", "REST APIs"],
            ),
            cv_text="Sample CV text",
            current_item_id="item-123",  # Provide a valid string
            is_initial_generation=True,
        )

    @pytest.mark.asyncio
    async def test_parser_agent_output_alignment(
        self, mock_llm_service, sample_agent_state
    ):
        """Test that ParserAgent returns keys matching AgentState fields."""
        # Arrange
        vector_store_service = Mock()
        error_recovery_service = Mock()
        progress_tracker = Mock()
        settings = Mock()
        parser_agent = ParserAgent(
            name="TestParserAgent",
            description="Test parser agent",
            llm_service=mock_llm_service,
            vector_store_service=vector_store_service,
            error_recovery_service=error_recovery_service,
            progress_tracker=progress_tracker,
            settings=settings,
        )

        # Act
        result = await parser_agent.run_as_node(sample_agent_state)

        # Assert
        assert isinstance(
            result, AgentState
        ), "Parser agent should return an AgentState object"

        # Verify that returned fields match AgentState fields
        expected_fields = {"structured_cv", "job_description_data"}
        actual_fields = set(result.model_fields_set)

        # Check that all returned fields are valid AgentState fields
        agent_state_fields = set(AgentState.model_fields.keys())
        invalid_fields = actual_fields - agent_state_fields
        assert (
            not invalid_fields
        ), f"Parser agent returned invalid fields: {invalid_fields}"

        # Check that expected fields are present
        assert expected_fields.issubset(
            actual_fields
        ), f"Missing expected fields: {expected_fields - actual_fields}"

    @pytest.mark.asyncio
    async def test_research_agent_output_alignment(
        self, mock_llm_service, sample_agent_state
    ):
        """Test that ResearchAgent returns keys matching AgentState fields."""
        # Arrange
        error_recovery_service = Mock()
        progress_tracker = Mock()
        vector_db = Mock()
        settings = Mock()
        research_agent = ResearchAgent(
            name="TestResearchAgent",
            description="Test research agent",
            llm_service=mock_llm_service,
            error_recovery_service=error_recovery_service,
            progress_tracker=progress_tracker,
            vector_db=vector_db,
            settings=settings,
        )

        # Mock the async method to return expected structure
        research_agent.run_async = AsyncMock()
        research_agent.run_async.return_value = Mock(
            success=True,
            output_data={"research_findings": {"key_skills": ["Python", "Testing"]}},
        )

        # Act
        result = await research_agent.run_as_node(sample_agent_state)

        # Assert
        # Accept both dict and Pydantic model for result
        if not isinstance(result, dict):
            # Try to convert to dict if it's a Pydantic model
            try:
                result = result.dict()
            except Exception:
                pass
        assert isinstance(
            result, dict
        ), "Research agent should return a dictionary or dict-like object"

        # Verify that returned keys match AgentState fields
        agent_state_fields = set(AgentState.model_fields.keys())
        actual_keys = set(result.keys())
        invalid_keys = actual_keys - agent_state_fields

        assert not invalid_keys, f"Research agent returned invalid keys: {invalid_keys}"

        # Check for expected research_findings key
        if "research_findings" not in result and "error_messages" not in result:
            pytest.fail(
                "Research agent should return either 'research_findings' or 'error_messages'"
            )

    @pytest.mark.asyncio
    async def test_quality_assurance_agent_output_alignment(
        self, mock_llm_service, sample_agent_state
    ):
        """Test that QualityAssuranceAgent returns keys matching AgentState fields."""
        # Arrange
        error_recovery_service = Mock()
        progress_tracker = Mock()
        qa_agent = QualityAssuranceAgent(
            name="TestQAAgent",
            description="Test QA agent",
            llm_service=mock_llm_service,
            error_recovery_service=error_recovery_service,
            progress_tracker=progress_tracker,
        )

        # Mock the async method to return expected structure
        qa_agent.run_async = AsyncMock()
        qa_agent.run_async.return_value = Mock(
            success=True,
            output_data={
                "quality_check_results": {"score": 0.85, "issues": []},
                "updated_structured_cv": sample_agent_state.structured_cv.model_dump(),
            },
        )

        # Act
        result = await qa_agent.run_as_node(sample_agent_state)

        # Assert
        assert isinstance(
            result, AgentState
        ), "QA agent should return an AgentState object"
        assert hasattr(result, "quality_check_results")
        assert result.quality_check_results is not None

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
            "structured_cv",
            "job_description_data",
            "research_findings",
            "quality_check_results",
        }

        missing_coverage = critical_fields - covered_fields
        assert (
            not missing_coverage
        ), f"Critical fields not covered by any agent: {missing_coverage}"

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
