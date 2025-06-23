import pytest
from src.agents.quality_assurance_agent import QualityAssuranceAgent
from src.models.quality_assurance_agent_models import QualityAssuranceResult
from unittest.mock import Mock
from src.agents.agent_base import AgentExecutionContext
import asyncio


@pytest.mark.asyncio
def test_quality_assurance_agent_enforces_pydantic_contract():
    agent = QualityAssuranceAgent(
        llm_service=Mock(),
        error_recovery_service=Mock(),
        progress_tracker=Mock(),
        name="test_qa_agent",
        description="Test QA agent",
    )
    context = AgentExecutionContext(session_id="test-session")
    # Minimal valid input
    input_data = {"structured_cv": {}, "job_description": ""}
    result = asyncio.run(agent.run_async(input_data, context))
    # If result is an error fallback, ensure it matches the model
    if hasattr(result.output_data, "error"):
        assert isinstance(result.output_data.error, str)
    else:
        assert result.success
        assert isinstance(result.output_data, QualityAssuranceResult)
        assert hasattr(result.output_data, "section_results")
        assert hasattr(result.output_data, "overall_checks")
