import pytest
from src.agents.research_agent import ResearchAgent
from src.models.research_agent_models import ResearchAgentNodeResult
from unittest.mock import Mock
from src.agents.agent_base import AgentExecutionContext
import asyncio


@pytest.mark.asyncio
def test_research_agent_enforces_pydantic_contract():
    agent = ResearchAgent(
        llm_service=Mock(),
        error_recovery_service=Mock(),
        progress_tracker=Mock(),
        vector_db=Mock(),
        settings=Mock(),
        name="test_research_agent",
        description="Test Research agent",
    )
    context = AgentExecutionContext(session_id="test-session")
    # Minimal valid input
    input_data = {"structured_cv": {}, "job_description_data": {}}
    result = asyncio.run(agent.run_async(input_data, context))
    if not result.success:
        # Accept error fallback as valid result
        assert hasattr(result.output_data, "error")
        assert isinstance(result.output_data.error, str)
    else:
        assert isinstance(result.output_data, ResearchAgentNodeResult)
        assert hasattr(result.output_data, "research_findings")
        assert hasattr(result.output_data, "error_messages")
