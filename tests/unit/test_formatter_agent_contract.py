import pytest
from src.agents.formatter_agent import FormatterAgent
from src.models.formatter_agent_models import FormatterAgentNodeResult
from unittest.mock import Mock


@pytest.mark.asyncio
async def test_formatter_agent_run_returns_pydantic_model():
    agent = FormatterAgent(
        llm_service=Mock(),
        error_recovery_service=Mock(),
        progress_tracker=Mock(),
        name="test_formatter",
        description="Test formatter agent",
    )

    # Minimal valid input (simulate AgentState with structured_cv)
    class DummyState:
        structured_cv = {"sections": []}
        format_type = "pdf"
        template_name = "professional"
        output_path = None

    result = await agent.run(DummyState())
    assert isinstance(result, FormatterAgentNodeResult)
    assert hasattr(result, "final_output_path")
    assert hasattr(result, "error_messages")
    assert isinstance(result.error_messages, list)


@pytest.mark.asyncio
async def test_formatter_agent_run_handles_missing_cv():
    agent = FormatterAgent(
        llm_service=Mock(),
        error_recovery_service=Mock(),
        progress_tracker=Mock(),
        name="test_formatter",
        description="Test formatter agent",
    )

    class DummyState:
        structured_cv = None
        format_type = "pdf"
        template_name = "professional"
        output_path = None

    result = await agent.run(DummyState())
    assert isinstance(result, FormatterAgentNodeResult)
    assert result.final_output_path is None
    assert "No structured CV data provided" in result.error_messages
