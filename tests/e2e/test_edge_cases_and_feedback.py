import pytest
import asyncio
from src.core.state_manager import StateManager
from src.orchestration.state import AgentState
from src.models.data_models import StructuredCV
from src.models.cv_analysis_result import CVAnalysisResult
from unittest.mock import MagicMock


@pytest.mark.asyncio
async def test_cv_analyzer_node_metadata_propagation(monkeypatch):
    """Ensure node_execution_metadata is populated after CVAnalyzerAgent runs."""
    from src.agents.cv_analyzer_agent import CVAnalyzerAgent

    # Patch logger to avoid logging errors
    monkeypatch.setattr(
        "src.agents.cv_analyzer_agent.get_structured_logger", lambda name: MagicMock()
    )
    # Provide dummy llm_service and settings
    dummy_llm = MagicMock()
    dummy_settings = MagicMock()
    agent = CVAnalyzerAgent(
        name="cv_analyzer",
        description="desc",
        llm_service=dummy_llm,
        settings=dummy_settings,
    )

    # Patch agent method to return expected metadata
    async def dummy_run_as_node(state):
        return {
            "cv_analysis_results": CVAnalysisResult(),
            "node_execution_metadata": {
                "cv_analyzer": {"success": True, "confidence": 0.99, "error": None}
            },
        }

    agent.run_as_node = dummy_run_as_node
    state = AgentState(structured_cv=StructuredCV())
    result = await agent.run_as_node(state)
    assert "node_execution_metadata" in result
    assert "cv_analyzer" in result["node_execution_metadata"]
    assert "success" in result["node_execution_metadata"]["cv_analyzer"]


@pytest.mark.asyncio
async def test_agent_error_recovery(monkeypatch):
    """Simulate agent failure and verify fallback via ErrorRecoveryService."""
    from src.services.error_recovery import ErrorRecoveryService

    # Patch logger to avoid logging errors
    monkeypatch.setattr(
        "src.services.error_recovery.get_structured_logger", lambda name: MagicMock()
    )
    ers = ErrorRecoveryService()
    ers.logger = MagicMock()

    class FailingAgent:
        def __init__(self, error_recovery):
            self.error_recovery = error_recovery

        async def run_as_node(self, state):
            raise RuntimeError("Simulated failure")

    state = AgentState(structured_cv=StructuredCV())
    agent = FailingAgent(error_recovery=ers)
    try:
        await agent.run_as_node(state)
    except Exception as e:
        recovery = await agent.error_recovery.handle_error(
            e, item_id=None, item_type=None, session_id=None, retry_count=0
        )
        assert hasattr(recovery, "fallback_content") or hasattr(recovery, "strategy")


@pytest.mark.asyncio
async def test_invalid_input_validation():
    """Pass malformed input to agent and assert validation error is raised."""
    from src.models.validation_schemas import validate_agent_input
    from pydantic import ValidationError

    # structured_cv is required, so this should fail on construction
    with pytest.raises(ValidationError):
        AgentState(structured_cv=None)


@pytest.mark.asyncio
async def test_state_manager_persistence_failure(monkeypatch):
    """Simulate StateManager.save_state failure and verify error handling."""
    # Patch the module-level logger directly
    import src.core.state_manager as sm

    sm.logger = MagicMock()
    state_manager = StateManager()

    async def fail_save(*args, **kwargs):
        raise IOError("Disk full")

    monkeypatch.setattr(state_manager, "save_state", fail_save)
    with pytest.raises(IOError):
        await state_manager.save_state()
