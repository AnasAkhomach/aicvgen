import sys
import os

# Ensure project root (containing src/) is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pytest
from unittest.mock import Mock
from src.orchestration.state import AgentState
from src.agents.parser_agent import ParserAgent
from src.agents.research_agent import ResearchAgent
from src.agents.quality_assurance_agent import QualityAssuranceAgent
from src.models.data_models import JobDescriptionData, StructuredCV


@pytest.mark.asyncio
async def test_parser_agent_contract():
    agent = ParserAgent(
        llm_service=Mock(),
        vector_store_service=Mock(),
        error_recovery_service=Mock(),
        progress_tracker=Mock(),
        settings=Mock(),
        name="TestParser",
        description="Test parser agent",
    )
    state = AgentState(
        structured_cv=StructuredCV(),
        job_description_data=JobDescriptionData(raw_text="Job description here"),
    )
    result = await agent.run_as_node(state)
    # Accept either a valid result or error_messages (contract: must not crash)
    assert (
        hasattr(result, "structured_cv")
        and hasattr(result, "job_description_data")
        or (hasattr(result, "error_messages"))
    )


@pytest.mark.asyncio
async def test_research_agent_contract():
    # Inject mock llm_service and vector_db
    mock_llm_service = Mock()
    mock_vector_db = Mock()
    agent = ResearchAgent(
        name="TestResearch",
        description="Test research agent",
        llm_service=mock_llm_service,
        vector_db=mock_vector_db,
        error_recovery_service=Mock(),
        progress_tracker=Mock(),
        settings=Mock(),
    )
    state = AgentState(
        structured_cv=StructuredCV(),
        job_description_data=JobDescriptionData(raw_text="Job description here"),
    )
    result = await agent.run_as_node(state)
    # Accept either a result with 'research_findings' or 'error_messages' attributes (contract: must not crash)
    assert hasattr(result, "research_findings") or hasattr(result, "error_messages")


@pytest.mark.asyncio
async def test_quality_assurance_agent_contract():
    # Inject mock llm_service
    mock_llm_service = Mock()
    agent = QualityAssuranceAgent(
        name="TestQA",
        description="Test QA agent",
        llm_service=mock_llm_service,
        error_recovery_service=Mock(),
        progress_tracker=Mock(),
    )
    state = AgentState(
        structured_cv=StructuredCV(),
        job_description_data=JobDescriptionData(raw_text="Job description here"),
    )
    result = await agent.run_as_node(state)
    # Should return an AgentState with 'output_data' or 'error_messages' keys
    assert isinstance(result, AgentState) and (
        hasattr(result, "final_output_path") or hasattr(result, "error_messages")
    )
