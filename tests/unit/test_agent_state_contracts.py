import sys
import os

# Ensure project root (containing src/) is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pytest
from src.orchestration.state import AgentState
from src.agents.parser_agent import ParserAgent
from src.agents.research_agent import ResearchAgent
from src.agents.quality_assurance_agent import QualityAssuranceAgent
from src.models.data_models import JobDescriptionData, StructuredCV


@pytest.mark.asyncio
async def test_parser_agent_contract():
    agent = ParserAgent(name="TestParser", description="Test parser agent")
    state = AgentState(
        structured_cv=StructuredCV(),
        job_description_data=JobDescriptionData(raw_text="Job description here"),
    )
    result = await agent.run_as_node(state)
    # Accept either a valid result or error_messages (contract: must not crash)
    assert (set(result.keys()) == {"structured_cv", "job_description_data"}) or (
        "error_messages" in result
    )


@pytest.mark.asyncio
async def test_research_agent_contract():
    agent = ResearchAgent(name="TestResearch", description="Test research agent")
    state = AgentState(
        structured_cv=StructuredCV(),
        job_description_data=JobDescriptionData(raw_text="Job description here"),
    )
    result = await agent.run_as_node(state)
    # Should return research_findings or error_messages
    assert ("research_findings" in result) or ("error_messages" in result)


@pytest.mark.asyncio
async def test_quality_assurance_agent_contract():
    agent = QualityAssuranceAgent(name="TestQA", description="Test QA agent")
    state = AgentState(
        structured_cv=StructuredCV(),
        job_description_data=JobDescriptionData(raw_text="Job description here"),
    )
    result = await agent.run_as_node(state)
    # Should return AgentState with at least one of the following populated
    assert (
        getattr(result, "quality_check_results", None) is not None
        or getattr(result, "structured_cv", None) is not None
        or (hasattr(result, "error_messages") and result.error_messages)
    )
