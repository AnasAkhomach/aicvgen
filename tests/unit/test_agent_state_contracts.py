import pytest
from src.orchestration.state import AgentState
from src.agents.parser_agent import ParserAgent
from src.agents.research_agent import ResearchAgent
from src.agents.quality_assurance_agent import QualityAssuranceAgent
from src.models.data_models import JobDescriptionData, StructuredCV


@pytest.mark.asyncio
async def test_parser_agent_contract():
    agent = ParserAgent()
    state = AgentState(
        structured_cv=StructuredCV(),
        job_description_data=JobDescriptionData(raw_text="Job description here"),
    )
    result = await agent.run_as_node(state)
    assert set(result.keys()) == {"structured_cv", "job_description_data"}


@pytest.mark.asyncio
async def test_research_agent_contract():
    agent = ResearchAgent()
    state = AgentState(
        structured_cv=StructuredCV(),
        job_description_data=JobDescriptionData(raw_text="Job description here"),
    )
    result = await agent.run_as_node(state)
    # Should return research_findings or error_messages
    assert ("research_findings" in result) or ("error_messages" in result)


@pytest.mark.asyncio
async def test_quality_assurance_agent_contract():
    agent = QualityAssuranceAgent()
    state = AgentState(
        structured_cv=StructuredCV(),
        job_description_data=JobDescriptionData(raw_text="Job description here"),
    )
    result = await agent.run_as_node(state)
    # Should return quality_check_results, structured_cv, or error_messages
    assert any(
        k in result
        for k in ("quality_check_results", "structured_cv", "error_messages")
    )
