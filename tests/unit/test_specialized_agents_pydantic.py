import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import pytest
from src.agents.specialized_agents import CVAnalysisAgent, EnhancedParserAgent
from src.models.data_models import StructuredCV, JobDescriptionData
from src.models.cv_analysis_result import CVAnalysisResult
from src.agents.agent_base import AgentExecutionContext, AgentResult
from src.orchestration.state import AgentState


@pytest.mark.asyncio
async def test_cv_analysis_agent_enforces_pydantic_contracts():
    agent = CVAnalysisAgent()
    cv = StructuredCV()
    jd = JobDescriptionData(raw_text="Software Engineer, Python, 5+ years experience")
    context = AgentExecutionContext(session_id="test-session")
    input_data = {"cv_data": cv, "job_description": jd}
    result: AgentResult = await agent.run_async(input_data, context)
    assert result.success
    assert isinstance(result.output_data, CVAnalysisResult)
    assert hasattr(result.output_data, "skill_matches")
    assert hasattr(result.output_data, "match_score")


@pytest.mark.asyncio
async def test_enhanced_parser_agent_enforces_pydantic_contracts():
    agent = EnhancedParserAgent()
    context = AgentExecutionContext(session_id="test-session")
    # Create a minimal AgentState for the parser
    state = AgentState(
        trace_id="test-trace",
        structured_cv=StructuredCV(),
        job_description_data=JobDescriptionData(raw_text="Python, leadership"),
    )
    # The EnhancedParserAgent expects run_async to receive a dict with AgentState-like keys
    result: AgentResult = await agent.run_async(state, context)
    assert result.success
    assert isinstance(result.output_data, dict)
    assert isinstance(result.output_data["structured_cv"], StructuredCV)
    assert isinstance(result.output_data["job_data"], JobDescriptionData)
