import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import pytest
from unittest.mock import Mock
from src.agents.specialized_agents import CVAnalysisAgent, EnhancedParserAgent
from src.models.data_models import StructuredCV, JobDescriptionData
from src.models.cv_analysis_result import CVAnalysisResult
from src.agents.agent_base import AgentExecutionContext, AgentResult
from src.orchestration.state import AgentState
from src.config.settings import AppConfig
from src.services.llm_service import EnhancedLLMService
from src.services.llm_client import LLMClient
from src.services.llm_retry_handler import LLMRetryHandler
from src.services.vector_store_service import VectorStoreService
from src.services.error_recovery import get_error_recovery_service
from src.services.progress_tracker import get_progress_tracker


@pytest.mark.asyncio
async def test_cv_analysis_agent_enforces_pydantic_contracts():
    # Provide all required dependencies
    settings = AppConfig()
    llm_client = Mock()
    llm_retry_handler = Mock()
    llm_service = EnhancedLLMService(
        settings=settings,
        llm_client=llm_client,
        llm_retry_handler=llm_retry_handler,
        cache=Mock(),
    )
    agent = CVAnalysisAgent(llm_service=llm_service, settings=settings)
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
    # Provide all required dependencies
    settings = AppConfig()
    llm_client = Mock()
    llm_retry_handler = Mock()
    llm_service = EnhancedLLMService(
        settings=settings,
        llm_client=llm_client,
        llm_retry_handler=llm_retry_handler,
        cache=Mock(),
    )
    vector_store_service = VectorStoreService(settings)
    error_recovery_service = get_error_recovery_service()
    progress_tracker = get_progress_tracker()
    agent = EnhancedParserAgent(
        llm_service=llm_service,
        vector_store_service=vector_store_service,
        error_recovery_service=error_recovery_service,
        progress_tracker=progress_tracker,
        settings=settings,
    )
    context = AgentExecutionContext(session_id="test-session")
    # Pass an AgentState object, not a dict, to match EnhancedParserAgent expectations
    state = AgentState(
        trace_id="test-trace",
        structured_cv=StructuredCV(),
        job_description_data=JobDescriptionData(raw_text="Python, leadership"),
        cv_text="John Doe, Python Developer, 5+ years experience",
    )
    result: AgentResult = await agent.run_async(state, context)
    assert result.success
    assert isinstance(result.output_data, agent.EnhancedParserAgentOutput)
    assert isinstance(result.output_data.structured_cv, StructuredCV)
    assert isinstance(result.output_data.job_data, JobDescriptionData)
