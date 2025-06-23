import pytest
from unittest.mock import Mock
from src.agents.specialized_agents import CVAnalysisAgent
from src.models.cv_analysis_result import CVAnalysisResult
from src.models.data_models import StructuredCV, JobDescriptionData
from src.agents.agent_base import AgentExecutionContext
from src.config.settings import AppConfig
from src.services.llm_service import EnhancedLLMService


@pytest.mark.asyncio
async def test_cv_analysis_agent_enforces_pydantic_contract():
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
    context = AgentExecutionContext(session_id="test-session")
    input_data = {
        "cv_data": StructuredCV(),
        "job_description": JobDescriptionData(raw_text="Python, leadership"),
    }
    result = await agent.run_async(input_data, context)
    assert result.success
    assert isinstance(result.output_data, CVAnalysisResult)
    assert hasattr(result.output_data, "skill_matches")
    assert hasattr(result.output_data, "match_score")
