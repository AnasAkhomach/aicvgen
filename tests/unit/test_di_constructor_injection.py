import pytest
from unittest.mock import Mock
from src.agents.enhanced_content_writer import EnhancedContentWriterAgent
from src.services.llm_service import EnhancedLLMService
from src.services.error_recovery import ErrorRecoveryService
from src.services.progress_tracker import ProgressTracker
from src.models.data_models import ContentType


def test_enhanced_content_writer_agent_constructor_injection():
    from src.agents.parser_agent import ParserAgent
    from src.config.settings import AppConfig

    llm_service = Mock(spec=EnhancedLLMService)
    error_recovery_service = Mock(spec=ErrorRecoveryService)
    progress_tracker = Mock(spec=ProgressTracker)
    parser_agent = Mock(spec=ParserAgent)
    settings = AppConfig()
    agent = EnhancedContentWriterAgent(
        llm_service=llm_service,
        error_recovery_service=error_recovery_service,
        progress_tracker=progress_tracker,
        parser_agent=parser_agent,
        settings=settings,
        name="TestAgent",
        description="Test agent for DI",
        content_type=ContentType.QUALIFICATION,
    )
    assert agent.llm_service is llm_service
    assert agent.error_recovery_service is error_recovery_service
    assert agent.progress_tracker is progress_tracker
    assert agent.parser_agent is parser_agent
    assert agent.settings is settings
    assert agent.name == "TestAgent"
    assert agent.description == "Test agent for DI"
