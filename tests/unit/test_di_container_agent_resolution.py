import pytest
from src.core.dependency_injection import DependencyContainer
from src.services.llm_service import EnhancedLLMService
from src.services.error_recovery import ErrorRecoveryService
from src.services.progress_tracker import ProgressTracker
from src.agents.enhanced_content_writer import EnhancedContentWriterAgent
from unittest.mock import Mock


def test_di_container_resolves_enhanced_content_writer_agent():
    from src.agents.parser_agent import ParserAgent
    from src.config.settings import AppConfig

    container = DependencyContainer()
    # Register dependencies
    container.register_singleton(
        "llm_service", EnhancedLLMService, factory=lambda: Mock(spec=EnhancedLLMService)
    )
    container.register_singleton(
        "error_recovery_service",
        ErrorRecoveryService,
        factory=lambda: Mock(spec=ErrorRecoveryService),
    )
    container.register_singleton(
        "progress_tracker", ProgressTracker, factory=lambda: Mock(spec=ProgressTracker)
    )
    container.register_singleton(
        "parser_agent", ParserAgent, factory=lambda: Mock(spec=ParserAgent)
    )
    container.register_singleton("settings", AppConfig, factory=lambda: AppConfig())
    # Register agent
    container.register_singleton(
        "EnhancedContentWriterAgent",
        EnhancedContentWriterAgent,
        factory=lambda: EnhancedContentWriterAgent(
            llm_service=container.get(EnhancedLLMService, "llm_service"),
            error_recovery_service=container.get(
                ErrorRecoveryService, "error_recovery_service"
            ),
            progress_tracker=container.get(ProgressTracker, "progress_tracker"),
            parser_agent=container.get(ParserAgent, "parser_agent"),
            settings=container.get(AppConfig, "settings"),
            name="TestAgent",
            description="Test agent for DI",
            content_type=None,
        ),
    )
    agent = container.get(EnhancedContentWriterAgent, "EnhancedContentWriterAgent")
    assert isinstance(agent, EnhancedContentWriterAgent)
    assert agent.llm_service is not None
    assert agent.error_recovery_service is not None
    assert agent.progress_tracker is not None
    assert agent.parser_agent is not None
    assert agent.settings is not None
