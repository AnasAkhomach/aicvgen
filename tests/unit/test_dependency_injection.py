import pytest
from src.core.dependency_injection import DependencyContainer, configure_container
from src.services.llm_service import EnhancedLLMService
from src.config.settings import Settings
from src.agents.cv_analyzer_agent import CVAnalyzerAgent
from src.agents.enhanced_content_writer import EnhancedContentWriterAgent


def test_configure_container_registers_all_dependencies():
    container = DependencyContainer()
    configure_container(container)

    # Core services
    assert container.get(Settings, "settings")
    assert container.get(EnhancedLLMService, "EnhancedLLMService")

    # Agents
    assert container.get(CVAnalyzerAgent, "CVAnalyzerAgent")
    assert container.get(EnhancedContentWriterAgent, "EnhancedContentWriterAgent")

    # Check that repeated gets return the same singleton instance
    s1 = container.get(Settings, "settings")
    s2 = container.get(Settings, "settings")
    assert s1 is s2

    # Should not raise for any registered agent/service
    agent_names = [
        "ParserAgent",
        "EnhancedContentWriterAgent",
        "QualityAssuranceAgent",
        "FormatterAgent",
        "ResearchAgent",
        "CVAnalyzerAgent",
        "CleaningAgent",
    ]
    for name in agent_names:
        assert container.get_by_name(name)
