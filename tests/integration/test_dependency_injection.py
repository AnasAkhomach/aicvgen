"""Integration test for the consolidated dependency injection container."""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from core.container import get_container


def test_container_singleton_behavior():
    """Test that the container returns the same instance when called multiple times."""
    container1 = get_container()
    container2 = get_container()

    # Both calls should return the same container instance
    assert container1 is container2


def test_container_provides_services():
    """Test that the container can provide core services."""
    container = get_container()

    # Test that core services can be instantiated
    config = container.config()
    assert config is not None

    llm_service = container.llm_service()
    assert llm_service is not None

    vector_store = container.vector_store_service()
    assert vector_store is not None


def test_container_provides_agents():
    """Test that the container can provide agent factories."""
    container = get_container()

    # Test that agent factories can be called
    jd_parser_agent = container.job_description_parser_agent()
    assert jd_parser_agent is not None

    cv_parser_agent = container.user_cv_parser_agent()
    assert cv_parser_agent is not None

    cv_analyzer = container.cv_analyzer_agent()
    assert cv_analyzer is not None

    content_writer = container.enhanced_content_writer_agent()
    assert content_writer is not None


def test_singleton_services_are_consistent():
    """Test that singleton services return the same instance."""
    container = get_container()

    # Singleton services should return the same instance
    config1 = container.config()
    config2 = container.config()
    assert config1 is config2

    llm_service1 = container.llm_service()
    llm_service2 = container.llm_service()
    assert llm_service1 is llm_service2


def test_factory_services_return_new_instances():
    """Test that factory services return new instances each time."""
    container = get_container()

    # Factory services should return different instances
    agent1 = container.job_description_parser_agent()
    agent2 = container.job_description_parser_agent()
    assert agent1 is not agent2

    tracker1 = container.progress_tracker()
    tracker2 = container.progress_tracker()
    assert tracker1 is not tracker2
