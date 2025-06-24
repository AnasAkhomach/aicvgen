import pytest
import uuid
from src.core.dependency_injection import (
    get_container,
    DependencyContainer,
    LifecycleScope,
)
from src.core.application_startup import ApplicationStartup
from src.services.llm_service import EnhancedLLMService
from src.config.settings import Settings, AppConfig
from src.agents.parser_agent import ParserAgent


@pytest.fixture(scope="module")
def initialized_container() -> DependencyContainer:
    """Fixture to provide an initialized dependency container."""
    startup = ApplicationStartup()
    startup.initialize_application()
    return startup.container


def test_singleton_scope(initialized_container: DependencyContainer):
    """Test that singleton dependencies return the same instance."""
    s1 = initialized_container.get_by_name("settings")
    s2 = initialized_container.get_by_name("settings")
    assert s1 is s2
    # Both Settings and AppConfig are valid, as one is an alias.
    assert isinstance(s1, (Settings, AppConfig))

    llm1 = initialized_container.get_by_name("llm_service")
    llm2 = initialized_container.get_by_name("llm_service")
    assert llm1 is llm2
    assert isinstance(llm1, EnhancedLLMService)


def test_session_scope(initialized_container: DependencyContainer):
    """Test that session-scoped dependencies create new instances per session."""
    session_id_1 = str(uuid.uuid4())
    session_id_2 = str(uuid.uuid4())

    p1_session1 = initialized_container.get_by_name(
        "parser_agent", session_id=session_id_1
    )
    p2_session1 = initialized_container.get_by_name(
        "parser_agent", session_id=session_id_1
    )
    assert p1_session1 is p2_session1  # Same instance within the same session
    assert isinstance(p1_session1, ParserAgent)

    p1_session2 = initialized_container.get_by_name(
        "parser_agent", session_id=session_id_2
    )
    assert p1_session1 is not p1_session2  # Different instances for different sessions


def test_get_container_returns_singleton():
    """Test that get_container() always returns the same container instance."""
    c1 = get_container()
    c2 = get_container()
    assert c1 is c2


def test_all_agents_and_services_are_registered(
    initialized_container: DependencyContainer,
):
    """Check that all expected components are registered in the container."""
    # Services (Singletons)
    assert (
        initialized_container._registrations["settings"].scope
        == LifecycleScope.SINGLETON
    )
    assert (
        initialized_container._registrations["llm_service"].scope
        == LifecycleScope.SINGLETON
    )
    assert (
        initialized_container._registrations["vector_store_service"].scope
        == LifecycleScope.SINGLETON
    )
    assert (
        initialized_container._registrations["session_manager"].scope
        == LifecycleScope.SINGLETON
    )

    # Agents (Session-scoped)
    assert (
        initialized_container._registrations["parser_agent"].scope
        == LifecycleScope.SESSION
    )
    assert (
        initialized_container._registrations["research_agent"].scope
        == LifecycleScope.SESSION
    )
    assert (
        initialized_container._registrations["writer_agent"].scope
        == LifecycleScope.SESSION
    )

    # Orchestration (Session-scoped)
    assert (
        initialized_container._registrations["cv_workflow_graph"].scope
        == LifecycleScope.SESSION
    )
