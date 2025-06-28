"""Unit test for agent dependency injection."""

import sys
from pathlib import Path
import unittest.mock as mock

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from core.container import get_container
from agents.parser_agent import ParserAgent


def test_parser_agent_dependency_injection():
    """Test that ParserAgent receives its dependencies correctly from the container."""
    container = get_container()

    # Mock the dependencies
    with mock.patch.object(
        container, "llm_service"
    ) as mock_llm_service, mock.patch.object(
        container, "vector_store_service"
    ) as mock_vector_store, mock.patch.object(
        container, "template_manager"
    ) as mock_template_manager:

        # Set up mock return values
        mock_llm_service.return_value = mock.Mock()
        mock_vector_store.return_value = mock.Mock()
        mock_template_manager.return_value = mock.Mock()

        # Create agent through container
        agent = container.parser_agent()

        # Verify agent was created
        assert isinstance(agent, ParserAgent)

        # Verify dependencies were injected
        assert agent.llm_service is not None
        assert agent.vector_store_service is not None
        assert agent.template_manager is not None


def test_container_agent_factory_behavior():
    """Test that agent factories work correctly."""
    container = get_container()

    # Create multiple agents
    agent1 = container.parser_agent()
    agent2 = container.parser_agent()

    # Should be different instances (factory behavior)
    assert agent1 is not agent2
    assert isinstance(agent1, ParserAgent)
    assert isinstance(agent2, ParserAgent)
