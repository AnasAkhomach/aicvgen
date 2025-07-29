"""Integration test for JobDescriptionParserAgent dependency injection with container."""

from unittest.mock import Mock, patch

import pytest

from src.agents.job_description_parser_agent import JobDescriptionParserAgent
from src.core.container import get_container
from src.services.llm_cv_parser_service import LLMCVParserService


class TestJobDescriptionParserAgentDIContainer:
    """Integration test for JobDescriptionParserAgent with DI container."""

    @pytest.fixture
    def container(self):
        """Get the DI container instance."""
        return get_container()

    def test_container_provides_llm_cv_parser_service(self, container):
        """Test that the container can provide LLMCVParserService."""
        # Test that the provider exists
        assert hasattr(container, "llm_cv_parser_service")

        # Test that we can get an instance (this will test the provider configuration)
        with patch(
            "src.services.llm_cv_parser_service.LLMCVParserService.__init__",
            return_value=None,
        ):
            service = container.llm_cv_parser_service()
            assert service is not None

    def test_container_provides_job_description_parser_agent_with_di(self, container):
        """Test that the container provides JobDescriptionParserAgent with proper DI."""
        test_session_id = "test-session-di"

        # Mock the dependencies to avoid actual initialization
        with patch(
            "src.services.llm_cv_parser_service.LLMCVParserService.__init__",
            return_value=None,
        ), patch(
            "src.agents.job_description_parser_agent.JobDescriptionParserAgent.__init__",
            return_value=None,
        ) as mock_init:
            # Get the agent from the container with session_id as runtime argument
            agent = container.job_description_parser_agent(session_id=test_session_id)

            # Verify the agent was created
            assert agent is not None

            # Verify the constructor was called with the expected arguments
            mock_init.assert_called_once()
            call_args = mock_init.call_args

            # Check that llm_cv_parser_service was passed as an argument
            assert "llm_cv_parser_service" in call_args.kwargs
            assert call_args.kwargs["llm_cv_parser_service"] is not None
            # Check that session_id was passed correctly
            assert call_args.kwargs["session_id"] == test_session_id

    def test_container_dependency_injection_flow(self, container):
        """Test the complete dependency injection flow."""
        test_session_id = "test-session-flow"

        # Mock all the dependencies to trace the flow
        with patch("src.config.settings.get_config") as mock_config, patch(
            "src.templates.content_templates.ContentTemplateManager.__init__",
            return_value=None,
        ), patch(
            "src.services.llm_service.EnhancedLLMService.__init__", return_value=None
        ), patch(
            "src.services.llm_cv_parser_service.LLMCVParserService.__init__",
            return_value=None,
        ) as mock_parser_init, patch(
            "src.agents.job_description_parser_agent.JobDescriptionParserAgent.__init__",
            return_value=None,
        ) as mock_agent_init:
            # Configure mock config
            mock_config.return_value = Mock()
            mock_config.return_value.agent_settings.model_dump.return_value = {
                "test": "settings"
            }

            # Get the agent from the container with session_id as runtime argument
            agent = container.job_description_parser_agent(session_id=test_session_id)

            # Verify the LLMCVParserService was created with proper dependencies
            mock_parser_init.assert_called_once()
            parser_call_args = mock_parser_init.call_args

            # Verify the agent was created with the parser service
            mock_agent_init.assert_called_once()
            agent_call_args = mock_agent_init.call_args

            # Verify llm_cv_parser_service is in the agent's arguments
            assert "llm_cv_parser_service" in agent_call_args.kwargs
            # Verify session_id was passed correctly
            assert agent_call_args.kwargs["session_id"] == test_session_id

    def test_container_singleton_behavior_for_services(self, container):
        """Test that services are properly managed as singletons where expected."""
        with patch(
            "src.services.llm_cv_parser_service.LLMCVParserService.__init__",
            return_value=None,
        ):
            # Get the same service multiple times
            service1 = container.llm_cv_parser_service()
            service2 = container.llm_cv_parser_service()

            # For Factory providers, we expect different instances
            # This tests that the provider is correctly configured as Factory
            assert service1 is not service2

    def test_container_agent_factory_behavior(self, container):
        """Test that agents are created as new instances (Factory behavior)."""
        test_session_id = "test-session-factory"

        with patch(
            "src.services.llm_cv_parser_service.LLMCVParserService.__init__",
            return_value=None,
        ), patch(
            "src.agents.job_description_parser_agent.JobDescriptionParserAgent.__init__",
            return_value=None,
        ):
            # Get multiple agent instances with session_id as runtime argument
            agent1 = container.job_description_parser_agent(session_id=test_session_id)
            agent2 = container.job_description_parser_agent(session_id=test_session_id)

            # Agents should be different instances (Factory behavior)
            assert agent1 is not agent2

    def test_container_session_id_propagation(self, container):
        """Test that session ID is properly propagated to agents."""
        test_session_id = "test-session-propagation"

        with patch(
            "src.services.llm_cv_parser_service.LLMCVParserService.__init__",
            return_value=None,
        ), patch(
            "src.agents.job_description_parser_agent.JobDescriptionParserAgent.__init__",
            return_value=None,
        ) as mock_agent_init:
            # Get the agent with session_id as runtime argument
            container.job_description_parser_agent(session_id=test_session_id)

            # Verify session_id was passed correctly
            mock_agent_init.assert_called_once()
            call_args = mock_agent_init.call_args
            assert call_args.kwargs["session_id"] == test_session_id
