"""Tests for dynamic session ID injection in the container and agent factory."""

import uuid
from unittest.mock import patch

import pytest

from src.core.container import Container, get_container
from src.core.factories.agent_factory import AgentFactory


class TestDynamicSessionId:
    """Test cases for dynamic session ID functionality."""

    def test_container_session_id_default(self):
        """Test that container has default session ID initially."""
        assert Container.get_current_session_id() == "default"

    def test_container_set_session_id(self):
        """Test that container session ID can be set and retrieved."""
        test_session_id = str(uuid.uuid4())
        Container.set_session_id(test_session_id)
        
        assert Container.get_current_session_id() == test_session_id
        
        # Reset to default for other tests
        Container.set_session_id("default")

    def test_agent_factory_uses_dynamic_session_id(self):
        """Test that agent factory uses dynamic session ID from container."""
        # Set a unique session ID
        test_session_id = str(uuid.uuid4())
        Container.set_session_id(test_session_id)
        
        # Mock the dependencies
        with patch('src.core.factories.agent_factory.CVAnalyzerAgent') as mock_agent:
            # Create a mock factory with minimal dependencies
            factory = AgentFactory(
                llm_service=None,
                template_manager=None,
                vector_store_service=None,
                session_id_provider=Container.get_current_session_id
            )
            
            # Create an agent without specifying session_id
            factory.create_cv_analyzer_agent()
            
            # Verify the agent was created with the dynamic session ID
            mock_agent.assert_called_once_with(
                llm_service=None,
                session_id=test_session_id
            )
        
        # Reset to default
        Container.set_session_id("default")

    def test_agent_factory_explicit_session_id_override(self):
        """Test that explicit session ID overrides container session ID."""
        # Set a container session ID
        container_session_id = str(uuid.uuid4())
        Container.set_session_id(container_session_id)
        
        # Use a different explicit session ID
        explicit_session_id = str(uuid.uuid4())
        
        with patch('src.core.factories.agent_factory.CVAnalyzerAgent') as mock_agent:
            factory = AgentFactory(
                llm_service=None,
                template_manager=None,
                vector_store_service=None,
                session_id_provider=Container.get_current_session_id
            )
            
            # Create an agent with explicit session_id
            factory.create_cv_analyzer_agent(session_id=explicit_session_id)
            
            # Verify the agent was created with the explicit session ID
            mock_agent.assert_called_once_with(
                llm_service=None,
                session_id=explicit_session_id
            )
        
        # Reset to default
        Container.set_session_id("default")

    def test_container_providers_use_dynamic_session_id(self):
        """Test that container providers use dynamic session ID."""
        test_session_id = str(uuid.uuid4())
        Container.set_session_id(test_session_id)
        
        # Verify that the callable provider returns the correct session ID
        assert Container.get_current_session_id() == test_session_id
        
        # Test that the lambda function in the provider works correctly
        session_id_callable = lambda: Container.get_current_session_id()
        assert session_id_callable() == test_session_id
        
        # Reset to default
        Container.set_session_id("default")
        assert Container.get_current_session_id() == "default"

    def test_multiple_session_ids_isolation(self):
        """Test that different session IDs are properly isolated."""
        session_id_1 = str(uuid.uuid4())
        session_id_2 = str(uuid.uuid4())
        
        # Test first session
        Container.set_session_id(session_id_1)
        assert Container.get_current_session_id() == session_id_1
        
        # Test second session
        Container.set_session_id(session_id_2)
        assert Container.get_current_session_id() == session_id_2
        
        # Verify the change persisted
        assert Container.get_current_session_id() != session_id_1
        
        # Reset to default
        Container.set_session_id("default")