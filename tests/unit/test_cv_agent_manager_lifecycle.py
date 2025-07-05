"""Unit tests for CV Agent Manager lifecycle management."""

import pytest
from unittest.mock import Mock, patch, MagicMock

from src.integration.cv_agent_manager import CVAgentManager
from src.error_handling.exceptions import AgentExecutionError


class TestCVAgentManagerLifecycle:
    """Test cases for CV Agent Manager lifecycle management."""

    def setup_method(self):
        """Set up test fixtures."""
        self.session_id = "test_session"
        
    @patch('src.integration.cv_agent_manager.get_agent_lifecycle_manager')
    @patch('src.integration.cv_agent_manager.AgentErrorHandler')
    def test_initialization_success(self, mock_error_handler, mock_get_lifecycle_manager):
        """Test successful initialization of CV Agent Manager."""
        # Setup mocks
        mock_lifecycle_manager = Mock()
        mock_get_lifecycle_manager.return_value = mock_lifecycle_manager
        mock_error_handler.return_value = Mock()
        
        # Create manager
        manager = CVAgentManager(session_id=self.session_id)
        
        # Verify initialization
        assert manager.is_initialized
        assert manager.session_id == self.session_id
        mock_get_lifecycle_manager.assert_called_once()
        
    @patch('src.integration.cv_agent_manager.get_agent_lifecycle_manager')
    def test_initialization_failure(self, mock_get_lifecycle_manager):
        """Test initialization failure handling."""
        # Setup mock to return None (failure case)
        mock_get_lifecycle_manager.return_value = None
        
        # Verify initialization fails
        with pytest.raises(RuntimeError, match="Agent lifecycle manager not available"):
            CVAgentManager(session_id=self.session_id)
            
    @patch('src.integration.cv_agent_manager.get_agent_lifecycle_manager')
    @patch('src.integration.cv_agent_manager.AgentErrorHandler')
    def test_get_agent_success(self, mock_error_handler, mock_get_lifecycle_manager):
        """Test successful agent retrieval."""
        # Setup mocks
        mock_agent = Mock()
        mock_lifecycle_manager = Mock()
        mock_lifecycle_manager.get_agent.return_value = mock_agent
        mock_get_lifecycle_manager.return_value = mock_lifecycle_manager
        mock_error_handler.return_value = Mock()
        
        # Create manager and get agent
        manager = CVAgentManager(session_id=self.session_id)
        result = manager.get_agent("test_agent")
        
        # Verify agent retrieval
        assert result == mock_agent
        mock_lifecycle_manager.get_agent.assert_called_once_with(
            agent_type="test_agent",
            session_id=self.session_id
        )
        
    @patch('src.integration.cv_agent_manager.get_agent_lifecycle_manager')
    @patch('src.integration.cv_agent_manager.AgentErrorHandler')
    def test_get_agent_with_session_override(self, mock_error_handler, mock_get_lifecycle_manager):
        """Test agent retrieval with session ID override."""
        # Setup mocks
        mock_agent = Mock()
        mock_lifecycle_manager = Mock()
        mock_lifecycle_manager.get_agent.return_value = mock_agent
        mock_get_lifecycle_manager.return_value = mock_lifecycle_manager
        mock_error_handler.return_value = Mock()
        
        override_session = "override_session"
        
        # Create manager and get agent with override
        manager = CVAgentManager(session_id=self.session_id)
        result = manager.get_agent("test_agent", session_id=override_session)
        
        # Verify session override is used
        assert result == mock_agent
        mock_lifecycle_manager.get_agent.assert_called_once_with(
            agent_type="test_agent",
            session_id=override_session
        )
        
    @patch('src.integration.cv_agent_manager.get_agent_lifecycle_manager')
    @patch('src.integration.cv_agent_manager.AgentErrorHandler')
    def test_get_agent_error_handling(self, mock_error_handler, mock_get_lifecycle_manager):
        """Test error handling during agent retrieval."""
        # Setup mocks
        mock_lifecycle_manager = Mock()
        mock_lifecycle_manager.get_agent.side_effect = AgentExecutionError("test_agent", "Test error")
        mock_get_lifecycle_manager.return_value = mock_lifecycle_manager
        
        mock_error_handler_instance = Mock()
        mock_fallback_agent = Mock()
        mock_error_handler_instance.handle_general_error.return_value = mock_fallback_agent
        mock_error_handler.return_value = mock_error_handler_instance
        
        # Create manager and get agent (should handle error)
        manager = CVAgentManager(session_id=self.session_id)
        result = manager.get_agent("test_agent")
        
        # Verify error handling
        assert result == mock_fallback_agent
        mock_error_handler_instance.handle_general_error.assert_called_once()
        
    @patch('src.integration.cv_agent_manager.get_agent_lifecycle_manager')
    @patch('src.integration.cv_agent_manager.AgentErrorHandler')
    def test_list_agents_success(self, mock_error_handler, mock_get_lifecycle_manager):
        """Test successful agent listing."""
        # Setup mocks
        mock_container = Mock()
        
        # Create mock attributes for the container
        test_agent_mock = Mock()
        another_agent_mock = Mock()
        not_an_agent_mock = Mock()
        private_agent_mock = Mock()
        
        # Set attributes on the mock container
        setattr(mock_container, 'test_agent', test_agent_mock)
        setattr(mock_container, 'another_agent', another_agent_mock)
        setattr(mock_container, 'some_service', not_an_agent_mock)  # This doesn't end with _agent
        setattr(mock_container, '_private_agent', private_agent_mock)
        
        mock_lifecycle_manager = Mock()
        mock_lifecycle_manager.container = mock_container
        mock_get_lifecycle_manager.return_value = mock_lifecycle_manager
        mock_error_handler.return_value = Mock()
        
        # Mock dir() to return our test attributes
        with patch('builtins.dir') as mock_dir:
            mock_dir.return_value = ['test_agent', 'another_agent', 'some_service', '_private_agent']
            
            # Create manager and list agents
            manager = CVAgentManager(session_id=self.session_id)
            result = manager.list_agents()
            
            # Verify agent listing (should only include *_agent that don't start with _)
            assert 'test_agent' in result
            assert 'another_agent' in result
            assert 'some_service' not in result  # Doesn't end with _agent
            assert '_private_agent' not in result  # Starts with _
            
    @patch('src.integration.cv_agent_manager.get_agent_lifecycle_manager')
    @patch('src.integration.cv_agent_manager.AgentErrorHandler')
    def test_get_metrics_success(self, mock_error_handler, mock_get_lifecycle_manager):
        """Test successful metrics retrieval."""
        # Setup mocks
        mock_metrics = {"total_agents": 5, "agent_metrics": {}}
        mock_lifecycle_manager = Mock()
        mock_lifecycle_manager.get_metrics.return_value = mock_metrics
        mock_get_lifecycle_manager.return_value = mock_lifecycle_manager
        mock_error_handler.return_value = Mock()
        
        # Create manager and get metrics
        manager = CVAgentManager(session_id=self.session_id)
        result = manager.get_metrics()
        
        # Verify metrics retrieval
        assert result == mock_metrics
        mock_lifecycle_manager.get_metrics.assert_called_once()
        
    @patch('src.integration.cv_agent_manager.get_agent_lifecycle_manager')
    @patch('src.integration.cv_agent_manager.AgentErrorHandler')
    def test_cleanup_success(self, mock_error_handler, mock_get_lifecycle_manager):
        """Test successful cleanup."""
        # Setup mocks
        mock_lifecycle_manager = Mock()
        mock_lifecycle_manager.cleanup = Mock()
        mock_get_lifecycle_manager.return_value = mock_lifecycle_manager
        mock_error_handler.return_value = Mock()
        
        # Create manager and cleanup
        manager = CVAgentManager(session_id=self.session_id)
        assert manager.is_initialized
        
        manager.cleanup()
        
        # Verify cleanup
        assert not manager.is_initialized
        mock_lifecycle_manager.cleanup.assert_called_once()
        
    @patch('src.integration.cv_agent_manager.get_agent_lifecycle_manager')
    @patch('src.integration.cv_agent_manager.AgentErrorHandler')
    def test_context_manager(self, mock_error_handler, mock_get_lifecycle_manager):
        """Test context manager functionality."""
        # Setup mocks
        mock_lifecycle_manager = Mock()
        mock_lifecycle_manager.cleanup = Mock()
        mock_get_lifecycle_manager.return_value = mock_lifecycle_manager
        mock_error_handler.return_value = Mock()
        
        # Use as context manager
        with CVAgentManager(session_id=self.session_id) as manager:
            assert manager.is_initialized
            assert manager.session_id == self.session_id
            
        # Verify cleanup was called
        mock_lifecycle_manager.cleanup.assert_called_once()
        
    def test_uninitialized_operations(self):
        """Test that operations fail on uninitialized manager."""
        # Create manager but prevent initialization
        with patch('src.integration.cv_agent_manager.get_agent_lifecycle_manager') as mock_get_lifecycle_manager:
            mock_get_lifecycle_manager.return_value = None
            
            # This should fail during initialization
            with pytest.raises(RuntimeError):
                CVAgentManager(session_id=self.session_id)