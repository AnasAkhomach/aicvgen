"""Tests for CB-004 fix: Dynamic ContentType in error_handler_node."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.orchestration.state import AgentState
from src.models.cv_models import StructuredCV
from src.models.workflow_models import ContentType


@pytest.fixture
def sample_state():
    """Create a sample AgentState for testing."""
    return AgentState(
        session_id="test-session",
        trace_id="test-trace",
        structured_cv=StructuredCV(),
        cv_text="Sample CV content",
        error_messages=["Test error message"]
    )


@pytest.mark.asyncio
class TestCB004ErrorHandlerNodeFix:
    """Test cases for CB-004 fix in error_handler_node."""

    async def test_error_handler_uses_current_content_type_experience(self, sample_state):
        """Test that error_handler_node uses current_content_type for EXPERIENCE."""
        # Arrange
        sample_state.error_messages = ["Test error"]
        sample_state.current_content_type = ContentType.EXPERIENCE
        sample_state.current_item_id = "test-item-123"
        sample_state.trace_id = "test-trace-456"
        
        with patch('src.services.error_recovery.ErrorRecoveryService') as mock_service_class:
            mock_service_instance = MagicMock()
            mock_service_class.return_value = mock_service_instance
            
            # Mock recovery action with strategy
            mock_recovery_action = MagicMock()
            mock_recovery_action.strategy.value = "immediate_retry"
            mock_service_instance.handle_error = AsyncMock(return_value=mock_recovery_action)
            
            # Import and test the method directly
            from src.orchestration.cv_workflow_graph import CVWorkflowGraph
            
            # Create a minimal instance with mocked dependencies
            workflow_graph = CVWorkflowGraph.__new__(CVWorkflowGraph)
            workflow_graph.session_id = "test-session"
            
            # Act
            result = await workflow_graph.error_handler_node(sample_state)
            
            # Assert
            mock_service_instance.handle_error.assert_called_once()
            call_args = mock_service_instance.handle_error.call_args
            # Check that ContentType.EXPERIENCE was passed as the 3rd argument
            assert call_args[0][2] == ContentType.EXPERIENCE
            assert result.error_messages == []  # Should clear error messages

    async def test_error_handler_fallback_to_qualification_when_none(self, sample_state):
        """Test that error_handler_node falls back to QUALIFICATION when current_content_type is None."""
        # Arrange
        sample_state.error_messages = ["Test error"]
        sample_state.current_content_type = None
        sample_state.current_item_id = "test-item-123"
        sample_state.trace_id = "test-trace-456"
        
        with patch('src.services.error_recovery.ErrorRecoveryService') as mock_service:
            mock_service_instance = MagicMock()
            mock_service.return_value = mock_service_instance
            
            # Mock recovery action with strategy
            mock_recovery_action = MagicMock()
            mock_recovery_action.strategy.value = "immediate_retry"
            mock_service_instance.handle_error = AsyncMock(return_value=mock_recovery_action)
            
            # Import and test the method directly
            from src.orchestration.cv_workflow_graph import CVWorkflowGraph
            
            # Create a minimal instance with mocked dependencies
            workflow_graph = CVWorkflowGraph.__new__(CVWorkflowGraph)
            workflow_graph.session_id = "test-session"
            
            # Act
            result = await workflow_graph.error_handler_node(sample_state)
            
            # Assert
            mock_service_instance.handle_error.assert_called_once()
            call_args = mock_service_instance.handle_error.call_args
            # Check that ContentType.QUALIFICATION was passed as the 3rd argument (fallback)
            assert call_args[0][2] == ContentType.QUALIFICATION
            assert result.error_messages == []  # Should clear error messages

    async def test_error_handler_with_no_error_messages(self, sample_state):
        """Test that error_handler_node returns state unchanged when no error messages."""
        # Arrange
        sample_state.error_messages = []
        sample_state.current_content_type = ContentType.EXPERIENCE
        
        # Import and test the method directly
        from src.orchestration.cv_workflow_graph import CVWorkflowGraph
        
        # Create a minimal instance with mocked dependencies
        workflow_graph = CVWorkflowGraph.__new__(CVWorkflowGraph)
        workflow_graph.session_id = "test-session"
        
        # Act
        result = await workflow_graph.error_handler_node(sample_state)
        
        # Assert
        assert result == sample_state
        assert isinstance(result, AgentState)