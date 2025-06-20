"""Unit tests for FormatterAgent error handling fixes."""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from src.agents.formatter_agent import FormatterAgent
from src.agents.agent_base import AgentExecutionContext, AgentResult
from src.orchestration.state import AgentState
from src.utils.agent_error_handling import AgentErrorHandler
from src.models.validation_schemas import ValidationError


class TestFormatterAgentErrorHandling:
    """Test error handling in FormatterAgent."""

    @pytest.fixture
    def formatter_agent(self):
        """Create a FormatterAgent instance for testing."""
        return FormatterAgent(name="test_formatter", description="Test formatter agent")

    @pytest.fixture
    def mock_state(self):
        """Create a mock AgentState."""
        state = Mock(spec=AgentState)
        state.error_messages = []
        state.structured_cv = {"sections": []}
        return state

    @pytest.fixture
    def mock_context(self):
        """Create a mock execution context."""
        context = Mock(spec=AgentExecutionContext)
        context.session_id = "test-session-123"
        context.input_data = {
            "content_data": {"sections": []},
            "format_specs": {"format": "pdf"}
        }
        return context

    @pytest.mark.asyncio
    async def test_node_error_handling_fix(self, formatter_agent, mock_state):
        """Test that handle_node_error returns correct dictionary format."""
        with patch.object(formatter_agent, 'run', side_effect=Exception("Test error")):
            result = await formatter_agent.run_as_node(mock_state)
            
            # Verify the result is a dictionary with error_messages
            assert isinstance(result, dict)
            assert "error_messages" in result
            assert isinstance(result["error_messages"], list)
            assert len(result["error_messages"]) > 0
            assert "FormatterAgent Error" in result["error_messages"][0]

    @pytest.mark.asyncio
    async def test_validation_error_handling_fix(self, formatter_agent, mock_context):
        """Test that validation errors are handled correctly."""
        # Mock validation to raise ValidationError
        with patch('src.agents.formatter_agent.validate_agent_input', side_effect=ValidationError("Invalid input")):
            result = await formatter_agent.run_async(mock_context.input_data, mock_context)
            
            # Verify the result is an AgentResult with validation error
            assert isinstance(result, AgentResult)
            assert not result.success
            assert result.confidence_score == 0.0
            assert "validation" in result.error_message.lower() or "invalid" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_successful_validation(self, formatter_agent, mock_context):
        """Test that successful validation works correctly."""
        # Mock successful validation
        mock_validated = Mock()
        mock_validated.model_dump.return_value = {
            "content_data": {"sections": []},
            "format_specs": {"format": "pdf"}
        }
        
        with patch('src.agents.formatter_agent.validate_agent_input', return_value=mock_validated):
            with patch.object(formatter_agent, 'format_content', return_value="Formatted CV"):
                result = await formatter_agent.run_async(mock_context.input_data, mock_context)
                
                # Verify successful result
                assert isinstance(result, AgentResult)
                assert result.success
                assert "formatted_cv_text" in result.output_data
                assert result.output_data["formatted_cv_text"] == "Formatted CV"

    @pytest.mark.asyncio
    async def test_no_error_message_attribute_access(self, formatter_agent, mock_state):
        """Test that we don't access non-existent error_message attribute."""
        with patch.object(formatter_agent, 'run', side_effect=Exception("Test error")):
            # This should not raise AttributeError anymore
            result = await formatter_agent.run_as_node(mock_state)
            
            # Verify we get a proper dictionary result
            assert isinstance(result, dict)
            assert "error_messages" in result

    @pytest.mark.asyncio
    async def test_handle_validation_error_correct_usage(self, formatter_agent, mock_context):
        """Test that handle_validation_error is used correctly with ValidationError instance."""
        validation_error = ValidationError("Test validation error")
        
        with patch('src.agents.formatter_agent.validate_agent_input', side_effect=validation_error):
            with patch.object(AgentErrorHandler, 'handle_validation_error') as mock_handler:
                mock_handler.return_value = AgentResult(
                    success=False,
                    output_data={},
                    confidence_score=0.0,
                    error_message="Validation failed"
                )
                
                result = await formatter_agent.run_async(mock_context.input_data, mock_context)
                
                # Verify the correct method was called with ValidationError instance
                mock_handler.assert_called_once_with(validation_error, "FormatterAgent")
                assert not result.success