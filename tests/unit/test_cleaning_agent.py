"""Unit tests for CleaningAgent error handling fixes."""

import sys
import os

# Ensure project root (containing src/) is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pytest
from unittest.mock import Mock, patch

from src.agents.cleaning_agent import CleaningAgent
from src.agents.agent_base import AgentExecutionContext, AgentResult
from src.utils.agent_error_handling import AgentErrorHandler
from src.models.data_models import ErrorFallbackModel


class TestCleaningAgentErrorHandling:
    """Test error handling in CleaningAgent."""

    @pytest.fixture
    def cleaning_agent(self):
        """Create a CleaningAgent instance for testing."""
        return CleaningAgent(
            llm_service=Mock(),
            error_recovery_service=Mock(),
            progress_tracker=Mock(),
        )

    @pytest.fixture
    def mock_context(self):
        """Create a mock execution context."""
        context = Mock(spec=AgentExecutionContext)
        context.session_id = "test-session-123"
        context.input_data = {
            "raw_output": "Test output",
            "output_type": "big_10_skills",
        }
        return context

    @pytest.mark.asyncio
    async def test_error_handler_call_fix(self, cleaning_agent, mock_context):
        """Test that the correct error handler method is called."""
        from unittest.mock import ANY

        with patch.object(AgentErrorHandler, "handle_general_error") as mock_handler:
            mock_handler.return_value = AgentResult(
                success=False,
                output_data=ErrorFallbackModel(error="Test error"),
                confidence_score=0.0,
                error_message="Test error",
            )

            # Mock the cleaning method to raise an exception
            with patch.object(
                cleaning_agent,
                "_clean_big_10_skills",
                side_effect=Exception("Test error"),
            ):
                # Pass both input_data and context as required by run_async signature
                result = await cleaning_agent.run_async(
                    mock_context.input_data, mock_context
                )

                # Verify the correct error handler method was called
                mock_handler.assert_called_once_with(
                    ANY, "CleaningAgent", context="run_async"
                )

                # Verify the result is properly formatted
                assert not result.success
                assert result.error_message == "Test error"
                assert result.confidence_score == 0.0
                assert isinstance(result.output_data, ErrorFallbackModel)

    @pytest.mark.asyncio
    async def test_no_handle_agent_error_method_called(
        self, cleaning_agent, mock_context
    ):
        """Test that the non-existent handle_agent_error method is not called."""
        # Verify that handle_agent_error doesn't exist
        assert not hasattr(AgentErrorHandler, "handle_agent_error")

        # Mock the cleaning method to raise an exception
        with patch.object(
            cleaning_agent, "_clean_big_10_skills", side_effect=Exception("Test error")
        ):
            # This should not raise AttributeError anymore
            result = await cleaning_agent.run_async(
                mock_context.input_data, mock_context
            )
            assert isinstance(result, AgentResult)
            assert not result.success
            assert isinstance(result.output_data, ErrorFallbackModel)

    @pytest.mark.asyncio
    async def test_successful_execution_no_error_handler(
        self, cleaning_agent, mock_context
    ):
        """Test that error handler is not called on successful execution."""
        with patch.object(AgentErrorHandler, "handle_general_error") as mock_handler:
            with patch.object(
                cleaning_agent,
                "_clean_big_10_skills",
                return_value=["Python", "JavaScript"],
            ), patch.object(
                cleaning_agent,
                "_calculate_confidence_score",
                return_value=1.0,
            ):
                result = await cleaning_agent.run_async(
                    mock_context.input_data, mock_context
                )

                # Verify error handler was not called
                mock_handler.assert_not_called()

                # Verify successful result
                assert result.success
                assert hasattr(result.output_data, "cleaned_data")
                assert result.output_data.cleaned_data == ["Python", "JavaScript"]
