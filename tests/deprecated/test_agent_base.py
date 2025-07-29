"""
Unit tests for AgentBase class.

This module tests the Template Method pattern implementation in AgentBase,
including error handling and validation logic.
"""

from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
from pydantic import BaseModel

from src.agents.agent_base import AgentBase
from src.error_handling.exceptions import AgentExecutionError
from src.models.agent_models import AgentResult


class MockOutputData(BaseModel):
    """Mock output data model for testing."""

    result: str


class MockAgentSuccess(AgentBase):
    """Mock agent that always succeeds for testing."""

    def _validate_inputs(self, input_data: dict) -> None:
        """Mock validation that always passes."""
        if not input_data.get("valid"):
            raise AgentExecutionError(
                agent_name=self.name,
                message="Validation failed - 'valid' field is required",
            )

    async def _execute(self, **kwargs: Any) -> AgentResult:
        """Mock execution that always succeeds."""
        return AgentResult(
            success=True,
            output_data=MockOutputData(result="success"),
            metadata={"agent_name": self.name, "message": "Mock execution successful"},
        )


class MockAgentValidationError(AgentBase):
    """Mock agent that always fails validation."""

    def _validate_inputs(self, input_data: dict) -> None:
        """Mock validation that always fails."""
        raise AgentExecutionError(agent_name=self.name, message="Mock validation error")

    async def _execute(self, **kwargs: Any) -> AgentResult:
        """This should never be called due to validation failure."""
        return AgentResult(
            success=True,
            output_data=MockOutputData(result="should not reach here"),
            metadata={"agent_name": self.name, "message": "Should not reach here"},
        )


class MockAgentExecutionError(AgentBase):
    """Mock agent that fails during execution."""

    def _validate_inputs(self, input_data: dict) -> None:
        """Mock validation that always passes."""
        pass

    async def _execute(self, **kwargs: Any) -> AgentResult:
        """Mock execution that always fails with AgentExecutionError."""
        raise AgentExecutionError(agent_name=self.name, message="Mock execution error")


class MockAgentUnexpectedError(AgentBase):
    """Mock agent that fails with unexpected error."""

    def _validate_inputs(self, input_data: dict) -> None:
        """Mock validation that always passes."""
        pass

    async def _execute(self, **kwargs: Any) -> AgentResult:
        """Mock execution that always fails with unexpected error."""
        raise ValueError("Unexpected error occurred")


class TestAgentBase:
    """Test suite for AgentBase template method pattern."""

    def setup_method(self):
        """Set up test fixtures."""
        self.session_id = "test-session-123"

    @pytest.mark.asyncio
    async def test_successful_execution_path(self):
        """Test the successful execution path through the template method."""
        agent = MockAgentSuccess(
            name="TestAgent",
            description="Test agent for successful execution",
            session_id=self.session_id,
        )

        # Mock the progress tracker
        agent.progress_tracker = Mock()

        result = await agent.run(input_data={"valid": True})

        assert result.success is True
        assert result.output_data.result == "success"
        assert result.error_message is None
        assert result.metadata["agent_name"] == "TestAgent"

    @pytest.mark.asyncio
    async def test_validation_error_handling(self):
        """Test that validation errors are properly caught and handled."""
        agent = MockAgentValidationError(
            name="TestAgent",
            description="Test agent for validation error",
            session_id=self.session_id,
        )

        # Mock the progress tracker
        agent.progress_tracker = Mock()

        result = await agent.run(input_data={"invalid": True})

        assert result.success is False
        assert "Mock validation error" in result.error_message
        assert result.output_data is None
        assert result.metadata["agent_name"] == "TestAgent"

    @pytest.mark.asyncio
    async def test_agent_execution_error_handling(self):
        """Test that AgentExecutionError from _execute is properly handled."""
        agent = MockAgentExecutionError(
            name="TestAgent",
            description="Test agent for execution error",
            session_id=self.session_id,
        )

        # Mock the progress tracker
        agent.progress_tracker = Mock()

        result = await agent.run(input_data={"valid": True})

        assert result.success is False
        assert "Mock execution error" in result.error_message
        assert result.output_data is None
        assert result.metadata["agent_name"] == "TestAgent"

    @pytest.mark.asyncio
    async def test_unexpected_error_handling(self):
        """Test that unexpected errors are caught and handled."""
        agent = MockAgentUnexpectedError(
            name="TestAgent",
            description="Test agent for unexpected error",
            session_id=self.session_id,
        )

        # Mock the progress tracker
        agent.progress_tracker = Mock()

        result = await agent.run(input_data={"valid": True})

        assert result.success is False
        assert (
            "An unexpected error occurred: Unexpected error occurred"
            in result.error_message
        )
        assert result.output_data is None
        assert result.metadata["agent_name"] == "TestAgent"

    @pytest.mark.asyncio
    async def test_input_validation_with_empty_input_data(self):
        """Test the template method handles empty input_data correctly."""
        agent = MockAgentSuccess(
            name="TestAgent",
            description="Test agent for empty input",
            session_id=self.session_id,
        )

        # Mock the progress tracker
        agent.progress_tracker = Mock()

        # Should fail validation because 'valid' field is missing
        result = await agent.run(input_data={})

        assert result.success is False
        assert "Validation failed" in result.error_message

    @pytest.mark.asyncio
    async def test_input_validation_with_no_input_data(self):
        """Test the template method handles missing input_data correctly."""
        agent = MockAgentSuccess(
            name="TestAgent",
            description="Test agent for missing input",
            session_id=self.session_id,
        )

        # Mock the progress tracker
        agent.progress_tracker = Mock()

        # Should fail validation because 'valid' field is missing
        result = await agent.run()

        assert result.success is False
        assert "Validation failed" in result.error_message

    @pytest.mark.asyncio
    async def test_progress_tracking_calls(self):
        """Test that progress tracking is called at expected points."""
        agent = MockAgentSuccess(
            name="TestAgent",
            description="Test agent for progress tracking",
            session_id=self.session_id,
        )

        # Mock the progress tracker
        mock_progress_tracker = Mock()
        agent.progress_tracker = mock_progress_tracker

        result = await agent.run(input_data={"valid": True})

        assert result.success is True

        # Check that update_progress was called
        mock_progress_tracker.update_progress.assert_called()

        # Verify the calls were made with expected parameters
        calls = mock_progress_tracker.update_progress.call_args_list
        assert len(calls) >= 2  # At least start and validation passed calls

        # Check first call (start)
        assert calls[0][0][0] == "TestAgent"  # agent name
        assert calls[0][0][1] == 0  # progress 0%
        assert "Starting TestAgent execution" in calls[0][0][2]  # message

        # Check second call (validation passed)
        assert calls[1][0][0] == "TestAgent"  # agent name
        assert calls[1][0][1] == 20  # progress 20%
        assert "Input validation passed" in calls[1][0][2]  # message

    def test_agent_initialization(self):
        """Test that agents are properly initialized."""
        agent = MockAgentSuccess(
            name="TestAgent",
            description="Test agent description",
            session_id=self.session_id,
        )

        assert agent.name == "TestAgent"
        assert agent.description == "Test agent description"
        assert agent.session_id == self.session_id
        assert agent.logger is not None
        assert agent.progress_tracker is None  # Should be None until set

    def test_progress_tracker_setter(self):
        """Test the progress tracker can be set and used."""
        agent = MockAgentSuccess(
            name="TestAgent", description="Test agent", session_id=self.session_id
        )

        mock_tracker = Mock()
        agent.set_progress_tracker(mock_tracker)

        assert agent.progress_tracker is mock_tracker

        # Test update_progress method
        agent.update_progress(50, "Test message")
        mock_tracker.update_progress.assert_called_once_with(
            "TestAgent", 50, "Test message"
        )
