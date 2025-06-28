"""Unit test for agent architecture refactoring."""

import pytest
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.agent_models import AgentResult, AgentExecutionContext
from src.agents.agent_base import AgentBase
from pydantic import BaseModel


class TestOutput(BaseModel):
    """Test output model."""

    result: str
    status: str = "success"


class MockAgent(AgentBase):
    """Mock agent for testing."""

    def __init__(self, session_id: str = "test"):
        super().__init__(
            name="MockAgent",
            description="A mock agent for testing",
            session_id=session_id,
        )

    async def run(self, **kwargs) -> AgentResult:
        """Mock run method."""
        test_output = TestOutput(result="success")
        return AgentResult(
            success=True,
            output_data=test_output,
            metadata={
                "agent_name": self.name,
                "message": "Mock agent executed successfully",
            },
        )


def test_agent_base_structure():
    """Test that the base agent structure is correct."""
    agent = MockAgent()

    assert agent.name == "MockAgent"
    assert agent.description == "A mock agent for testing"
    assert agent.session_id == "test"
    assert hasattr(agent, "logger")
    assert hasattr(agent, "progress_tracker")


def test_agent_result_creation():
    """Test AgentResult creation methods."""
    # Test basic construction
    test_output = TestOutput(result="success")
    success_result = AgentResult(
        success=True,
        output_data=test_output,
        metadata={"agent_name": "TestAgent", "message": "Test successful"},
    )

    assert success_result.success is True
    assert success_result.output_data.result == "success"
    assert success_result.was_successful() is True
    assert success_result.get_error_message() is None

    # Test failure result
    failure_result = AgentResult(
        success=False,
        output_data=None,
        error_message="Test failed",
        metadata={"agent_name": "TestAgent"},
    )

    assert failure_result.success is False
    assert failure_result.output_data is None
    assert failure_result.was_successful() is False
    assert failure_result.get_error_message() == "Test failed"


def test_agent_execution_context():
    """Test AgentExecutionContext structure."""
    context = AgentExecutionContext(
        session_id="test_session",
        item_id="item_123",
        content_type="cv",
        retry_count=1,
        metadata={"test": "data"},
        input_data={"input": "test"},
        processing_options={"option": "value"},
    )

    assert context.session_id == "test_session"
    assert context.item_id == "item_123"
    assert context.content_type == "cv"
    assert context.retry_count == 1
    assert context.metadata == {"test": "data"}
    assert context.input_data == {"input": "test"}
    assert context.processing_options == {"option": "value"}


async def test_mock_agent_run():
    """Test that the mock agent can execute."""
    agent = MockAgent()

    result = await agent.run(test_input="test_data")

    assert isinstance(result, AgentResult)
    assert result.was_successful()
    assert result.output_data.result == "success"


if __name__ == "__main__":
    import asyncio

    # Run basic tests
    print("Testing agent base structure...")
    test_agent_base_structure()
    print("✓ Agent base structure test passed")

    print("Testing AgentResult creation...")
    test_agent_result_creation()
    print("✓ AgentResult creation test passed")

    print("Testing AgentExecutionContext...")
    test_agent_execution_context()
    print("✓ AgentExecutionContext test passed")

    print("Testing mock agent execution...")
    asyncio.run(test_mock_agent_run())
    print("✓ Mock agent execution test passed")

    print("\nAll agent architecture tests passed!")
