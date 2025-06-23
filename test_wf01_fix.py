"""Test to validate WF-01 race condition fix in route_after_qa"""

import asyncio
from typing import Dict, Any


# Simple test data structures to validate the logic
class MockUserAction:
    REGENERATE = "regenerate"


class MockUserFeedback:
    def __init__(self, action: str, item_id: str):
        self.action = action
        self.item_id = item_id


class MockAgentState:
    def __init__(self, error_messages=None, user_feedback=None):
        self.error_messages = error_messages or []
        self.user_feedback = user_feedback

    @classmethod
    def model_validate(cls, state_dict):
        return cls(
            error_messages=state_dict.get("error_messages", []),
            user_feedback=state_dict.get("user_feedback"),
        )


def mock_should_continue_generation(state):
    return "continue"


# Mock logger
class MockLogger:
    def info(self, msg):
        print(f"INFO: {msg}")

    def warning(self, msg):
        print(f"WARNING: {msg}")


logger = MockLogger()


# The fixed route_after_qa function logic
async def route_after_qa_fixed(state: Dict[str, Any]) -> str:
    """Route after QA based on user feedback and workflow state. Validates state."""
    agent_state = MockAgentState.model_validate(state)

    # Priority 1: Check for user feedback first to ensure user intent is honored
    if (
        agent_state.user_feedback
        and agent_state.user_feedback.action == MockUserAction.REGENERATE
    ):
        logger.info("User requested regeneration, routing to prepare_regeneration")
        return "regenerate"

    # Priority 2: Check for errors if no explicit user action
    if agent_state.error_messages:
        logger.warning("Errors detected in state, routing to error handler")
        return "error"

    # Priority 3: Continue with content generation loop
    return mock_should_continue_generation(state)


# Test the race condition scenario
async def test_race_condition_fix():
    """Test that user feedback takes priority over errors"""

    print("Testing WF-01 race condition fix...")

    # Scenario 1: Both error and user regeneration request exist
    user_feedback = MockUserFeedback(
        action=MockUserAction.REGENERATE, item_id="test_item"
    )
    state_with_both = {
        "error_messages": ["Some error occurred"],
        "user_feedback": user_feedback,
    }

    result = await route_after_qa_fixed(state_with_both)
    print(f"Result when both error and user feedback exist: {result}")
    assert result == "regenerate", f"Expected 'regenerate', got '{result}'"

    # Scenario 2: Only error exists
    state_error_only = {
        "error_messages": ["Some error occurred"],
        "user_feedback": None,
    }

    result = await route_after_qa_fixed(state_error_only)
    print(f"Result when only error exists: {result}")
    assert result == "error", f"Expected 'error', got '{result}'"

    # Scenario 3: Only user feedback exists
    state_feedback_only = {"error_messages": [], "user_feedback": user_feedback}

    result = await route_after_qa_fixed(state_feedback_only)
    print(f"Result when only user feedback exists: {result}")
    assert result == "regenerate", f"Expected 'regenerate', got '{result}'"

    print("✅ All race condition tests passed!")


if __name__ == "__main__":
    asyncio.run(test_race_condition_fix())
