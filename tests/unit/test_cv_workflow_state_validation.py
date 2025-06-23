import sys
import os
import pytest

# Ensure project root (containing src/) is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.orchestration.cv_workflow_graph import (
    should_continue_generation,
    route_after_qa,
)
from src.orchestration.state import AgentState
from src.models.data_models import (
    StructuredCV,
    JobDescriptionData,
    UserFeedback,
    UserAction,
)


def minimal_state(**kwargs):
    return AgentState(
        structured_cv=StructuredCV(sections=[]),
        job_description_data=JobDescriptionData(raw_text="test"),
        **kwargs
    )


def test_should_continue_generation_error():
    state = minimal_state(error_messages=["fail"])
    result = should_continue_generation(state.model_dump())
    assert result == "error"


def test_should_continue_generation_continue():
    state = minimal_state(content_generation_queue=["id1", "id2"])
    result = should_continue_generation(state.model_dump())
    assert result == "continue"


def test_should_continue_generation_complete():
    state = minimal_state()
    result = should_continue_generation(state.model_dump())
    assert result == "complete"


@pytest.mark.asyncio
async def test_route_after_qa_error():
    state = minimal_state(error_messages=["fail"])
    result = await route_after_qa(state.model_dump())
    assert result == "error"


@pytest.mark.asyncio
async def test_route_after_qa_regenerate():
    feedback = UserFeedback(action=UserAction.REGENERATE, item_id="id1")
    state = minimal_state(user_feedback=feedback)
    result = await route_after_qa(state.model_dump())
    assert result == "regenerate"


@pytest.mark.asyncio
async def test_route_after_qa_continue():
    state = minimal_state()
    result = await route_after_qa(state.model_dump())
    assert result == "complete"


@pytest.mark.asyncio
async def test_route_after_qa_regenerate_takes_precedence_over_error():
    """If both user feedback (regenerate) and error are present, user intent should win."""
    feedback = UserFeedback(action=UserAction.REGENERATE, item_id="id1")
    state = minimal_state(user_feedback=feedback, error_messages=["fail"])
    result = await route_after_qa(state.model_dump())
    assert result == "regenerate"
