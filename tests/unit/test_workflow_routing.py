"""Unit tests for workflow routing logic.

Tests the route_after_review function and related workflow routing decisions.
"""

import pytest
import pytest_asyncio
from unittest.mock import Mock
from typing import Dict, Any, List

from src.orchestration.cv_workflow_graph import route_after_review
from src.orchestration.state import AgentState
from src.models.data_models import (
    StructuredCV,
    Section,
    Item,
    ItemStatus,
    UserFeedback,
    UserAction,
)


class TestWorkflowRouting:
    """Test cases for workflow routing logic."""

    @pytest.fixture
    def sample_structured_cv(self):
        """Create a sample structured CV for testing."""
        return StructuredCV(
            sections=[
                Section(
                    name="experience",
                    items=[
                        Item(
                            content="Software Engineer at TechCorp",
                            status=ItemStatus.GENERATED,
                        ),
                        Item(
                            content="Senior Developer at StartupXYZ",
                            status=ItemStatus.INITIAL,
                        ),
                    ],
                ),
                Section(
                    name="education",
                    items=[
                        Item(content="BS Computer Science", status=ItemStatus.INITIAL)
                    ],
                ),
            ]
        )

    @pytest.fixture
    def base_state(self, sample_structured_cv):
        """Create a base agent state for testing."""
        from src.models.data_models import JobDescriptionData

        return AgentState(
            structured_cv=sample_structured_cv,
            job_description_data=JobDescriptionData(
                raw_text="Sample job description",
                parsed_data={"title": "Software Engineer"},
            ),
            current_section_key="key_qualifications",
            current_item_id="exp_1",
            items_to_process_queue=["exp_2"],
            user_feedback=None,
        )

    @pytest.mark.asyncio
    async def test_route_after_review_regenerate_feedback(self, base_state):
        """Test routing when user provides regenerate feedback."""
        # Add regenerate feedback
        feedback = UserFeedback(
            item_id="exp_1",
            action=UserAction.REGENERATE,
            feedback_text="Please make this more detailed",
        )
        base_state.user_feedback = feedback

        result = await route_after_review(base_state.model_dump())

        assert result == "regenerate"

    @pytest.mark.asyncio
    async def test_route_after_review_accept_feedback_with_queue(self, base_state):
        """Test routing when user accepts and there are items in queue."""
        # Add accept feedback
        feedback = UserFeedback(
            item_id="exp_1", action=UserAction.ACCEPT, feedback_text="Looks good"
        )
        base_state.user_feedback = feedback

        result = await route_after_review(base_state.model_dump())

        assert result == "next_item"

    @pytest.mark.asyncio
    async def test_route_after_review_accept_feedback_empty_queue_has_next_section(
        self, base_state
    ):
        """Test routing when queue is empty but next section exists."""
        # Clear queue and add accept feedback
        base_state.items_to_process_queue = []
        feedback = UserFeedback(
            item_id="exp_1", action=UserAction.ACCEPT, feedback_text="Looks good"
        )
        base_state.user_feedback = feedback

        result = await route_after_review(base_state.model_dump())

        assert result == "next_section"

    @pytest.mark.asyncio
    async def test_route_after_review_accept_feedback_no_next_section(self, base_state):
        """Test routing when no more sections to process."""
        # Set to last section and clear queue
        base_state.current_section_key = (
            "executive_summary"  # Last section in WORKFLOW_SEQUENCE
        )
        base_state.items_to_process_queue = []
        feedback = UserFeedback(
            item_id="edu_1", action=UserAction.ACCEPT, feedback_text="Looks good"
        )
        base_state.user_feedback = feedback

        result = await route_after_review(base_state.model_dump())

        assert result == "complete"

    @pytest.mark.asyncio
    async def test_route_after_review_no_feedback_with_queue(self, base_state):
        """Test routing when no feedback but items in queue."""
        # No feedback, but items in queue
        base_state.user_feedback = None

        result = await route_after_review(base_state.model_dump())

        assert result == "next_item"

    @pytest.mark.asyncio
    async def test_route_after_review_no_feedback_empty_queue_has_next_section(
        self, base_state
    ):
        """Test routing when no feedback, empty queue, but next section exists."""
        # No feedback, empty queue
        base_state.user_feedback = None
        base_state.items_to_process_queue = []

        result = await route_after_review(base_state.model_dump())

        assert result == "next_section"

    @pytest.mark.asyncio
    async def test_route_after_review_no_feedback_no_next_section(self, base_state):
        """Test routing when no feedback and no more sections."""
        # Set to last section, no feedback, empty queue
        base_state.current_section_key = "education"
        base_state.user_feedback = None
        base_state.items_to_process_queue = []

        result = await route_after_review(base_state.model_dump())

        assert result == "complete"

    @pytest.mark.asyncio
    async def test_route_after_review_multiple_feedback_latest_wins(self, base_state):
        """Test that latest feedback takes precedence."""
        # Set the latest feedback (simulating the latest user action)
        feedback = UserFeedback(
            item_id="exp_1",
            action=UserAction.REGENERATE,
            feedback_text="Latest feedback",
        )
        base_state.user_feedback = feedback

        result = await route_after_review(base_state.model_dump())

        assert result == "regenerate"

    @pytest.mark.asyncio
    async def test_route_after_review_feedback_for_different_item(self, base_state):
        """Test routing when feedback is for a different item."""
        # Add feedback for different item
        feedback = UserFeedback(
            item_id="exp_2",  # Different from current_item_id
            action=UserAction.REGENERATE,
            feedback_text="Regenerate other item",
        )
        base_state.user_feedback = feedback

        result = await route_after_review(base_state.model_dump())

        # Should still regenerate since feedback action is REGENERATE
        assert result == "regenerate"

    @pytest.mark.asyncio
    async def test_route_after_review_edge_case_empty_structured_cv(self):
        """Test routing with empty structured CV."""
        from src.models.data_models import JobDescriptionData

        state = AgentState(
            structured_cv=StructuredCV(sections=[]),
            job_description_data=JobDescriptionData(
                raw_text="Sample job description",
                parsed_data={"title": "Software Engineer"},
            ),
            current_section_key="experience",
            current_item_id="exp_1",
            items_to_process_queue=[],
            user_feedback=None,
        )

        result = await route_after_review(state.model_dump())

        # Should go to complete when no sections exist
        assert result == "complete"
