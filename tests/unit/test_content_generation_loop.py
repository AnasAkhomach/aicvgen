#!/usr/bin/env python3
"""
Unit tests for the new content generation loop nodes and router functions.
Tests the explicit item-by-item processing workflow.
"""

import pytest
from unittest.mock import Mock, AsyncMock
from typing import Dict, Any, List

from src.orchestration.cv_workflow_graph import (
    setup_generation_queue_node,
    pop_next_item_node,
    prepare_regeneration_node,
    should_continue_generation,
    route_after_qa
)
from src.orchestration.state import AgentState
from src.models.data_models import (
    StructuredCV, Section, Item, ItemStatus, ItemType,
    UserFeedback, UserAction, JobDescriptionData
)


class TestSetupGenerationQueueNode:
    """Test the setup_generation_queue_node function."""

    @pytest.fixture
    def mock_state(self):
        """Create a mock AgentState with structured CV data."""
        # Create mock items
        item1 = Item(
            content="Test experience 1",
            status=ItemStatus.PENDING,
            item_type=ItemType.EXPERIENCE
        )
        item2 = Item(
            content="Test experience 2",
            status=ItemStatus.PENDING,
            item_type=ItemType.EXPERIENCE
        )
        item3 = Item(
            content="Test project 1",
            status=ItemStatus.PENDING,
            item_type=ItemType.PROJECT
        )

        # Create mock sections
        experience_section = Section(
            name="Professional Experience",
            items=[item1, item2]
        )
        project_section = Section(
            name="Project Experience",
            items=[item3]
        )

        # Create mock structured CV
        structured_cv = StructuredCV(
            sections=[experience_section, project_section]
        )

        return AgentState(
            structured_cv=structured_cv,
            job_description_data=JobDescriptionData(raw_text="Test job description")
        )

    @pytest.mark.asyncio
    async def test_setup_generation_queue_success(self, mock_state):
        """Test successful setup of content generation queue."""
        result = await setup_generation_queue_node(mock_state)
        
        assert "content_generation_queue" in result
        queue = result["content_generation_queue"]
        
        # Should have 3 items (2 from experience + 1 from projects)
        assert len(queue) == 3
        
        # All items should be string IDs
        for item_id in queue:
            assert isinstance(item_id, str)

    @pytest.mark.asyncio
    async def test_setup_generation_queue_with_subsections(self):
        """Test setup with subsections."""
        # Create items for subsection
        subsection_item = Item(
            content="Subsection item",
            status=ItemStatus.PENDING,
            item_type=ItemType.QUALIFICATION
        )
        
        # Create subsection
        subsection = Section(
            name="Key Skills",
            items=[subsection_item]
        )
        
        # Create main section with subsection
        main_section = Section(
            name="Qualifications",
            items=[],
            subsections=[subsection]
        )
        
        structured_cv = StructuredCV(sections=[main_section])
        state = AgentState(
            structured_cv=structured_cv,
            job_description_data=JobDescriptionData(raw_text="Test job description")
        )
        
        result = await setup_generation_queue_node(state)
        
        assert "content_generation_queue" in result
        queue = result["content_generation_queue"]
        
        # Should have 1 item from subsection
        assert len(queue) == 1


class TestPopNextItemNode:
    """Test the pop_next_item_node function."""

    @pytest.mark.asyncio
    async def test_pop_next_item_success(self):
        """Test successful popping of next item from queue."""
        state = AgentState(
            structured_cv=StructuredCV(sections=[]),
            job_description_data=JobDescriptionData(raw_text="Test"),
            content_generation_queue=["item1", "item2", "item3"]
        )
        
        result = await pop_next_item_node(state)
        
        assert "current_item_id" in result
        assert "content_generation_queue" in result
        
        assert result["current_item_id"] == "item1"
        assert result["content_generation_queue"] == ["item2", "item3"]

    @pytest.mark.asyncio
    async def test_pop_next_item_empty_queue(self):
        """Test popping from empty queue."""
        state = AgentState(
            structured_cv=StructuredCV(sections=[]),
            job_description_data=JobDescriptionData(raw_text="Test"),
            content_generation_queue=[]
        )
        
        result = await pop_next_item_node(state)
        
        # Should return empty dict when queue is empty
        assert result == {}

    @pytest.mark.asyncio
    async def test_pop_next_item_single_item(self):
        """Test popping the last item from queue."""
        state = AgentState(
            structured_cv=StructuredCV(sections=[]),
            job_description_data=JobDescriptionData(raw_text="Test"),
            content_generation_queue=["last_item"]
        )
        
        result = await pop_next_item_node(state)
        
        assert result["current_item_id"] == "last_item"
        assert result["content_generation_queue"] == []


class TestPrepareRegenerationNode:
    """Test the prepare_regeneration_node function."""

    @pytest.mark.asyncio
    async def test_prepare_regeneration_success(self):
        """Test successful preparation of regeneration."""
        user_feedback = UserFeedback(
            action=UserAction.REGENERATE,
            item_id="target_item_123",
            feedback_text="Please improve this content"
        )
        
        state = AgentState(
            structured_cv=StructuredCV(sections=[]),
            job_description_data=JobDescriptionData(raw_text="Test"),
            user_feedback=user_feedback
        )
        
        result = await prepare_regeneration_node(state)
        
        assert "content_generation_queue" in result
        assert "current_item_id" in result
        assert "is_initial_generation" in result
        
        assert result["content_generation_queue"] == ["target_item_123"]
        assert result["current_item_id"] is None
        assert result["is_initial_generation"] is False

    @pytest.mark.asyncio
    async def test_prepare_regeneration_no_feedback(self):
        """Test preparation when no user feedback is provided."""
        state = AgentState(
            structured_cv=StructuredCV(sections=[]),
            job_description_data=JobDescriptionData(raw_text="Test"),
            user_feedback=None
        )
        
        result = await prepare_regeneration_node(state)
        
        assert "error_messages" in result
        assert "No item specified for regeneration" in result["error_messages"][0]

    @pytest.mark.asyncio
    async def test_prepare_regeneration_no_item_id(self):
        """Test preparation when user feedback has no item_id."""
        user_feedback = UserFeedback(
            action=UserAction.REGENERATE,
            item_id=None,
            feedback_text="General feedback"
        )
        
        state = AgentState(
            structured_cv=StructuredCV(sections=[]),
            job_description_data=JobDescriptionData(raw_text="Test"),
            user_feedback=user_feedback
        )
        
        result = await prepare_regeneration_node(state)
        
        assert "error_messages" in result
        assert "No item specified for regeneration" in result["error_messages"][0]


class TestShouldContinueGeneration:
    """Test the should_continue_generation router function."""

    def test_should_continue_with_items_in_queue(self):
        """Test router when there are items in the content generation queue."""
        state_dict = {
            "structured_cv": StructuredCV(sections=[]).model_dump(),
            "job_description_data": JobDescriptionData(raw_text="Test").model_dump(),
            "content_generation_queue": ["item1", "item2"],
            "error_messages": []
        }
        
        result = should_continue_generation(state_dict)
        assert result == "continue"

    def test_should_continue_empty_queue(self):
        """Test router when content generation queue is empty."""
        state_dict = {
            "structured_cv": StructuredCV(sections=[]).model_dump(),
            "job_description_data": JobDescriptionData(raw_text="Test").model_dump(),
            "content_generation_queue": [],
            "error_messages": []
        }
        
        result = should_continue_generation(state_dict)
        assert result == "complete"

    def test_should_continue_with_errors(self):
        """Test router when there are error messages."""
        state_dict = {
            "structured_cv": StructuredCV(sections=[]).model_dump(),
            "job_description_data": JobDescriptionData(raw_text="Test").model_dump(),
            "content_generation_queue": ["item1"],
            "error_messages": ["Some error occurred"]
        }
        
        result = should_continue_generation(state_dict)
        assert result == "error"


class TestRouteAfterQA:
    """Test the route_after_qa router function."""

    @pytest.mark.asyncio
    async def test_route_after_qa_with_errors(self):
        """Test routing when there are error messages."""
        state_dict = {
            "structured_cv": StructuredCV(sections=[]).model_dump(),
            "job_description_data": JobDescriptionData(raw_text="Test").model_dump(),
            "content_generation_queue": ["item1"],
            "error_messages": ["QA failed"],
            "user_feedback": None
        }
        
        result = await route_after_qa(state_dict)
        assert result == "error"

    @pytest.mark.asyncio
    async def test_route_after_qa_with_regeneration_feedback(self):
        """Test routing when user requests regeneration."""
        user_feedback = UserFeedback(
            action=UserAction.REGENERATE,
            item_id="item123",
            feedback_text="Please improve"
        )
        
        state_dict = {
            "structured_cv": StructuredCV(sections=[]).model_dump(),
            "job_description_data": JobDescriptionData(raw_text="Test").model_dump(),
            "content_generation_queue": ["item1"],
            "error_messages": [],
            "user_feedback": user_feedback.model_dump()
        }
        
        result = await route_after_qa(state_dict)
        assert result == "regenerate"

    @pytest.mark.asyncio
    async def test_route_after_qa_continue_generation(self):
        """Test routing when should continue with generation loop."""
        state_dict = {
            "structured_cv": StructuredCV(sections=[]).model_dump(),
            "job_description_data": JobDescriptionData(raw_text="Test").model_dump(),
            "content_generation_queue": ["item1", "item2"],
            "error_messages": [],
            "user_feedback": None
        }
        
        result = await route_after_qa(state_dict)
        assert result == "continue"

    @pytest.mark.asyncio
    async def test_route_after_qa_complete(self):
        """Test routing when generation is complete."""
        state_dict = {
            "structured_cv": StructuredCV(sections=[]).model_dump(),
            "job_description_data": JobDescriptionData(raw_text="Test").model_dump(),
            "content_generation_queue": [],
            "error_messages": [],
            "user_feedback": None
        }
        
        result = await route_after_qa(state_dict)
        assert result == "complete"