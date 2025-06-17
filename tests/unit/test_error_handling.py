"""Unit tests for error handling in the CV workflow graph."""

import pytest
from unittest.mock import AsyncMock, patch
from src.orchestration.cv_workflow_graph import error_handler_node, route_after_review
from src.orchestration.state import AgentState
from src.models.data_models import JobDescriptionData, StructuredCV, UserAction


class TestErrorHandlerNode:
    """Test cases for the error_handler_node function."""

    @pytest.mark.asyncio
    async def test_error_handler_logs_errors(self):
        """Test that error_handler_node logs errors and returns empty dict."""
        # Arrange
        from src.models.data_models import JobDescriptionData, StructuredCV

        test_state = {
            "error_messages": ["Error 1", "Error 2"],
            "user_feedback": None,
            "items_to_process_queue": ["item1"],
            "current_section_key": "key_qualifications",
            "structured_cv": StructuredCV(),
            "job_description_data": JobDescriptionData(raw_text="test job"),
        }

        # Act
        result = await error_handler_node(test_state)

        # Assert
        assert result == {}  # error_handler_node returns empty dict for termination

    @pytest.mark.asyncio
    async def test_error_handler_empty_errors(self):
        """Test error_handler_node with empty error list."""
        # Arrange
        from src.models.data_models import JobDescriptionData, StructuredCV

        test_state = {
            "error_messages": [],
            "user_feedback": None,
            "items_to_process_queue": [],
            "current_section_key": None,
            "structured_cv": StructuredCV(),
            "job_description_data": JobDescriptionData(raw_text="test job"),
        }

        # Act
        result = await error_handler_node(test_state)

        # Assert
        assert result == {}  # error_handler_node returns empty dict for termination


class TestRouteAfterReview:
    """Test cases for the route_after_review function."""

    @pytest.mark.asyncio
    async def test_route_with_errors_returns_error(self):
        """Test that route_after_review returns 'error' when errors are present."""
        # Arrange
        from src.models.data_models import JobDescriptionData, StructuredCV

        test_state = {
            "error_messages": ["Some error occurred"],
            "user_feedback": None,
            "items_to_process_queue": ["item1", "item2"],
            "current_section_key": "key_qualifications",
            "structured_cv": StructuredCV(),
            "job_description_data": JobDescriptionData(raw_text="test job"),
        }

        # Act
        result = await route_after_review(test_state)

        # Assert
        assert result == "error"

    @pytest.mark.asyncio
    async def test_route_with_regenerate_action(self):
        """Test routing when user requests regeneration."""
        # Arrange
        from src.models.data_models import (
            UserFeedback,
            JobDescriptionData,
            StructuredCV,
        )

        test_state = {
            "error_messages": [],
            "user_feedback": UserFeedback(
                action=UserAction.REGENERATE,
                item_id="test_item",
                feedback_text="Please regenerate",
            ),
            "items_to_process_queue": ["item1"],
            "current_section_key": "key_qualifications",
            "structured_cv": StructuredCV(),
            "job_description_data": JobDescriptionData(raw_text="test job"),
        }

        # Act
        result = await route_after_review(test_state)

        # Assert
        assert result == "regenerate"

    @pytest.mark.asyncio
    async def test_route_with_items_in_queue(self):
        """Test routing when there are more items to process."""
        # Arrange
        from src.models.data_models import JobDescriptionData, StructuredCV

        test_state = {
            "error_messages": [],
            "user_feedback": None,
            "items_to_process_queue": ["item1", "item2"],
            "current_section_key": "key_qualifications",
            "structured_cv": StructuredCV(),
            "job_description_data": JobDescriptionData(raw_text="test job"),
        }

        # Act
        result = await route_after_review(test_state)

        # Assert
        assert result == "next_item"

    @pytest.mark.asyncio
    async def test_route_to_next_section(self):
        """Test routing to next section when current section is complete."""
        # Arrange
        from src.models.data_models import JobDescriptionData, StructuredCV

        test_state = {
            "error_messages": [],
            "user_feedback": None,
            "items_to_process_queue": [],
            "current_section_key": "key_qualifications",  # First in WORKFLOW_SEQUENCE
            "structured_cv": StructuredCV(),
            "job_description_data": JobDescriptionData(raw_text="test job"),
        }

        # Act
        result = await route_after_review(test_state)

        # Assert
        assert result == "next_section"

    @pytest.mark.asyncio
    async def test_route_complete_workflow(self):
        """Test routing when all sections are complete."""
        # Arrange
        from src.models.data_models import JobDescriptionData, StructuredCV

        test_state = {
            "error_messages": [],
            "user_feedback": None,
            "items_to_process_queue": [],
            "current_section_key": "executive_summary",  # Last in WORKFLOW_SEQUENCE
            "structured_cv": StructuredCV(),
            "job_description_data": JobDescriptionData(raw_text="test job"),
        }

        # Act
        result = await route_after_review(test_state)

        # Assert
        assert result == "complete"

    @pytest.mark.asyncio
    async def test_error_priority_over_other_conditions(self):
        """Test that error routing takes priority over other conditions."""
        # Arrange
        from src.models.data_models import (
            UserFeedback,
            JobDescriptionData,
            StructuredCV,
        )

        test_state = {
            "error_messages": ["Critical error"],
            "user_feedback": UserFeedback(
                action=UserAction.REGENERATE,
                item_id="test_item",
                feedback_text="Regenerate",
            ),
            "items_to_process_queue": ["item1", "item2"],
            "current_section_key": "key_qualifications",
            "structured_cv": StructuredCV(),
            "job_description_data": JobDescriptionData(raw_text="test job"),
        }

        # Act
        result = await route_after_review(test_state)

        # Assert
        assert result == "error"  # Error should take priority over regenerate
