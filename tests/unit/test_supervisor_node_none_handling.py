"""Test supervisor node handling of None current_section_index.

This test verifies that the supervisor_node function correctly handles
the case where current_section_index is None (no sections with items).
"""

import pytest
from src.orchestration.nodes.workflow_nodes import supervisor_node
from src.orchestration.state import create_global_state
from src.core.enums import WorkflowNodes


class TestSupervisorNodeNoneHandling:
    """Test supervisor node None handling."""

    @pytest.mark.asyncio
    async def test_supervisor_node_handles_none_current_section_index(self):
        """Test that supervisor_node handles None current_section_index correctly."""
        # Arrange - Create state with None current_section_index
        state = create_global_state(
            cv_text="test cv",
            current_section_index=None,
            workflow_sections=["key_qualifications", "professional_experience"],
        )

        # Act
        result = await supervisor_node(state)

        # Assert
        assert (
            result["next_node"] == "END"
        )  # MVP: Direct completion instead of formatter
        assert result["last_executed_node"] == "SUPERVISOR"
        assert "current_section_index" in result

    @pytest.mark.asyncio
    async def test_supervisor_node_handles_valid_current_section_index(self):
        """Test that supervisor_node handles valid current_section_index correctly."""
        # Arrange - Create state with valid current_section_index
        state = create_global_state(
            cv_text="test cv",
            current_section_index=0,
            workflow_sections=["key_qualifications", "professional_experience"],
        )

        # Act
        result = await supervisor_node(state)

        # Assert
        assert "next_node" in result
        assert result["last_executed_node"] == "SUPERVISOR"
        assert "current_section_index" in result

    @pytest.mark.asyncio
    async def test_supervisor_node_handles_errors_first(self):
        """Test that supervisor_node prioritizes error handling."""
        # Arrange - Create state with errors and None current_section_index
        state = create_global_state(
            cv_text="test cv", current_section_index=None, error_messages=["Test error"]
        )

        # Act
        result = await supervisor_node(state)

        # Assert
        assert result["next_node"] == WorkflowNodes.ERROR_HANDLER.value
        assert result["last_executed_node"] == "SUPERVISOR"
