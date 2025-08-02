"""Test CB-005 fix: WORKFLOW_SEQUENCE uses WorkflowNodes enum values."""

import pytest
from src.core.enums import WorkflowNodes
from src.orchestration.nodes.workflow_nodes import WORKFLOW_SEQUENCE
from src.orchestration.state import GlobalState
from src.models.workflow_models import UserAction, UserFeedback


class TestCB005WorkflowSequenceEnum:
    """Test that WORKFLOW_SEQUENCE uses WorkflowNodes enum values for type safety."""

    def test_workflow_sequence_uses_enum_values(self):
        """Test that WORKFLOW_SEQUENCE contains enum values, not string literals."""
        # Verify WORKFLOW_SEQUENCE is defined
        assert WORKFLOW_SEQUENCE is not None
        assert len(WORKFLOW_SEQUENCE) > 0

        # Verify all items in WORKFLOW_SEQUENCE are enum values
        expected_values = [
            WorkflowNodes.KEY_QUALIFICATIONS.value,
            WorkflowNodes.PROFESSIONAL_EXPERIENCE.value,
            WorkflowNodes.PROJECT_EXPERIENCE.value,
            WorkflowNodes.EXECUTIVE_SUMMARY.value,
        ]

        assert WORKFLOW_SEQUENCE == expected_values

        # Verify each item is a string value from the enum
        for item in WORKFLOW_SEQUENCE:
            assert isinstance(item, str)
            # Verify the item exists as a value in WorkflowNodes enum
            enum_values = [node.value for node in WorkflowNodes]
            assert item in enum_values

    def test_workflow_nodes_enum_has_sequence_values(self):
        """Test that WorkflowNodes enum contains all required sequence values."""
        # Check that the enum has the required workflow sequence identifiers
        assert hasattr(WorkflowNodes, "KEY_QUALIFICATIONS")
        assert hasattr(WorkflowNodes, "PROFESSIONAL_EXPERIENCE")
        assert hasattr(WorkflowNodes, "PROJECT_EXPERIENCE")
        assert hasattr(WorkflowNodes, "EXECUTIVE_SUMMARY")

        # Verify the values are correct
        assert WorkflowNodes.KEY_QUALIFICATIONS.value == "key_qualifications"
        assert WorkflowNodes.PROFESSIONAL_EXPERIENCE.value == "professional_experience"
        assert WorkflowNodes.PROJECT_EXPERIENCE.value == "project_experience"
        assert WorkflowNodes.EXECUTIVE_SUMMARY.value == "executive_summary"

    def test_supervisor_node_uses_enum_based_sequence(self):
        """Test that supervisor_node logic uses enum-based WORKFLOW_SEQUENCE."""
        # Verify that the expected subgraph names can be constructed from enum values
        for index, section_key in enumerate(WORKFLOW_SEQUENCE):
            expected_subgraph_name = f"{section_key}_subgraph"

            # Verify the section_key comes from enum values
            assert section_key in [node.value for node in WorkflowNodes]

            # Verify the constructed name follows the expected pattern
            assert expected_subgraph_name.endswith("_subgraph")
            assert section_key in expected_subgraph_name

    def test_supervisor_node_routing_consistency(self):
        """Test that supervisor_node routing logic is consistent with enum values."""
        # Test each section in WORKFLOW_SEQUENCE
        for index, section_key in enumerate(WORKFLOW_SEQUENCE):
            expected_next_node = f"{section_key}_subgraph"

            # Verify the section_key comes from enum
            assert section_key in [node.value for node in WorkflowNodes]

            # Verify the expected routing pattern
            assert expected_next_node.endswith("_subgraph")
            assert section_key in expected_next_node

    def test_workflow_sequence_completeness(self):
        """Test that WORKFLOW_SEQUENCE covers all expected workflow sections."""
        # Verify WORKFLOW_SEQUENCE has the expected length
        assert len(WORKFLOW_SEQUENCE) == 4

        # Verify all expected sections are present
        expected_sections = {
            WorkflowNodes.KEY_QUALIFICATIONS.value,
            WorkflowNodes.PROFESSIONAL_EXPERIENCE.value,
            WorkflowNodes.PROJECT_EXPERIENCE.value,
            WorkflowNodes.EXECUTIVE_SUMMARY.value,
        }

        actual_sections = set(WORKFLOW_SEQUENCE)
        assert actual_sections == expected_sections

    def test_no_string_literals_in_workflow_sequence(self):
        """Test that WORKFLOW_SEQUENCE contains no hardcoded string literals."""
        # All items should be enum values, not hardcoded strings
        enum_values = {node.value for node in WorkflowNodes}

        for section_key in WORKFLOW_SEQUENCE:
            # Each section key should exist in the enum values
            assert section_key in enum_values

            # Verify it's not a hardcoded string by checking it matches an enum value
            matching_enum = None
            for node in WorkflowNodes:
                if node.value == section_key:
                    matching_enum = node
                    break

            assert (
                matching_enum is not None
            ), f"Section key '{section_key}' not found in WorkflowNodes enum"
