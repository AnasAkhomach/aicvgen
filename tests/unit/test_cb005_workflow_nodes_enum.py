"""Unit tests for CB-005: WorkflowNodes enum implementation.

Tests that the WorkflowNodes enum provides type safety for routing decisions
and that all string literals have been replaced with enum values.
"""

import pytest
from unittest.mock import Mock, patch
from src.orchestration.graphs.main_graph import create_cv_workflow_graph_with_di
from src.core.enums import WorkflowNodes
from src.orchestration.state import AgentState
from src.models.workflow_models import UserAction, UserFeedback


class TestWorkflowNodesEnum:
    """Test suite for WorkflowNodes enum implementation."""

    def test_workflow_nodes_enum_exists(self):
        """Test that WorkflowNodes enum is properly defined."""
        # Test that all expected enum members exist
        expected_nodes = [
            "JD_PARSER",
            "CV_PARSER",
            "RESEARCH",
            "CV_ANALYZER",
            "SUPERVISOR",
            "FORMATTER",
            "ERROR_HANDLER",
            "KEY_QUALIFICATIONS_SUBGRAPH",
            "PROFESSIONAL_EXPERIENCE_SUBGRAPH",
            "PROJECTS_SUBGRAPH",
            "EXECUTIVE_SUMMARY_SUBGRAPH",
            "GENERATE",
            "QA",
            "HANDLE_FEEDBACK",
            "REGENERATE",
            "CONTINUE",
            "ERROR",
            "PREPARE_REGENERATION",
        ]

        for node_name in expected_nodes:
            assert hasattr(
                WorkflowNodes, node_name
            ), f"Missing enum member: {node_name}"
            assert isinstance(getattr(WorkflowNodes, node_name).value, str)

    def test_enum_values_are_strings(self):
        """Test that all enum values are strings."""
        for node in WorkflowNodes:
            assert isinstance(
                node.value, str
            ), f"Enum value {node.name} is not a string"

    # Removed test_supervisor_node_uses_enum_values as CVWorkflowGraph no longer exists
    # The enum usage is now tested through the actual workflow graph implementation

    # Removed test_route_after_content_generation_uses_enum_values as CVWorkflowGraph no longer exists
    # The enum usage is now tested through the actual workflow graph implementation

    # Removed test_route_from_supervisor_uses_enum_default as CVWorkflowGraph no longer exists
    # The enum usage is now tested through the actual workflow graph implementation

    # Removed test_build_graph_uses_enum_values as CVWorkflowGraph no longer exists
    # The enum usage is now tested through the actual workflow graph implementation

    def test_enum_provides_type_safety(self):
        """Test that enum provides type safety and prevents typos."""
        # This test demonstrates that using the enum prevents typos
        # that would only be caught at runtime with string literals

        # Valid enum access
        assert WorkflowNodes.SUPERVISOR.value == "supervisor"
        assert WorkflowNodes.FORMATTER.value == "formatter"

        # Invalid enum access would raise AttributeError at development time
        with pytest.raises(AttributeError):
            _ = WorkflowNodes.INVALID_NODE  # This would be caught by IDE/linter

    def test_enum_completeness(self):
        """Test that enum covers all routing scenarios."""
        # Verify that all routing outcomes are covered
        routing_outcomes = [
            WorkflowNodes.REGENERATE.value,
            WorkflowNodes.CONTINUE.value,
            WorkflowNodes.ERROR.value,
            WorkflowNodes.PREPARE_REGENERATION.value,
        ]

        expected_outcomes = ["regenerate", "continue", "error", "prepare_regeneration"]

        for expected in expected_outcomes:
            assert expected in routing_outcomes, f"Missing routing outcome: {expected}"

    def test_enum_consistency(self):
        """Test that enum values are consistent with expected string values."""
        # Test main graph nodes
        assert WorkflowNodes.JD_PARSER.value == "jd_parser"
        assert WorkflowNodes.CV_PARSER.value == "cv_parser"
        assert WorkflowNodes.RESEARCH.value == "research"
        assert WorkflowNodes.CV_ANALYZER.value == "cv_analyzer"
        assert WorkflowNodes.SUPERVISOR.value == "supervisor"
        assert WorkflowNodes.FORMATTER.value == "formatter"
        assert WorkflowNodes.ERROR_HANDLER.value == "error_handler"

        # Test subgraph nodes
        assert (
            WorkflowNodes.KEY_QUALIFICATIONS_SUBGRAPH.value
            == "key_qualifications_subgraph"
        )
        assert (
            WorkflowNodes.PROFESSIONAL_EXPERIENCE_SUBGRAPH.value
            == "professional_experience_subgraph"
        )
        assert WorkflowNodes.PROJECTS_SUBGRAPH.value == "projects_subgraph"
        assert (
            WorkflowNodes.EXECUTIVE_SUMMARY_SUBGRAPH.value
            == "executive_summary_subgraph"
        )

        # Test internal subgraph nodes
        assert WorkflowNodes.GENERATE.value == "generate"
        assert WorkflowNodes.QA.value == "qa"
        assert WorkflowNodes.HANDLE_FEEDBACK.value == "handle_feedback"

        # Test routing outcomes
        assert WorkflowNodes.REGENERATE.value == "regenerate"
        assert WorkflowNodes.CONTINUE.value == "continue"
        assert WorkflowNodes.ERROR.value == "error"
        assert WorkflowNodes.PREPARE_REGENERATION.value == "prepare_regeneration"
