"""Test GlobalState field definitions and accessibility.

This test ensures that all fields used in workflow nodes are properly
defined in the GlobalState TypedDict and can be accessed without errors.
"""

import pytest
from src.orchestration.state import GlobalState, create_global_state


class TestGlobalStateFields:
    """Test GlobalState field definitions."""

    def test_create_global_state_basic(self):
        """Test basic GlobalState creation."""
        state = create_global_state(cv_text="test cv")

        # Test required fields
        assert state["cv_text"] == "test cv"
        assert "session_id" in state
        assert "trace_id" in state

        # Test routing fields
        assert "entry_route" in state
        assert "next_node" in state
        assert "last_executed_node" in state

        # Test agent output fields
        assert "parsed_jd" in state
        assert "research_data" in state
        assert "cv_analysis" in state

    def test_routing_fields_accessibility(self):
        """Test that routing fields can be set and accessed."""
        state = create_global_state(
            cv_text="test cv",
            entry_route="JD_PARSER",
            next_node="SUPERVISOR",
            last_executed_node="ENTRY_ROUTER",
        )

        assert state["entry_route"] == "JD_PARSER"
        assert state["next_node"] == "SUPERVISOR"
        assert state["last_executed_node"] == "ENTRY_ROUTER"

    def test_agent_output_fields_accessibility(self):
        """Test that agent output fields can be set and accessed."""
        test_parsed_jd = {"title": "Software Engineer", "requirements": []}
        test_research_data = {"company_info": "Test Company"}
        test_cv_analysis = {"skills": ["Python", "JavaScript"]}

        state = create_global_state(
            cv_text="test cv",
            parsed_jd=test_parsed_jd,
            research_data=test_research_data,
            cv_analysis=test_cv_analysis,
        )

        assert state["parsed_jd"] == test_parsed_jd
        assert state["research_data"] == test_research_data
        assert state["cv_analysis"] == test_cv_analysis

    def test_state_update_with_new_fields(self):
        """Test that state can be updated with new field values."""
        state = create_global_state(cv_text="test cv")

        # Simulate what entry_router_node does
        updated_state = {
            **state,
            "entry_route": "JD_PARSER",
            "last_executed_node": "ENTRY_ROUTER",
        }

        assert updated_state["entry_route"] == "JD_PARSER"
        assert updated_state["last_executed_node"] == "ENTRY_ROUTER"

        # Simulate what jd_parser_node does
        updated_state = {
            **updated_state,
            "parsed_jd": {"title": "Test Job"},
            "last_executed_node": "JD_PARSER",
        }

        assert updated_state["parsed_jd"] == {"title": "Test Job"}
        assert updated_state["last_executed_node"] == "JD_PARSER"

    def test_all_workflow_fields_present(self):
        """Test that all fields used in workflow nodes are present in GlobalState."""
        state = create_global_state(cv_text="test cv")

        # Fields used in entry_router_node
        assert "parsed_jd" in state
        assert "research_data" in state
        assert "cv_analysis" in state
        assert "entry_route" in state
        assert "last_executed_node" in state

        # Fields used in routing_nodes
        assert "entry_route" in state
        assert "next_node" in state

        # Fields used in parsing_nodes
        assert "job_description_data" in state
        assert "parsed_jd" in state
        assert "research_data" in state
        assert "error_messages" in state
        assert "last_executed_node" in state
