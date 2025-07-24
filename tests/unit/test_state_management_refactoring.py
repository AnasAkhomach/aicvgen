"""Tests for STATE-MGT-03: Declarative list updates using typing.Annotated."""

import pytest
from typing import get_type_hints
import operator
from typing import Annotated

from src.orchestration.state import GlobalState, create_global_state


class TestStateManagementRefactoring:
    """Test suite for STATE-MGT-03: Declarative list updates."""

    def test_error_messages_type_annotation(self):
        """Test that error_messages has the correct Annotated type annotation."""
        from typing import get_origin, get_args, List
        
        # Get the type annotation from GlobalState
        annotations = GlobalState.__annotations__
        error_messages_type = annotations.get('error_messages')
        
        # Check that it's the expected type
        assert error_messages_type is not None
        
        # Check that it's an Annotated type
        assert get_origin(error_messages_type) is not None
        
        # Check that the first argument is List[str]
        args = get_args(error_messages_type)
        assert len(args) >= 2
        assert args[0] == List[str]
        assert args[1] == operator.add

    def test_create_global_state_initializes_error_messages(self):
        """Test that create_global_state properly initializes error_messages as empty list."""
        state = create_global_state(
            cv_text="Sample CV text",
            session_id="test_session",
            trace_id="test_trace",
            automated_mode=False
        )
        
        assert "error_messages" in state
        assert isinstance(state["error_messages"], list)
        assert len(state["error_messages"]) == 0

    def test_error_messages_list_behavior(self):
        """Test that error_messages behaves as a proper list."""
        state = create_global_state(
            cv_text="Sample CV text",
            session_id="test_session",
            trace_id="test_trace",
            automated_mode=False
        )
        
        # Test adding single error message
        test_error = "Test error message"
        state["error_messages"].append(test_error)
        assert len(state["error_messages"]) == 1
        assert state["error_messages"][0] == test_error
        
        # Test adding multiple error messages
        additional_errors = ["Error 2", "Error 3"]
        state["error_messages"].extend(additional_errors)
        assert len(state["error_messages"]) == 3
        assert state["error_messages"][1] == "Error 2"
        assert state["error_messages"][2] == "Error 3"

    def test_node_return_pattern_simulation(self):
        """Test that nodes can return simple error_messages lists that will be automatically appended."""
        # Simulate initial state
        state = create_global_state(
            cv_text="Sample CV text",
            session_id="test_session",
            trace_id="test_trace",
            automated_mode=False
        )
        
        # Add some initial errors
        state["error_messages"] = ["Initial error"]
        
        # Simulate a node returning new error messages
        # With Annotated[List[str], operator.add], LangGraph should automatically
        # append the returned list to the existing list
        node_return = {"error_messages": ["New error from node"]}
        
        # Manually simulate what LangGraph would do with operator.add
        # This is what the framework will do automatically
        combined_errors = state["error_messages"] + node_return["error_messages"]
        state["error_messages"] = combined_errors
        
        # Verify the result
        assert len(state["error_messages"]) == 2
        assert "Initial error" in state["error_messages"]
        assert "New error from node" in state["error_messages"]

    def test_multiple_node_returns_simulation(self):
        """Test multiple nodes returning error messages in sequence."""
        # Simulate initial state
        state = create_global_state(
            cv_text="Sample CV text",
            session_id="test_session",
            trace_id="test_trace",
            automated_mode=False
        )
        
        # Simulate multiple nodes returning errors
        node_returns = [
            {"error_messages": ["Parser error"]},
            {"error_messages": ["Validation error"]},
            {"error_messages": ["Processing error"]}
        ]
        
        # Simulate what LangGraph would do with each node return
        for node_return in node_returns:
            if "error_messages" in node_return:
                state["error_messages"] = state["error_messages"] + node_return["error_messages"]
        
        # Verify all errors are accumulated
        assert len(state["error_messages"]) == 3
        assert "Parser error" in state["error_messages"]
        assert "Validation error" in state["error_messages"]
        assert "Processing error" in state["error_messages"]

    def test_empty_error_messages_return(self):
        """Test that returning empty error_messages list doesn't affect state."""
        # Simulate initial state with existing errors
        state = create_global_state(
            cv_text="Sample CV text",
            session_id="test_session",
            trace_id="test_trace",
            automated_mode=False
        )
        state["error_messages"] = ["Existing error"]
        
        # Simulate node returning empty error_messages
        node_return = {"error_messages": []}
        
        # Simulate LangGraph behavior
        state["error_messages"] = state["error_messages"] + node_return["error_messages"]
        
        # Verify existing errors are preserved
        assert len(state["error_messages"]) == 1
        assert "Existing error" in state["error_messages"]

    def test_node_not_returning_error_messages(self):
        """Test that nodes not returning error_messages don't affect the state."""
        # Simulate initial state with existing errors
        state = create_global_state(
            cv_text="Sample CV text",
            session_id="test_session",
            trace_id="test_trace",
            automated_mode=False
        )
        state["error_messages"] = ["Existing error"]
        
        # Simulate node returning other data but no error_messages
        node_return = {"some_other_field": "some_value"}
        
        # Simulate LangGraph behavior - only update fields that are returned
        for key, value in node_return.items():
            if key != "error_messages":
                state[key] = value
        
        # Verify existing errors are preserved
        assert len(state["error_messages"]) == 1
        assert "Existing error" in state["error_messages"]
        assert state["some_other_field"] == "some_value"