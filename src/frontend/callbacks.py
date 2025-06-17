# In src/frontend/callbacks.py
import streamlit as st
import asyncio
from typing import Optional
from src.orchestration.state import AgentState, UserFeedback


def handle_user_action(action: str, item_id: str):
    """
    Handle user actions like 'accept' or 'regenerate' for CV items.
    Updates the agent_state in session state and sets a flag to run the backend workflow.
    """
    # Get current state
    agent_state: Optional[AgentState] = st.session_state.get("agent_state")

    if not agent_state:
        st.error("No agent state found")
        return

    # Create user feedback based on action
    if action == "accept":
        feedback_type = "accept"
        feedback_data = {"item_id": item_id}
    elif action == "regenerate":
        feedback_type = "regenerate"
        feedback_data = {"item_id": item_id}
    else:
        st.error(f"Unknown action: {action}")
        return

    # Create UserFeedback object
    user_feedback = UserFeedback(feedback_type=feedback_type, data=feedback_data)

    # Update the agent state with user feedback
    agent_state.user_feedback = user_feedback
    st.session_state.agent_state = agent_state

    # Set flag to trigger backend workflow
    st.session_state.run_workflow = True

    # Show feedback to user
    if action == "accept":
        st.success(f"✅ Item accepted")
    elif action == "regenerate":
        st.info(f"🔄 Regenerating item...")

    # Trigger rerun to process the feedback
    st.rerun()


import threading


def _execute_workflow_in_thread(initial_state: AgentState, trace_id: str):
    """
    The target function for the background thread.
    Executes the LangGraph workflow in a separate thread to avoid blocking the UI.
    """
    from src.orchestration.cv_workflow_graph import cv_graph_app
    from src.config.logging_config import setup_logging

    logger = setup_logging()

    try:
        # Each thread needs its own event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        logger.info(
            "Starting LangGraph workflow in background thread",
            extra={
                "trace_id": trace_id,
                "session_id": st.session_state.get("session_id"),
            },
        )

        # Run the async workflow
        final_state = loop.run_until_complete(
            cv_graph_app.ainvoke(
                initial_state, {"configurable": {"trace_id": trace_id}}
            )
        )

        # Store the result in session_state for the main thread to pick up
        st.session_state.workflow_result = final_state

        logger.info(
            "LangGraph workflow completed successfully in background thread",
            extra={
                "trace_id": trace_id,
                "session_id": st.session_state.get("session_id"),
            },
        )

    except Exception as e:
        logger.error(
            f"LangGraph workflow execution failed in background thread: {str(e)}",
            extra={
                "trace_id": trace_id,
                "session_id": st.session_state.get("session_id"),
                "error_type": type(e).__name__,
            },
        )
        st.session_state.workflow_error = e
    finally:
        st.session_state.processing = False
        loop.close()


def handle_workflow_execution(trace_id: str = None):
    """
    Starts the CV generation workflow in a separate thread to avoid
    blocking the Streamlit UI.

    Args:
        trace_id: Optional trace ID for observability tracking
    """
    from src.config.logging_config import setup_logging
    import uuid

    logger = setup_logging()

    # Generate trace_id if not provided
    if trace_id is None:
        trace_id = str(uuid.uuid4())

    try:
        # If it's the first run, create a new state from UI inputs
        if (
            "agent_state" not in st.session_state
            or st.session_state.agent_state is None
        ):
            from src.core.state_helpers import create_agent_state_from_ui

            st.session_state.agent_state = create_agent_state_from_ui()

        initial_state = st.session_state.agent_state
        initial_state.trace_id = trace_id

        # Clear previous results/errors
        st.session_state.workflow_result = None
        st.session_state.workflow_error = None

        logger.info(
            "Starting non-blocking workflow execution",
            extra={
                "trace_id": trace_id,
                "session_id": st.session_state.get("session_id"),
            },
        )

        # Run the workflow in a background thread
        thread = threading.Thread(
            target=_execute_workflow_in_thread,
            args=(initial_state, trace_id),
        )
        thread.start()

        # The UI will now remain responsive. The main loop will check for the result.

    except Exception as e:
        logger.error(
            f"Failed to start workflow execution: {str(e)}",
            extra={
                "trace_id": trace_id,
                "session_id": st.session_state.get("session_id"),
                "error_type": type(e).__name__,
            },
        )
        st.error(f"Failed to start CV generation: {e}")
        st.session_state.processing = False
