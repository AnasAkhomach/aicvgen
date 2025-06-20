# In src/frontend/callbacks.py
import streamlit as st
import asyncio
from typing import Optional
from ..orchestration.state import AgentState, UserFeedback
from ..services.llm_service import get_llm_service
from ..utils.exceptions import ConfigurationError


def handle_user_action(action: str, item_id: str):
    """
    Handle user actions like 'accept', 'regenerate', or 'validate_api_key'.
    Updates the agent_state in session state and sets a flag to run the backend workflow.
    """
    # Handle API key validation separately
    if action == "validate_api_key":
        handle_api_key_validation()
        return
        
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


def handle_api_key_validation():
    """
    Handle API key validation by calling the LLM service validate_api_key method.
    Updates session state with validation results.
    """
    user_api_key = st.session_state.get("user_gemini_api_key", "")
    
    if not user_api_key:
        st.error("Please enter an API key first")
        return
    
    # Reset validation states
    st.session_state.api_key_validated = False
    st.session_state.api_key_validation_failed = False
    
    try:
        # Show validation in progress
        with st.spinner("Validating API key..."):
            # Get LLM service instance
            llm_service = get_llm_service(user_api_key=user_api_key)
            
            # Run validation asynchronously
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                is_valid = loop.run_until_complete(llm_service.validate_api_key())
            finally:
                loop.close()
            
            if is_valid:
                st.session_state.api_key_validated = True
                st.success("✅ API key is valid and ready to use!")
            else:
                st.session_state.api_key_validation_failed = True
                st.error("❌ API key validation failed. Please check your key.")
                
    except ConfigurationError as e:
        st.session_state.api_key_validation_failed = True
        st.error(f"❌ Configuration error: {e}")
    except Exception as e:
        st.session_state.api_key_validation_failed = True
        st.error(f"❌ Validation failed: {e}")
    
    # Trigger rerun to update UI
    st.rerun()


import threading


def _execute_workflow_in_thread(initial_state: AgentState, trace_id: str):
    """
    The target function for the background thread.
    Executes the LangGraph workflow in a separate thread to avoid blocking the UI.
    """
    from ..orchestration.cv_workflow_graph import cv_graph_app
    from ..config.logging_config import setup_logging

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
    from ..config.logging_config import setup_logging
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
            from ..core.state_helpers import create_initial_agent_state

            st.session_state.agent_state = create_initial_agent_state()

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
