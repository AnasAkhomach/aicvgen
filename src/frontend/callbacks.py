# In src/frontend/callbacks.py
import streamlit as st
import asyncio
import threading
import uuid
from typing import Optional

from ..orchestration.state import AgentState, UserFeedback
from ..utils.exceptions import ConfigurationError
from ..core.dependency_injection import (
    get_container,
    build_llm_service,
    register_core_services,
)
from ..core.state_helpers import create_initial_agent_state

# Import the workflow graph
from ..orchestration.cv_workflow_graph import cv_graph_app


def _execute_workflow_in_thread(state_to_run: AgentState, trace_id: str):
    """
    The target function for the background thread.
    Executes the LangGraph workflow, updating the agent_state directly upon completion.
    """
    from ..config.logging_config import setup_logging

    logger = setup_logging()
    try:        # Each thread needs its own event loop
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
            cv_graph_app.ainvoke(state_to_run, {"configurable": {"trace_id": trace_id}})
        )

        # Update the session state with the final results
        st.session_state.agent_state = final_state

        logger.info(
            "LangGraph workflow completed successfully",
            extra={"trace_id": trace_id},
        )

    except Exception as e:
        logger.error(
            f"LangGraph workflow failed in background thread: {str(e)}",
            extra={"trace_id": trace_id, "error_type": type(e).__name__},
            exc_info=True,
        )
        st.session_state.workflow_error = e
    finally:
        st.session_state.is_processing = False
        st.session_state.just_finished = True
        loop.close()


def _start_workflow_thread(state_to_run: AgentState):
    """Helper to configure and start the background workflow thread."""
    from ..config.logging_config import setup_logging

    logger = setup_logging()
    trace_id = str(uuid.uuid4())
    state_to_run.trace_id = trace_id    # Set flags to indicate processing has started
    st.session_state.is_processing = True
    st.session_state.workflow_error = None  # Clear previous errors
    st.session_state.just_finished = False

    logger.info(
        "Starting workflow thread",
        extra={"trace_id": trace_id, "session_id": st.session_state.get("session_id")},
    )
    
    thread = threading.Thread(
        target=_execute_workflow_in_thread, args=(state_to_run, trace_id), daemon=True
    )
    thread.start()
    # Note: st.rerun() removed - calling within callback is not allowed


def start_cv_generation():
    """Initializes state from UI inputs and starts the CV generation workflow."""
    initial_state = create_initial_agent_state()
    st.session_state.agent_state = initial_state
    _start_workflow_thread(initial_state)


def handle_user_action(action: str, item_id: str):
    """
    Handle user actions like 'accept', 'regenerate', or 'validate_api_key'.
    """
    if action == "validate_api_key":
        handle_api_key_validation()
        return

    agent_state: Optional[AgentState] = st.session_state.get("agent_state")
    if not agent_state:
        st.error("No agent state found")
        return

    # Create user feedback based on action
    if action == "accept":
        feedback_type = "accept"
        feedback_data = {"item_id": item_id}
        st.success(f"✅ Item accepted")
    elif action == "regenerate":
        feedback_type = "regenerate"
        feedback_data = {"item_id": item_id}
        st.info(f"🔄 Regenerating item...")
    else:
        st.error(f"Unknown action: {action}")
        return

    agent_state.user_feedback = UserFeedback(
        feedback_type=feedback_type, data=feedback_data
    )
    st.session_state.agent_state = agent_state

    # Start the workflow to process the user feedback
    _start_workflow_thread(agent_state)


def handle_api_key_validation(llm_service=None):
    """
    Handle API key validation by calling the LLM service validate_api_key method.
    Updates session state with validation results.
    Accepts an optional llm_service for DI/testing.
    """
    import logging

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
            # Use injected llm_service if provided, else create one using DI
            if llm_service is None:
                container = get_container()
                # Ensure core services like settings are registered
                if not container._registrations.get("settings"):
                    register_core_services(container)

                # Build the LLM service with the user-provided API key
                llm_service = build_llm_service(container, user_api_key=user_api_key)

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
                logging.error(
                    "Gemini API key validation failed: No exception, returned False."
                )

    except ConfigurationError as e:
        st.session_state.api_key_validation_failed = True
        st.error(f"❌ Configuration error: {e}")
        logging.error(f"Gemini API key configuration error: {e}")
    except Exception as e:
        st.session_state.api_key_validation_failed = True
        st.error(f"❌ Validation failed: {e}")
        logging.error(f"Gemini API key validation exception: {e}", exc_info=True)

    # Note: st.rerun() removed - calling within callback is not allowed
