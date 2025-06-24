# In src/frontend/callbacks.py
import streamlit as st
import asyncio
import threading
import uuid
from typing import Optional

from ..orchestration.state import AgentState, UserFeedback
from ..utils.exceptions import ConfigurationError
from ..core.dependency_injection import get_container
from ..utils.state_utils import create_initial_agent_state
from ..core.application_startup import get_startup_manager
from ..models.data_models import UserAction

# Import the workflow graph class
from ..orchestration.cv_workflow_graph import CVWorkflowGraph
from ..services.llm_service import EnhancedLLMService


def get_workflow_graph() -> CVWorkflowGraph:
    """Get or create the workflow graph for the current session."""
    if "workflow_graph" not in st.session_state:
        session_id = st.session_state.get("session_id")
        if not session_id:
            session_id = str(uuid.uuid4())
            st.session_state.session_id = session_id
        # Ensure application is initialized before creating workflow
        startup_manager = get_startup_manager()
        if not startup_manager.is_initialized:
            startup_manager.initialize_application()
        st.session_state.workflow_graph = CVWorkflowGraph(session_id=session_id)
    return st.session_state.workflow_graph


def _execute_workflow_in_thread(state_to_run: AgentState, trace_id: str):
    """
    The target function for the background thread.
    Executes the LangGraph workflow, updating the agent_state directly upon completion.
    """
    from ..config.logging_config import setup_logging
    import logging

    setup_logging()
    logger = logging.getLogger(__name__)
    try:  # Each thread needs its own event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        logger.info(
            "Starting LangGraph workflow in background thread",
            extra={
                "trace_id": trace_id,
                "session_id": st.session_state.get("session_id"),
            },
        )

        # Get the session-specific workflow graph
        workflow_graph = get_workflow_graph()

        # Run the async workflow
        final_state = loop.run_until_complete(workflow_graph.invoke(state_to_run))

        # Update the session state with the final results
        st.session_state.agent_state = final_state

        logger.info(
            "LangGraph workflow completed successfully",
            extra={"trace_id": trace_id},
        )

    except Exception as e:  # noqa: BLE001
        logger.error(
            "LangGraph workflow failed in background thread: %s",
            str(e),
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
    import logging

    setup_logging()
    logger = logging.getLogger(__name__)
    trace_id = str(uuid.uuid4())
    state_to_run.trace_id = trace_id  # Set flags to indicate processing has started
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


def start_cv_generation():
    """Initializes state from UI inputs and starts the CV generation workflow."""
    job_desc_raw = st.session_state.get("job_description_input", "")
    cv_text = st.session_state.get("cv_text_input", "")
    start_from_scratch = st.session_state.get("start_from_scratch_input", False)

    initial_state = create_initial_agent_state(
        job_description_raw=job_desc_raw,
        cv_text=cv_text,
        start_from_scratch=start_from_scratch,
    )
    st.session_state.agent_state = initial_state
    _start_workflow_thread(initial_state)


def handle_user_action(action: str, item_id: str):
    """
    Handle user actions like 'accept' or 'regenerate'.
    """
    agent_state: Optional[AgentState] = st.session_state.get("agent_state")
    if not agent_state:
        st.error("No agent state found")
        return

    # Create user feedback based on action
    if action == "accept":
        user_action = UserAction.ACCEPT
        feedback_text = "User accepted the item."
        st.success("✅ Item accepted")
    elif action == "regenerate":
        user_action = UserAction.REQUEST_REFINEMENT
        feedback_text = "User requested to regenerate the item."
        st.info("🔄 Regenerating item...")
    else:
        st.error(f"Unknown action: {action}")
        return

    agent_state.user_action = user_action
    agent_state.user_feedback = UserFeedback(item_id=item_id, feedback=feedback_text)
    st.session_state.agent_state = agent_state

    # Start the workflow to process the user feedback
    _start_workflow_thread(agent_state)


def handle_api_key_validation():
    """
    Handle API key validation by calling the LLM service validate_api_key method.
    Updates session state with validation results.
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
            # Get container and update settings with the new key
            container = get_container()
            settings = container.get_by_name("settings")
            settings.gemini_api_key = user_api_key

            # The LLM service is a singleton. We assume it's designed to handle
            # settings changes, or it's re-initialized on next use internally.
            llm_service: EnhancedLLMService = container.get_by_name("llm_service")

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
        logging.error("Gemini API key configuration error: %s", e)
    except Exception as e:  # noqa: BLE001
        st.session_state.api_key_validation_failed = True
        st.error(f"❌ Validation failed: {e}")
        logging.error("Gemini API key validation exception: %s", e, exc_info=True)
