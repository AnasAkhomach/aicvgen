# In src/frontend/callbacks.py
import asyncio
import concurrent.futures
import uuid
import streamlit as st

from src.config.logging_config import get_logger
from src.core.application_startup import get_application_startup
from src.core.container import get_container
from src.services.session_manager import SessionManager
from src.error_handling.exceptions import ConfigurationError, AgentExecutionError
from src.frontend.workflow_controller import WorkflowController
from src.integration.enhanced_cv_system import get_enhanced_cv_integration
from src.services.llm_service import EnhancedLLMService
from src.utils.state_utils import create_initial_agent_state
from src.core.container import Container

# Initialize logger
logger = get_logger(__name__)


def get_enhanced_cv_integration_instance():
    """Get or create the enhanced CV integration instance for the current session."""
    if "cv_integration" not in st.session_state:
        session_id = st.session_state.get("session_id")
        if not session_id:
            container = get_container()
            session_manager = container.session_manager()
            session_id = session_manager.create_session()
            st.session_state.session_id = session_id

        logger.info(
            "Created session ID for CV integration",
            extra={"session_id": session_id},
        )

        # Application should already be initialized by the time a callback is called.
        startup_manager = get_application_startup()
        if not startup_manager.is_initialized:
            # This should ideally never happen if the app is started via main.py
            st.error("FATAL: Integration started before application was initialized.")
            raise ConfigurationError("Application not initialized.")

        st.session_state.cv_integration = get_enhanced_cv_integration()
    return st.session_state.cv_integration


def _get_or_create_workflow_controller() -> WorkflowController:
    """
    Get or create the WorkflowController instance from session state.

    Returns:
        WorkflowController: The singleton WorkflowController instance
    """
    if "workflow_controller" not in st.session_state:
        # Get the workflow manager from the container
        container = get_container()
        workflow_manager = container.workflow_manager()

        # Create and store the WorkflowController
        st.session_state.workflow_controller = WorkflowController(workflow_manager)
        logger.info("Created new WorkflowController instance")

    return st.session_state.workflow_controller

def start_cv_generation():
    """
    Initializes state from UI inputs and starts the CV generation workflow
    using WorkflowController.
    """
    job_desc_raw = st.session_state.get("job_description_input", "")
    cv_text = st.session_state.get("cv_text_input", "")
    start_from_scratch = st.session_state.get("start_from_scratch_input", False)

    # Get WorkflowController
    workflow_controller = _get_or_create_workflow_controller()

    # Create new workflow and get session_id
    container = get_container()
    workflow_manager = container.workflow_manager()
    session_id = workflow_manager.create_new_workflow(
        cv_text=cv_text, jd_text=job_desc_raw
    )

    # Store session_id in session state
    st.session_state.workflow_session_id = session_id

    # Create initial agent state
    initial_state = create_initial_agent_state(
        job_description_raw=job_desc_raw,
        cv_text=cv_text,
        start_from_scratch=start_from_scratch
    )

    # Store initial state in session state
    st.session_state.agent_state = initial_state

    # Start workflow execution using WorkflowController
    workflow_controller.start_generation(initial_state, session_id)


def handle_user_action(action: str, item_id: str):
    """
    Handle user actions like 'accept' or 'regenerate' using WorkflowController.
    """
    agent_state = st.session_state.get("agent_state")
    if not agent_state:
        st.error("No agent state found")
        return

    # Get workflow session ID
    workflow_session_id = st.session_state.get("workflow_session_id")
    if not workflow_session_id:
        st.error("No workflow session found")
        return

    try:
        # Get WorkflowController
        workflow_controller = _get_or_create_workflow_controller()

        # Submit user feedback using WorkflowController
        success = workflow_controller.submit_user_feedback(
            action=action, item_id=item_id, workflow_session_id=workflow_session_id
        )

        if success:
            st.success(f"Action '{action}' processed successfully")
        else:
            st.error(f"Failed to process action '{action}'")

    except (ConfigurationError, AgentExecutionError, RuntimeError) as e:
        logger.error(f"Error handling user action: {e}", exc_info=True)
        st.error(f"Failed to handle {action} action: {e}")
    except (ValueError, OSError, IOError, TimeoutError,
            concurrent.futures.CancelledError) as e:
        logger.error(
            f"Error handling user action: {e}",
            exc_info=True
        )
        st.error(f"An error occurred while handling {action} action")
    except Exception as e:  # pylint: disable=broad-exception-caught
        # Minimal catch-all for truly unexpected errors to prevent UI crashes
        logger.error(
            f"Unexpected error handling user action: {e}",
            exc_info=True
        )
        st.error(f"An unexpected error occurred while handling {action} action")


def handle_api_key_validation():
    """
    Handle API key validation by calling the LLM service validate_api_key method.
    Updates session state with validation results.
    """
    logger.info("Attempting to validate API key.")
    user_api_key = st.session_state.get("user_gemini_api_key", "")

    if not user_api_key:
        st.error("Please enter an API key first")
        logger.warning("API key validation called without an API key.")
        return

    # Reset validation states
    st.session_state.api_key_validated = False
    st.session_state.api_key_validation_failed = False

    try:
        # Show validation in progress
        with st.spinner("Validating API key..."):
            logger.debug("Getting container and services for validation.")
            # Get container and update settings with the new key
            container = get_container()
            settings = container.config()
            settings.gemini_api_key = user_api_key

            # The LLM service is a singleton. We assume it's designed to handle
            # settings changes, or it's re-initialized on next use internally.
            llm_service: EnhancedLLMService = container.llm_service()

            logger.debug("Running async validation.")
            # Run validation asynchronously
            # Using a new event loop can be problematic in some environments,
            # but it's a common pattern for running async code from sync Streamlit callbacks.
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                is_valid = loop.run_until_complete(llm_service.validate_api_key())
            finally:
                loop.close()

            if is_valid:
                st.session_state.api_key_validated = True
                st.success("✅ API key is valid and ready to use!")
                logger.info("API key validation successful.")
            else:
                st.session_state.api_key_validation_failed = True
                st.error("❌ API key validation failed. Please check your key.")
                logger.error(
                    "Gemini API key validation failed: No exception, returned False."
                )

    except ConfigurationError as e:
        st.session_state.api_key_validation_failed = True
        st.error(f"❌ Configuration error: {e}")
        logger.error("Gemini API key configuration error", error=str(e), exc_info=True)
    except (AgentExecutionError, RuntimeError) as e:
        st.session_state.api_key_validation_failed = True
        st.error(f"❌ Validation failed: {e}")
        logger.error("Gemini API key validation exception", error=str(e), exc_info=True)
