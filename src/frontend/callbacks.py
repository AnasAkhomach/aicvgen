# In src/frontend/callbacks.py
import asyncio
import threading
import uuid
from typing import Optional

import streamlit as st

from src.config.logging_config import get_logger
from src.core.application_startup import get_application_startup
from src.core.container import get_container
from src.error_handling.exceptions import (AgentExecutionError, ConfigurationError)
from src.integration.enhanced_cv_system import get_enhanced_cv_integration
from src.models.data_models import UserAction, WorkflowType
from src.orchestration.state import AgentState, UserFeedback
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
            session_id = str(uuid.uuid4())
            st.session_state.session_id = session_id

        # Set the session ID in the container for dynamic agent creation
        Container.set_session_id(session_id)

        logger.info(
            "Set container session ID for dynamic agent creation",
            extra={"session_id": session_id}
        )

        # Application should already be initialized by the time a callback is called.
        startup_manager = get_application_startup()
        if not startup_manager.is_initialized:
            # This should ideally never happen if the app is started via main.py
            st.error("FATAL: Integration started before application was initialized.")
            raise ConfigurationError("Application not initialized.")

        st.session_state.cv_integration = get_enhanced_cv_integration()
    return st.session_state.cv_integration


def _execute_workflow_in_thread(state_to_run: AgentState, trace_id: str):
    """
    The target function for the background thread.
    Executes the CV workflow using the EnhancedCVIntegration layer.
    """
    thread_logger = logger
    try:  # Each thread needs its own event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        thread_logger.info(
            "Starting CV workflow in background thread",
            extra={
                "trace_id": trace_id,
                "session_id": st.session_state.get("session_id"),
            },
        )

        # Get the enhanced CV integration instance
        cv_integration = get_enhanced_cv_integration_instance()
        session_id = st.session_state.get("session_id")

        # Execute the workflow using the integration layer
        # Determine workflow type based on the state content
        workflow_type = WorkflowType.COMPREHENSIVE_CV  # Use comprehensive CV workflow

        result = loop.run_until_complete(
            cv_integration.execute_workflow(
                workflow_type=workflow_type,
                input_data=state_to_run,
                session_id=session_id,
            )
        )

        # Update the session state with the final results
        # The result contains the final agent state and other metadata
        if "final_state" in result:
            st.session_state.agent_state = result["final_state"]
        else:
            # Fallback: use the result as the final state if it's an AgentState
            st.session_state.agent_state = result

        thread_logger.info(
            "CV workflow completed successfully",
            extra={"trace_id": trace_id},
        )

    except (AgentExecutionError, ConfigurationError, RuntimeError) as e:
        thread_logger.error(
            "CV workflow failed in background thread: %s",
            e,
            extra={"trace_id": trace_id, "error_type": type(e).__name__},
            exc_info=True,
        )
        st.session_state.workflow_error = e
    finally:
        st.session_state.is_processing = False
        st.session_state.just_finished = True
        loop.close()


def _execute_workflow_manager_in_thread(state_to_run: AgentState, trace_id: str, workflow_session_id: str, workflow_manager):
    """
    The target function for the background thread.
    Executes the CV workflow using the WorkflowManager.
    """
    thread_logger = logger
    try:  # Each thread needs its own event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        thread_logger.info(
            "Starting CV workflow using WorkflowManager in background thread",
            extra={
                "trace_id": trace_id,
                "workflow_session_id": workflow_session_id,
            },
        )

        # Execute the workflow step using WorkflowManager
        result_state = loop.run_until_complete(
            workflow_manager.trigger_workflow_step(
                session_id=workflow_session_id,
                agent_state=state_to_run
            )
        )

        # Update the session state with the final results
        st.session_state.agent_state = result_state

        thread_logger.info(
            "CV workflow completed successfully using WorkflowManager",
            extra={"trace_id": trace_id, "workflow_session_id": workflow_session_id},
        )

    except (AgentExecutionError, ConfigurationError, RuntimeError) as e:
        thread_logger.error(
            "CV workflow failed in background thread: %s",
            e,
            extra={"trace_id": trace_id, "workflow_session_id": workflow_session_id, "error_type": type(e).__name__},
            exc_info=True,
        )
        st.session_state.workflow_error = e
    finally:
        st.session_state.is_processing = False
        st.session_state.just_finished = True
        loop.close()


def _start_workflow_thread(state_to_run: AgentState):
    """Helper to configure and start the background workflow thread."""
    start_logger = logger
    trace_id = str(uuid.uuid4())
    state_to_run.trace_id = trace_id  # Set flags to indicate processing has started
    st.session_state.is_processing = True
    st.session_state.workflow_error = None  # Clear previous errors
    st.session_state.just_finished = False

    start_logger.info(
        "Starting workflow thread",
        extra={"trace_id": trace_id, "session_id": st.session_state.get("session_id")},
    )

    thread = threading.Thread(
        target=_execute_workflow_in_thread, args=(state_to_run, trace_id), daemon=True
    )
    thread.start()


def _start_workflow_manager_thread(state_to_run: AgentState, workflow_session_id: str, workflow_manager):
    """Helper to configure and start the background workflow thread using WorkflowManager."""
    start_logger = logger
    trace_id = str(uuid.uuid4())
    state_to_run.trace_id = trace_id  # Set flags to indicate processing has started
    st.session_state.is_processing = True
    st.session_state.workflow_error = None  # Clear previous errors
    st.session_state.just_finished = False

    start_logger.info(
        "Starting workflow thread using WorkflowManager",
        extra={"trace_id": trace_id, "workflow_session_id": workflow_session_id},
    )

    thread = threading.Thread(
        target=_execute_workflow_manager_in_thread, 
        args=(state_to_run, trace_id, workflow_session_id, workflow_manager), 
        daemon=True
    )
    thread.start()


def start_cv_generation():
    """Initializes state from UI inputs and starts the CV generation workflow using WorkflowManager."""
    job_desc_raw = st.session_state.get("job_description_input", "")
    cv_text = st.session_state.get("cv_text_input", "")
    start_from_scratch = st.session_state.get("start_from_scratch_input", False)

    # Get WorkflowManager from container
    container = get_container()
    workflow_manager = container.workflow_manager()
    
    # Create new workflow and get session_id
    session_id = workflow_manager.create_new_workflow(
        cv_text=cv_text,
        jd_text=job_desc_raw
    )
    
    # Store session_id in Streamlit session state
    st.session_state.workflow_session_id = session_id
    
    # Create initial agent state
    initial_state = create_initial_agent_state(
        job_description_raw=job_desc_raw,
        cv_text=cv_text,
        start_from_scratch=start_from_scratch,
    )
    st.session_state.agent_state = initial_state
    
    # Start workflow using WorkflowManager in background thread
    _start_workflow_manager_thread(initial_state, session_id, workflow_manager)


def handle_user_action(action: str, item_id: str):
    """
    Handle user actions like 'accept' or 'regenerate' with async execution.
    
    This function implements the feedback callback requirements:
    - Calls workflow_manager.send_feedback() with appropriate action
    - Calls asyncio.run(workflow_manager.trigger_workflow_step()) to resume workflow
    """
    import asyncio
    
    agent_state: Optional[AgentState] = st.session_state.get("agent_state")
    if not agent_state:
        st.error("No agent state found")
        return

    # Get workflow session ID and WorkflowManager
    workflow_session_id = st.session_state.get("workflow_session_id")
    if not workflow_session_id:
        st.error("No workflow session found")
        return
        
    container = get_container()
    workflow_manager = container.workflow_manager()

    # Map UI actions to feedback actions as required by the task
    if action == "accept":
        feedback_action = "approve"  # Task requires "approve" for Accept button
        user_action = UserAction.APPROVE
        feedback_text = "User approved the item."
        st.success("✅ Item approved")
    elif action == "regenerate":
        feedback_action = "regenerate"  # Task requires "regenerate" for Regenerate button
        user_action = UserAction.REGENERATE
        feedback_text = "User requested to regenerate the item."
        st.info("🔄 Regenerating item...")
    else:
        st.error(f"Unknown action: {action}")
        return

    try:
        # Create user feedback with proper validation
        user_feedback = UserFeedback(
            action=user_action,
            item_id=item_id, 
            feedback_text=feedback_text
        )
        
        # Step 1: Send feedback to workflow manager
        feedback_sent = workflow_manager.send_feedback(workflow_session_id, user_feedback)
        if not feedback_sent:
            st.error("Failed to send feedback to workflow")
            return
            
        # Step 2: Get updated agent state after feedback
        updated_agent_state = workflow_manager.get_workflow_status(workflow_session_id)
        if not updated_agent_state:
            st.error("Failed to get updated workflow state")
            return
            
        # Step 3: Trigger workflow step with async execution as required by task
        try:
            result_state = asyncio.run(
                workflow_manager.trigger_workflow_step(workflow_session_id, updated_agent_state)
            )
            
            # Update session state with result
            st.session_state.agent_state = result_state
            st.success(f"Workflow resumed successfully after {feedback_action}")
            
        except Exception as async_error:
            logger.error(f"Error in async workflow execution: {async_error}")
            st.error(f"Failed to resume workflow: {async_error}")
            
    except Exception as e:
        logger.error(f"Error handling user action {action}: {e}")
        st.error(f"Error processing {action} action: {e}")


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
