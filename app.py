#!/usr/bin/env python3
"""
Main launcher for the AI CV Generator Streamlit application.
This file implements the Interactive UI Loop driven by workflow status.
"""
import time
import asyncio
import threading
import streamlit as st
from src.core.application_startup import get_container
from src.models.workflow_models import UserFeedback, UserAction
from src.frontend.ui_components import display_sidebar, display_input_form
from src.core.state_manager import StateManager
from src.error_handling.boundaries import safe_streamlit_component
from src.error_handling.exceptions import (
    CATCHABLE_EXCEPTIONS,
    WorkflowError,
    StateManagerError,
    ConfigurationError
)
from src.config.logging_config import get_logger, setup_logging

# Initialize logging first
setup_logging()

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="AI CV Generator",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

logger = get_logger(__name__)


def run_async_workflow_in_thread(workflow_func, *args, **kwargs):
    """Run an async workflow function in a separate thread with its own event loop.
    
    This is necessary to avoid 'Event loop is closed' errors in Streamlit,
    which already runs in an event loop context.
    
    Args:
        workflow_func: The async function to run
        *args: Positional arguments for the workflow function
        **kwargs: Keyword arguments for the workflow function
    """
    def run_in_thread():
        """Run the workflow in a separate thread with its own event loop."""
        try:
            # Create a new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(workflow_func(*args, **kwargs))
            finally:
                loop.close()
        except CATCHABLE_EXCEPTIONS as e:
            logger.error("Error in workflow thread: %s", e)
    
    # Start workflow in background thread
    workflow_thread = threading.Thread(target=run_in_thread, daemon=True)
    workflow_thread.start()
    return workflow_thread


def on_start_generation():
    """Create workflow and trigger its first step."""
    try:
        # Get inputs from UI
        cv_text = st.session_state.get("cv_text_input", "")
        jd_text = st.session_state.get("job_description_input", "")

        if not cv_text or not jd_text:
            st.error("Please provide both CV text and job description.")
            return

        # Get workflow manager from container
        container = get_container()
        manager = container.workflow_manager()

        # Create new workflow and get session_id
        session_id = manager.create_new_workflow(cv_text, jd_text)
        st.session_state.session_id = session_id

        # Get initial state and trigger first workflow step
        initial_state = manager.get_workflow_status(session_id)
        if initial_state is None:
            logger.error("Failed to get initial state for session: %s", session_id)
            st.error("Failed to initialize workflow. Please try again.")
            return
            
        # Use thread-based approach to avoid "Event loop is closed" error in Streamlit
        run_async_workflow_in_thread(manager.trigger_workflow_step, session_id, initial_state)
        
        st.info("⏳ Starting workflow processing...")
        time.sleep(1)  # Brief pause to let thread start
        
        logger.info("Started workflow with session_id: %s", session_id)
        st.rerun()

    except (WorkflowError, StateManagerError, ConfigurationError) as e:
        logger.error("Workflow error starting generation: %s", e)
        st.error(f"Failed to start generation: {e}")
    except CATCHABLE_EXCEPTIONS as e:
        logger.error("Error starting generation: %s", e)
        st.error(f"Failed to start generation: {e}")


def on_approve():
    """Send approve feedback and resume workflow."""
    try:
        session_id = st.session_state.get("session_id")
        if not session_id:
            st.error("No active workflow session found.")
            return

        container = get_container()
        manager = container.workflow_manager()

        # Create user feedback for approve action
        user_feedback = UserFeedback(
            action=UserAction.APPROVE,
            item_id="current_section",
            feedback_text="User approved the content."
        )

        # Send feedback and get updated state
        manager.send_feedback(session_id, user_feedback)
        updated_state = manager.get_workflow_status(session_id)

        # Resume workflow - use thread-based approach for Streamlit
        run_async_workflow_in_thread(manager.trigger_workflow_step, session_id, updated_state)
        
        st.info("⏳ Processing approval...")
        time.sleep(1)  # Brief pause to let thread start
        st.success("✅ Content approved and workflow resumed.")
        logger.info("Approved content for session: %s", session_id)
        st.rerun()

    except (WorkflowError, StateManagerError) as e:
        logger.error("Workflow error in approve action: %s", e)
        st.error(f"Failed to approve content: {e}")
    except CATCHABLE_EXCEPTIONS as e:
        logger.error("Error in approve action: %s", e)
        st.error(f"Failed to approve content: {e}")


def on_regenerate():
    """Send regenerate feedback and resume workflow."""
    try:
        session_id = st.session_state.get("session_id")
        if not session_id:
            st.error("No active workflow session found.")
            return

        container = get_container()
        manager = container.workflow_manager()

        # Create user feedback for regenerate action
        user_feedback = UserFeedback(
            action=UserAction.REGENERATE,
            item_id="current_section",
            feedback_text="User requested to regenerate the content."
        )

        # Send feedback and get updated state
        manager.send_feedback(session_id, user_feedback)
        updated_state = manager.get_workflow_status(session_id)

        # Resume workflow - use thread-based approach for Streamlit
        run_async_workflow_in_thread(manager.trigger_workflow_step, session_id, updated_state)
        
        st.info("⏳ Processing regeneration...")
        time.sleep(1)  # Brief pause to let thread start
        st.info("🔄 Content regeneration requested and workflow resumed.")
        logger.info("Regenerate requested for session: %s", session_id)
        st.rerun()

    except (WorkflowError, StateManagerError) as e:
        logger.error("Workflow error in regenerate action: %s", e)
        st.error(f"Failed to regenerate content: {e}")
    except CATCHABLE_EXCEPTIONS as e:
        logger.error("Error in regenerate action: %s", e)
        st.error(f"Failed to regenerate content: {e}")


def render_awaiting_feedback_ui(state):
    """Render the interactive UI when workflow is awaiting feedback."""
    st.subheader("Please Review This Section")

    # Display content from state.ui_display_data
    ui_data = state.ui_display_data or {}
    section_name = ui_data.get("section", "Current Section")
    content_preview = ui_data.get("content_preview")
    
    st.write(f"**Section:** {section_name.replace('_', ' ').title()}")
    
    if content_preview and isinstance(content_preview, dict):
        # Display section preview with items
        section_display_name = content_preview.get("section_name", section_name)
        items_count = content_preview.get("items_count", 0)
        items = content_preview.get("items", [])
        
        st.write(f"**Generated {items_count} items for {section_display_name}:**")
        
        # Display each item
        for i, item in enumerate(items, 1):
            with st.expander(f"Item {i} (ID: {item.get('id', 'unknown')})", expanded=True):
                st.write(item.get('content', 'No content available'))
    else:
        # Fallback display
        content_to_review = ui_data.get("content", "Nothing to display.")
        st.text_area("Generated Content", value=content_to_review, height=300, disabled=True)

    # Feedback buttons
    col1, col2 = st.columns(2)
    with col1:
        st.button("Approve and Continue", on_click=on_approve, type="primary")
    with col2:
        st.button("Regenerate Section", on_click=on_regenerate)


def render_completed_ui(state):
    """Render UI when workflow is completed."""
    st.success("🎉 CV Generation Complete!")

    # Check if final output is available
    final_output_path = getattr(state, 'final_output_path', None)
    if final_output_path:
        try:
            with open(final_output_path, 'rb') as file:
                st.download_button(
                    label="📄 Download Generated CV",
                    data=file.read(),
                    file_name="generated_cv.pdf",
                    mime="application/pdf"
                )
        except (IOError, OSError, FileNotFoundError) as e:
            logger.error("Error reading final output file: %s", e)
            st.error("Generated CV file not found or cannot be read.")
    else:
        st.info("CV generation completed but download file is not available.")


def render_error_ui(state):
    """Render UI when workflow has an error."""
    error_message = "An unknown error occurred."
    if hasattr(state, 'error_messages') and state.error_messages:
        error_message = state.error_messages[-1]

    st.error(f"❌ An error occurred: {error_message}")

    # Option to restart
    if st.button("🔄 Start New Generation"):
        # Clear session state to restart
        if "session_id" in st.session_state:
            del st.session_state.session_id
        st.rerun()


@safe_streamlit_component(component_name="main_app")
def main():
    """Main application entry point with workflow-driven UI."""
    # Initialize application
    StateManager()

    # Check for UI refresh signal from background thread
    if st.session_state.get("needs_rerun", False):
        st.session_state.needs_rerun = False
        st.rerun()

    # Display header
    st.title("🤖 AI CV Generator")
    st.markdown("Generate tailored CVs using AI-powered analysis")

    # Display sidebar
    display_sidebar()

    # Get workflow manager
    container = get_container()
    manager = container.workflow_manager()

    # Main UI logic based on workflow status
    session_id = st.session_state.get("session_id")

    if not session_id:
        # No active workflow - render initial input form
        st.header("📝 Input & Generate")
        display_input_form()

        # Start Generation button
        if st.button("🚀 Start Generation", on_click=on_start_generation, type="primary"):
            pass  # on_click handles the logic

    else:
        # Active workflow - get status and render appropriate UI
        try:
            state = manager.get_workflow_status(session_id)

            if not state:
                st.error("Workflow session not found. Please start a new generation.")
                if st.button("🔄 Start New Generation"):
                    del st.session_state.session_id
                    st.rerun()
                return

            # Render UI based on workflow status
            if state.workflow_status == "AWAITING_FEEDBACK":
                render_awaiting_feedback_ui(state)

            elif state.workflow_status == "PROCESSING":
                st.info("⏳ Processing your request...")
                with st.spinner("Generating CV content..."):
                    # Auto-refresh every 2 seconds to check status
                    time.sleep(2)
                    st.rerun()

            elif state.workflow_status == "COMPLETED":
                render_completed_ui(state)

            elif state.workflow_status == "ERROR":
                render_error_ui(state)

            else:
                st.warning(f"Unknown workflow status: {state.workflow_status}")

        except (WorkflowError, StateManagerError) as e:
            logger.error("Workflow error getting status: %s", e)
            st.error(f"Failed to get workflow status: {e}")
            if st.button("🔄 Start New Generation"):
                if "session_id" in st.session_state:
                    del st.session_state.session_id
                st.rerun()
        except CATCHABLE_EXCEPTIONS as e:
            logger.error("Error getting workflow status: %s", e)
            st.error(f"Failed to get workflow status: {e}")
            if st.button("🔄 Start New Generation"):
                if "session_id" in st.session_state:
                    del st.session_state.session_id
                st.rerun()


if __name__ == "__main__":
    main()
