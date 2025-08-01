"""Consolidated UI management module for the Streamlit application.

This module provides a centralized UIManager class that handles all UI rendering
and user interaction logic, keeping presentation concerns separate from business logic.
Consolidated from src/ui/ui_manager.py and src/frontend/workflow_controller.py.
"""

import asyncio
import threading
import time
import traceback
from typing import Optional, Tuple

import streamlit as st

from src.config.logging_config import get_logger
from src.core.container import get_container
from src.core.state_manager import StateManager
from src.error_handling.boundaries import CATCHABLE_EXCEPTIONS
from src.frontend.ui_components import (
    display_export_tab,
    display_input_form,
    display_review_and_edit_tab,
    display_sidebar,
)
from src.frontend.streamlit_callback import StreamlitCallbackHandler
from src.models.workflow_models import UserFeedback, UserAction
from src.orchestration.state import GlobalState

logger = get_logger(__name__)


class UIManager:
    """Handles all UI rendering and user interaction logic.

    This class encapsulates all Streamlit UI components and rendering logic,
    providing a clean separation between presentation and business logic.
    Consolidated from both ui_manager.py and workflow_controller.py.
    """

    def __init__(self, state_manager: StateManager):
        """Initialize the UIManager with a state manager.

        Args:
            state_manager: The StateManager instance for state access
        """
        self.state = state_manager
        self._configure_page()

        # Workflow controller attributes
        self.facade = None
        self._background_thread = None
        self._background_loop = None
        self._is_running = False

        # Initialize StreamlitCallbackHandler for real-time updates
        self.callback_handler = StreamlitCallbackHandler("workflow_stream_output")

        logger.info("UIManager initialized with StreamlitCallbackHandler")

    def _configure_page(self):
        """Configure the Streamlit page settings."""
        # Page configuration is handled in app.py to avoid conflicts
        pass

    def render_header(self):
        """Render the application header."""
        st.title("🤖 AI CV Generator")
        st.markdown(
            "Transform your CV with AI-powered optimization tailored to specific job descriptions."
        )
        st.divider()

    def render_sidebar(self):
        """Render the sidebar with navigation and settings."""
        return display_sidebar()

    def render_processing_indicator(self):
        """Render a processing indicator for ongoing operations."""
        with st.spinner("Processing your request..."):
            st.info("⏳ Your CV is being generated. This may take a few moments.")

    def render_status_messages(self, messages: list):
        """Render status messages to the user.

        Args:
            messages: List of status messages to display
        """
        for message in messages:
            if message.get("type") == "error":
                st.error(message.get("text", "An error occurred"))
            elif message.get("type") == "warning":
                st.warning(message.get("text", "Warning"))
            elif message.get("type") == "success":
                st.success(message.get("text", "Success"))
            else:
                st.info(message.get("text", "Information"))

    def render_main_tabs(self):
        """Render the main application tabs.

        Returns:
            Tuple of tab objects for input, review, and export
        """
        return st.tabs(["📝 Input", "🔍 Review & Edit", "📄 Export"])

    def _render_input_tab(self, tab):
        """Render the input tab content.

        Args:
            tab: The Streamlit tab object
        """
        with tab:
            return display_input_form()

    def _render_review_tab(self, tab):
        """Render the review and edit tab content.

        Args:
            tab: The Streamlit tab object
        """
        with tab:
            agent_state = self.state.get_agent_state()
            if agent_state and agent_state.get("structured_cv"):
                return display_review_and_edit_tab(agent_state["structured_cv"])
            else:
                st.info("No CV data available for review. Please generate a CV first.")
                return None

    def _render_export_tab(self, tab):
        """Render the export tab content.

        Args:
            tab: The Streamlit tab object
        """
        with tab:
            agent_state = self.state.get_agent_state()
            if agent_state and agent_state.get("structured_cv"):
                return display_export_tab(agent_state["structured_cv"])
            else:
                st.info("No CV data available for export. Please generate a CV first.")
                return None

    def render_debug_info(self):
        """Render debug information if enabled."""
        if st.sidebar.checkbox("Show Debug Info", value=False):
            with st.expander("Debug Information", expanded=False):
                st.json(
                    {
                        "session_state_keys": list(st.session_state.keys()),
                        "agent_state_available": bool(self.state.get_agent_state()),
                        "workflow_session_id": self.state.get_workflow_session_id(),
                    }
                )

    def render_status_driven_ui(self, workflow_status: str, agent_state: GlobalState):
        """Render UI components based on the current workflow status.

        Args:
            workflow_status: Current workflow status
            agent_state: Current agent state
        """
        if workflow_status == "AWAITING_FEEDBACK":
            self._render_awaiting_feedback_ui(agent_state)
        elif workflow_status == "PROCESSING":
            self.render_processing_indicator()
        elif workflow_status == "COMPLETED":
            self._render_completed_ui(agent_state)
        elif workflow_status == "ERROR":
            self._render_error_ui(agent_state)
        else:
            st.info(f"Workflow status: {workflow_status}")

    def _render_awaiting_feedback_ui(self, agent_state: GlobalState):
        """Render UI for when user feedback is needed.

        Args:
            agent_state: Current agent state
        """
        st.subheader("🔍 Review Generated Content")
        st.info("Please review the generated content and provide your feedback.")

        # Display generated content
        if "generated_content" in agent_state:
            st.markdown("### Generated Content:")
            st.markdown(agent_state["generated_content"])

        # Feedback buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Approve", type="primary", use_container_width=True):
                self._handle_approve_action(agent_state)
        with col2:
            if st.button("🔄 Regenerate", use_container_width=True):
                self._handle_regenerate_action(agent_state)

    def _render_completed_ui(self, agent_state: GlobalState):
        """Render UI for completed workflow.

        Args:
            agent_state: Current agent state
        """
        st.success("🎉 CV Generation Completed Successfully!")

        # Display final CV
        if "structured_cv" in agent_state:
            st.markdown("### Your Optimized CV:")
            # Display CV content here

        # Download button
        if "pdf_content" in agent_state:
            st.download_button(
                label="📄 Download PDF",
                data=agent_state["pdf_content"],
                file_name="optimized_cv.pdf",
                mime="application/pdf",
                type="primary",
            )

    def _render_error_ui(self, agent_state: GlobalState):
        """Render UI for error states.

        Args:
            agent_state: Current agent state
        """
        st.error("❌ An error occurred during CV generation")

        # Display error details
        if "error_messages" in agent_state and agent_state["error_messages"]:
            with st.expander("Error Details", expanded=False):
                for error in agent_state["error_messages"]:
                    st.code(error)

        # Restart button
        if st.button("🔄 Start Over", type="primary"):
            self._handle_restart_workflow()

    def _handle_approve_action(self, agent_state: GlobalState):
        """Handle user approval action.

        Args:
            agent_state: Current agent state
        """
        try:
            session_id = self.state.get_workflow_session_id()
            if session_id and self.facade:
                feedback = UserFeedback(
                    action=UserAction.ACCEPT,
                    item_id="generated_content",
                    feedback_text="User approved the content",
                )
                success = self.facade.submit_user_feedback(session_id, feedback)
                if success:
                    st.success("Feedback submitted successfully!")
                else:
                    st.error("Failed to submit feedback")
        except Exception as e:
            logger.error(f"Error handling approve action: {e}")
            st.error("An error occurred while processing your approval")

    def _handle_regenerate_action(self, agent_state: GlobalState):
        """Handle user regenerate action.

        Args:
            agent_state: Current agent state
        """
        try:
            session_id = self.state.get_workflow_session_id()
            if session_id and self.facade:
                feedback = UserFeedback(
                    action=UserAction.REGENERATE,
                    item_id="generated_content",
                    feedback_text="User requested regeneration",
                )
                success = self.facade.submit_user_feedback(session_id, feedback)
                if success:
                    st.success("Regeneration request submitted!")
                else:
                    st.error("Failed to submit regeneration request")
        except Exception as e:
            logger.error(f"Error handling regenerate action: {e}")
            st.error("An error occurred while processing your regeneration request")

    def _handle_restart_workflow(self):
        """Handle workflow restart action."""
        try:
            # Clear session state
            self.state.clear_workflow_state()
            st.success("Workflow restarted successfully!")
        except Exception as e:
            logger.error(f"Error restarting workflow: {e}")
            st.error("An error occurred while restarting the workflow")

    def render_full_ui(self):
        """Render the complete UI based on current state."""
        try:
            # Render header
            self.render_header()

            # Render sidebar
            self.render_sidebar()

            # Check for active workflow
            workflow_session_id = self.state.get_workflow_session_id()
            if workflow_session_id:
                # Get workflow status
                container = get_container()
                manager = container.workflow_manager()
                agent_state = manager.get_workflow_status(workflow_session_id)

                if agent_state:
                    workflow_status = agent_state.get("workflow_status", "UNKNOWN")
                    self.render_status_driven_ui(workflow_status, agent_state)
                else:
                    st.error("Failed to retrieve workflow status")
            else:
                # Render main tabs for initial input
                input_tab, review_tab, export_tab = self.render_main_tabs()

                # Render tab content
                user_inputs = self._render_input_tab(input_tab)
                self._render_review_tab(review_tab)
                self._render_export_tab(export_tab)

                return user_inputs

            # Render debug info
            self.render_debug_info()

        except CATCHABLE_EXCEPTIONS as e:
            logger.error(f"Error rendering UI: {e}")
            st.error("An error occurred while rendering the interface")
            return None

    def get_user_inputs(self) -> Optional[Tuple[str, str]]:
        """Get user inputs from the form.

        Returns:
            Tuple of (cv_text, job_description) if available, None otherwise
        """
        try:
            cv_text = self.state.get_cv_text()
            job_description = self.state.get_job_description()

            if cv_text and job_description:
                return cv_text, job_description
            return None
        except Exception as e:
            logger.error(f"Error getting user inputs: {e}")
            return None

    def show_startup_error(self, error_message: str):
        """Show startup error message.

        Args:
            error_message: The error message to display
        """
        st.error(f"🚨 Application Startup Error: {error_message}")
        st.info("Please check the application configuration and try again.")

    def show_validation_error(self, error_message: str):
        """Show validation error message.

        Args:
            error_message: The validation error message to display
        """
        st.error(f"❌ Validation Error: {error_message}")

    def show_unexpected_error(self, error_message: str):
        """Show unexpected error message.

        Args:
            error_message: The unexpected error message to display
        """
        st.error(f"💥 Unexpected Error: {error_message}")
        st.info(
            "Please try refreshing the page or contact support if the issue persists."
        )

    # Workflow Controller Methods
    def initialize_facade(self, facade):
        """Initialize the CV generation facade for this UI instance.

        Args:
            facade: The CvGenerationFacade instance
        """
        self.facade = facade
        self._start_background_thread()

    def _start_background_thread(self):
        """Start the background thread with a persistent event loop."""
        if self._is_running or not self.facade:
            return

        logger.info("Starting UIManager background thread")
        self._background_thread = threading.Thread(
            target=self._run_background_event_loop,
            daemon=True,
            name="UIManager-AsyncLoop",
        )
        self._background_thread.start()
        self._is_running = True

    def _run_background_event_loop(self):
        """Run the background asyncio event loop."""
        try:
            self._background_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._background_loop)
            logger.info("Background event loop started")
            self._background_loop.run_forever()
        except Exception as e:
            logger.error(f"Background event loop error: {e}")
        finally:
            logger.info("Background event loop stopped")

    def start_generation(self, cv_text: str, job_description: str) -> str:
        """Start the CV generation workflow.

        Args:
            cv_text: The CV text content
            job_description: The job description text

        Returns:
            The session ID of the created workflow
        """
        if not self.facade:
            raise RuntimeError("CV generation facade not initialized")

        try:
            # Create new workflow using facade
            session_id = self.facade.start_cv_generation(
                cv_content=cv_text, job_description=job_description
            )

            logger.info(f"CV generation workflow started for session {session_id}")
            return session_id

        except Exception as e:
            logger.error(f"Failed to start generation: {e}")
            raise

    def start_cv_generation(self, cv_text: str, job_description: str) -> str:
        """Start CV generation workflow (alias for start_generation).

        Args:
            cv_text: The CV text content
            job_description: The job description text

        Returns:
            The session ID of the created workflow
        """
        return self.start_generation(cv_text, job_description)

    async def stream_cv_generation(
        self, cv_text: str, job_description: str, callback_handler=None
    ):
        """Stream CV generation workflow with real-time updates.

        Args:
            cv_text: The CV text content
            job_description: The job description text
            callback_handler: Optional callback handler for streaming updates

        Yields:
            dict: Workflow state updates during execution
        """
        if not self.facade:
            raise RuntimeError("CV generation facade not initialized")

        try:
            # Use provided callback handler or fallback to instance handler
            handler = callback_handler or self.callback_handler

            # Stream workflow execution using facade
            async for state_update in self.facade.stream_cv_generation(
                cv_text=cv_text, jd_text=job_description, callback_handler=handler
            ):
                yield state_update

        except Exception as e:
            logger.error(f"Failed to stream CV generation: {e}")
            raise

    async def _execute_initial_workflow(
        self, session_id: str, initial_state: GlobalState
    ):
        """Execute the initial workflow step.

        Args:
            session_id: The workflow session ID
            initial_state: The initial workflow state
        """
        try:
            # This method is no longer needed with facade pattern
            pass
            logger.info(f"Initial workflow step completed for session {session_id}")
        except Exception as e:
            logger.error(
                f"Initial workflow execution failed for session {session_id}: {e}"
            )

    def submit_user_feedback(
        self, action: str, item_id: str, workflow_session_id: str
    ) -> bool:
        """Submit user feedback to the workflow.

        Args:
            action: The user action (accept/regenerate)
            item_id: The ID of the item being acted upon
            workflow_session_id: The workflow session ID

        Returns:
            True if feedback was submitted successfully, False otherwise
        """
        try:
            # Create user feedback object
            user_action = (
                UserAction.ACCEPT
                if action.lower() == "accept"
                else UserAction.REGENERATE
            )
            feedback = UserFeedback(
                action=user_action,
                item_id=item_id,
                feedback_text=f"User {action} action",
            )

            success = self.facade.submit_user_feedback(
                session_id=workflow_session_id, feedback=feedback
            )

            return success

        except Exception as e:
            logger.error(f"Failed to submit user feedback: {e}")
            return False

    async def _execute_feedback_workflow(self, session_id: str, state: GlobalState):
        """Execute workflow continuation after user feedback.

        Args:
            session_id: The workflow session ID
            state: The current workflow state
        """
        try:
            # This method is no longer needed with facade pattern
            pass
            logger.info(f"Feedback workflow step completed for session {session_id}")
        except Exception as e:
            logger.error(
                f"Feedback workflow execution failed for session {session_id}: {e}"
            )

    def is_ready(self) -> bool:
        """Check if the UI manager is ready for operations.

        Returns:
            True if ready, False otherwise
        """
        return self._is_running and self.facade is not None

    def shutdown(self):
        """Shutdown the UI manager and clean up resources."""
        if self._background_loop and self._background_loop.is_running():
            # Stop the event loop
            self._background_loop.call_soon_threadsafe(self._background_loop.stop)

        if self._background_thread and self._background_thread.is_alive():
            # Wait for thread to finish
            self._background_thread.join(timeout=5.0)

        self._is_running = False
        logger.info("UIManager shutdown completed")

    def __del__(self):
        """Cleanup on garbage collection."""
        try:
            self.shutdown()
        except Exception:
            pass  # Ignore errors during cleanup
