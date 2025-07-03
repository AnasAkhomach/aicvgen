"""UI management module for the Streamlit application.

This module provides a centralized UIManager class that handles all UI rendering
and user interaction logic, keeping presentation concerns separate from business logic.
"""
from typing import Optional, Tuple
import traceback
import time
import streamlit as st


from ..core.state_manager import StateManager
from ..config.logging_config import get_logger
from ..error_handling.boundaries import CATCHABLE_EXCEPTIONS
from ..frontend.ui_components import (
    display_sidebar,
    display_input_form,
    display_review_and_edit_tab,
    display_export_tab,
)

logger = get_logger(__name__)


class UIManager:
    """Handles all UI rendering and user interaction logic.

    This class encapsulates all Streamlit UI components and rendering logic,
    providing a clean separation between presentation and business logic.
    """

    def __init__(self, state_manager: StateManager):
        """Initialize the UIManager with a state manager.

        Args:
            state_manager: The StateManager instance for state access
        """
        self.state = state_manager
        self._configure_page()

    def _configure_page(self) -> None:
        """Configure the Streamlit page settings."""
        st.set_page_config(
            page_title="🤖 AI CV Generator",
            page_icon="🤖",
            layout="wide",
            initial_sidebar_state="expanded",
        )

    def render_header(self) -> None:
        """Render the main application header."""
        st.title("🤖 AI CV Generator")
        st.markdown(
            "Transform your CV to match any job description using advanced AI. "
            "Get personalized, ATS-friendly CVs that highlight your most relevant skills and experience."
        )

    def render_sidebar(self) -> None:
        """Render the sidebar using existing UI components."""
        try:
            display_sidebar()
        except CATCHABLE_EXCEPTIONS as e:
            logger.error("Error rendering sidebar: %s", e)
            st.sidebar.error("Error loading sidebar components")

    def render_processing_indicator(self) -> None:
        """Render processing indicators and handle processing state."""
        if self.state.is_processing:
            with st.spinner("Processing your CV... Please wait."):
                # Allow UI to feel responsive while waiting for background processing
                while self.state.is_processing:
                    time.sleep(0.1)

    def render_status_messages(self) -> None:
        """Render status messages (success, error, etc.)."""
        # Show success message if just finished
        if self.state.just_finished:
            st.success("CV Generation Complete!")
            self.state.just_finished = False

        # Show workflow errors
        if self.state.workflow_error:
            st.error(
                f"An error occurred during CV generation: {self.state.workflow_error}"
            )
            self.state.workflow_error = None

        # Show agent state errors
        if self.state.agent_state and self.state.agent_state.error_messages:
            for error in self.state.agent_state.error_messages:
                st.error(error)

    def render_main_tabs(self) -> None:
        """Render the main application tabs."""
        tab1, tab2, tab3 = st.tabs(
            ["📝 Input & Generate", "✏️ Review & Edit", "📄 Export"]
        )

        with tab1:
            self._render_input_tab()

        with tab2:
            self._render_review_tab()

        with tab3:
            self._render_export_tab()

    def _render_input_tab(self) -> None:
        """Render the input and generation tab."""
        try:
            display_input_form()
        except CATCHABLE_EXCEPTIONS as e:
            logger.error("Error rendering input form: %s", e)
            st.error("Error loading input form")

    def _render_review_tab(self) -> None:
        """Render the review and edit tab."""
        if self.state.agent_state:
            try:
                display_review_and_edit_tab(self.state.agent_state)
            except CATCHABLE_EXCEPTIONS as e:
                logger.error("Error rendering review tab: %s", e)
                st.error("Error loading review interface")
        else:
            st.info("Start CV generation in the first tab to see results here.")

    def _render_export_tab(self) -> None:
        """Render the export tab."""
        if self.state.agent_state:
            try:
                display_export_tab(self.state.agent_state)
            except CATCHABLE_EXCEPTIONS as e:
                logger.error("Error rendering export tab: %s", e)
                st.error("Error loading export interface")
        else:
            st.info("Complete CV generation to export your results.")

    def render_debug_info(self, show_debug: bool = False) -> None:
        """Render debug information if enabled.

        Args:
            show_debug: Whether to show debug information
        """
        if show_debug:
            with st.expander("🔍 Debug Information", expanded=False):
                st.json(self.state.get_state_summary())

    def render_full_ui(self, show_debug: bool = False) -> None:
        """Render the complete UI.

        Args:
            show_debug: Whether to show debug information
        """
        # Render sidebar
        self.render_sidebar()

        # Render main content
        self.render_header()

        # Handle processing state
        self.render_processing_indicator()

        # Show status messages
        self.render_status_messages()

        # Render main tabs
        self.render_main_tabs()

        # Optional debug info
        if show_debug:
            self.render_debug_info(True)

    def get_user_inputs(self) -> Tuple[Optional[str], Optional[str]]:
        """Get current user inputs from state.

        Returns:
            Tuple of (cv_text, job_description_text)
        """
        return self.state.cv_text, self.state.job_description_text

    def show_startup_error(self, errors: list, services_info: dict) -> None:
        """Show startup error information.

        Args:
            errors: List of error messages
            services_info: Dictionary of service information
        """
        st.error("**Application Startup Failed:**")
        for error in errors:
            st.error(f"• {error}")
        st.warning("Please check your configuration and restart the application.")

        # Show startup details in expander
        with st.expander("🔍 Startup Details", expanded=False):
            st.json(
                {
                    "services": {
                        name: {
                            "status": (
                                "✅ Success" if service.initialized else "❌ Failed"
                            ),
                            "time": f"{service.initialization_time:.3f}s",
                            "error": service.error,
                        }
                        for name, service in services_info.items()
                    },
                }
            )

    def show_validation_error(self, validation_errors: list) -> None:
        """Show validation error information.

        Args:
            validation_errors: List of validation error messages
        """
        st.error("**Critical Service Validation Failed:**")
        for error in validation_errors:
            st.error(f"• {error}")

    def show_unexpected_error(self, error: Exception) -> None:
        """Show unexpected error information with debug details.

        Args:
            error: The exception that occurred
        """
        st.error(f"An unexpected error occurred: {error}")

        # Show error details in expander for debugging
        with st.expander("🔍 Error Details", expanded=False):
            st.code(traceback.format_exc(), language="text")
