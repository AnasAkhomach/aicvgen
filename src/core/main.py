import sys
import os
from pathlib import Path
import traceback
import logging

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


import streamlit as st
import time
import atexit

# Project imports
from ..core.application_startup import get_startup_manager
from ..orchestration.state import AgentState
from ..error_handling.exceptions import ConfigurationError, ServiceInitializationError
from ..error_handling.boundaries import CATCHABLE_EXCEPTIONS
from ..config.logging_config import get_logger, setup_logging
from ..frontend.ui_components import (
    display_sidebar,
    display_input_form,
    display_review_and_edit_tab,
    display_export_tab,
)
from ..config.settings import get_config

# Get logger (will be initialized by startup service)
logger = get_logger(__name__)


# --- Global Exception Hook ---
def handle_exception(exc_type, exc_value, exc_traceback):
    """Log any uncaught exceptions to the error log."""
    # Ensure logging is configured before trying to log
    # This is a fallback in case the exception occurs before normal startup
    try:
        # Check if a root logger has handlers, if not, configure it.
        if not logging.getLogger().hasHandlers():
            setup_logging()
    except CATCHABLE_EXCEPTIONS as setup_exc:
        # If logging setup fails, print to stderr as a last resort
        print(f"FATAL: Logging setup failed: {setup_exc}", file=sys.stderr)
        print(f"Original unhandled exception: {exc_value}", file=sys.stderr)
        traceback.print_exception(exc_type, exc_value, exc_traceback, file=sys.stderr)

    logger.critical(
        "Unhandled exception caught by global handler",
        exc_info=(exc_type, exc_value, exc_traceback),
    )


# It's important to set the hook as early as possible
sys.excepthook = handle_exception
# --- End Global Exception Hook ---


# Constants are now managed by AppConfig, no need for them here.


def main():
    """
    Main Streamlit application controller.
    Orchestrates the UI rendering and backend workflow invocations.
    """
    try:
        # 1. Initialize Application Services using the singleton manager
        startup_service = get_startup_manager()

        # Register the shutdown hook once.
        if not startup_service._shutdown_hook_registered:
            atexit.register(startup_service.shutdown_application)
            startup_service._shutdown_hook_registered = True

        # Initialize only if it hasn't been done already
        if not startup_service.is_initialized:
            # Before initializing, set up a basic logging config
            # This ensures that errors during startup are logged.
            setup_logging()

            user_api_key = st.session_state.get("user_gemini_api_key", "")
            startup_result = startup_service.initialize_application(
                user_api_key=user_api_key
            )

            if not startup_result.success:
                st.error("**Application Startup Failed:**")
                for error in startup_result.errors:
                    st.error(f"• {error}")
                st.warning(
                    "Please check your configuration and restart the application."
                )

                # Show startup details in expander
                with st.expander("🔍 Startup Details", expanded=False):
                    st.json(
                        {
                            "total_time": f"{startup_result.total_time:.2f}s",
                            "services": {
                                name: {
                                    "status": (
                                        "✅ Success"
                                        if service.initialized
                                        else "❌ Failed"
                                    ),
                                    "time": f"{service.initialization_time:.3f}s",
                                    "error": service.error,
                                }
                                for name, service in startup_result.services.items()
                            },
                        }
                    )
                st.stop()

            # Validate critical services
            validation_errors = startup_service.validate_application()
            if validation_errors:
                st.error("**Critical Service Validation Failed:**")
                for error in validation_errors:
                    st.error(f"• {error}")
                st.stop()

            logger.info(
                "Application started successfully in %.2fs", startup_result.total_time
            )
        # Session state initialization is now handled via create_initial_agent_state in utils.state_utils
        # Remove import of deleted state_helpers
        # from ..frontend.state_helpers import initialize_session_state
        # initialize_session_state()

        # 3. Display Static UI Components
        display_sidebar()
        st.title("🤖 AI CV Generator")
        st.markdown(
            "Transform your CV to match any job description using advanced AI. "
            "Get personalized, ATS-friendly CVs that highlight your most relevant skills and experience."
        )  # 4. Handle UI based on processing state
        if st.session_state.get("is_processing"):
            with st.spinner("Processing your CV... Please wait."):
                # The UI is blocked here by the spinner, but the work is in a thread.
                # We can use a simple time.sleep to allow the UI to feel responsive
                # while we wait for the background thread to update the state.
                while st.session_state.get("is_processing"):
                    time.sleep(0.1)
            # After the spinner, the state should be updated
            # Note: st.rerun() removed - UI will update automatically on next interaction

        # Check if the workflow just finished to show a success message
        if st.session_state.get("just_finished"):
            st.success("CV Generation Complete!")
            st.session_state.just_finished = False  # Reset flag

        # Display any errors from the workflow
        if st.session_state.get("workflow_error"):
            error = st.session_state.workflow_error
            st.error(f"An error occurred during CV generation: {error}")
            st.session_state.workflow_error = None  # Clear the error        # 5. Render UI Tabs based on the current state
        # The button clicks in the UI now directly trigger the callbacks,
        # so no explicit workflow management is needed here.
        tab1, tab2, tab3 = st.tabs(
            ["📝 Input & Generate", "✏️ Review & Edit", "📄 Export"]
        )

        with tab1:
            display_input_form()

        with tab2:
            # Ensure agent_state exists before rendering tabs that depend on it
            if st.session_state.get("agent_state"):
                display_review_and_edit_tab(st.session_state.agent_state)
            else:
                st.info("Start CV generation in the first tab to see results here.")

        with tab3:
            if st.session_state.get("agent_state"):
                display_export_tab(st.session_state.agent_state)
            else:
                st.info(
                    "Complete CV generation to export your results."
                )  # Display any errors from the agent_state model itself
        if (
            st.session_state.get("agent_state")
            and st.session_state.agent_state.error_messages
        ):
            for error in st.session_state.agent_state.error_messages:
                st.error(error)

    except (ConfigurationError, ServiceInitializationError) as e:
        st.error(f"**Application Startup Failed:**\n\n{e}")
        st.warning(
            "Please check your configuration (.env file, database paths, API keys) and restart."
        )
        st.stop()
    except CATCHABLE_EXCEPTIONS as e:
        st.error(f"An unexpected error occurred: {e}")
        logger.error("Main application error", error=str(e), exc_info=True)

        # Show error details in expander for debugging
        with st.expander("🔍 Error Details", expanded=False):
            import traceback

            st.code(traceback.format_exc(), language="text")

        st.stop()


if __name__ == "__main__":
    main()
