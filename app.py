#!/usr/bin/env python3
"""
Main launcher for the AI CV Generator Streamlit application.
This file implements the Interactive UI Loop driven by workflow status.

Refactored to use the centralized ApplicationStartupService as the single source of truth
for all initialization logic.
"""
import streamlit as st
from src.frontend.ui_manager import UIManager
from src.error_handling.boundaries import safe_streamlit_component
from src.config.logging_config import get_logger
from src.core.application_startup import get_startup_manager
from src.error_handling.exceptions import ConfigurationError, ServiceInitializationError
from src.error_handling.boundaries import CATCHABLE_EXCEPTIONS

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="AI CV Generator",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Get logger (will be properly initialized by startup service)
logger = get_logger(__name__)


@safe_streamlit_component(component_name="main_app")
def main():
    """Main application entry point with workflow-driven UI.

    Refactored to use the centralized ApplicationStartupService.
    """
    try:
        # Get the startup manager (single source of truth for initialization)
        startup_service = get_startup_manager()

        # Initialize application using the centralized service
        if not startup_service.is_initialized:
            startup_result = startup_service.initialize_application()

            if not startup_result.success:
                # Handle startup failure
                error_message = (
                    "\n".join(startup_result.errors)
                    if startup_result.errors
                    else "Unknown startup error"
                )
                st.error(f"**Application Startup Failed:**\n\n{error_message}")
                st.warning(
                    "Please check your configuration (.env file, database paths, API keys) and restart."
                )
                st.stop()
                return

            logger.info(
                f"Application started successfully in {startup_result.total_time:.2f}s"
            )

        # Validate application state
        validation_errors = startup_service.validate_application()
        if validation_errors:
            st.error("**Application Validation Failed:**")
            for error in validation_errors:
                st.error(f"• {error}")
            st.warning("Please check your configuration and restart.")
            st.stop()
            return

        # Get the initialized StateManager from the startup service
        state_manager = startup_service.get_state_manager()

        # Initialize UI Manager with the state manager
        ui_manager = UIManager(state_manager=state_manager)

        # Initialize facade from the container
        container = startup_service.container
        ui_manager.initialize_facade(container.cv_generation_facade())

        # Render the complete UI
        ui_manager.render_full_ui()

    except (ConfigurationError, ServiceInitializationError) as e:
        st.error(f"**Application Startup Failed:**\n\n{e}")
        st.warning(
            "Please check your configuration (.env file, database paths, API keys) and restart."
        )
        st.stop()
    except CATCHABLE_EXCEPTIONS as e:
        logger.error("Main application error: %s", str(e), exc_info=True)
        st.error(f"An unexpected error occurred: {e}")
        with st.expander("🔍 Error Details", expanded=False):
            import traceback

            st.code(traceback.format_exc(), language="text")
        st.stop()


if __name__ == "__main__":
    main()
