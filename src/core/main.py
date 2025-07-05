import atexit
import logging
import sys
import traceback
from pathlib import Path

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Third-party imports
import streamlit as st

from src.config.logging_config import get_logger, setup_logging
# Project imports
from src.core.application_startup import get_startup_manager
from src.core.state_manager import StateManager
from src.error_handling.boundaries import CATCHABLE_EXCEPTIONS
from src.error_handling.exceptions import (ConfigurationError,
                                         ServiceInitializationError)
from src.ui.ui_manager import UIManager

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


def initialize_application(state_manager: StateManager) -> bool:
    """Initialize the application services and return success status.

    Args:
        state_manager: The state manager instance

    Returns:
        True if initialization succeeded, False otherwise
    """
    try:
        # Get startup manager
        startup_service = get_startup_manager()

        # Register the shutdown hook once (ignoring protected member warning for this specific case)
        if not hasattr(startup_service, "_shutdown_hook_registered") or not getattr(
            startup_service, "_shutdown_hook_registered", False
        ):
            atexit.register(startup_service.shutdown_application)
            setattr(startup_service, "_shutdown_hook_registered", True)

        # Initialize only if it hasn't been done already
        if not startup_service.is_initialized:
            # Set up basic logging config before initialization
            setup_logging()

            # Get user API key from state
            user_api_key = state_manager.user_gemini_api_key
            startup_result = startup_service.initialize_application(
                user_api_key=user_api_key
            )

            if not startup_result.success:
                return False

            logger.info(
                "Application started successfully in %.2fs", startup_result.total_time
            )

        # Always validate critical services (even if already initialized)
        validation_errors = startup_service.validate_application()
        if validation_errors:
            return False

        return True

    except (ConfigurationError, ServiceInitializationError) as e:
        logger.error("Application startup failed: %s", str(e))
        return False
    except CATCHABLE_EXCEPTIONS as e:
        logger.error("Unexpected error during startup: %s", str(e), exc_info=True)
        return False


def main():
    """
    Main Streamlit application entry point.

    This function serves as a thin orchestrator that initializes the state manager,
    UI manager, and coordinates the application flow while maintaining clean
    separation of concerns.
    """
    try:
        # Initialize state manager
        state_manager = StateManager()

        # Initialize UI manager
        ui_manager = UIManager(state_manager)

        # Initialize application services
        if not initialize_application(state_manager):
            # Handle startup failure
            startup_service = get_startup_manager()
            try:
                # Try to get startup result from the service
                startup_result = getattr(startup_service, "last_startup_result", None)
                if startup_result:
                    ui_manager.show_startup_error(
                        startup_result.errors, startup_result.services
                    )
                else:
                    st.error("Application startup failed. Please check configuration.")
            except CATCHABLE_EXCEPTIONS as e:
                logger.error("Error showing startup failure details: %s", str(e))
                st.error("Application startup failed. Please check configuration.")
            st.stop()

        # Check for validation errors
        startup_service = get_startup_manager()
        validation_errors = startup_service.validate_application()
        if validation_errors:
            ui_manager.show_validation_error(validation_errors)
            st.stop()

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

        # Use UI manager to show error if available, otherwise fallback
        try:
            ui_manager = UIManager(StateManager())
            ui_manager.show_unexpected_error(e)
        except (
            CATCHABLE_EXCEPTIONS
        ):  # Catch any exception during fallback error handling
            st.error(f"An unexpected error occurred: {e}")
            with st.expander("🔍 Error Details", expanded=False):
                st.code(traceback.format_exc(), language="text")

        st.stop()


if __name__ == "__main__":
    main()
