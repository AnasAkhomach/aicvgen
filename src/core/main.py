import sys
import os
from pathlib import Path

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

"""
Main module for the AI CV Generator - A Streamlit application that helps users tailor their CVs
to specific job descriptions using AI. This version implements section-level control for editing
and regenerating CV content, which simplifies the user experience compared to the more granular
item-level approach.

Key features:
- Input handling for job descriptions and existing CVs
- Section-level editing and regeneration controls
- Multiple export formats
- Session management for saving and loading work

For more details, see the Software Design Document (SDD) v1.3 with Section-Level Control.
"""

import streamlit as st

# Page configuration - MUST be first Streamlit command when running directly
# Only set page config if this file is being run directly (not imported)
try:
    from ..utils.streamlit_utils import configure_page
    configure_page()
except Exception:
    # If we can't access the state or page config is already set, continue
    pass

# Now import everything else after set_page_config
from typing import Dict, Any
import asyncio

# Project imports
from ..core.application_startup import initialize_application, validate_application
from ..config.logging_config import get_logger
from ..frontend.ui_components import (
    display_sidebar,
    display_input_form,
    display_review_and_edit_tab,
    display_export_tab,
)
from ..frontend.state_helpers import initialize_session_state
from ..frontend.callbacks import handle_workflow_execution
from ..core.state_helpers import create_initial_agent_state
from ..orchestration.state import AgentState
from ..orchestration.cv_workflow_graph import (
    cv_graph_app,
)  # Assumes this is the compiled graph
from ..utils.exceptions import ConfigurationError, ServiceInitializationError
import uuid
import time

# Get logger (will be initialized by startup service)
logger = get_logger(__name__)

# Constants
TEMPLATE_FILE_PATH = "src/templates/cv_template.md"
SESSIONS_DIR = "data/sessions"
OUTPUT_DIR = "data/output"

# Ensure directories exist
os.makedirs(SESSIONS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    """
    Main Streamlit application controller.
    Orchestrates the UI rendering and backend workflow invocations.
    """
    try:
        # 1. Initialize Application Services
        user_api_key = st.session_state.get("user_gemini_api_key", "")
        startup_result = initialize_application(user_api_key=user_api_key)
        
        if not startup_result.success:
            st.error("**Application Startup Failed:**")
            for error in startup_result.errors:
                st.error(f"• {error}")
            st.warning("Please check your configuration and restart the application.")
            
            # Show startup details in expander
            with st.expander("🔍 Startup Details", expanded=False):
                st.json({
                    "total_time": f"{startup_result.total_time:.2f}s",
                    "services": {name: {
                        "status": "✅ Success" if service.initialized else "❌ Failed",
                        "time": f"{service.initialization_time:.3f}s",
                        "error": service.error
                    } for name, service in startup_result.services.items()}
                })
            st.stop()
        
        # Validate critical services
        validation_errors = validate_application()
        if validation_errors:
            st.error("**Critical Service Validation Failed:**")
            for error in validation_errors:
                st.error(f"• {error}")
            st.stop()
        
        logger.info(f"Application started successfully in {startup_result.total_time:.2f}s")

        # 2. Initialize Session State
        initialize_session_state()

        # 3. Display Static UI Components
        display_sidebar(st.session_state.agent_state)
        st.title("🤖 AI CV Generator")
        st.markdown(
            "Transform your CV to match any job description using advanced AI. "
            "Get personalized, ATS-friendly CVs that highlight your most relevant skills and experience."
        )

        # 4. Handle background thread results
        if st.session_state.get("processing"):
            st.spinner("Processing your CV... Please wait.")
            # The UI is not blocked here. You could add a progress bar that updates.

        if "workflow_result" in st.session_state and st.session_state.workflow_result:
            # Workflow is done, process the result
            final_state_dict = st.session_state.workflow_result
            st.session_state.agent_state = AgentState.model_validate(final_state_dict)
            st.session_state.workflow_result = None  # Clear the result

            # Clear feedback so the same action doesn't run again on the next rerun
            if st.session_state.agent_state.user_feedback:
                st.session_state.agent_state.user_feedback = None

            st.success("CV Generation Complete!")
            st.rerun()  # Rerun to display the new state in the 'Review & Edit' tab

        if "workflow_error" in st.session_state and st.session_state.workflow_error:
            # Workflow failed
            error = st.session_state.workflow_error
            st.error(f"An error occurred during CV generation: {error}")
            st.session_state.workflow_error = None  # Clear the error

        # 5. Main Interaction & Backend Loop
        if st.session_state.get("run_workflow"):
            st.session_state.processing = True
            st.session_state.run_workflow = False  # Reset flag

            # Generate trace_id for this workflow execution
            trace_id = str(uuid.uuid4())

            logger.info(
                "Starting CV generation workflow",
                extra={
                    "trace_id": trace_id,
                    "session_id": st.session_state.get("session_id"),
                },
            )

            handle_workflow_execution(trace_id)
            st.rerun()  # Rerun to show the spinner immediately

        # 6. Render UI Tabs based on the current state
        tab1, tab2, tab3 = st.tabs(
            ["📝 Input & Generate", "✏️ Review & Edit", "📄 Export"]
        )

        with tab1:
            display_input_form(st.session_state.agent_state)

        with tab2:
            display_review_and_edit_tab(st.session_state.agent_state)

        with tab3:
            display_export_tab(st.session_state.agent_state)

        # Display any errors from the workflow
        if st.session_state.agent_state and st.session_state.agent_state.error_messages:
            for error in st.session_state.agent_state.error_messages:
                st.error(error)

    except (ConfigurationError, ServiceInitializationError) as e:
        st.error(f"**Application Startup Failed:**\n\n{e}")
        st.warning(
            "Please check your configuration (.env file, database paths, API keys) and restart."
        )
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        
        # Show error details in expander for debugging
        with st.expander("🔍 Error Details", expanded=False):
            import traceback
            st.code(traceback.format_exc(), language="text")
        
        st.stop()


if __name__ == "__main__":
    main()
