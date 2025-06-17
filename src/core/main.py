import sys
import os
import sys
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
    # Check if page config has already been set
    import streamlit.runtime.state as state

    if not hasattr(state, "_page_config_set"):
        st.set_page_config(
            page_title="AI CV Generator",
            page_icon="📄",
            layout="wide",
            initial_sidebar_state="expanded",
        )
        state._page_config_set = True
except:
    # If we can't access the state or page config is already set, continue
    pass

# Now import everything else after set_page_config
from typing import Dict, Any
import asyncio

# Project imports
from src.config.logging_config import setup_logging
from src.frontend.ui_components import (
    display_sidebar,
    display_input_form,
    display_review_and_edit_tab,
    display_export_tab
)
from src.frontend.state_helpers import initialize_session_state
from src.frontend.callbacks import handle_workflow_execution
from src.core.state_helpers import create_agent_state_from_ui
from src.orchestration.state import AgentState
from src.orchestration.cv_workflow_graph import cv_graph_app  # Assumes this is the compiled graph
import uuid
import time

# Initialize logging
logger = setup_logging()

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
        # 1. Initialize State
        initialize_session_state()

        # 2. Display Static UI Components
        display_sidebar(st.session_state.agent_state)
        st.title("🤖 AI CV Generator")
        st.markdown(
            "Transform your CV to match any job description using advanced AI. "
            "Get personalized, ATS-friendly CVs that highlight your most relevant skills and experience."
        )

        # 3. Main Interaction & Backend Loop
        if st.session_state.get('run_workflow'):
            st.session_state.processing = True
            st.session_state.run_workflow = False  # Reset flag
            
            # Generate trace_id for this workflow execution
            trace_id = str(uuid.uuid4())
            start_time = time.time()
            
            logger.info(
                "Starting CV generation workflow",
                extra={'trace_id': trace_id, 'session_id': st.session_state.get('session_id')}
            )

            with st.spinner("Processing..."):
                try:
                    handle_workflow_execution(trace_id)
                    duration = time.time() - start_time
                    logger.info(
                        "CV generation workflow completed successfully",
                        extra={
                            'trace_id': trace_id,
                            'session_id': st.session_state.get('session_id'),
                            'duration_seconds': duration
                        }
                    )
                except Exception as e:
                    duration = time.time() - start_time
                    logger.error(
                        f"CV generation workflow failed: {str(e)}",
                        extra={
                            'trace_id': trace_id,
                            'session_id': st.session_state.get('session_id'),
                            'duration_seconds': duration,
                            'error_type': type(e).__name__
                        }
                    )
                    raise

        # 4. Render UI Tabs based on the current state
        tab1, tab2, tab3 = st.tabs(["📝 Input & Generate", "✏️ Review & Edit", "📄 Export"])

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
                
    except Exception as e:
        logger.error(
            f"Error in main application: {str(e)}",
            extra={
                'session_id': st.session_state.get('session_id'),
                'error_type': type(e).__name__
            }
        )
        st.error(f"❌ Application error: {str(e)}")
        
        # Show error details in expander for debugging
        with st.expander("🔍 Error Details", expanded=False):
            import traceback
            st.code(traceback.format_exc(), language="text")


if __name__ == "__main__":
    main()
