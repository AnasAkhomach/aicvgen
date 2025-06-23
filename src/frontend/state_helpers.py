# In src/frontend/state_helpers.py
import streamlit as st
from typing import Optional
from ..orchestration.state import AgentState
import uuid

# NOTE: This module should only manage raw UI state (user input, flags, progress). Do NOT store complex objects like StructuredCV or AgentState here. See src/core/state_manager.py for architectural boundaries.


def initialize_session_state():
    """
    Initialize Streamlit session state with default values.
    This ensures all required keys exist before the UI components try to access them.
    """
    # Session management
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    # Core state
    if "agent_state" not in st.session_state:
        st.session_state.agent_state = None

    # Workflow control
    if "run_workflow" not in st.session_state:
        st.session_state.run_workflow = False

    if "processing" not in st.session_state:
        st.session_state.processing = False

    if "stop_processing" not in st.session_state:
        st.session_state.stop_processing = False

    # API configuration
    if "user_gemini_api_key" not in st.session_state:
        st.session_state.user_gemini_api_key = ""

    if "api_key_validated" not in st.session_state:
        st.session_state.api_key_validated = False

    # Token tracking
    if "session_tokens_used" not in st.session_state:
        st.session_state.session_tokens_used = 0

    if "session_token_limit" not in st.session_state:
        st.session_state.session_token_limit = 10000

    # Progress tracking
    if "current_step" not in st.session_state:
        st.session_state.current_step = ""

    if "progress" not in st.session_state:
        st.session_state.progress = 0.0

    if "status_message" not in st.session_state:
        st.session_state.status_message = ""

    # Input form data (temporary storage)
    if "job_description_input" not in st.session_state:
        st.session_state.job_description_input = ""

    if "cv_text_input" not in st.session_state:
        st.session_state.cv_text_input = ""

    if "start_from_scratch_input" not in st.session_state:
        st.session_state.start_from_scratch_input = False

    # Error handling
    if "error_messages" not in st.session_state:
        st.session_state.error_messages = []

    # Workflow execution state
    if "workflow_result" not in st.session_state:
        st.session_state.workflow_result = None

    if "workflow_error" not in st.session_state:
        st.session_state.workflow_error = None
