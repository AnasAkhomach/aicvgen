# In src/frontend/ui_components.py
from pathlib import Path
import streamlit as st

from src.config.logging_config import get_logger
from src.frontend.callbacks import (handle_api_key_validation, handle_user_action, start_cv_generation)
from src.orchestration.state import AgentState

# Initialize logger
logger = get_logger(__name__)


def display_sidebar():
    """Renders the sidebar for session management and settings."""
    with st.sidebar:
        st.title("🔧 Session Management")

        # API Key Configuration Section
        st.subheader("🔑 API Key Configuration")

        # User API Key Input
        user_api_key = st.text_input(
            "Enter your Gemini API Key",
            value=st.session_state.get("user_gemini_api_key", ""),
            type="password",
            help="Get your free API key from https://aistudio.google.com/app/apikey",
            placeholder="Enter your Gemini API key here...",
        )

        # Validate and save API key
        if user_api_key and user_api_key != st.session_state.get(
            "user_gemini_api_key", ""
        ):
            st.session_state.user_gemini_api_key = user_api_key
            st.session_state.api_key_validated = (
                False  # Reset validation when key changes
            )

        # API Key Validation Button and Status
        if st.session_state.get("user_gemini_api_key"):
            col1, col2 = st.columns([2, 1])

            with col1:
                if st.session_state.get("api_key_validated"):
                    st.success("✅ API Key validated and ready to use!")
                elif st.session_state.get("api_key_validation_failed"):
                    st.error("❌ API Key validation failed. Please check your key.")
                else:
                    st.info("🔄 Click 'Validate' to test your API key")

            with col2:
                st.button(
                    "🔍 Validate",
                    key="validate_api_key",
                    help="Test your API key",
                    on_click=handle_api_key_validation,  # Use on_click for direct callback
                )

        else:
            st.error("⚠️ Please enter your Gemini API key to use the application")
            st.markdown(
                """
            **How to get your API key:**
            1. Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
            2. Sign in with your Google account
            3. Click "Create API Key"
            4. Copy and paste the key above
            """
            )

        st.divider()  # Safety Controls Section
        st.subheader("🛡️ Safety Controls")  # Manual Stop Button
        if st.session_state.get("is_processing"):
            if st.button(
                "🛑 STOP PROCESSING", type="primary", use_container_width=True
            ):
                st.session_state.stop_processing = True
                st.session_state.is_processing = False
                st.warning("Processing stopped by user")
                # Note: st.rerun() removed - calling within callback is not allowed

        # Token Usage Display
        st.subheader("📊 Token Usage")

        # Session tokens
        session_usage_pct = 0
        if (
            "session_token_limit" in st.session_state
            and st.session_state.session_token_limit > 0
        ):
            session_usage_pct = (
                st.session_state.get("session_tokens_used", 0)
                / st.session_state.session_token_limit
            ) * 100

        st.metric(
            "Session Tokens",
            f"{st.session_state.get('session_tokens_used', 0):,}",
            f"{session_usage_pct:.1f}% of limit",
        )

        # Budget Limits Configuration
        st.subheader("💰 Budget Limits")

        session_limit = st.number_input(
            "Session Token Limit",
            min_value=1000,
            max_value=100000,
            value=st.session_state.get("session_token_limit", 10000),
            step=1000,
            help="Maximum tokens allowed per session",
        )

        if session_limit != st.session_state.get("session_token_limit"):
            st.session_state.session_token_limit = session_limit

        st.divider()

        # Session Management
        st.subheader("💾 Session Management")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Save", use_container_width=True):
                # Handle save session
                pass

        with col2:
            if st.button("📂 Load", use_container_width=True):
                # Handle load session
                pass

        if st.button("🆕 New Session", use_container_width=True):
            # Handle new session
            pass


def display_input_form():
    """Renders the initial input form for job description and CV text."""
    st.header("1. Input Your Information")

    # Check if we have a valid API key before allowing input
    if not st.session_state.get("user_gemini_api_key"):
        st.warning(
            "⚠️ Please enter your Gemini API key in the sidebar before proceeding."
        )
        return

    # Job Description Input
    job_description = st.text_area(
        "🎯 Job Description",
        height=200,
        key="job_description_input",  # Use key expected by state_helpers
        placeholder="Paste the job description here...",
        help="Provide the complete job description to tailor your CV accordingly.",
    )

    # CV Content Input
    cv_content = st.text_area(
        "📄 Your Current CV",
        height=300,
        key="cv_text_input",  # Use key expected by state_helpers
        placeholder="Paste your current CV content here, or leave empty to start from scratch...",
        help="Provide your existing CV content, or leave empty to create a new CV structure.",
    )

    # Start from scratch option
    start_from_scratch = st.checkbox(
        "🆕 Start from scratch (ignore existing CV content)",
        key="start_from_scratch_input",  # Use key expected by state_helpers
        help="Check this to create a completely new CV structure based only on the job description.",
    )

    # Generate button
    can_generate = bool(job_description.strip()) and (
        bool(cv_content.strip()) or start_from_scratch
    )

    # Check API key validation status
    api_key_validated = st.session_state.get("api_key_validated", False)

    # Show validation warning if API key is not validated
    if not api_key_validated and st.session_state.get("user_gemini_api_key"):
        st.warning(
            "⚠️ Please validate your API key in the sidebar before generating a CV."
        )

    if st.button(
        "🚀 Generate Tailored CV",
        type="primary",
        use_container_width=True,
        disabled=not can_generate
        or not api_key_validated
        or st.session_state.get("is_processing", False),
        on_click=start_cv_generation,
    ):
        # The on_click callback now handles all logic.
        # The widgets write directly to the correct session_state keys.
        pass

    if not can_generate and not st.session_state.get("is_processing", False):
        if not job_description.strip():
            st.info("💡 Please provide a job description to get started.")
        elif not cv_content.strip() and not start_from_scratch:
            st.info("💡 Please provide your CV content or check 'Start from scratch'.")


def display_review_and_edit_tab(state: AgentState):
    """Renders the 'Review & Edit' tab with section-based controls."""
    if not state or not state.structured_cv:
        st.info("Please generate a CV first to review it here.")
        return

    for section in state.structured_cv.sections:
        with st.expander(f"### {section.name}", expanded=True):
            if section.items:
                for item in section.items:
                    _display_reviewable_item(item)
            if section.subsections:
                for sub in section.subsections:
                    st.markdown(f"#### {sub.name}")
                    for item in sub.items:
                        _display_reviewable_item(item)


def _display_reviewable_item(item):
    """Displays a single reviewable item with Accept/Regenerate buttons."""
    item_id = str(item.id)
    st.markdown(f"> {item.content}")

    # Raw LLM Output for transparency
    if hasattr(item, "raw_llm_output") and item.raw_llm_output:
        with st.expander("🔍 View Raw LLM Output", expanded=False):
            st.code(item.raw_llm_output, language="text")

    cols = st.columns([1, 1, 5])
    with cols[0]:
        st.button(
            "✅ Accept",
            key=f"accept_{item_id}",
            on_click=handle_user_action,
            args=("accept", item_id),
        )
    with cols[1]:
        st.button(
            "🔄 Regenerate",
            key=f"regenerate_{item_id}",
            on_click=handle_user_action,
            args=("regenerate", item_id),
        )


def display_export_tab(state: AgentState):
    """Renders the 'Export' tab."""
    if state and hasattr(state, "final_output_path") and state.final_output_path:
        st.success("✅ Your CV has been generated!")
        with open(state.final_output_path, "rb") as f:
            st.download_button(
                label="📥 Download PDF",
                data=f,
                file_name=Path(state.final_output_path).name,
                mime="application/pdf",
            )
    else:
        st.info("Generate and finalize your CV to enable export options.")
