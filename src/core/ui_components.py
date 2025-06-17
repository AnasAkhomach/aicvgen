"""UI Components Module (DEPRECATED)

UI Components for the AI CV Generator Streamlit.

This module contains legacy UI rendering functions.
NEW CODE SHOULD USE: src.frontend.ui_components instead.

This module is maintained for backward compatibility only.
"""

# DEPRECATION WARNING
import warnings
warnings.warn(
    "src.core.ui_components is deprecated. Use src.frontend.ui_components instead.",
    DeprecationWarning,
    stacklevel=2
)

import streamlit as st
import time
from datetime import datetime
from typing import Dict, Any, Optional, List
from src.config.logging_config import setup_logging
from src.core.state_manager import StateManager
from src.models.data_models import StructuredCV, JobDescriptionData

# Initialize logging
logger = setup_logging()


def display_sidebar() -> None:
    """Render the sidebar with session management and settings."""
    with st.sidebar:
        st.title("🔧 Session Management")

        # API Key Configuration Section
        _display_api_key_section()
        st.divider()

        # Safety Controls Section
        _display_safety_controls()

        # Token Usage Display
        _display_token_usage()

        # Budget Limits Configuration
        _display_budget_limits()

        st.divider()

        # Session Management
        _display_session_management()


def _display_api_key_section() -> None:
    """Display API key configuration section."""
    st.subheader("🔑 API Key Configuration")

    # User API Key Input
    user_api_key = st.text_input(
        "Enter your Gemini API Key",
        value=st.session_state.user_gemini_api_key,
        type="password",
        help="Get your free API key from https://aistudio.google.com/app/apikey",
        placeholder="Enter your Gemini API key here...",
    )

    # Validate and save API key
    if user_api_key and user_api_key != st.session_state.user_gemini_api_key:
        st.session_state.user_gemini_api_key = user_api_key
        st.session_state.api_key_validated = False  # Reset validation when key changes

    # Show validation status
    if st.session_state.user_gemini_api_key:
        if st.session_state.api_key_validated:
            st.success("✅ API Key validated and ready to use!")
        else:
            st.info("🔄 API Key will be validated when you start processing")
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


def _display_safety_controls() -> None:
    """Display safety controls section."""
    st.subheader("🛡️ Safety Controls")

    # Manual Stop Button
    if st.session_state.processing:
        if st.button("🛑 STOP PROCESSING", type="primary", use_container_width=True):
            st.session_state.stop_processing = True
            st.session_state.processing = False
            st.warning("Processing stopped by user")
            st.rerun()


def _display_token_usage() -> None:
    """Display token usage metrics."""
    st.subheader("📊 Token Usage")

    # Session tokens
    session_usage_pct = 0
    if (
        "session_token_limit" in st.session_state
        and st.session_state.session_token_limit > 0
    ):
        session_usage_pct = (
            st.session_state.token_usage["session_tokens"]
            / st.session_state.session_token_limit
        ) * 100

    st.metric(
        "Session Tokens",
        f"{st.session_state.token_usage['session_tokens']:,} / {st.session_state.session_token_limit:,}",
        f"{session_usage_pct:.1f}%",
    )
    st.progress(min(session_usage_pct / 100, 1.0))

    # Daily tokens
    daily_usage_pct = 0
    if (
        "daily_token_limit" in st.session_state
        and st.session_state.daily_token_limit > 0
    ):
        daily_usage_pct = (
            st.session_state.token_usage["daily_tokens"]
            / st.session_state.daily_token_limit
        ) * 100

    st.metric(
        "Daily Tokens",
        f"{st.session_state.token_usage['daily_tokens']:,} / {st.session_state.daily_token_limit:,}",
        f"{daily_usage_pct:.1f}%",
    )
    st.progress(min(daily_usage_pct / 100, 1.0))

    # Warning indicators
    if session_usage_pct > 80:
        st.error("⚠️ Session token limit nearly reached!")
    elif session_usage_pct > 60:
        st.warning("⚠️ High session token usage")

    if daily_usage_pct > 80:
        st.error("⚠️ Daily token limit nearly reached!")
    elif daily_usage_pct > 60:
        st.warning("⚠️ High daily token usage")


def _display_budget_limits() -> None:
    """Display budget limits configuration."""
    st.subheader("💰 Budget Limits")

    new_session_limit = st.number_input(
        "Session Token Limit",
        min_value=1000,
        max_value=100000,
        value=st.session_state.session_token_limit,
        step=1000,
    )

    new_daily_limit = st.number_input(
        "Daily Token Limit",
        min_value=10000,
        max_value=500000,
        value=st.session_state.daily_token_limit,
        step=5000,
    )

    if st.button("Update Limits"):
        st.session_state.session_token_limit = new_session_limit
        st.session_state.daily_token_limit = new_daily_limit
        st.success("Limits updated!")


def _display_session_management() -> None:
    """Display session management controls."""
    from src.core.session_utils import save_session, get_available_sessions, load_session
    
    st.subheader("💾 Sessions")

    # Current session info
    st.write(f"**Current Session:** `{st.session_state.session_id[:8]}...`")

    # Save current session
    if st.button("💾 Save Session", use_container_width=True):
        if save_session(st.session_state.state_manager):
            st.success("Session saved!")
        else:
            st.error("Failed to save session")

    # Load existing session
    sessions = get_available_sessions()
    if sessions:
        st.write("**Load Existing Session:**")
        session_options = [f"{s['id'][:8]}... ({s['step']})" for s in sessions]
        selected_session = st.selectbox("Select session", session_options, index=None)

        if selected_session and st.button("📂 Load Session", use_container_width=True):
            session_id = sessions[session_options.index(selected_session)]["id"]
            if load_session(session_id):
                st.success("Session loaded!")
                st.rerun()
            else:
                st.error("Failed to load session")

    # New session
    if st.button("🆕 New Session", use_container_width=True):
        # Reset session state
        for key in list(st.session_state.keys()):
            if key not in ["session_token_limit", "daily_token_limit"]:
                del st.session_state[key]
        st.rerun()


def display_input_form() -> None:
    """Render the input form tab."""
    st.header("Input Your Information")

    # Job description input
    st.subheader("🎯 Job Description")
    job_description = st.text_area(
        "Paste the job description here:",
        value=st.session_state.job_description,
        height=200,
        help="Paste the complete job description to tailor your CV accordingly",
    )

    # CV content input
    st.subheader("📄 Your Current CV")
    cv_content = st.text_area(
        "Paste your current CV content here:",
        value=st.session_state.cv_content,
        height=300,
        help="Paste your existing CV content that will be tailored to the job",
    )

    # Update session state
    st.session_state.job_description = job_description
    st.session_state.cv_content = cv_content

    # Generate button
    _display_generate_button(job_description, cv_content)


def _display_generate_button(job_description: str, cv_content: str) -> None:
    """Display the generate CV button with validation."""
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button(
            "🚀 Generate Tailored CV",
            type="primary",
            use_container_width=True,
            disabled=st.session_state.processing,
        ):
            if not job_description.strip():
                st.error("Please provide a job description")
            elif not cv_content.strip():
                st.error("Please provide your CV content")
            else:
                # Check budget limits before starting
                session_usage_pct = (
                    st.session_state.token_usage["session_tokens"]
                    / st.session_state.session_token_limit
                ) * 100
                daily_usage_pct = (
                    st.session_state.token_usage["daily_tokens"]
                    / st.session_state.daily_token_limit
                ) * 100

                if session_usage_pct >= 100:
                    st.error(
                        "❌ Session token limit exceeded! Please start a new session or increase limits."
                    )
                elif daily_usage_pct >= 100:
                    st.error(
                        "❌ Daily token limit exceeded! Please try again tomorrow or increase limits."
                    )
                else:
                    # Trigger CV generation workflow
                    from src.core.callbacks import handle_cv_generation
                    handle_cv_generation(job_description, cv_content)


def display_review_edit_tab() -> None:
    """Render the review and edit tab."""
    st.header("Review & Edit Your CV")

    if st.session_state.state_manager and st.session_state.state_manager.current_cv:
        cv_data = st.session_state.state_manager.current_cv
        
        # Display structured CV sections
        if hasattr(cv_data, 'sections') and cv_data.sections:
            for section in cv_data.sections:
                display_section(section, st.session_state.state_manager)
        else:
            st.info("No CV sections available. Please generate a CV first.")
    else:
        st.info(
            "No CV data available. Please generate a CV first in the 'Input & Generate' tab."
        )


def display_export_tab() -> None:
    """Render the export tab."""
    st.header("Export Your CV")

    if st.session_state.state_manager and st.session_state.state_manager.current_cv:
        cv_data = st.session_state.state_manager.current_cv
        
        # Get content data for rendering
        if hasattr(cv_data, 'model_dump'):
            content_data = cv_data.model_dump().get("content", {})
        else:
            content_data = getattr(cv_data, 'content', {})

        if content_data:
            # Add toggle for showing raw LLM output
            show_raw_output = st.checkbox(
                "🔍 Show Raw LLM Output",
                value=False,
                help="Toggle to see the raw LLM responses for transparency"
            )

            # Generate rendered Markdown
            rendered_cv = _generate_cv_markdown(content_data, show_raw_output)

            # Show preview
            st.markdown(rendered_cv)

            # Export options
            _display_export_options(rendered_cv)

        else:
            # Create a simple preview
            preview = _generate_simple_preview(content_data)
            st.markdown(preview)
            st.info(
                "Complete your CV in the 'Review & Edit' tab to see the full preview."
            )
    else:
        st.info(
            "No CV data available. Please generate a CV first in the 'Input & Generate' tab."
        )


def _generate_cv_markdown(content_data: Dict[str, Any], show_raw_output: bool) -> str:
    """Generate markdown representation of the CV."""
    rendered_cv = f"""
# {content_data.get('name', 'Your Name')}

{content_data.get('contact_info', 'Contact information')}

---

## Executive Summary

{content_data.get('executive_summary', 'Executive summary content')}

## Key Qualifications

{content_data.get('key_qualifications', 'Key qualifications content')}

## 🎯 Big 10 Skills
"""

    # Add Big 10 skills display
    big_10_skills = content_data.get('big_10_skills', [])
    if big_10_skills:
        for i, skill in enumerate(big_10_skills, 1):
            rendered_cv += f"\n{i}. {skill}"
    else:
        rendered_cv += "\nBig 10 skills will appear here after CV generation."

    # Add raw output section if toggled
    if show_raw_output:
        raw_output = content_data.get('big_10_skills_raw_output', '')
        if raw_output:
            rendered_cv += f"\n\n### 🔍 Raw LLM Output for Big 10 Skills\n\n```\n{raw_output}\n```"

    rendered_cv += "\n\n## Professional Experience\n"
    # Add experience bullets
    for bullet in content_data.get("experience_bullets", []):
        rendered_cv += f"\n* {bullet}"

    rendered_cv += "\n\n## Projects\n"
    for project in content_data.get("projects", []):
        rendered_cv += f"\n### {project.get('title', 'Project Title')}\n{project.get('description', 'Project description')}"

    return rendered_cv


def _generate_simple_preview(content_data: Dict[str, Any]) -> str:
    """Generate a simple preview of the CV."""
    preview = f"""
# {content_data.get('name', 'Your Name')}

{content_data.get('contact_info', 'Contact information will appear here')}

---

## Experience
"""
    for bullet in content_data.get("experience_bullets", [])[:3]:  # Show first 3 bullets
        preview += f"\n* {bullet}"

    if len(content_data.get("experience_bullets", [])) > 3:
        preview += "\n* ..."

    return preview


def _display_export_options(rendered_cv: str) -> None:
    """Display export options for the CV."""
    st.subheader("💾 Export Options")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📄 Download as Markdown", use_container_width=True):
            st.download_button(
                label="Download MD",
                data=rendered_cv,
                file_name=f"tailored_cv_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
            )

    with col2:
        if st.button("📝 Download as Text", use_container_width=True):
            # Convert markdown to plain text (simple conversion)
            plain_text = (
                rendered_cv.replace("#", "")
                .replace("*", "")
                .replace("---", "")
            )
            st.download_button(
                label="Download TXT",
                data=plain_text,
                file_name=f"tailored_cv_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
            )

    with col3:
        st.button(
            "📊 Generate PDF",
            use_container_width=True,
            disabled=True,
            help="PDF export coming soon!",
        )


def display_section(section: Dict[str, Any], state_manager: StateManager) -> None:
    """Display a section with its items or subsections in a card-based UI."""
    if not section:
        return

    # Create a container for the section
    with st.container():
        # Section header with controls
        col1, col2, col3 = st.columns([3, 1, 1])

        with col1:
            st.subheader(section.get("title", "Untitled Section"))

        with col2:
            if st.button(f"✏️ Edit", key=f"edit_section_{section.get('id', 'unknown')}"):
                st.session_state[f"editing_section_{section.get('id', 'unknown')}"] = True

        with col3:
            if st.button(f"🔄 Regenerate", key=f"regen_section_{section.get('id', 'unknown')}"):
                # Trigger regeneration for this section
                st.session_state[f"regenerate_section_{section.get('id', 'unknown')}"] = True
                st.rerun()

        # Check if we're in editing mode for this section
        if st.session_state.get(f"editing_section_{section.get('id', 'unknown')}", False):
            _display_section_edit_mode(section, state_manager)
        else:
            _display_section_view_mode(section, state_manager)


def _display_section_edit_mode(section: Dict[str, Any], state_manager: StateManager) -> None:
    """Display section in edit mode."""
    # Show editing interface
    new_title = st.text_input(
        "Section Title",
        value=section.get("title", ""),
        key=f"title_input_{section.get('id', 'unknown')}",
    )

    # Handle subsections if they exist
    if "subsections" in section:
        st.write("**Subsections:**")
        for subsection in section["subsections"]:
            display_subsection(subsection, section, state_manager)

    # Handle items if they exist
    elif "items" in section:
        st.write("**Items:**")
        for item in section["items"]:
            display_item(item, section, None, state_manager)

    # Save/Cancel buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 Save", key=f"save_section_{section.get('id', 'unknown')}"):
            # Update the section title
            section["title"] = new_title
            state_manager.update_section(section)
            st.session_state[f"editing_section_{section.get('id', 'unknown')}"] = False
            st.success("Section updated!")
            st.rerun()

    with col2:
        if st.button("❌ Cancel", key=f"cancel_section_{section.get('id', 'unknown')}"):
            st.session_state[f"editing_section_{section.get('id', 'unknown')}"] = False
            st.rerun()


def _display_section_view_mode(section: Dict[str, Any], state_manager: StateManager) -> None:
    """Display section in view mode."""
    if "subsections" in section:
        for subsection in section["subsections"]:
            display_subsection(subsection, section, state_manager)
    elif "items" in section:
        for item in section["items"]:
            display_item(item, section, None, state_manager)
    else:
        st.write("*No content available*")


def display_subsection(subsection: Dict[str, Any], parent_section: Dict[str, Any], state_manager: StateManager) -> None:
    """Display a subsection with its items."""
    with st.expander(f"📁 {subsection.get('title', 'Untitled Subsection')}", expanded=True):
        # Subsection controls
        col1, col2 = st.columns([1, 1])

        with col1:
            if st.button(
                f"✏️ Edit Subsection",
                key=f"edit_subsection_{subsection.get('id', 'unknown')}",
            ):
                st.session_state[f"editing_subsection_{subsection.get('id', 'unknown')}"] = True

        # Display items in the subsection
        if "items" in subsection:
            for item in subsection["items"]:
                display_item(item, parent_section, subsection, state_manager)


def display_item(item: Dict[str, Any], parent_section: Dict[str, Any], parent_subsection: Optional[Dict[str, Any]], state_manager: StateManager) -> None:
    """Display an individual item with edit/regenerate controls."""
    item_id = item.get('id', 'unknown')
    
    with st.container():
        # Item header
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.write(f"**{item.get('title', 'Untitled Item')}**")
        
        with col2:
            if st.button(f"✏️ Edit", key=f"edit_item_{item_id}"):
                st.session_state[f"editing_item_{item_id}"] = True
                st.rerun()
        
        with col3:
            if st.button(f"🔄 Regenerate", key=f"regen_item_{item_id}"):
                # Trigger item regeneration
                from src.core.callbacks import handle_item_regeneration
                handle_item_regeneration(item_id, parent_section, parent_subsection, item, state_manager)
        
        # Display item content
        if st.session_state.get(f"editing_item_{item_id}", False):
            _display_item_edit_mode(item, parent_section, parent_subsection, state_manager)
        else:
            _display_item_view_mode(item)


def _display_item_edit_mode(item: Dict[str, Any], parent_section: Dict[str, Any], parent_subsection: Optional[Dict[str, Any]], state_manager: StateManager) -> None:
    """Display item in edit mode."""
    item_id = item.get('id', 'unknown')
    
    # Edit form
    new_title = st.text_input(
        "Item Title",
        value=item.get("title", ""),
        key=f"title_input_item_{item_id}",
    )
    
    new_content = st.text_area(
        "Item Content",
        value=item.get("content", ""),
        key=f"content_input_item_{item_id}",
        height=100,
    )
    
    # Save/Cancel buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 Save", key=f"save_item_{item_id}"):
            # Update the item
            item["title"] = new_title
            item["content"] = new_content
            state_manager.update_item(item)
            st.session_state[f"editing_item_{item_id}"] = False
            st.success("Item updated!")
            st.rerun()
    
    with col2:
        if st.button("❌ Cancel", key=f"cancel_item_{item_id}"):
            st.session_state[f"editing_item_{item_id}"] = False
            st.rerun()


def _display_item_view_mode(item: Dict[str, Any]) -> None:
    """Display item in view mode."""
    content = item.get('content', 'No content available')
    status = item.get('status', 'unknown')
    
    # Status indicator
    if status == 'completed':
        st.success(f"✅ {content}")
    elif status == 'processing':
        st.info(f"🔄 {content}")
    elif status == 'error':
        st.error(f"❌ {content}")
    else:
        st.write(content)


def show_template_info() -> None:
    """Display template information."""
    with st.expander("ℹ️ About the CV Template", expanded=False):
        st.markdown(
            """
        This AI CV Generator uses a structured template to create professional CVs.
        
        **Features:**
        - 🎯 **Big 10 Skills**: Top 10 most relevant skills for the job
        - 📝 **Tailored Content**: Content specifically adapted to the job description
        - 🔍 **Transparency**: View raw LLM outputs for full transparency
        - ✏️ **Editable**: Review and edit all generated content
        - 📊 **Multiple Formats**: Export as Markdown, Text, or PDF
        
        **How it works:**
        1. Paste your job description and current CV
        2. AI agents analyze and tailor your content
        3. Review and edit the generated sections
        4. Export in your preferred format
        """
        )


def display_progress_indicator() -> None:
    """Display progress indicator when processing."""
    if st.session_state.processing:
        st.info(f"🔄 {st.session_state.status_message}")
        progress_bar = st.progress(st.session_state.progress / 100)

        # Check for stop signal
        if st.session_state.stop_processing:
            st.warning("⏹️ Processing interrupted by user")
            st.session_state.processing = False
            st.session_state.stop_processing = False