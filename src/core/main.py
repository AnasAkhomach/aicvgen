#!/usr/bin/env python3
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
import json
import uuid
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging
import traceback
import enum


def enum_to_value(obj):
    """Recursively convert enum objects to their values for JSON serialization."""
    if isinstance(obj, dict):
        return {k: enum_to_value(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [enum_to_value(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(enum_to_value(v) for v in obj)
    elif isinstance(obj, set):
        return {enum_to_value(v) for v in obj}
    elif isinstance(obj, enum.Enum):
        return obj.value
    elif hasattr(obj, 'to_dict') and callable(getattr(obj, 'to_dict')):
        # Handle objects with to_dict method (like EnhancedCVConfig)
        try:
            return enum_to_value(obj.to_dict())
        except Exception:
            # Fall back to attribute inspection if to_dict fails
            pass
    elif hasattr(obj, '__dict__'):
        # Handle objects with attributes that might contain enums
        try:
            return {k: enum_to_value(v) for k, v in obj.__dict__.items() if not k.startswith('_')}
        except:
            return str(obj)
    else:
        return obj


class EnumEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, enum.Enum):
            return obj.value
        return super().default(obj)


# Import project modules after Streamlit configuration
from src.integration.enhanced_cv_system import (
    get_enhanced_cv_integration,
    EnhancedCVConfig,
    IntegrationMode,
)
from src.core.state_manager import (
    VectorStoreConfig,
    CVData,
    WorkflowState,
    StateManager,
)
from src.models.data_models import (
    JobDescriptionData,
    StructuredCV,
    Section,
    Subsection,
    Item,
)
from src.services.llm import LLM

# ContentWriterAgent no longer needed - using enhanced CV integration
from src.agents.cv_analyzer_agent import CVAnalyzerAgent
from src.agents.formatter_agent import FormatterAgent
from src.agents.quality_assurance_agent import QualityAssuranceAgent
from src.agents.research_agent import ResearchAgent

# ToolsAgent no longer needed - using enhanced CV integration
from src.services.session_manager import SessionManager
from src.services.progress_tracker import ProgressTracker
from src.services.rate_limiter import RateLimiter
from src.services.error_recovery import ErrorRecoveryService
from src.services.item_processor import ItemProcessor
from src.config.logging_config import setup_logging
from src.config.settings import AppConfig
from src.config.environment import Environment

# Initialize logging
logger = setup_logging()

# Constants
TEMPLATE_FILE_PATH = "src/templates/cv_template.md"
SESSIONS_DIR = "data/sessions"
OUTPUT_DIR = "data/output"

# Ensure directories exist
os.makedirs(SESSIONS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# Helper functions for the UI
def display_section(section, state_manager):
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
                st.session_state[f"editing_section_{section.get('id', 'unknown')}"] = (
                    True
                )

        with col3:
            if st.button(
                f"🔄 Regenerate", key=f"regen_section_{section.get('id', 'unknown')}"
            ):
                # Trigger regeneration for this section
                st.session_state[
                    f"regenerate_section_{section.get('id', 'unknown')}"
                ] = True
                st.rerun()

        # Check if we're in editing mode for this section
        if st.session_state.get(
            f"editing_section_{section.get('id', 'unknown')}", False
        ):
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
                if st.button(
                    "💾 Save", key=f"save_section_{section.get('id', 'unknown')}"
                ):
                    # Update the section title
                    section["title"] = new_title
                    state_manager.update_section(section)
                    st.session_state[
                        f"editing_section_{section.get('id', 'unknown')}"
                    ] = False
                    st.success("Section updated!")
                    st.rerun()

            with col2:
                if st.button(
                    "❌ Cancel", key=f"cancel_section_{section.get('id', 'unknown')}"
                ):
                    st.session_state[
                        f"editing_section_{section.get('id', 'unknown')}"
                    ] = False
                    st.rerun()

        else:
            # Display mode
            if "subsections" in section:
                for subsection in section["subsections"]:
                    display_subsection(subsection, section, state_manager)
            elif "items" in section:
                for item in section["items"]:
                    display_item(item, section, None, state_manager)
            else:
                st.write("*No content available*")


def display_subsection(subsection, parent_section, state_manager):
    """Display a subsection with its items."""
    with st.expander(
        f"📁 {subsection.get('title', 'Untitled Subsection')}", expanded=True
    ):
        # Subsection controls
        col1, col2 = st.columns([1, 1])

        with col1:
            if st.button(
                f"✏️ Edit Subsection",
                key=f"edit_subsection_{subsection.get('id', 'unknown')}",
            ):
                st.session_state[
                    f"editing_subsection_{subsection.get('id', 'unknown')}"
                ] = True

        with col2:
            if st.button(
                f"🔄 Regenerate Subsection",
                key=f"regen_subsection_{subsection.get('id', 'unknown')}",
            ):
                st.session_state[
                    f"regenerate_subsection_{subsection.get('id', 'unknown')}"
                ] = True
                st.rerun()

        # Check if we're editing this subsection
        if st.session_state.get(
            f"editing_subsection_{subsection.get('id', 'unknown')}", False
        ):
            new_title = st.text_input(
                "Subsection Title",
                value=subsection.get("title", ""),
                key=f"subsection_title_{subsection.get('id', 'unknown')}",
            )

            # Save/Cancel buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button(
                    "💾 Save", key=f"save_subsection_{subsection.get('id', 'unknown')}"
                ):
                    subsection["title"] = new_title
                    state_manager.update_subsection(parent_section, subsection)
                    st.session_state[
                        f"editing_subsection_{subsection.get('id', 'unknown')}"
                    ] = False
                    st.success("Subsection updated!")
                    st.rerun()

            with col2:
                if st.button(
                    "❌ Cancel",
                    key=f"cancel_subsection_{subsection.get('id', 'unknown')}",
                ):
                    st.session_state[
                        f"editing_subsection_{subsection.get('id', 'unknown')}"
                    ] = False
                    st.rerun()

        # Display items in this subsection
        if "items" in subsection:
            for item in subsection["items"]:
                display_item(item, parent_section, subsection, state_manager)
        else:
            st.write("*No items in this subsection*")


def display_item(item, section, subsection, state_manager):
    """Display an individual item with editing and feedback controls."""
    # Create a card-like container for the item
    with st.container():
        # Create a border using CSS
        st.markdown(
            """
            <div style="
                border: 1px solid #ddd;
                border-radius: 8px;
                padding: 16px;
                margin: 8px 0;
                background-color: #fafafa;
            ">
            """,
            unsafe_allow_html=True,
        )

        # Item header with controls
        col1, col2, col3, col4 = st.columns([3, 1, 1, 1])

        with col1:
            st.write(f"**{item.get('title', 'Untitled Item')}**")

        with col2:
            if st.button(
                f"✏️", key=f"edit_item_{item.get('id', 'unknown')}", help="Edit item"
            ):
                st.session_state[f"editing_item_{item.get('id', 'unknown')}"] = True

        with col3:
            if st.button(
                f"🔄",
                key=f"regen_item_{item.get('id', 'unknown')}",
                help="Regenerate item",
            ):
                st.session_state[f"regenerate_item_{item.get('id', 'unknown')}"] = True
                st.rerun()

        with col4:
            # Feedback buttons
            feedback_col1, feedback_col2 = st.columns(2)
            with feedback_col1:
                if st.button(
                    f"👍", key=f"like_item_{item.get('id', 'unknown')}", help="Good"
                ):
                    item["feedback"] = "positive"
                    state_manager.update_item_feedback(
                        section, subsection, item, "positive"
                    )
                    st.success("Feedback recorded!")

            with feedback_col2:
                if st.button(
                    f"👎",
                    key=f"dislike_item_{item.get('id', 'unknown')}",
                    help="Needs improvement",
                ):
                    item["feedback"] = "negative"
                    state_manager.update_item_feedback(
                        section, subsection, item, "negative"
                    )
                    st.warning("Feedback recorded. This item will be improved.")

        # Show current feedback if any
        if item.get("feedback"):
            if item["feedback"] == "positive":
                st.success("✅ Marked as good")
            elif item["feedback"] == "negative":
                st.error("❌ Marked for improvement")

        # Check if we're in editing mode for this item
        if st.session_state.get(f"editing_item_{item.get('id', 'unknown')}", False):
            # Show editing interface
            new_title = st.text_input(
                "Item Title",
                value=item.get("title", ""),
                key=f"item_title_input_{item.get('id', 'unknown')}",
            )
            new_content = st.text_area(
                "Item Content",
                value=item.get("content", ""),
                key=f"item_content_input_{item.get('id', 'unknown')}",
                height=100,
            )

            # Save/Cancel buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button("💾 Save", key=f"save_item_{item.get('id', 'unknown')}"):
                    item["title"] = new_title
                    item["content"] = new_content
                    state_manager.update_item(section, subsection, item)
                    st.session_state[f"editing_item_{item.get('id', 'unknown')}"] = (
                        False
                    )
                    st.success("Item updated!")
                    st.rerun()

            with col2:
                if st.button(
                    "❌ Cancel", key=f"cancel_item_{item.get('id', 'unknown')}"
                ):
                    st.session_state[f"editing_item_{item.get('id', 'unknown')}"] = (
                        False
                    )
                    st.rerun()

        else:
            # Display mode
            st.write(item.get("content", "*No content*"))

        # Close the card container
        st.markdown("</div>", unsafe_allow_html=True)


def load_template():
    """Load the CV template from file."""
    # Check if the template file exists, if not, create it from the default template
    if not os.path.exists(TEMPLATE_FILE_PATH):
        default_template = """**Anas AKHOMACH** | 📞 (+212) 600310536 | 📧 [anasakhomach205@gmail.com](mailto:anasakhomach205@gmail.com) | 🔗 [LinkedIn](https://www.linkedin.com/in/anas-akhomach/) | 💻 [GitHub](https://github.com/AnasAkhomach)
---

## Executive Summary

Dynamic and results-driven Software Engineer with 3+ years of experience in full-stack development, specializing in Python, JavaScript, and cloud technologies. Proven track record of delivering scalable web applications and leading cross-functional teams. Passionate about leveraging cutting-edge technologies to solve complex business problems and drive innovation.

## Key Qualifications

* **Programming Languages:** Python, JavaScript, TypeScript, Java, C++
* **Web Technologies:** React, Node.js, Express.js, Django, Flask, HTML5, CSS3
* **Databases:** PostgreSQL, MongoDB, Redis, MySQL
* **Cloud & DevOps:** AWS (EC2, S3, Lambda, RDS), Docker, Kubernetes, CI/CD pipelines
* **Tools & Frameworks:** Git, Jenkins, Terraform, Microservices Architecture

## Professional Experience

### Senior Software Engineer | TechCorp Solutions | 2022 - Present

* Led development of a microservices-based e-commerce platform serving 100K+ users, resulting in 40% improvement in system performance
* Architected and implemented RESTful APIs using Python/Django and Node.js, reducing response times by 35%
* Mentored junior developers and established code review processes, improving team productivity by 25%
* Collaborated with product managers and designers to deliver features ahead of schedule

### Software Engineer | InnovateTech | 2021 - 2022

* Developed responsive web applications using React and TypeScript, enhancing user experience for 50K+ monthly active users
* Implemented automated testing strategies, reducing bug reports by 60%
* Optimized database queries and implemented caching strategies, improving application load times by 45%
* Participated in agile development processes and sprint planning

### Junior Software Developer | StartupXYZ | 2020 - 2021

* Built and maintained web applications using Python/Flask and JavaScript
* Contributed to the development of a customer management system used by 200+ clients
* Assisted in database design and optimization
* Gained experience in version control, testing, and deployment processes

## Education

### Master of Science in Computer Science | University of Technology | 2020
* **Relevant Coursework:** Advanced Algorithms, Software Engineering, Database Systems, Machine Learning
* **Thesis:** "Scalable Web Application Architecture for High-Traffic Systems"
* **GPA:** 3.8/4.0

### Bachelor of Science in Software Engineering | Tech University | 2018
* **Relevant Coursework:** Data Structures, Object-Oriented Programming, Web Development, Computer Networks
* **Capstone Project:** E-commerce platform with real-time inventory management
* **GPA:** 3.7/4.0

## Projects

### AI-Powered Task Management System | 2023
* Developed a full-stack application using React, Node.js, and OpenAI API
* Implemented machine learning algorithms for task prioritization and scheduling
* Deployed on AWS with auto-scaling capabilities
* **Technologies:** React, Node.js, MongoDB, OpenAI API, AWS

### Real-time Chat Application | 2022
* Built a scalable chat application supporting 1000+ concurrent users
* Implemented WebSocket connections and real-time messaging
* Used Redis for session management and message caching
* **Technologies:** Socket.io, Express.js, Redis, PostgreSQL

## Certifications

* AWS Certified Solutions Architect - Associate (2023)
* Google Cloud Professional Developer (2022)
* MongoDB Certified Developer (2021)

## Languages

* Arabic (native) | English (B2) | German (B2) | French (B2) | Spanish (B1)
"""
        # Create the template file
        os.makedirs(os.path.dirname(TEMPLATE_FILE_PATH), exist_ok=True)
        with open(TEMPLATE_FILE_PATH, "w", encoding="utf-8") as f:
            f.write(default_template)

    # Load the template
    with open(TEMPLATE_FILE_PATH, "r", encoding="utf-8") as f:
        return f.read()


def show_template_info():
    """Show information about the CV template."""
    with st.expander("📋 CV Template Information", expanded=False):
        st.subheader("CV Template Settings")
        st.info(
            """Your CV template is being used as the base.

        Dynamic sections (will be tailored to match the job):
        - Executive Summary
        - Key Qualifications
        - Professional Experience (bullets)
        - Projects (descriptions)

        Static sections (will remain unchanged):
        - Contact Information
        - Education
        - Certifications
        - Languages
        """
        )


def initialize_session_state():
    """Initialize session state variables."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    if "state_manager" not in st.session_state:
        st.session_state.state_manager = StateManager()

    if "orchestrator_config" not in st.session_state:
        st.session_state.orchestrator_config = None

    if "job_description" not in st.session_state:
        st.session_state.job_description = ""

    if "cv_content" not in st.session_state:
        st.session_state.cv_content = ""

    if "processing" not in st.session_state:
        st.session_state.processing = False

    if "current_step" not in st.session_state:
        st.session_state.current_step = "input"

    if "progress" not in st.session_state:
        st.session_state.progress = 0

    if "status_message" not in st.session_state:
        st.session_state.status_message = ""

    # Initialize safety features
    if "stop_processing" not in st.session_state:
        st.session_state.stop_processing = False

    if "token_usage" not in st.session_state:
        st.session_state.token_usage = {
            "session_tokens": 0,
            "daily_tokens": 0,
            "session_requests": 0,
            "daily_requests": 0,
        }

    if "session_token_limit" not in st.session_state:
        st.session_state.session_token_limit = 50000  # Default session limit

    if "daily_token_limit" not in st.session_state:
        st.session_state.daily_token_limit = 200000  # Default daily limit

    # Initialize user API key
    if "user_gemini_api_key" not in st.session_state:
        st.session_state.user_gemini_api_key = ""

    if "api_key_validated" not in st.session_state:
        st.session_state.api_key_validated = False


def save_session(state_manager):
    """Save the current session to disk."""
    try:
        session_dir = os.path.join(SESSIONS_DIR, st.session_state.session_id)
        os.makedirs(session_dir, exist_ok=True)

        # Save session data
        session_data = {
            "session_id": st.session_state.session_id,
            "job_description": st.session_state.job_description,
            "cv_content": st.session_state.cv_content,
            "current_step": st.session_state.current_step,
            "progress": st.session_state.progress,
            "status_message": st.session_state.status_message,
            "timestamp": datetime.now().isoformat(),
            "token_usage": st.session_state.token_usage,
            "user_gemini_api_key": st.session_state.user_gemini_api_key,
            "api_key_validated": st.session_state.api_key_validated,
        }

        session_file = os.path.join(session_dir, "session.json")
        with open(session_file, "w", encoding="utf-8") as f:
            json.dump(session_data, f, indent=2, ensure_ascii=False)

        # Save state manager data
        state_manager.save_session(session_dir)

        return True
    except Exception as e:
        logger.error(f"Error saving session: {e}")
        return False


def load_session(session_id):
    """Load a session from disk."""
    try:
        session_dir = os.path.join(SESSIONS_DIR, session_id)
        session_file = os.path.join(session_dir, "session.json")

        if not os.path.exists(session_file):
            return False

        with open(session_file, "r", encoding="utf-8") as f:
            session_data = json.load(f)

        # Restore session state
        st.session_state.session_id = session_data["session_id"]
        st.session_state.job_description = session_data.get("job_description", "")
        st.session_state.cv_content = session_data.get("cv_content", "")
        st.session_state.current_step = session_data.get("current_step", "input")
        st.session_state.progress = session_data.get("progress", 0)
        st.session_state.status_message = session_data.get("status_message", "")
        st.session_state.token_usage = session_data.get(
            "token_usage",
            {
                "session_tokens": 0,
                "daily_tokens": 0,
                "session_requests": 0,
                "daily_requests": 0,
            },
        )
        st.session_state.user_gemini_api_key = session_data.get(
            "user_gemini_api_key", ""
        )
        st.session_state.api_key_validated = session_data.get(
            "api_key_validated", False
        )

        # Load state manager
        state_manager = StateManager()
        state_manager.load_session(session_dir)
        st.session_state.state_manager = state_manager

        return True
    except Exception as e:
        logger.error(f"Error loading session: {e}")
        return False


def get_available_sessions():
    """Get list of available sessions."""
    sessions = []
    if os.path.exists(SESSIONS_DIR):
        for session_id in os.listdir(SESSIONS_DIR):
            session_dir = os.path.join(SESSIONS_DIR, session_id)
            session_file = os.path.join(session_dir, "session.json")
            if os.path.exists(session_file):
                try:
                    with open(session_file, "r", encoding="utf-8") as f:
                        session_data = json.load(f)
                    sessions.append(
                        {
                            "id": session_id,
                            "timestamp": session_data.get("timestamp", ""),
                            "step": session_data.get("current_step", "unknown"),
                        }
                    )
                except Exception as e:
                    logger.error(f"Error reading session {session_id}: {e}")
    return sessions


def main():
    """Main Streamlit application."""

    # Initialize session state
    initialize_session_state()

    # Sidebar for session management and settings
    with st.sidebar:
        st.title("🔧 Session Management")

        # API Key Configuration Section
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
            st.session_state.api_key_validated = (
                False  # Reset validation when key changes
            )

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

        st.divider()

        # Safety Controls Section
        st.subheader("🛡️ Safety Controls")

        # Manual Stop Button
        if st.session_state.processing:
            if st.button(
                "🛑 STOP PROCESSING", type="primary", use_container_width=True
            ):
                st.session_state.stop_processing = True
                st.session_state.processing = False
                st.warning("Processing stopped by user")
                st.rerun()

        # Token Usage Display
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

        # Budget Limits Configuration
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

        # Warning indicators
        if session_usage_pct > 80:
            st.error("⚠️ Session token limit nearly reached!")
        elif session_usage_pct > 60:
            st.warning("⚠️ High session token usage")

        if daily_usage_pct > 80:
            st.error("⚠️ Daily token limit nearly reached!")
        elif daily_usage_pct > 60:
            st.warning("⚠️ High daily token usage")

        st.divider()

        # Session Management
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
            selected_session = st.selectbox(
                "Select session", session_options, index=None
            )

            if selected_session and st.button(
                "📂 Load Session", use_container_width=True
            ):
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

    # Main content area
    st.title("🤖 AI CV Generator")
    st.markdown("*Tailor your CV to any job description using AI*")

    # Show template information
    show_template_info()

    # Progress indicator
    if st.session_state.processing:
        st.info(f"🔄 {st.session_state.status_message}")
        progress_bar = st.progress(st.session_state.progress / 100)

        # Check for stop signal
        if st.session_state.stop_processing:
            st.warning("⏹️ Processing interrupted by user")
            st.session_state.processing = False
            st.session_state.stop_processing = False

    # Main tabs
    tab1, tab2, tab3 = st.tabs(["📝 Input & Generate", "✏️ Review & Edit", "📄 Export"])

    with tab1:
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
                        # Start processing
                        st.session_state.processing = True
                        st.session_state.stop_processing = False
                        st.session_state.current_step = "processing"
                        st.session_state.progress = 0
                        st.session_state.status_message = "Initializing AI agents..."

                        try:
                            # Check if user has provided API key
                            if not st.session_state.user_gemini_api_key:
                                st.error(
                                    "⚠️ Please enter your Gemini API key in the sidebar before starting."
                                )
                                st.session_state.processing = False
                                return

                            # Initialize enhanced CV integration if not already done
                            if st.session_state.orchestrator_config is None:
                                config = EnhancedCVConfig(
                                    mode=IntegrationMode.PRODUCTION,
                                    api_key=st.session_state.user_gemini_api_key,
                                )
                                # Store only serializable config to avoid JSON serialization errors
                                st.session_state.orchestrator_config = config.to_dict()
                                st.session_state.api_key_validated = True

                            # Get enhanced content writer agent from integration
                            # Reconstruct the object from stored config
                            config = EnhancedCVConfig.from_dict(
                                st.session_state.orchestrator_config
                            )
                            enhanced_cv_integration = get_enhanced_cv_integration(
                                config
                            )
                            enhanced_content_writer = enhanced_cv_integration.get_agent(
                                "enhanced_content_writer"
                            )

                            # Budget checking function
                            def check_budget_limits():
                                """Check if budget limits are exceeded and stop processing if needed."""
                                if (
                                    "session_token_limit" in st.session_state
                                    and "daily_token_limit" in st.session_state
                                ):
                                    session_usage_pct = (
                                        st.session_state.token_usage["session_tokens"]
                                        / st.session_state.session_token_limit
                                    ) * 100
                                    daily_usage_pct = (
                                        st.session_state.token_usage["daily_tokens"]
                                        / st.session_state.daily_token_limit
                                    ) * 100

                                    if (
                                        session_usage_pct >= 100
                                        or daily_usage_pct >= 100
                                    ):
                                        st.session_state.stop_processing = True
                                        st.session_state.processing = False
                                        if session_usage_pct >= 100:
                                            st.error(
                                                "❌ Session token limit exceeded! Processing stopped."
                                            )
                                        if daily_usage_pct >= 100:
                                            st.error(
                                                "❌ Daily token limit exceeded! Processing stopped."
                                            )
                                        return True
                                return False

                            # Token usage tracking function
                            def update_token_usage(tokens_used, requests_made=1):
                                """Update token usage counters."""
                                st.session_state.token_usage[
                                    "session_tokens"
                                ] += tokens_used
                                st.session_state.token_usage[
                                    "daily_tokens"
                                ] += tokens_used
                                st.session_state.token_usage[
                                    "session_requests"
                                ] += requests_made
                                st.session_state.token_usage[
                                    "daily_requests"
                                ] += requests_made

                            # Check budget before starting
                            if check_budget_limits():
                                st.rerun()
                                return

                            # Check for stop signal
                            if st.session_state.stop_processing:
                                st.warning("⏹️ Processing stopped by user")
                                st.session_state.processing = False
                                st.rerun()
                                return

                            # Update progress
                            st.session_state.progress = 20
                            st.session_state.status_message = (
                                "Analyzing job description and CV..."
                            )

                            # Estimate token usage for this operation (rough estimate)
                            estimated_tokens = (
                                len(job_description.split()) * 1.3
                                + len(cv_content.split()) * 1.3
                                + 2000
                            )  # Rough estimate

                            # Process with enhanced CV integration
                            # Parse CV content into user data format
                            user_data = {
                                "personal_info": {},  # Add personal_info key
                                "experience": [cv_content],  # Use 'experience' not 'experiences'
                                "experiences": [cv_content],  # Keep for backward compatibility
                                "qualifications": [],
                                "projects": [],
                                "education": [],
                            }

                            # Start enhanced CV generation workflow
                            import asyncio

                            async def generate_cv():
                                try:
                                    # Use the enhanced CV integration's generate_job_tailored_cv method
                                    result = await enhanced_cv_integration.generate_job_tailored_cv(
                                        personal_info=user_data.get('personal_info', {}),
                                        experience=user_data.get('experience', []),
                                        job_description=job_description
                                    )
                                    
                                    # Apply enum_to_value to the entire result to handle IntegrationMode objects
                                    logger.info(f"Raw result type: {type(result)}")
                                    if result:
                                        logger.info(f"Raw result keys: {result.keys() if isinstance(result, dict) else 'Not a dict'}")
                                        
                                        # Clean the result of any enum objects
                                        cleaned_result = enum_to_value(result)
                                        logger.info(f"Cleaned result type: {type(cleaned_result)}")
                                        
                                        # Test serialization
                                        try:
                                            import json
                                            json.dumps(cleaned_result, cls=EnumEncoder)
                                            logger.info("Result serialization test passed")
                                        except Exception as serialize_error:
                                            logger.error(f"Result serialization test failed: {serialize_error}")
                                            logger.error(f"Problematic result structure: {cleaned_result}")
                                            raise serialize_error
                                        
                                        return cleaned_result
                                    return result
                                except Exception as workflow_error:
                                    logger.error(f"Error in workflow execution: {workflow_error}")
                                    logger.error(f"Workflow error type: {type(workflow_error)}")
                                    import traceback
                                    logger.error(f"Workflow traceback: {traceback.format_exc()}")
                                    raise workflow_error

                            # Run the async workflow
                            try:
                                loop = asyncio.new_event_loop()
                                asyncio.set_event_loop(loop)
                                result = loop.run_until_complete(generate_cv())
                                # The result is already in the correct format from generate_job_tailored_cv
                            except Exception as e:
                                logger.error(f"Error in CV generation loop: {str(e)}")
                                logger.error(f"Loop error type: {type(e)}")
                                if "IntegrationMode" in str(e):
                                    logger.error("IntegrationMode serialization error detected in workflow execution")
                                st.error(f"Error in CV generation: {str(e)}")
                                result = None
                            finally:
                                loop.close()

                            # Update token usage (using estimated tokens for now)
                            update_token_usage(int(estimated_tokens))

                            # Check budget after processing
                            if check_budget_limits():
                                st.rerun()
                                return

                            # Check for stop signal
                            if st.session_state.stop_processing:
                                st.warning("⏹️ Processing stopped by user")
                                st.session_state.processing = False
                                st.rerun()
                                return

                            # Update progress
                            st.session_state.progress = 80
                            st.session_state.status_message = (
                                "Finalizing tailored CV..."
                            )

                            # Store the result in state manager
                            if result:
                                try:
                                    # First, clean the entire result object to remove any IntegrationMode enums
                                    logger.info(f"Raw result type: {type(result)}")
                                    logger.info(f"Raw result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
                                    
                                    # Apply enum_to_value to the entire result first
                                    cleaned_result = enum_to_value(result)
                                    
                                    # Test if the cleaned result is serializable
                                    try:
                                        import json
                                        json.dumps(cleaned_result, cls=EnumEncoder)
                                        logger.info("Result successfully cleaned and serializable")
                                    except Exception as serialize_error:
                                        import json
                                        logger.error(f"Result still not serializable after cleaning: {serialize_error}")
                                        # Force convert any remaining problematic objects to strings
                                        cleaned_result = json.loads(json.dumps(cleaned_result, cls=EnumEncoder, default=str))
                                    
                                    if cleaned_result and cleaned_result.get("success") and cleaned_result.get("results"):
                                        # Debug: Log the structure of results
                                        logger.info(f"Results array length: {len(cleaned_result['results'])}")
                                        for i, task_result in enumerate(cleaned_result["results"]):
                                            logger.info(f"Task result {i}: keys={list(task_result.keys()) if isinstance(task_result, dict) else 'Not a dict'}")
                                            if isinstance(task_result, dict):
                                                logger.info(f"Task result {i} content keys: {list(task_result.get('content', {}).keys()) if task_result.get('content') else 'No content key'}")
                                        
                                        # Use ContentAggregator to properly aggregate individual content pieces
                                        from .content_aggregator import ContentAggregator
                                        
                                        # Add detailed logging for each task result
                                        for i, task_result in enumerate(cleaned_result["results"]):
                                            agent_type = task_result.get("agent_type", "unknown")
                                            content_keys = list(task_result.get("content", {}).keys()) if task_result.get("content") else []
                                            logger.info(f"Task {i} ({agent_type}): content_keys={content_keys}")
                                        
                                        # Create aggregator and process all task results
                                        aggregator = ContentAggregator()
                                        cv_content = aggregator.aggregate_results(cleaned_result["results"])
                                        
                                        # Validate aggregated content
                                        if cv_content and aggregator.validate_content_data(cv_content):
                                            logger.info(f"Content aggregation successful. Populated fields: {[k for k, v in cv_content.items() if v]}")
                                            
                                            # Ensure content is serializable
                                            serializable_content = json.loads(json.dumps(cv_content, cls=EnumEncoder, default=str))
                                            st.session_state.state_manager.update_cv_data(serializable_content)
                                        else:
                                            logger.error("Content aggregation failed or no valid content found")
                                            st.error("Failed to aggregate CV content from agent results")
                                            st.session_state.processing = False
                                            return
                                    else:
                                        logger.error(f"Result structure invalid: success={cleaned_result.get('success') if isinstance(cleaned_result, dict) else 'N/A'}")
                                        st.error("Invalid result structure from CV generation")
                                        st.session_state.processing = False
                                        return
                                        
                                except Exception as e:
                                    logger.error(f"Error processing CV results: {e}")
                                    logger.error(f"Error type: {type(e)}")
                                    import traceback
                                    logger.error(f"Full traceback: {traceback.format_exc()}")
                                    
                                    # Try to log the problematic result structure
                                    try:
                                        logger.error(f"Result structure: {json.dumps(result, cls=EnumEncoder, default=str, indent=2)}")
                                    except:
                                        logger.error(f"Could not serialize result for logging: {type(result)}")
                                    
                                    st.error(f"An error occurred during CV processing: {str(e)}")
                                    st.session_state.processing = False
                                    return
                                st.session_state.current_step = "review"
                                st.session_state.progress = 100
                                st.session_state.status_message = "CV generation completed!"
                                st.session_state.processing = False
                                st.success("✅ CV tailored successfully!")
                            else:
                                st.error("Failed to generate tailored CV")
                                st.session_state.processing = False

                            # Small delay to show completion
                            time.sleep(1)
                            st.rerun()

                        except Exception as e:
                            logger.error(f"Error during CV generation: {e}")
                            st.error(f"An error occurred: {str(e)}")
                            st.session_state.processing = False
                            st.rerun()

    with tab2:
        st.header("Review & Edit Your CV")

        if st.session_state.state_manager.cv_data:
            # Get current CV data
            cv_data = st.session_state.state_manager.cv_data

            # Display sections
            if "sections" in cv_data:
                for section in cv_data["sections"]:
                    display_section(section, st.session_state.state_manager)
            else:
                st.info("No CV sections available. Please generate a CV first.")

            # Manual tailoring section
            st.subheader("🎯 Manual Tailoring")
            st.write(
                "Need to make specific adjustments? Describe what you'd like to change:"
            )

            tailoring_request = st.text_area(
                "Tailoring Instructions",
                placeholder="e.g., 'Emphasize my Python experience more', 'Add more details about my leadership skills', 'Focus on cloud technologies'",
                height=100,
            )

            if st.button("🔧 Apply Tailoring", disabled=st.session_state.processing):
                if tailoring_request.strip():
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
                        st.session_state.processing = True
                        st.session_state.stop_processing = False
                        st.session_state.status_message = "Applying manual tailoring..."

                        try:
                            # Get enhanced CV integration for tailoring
                            if "enhanced_cv_integration_config" not in st.session_state:
                                config = EnhancedCVConfig(
                                    mode=IntegrationMode.STREAMLIT,
                                    enable_caching=True,
                                    enable_monitoring=True,
                                )
                                # Store only serializable config to avoid JSON serialization errors
                                st.session_state.enhanced_cv_integration_config = (
                                    config.to_dict()
                                )

                            # Reconstruct the object from stored config
                            config = EnhancedCVConfig.from_dict(
                                st.session_state.enhanced_cv_integration_config
                            )
                            enhanced_cv_integration = get_enhanced_cv_integration(
                                config
                            )
                            enhanced_content_writer = (
                                enhanced_cv_integration.get_content_writer()
                            )

                            # Budget checking function
                            def check_budget_limits():
                                """Check if budget limits are exceeded and stop processing if needed."""
                                if (
                                    "session_token_limit" in st.session_state
                                    and "daily_token_limit" in st.session_state
                                ):
                                    session_usage_pct = (
                                        st.session_state.token_usage["session_tokens"]
                                        / st.session_state.session_token_limit
                                    ) * 100
                                    daily_usage_pct = (
                                        st.session_state.token_usage["daily_tokens"]
                                        / st.session_state.daily_token_limit
                                    ) * 100

                                    if (
                                        session_usage_pct >= 100
                                        or daily_usage_pct >= 100
                                    ):
                                        st.session_state.stop_processing = True
                                        st.session_state.processing = False
                                        if session_usage_pct >= 100:
                                            st.error(
                                                "❌ Session token limit exceeded! Processing stopped."
                                            )
                                        if daily_usage_pct >= 100:
                                            st.error(
                                                "❌ Daily token limit exceeded! Processing stopped."
                                            )
                                        return True
                                return False

                            # Token usage tracking function
                            def update_token_usage(tokens_used, requests_made=1):
                                """Update token usage counters."""
                                st.session_state.token_usage[
                                    "session_tokens"
                                ] += tokens_used
                                st.session_state.token_usage[
                                    "daily_tokens"
                                ] += tokens_used
                                st.session_state.token_usage[
                                    "session_requests"
                                ] += requests_made
                                st.session_state.token_usage[
                                    "daily_requests"
                                ] += requests_made

                            # Check budget before starting
                            if check_budget_limits():
                                st.rerun()
                                return

                            # Check for stop signal
                            if st.session_state.stop_processing:
                                st.warning("⏹️ Processing stopped by user")
                                st.session_state.processing = False
                                st.rerun()
                                return

                            # Estimate token usage for this operation
                            estimated_tokens = (
                                len(tailoring_request.split()) * 1.3 + 1500
                            )  # Rough estimate

                            # Apply tailoring using enhanced CV integration
                            # Parse current CV data into user data format
                            user_data = {
                                "personal_info": {},  # Add personal_info key
                                "experience": [str(cv_data)],  # Use 'experience' not 'experiences'
                                "experiences": [str(cv_data)],  # Keep for backward compatibility
                                "qualifications": [],
                                "projects": [],
                                "education": [],
                            }

                            # Apply tailoring with enhanced workflow
                            import asyncio

                            async def apply_tailoring():
                                # Create a tailored job description that includes the tailoring request
                                tailored_job_desc = f"{st.session_state.job_description}\n\nAdditional Requirements: {tailoring_request}"
                                # Use the enhanced CV integration's generate_job_tailored_cv method
                                result = await enhanced_cv_integration.generate_job_tailored_cv(
                                    personal_info=user_data.get('personal_info', {}),
                                    experience=user_data.get('experience', []),
                                    job_description=tailored_job_desc
                                )
                                return result

                            # Run the async tailoring workflow
                            try:
                                loop = asyncio.new_event_loop()
                                asyncio.set_event_loop(loop)
                                result = loop.run_until_complete(apply_tailoring())
                                # The result is already in the correct format from generate_job_tailored_cv
                            except Exception as e:
                                st.error(f"Error in CV tailoring: {str(e)}")
                                result = None
                            finally:
                                loop.close()

                            # Update token usage
                            update_token_usage(int(estimated_tokens))

                            # Check budget after processing
                            if check_budget_limits():
                                st.rerun()
                                return

                            # Check for stop signal
                            if st.session_state.stop_processing:
                                st.warning("⏹️ Processing stopped by user")
                                st.session_state.processing = False
                                st.rerun()
                                return

                            if result and "content" in result:
                                # Ensure the content is JSON serializable before storing
                                import json

                                try:
                                    # Recursively convert all enums to their value before serialization
                                    cleaned_content = enum_to_value(result["content"])
                                    serializable_content = json.loads(json.dumps(cleaned_content))
                                    st.session_state.state_manager.update_cv_data(
                                        serializable_content
                                    )
                                except Exception as e:
                                    raise e
                                st.success("✅ Tailoring applied successfully!")
                            else:
                                st.error("Failed to apply tailoring")

                            st.session_state.processing = False
                            st.rerun()

                        except Exception as e:
                            logger.error(f"Error during manual tailoring: {e}")
                            st.error(f"An error occurred: {str(e)}")
                            st.session_state.processing = False
                            st.rerun()
                else:
                    st.warning("Please provide tailoring instructions")
        else:
            st.info(
                "No CV data available. Please generate a CV first in the 'Input & Generate' tab."
            )

    with tab3:
        st.header("Export Your CV")

        if st.session_state.state_manager.cv_data:
            cv_data = st.session_state.state_manager.cv_data

            # Preview section
            st.subheader("📋 Preview")

            # Get content data for rendering
            content_data = cv_data.get("content", {})

            if content_data:
                # Generate rendered Markdown (placeholder for real rendering)
                rendered_cv = f"""
# {content_data.get('name', 'Your Name')}

{content_data.get('contact_info', 'Contact information')}

---

## Executive Summary

{content_data.get('executive_summary', 'Executive summary content')}

## Key Qualifications

{content_data.get('key_qualifications', 'Key qualifications content')}

## Professional Experience
"""
                # Add experience bullets
                for bullet in content_data.get("experience_bullets", []):
                    rendered_cv += f"\n* {bullet}"

                rendered_cv += "\n\n## Projects\n"
                for project in content_data.get("projects", []):
                    rendered_cv += f"\n### {project.get('title', 'Project Title')}\n{project.get('description', 'Project description')}"

                # Show preview
                st.markdown(rendered_cv)

                # Export options
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

            else:
                # Create a simple preview of the current content
                preview = f"""
# {content_data.get('name', 'Your Name')}

{content_data.get('contact_info', 'Contact information will appear here')}

---

## Experience
"""
                for bullet in content_data.get("experience_bullets", [])[
                    :3
                ]:  # Show first 3 bullets
                    preview += f"\n* {bullet}"

                if len(content_data.get("experience_bullets", [])) > 3:
                    preview += "\n* ..."

                st.markdown(preview)
                st.info(
                    "Complete your CV in the 'Review & Edit' tab to see the full preview."
                )

        else:
            st.info(
                "No CV data available. Please generate a CV first in the 'Input & Generate' tab."
            )


# This module can be imported and executed by app.py launcher or run directly
if __name__ == "__main__":
    main()
