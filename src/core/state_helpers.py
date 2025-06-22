"""State Management Helpers for the AI CV Generator.

This module contains functions for managing Streamlit session state,
initializing state variables, and handling state transitions.
"""

import streamlit as st
import uuid
from typing import Dict, Any, Optional
from ..config.logging_config import setup_logging
from ..core.state_manager import StateManager
from ..orchestration.state import AgentState
from ..models.data_models import JobDescriptionData, StructuredCV

# Initialize logging
logger = setup_logging()


def initialize_session_state() -> None:
    """Initialize all session state variables with default values."""
    # Core session management
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    if "state_manager" not in st.session_state:
        st.session_state.state_manager = StateManager(
            session_id=st.session_state.session_id
        )

    # API Configuration
    if "user_gemini_api_key" not in st.session_state:
        st.session_state.user_gemini_api_key = ""

    if "api_key_validated" not in st.session_state:
        st.session_state.api_key_validated = False

    # Processing state
    if "processing" not in st.session_state:
        st.session_state.processing = False

    if "stop_processing" not in st.session_state:
        st.session_state.stop_processing = False

    if "current_step" not in st.session_state:
        st.session_state.current_step = "input"

    if "progress" not in st.session_state:
        st.session_state.progress = 0

    if "status_message" not in st.session_state:
        st.session_state.status_message = "Ready to start"

    # Input data
    if "job_description" not in st.session_state:
        st.session_state.job_description = ""

    if "cv_content" not in st.session_state:
        st.session_state.cv_content = ""

    if "cv_text" not in st.session_state:
        st.session_state.cv_text = ""

    if "start_from_scratch" not in st.session_state:
        st.session_state.start_from_scratch = False

    # Token usage and budget management
    if "token_usage" not in st.session_state:
        st.session_state.token_usage = {
            "session_tokens": 0,
            "daily_tokens": 0,
            "total_tokens": 0,
        }

    if "session_token_limit" not in st.session_state:
        st.session_state.session_token_limit = 50000

    if "daily_token_limit" not in st.session_state:
        st.session_state.daily_token_limit = 200000

    # Orchestrator configuration
    if "orchestrator_config" not in st.session_state:
        st.session_state.orchestrator_config = None

    # Enhanced CV integration configuration
    if "enhanced_cv_integration_config" not in st.session_state:
        st.session_state.enhanced_cv_integration_config = None

    # Workflow state
    if "workflow_state" not in st.session_state:
        st.session_state.workflow_state = None

    # Error handling
    if "error_messages" not in st.session_state:
        st.session_state.error_messages = []

    # UI state management
    _initialize_ui_state()


def _initialize_ui_state() -> None:
    """Initialize UI-specific session state variables."""
    # Tab state
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = "input"

    # Editing states for sections and items
    if "editing_states" not in st.session_state:
        st.session_state.editing_states = {}

    # Processing states for individual items
    if "item_processing_states" not in st.session_state:
        st.session_state.item_processing_states = {}

    # Regeneration flags
    if "regeneration_flags" not in st.session_state:
        st.session_state.regeneration_flags = {}


def create_initial_agent_state() -> AgentState:
    """Create an AgentState object from current UI session state.

    This function acts as the single point of conversion from UI state to AgentState,
    establishing AgentState as the source of truth for workflow execution.
    Uses only raw input data stored by the frontend components.

    Returns:
        AgentState: A fully initialized AgentState object ready for the workflow.
    """
    # 1. Initialize JobDescriptionData from UI
    job_desc_raw = st.session_state.get("job_description_input", "")
    job_description_data = JobDescriptionData(raw_text=job_desc_raw)

    # 2. Initialize StructuredCV based on user choice
    cv_text = st.session_state.get("cv_text_input", "")
    start_from_scratch = st.session_state.get("start_from_scratch_input", False)

    # The ParserAgent will later populate the full StructuredCV from cv_text.
    # For initialization, we only need the raw text and metadata.
    structured_cv = StructuredCV(
        metadata={"original_cv_text": cv_text, "start_from_scratch": start_from_scratch}
    )

    # 3. Construct the final AgentState
    initial_state = AgentState(
        structured_cv=structured_cv,
        job_description_data=job_description_data,
        cv_text=cv_text,  # <-- Ensure cv_text is set for parser validation
        # Initialize other fields with sensible defaults
        user_feedback=None,
        error_messages=[],
        # The workflow itself will manage these control fields
        current_section_key=None,
        current_item_id=None,
        items_to_process_queue=[],
        is_initial_generation=True,
        final_output_path=None,
        research_findings=None,
    )

    return initial_state


def update_processing_state(
    processing: bool, step: str = None, progress: int = None, message: str = None
) -> None:
    """Update the processing state in session state.

    Args:
        processing: Whether processing is active
        step: Current processing step
        progress: Progress percentage (0-100)
        message: Status message to display
    """
    st.session_state.processing = processing

    if step is not None:
        st.session_state.current_step = step

    if progress is not None:
        st.session_state.progress = progress

    if message is not None:
        st.session_state.status_message = message

    logger.info(
        f"Processing state updated: {processing}, step: {step}, progress: {progress}%"
    )


def update_token_usage(session_tokens: int = 0, daily_tokens: int = 0) -> None:
    """Update token usage counters.

    Args:
        session_tokens: Tokens used in current session
        daily_tokens: Tokens used today
    """
    if "token_usage" not in st.session_state:
        st.session_state.token_usage = {
            "session_tokens": 0,
            "daily_tokens": 0,
            "total_tokens": 0,
        }

    st.session_state.token_usage["session_tokens"] += session_tokens
    st.session_state.token_usage["daily_tokens"] += daily_tokens
    st.session_state.token_usage["total_tokens"] += session_tokens

    logger.info(
        f"Token usage updated: +{session_tokens} session, +{daily_tokens} daily"
    )


def check_budget_limits() -> Dict[str, Any]:
    """Check if budget limits are exceeded.

    Returns:
        Dict containing budget status information
    """
    session_usage_pct = 0
    daily_usage_pct = 0

    if (
        "session_token_limit" in st.session_state
        and st.session_state.session_token_limit > 0
    ):
        session_usage_pct = (
            st.session_state.token_usage["session_tokens"]
            / st.session_state.session_token_limit
        ) * 100

    if (
        "daily_token_limit" in st.session_state
        and st.session_state.daily_token_limit > 0
    ):
        daily_usage_pct = (
            st.session_state.token_usage["daily_tokens"]
            / st.session_state.daily_token_limit
        ) * 100

    return {
        "session_usage_pct": session_usage_pct,
        "daily_usage_pct": daily_usage_pct,
        "session_exceeded": session_usage_pct >= 100,
        "daily_exceeded": daily_usage_pct >= 100,
        "session_warning": session_usage_pct > 80,
        "daily_warning": daily_usage_pct > 80,
    }


def set_editing_state(item_type: str, item_id: str, editing: bool) -> None:
    """Set editing state for a specific item.

    Args:
        item_type: Type of item ('section', 'subsection', 'item')
        item_id: Unique identifier for the item
        editing: Whether the item is being edited
    """
    key = f"editing_{item_type}_{item_id}"
    st.session_state[key] = editing

    # Also track in centralized editing states
    if "editing_states" not in st.session_state:
        st.session_state.editing_states = {}

    st.session_state.editing_states[key] = editing

    logger.debug(f"Editing state set: {key} = {editing}")


def get_editing_state(item_type: str, item_id: str) -> bool:
    """Get editing state for a specific item.

    Args:
        item_type: Type of item ('section', 'subsection', 'item')
        item_id: Unique identifier for the item

    Returns:
        bool: Whether the item is being edited
    """
    key = f"editing_{item_type}_{item_id}"
    return st.session_state.get(key, False)


def set_processing_state(item_id: str, processing: bool) -> None:
    """Set processing state for a specific item.

    Args:
        item_id: Unique identifier for the item
        processing: Whether the item is being processed
    """
    key = f"processing_item_{item_id}"
    st.session_state[key] = processing

    # Also track in centralized processing states
    if "item_processing_states" not in st.session_state:
        st.session_state.item_processing_states = {}

    st.session_state.item_processing_states[key] = processing

    logger.debug(f"Processing state set: {key} = {processing}")


def get_processing_state(item_id: str) -> bool:
    """Get processing state for a specific item.

    Args:
        item_id: Unique identifier for the item

    Returns:
        bool: Whether the item is being processed
    """
    key = f"processing_item_{item_id}"
    return st.session_state.get(key, False)


def set_regeneration_flag(item_type: str, item_id: str, regenerate: bool) -> None:
    """Set regeneration flag for a specific item.

    Args:
        item_type: Type of item ('section', 'subsection', 'item')
        item_id: Unique identifier for the item
        regenerate: Whether the item should be regenerated
    """
    key = f"regenerate_{item_type}_{item_id}"
    st.session_state[key] = regenerate

    # Also track in centralized regeneration flags
    if "regeneration_flags" not in st.session_state:
        st.session_state.regeneration_flags = {}

    st.session_state.regeneration_flags[key] = regenerate

    logger.debug(f"Regeneration flag set: {key} = {regenerate}")


def get_regeneration_flag(item_type: str, item_id: str) -> bool:
    """Get regeneration flag for a specific item.

    Args:
        item_type: Type of item ('section', 'subsection', 'item')
        item_id: Unique identifier for the item

    Returns:
        bool: Whether the item should be regenerated
    """
    key = f"regenerate_{item_type}_{item_id}"
    return st.session_state.get(key, False)


def clear_regeneration_flag(item_type: str, item_id: str) -> None:
    """Clear regeneration flag for a specific item.

    Args:
        item_type: Type of item ('section', 'subsection', 'item')
        item_id: Unique identifier for the item
    """
    key = f"regenerate_{item_type}_{item_id}"
    if key in st.session_state:
        del st.session_state[key]

    if (
        "regeneration_flags" in st.session_state
        and key in st.session_state.regeneration_flags
    ):
        del st.session_state.regeneration_flags[key]

    logger.debug(f"Regeneration flag cleared: {key}")


def add_error_message(message: str) -> None:
    """Add an error message to the session state.

    Args:
        message: Error message to add
    """
    if "error_messages" not in st.session_state:
        st.session_state.error_messages = []

    st.session_state.error_messages.append(message)
    logger.error(f"Error added to session: {message}")


def clear_error_messages() -> None:
    """Clear all error messages from session state."""
    st.session_state.error_messages = []
    logger.debug("Error messages cleared")


def get_error_messages() -> list:
    """Get all error messages from session state.

    Returns:
        List of error messages
    """
    return st.session_state.get("error_messages", [])


def reset_session_state(preserve_keys: Optional[list] = None) -> None:
    """Reset session state, optionally preserving specific keys.

    Args:
        preserve_keys: List of keys to preserve during reset
    """
    if preserve_keys is None:
        preserve_keys = ["session_token_limit", "daily_token_limit"]

    # Store values to preserve
    preserved_values = {}
    for key in preserve_keys:
        if key in st.session_state:
            preserved_values[key] = st.session_state[key]

    # Clear all session state
    for key in list(st.session_state.keys()):
        del st.session_state[key]

    # Restore preserved values
    for key, value in preserved_values.items():
        st.session_state[key] = value

    # Reinitialize session state
    initialize_session_state()

    logger.info(f"Session state reset, preserved keys: {list(preserved_values.keys())}")


def validate_session_state() -> Dict[str, Any]:
    """Validate the current session state and return validation results.

    Returns:
        Dict containing validation results
    """
    validation_results = {"valid": True, "errors": [], "warnings": []}

    # Check required fields
    required_fields = [
        "session_id",
        "user_gemini_api_key",
        "token_usage",
        "session_token_limit",
        "daily_token_limit",
    ]

    for field in required_fields:
        if field not in st.session_state:
            validation_results["errors"].append(f"Missing required field: {field}")
            validation_results["valid"] = False

    # Check API key
    if not st.session_state.get("user_gemini_api_key"):
        validation_results["warnings"].append("No API key provided")

    # Check budget limits
    budget_status = check_budget_limits()
    if budget_status["session_exceeded"]:
        validation_results["errors"].append("Session token limit exceeded")
        validation_results["valid"] = False
    elif budget_status["session_warning"]:
        validation_results["warnings"].append("Session token usage high")

    if budget_status["daily_exceeded"]:
        validation_results["errors"].append("Daily token limit exceeded")
        validation_results["valid"] = False
    elif budget_status["daily_warning"]:
        validation_results["warnings"].append("Daily token usage high")

    return validation_results
