"""State management module for the Streamlit application.

This module provides a centralized StateManager class that encapsulates all
session state logic, providing a clean, testable interface for state management.
"""

from typing import Any, Optional, Dict
import streamlit as st
from ..orchestration.state import AgentState
from ..config.logging_config import get_logger

logger = get_logger(__name__)


class StateManager:
    """Handles all session state logic for the Streamlit app.

    This class provides a clean abstraction over Streamlit's session_state,
    ensuring consistent state initialization and providing type-safe access
    to common state variables.
    """

    def __init__(self):
        """Initialize the StateManager and set up default session state."""
        self._initialize_state()

    def _initialize_state(self):
        """Initializes the session state with default values."""
        defaults = {
            # Core application state
            "agent_state": None,
            "user_gemini_api_key": "",
            # File processing state
            "uploaded_cv_file": None,
            "uploaded_job_file": None,
            "cv_text": None,
            "job_description_text": None,
            # Processing state
            "is_processing": False,
            "just_finished": False,
            "workflow_error": None,
            # UI state
            "current_tab": 0,
            # Session management
            "session_id": None,
        }

        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
                logger.debug("Initialized session state key: %s", key)

    def get(self, key: str, default: Any = None) -> Any:
        """Gets a value from the session state.

        Args:
            key: The session state key to retrieve
            default: Default value if key doesn't exist

        Returns:
            The value from session state or default
        """
        return st.session_state.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Sets a value in the session state.

        Args:
            key: The session state key to set
            value: The value to set
        """
        st.session_state[key] = value
        logger.debug("Set session state key: %s", key)

    def clear_key(self, key: str) -> None:
        """Removes a key from session state if it exists.

        Args:
            key: The session state key to remove
        """
        if key in st.session_state:
            del st.session_state[key]
            logger.debug("Cleared session state key: %s", key)

    def reset_processing_state(self) -> None:
        """Resets all processing-related state variables."""
        self.set("is_processing", False)
        self.set("just_finished", False)
        self.set("workflow_error", None)
        logger.debug("Reset processing state")

    # Properties for type-safe access to common state variables

    @property
    def agent_state(self) -> Optional[AgentState]:
        """Get the current AgentState."""
        return self.get("agent_state")

    @agent_state.setter
    def agent_state(self, state: Optional[AgentState]) -> None:
        """Set the AgentState."""
        self.set("agent_state", state)

    @property
    def user_gemini_api_key(self) -> str:
        """Get the user's Gemini API key."""
        return self.get("user_gemini_api_key", "")

    @user_gemini_api_key.setter
    def user_gemini_api_key(self, key: str) -> None:
        """Set the user's Gemini API key."""
        self.set("user_gemini_api_key", key)

    @property
    def cv_text(self) -> Optional[str]:
        """Get the processed CV text."""
        return self.get("cv_text")

    @cv_text.setter
    def cv_text(self, text: Optional[str]) -> None:
        """Set the processed CV text."""
        self.set("cv_text", text)

    @property
    def job_description_text(self) -> Optional[str]:
        """Get the processed job description text."""
        return self.get("job_description_text")

    @job_description_text.setter
    def job_description_text(self, text: Optional[str]) -> None:
        """Set the processed job description text."""
        self.set("job_description_text", text)

    @property
    def is_processing(self) -> bool:
        """Check if processing is currently in progress."""
        return self.get("is_processing", False)

    @is_processing.setter
    def is_processing(self, processing: bool) -> None:
        """Set the processing state."""
        self.set("is_processing", processing)

    @property
    def just_finished(self) -> bool:
        """Check if processing just finished."""
        return self.get("just_finished", False)

    @just_finished.setter
    def just_finished(self, finished: bool) -> None:
        """Set the just finished state."""
        self.set("just_finished", finished)

    @property
    def workflow_error(self) -> Optional[str]:
        """Get the current workflow error message."""
        return self.get("workflow_error")

    @workflow_error.setter
    def workflow_error(self, error: Optional[str]) -> None:
        """Set the workflow error message."""
        self.set("workflow_error", error)

    def has_required_data(self) -> bool:
        """Check if both CV and job description data are available."""
        return self.cv_text is not None and self.job_description_text is not None

    def get_state_summary(self) -> Dict[str, Any]:
        """Get a summary of current state for debugging/logging."""
        return {
            "has_agent_state": self.agent_state is not None,
            "has_cv_text": self.cv_text is not None,
            "has_job_description": self.job_description_text is not None,
            "is_processing": self.is_processing,
            "just_finished": self.just_finished,
            "has_workflow_error": self.workflow_error is not None,
            "session_id": self.get("session_id"),
        }
