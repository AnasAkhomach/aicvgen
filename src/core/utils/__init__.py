"""Utilities module for the aicvgen application."""

from src.error_handling.boundaries import StreamlitErrorBoundary
from src.error_handling.exceptions import (
    AicvgenError,
    ConfigurationError,
    StateManagerError,
)

from .decorators import create_async_sync_decorator
from .import_fallbacks import get_dotenv
from .latex_utils import escape_latex, recursively_escape_latex
from .node_validation import validate_node_output
from .performance import get_performance_monitor, monitor_performance
from .security_utils import redact_log_message, redact_sensitive_data
from .state_utils import create_initial_agent_state
from .streamlit_utils import configure_page


# Lazy import to avoid circular dependency
def get_error_boundary():
    """Get StreamlitErrorBoundary with lazy import to avoid circular dependency."""
    return StreamlitErrorBoundary


__all__ = [
    "AicvgenError",
    "ConfigurationError",
    "StateManagerError",
    "redact_sensitive_data",
    "redact_log_message",
    "monitor_performance",
    "get_performance_monitor",
    "escape_latex",
    "recursively_escape_latex",
    "create_async_sync_decorator",
    "configure_page",
    "validate_node_output",
    "create_initial_agent_state",
    "get_error_boundary",
    "get_dotenv",
]
