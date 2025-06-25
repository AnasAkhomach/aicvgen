"""Utilities module for the aicvgen application."""

from ..error_handling.boundaries import StreamlitErrorBoundary as ErrorBoundary
from ..error_handling.decorators import handle_errors as handle_error
from .exceptions import AicvgenError, ConfigurationError, StateManagerError
from .security_utils import redact_sensitive_data, redact_log_message
from .performance import monitor_performance, get_performance_monitor
from .latex_utils import escape_latex, recursively_escape_latex
from .decorators import create_async_sync_decorator
from .streamlit_utils import configure_page

__all__ = [
    "ErrorBoundary",
    "handle_error",
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
]
