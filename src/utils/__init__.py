"""Backward compatibility module for utils imports.

This module provides backward compatibility for imports that expect
utils to be at src.utils. All imports are redirected to the new
location at src.core.utils.
"""


# Lazy imports to avoid circular dependencies
def _get_core_utils():
    """Get core utils with lazy import to avoid circular dependencies."""
    from src.core.utils import (
        AicvgenError,
        ConfigurationError,
        StateManagerError,
        configure_page,
        create_async_sync_decorator,
        create_initial_agent_state,
        escape_latex,
        get_error_boundary,
        get_performance_monitor,
        monitor_performance,
        recursively_escape_latex,
        redact_log_message,
        redact_sensitive_data,
        validate_node_output,
    )

    return {
        "AicvgenError": AicvgenError,
        "ConfigurationError": ConfigurationError,
        "StateManagerError": StateManagerError,
        "redact_sensitive_data": redact_sensitive_data,
        "redact_log_message": redact_log_message,
        "monitor_performance": monitor_performance,
        "get_performance_monitor": get_performance_monitor,
        "escape_latex": escape_latex,
        "recursively_escape_latex": recursively_escape_latex,
        "create_async_sync_decorator": create_async_sync_decorator,
        "configure_page": configure_page,
        "validate_node_output": validate_node_output,
        "create_initial_agent_state": create_initial_agent_state,
        "get_error_boundary": get_error_boundary,
    }


# Dynamically expose all exports
_utils = _get_core_utils()
for name, obj in _utils.items():
    globals()[name] = obj

__all__ = list(_utils.keys())
