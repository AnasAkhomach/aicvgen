"""Core module for the aicvgen application."""


# Lazy imports to avoid circular dependencies
def get_container():
    """Get container instance with lazy import."""
    from src.core.containers.main_container import get_container as _get_container

    return _get_container()


def get_workflow_manager():
    """Get workflow manager with lazy import."""
    from src.core.managers.workflow_manager import WorkflowManager

    return WorkflowManager


def get_session_manager():
    """Get session manager with lazy import."""
    from src.core.managers.session_manager import SessionManager

    return SessionManager


__all__ = ["get_container", "get_workflow_manager", "get_session_manager"]
