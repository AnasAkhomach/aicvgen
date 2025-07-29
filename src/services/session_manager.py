"""Backward compatibility module for session_manager imports.

This module provides backward compatibility for imports that expect
the session_manager to be at src.services.session_manager. All imports are redirected
to the new location at src.core.managers.session_manager.
"""

# Import everything from the new location
from src.core.managers.session_manager import SessionManager, SessionStatus

__all__ = ["SessionManager", "SessionStatus"]
