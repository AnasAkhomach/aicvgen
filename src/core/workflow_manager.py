"""Backward compatibility module for workflow_manager imports.

This module provides backward compatibility for imports that expect
the workflow_manager to be at src.core.workflow_manager. All imports are redirected
to the new location at src.core.managers.workflow_manager.
"""

# Import everything from the new location
from src.core.managers.workflow_manager import WorkflowManager

__all__ = ["WorkflowManager"]
