"""Standardized import fallback utilities for optional dependencies.

This module provides a consistent approach to handling optional dependencies
across the entire codebase, ensuring graceful degradation when dependencies
are not available.
"""

import logging
from typing import Any, Callable, Optional, Tuple, Union

# Get logger for import fallback notifications
logger = logging.getLogger(__name__)


class OptionalDependency:
    """Context manager for handling optional dependencies with consistent logging."""

    def __init__(
        self,
        dependency_name: str,
        feature_description: str = "",
        log_level: int = logging.WARNING,
        silent: bool = False,
    ):
        """
        Initialize optional dependency handler.

        Args:
            dependency_name: Name of the dependency being imported
            feature_description: Description of what feature will be disabled
            log_level: Logging level for import failures
            silent: If True, suppress logging of import failures
        """
        self.dependency_name = dependency_name
        self.feature_description = feature_description
        self.log_level = log_level
        self.silent = silent
        self.available = False
        self.module = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is ImportError:
            self.available = False
            if not self.silent:
                message = f"Optional dependency '{self.dependency_name}' not available"
                if self.feature_description:
                    message += f": {self.feature_description}"
                logger.log(self.log_level, message)
            return True  # Suppress the ImportError
        return False

    def import_module(self, module_path: str) -> Optional[Any]:
        """Import a module within the optional dependency context."""
        try:
            import importlib

            self.module = importlib.import_module(module_path)
            self.available = True
            return self.module
        except ImportError:
            self.available = False
            if not self.silent:
                message = f"Optional dependency '{self.dependency_name}' not available"
                if self.feature_description:
                    message += f": {self.feature_description}"
                logger.log(self.log_level, message)
            return None


def safe_import(
    module_path: str,
    dependency_name: str = None,
    feature_description: str = "",
    fallback_value: Any = None,
    log_level: int = logging.WARNING,
    silent: bool = False,
) -> Tuple[Any, bool]:
    """Safely import a module with standardized error handling.

    Args:
        module_path: The module path to import (e.g., 'weasyprint')
        dependency_name: Human-readable name of the dependency
        feature_description: Description of what feature will be disabled
        fallback_value: Value to return if import fails
        log_level: Logging level for import failures
        silent: If True, suppress logging of import failures

    Returns:
        Tuple of (imported_module_or_fallback, is_available)
    """
    if dependency_name is None:
        dependency_name = module_path

    try:
        import importlib

        module = importlib.import_module(module_path)
        return module, True
    except ImportError:
        if not silent:
            message = f"Optional dependency '{dependency_name}' not available"
            if feature_description:
                message += f": {feature_description}"
            logger.log(log_level, message)
        return fallback_value, False


def safe_import_from(
    module_path: str,
    import_names: Union[str, list],
    dependency_name: str = None,
    feature_description: str = "",
    fallback_factory: Callable = None,
    log_level: int = logging.WARNING,
    silent: bool = False,
) -> Tuple[Any, bool]:
    """Safely import specific items from a module with standardized error handling.

    Args:
        module_path: The module path to import from
        import_names: Name(s) to import from the module
        dependency_name: Human-readable name of the dependency
        feature_description: Description of what feature will be disabled
        fallback_factory: Function that returns fallback implementations
        log_level: Logging level for import failures
        silent: If True, suppress logging of import failures

    Returns:
        Tuple of (imported_items_or_fallback, is_available)
    """
    if dependency_name is None:
        dependency_name = module_path

    try:
        import importlib

        module = importlib.import_module(module_path)

        if isinstance(import_names, str):
            return getattr(module, import_names), True
        else:
            return tuple(getattr(module, name) for name in import_names), True

    except (ImportError, AttributeError):
        if not silent:
            message = f"Optional dependency '{dependency_name}' not available"
            if feature_description:
                message += f": {feature_description}"
            logger.log(log_level, message)

        if fallback_factory:
            return fallback_factory(), False
        else:
            return None, False


def get_security_utils():
    """Get security utilities with fallback implementations.

    Returns:
        Tuple of (redact_sensitive_data, redact_log_message) functions
    """

    def create_fallback_security_utils():
        """Create fallback security utility functions."""

        def redact_sensitive_data(data):
            return data

        def redact_log_message(message):
            return message

        return redact_sensitive_data, redact_log_message

    result, available = safe_import_from(
        module_path="src.utils.security_utils",
        import_names=["redact_sensitive_data", "redact_log_message"],
        dependency_name="security_utils",
        feature_description="advanced security features will use basic implementations",
        fallback_factory=create_fallback_security_utils,
        silent=True,  # This is an internal utility, don't log
    )

    return result


def get_weasyprint():
    """Get WeasyPrint with standardized fallback handling.

    Returns:
        Tuple of (weasyprint_module_or_None, is_available)
    """
    return safe_import(
        module_path="weasyprint",
        dependency_name="WeasyPrint",
        feature_description="PDF generation will be disabled",
        fallback_value=None,
    )


def get_google_exceptions():
    """Get Google API exceptions with standardized fallback handling.

    Returns:
        Tuple of (google_exceptions_module_or_None, is_available)
    """
    return safe_import(
        module_path="google.api_core.exceptions",
        dependency_name="Google API Core",
        feature_description="Google-specific error handling will use generic patterns",
        fallback_value=None,
    )


def get_dotenv():
    """Get python-dotenv with standardized fallback handling.

    Returns:
        Tuple of (load_dotenv_function_or_None, is_available)
    """
    result, available = safe_import_from(
        module_path="dotenv",
        import_names="load_dotenv",
        dependency_name="python-dotenv",
        feature_description="environment variables must be set manually",
        fallback_factory=lambda: lambda *args, **kwargs: None,
    )

    return result, available
