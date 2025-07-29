"""Backward compatibility module for container imports.

This module provides backward compatibility for imports that expect
the container to be at src.core.container. All imports are redirected
to the new location at src.core.containers.main_container.
"""

# Import everything from the new location
from src.core.containers.main_container import (
    Container,
    ContainerSingleton,
    get_container,
    validate_prompts_directory,
)

__all__ = [
    "Container",
    "ContainerSingleton",
    "get_container",
    "validate_prompts_directory",
]
