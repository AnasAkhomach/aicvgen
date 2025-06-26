"""Dependency injection system for agent lifecycle management and optimization."""

import threading
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type, TypeVar, Callable, List
from dataclasses import dataclass, field
from enum import Enum
import time
from datetime import datetime, timedelta

# Removed specific service/agent imports to break circular dependencies.
# Registration will now be handled in `application_startup.py`
from src.config.logging_config import get_structured_logger
from src.error_handling.models import (
    ErrorCategory,
    ErrorSeverity,
    ErrorContext,
)
from src.error_handling.boundaries import CATCHABLE_EXCEPTIONS
from src.error_handling.agent_error_handler import AgentErrorHandler as ErrorHandler


logger = get_structured_logger(__name__)
T = TypeVar("T")


class LifecycleScope(Enum):
    """Defines the lifecycle scope of dependencies."""

    SINGLETON = "singleton"  # Single instance for entire application
    SESSION = "session"  # One instance per session
    TRANSIENT = "transient"  # New instance every time


@dataclass
class DependencyMetadata:
    """Metadata for dependency registration and management."""

    name: str
    dependency_type: Type
    scope: LifecycleScope
    factory: Optional[Callable[..., Any]] = None
    dependencies: List[str] = field(default_factory=list)
    lazy: bool = True


class DependencyContainer:
    """Main dependency injection container with lifecycle management."""

    def __init__(self):
        self._lock = threading.RLock()
        self._registrations: Dict[str, DependencyMetadata] = {}
        self._singleton_instances: Dict[str, Any] = {}
        self._session_instances: Dict[str, Dict[str, Any]] = (
            {}
        )  # session_id -> {dep_name: instance}
        self._creating: set = set()
        self._error_handler = ErrorHandler()
        logger.info("Dependency container initialized")

    def register(self, metadata: DependencyMetadata) -> None:
        """Register a dependency with the container."""
        with self._lock:
            if metadata.name in self._registrations:
                logger.warning(
                    f"Dependency '{metadata.name}' already registered, overwriting"
                )
            self._registrations[metadata.name] = metadata
            logger.debug(
                f"Dependency registered: {metadata.name}", scope=metadata.scope.value
            )

    def register_singleton(
        self,
        name: str,
        dependency_type: Type[T],
        factory: Optional[Callable[..., T]] = None,
        dependencies: Optional[List[str]] = None,
    ) -> None:
        """Register a singleton dependency."""
        self.register(
            DependencyMetadata(
                name=name,
                dependency_type=dependency_type,
                scope=LifecycleScope.SINGLETON,
                factory=factory,
                dependencies=dependencies or [],
            )
        )

    def register_transient(
        self,
        name: str,
        dependency_type: Type[T],
        factory: Callable[..., T],
        dependencies: Optional[List[str]] = None,
    ) -> None:
        """Register a transient dependency."""
        self.register(
            DependencyMetadata(
                name=name,
                dependency_type=dependency_type,
                scope=LifecycleScope.TRANSIENT,
                factory=factory,
                dependencies=dependencies or [],
            )
        )

    def register_session(
        self,
        name: str,
        dependency_type: Type[T],
        factory: Optional[Callable[..., T]] = None,
        dependencies: Optional[List[str]] = None,
    ) -> None:
        """Register a session-scoped dependency."""
        self.register(
            DependencyMetadata(
                name=name,
                dependency_type=dependency_type,
                scope=LifecycleScope.SESSION,
                factory=factory,
                dependencies=dependencies or [],
            )
        )

    def _create_instance(
        self, name: str, metadata: DependencyMetadata, session_id: Optional[str] = None
    ) -> Any:
        """Create a new instance of a dependency."""
        if name in self._creating:
            raise ValueError(f"Circular dependency detected for '{name}'")

        self._creating.add(name)
        try:
            logger.debug("Creating dependency instance: %s", name)

            resolved_deps = {}
            for dep_name in metadata.dependencies:
                # Allow for dependency names that are not valid python identifiers
                arg_name = dep_name.replace("-", "_")
                resolved_deps[arg_name] = self.get_by_name(
                    dep_name, session_id=session_id
                )

            if metadata.factory:
                instance = metadata.factory(**resolved_deps)
            else:
                instance = metadata.dependency_type(**resolved_deps)

            logger.info(
                f"Dependency created successfully: {name}", scope=metadata.scope.value
            )
            return instance
        except CATCHABLE_EXCEPTIONS as e:
            error_msg = f"Failed to create dependency '{name}': {str(e)}"
            self._error_handler.handle_error(
                error_msg,
                ErrorCategory.CONFIGURATION,
                ErrorSeverity.HIGH,
                context=ErrorContext(
                    additional_data={
                        "dependency_name": name,
                        "dependency_type": str(metadata.dependency_type),
                    }
                ),
            )
            raise
        finally:
            if name in self._creating:
                self._creating.remove(name)

    def get_by_name(self, name: str, session_id: Optional[str] = None) -> Any:
        """Get a dependency by its registered name."""
        with self._lock:
            metadata = self._registrations.get(name)
            if not metadata:
                raise ValueError(f"No dependency registered with name '{name}'")

            if metadata.scope == LifecycleScope.SINGLETON:
                if name not in self._singleton_instances:
                    self._singleton_instances[name] = self._create_instance(
                        name, metadata
                    )
                return self._singleton_instances[name]

            elif metadata.scope == LifecycleScope.SESSION:
                if not session_id:
                    raise ValueError(
                        "Session ID is required for session-scoped dependencies."
                    )
                if session_id not in self._session_instances:
                    self._session_instances[session_id] = {}
                if name not in self._session_instances[session_id]:
                    self._session_instances[session_id][name] = self._create_instance(
                        name, metadata, session_id=session_id
                    )
                return self._session_instances[session_id][name]

            elif metadata.scope == LifecycleScope.TRANSIENT:
                return self._create_instance(name, metadata, session_id=session_id)

            else:
                raise ValueError(f"Unknown lifecycle scope: {metadata.scope}")

    def get(self, dependency_type: Type[T], session_id: Optional[str] = None) -> T:
        """Get a dependency instance by its type.

        Raises ValueError if no dependency or multiple dependencies are found for the given type.
        """
        with self._lock:
            candidates = [
                name
                for name, meta in self._registrations.items()
                if issubclass(meta.dependency_type, dependency_type)
            ]

            if not candidates:
                raise ValueError(
                    f"No dependency of type {dependency_type.__name__} registered."
                )
            if len(candidates) > 1:
                raise ValueError(
                    f"Multiple dependencies of type {dependency_type.__name__} registered. "
                    f"Use get_by_name(). Candidates: {candidates}"
                )

            return self.get_by_name(candidates[0], session_id=session_id)

    def dispose_session(self, session_id: str):
        """Dispose of all instances for a given session."""
        with self._lock:
            if session_id in self._session_instances:
                del self._session_instances[session_id]
                logger.info("Disposed instances for session %s", session_id)


# --- Singleton Implementation ---

_container_instance: Optional[DependencyContainer] = None
_container_lock = threading.Lock()


def get_container() -> DependencyContainer:
    """Get the singleton instance of the dependency container."""
    global _container_instance
    if _container_instance is None:
        with _container_lock:
            if _container_instance is None:
                _container_instance = DependencyContainer()
    return _container_instance
