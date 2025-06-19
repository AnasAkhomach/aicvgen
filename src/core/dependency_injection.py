"""Dependency injection system for agent lifecycle management and optimization."""

import threading
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type, TypeVar, Callable, List, Union
from dataclasses import dataclass, field
from enum import Enum
import weakref
import time
from datetime import datetime, timedelta

from ..config.logging_config import get_structured_logger
from ..utils.error_handling import (
    ErrorHandler,
    ErrorCategory,
    ErrorSeverity,
    ErrorContext,
)

logger = get_structured_logger(__name__)
T = TypeVar("T")


class LifecycleScope(Enum):
    """Defines the lifecycle scope of dependencies."""

    SINGLETON = "singleton"  # Single instance for entire application
    SESSION = "session"  # One instance per session
    REQUEST = "request"  # New instance per request
    TRANSIENT = "transient"  # New instance every time
    PROTOTYPE = "prototype"  # New instance with custom lifecycle


class DependencyState(Enum):
    """States of a dependency during its lifecycle."""

    NOT_CREATED = "not_created"
    CREATING = "creating"
    CREATED = "created"
    INITIALIZING = "initializing"
    READY = "ready"
    DISPOSING = "disposing"
    DISPOSED = "disposed"
    ERROR = "error"


@dataclass
class DependencyMetadata:
    """Metadata for dependency registration and management."""

    name: str
    dependency_type: Type
    scope: LifecycleScope
    factory: Optional[Callable[[], Any]] = None
    dependencies: List[str] = field(default_factory=list)
    lazy: bool = True
    auto_dispose: bool = True
    max_idle_time: Optional[timedelta] = None
    priority: int = 0  # Higher priority dependencies are created first
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DependencyInstance:
    """Represents an instance of a dependency with its lifecycle information."""

    instance: Any
    metadata: DependencyMetadata
    state: DependencyState = DependencyState.CREATED
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    session_id: Optional[str] = None
    error: Optional[Exception] = None

    def mark_accessed(self):
        """Mark the instance as accessed."""
        self.last_accessed = datetime.now()
        self.access_count += 1


class IDependencyProvider(ABC):
    """Interface for dependency providers."""

    @abstractmethod
    def get(self, dependency_type: Type[T], name: Optional[str] = None) -> T:
        """Get a dependency instance."""
        pass

    @abstractmethod
    def register(self, metadata: DependencyMetadata) -> None:
        """Register a dependency."""
        pass

    @abstractmethod
    def dispose(self, name: str) -> None:
        """Dispose of a dependency."""
        pass


class DependencyContainer(IDependencyProvider):
    """Main dependency injection container with lifecycle management."""

    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id
        self._lock = threading.RLock()
        self._registrations: Dict[str, DependencyMetadata] = {}
        self._instances: Dict[str, DependencyInstance] = {}
        self._session_instances: Dict[str, Dict[str, DependencyInstance]] = {}
        self._creating: set = (
            set()
        )  # Track dependencies being created to prevent cycles
        self._error_handler = ErrorHandler()
        self._cleanup_thread: Optional[threading.Thread] = None
        self._shutdown = False

        # Performance tracking
        self._stats = {
            "total_created": 0,
            "total_disposed": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "creation_time_total": 0.0,
            "average_creation_time": 0.0,
        }

        self._start_cleanup_thread()
        logger.info("Dependency container initialized", session_id=self.session_id)

    def register(self, metadata: DependencyMetadata) -> None:
        """Register a dependency with the container."""
        with self._lock:
            if metadata.name in self._registrations:
                logger.warning(
                    f"Dependency '{metadata.name}' already registered, overwriting"
                )

            self._registrations[metadata.name] = metadata
            logger.debug(
                f"Dependency registered: {metadata.name}",
                scope=metadata.scope.value,
                lazy=metadata.lazy,
            )

    def register_singleton(
        self,
        name: str,
        dependency_type: Type[T],
        factory: Optional[Callable[[], T]] = None,
        dependencies: Optional[List[str]] = None,
    ) -> None:
        """Register a singleton dependency."""
        metadata = DependencyMetadata(
            name=name,
            dependency_type=dependency_type,
            scope=LifecycleScope.SINGLETON,
            factory=factory,
            dependencies=dependencies or [],
            lazy=True,
        )
        self.register(metadata)

    def register_transient(
        self,
        name: str,
        dependency_type: Type[T],
        factory: Callable[[], T],
        dependencies: Optional[List[str]] = None,
    ) -> None:
        """Register a transient dependency."""
        metadata = DependencyMetadata(
            name=name,
            dependency_type=dependency_type,
            scope=LifecycleScope.TRANSIENT,
            factory=factory,
            dependencies=dependencies or [],
            lazy=True,
        )
        self.register(metadata)

    def register_session(
        self,
        name: str,
        dependency_type: Type[T],
        factory: Optional[Callable[[], T]] = None,
        dependencies: Optional[List[str]] = None,
    ) -> None:
        """Register a session-scoped dependency."""
        metadata = DependencyMetadata(
            name=name,
            dependency_type=dependency_type,
            scope=LifecycleScope.SESSION,
            factory=factory,
            dependencies=dependencies or [],
            lazy=True,
        )
        self.register(metadata)

    def get(self, dependency_type: Type[T], name: Optional[str] = None) -> T:
        """Get a dependency instance."""
        if name is None:
            name = dependency_type.__name__

        with self._lock:
            if name not in self._registrations:
                raise ValueError(f"Dependency '{name}' not registered")

            metadata = self._registrations[name]

            # Check for existing instance based on scope
            existing_instance = self._get_existing_instance(name, metadata)
            if existing_instance:
                existing_instance.mark_accessed()
                self._stats["cache_hits"] += 1
                return existing_instance.instance

            # Create new instance
            self._stats["cache_misses"] += 1
            return self._create_instance(name, metadata)

    def get_or_create(
        self,
        dependency_type: Type[T],
        name: Optional[str] = None,
        factory: Optional[Callable[[], T]] = None,
        scope: LifecycleScope = LifecycleScope.SINGLETON,
    ) -> T:
        """Get an existing dependency or create and register it if it doesn't exist."""
        if name is None:
            name = dependency_type.__name__

        with self._lock:
            # If already registered, just get it
            if name in self._registrations:
                return self.get(dependency_type, name)

            # Register and create new dependency
            metadata = DependencyMetadata(
                name=name,
                dependency_type=dependency_type,
                scope=scope,
                factory=factory,
                dependencies=[],
                lazy=True,
            )
            self.register(metadata)

            # Now get the newly registered dependency
            return self.get(dependency_type, name)

    def _get_existing_instance(
        self, name: str, metadata: DependencyMetadata
    ) -> Optional[DependencyInstance]:
        """Get existing instance if available based on scope."""
        if metadata.scope == LifecycleScope.SINGLETON:
            return self._instances.get(name)
        elif metadata.scope == LifecycleScope.SESSION and self.session_id:
            session_instances = self._session_instances.get(self.session_id, {})
            return session_instances.get(name)
        elif metadata.scope == LifecycleScope.TRANSIENT:
            return None  # Always create new instance

        return None

    def _create_instance(self, name: str, metadata: DependencyMetadata) -> Any:
        """Create a new instance of a dependency."""
        if name in self._creating:
            raise ValueError(f"Circular dependency detected for '{name}'")

        start_time = time.time()
        self._creating.add(name)

        try:
            logger.debug(f"Creating dependency instance: {name}")

            # Resolve dependencies first
            resolved_deps = {}
            for dep_name in metadata.dependencies:
                if dep_name in self._registrations:
                    dep_metadata = self._registrations[dep_name]
                    resolved_deps[dep_name] = self.get(
                        dep_metadata.dependency_type, dep_name
                    )

            # Create instance
            if metadata.factory:
                if metadata.dependencies:
                    # Pass resolved dependencies to factory if it accepts them
                    try:
                        instance = metadata.factory(**resolved_deps)
                    except TypeError:
                        # Factory doesn't accept dependencies, call without args
                        instance = metadata.factory()
                else:
                    instance = metadata.factory()
            else:
                # Use default constructor
                instance = metadata.dependency_type()

            # Create dependency instance wrapper
            dep_instance = DependencyInstance(
                instance=instance,
                metadata=metadata,
                state=DependencyState.READY,
                session_id=self.session_id,
            )

            # Store based on scope
            if metadata.scope == LifecycleScope.SINGLETON:
                self._instances[name] = dep_instance
            elif metadata.scope == LifecycleScope.SESSION and self.session_id:
                if self.session_id not in self._session_instances:
                    self._session_instances[self.session_id] = {}
                self._session_instances[self.session_id][name] = dep_instance

            # Update statistics
            creation_time = time.time() - start_time
            self._stats["total_created"] += 1
            self._stats["creation_time_total"] += creation_time
            self._stats["average_creation_time"] = (
                self._stats["creation_time_total"] / self._stats["total_created"]
            )

            logger.info(
                f"Dependency created successfully: {name}",
                creation_time=creation_time,
                scope=metadata.scope.value,
            )

            return instance

        except Exception as e:
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
            self._creating.discard(name)

    def dispose(self, name: str) -> None:
        """Dispose of a dependency instance."""
        with self._lock:
            # Remove from singleton instances
            if name in self._instances:
                instance = self._instances.pop(name)
                self._dispose_instance(instance)

            # Remove from session instances
            for session_id, session_instances in self._session_instances.items():
                if name in session_instances:
                    instance = session_instances.pop(name)
                    self._dispose_instance(instance)

    def _dispose_instance(self, dep_instance: DependencyInstance) -> None:
        """Dispose of a dependency instance."""
        try:
            dep_instance.state = DependencyState.DISPOSING

            # Call dispose method if available
            if hasattr(dep_instance.instance, "dispose"):
                dep_instance.instance.dispose()
            elif hasattr(dep_instance.instance, "close"):
                dep_instance.instance.close()

            dep_instance.state = DependencyState.DISPOSED
            self._stats["total_disposed"] += 1

            logger.debug(f"Dependency disposed: {dep_instance.metadata.name}")

        except Exception as e:
            dep_instance.state = DependencyState.ERROR
            dep_instance.error = e
            logger.error(
                f"Error disposing dependency {dep_instance.metadata.name}: {e}"
            )

    def dispose_session(self, session_id: str) -> None:
        """Dispose all dependencies for a specific session."""
        with self._lock:
            if session_id in self._session_instances:
                session_instances = self._session_instances.pop(session_id)
                for instance in session_instances.values():
                    self._dispose_instance(instance)

                logger.info(
                    f"Session dependencies disposed",
                    session_id=session_id,
                    count=len(session_instances),
                )

    def cleanup_idle_instances(self) -> None:
        """Clean up idle instances that exceed their max idle time."""
        with self._lock:
            current_time = datetime.now()
            to_dispose = []

            # Check singleton instances
            for name, instance in self._instances.items():
                if (
                    instance.metadata.max_idle_time
                    and current_time - instance.last_accessed
                    > instance.metadata.max_idle_time
                ):
                    to_dispose.append((name, instance))

            # Check session instances
            for session_id, session_instances in self._session_instances.items():
                for name, instance in session_instances.items():
                    if (
                        instance.metadata.max_idle_time
                        and current_time - instance.last_accessed
                        > instance.metadata.max_idle_time
                    ):
                        to_dispose.append((name, instance))

            # Dispose idle instances
            for name, instance in to_dispose:
                logger.info(f"Disposing idle dependency: {name}")
                self.dispose(name)

    def get_statistics(self) -> Dict[str, Any]:
        """Get container statistics."""
        with self._lock:
            return {
                **self._stats,
                "registered_dependencies": len(self._registrations),
                "singleton_instances": len(self._instances),
                "session_instances": sum(
                    len(instances) for instances in self._session_instances.values()
                ),
                "total_instances": len(self._instances)
                + sum(len(instances) for instances in self._session_instances.values()),
            }

    def _start_cleanup_thread(self) -> None:
        """Start background cleanup thread."""

        def cleanup_worker():
            while not self._shutdown:
                try:
                    time.sleep(60)  # Run cleanup every minute
                    if not self._shutdown:
                        self.cleanup_idle_instances()
                except Exception as e:
                    logger.error(f"Error in cleanup thread: {e}")

        self._cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        self._cleanup_thread.start()

    def shutdown(self) -> None:
        """Shutdown the container and dispose all instances."""
        with self._lock:
            self._shutdown = True

            # Dispose all instances
            for instance in self._instances.values():
                self._dispose_instance(instance)
            self._instances.clear()

            for session_instances in self._session_instances.values():
                for instance in session_instances.values():
                    self._dispose_instance(instance)
            self._session_instances.clear()

            logger.info("Dependency container shutdown complete")


# Global container instance
_global_container: Optional[DependencyContainer] = None
_container_lock = threading.Lock()


def get_container(session_id: Optional[str] = None) -> DependencyContainer:
    """Get the global dependency container."""
    global _global_container

    with _container_lock:
        if _global_container is None:
            _global_container = DependencyContainer(session_id)
        return _global_container


def reset_container() -> None:
    """Reset the global container (mainly for testing)."""
    global _global_container

    with _container_lock:
        if _global_container:
            _global_container.shutdown()
        _global_container = None
