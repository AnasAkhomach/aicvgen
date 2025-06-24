"""Dependency injection system for agent lifecycle management and optimization."""

import threading
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type, TypeVar, Callable, List
from dataclasses import dataclass, field
from enum import Enum
import time
from datetime import datetime, timedelta
import google.generativeai as genai

from ..config.logging_config import get_structured_logger
from ..utils.error_handling import (
    ErrorHandler,
    ErrorCategory,
    ErrorSeverity,
    ErrorContext,
)
from ..services.llm_client import LLMClient
from ..services.llm_retry_handler import LLMRetryHandler
from ..services.llm_service import EnhancedLLMService, AdvancedCache
from ..utils.error_classification import is_retryable_error
from ..config.settings import Settings
from ..services.rate_limiter import RateLimiter
from ..services.error_recovery import ErrorRecoveryService
from ..core.performance_optimizer import PerformanceOptimizer

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

    @abstractmethod
    def register(self, metadata: DependencyMetadata) -> None:
        """Register a dependency."""

    @abstractmethod
    def dispose(self, name: str) -> None:
        """Dispose of a dependency."""


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

    def _create_instance(
        self, name: str, metadata: DependencyMetadata
    ) -> DependencyInstance:
        """Create a new instance of a dependency and return a DependencyInstance."""
        if name in self._creating:
            raise ValueError(f"Circular dependency detected for '{name}'")

        start_time = time.time()
        self._creating.add(name)

        try:
            logger.debug(
                f"Creating dependency instance: {name}"
            )  # Resolve dependencies first
            resolved_deps = {}
            for dep_name in metadata.dependencies:
                if dep_name in self._registrations:
                    resolved_deps[dep_name] = self._get_by_name_internal(
                        dep_name
                    ).instance

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

            return dep_instance

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
            return self._create_instance(name, metadata).instance

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

    def dispose(self, name: str) -> None:
        """Dispose of a dependency instance."""
        with self._lock:
            # Remove from singleton instances
            if name in self._instances:
                instance = self._instances.pop(name)
                self._dispose_instance(instance)

            # Remove from session instances
            for session_instances in self._session_instances.values():
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

        except (
            Exception
        ) as exc:  # linter: justified, must catch all for resource cleanup
            dep_instance.state = DependencyState.ERROR
            dep_instance.error = exc
            logger.error(
                f"Error disposing dependency {dep_instance.metadata.name}: {exc}"
            )

    def dispose_session(self, session_id: str) -> None:
        """Dispose all dependencies for a specific session."""
        with self._lock:
            if session_id in self._session_instances:
                session_instances = self._session_instances.pop(session_id)
                for instance in session_instances.values():
                    self._dispose_instance(instance)

                logger.info(
                    "Session dependencies disposed",
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
            for session_instances in self._session_instances.values():
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
                except (
                    Exception
                ) as exc:  # linter: justified, must catch all for background thread
                    logger.error(f"Error in cleanup thread: {exc}")

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

    def _get_by_name_internal(self, name: str) -> DependencyInstance:
        """
        Internal method to get a dependency by name without acquiring locks.
        Should only be called when the caller already holds self._lock.
        """
        metadata = self._registrations.get(name)
        if not metadata:
            raise ValueError(f"No dependency named '{name}' is registered.")

        if metadata.scope == LifecycleScope.SINGLETON:
            if name not in self._instances:
                self._instances[name] = self._create_instance(name, metadata)
            dep_instance = self._instances[name]
            dep_instance.mark_accessed()
            return dep_instance

        if metadata.scope == LifecycleScope.SESSION:
            if not self.session_id:
                raise ValueError("Session scope requires a session_id.")
            if self.session_id not in self._session_instances:
                self._session_instances[self.session_id] = {}
            if name not in self._session_instances[self.session_id]:
                self._session_instances[self.session_id][name] = self._create_instance(
                    name, metadata
                )
            dep_instance = self._session_instances[self.session_id][name]
            dep_instance.mark_accessed()
            return dep_instance

        if metadata.scope == LifecycleScope.TRANSIENT:
            return self._create_instance(name, metadata)

        raise NotImplementedError(f"Lifecycle scope {metadata.scope} not implemented.")

    def get_by_name(self, name: str) -> Any:
        """Get a dependency instance by name."""
        with self._lock:
            return self._get_by_name_internal(name).instance


def build_llm_service(
    container: "DependencyContainer", user_api_key: Optional[str] = None
) -> EnhancedLLMService:
    """Factory to build the EnhancedLLMService with all its dependencies."""
    from ..utils.exceptions import ConfigurationError

    logger.info("Starting build_llm_service...")

    logger.info("Getting settings...")
    settings = container.get(Settings, "settings")
    logger.info("Settings obtained")

    logger.info("Getting rate_limiter...")
    rate_limiter = container.get(RateLimiter, "RateLimiter")
    logger.info("Rate limiter obtained")

    logger.info("Getting error_recovery...")
    error_recovery = container.get(ErrorRecoveryService, "ErrorRecoveryService")
    logger.info("Error recovery obtained")

    logger.info("Getting performance_optimizer...")
    performance_optimizer = container.get(PerformanceOptimizer, "PerformanceOptimizer")
    logger.info("Performance optimizer obtained")

    logger.info("Getting cache...")
    cache = container.get(AdvancedCache, "AdvancedCache")
    logger.info("Cache obtained")

    # Determine the API key to use (fail-fast if none available)
    logger.info("Determining API key...")
    api_key = None
    if user_api_key:
        api_key = user_api_key
    elif (
        hasattr(settings.llm, "gemini_api_key_primary")
        and settings.llm.gemini_api_key_primary
    ):
        api_key = settings.llm.gemini_api_key_primary
    elif (
        hasattr(settings.llm, "gemini_api_key_fallback")
        and settings.llm.gemini_api_key_fallback
    ):
        api_key = settings.llm.gemini_api_key_fallback

    if not api_key:
        raise ConfigurationError(
            "CRITICAL: No Gemini API key configured. "
            "Please set GEMINI_API_KEY or GEMINI_API_KEY_FALLBACK in your .env file. "
            "Application cannot start without a valid API key."
        )
    logger.info("API key determined")

    # Use timeout wrapper for potentially blocking operations
    def run_with_timeout(func, timeout_seconds=10):
        """Run a function with timeout to prevent hanging."""
        result = [None]
        exception = [None]

        def target():
            try:
                result[0] = func()
            except Exception as exc:  # linter: justified, must catch all for timeout
                exception[0] = exc

        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        thread.join(timeout_seconds)

        if thread.is_alive():
            raise ConfigurationError(
                f"LLM service initialization timed out after {timeout_seconds} seconds. "
                "This might be due to network issues or invalid API key."
            )

        if exception[0] is not None:
            if isinstance(exception[0], BaseException):
                raise exception[0]
            raise RuntimeError(
                f"Unexpected non-exception raised in run_with_timeout: {exception[0]}"
            )
        return result[0]

    # Configure the Google Generative AI client with the API key (with timeout)
    try:
        logger.info("Configuring Google Generative AI client...")
        run_with_timeout(lambda: genai.configure(api_key=api_key), timeout_seconds=10)
        logger.info("Google Generative AI client configured successfully")
    except Exception as exc:  # Narrowed variable name
        raise ConfigurationError(
            f"Failed to configure Google Generative AI client: {exc}. "
            "Please check your API key and network connection."
        ) from exc

    # Create the LLM model with proper error handling (with timeout)
    try:
        logger.info("Creating LLM model...")
        llm_model = run_with_timeout(
            lambda: genai.GenerativeModel(settings.llm_settings.default_model),
            timeout_seconds=10,
        )
        logger.info(
            "LLM model created successfully", model=settings.llm_settings.default_model
        )
    except Exception as exc:  # Narrowed variable name
        raise ConfigurationError(
            f"Failed to create LLM model '{settings.llm_settings.default_model}': {exc}. "
            "Please check your model name and API key."
        ) from exc

    logger.info("Creating LLMClient...")
    llm_client = LLMClient(llm_model)
    logger.info("LLMClient created")

    # Correctly define the retry handler logic
    def _is_retryable(exception: Exception) -> bool:
        return is_retryable_error(exception)

    logger.info("Creating LLMRetryHandler...")
    retry_handler = LLMRetryHandler(llm_client, _is_retryable)
    logger.info("LLMRetryHandler created")

    logger.info("Creating EnhancedLLMService...")
    service = EnhancedLLMService(
        settings=settings,
        llm_client=llm_client,
        llm_retry_handler=retry_handler,
        cache=cache,
        rate_limiter=rate_limiter,
        error_recovery=error_recovery,
        performance_optimizer=performance_optimizer,
        user_api_key=user_api_key,
        timeout=settings.llm.request_timeout,  # Use the correct timeout field
        async_optimizer=None,  # Assuming this is handled elsewhere or not needed for now
    )
    logger.info("EnhancedLLMService created successfully")

    return service


def configure_container(container: "DependencyContainer"):
    """Configure the dependency injection container with all agents and services."""
    # Settings and Core Services
    container.register_singleton("settings", Settings, factory=Settings)
    container.register_singleton("RateLimiter", RateLimiter)
    container.register_singleton("AdvancedCache", AdvancedCache)
    container.register_singleton("ErrorHandler", ErrorHandler)
    container.register_singleton(
        "PerformanceOptimizer",
        PerformanceOptimizer,
        factory=PerformanceOptimizer,
    )
    container.register_singleton(
        "ErrorRecoveryService",
        ErrorRecoveryService,
        factory=lambda: ErrorRecoveryService(
            logger=get_structured_logger("error_recovery")
        ),
    )

    # LLM Service Factory and Registration
    def llm_service_factory() -> EnhancedLLMService:
        return build_llm_service(container)

    container.register_singleton(
        "EnhancedLLMService",
        EnhancedLLMService,
        factory=llm_service_factory,
    )

    # Vector Store, Progress Tracker, Content Template Manager
    from ..services.vector_store_service import VectorStoreService
    from ..services.progress_tracker import ProgressTracker
    from ..templates.content_templates import ContentTemplateManager

    container.register_singleton(
        name="VectorStoreService",
        dependency_type=VectorStoreService,
        factory=lambda: VectorStoreService(
            settings=container.get(Settings, "settings")
        ),
    )
    container.register_singleton(
        name="ProgressTracker",
        dependency_type=ProgressTracker,
        factory=ProgressTracker,
    )
    container.register_singleton(
        name="ContentTemplateManager",
        dependency_type=ContentTemplateManager,
        factory=ContentTemplateManager,
    )

    # Agents
    from ..agents.enhanced_content_writer import EnhancedContentWriterAgent
    from ..agents.parser_agent import ParserAgent
    from ..agents.quality_assurance_agent import QualityAssuranceAgent
    from ..agents.formatter_agent import FormatterAgent
    from ..agents.research_agent import ResearchAgent
    from ..agents.cv_analyzer_agent import CVAnalyzerAgent
    from ..agents.cleaning_agent import CleaningAgent

    settings = container.get(Settings, "settings")
    template_manager = container.get(ContentTemplateManager, "ContentTemplateManager")

    container.register_singleton(
        name="ParserAgent",
        dependency_type=ParserAgent,
        factory=lambda: ParserAgent(
            llm_service=container.get(EnhancedLLMService, "EnhancedLLMService"),
            vector_store_service=container.get(
                VectorStoreService, "VectorStoreService"
            ),
            progress_tracker=container.get(ProgressTracker, "ProgressTracker"),
            settings=settings.agent if hasattr(settings, "agent") else settings,
            template_manager=template_manager,
        ),
        dependencies=[
            "EnhancedLLMService",
            "VectorStoreService",
            "ProgressTracker",
            "settings",
            "ContentTemplateManager",
        ],
    )

    container.register_singleton(
        name="EnhancedContentWriterAgent",
        dependency_type=EnhancedContentWriterAgent,
        factory=lambda: EnhancedContentWriterAgent(
            llm_service=container.get(EnhancedLLMService, "EnhancedLLMService"),
            progress_tracker=container.get(ProgressTracker, "ProgressTracker"),
            parser_agent=container.get(ParserAgent, "ParserAgent"),
            settings=settings.agent if hasattr(settings, "agent") else settings,
        ),
        dependencies=[
            "EnhancedLLMService",
            "ProgressTracker",
            "ParserAgent",
            "settings",
        ],
    )

    container.register_singleton(
        name="QualityAssuranceAgent",
        dependency_type=QualityAssuranceAgent,
        factory=lambda: QualityAssuranceAgent(
            llm_service=container.get(EnhancedLLMService, "EnhancedLLMService"),
            progress_tracker=container.get(ProgressTracker, "ProgressTracker"),
            template_manager=template_manager,
        ),
        dependencies=[
            "EnhancedLLMService",
            "ProgressTracker",
            "ContentTemplateManager",
        ],
    )

    container.register_singleton(
        name="FormatterAgent",
        dependency_type=FormatterAgent,
        factory=lambda: FormatterAgent(
            llm_service=container.get(EnhancedLLMService, "EnhancedLLMService"),
            progress_tracker=container.get(ProgressTracker, "ProgressTracker"),
            template_manager=template_manager,
        ),
        dependencies=[
            "EnhancedLLMService",
            "ProgressTracker",
            "ContentTemplateManager",
        ],
    )

    container.register_singleton(
        name="ResearchAgent",
        dependency_type=ResearchAgent,
        factory=lambda: ResearchAgent(
            llm_service=container.get(EnhancedLLMService, "EnhancedLLMService"),
            progress_tracker=container.get(ProgressTracker, "ProgressTracker"),
            vector_db=container.get(VectorStoreService, "VectorStoreService"),
            settings=settings.agent if hasattr(settings, "agent") else settings,
            template_manager=template_manager,
        ),
        dependencies=[
            "EnhancedLLMService",
            "ProgressTracker",
            "VectorStoreService",
            "settings",
            "ContentTemplateManager",
        ],
    )

    container.register_singleton(
        name="CVAnalyzerAgent",
        dependency_type=CVAnalyzerAgent,
        factory=lambda: CVAnalyzerAgent(
            config={
                "llm_service": container.get(EnhancedLLMService, "EnhancedLLMService"),
                "settings": settings.agent if hasattr(settings, "agent") else settings,
                "progress_tracker": container.get(ProgressTracker, "ProgressTracker"),
                "template_manager": template_manager,
                "name": "CVAnalyzerAgent",
                "description": "Agent responsible for analyzing the user's CV and extracting relevant information",
            }
        ),
        dependencies=[
            "EnhancedLLMService",
            "settings",
            "ProgressTracker",
            "ContentTemplateManager",
        ],
    )

    container.register_singleton(
        name="CleaningAgent",
        dependency_type=CleaningAgent,
        factory=lambda: CleaningAgent(
            llm_service=container.get(EnhancedLLMService, "EnhancedLLMService"),
            progress_tracker=container.get(ProgressTracker, "ProgressTracker"),
            template_manager=template_manager,
        ),
        dependencies=[
            "EnhancedLLMService",
            "ProgressTracker",
            "ContentTemplateManager",
        ],
    )
    logger.info(
        "All agents and services registered successfully with dependency injection"
    )


# Explicitly export configure_container for import
__all__ = [
    "DependencyContainer",
    "configure_container",
    "build_llm_service",
]
