"""Agent lifecycle manager for optimized agent initialization and management."""

import threading
import weakref
from typing import Dict, List, Optional, Type, Any, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import asyncio
import time

from .dependency_injection import (
    DependencyContainer,
    get_container,
    LifecycleScope,
    DependencyMetadata,
)
from ..agents.agent_base import EnhancedAgentBase, AgentExecutionContext
from ..config.logging_config import get_structured_logger
from ..utils.error_handling import ErrorHandler, ErrorCategory, ErrorSeverity
from ..models.data_models import ContentType
from ..services.session_manager import get_session_manager
from ..services.error_recovery import get_error_recovery_service
from ..services.progress_tracker import get_progress_tracker

logger = get_structured_logger(__name__)


class AgentState(Enum):
    """States of an agent during its lifecycle."""

    NOT_INITIALIZED = "not_initialized"
    INITIALIZING = "initializing"
    READY = "ready"
    BUSY = "busy"
    IDLE = "idle"
    WARMING_UP = "warming_up"
    COOLING_DOWN = "cooling_down"
    DISPOSED = "disposed"
    ERROR = "error"


class AgentPoolStrategy(Enum):
    """Strategies for agent pool management."""

    LAZY = "lazy"  # Create agents on demand
    EAGER = "eager"  # Pre-create agents at startup
    ADAPTIVE = "adaptive"  # Adjust pool size based on usage
    FIXED = "fixed"  # Fixed pool size


@dataclass
class AgentMetrics:
    """Performance metrics for an agent."""

    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    total_execution_time: float = 0.0
    average_execution_time: float = 0.0
    last_execution: Optional[datetime] = None
    creation_time: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    memory_usage: Optional[float] = None

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_executions == 0:
            return 0.0
        return self.successful_executions / self.total_executions

    def update_execution(self, success: bool, execution_time: float):
        """Update execution metrics."""
        self.total_executions += 1
        if success:
            self.successful_executions += 1
        else:
            self.failed_executions += 1

        self.total_execution_time += execution_time
        self.average_execution_time = self.total_execution_time / self.total_executions
        self.last_execution = datetime.now()
        self.last_accessed = datetime.now()


@dataclass
class AgentPoolConfig:
    """Configuration for agent pools."""

    agent_type: str
    factory: Callable[[], EnhancedAgentBase]
    min_instances: int = 0
    max_instances: int = 5
    strategy: AgentPoolStrategy = AgentPoolStrategy.LAZY
    idle_timeout: timedelta = timedelta(minutes=10)
    warmup_on_startup: bool = False
    preload_dependencies: bool = True
    content_types: Optional[List[ContentType]] = None
    priority: int = 0
    tags: List[str] = field(default_factory=list)


@dataclass
class ManagedAgent:
    """Wrapper for managed agent instances."""

    instance: EnhancedAgentBase
    config: AgentPoolConfig
    state: AgentState = AgentState.READY
    metrics: AgentMetrics = field(default_factory=AgentMetrics)
    session_id: Optional[str] = None
    lock: threading.RLock = field(default_factory=threading.RLock)

    def mark_busy(self):
        """Mark agent as busy."""
        with self.lock:
            self.state = AgentState.BUSY
            self.metrics.last_accessed = datetime.now()

    def mark_idle(self):
        """Mark agent as idle."""
        with self.lock:
            self.state = AgentState.IDLE

    def is_available(self) -> bool:
        """Check if agent is available for execution."""
        return self.state in [AgentState.READY, AgentState.IDLE]

    def is_idle_timeout_exceeded(self) -> bool:
        """Check if agent has exceeded idle timeout."""
        if self.state != AgentState.IDLE:
            return False

        idle_time = datetime.now() - self.metrics.last_accessed
        return idle_time > self.config.idle_timeout


class AgentLifecycleManager:
    """Manages agent lifecycle with optimization and dependency injection."""

    def __init__(self, container: Optional[DependencyContainer] = None):
        self.container = container or get_container()
        self._lock = threading.RLock()
        self._pools: Dict[str, List[ManagedAgent]] = {}
        self._pool_configs: Dict[str, AgentPoolConfig] = {}
        self._agent_registry: Dict[str, Callable[[], EnhancedAgentBase]] = {}
        self._session_agents: Dict[str, Set[str]] = {}  # session_id -> agent_ids
        self._error_handler = ErrorHandler()

        # Performance tracking
        self._global_metrics = {
            "total_agents_created": 0,
            "total_agents_disposed": 0,
            "pool_hits": 0,
            "pool_misses": 0,
            "average_pool_utilization": 0.0,
            "startup_time": 0.0,
        }

        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        self._monitoring_task: Optional[asyncio.Task] = None
        self._shutdown = False

        self._register_core_dependencies()
        logger.info("Agent lifecycle manager initialized")

    def _register_core_dependencies(self):
        """Register core dependencies with the DI container."""
        # Register common services as singletons
        from ..config.logging_config import get_structured_logger
        from ..services.error_recovery import get_error_recovery_service
        from ..services.progress_tracker import get_progress_tracker
        from ..services.session_manager import get_session_manager

        self.container.register_singleton(
            "logger",
            type(logger),
            factory=lambda: get_structured_logger("agent_lifecycle"),
        )

        self.container.register_singleton(
            "error_recovery",
            type(get_error_recovery_service()),
            factory=get_error_recovery_service,
        )

        self.container.register_singleton(
            "progress_tracker",
            type(get_progress_tracker()),
            factory=get_progress_tracker,
        )

        self.container.register_singleton(
            "session_manager", type(get_session_manager()), factory=get_session_manager
        )

    def register_agent_type(
        self,
        agent_type: str,
        factory: Callable[[], EnhancedAgentBase] = None,
        config: Optional[AgentPoolConfig] = None,
    ):
        """Register an agent type with the lifecycle manager."""
        with self._lock:
            # Use DI container to resolve dependencies for each agent type
            if agent_type == "cv_analyzer":

                def factory():
                    llm_service = self.container.get(
                        type(self.container.get("llm_service")), "llm_service"
                    )
                    settings = self.container.get(
                        type(self.container.get("settings")), "settings"
                    )
                    logger = self.container.get(
                        type(self.container.get("logger")), "logger"
                    )
                    error_recovery = self.container.get(
                        type(self.container.get("error_recovery")), "error_recovery"
                    )
                    progress_tracker = self.container.get(
                        type(self.container.get("progress_tracker")), "progress_tracker"
                    )
                    session_manager = self.container.get(
                        type(self.container.get("session_manager")), "session_manager"
                    )
                    from ..agents.cv_analyzer_agent import CVAnalyzerAgent

                    return CVAnalyzerAgent(
                        name="CVAnalyzerAgent",
                        description="Analyzes CV content and job requirements to provide optimization recommendations",
                        llm_service=llm_service,
                        settings=settings,
                        logger=logger,
                        error_recovery=error_recovery,
                        progress_tracker=progress_tracker,
                        session_manager=session_manager,
                    )

            # ...repeat for other agent types, using the correct constructor signature and dependencies...
            self._agent_registry[agent_type] = factory

            if config is None:
                config = AgentPoolConfig(agent_type=agent_type, factory=factory)

            self._pool_configs[agent_type] = config
            self._pools[agent_type] = []

            # Register with DI container
            scope = (
                LifecycleScope.SESSION
                if config.strategy == AgentPoolStrategy.FIXED
                else LifecycleScope.TRANSIENT
            )

            self.container.register(
                DependencyMetadata(
                    name=f"agent_{agent_type}",
                    dependency_type=EnhancedAgentBase,
                    scope=scope,
                    factory=factory,
                    lazy=not config.warmup_on_startup,
                    max_idle_time=config.idle_timeout,
                )
            )

            logger.info(
                f"Agent type registered: {agent_type}",
                strategy=config.strategy.value,
                min_instances=config.min_instances,
                max_instances=config.max_instances,
            )

    def get_agent(
        self,
        agent_type: str,
        session_id: Optional[str] = None,
        content_type: Optional[ContentType] = None,
    ) -> Optional[ManagedAgent]:
        """Get an agent instance from the pool or create a new one."""
        with self._lock:
            if agent_type not in self._pool_configs:
                logger.error(f"Unknown agent type: {agent_type}")
                return None

            config = self._pool_configs[agent_type]
            pool = self._pools[agent_type]

            # Try to get available agent from pool
            available_agent = self._get_available_agent(pool, content_type)
            if available_agent:
                available_agent.mark_busy()
                available_agent.session_id = session_id
                self._global_metrics["pool_hits"] += 1

                # Track session association
                if session_id:
                    if session_id not in self._session_agents:
                        self._session_agents[session_id] = set()
                    self._session_agents[session_id].add(id(available_agent))

                return available_agent

            # Create new agent if pool allows
            if len(pool) < config.max_instances:
                new_agent = self._create_agent(config, session_id)
                if new_agent:
                    pool.append(new_agent)
                    new_agent.mark_busy()  # Mark as busy before returning
                    self._global_metrics["pool_misses"] += 1
                    self._global_metrics["total_agents_created"] += 1

                    # Track session association
                    if session_id:
                        if session_id not in self._session_agents:
                            self._session_agents[session_id] = set()
                        self._session_agents[session_id].add(id(new_agent))

                    return new_agent

            logger.warning(
                f"No available agents for type {agent_type}, pool at capacity"
            )
            return None

    def _get_available_agent(
        self, pool: List[ManagedAgent], content_type: Optional[ContentType] = None
    ) -> Optional[ManagedAgent]:
        """Get an available agent from the pool."""
        for agent in pool:
            if agent.is_available():
                # Check content type compatibility if specified
                if content_type and agent.config.content_types:
                    if content_type not in agent.config.content_types:
                        continue
                return agent
        return None

    def _create_agent(
        self, config: AgentPoolConfig, session_id: Optional[str] = None
    ) -> Optional[ManagedAgent]:
        """Create a new managed agent instance."""
        try:
            start_time = time.time()

            # Create agent instance using factory
            agent_instance = config.factory()

            # Inject dependencies if agent supports it
            if hasattr(agent_instance, "inject_dependencies"):
                dependencies = {
                    "logger": logger,
                    "error_recovery": get_error_recovery_service(),
                    "progress_tracker": get_progress_tracker(),
                    "session_manager": get_session_manager(),
                }
                agent_instance.inject_dependencies(dependencies)

            managed_agent = ManagedAgent(
                instance=agent_instance,
                config=config,
                state=AgentState.READY,
                session_id=session_id,
            )

            creation_time = time.time() - start_time
            logger.info(
                f"Agent created: {config.agent_type}",
                creation_time=creation_time,
                session_id=session_id,
            )

            return managed_agent

        except Exception as e:
            error_msg = f"Failed to create agent {config.agent_type}: {str(e)}"
            self._error_handler.handle_error(
                error_msg,
                ErrorCategory.AGENT_LIFECYCLE,
                ErrorSeverity.HIGH,
                context={"agent_type": config.agent_type, "session_id": session_id},
            )
            return None

    def return_agent(self, agent: ManagedAgent):
        """Return an agent to the pool after use."""
        with self._lock:
            agent.mark_idle()
            logger.debug(f"Agent returned to pool: {agent.config.agent_type}")

    def dispose_agent(self, agent: ManagedAgent):
        """Dispose of an agent instance."""
        with self._lock:
            agent.state = AgentState.DISPOSED

            # Remove from pool
            pool = self._pools.get(agent.config.agent_type, [])
            if agent in pool:
                pool.remove(agent)

            # Remove from session tracking
            agent_id = id(agent)
            for session_id, agent_ids in self._session_agents.items():
                agent_ids.discard(agent_id)

            # Dispose agent instance
            try:
                if hasattr(agent.instance, "dispose"):
                    agent.instance.dispose()
            except Exception as e:
                logger.error(f"Error disposing agent: {e}")

            self._global_metrics["total_agents_disposed"] += 1
            logger.debug(f"Agent disposed: {agent.config.agent_type}")

    def dispose_session_agents(self, session_id: str):
        """Dispose all agents associated with a session."""
        with self._lock:
            if session_id not in self._session_agents:
                return

            agent_ids = self._session_agents.pop(session_id)
            disposed_count = 0

            for agent_type, pool in self._pools.items():
                agents_to_dispose = []
                for agent in pool:
                    if id(agent) in agent_ids:
                        agents_to_dispose.append(agent)

                for agent in agents_to_dispose:
                    self.dispose_agent(agent)
                    disposed_count += 1

            logger.info(
                f"Session agents disposed", session_id=session_id, count=disposed_count
            )

    def warmup_pools(self):
        """Warm up agent pools by pre-creating instances."""
        with self._lock:
            start_time = time.time()
            total_created = 0

            for agent_type, config in self._pool_configs.items():
                if config.warmup_on_startup and config.min_instances > 0:
                    pool = self._pools[agent_type]
                    current_size = len(pool)

                    for _ in range(config.min_instances - current_size):
                        agent = self._create_agent(config)
                        if agent:
                            pool.append(agent)
                            total_created += 1

            warmup_time = time.time() - start_time
            self._global_metrics["startup_time"] = warmup_time

            logger.info(
                f"Agent pools warmed up",
                agents_created=total_created,
                warmup_time=warmup_time,
            )

    async def start_background_tasks(self):
        """Start background monitoring and cleanup tasks."""
        self._cleanup_task = asyncio.create_task(self._cleanup_worker())
        self._monitoring_task = asyncio.create_task(self._monitoring_worker())
        logger.info("Background tasks started")

    async def _cleanup_worker(self):
        """Background task for cleaning up idle agents."""
        while not self._shutdown:
            try:
                await asyncio.sleep(60)  # Run every minute
                self._cleanup_idle_agents()
            except Exception as e:
                logger.error(f"Error in cleanup worker: {e}")

    async def _monitoring_worker(self):
        """Background task for monitoring agent performance."""
        while not self._shutdown:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                self._update_performance_metrics()
            except Exception as e:
                logger.error(f"Error in monitoring worker: {e}")

    def _cleanup_idle_agents(self):
        """Clean up agents that have exceeded their idle timeout."""
        with self._lock:
            disposed_count = 0

            for agent_type, pool in self._pools.items():
                config = self._pool_configs[agent_type]
                agents_to_dispose = []

                for agent in pool:
                    if (
                        len(pool) > config.min_instances
                        and agent.is_idle_timeout_exceeded()
                    ):
                        agents_to_dispose.append(agent)

                for agent in agents_to_dispose:
                    self.dispose_agent(agent)
                    disposed_count += 1

            if disposed_count > 0:
                logger.info(f"Idle agents cleaned up: {disposed_count}")

    def _update_performance_metrics(self):
        """Update global performance metrics."""
        with self._lock:
            total_agents = sum(len(pool) for pool in self._pools.values())
            busy_agents = sum(
                1
                for pool in self._pools.values()
                for agent in pool
                if agent.state == AgentState.BUSY
            )

            if total_agents > 0:
                self._global_metrics["average_pool_utilization"] = (
                    busy_agents / total_agents
                )

            logger.debug(
                "Performance metrics updated",
                total_agents=total_agents,
                busy_agents=busy_agents,
                utilization=self._global_metrics["average_pool_utilization"],
            )

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about agent lifecycle management."""
        with self._lock:
            pool_stats = {}
            for agent_type, pool in self._pools.items():
                config = self._pool_configs[agent_type]
                pool_stats[agent_type] = {
                    "current_size": len(pool),
                    "min_instances": config.min_instances,
                    "max_instances": config.max_instances,
                    "strategy": config.strategy.value,
                    "busy_agents": sum(
                        1 for agent in pool if agent.state == AgentState.BUSY
                    ),
                    "idle_agents": sum(
                        1 for agent in pool if agent.state == AgentState.IDLE
                    ),
                    "ready_agents": sum(
                        1 for agent in pool if agent.state == AgentState.READY
                    ),
                }

            return {
                "global_metrics": self._global_metrics,
                "pool_statistics": pool_stats,
                "active_sessions": len(self._session_agents),
                "container_stats": self.container.get_statistics(),
            }

    def shutdown(self):
        """Shutdown the lifecycle manager and dispose all agents."""
        with self._lock:
            self._shutdown = True

            # Cancel background tasks
            if self._cleanup_task:
                self._cleanup_task.cancel()
            if self._monitoring_task:
                self._monitoring_task.cancel()

            # Dispose all agents
            for pool in self._pools.values():
                for agent in pool[:]:
                    self.dispose_agent(agent)

            self._pools.clear()
            self._session_agents.clear()

            logger.info("Agent lifecycle manager shutdown complete")


# Global lifecycle manager instance
_global_manager: Optional[AgentLifecycleManager] = None
_manager_lock = threading.Lock()


def get_agent_lifecycle_manager() -> AgentLifecycleManager:
    """Get the global agent lifecycle manager."""
    global _global_manager

    with _manager_lock:
        if _global_manager is None:
            _global_manager = AgentLifecycleManager()
        return _global_manager


def reset_agent_lifecycle_manager() -> None:
    """Reset the global manager (mainly for testing)."""
    global _global_manager

    with _manager_lock:
        if _global_manager:
            _global_manager.shutdown()
        _global_manager = None
