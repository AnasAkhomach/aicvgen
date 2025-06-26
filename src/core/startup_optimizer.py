"""Startup optimization utilities for the CV generation system."""

import asyncio
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import threading
import psutil
import os
import logging

from .dependency_injection import get_container, reset_container
from .agent_lifecycle_manager import (
    get_agent_lifecycle_manager,
    reset_agent_lifecycle_manager,
)
from ..config.logging_config import get_structured_logger
from ..error_handling.models import (
    ErrorCategory,
    ErrorSeverity,
)
from ..error_handling.agent_error_handler import AgentErrorHandler as ErrorHandler

logger = get_structured_logger(__name__)


@dataclass
class StartupMetrics:
    """Metrics collected during startup optimization."""

    total_startup_time: float
    dependency_injection_time: float
    agent_pool_warmup_time: float
    memory_usage_before: float
    memory_usage_after: float
    memory_delta: float
    agents_preloaded: int
    dependencies_registered: int
    errors_encountered: int
    optimization_level: str
    timestamp: datetime


class StartupOptimizer:
    """Optimizes application startup performance."""

    def __init__(self):
        self.error_handler = ErrorHandler()
        self.metrics_history: List[StartupMetrics] = []
        self._optimization_strategies = {
            "minimal": self._minimal_startup,
            "balanced": self._balanced_startup,
            "aggressive": self._aggressive_startup,
            "development": self._development_startup,
        }

    async def optimize_startup(
        self,
        strategy: str = "balanced",
        warmup_pools: bool = True,
        preload_dependencies: bool = True,
    ) -> StartupMetrics:
        """Optimize application startup with the specified strategy."""
        start_time = time.time()
        memory_before = self._get_memory_usage()

        logger.info("Starting application optimization with strategy: %s", strategy)

        try:
            # Reset systems for clean startup
            reset_container()
            reset_agent_lifecycle_manager()

            # Execute optimization strategy
            if strategy not in self._optimization_strategies:
                raise ValueError(f"Unknown optimization strategy: {strategy}")

            di_time, warmup_time, agents_count, deps_count, errors = (
                await self._optimization_strategies[strategy](
                    warmup_pools, preload_dependencies
                )
            )

            total_time = time.time() - start_time
            memory_after = self._get_memory_usage()
            memory_delta = memory_after - memory_before

            metrics = StartupMetrics(
                total_startup_time=total_time,
                dependency_injection_time=di_time,
                agent_pool_warmup_time=warmup_time,
                memory_usage_before=memory_before,
                memory_usage_after=memory_after,
                memory_delta=memory_delta,
                agents_preloaded=agents_count,
                dependencies_registered=deps_count,
                errors_encountered=errors,
                optimization_level=strategy,
                timestamp=datetime.now(),
            )

            self.metrics_history.append(metrics)

            logger.info(
                "Startup optimization completed",
                strategy=strategy,
                total_time=total_time,
                memory_delta=memory_delta,
                agents_preloaded=agents_count,
                dependencies_registered=deps_count,
            )

            return metrics

        except (ValueError, RuntimeError, OSError) as e:
            self.error_handler.handle_error(
                f"Startup optimization failed: {str(e)}",
                ErrorCategory.CONFIGURATION,
                ErrorSeverity.HIGH,
            )
            raise
        # If truly unexpected, catch-all for logging only
        except Exception as e:
            self.error_handler.handle_error(
                f"Startup optimization failed (unexpected): {str(e)}",
                ErrorCategory.CONFIGURATION,
                ErrorSeverity.CRITICAL,
            )
            raise

    async def _minimal_startup(
        self, warmup_pools: bool, preload_dependencies: bool
    ) -> tuple[float, float, int, int, int]:
        """Minimal startup - only essential components."""
        di_start = time.time()

        # Initialize DI container with minimal dependencies
        container = get_container()
        deps_count = 0

        # Register only critical services
        from ..config.logging_config import get_structured_logger
        from logging import Logger

        container.register_singleton(
            "logger", Logger, factory=lambda: get_structured_logger("minimal_startup")
        )
        deps_count += 1

        di_time = time.time() - di_start

        # Minimal agent setup
        warmup_start = time.time()
        lifecycle_manager = get_agent_lifecycle_manager()
        agents_count = 0

        if warmup_pools:
            # Use dependency container for agent creation
            from .dependency_injection import get_container
            from ..agents.enhanced_content_writer import EnhancedContentWriterAgent
            from .agent_lifecycle_manager import AgentPoolConfig, AgentPoolStrategy

            container = get_container()
            container.register_agents()

            lifecycle_manager.register_agent_type(
                "content_writer",
                lambda: container.get(
                    EnhancedContentWriterAgent, "EnhancedContentWriterAgent"
                ),
                AgentPoolConfig(
                    agent_type="content_writer",
                    factory=lambda: container.get(
                        EnhancedContentWriterAgent, "EnhancedContentWriterAgent"
                    ),
                    min_instances=1,
                    max_instances=1,
                    strategy=AgentPoolStrategy.LAZY,
                ),
            )
            agents_count = 1

        warmup_time = time.time() - warmup_start

        return di_time, warmup_time, agents_count, deps_count, 0

    async def _balanced_startup(
        self, warmup_pools: bool, preload_dependencies: bool
    ) -> tuple[float, float, int, int, int]:
        """Balanced startup - good performance with reasonable resource usage."""
        di_start = time.time()

        # Initialize DI container with common dependencies
        container = get_container()
        deps_count = 0

        if preload_dependencies:
            # Register common services
            from ..config.logging_config import get_structured_logger
            from ..services.error_recovery import get_error_recovery_service
            from ..services.progress_tracker import get_progress_tracker
            from logging import Logger
            from ..services.error_recovery import ErrorRecoveryService
            from ..services.progress_tracker import ProgressTracker

            container.register_singleton(
                "logger",
                Logger,
                factory=lambda: get_structured_logger("balanced_startup"),
            )
            container.register_singleton(
                "error_recovery",
                ErrorRecoveryService,
                factory=lambda: ErrorRecoveryService(),
            )
            container.register_singleton(
                "progress_tracker", ProgressTracker, factory=lambda: ProgressTracker()
            )
            deps_count = 3

        di_time = time.time() - di_start

        # Balanced agent setup
        warmup_start = time.time()
        lifecycle_manager = get_agent_lifecycle_manager()
        agents_count = 0

        if warmup_pools:
            from ..agents.enhanced_content_writer import EnhancedContentWriterAgent
            from ..agents.specialized_agents import get_agent
            from .agent_lifecycle_manager import AgentPoolConfig, AgentPoolStrategy
            from ..models.data_models import ContentType  # Get dependency container
            from .dependency_injection import get_container

            container = get_container()
            container.register_agents()

            # Register core agents with balanced configuration
            agent_configs = [
                (
                    "content_writer",
                    lambda: container.get(
                        EnhancedContentWriterAgent, "EnhancedContentWriterAgent"
                    ),
                    AgentPoolStrategy.EAGER,
                    1,
                    2,
                    [ContentType.EXECUTIVE_SUMMARY],
                ),
                (
                    "cv_analysis",
                    lambda: get_agent("cv_analysis"),
                    AgentPoolStrategy.LAZY,
                    0,
                    1,
                    [ContentType.CV_ANALYSIS],
                ),
                (
                    "quality_assurance",
                    lambda: get_agent("quality_assurance"),
                    AgentPoolStrategy.LAZY,
                    0,
                    1,
                    [ContentType.QUALITY_CHECK],
                ),
            ]

            for (
                agent_type,
                factory,
                strategy,
                min_inst,
                max_inst,
                content_types,
            ) in agent_configs:
                try:
                    lifecycle_manager.register_agent_type(
                        agent_type,
                        factory,
                        AgentPoolConfig(
                            agent_type=agent_type,
                            factory=factory,
                            min_instances=min_inst,
                            max_instances=max_inst,
                            strategy=strategy,
                            content_types=content_types,
                            warmup_on_startup=(strategy == AgentPoolStrategy.EAGER),
                        ),
                    )
                    agents_count += 1
                except (TypeError, ValueError, RuntimeError) as e:
                    logger.warning("Failed to register agent %s: %s", agent_type, e)

            # Warm up eager agents
            lifecycle_manager.warmup_pools()

        warmup_time = time.time() - warmup_start

        return di_time, warmup_time, agents_count, deps_count, 0

    async def _aggressive_startup(
        self, warmup_pools: bool, preload_dependencies: bool
    ) -> tuple[float, float, int, int, int]:
        """Aggressive startup - maximum performance, higher resource usage."""
        di_start = time.time()

        # Initialize DI container with all dependencies
        container = get_container()
        deps_count = 0

        if preload_dependencies:
            # Register all services
            from ..config.logging_config import get_structured_logger
            from ..services.error_recovery import get_error_recovery_service
            from ..services.progress_tracker import get_progress_tracker
            from ..services.session_manager import get_session_manager

            from logging import Logger
            from ..services.error_recovery import ErrorRecoveryService
            from ..services.progress_tracker import ProgressTracker
            from ..services.session_manager import SessionManager

            services = [
                ("logger", Logger, lambda: get_structured_logger("aggressive_startup")),
                (
                    "error_recovery",
                    ErrorRecoveryService,
                    lambda: ErrorRecoveryService(),
                ),
                ("progress_tracker", ProgressTracker, lambda: ProgressTracker()),
                ("session_manager", SessionManager, get_session_manager),
            ]

            for name, service_type, factory in services:
                container.register_singleton(name, service_type, factory=factory)
                deps_count += 1

        di_time = time.time() - di_start

        # Aggressive agent setup
        warmup_start = time.time()
        lifecycle_manager = get_agent_lifecycle_manager()
        agents_count = 0

        if warmup_pools:
            from ..agents.enhanced_content_writer import EnhancedContentWriterAgent
            from ..agents.specialized_agents import get_agent
            from .agent_lifecycle_manager import AgentPoolConfig, AgentPoolStrategy
            from ..models.data_models import ContentType
            from .dependency_injection import get_container

            # Get dependency container
            container = get_container()
            container.register_agents()

            # Register all agents with aggressive configuration
            agent_configs = [
                (
                    "content_writer",
                    lambda: container.get(
                        EnhancedContentWriterAgent, "EnhancedContentWriterAgent"
                    ),
                    AgentPoolStrategy.EAGER,
                    2,
                    5,
                    [ContentType.EXECUTIVE_SUMMARY, ContentType.EXPERIENCE],
                ),
                (
                    "cv_analysis",
                    lambda: get_agent("cv_analysis"),
                    AgentPoolStrategy.EAGER,
                    1,
                    3,
                    [ContentType.CV_ANALYSIS],
                ),
                (
                    "cv_parser",
                    lambda: get_agent("cv_parser"),
                    AgentPoolStrategy.ADAPTIVE,
                    1,
                    2,
                    [ContentType.CV_PARSING],
                ),
                (
                    # "content_optimization" removed - was never implemented
                    AgentPoolStrategy.ADAPTIVE,
                    1,
                    2,
                    [ContentType.SKILLS, ContentType.ACHIEVEMENTS],
                ),
                (
                    "quality_assurance",
                    lambda: get_agent("quality_assurance"),
                    AgentPoolStrategy.LAZY,
                    0,
                    2,
                    [ContentType.QUALITY_CHECK],
                ),
            ]

            # Use parallel registration for faster startup
            async def register_agent(
                agent_type, factory, strategy, min_inst, max_inst, content_types
            ):
                try:
                    lifecycle_manager.register_agent_type(
                        agent_type,
                        factory,
                        AgentPoolConfig(
                            agent_type=agent_type,
                            factory=factory,
                            min_instances=min_inst,
                            max_instances=max_inst,
                            strategy=strategy,
                            content_types=content_types,
                            warmup_on_startup=(strategy == AgentPoolStrategy.EAGER),
                        ),
                    )
                    return 1
                except (TypeError, ValueError, RuntimeError) as e:
                    logger.warning("Failed to register agent %s: %s", agent_type, e)
                    return 0

            # Register agents in parallel
            registration_tasks = [
                register_agent(
                    agent_type, factory, strategy, min_inst, max_inst, content_types
                )
                for agent_type, factory, strategy, min_inst, max_inst, content_types in agent_configs
            ]

            results = await asyncio.gather(*registration_tasks, return_exceptions=True)
            agents_count = sum(r for r in results if isinstance(r, int))

            # Warm up all pools
            lifecycle_manager.warmup_pools()

        warmup_time = time.time() - warmup_start

        return di_time, warmup_time, agents_count, deps_count, 0

    async def _development_startup(
        self, warmup_pools: bool, preload_dependencies: bool
    ) -> tuple[float, float, int, int, int]:
        """Development startup - optimized for development workflow."""
        di_start = time.time()

        # Initialize DI container with development-friendly setup
        container = get_container()
        deps_count = 0

        if preload_dependencies:
            # Register services with development optimizations
            from ..config.logging_config import get_structured_logger
            from ..services.error_recovery import get_error_recovery_service

            # Enhanced logging for development
            from logging import Logger
            from ..services.error_recovery import ErrorRecoveryService

            dev_logger = get_structured_logger("development_startup")
            dev_logger.logger.setLevel(logging.DEBUG)

            container.register_singleton("logger", Logger, factory=lambda: dev_logger)
            container.register_singleton(
                "error_recovery",
                ErrorRecoveryService,
                factory=get_error_recovery_service,
            )
            deps_count = 2

        di_time = time.time() - di_start

        # Development-optimized agent setup
        warmup_start = time.time()
        lifecycle_manager = get_agent_lifecycle_manager()
        agents_count = 0

        if warmup_pools:
            from ..agents.enhanced_content_writer import EnhancedContentWriterAgent
            from .agent_lifecycle_manager import AgentPoolConfig, AgentPoolStrategy
            from ..models.data_models import ContentType
            from .dependency_injection import get_container

            # Get dependency container
            container = get_container()
            container.register_agents()

            # Register minimal agents for fast development cycles
            lifecycle_manager.register_agent_type(
                "content_writer",
                lambda: container.get(
                    EnhancedContentWriterAgent, "EnhancedContentWriterAgent"
                ),
                AgentPoolConfig(
                    agent_type="content_writer",
                    factory=lambda: container.get(
                        EnhancedContentWriterAgent, "EnhancedContentWriterAgent"
                    ),
                    min_instances=1,
                    max_instances=2,
                    strategy=AgentPoolStrategy.LAZY,
                    content_types=[ContentType.EXECUTIVE_SUMMARY],
                    idle_timeout=timedelta(
                        minutes=5
                    ),  # Shorter timeout for development
                ),
            )
            agents_count = 1

            # Quick warmup
            lifecycle_manager.warmup_pools()

        warmup_time = time.time() - warmup_start

        return di_time, warmup_time, agents_count, deps_count, 0

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024  # Convert to MB
        except (OSError, AttributeError, ImportError) as e:
            return 0.0

    def get_optimization_report(self) -> Dict[str, Any]:
        """Generate a comprehensive optimization report."""
        if not self.metrics_history:
            return {"error": "No optimization metrics available"}

        latest = self.metrics_history[-1]

        # Calculate averages if multiple runs
        if len(self.metrics_history) > 1:
            avg_startup_time = sum(
                m.total_startup_time for m in self.metrics_history
            ) / len(self.metrics_history)
            avg_memory_delta = sum(m.memory_delta for m in self.metrics_history) / len(
                self.metrics_history
            )
        else:
            avg_startup_time = latest.total_startup_time
            avg_memory_delta = latest.memory_delta

        return {
            "latest_metrics": {
                "total_startup_time": latest.total_startup_time,
                "dependency_injection_time": latest.dependency_injection_time,
                "agent_pool_warmup_time": latest.agent_pool_warmup_time,
                "memory_delta_mb": latest.memory_delta,
                "agents_preloaded": latest.agents_preloaded,
                "dependencies_registered": latest.dependencies_registered,
                "optimization_level": latest.optimization_level,
                "timestamp": latest.timestamp.isoformat(),
            },
            "performance_summary": {
                "average_startup_time": avg_startup_time,
                "average_memory_delta": avg_memory_delta,
                "total_optimization_runs": len(self.metrics_history),
                "fastest_startup": min(
                    m.total_startup_time for m in self.metrics_history
                ),
                "slowest_startup": max(
                    m.total_startup_time for m in self.metrics_history
                ),
            },
            "recommendations": self._generate_recommendations(latest),
        }

    def _generate_recommendations(self, metrics: StartupMetrics) -> List[str]:
        """Generate optimization recommendations based on metrics."""
        recommendations = []

        if metrics.total_startup_time > 5.0:
            recommendations.append(
                "Consider using 'minimal' or 'balanced' optimization strategy for faster startup"
            )

        if metrics.memory_delta > 100:  # More than 100MB
            recommendations.append(
                "High memory usage detected. Consider reducing agent pool sizes or using lazy loading"
            )

        if metrics.dependency_injection_time > metrics.agent_pool_warmup_time * 2:
            recommendations.append(
                "Dependency injection is taking longer than expected. Review service registration"
            )

        if metrics.agents_preloaded > 5:
            recommendations.append(
                "Many agents preloaded. Consider using adaptive or lazy strategies for less critical agents"
            )

        if metrics.errors_encountered > 0:
            recommendations.append(
                "Errors encountered during startup. Check logs for details"
            )

        if not recommendations:
            recommendations.append(
                "Startup performance looks good! No specific recommendations."
            )

        return recommendations


# Global optimizer instance
_global_optimizer: Optional[StartupOptimizer] = None
_optimizer_lock = threading.Lock()


def get_startup_optimizer() -> StartupOptimizer:
    """Get the global startup optimizer."""
    global _global_optimizer

    with _optimizer_lock:
        if _global_optimizer is None:
            _global_optimizer = StartupOptimizer()
        return _global_optimizer


def reset_startup_optimizer() -> None:
    """Reset the global optimizer (mainly for testing)."""
    global _global_optimizer

    with _optimizer_lock:
        _global_optimizer = None
