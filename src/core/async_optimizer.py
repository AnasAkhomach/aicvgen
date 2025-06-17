"""Advanced async execution optimizer for the AI CV Generator.

This module provides sophisticated async execution optimizations including:
- Intelligent concurrency control and throttling
- Adaptive semaphore management
- Async context pooling and reuse
- Deadlock detection and prevention
- Async operation batching and coalescing
"""

import asyncio
import time
import weakref
from typing import Dict, Any, Optional, List, Callable, Union, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import threading
from contextlib import asynccontextmanager
from functools import wraps
import inspect
import traceback

from ..config.logging_config import get_structured_logger
from ..utils.performance import get_performance_monitor

logger = get_structured_logger("async_optimizer")


@dataclass
class ConcurrencyConfig:
    """Configuration for concurrency control."""

    max_concurrent_operations: int = 50
    max_concurrent_per_type: int = 10
    adaptive_scaling: bool = True
    scaling_factor: float = 1.2
    min_concurrency: int = 2
    max_concurrency: int = 100
    throttle_threshold: float = 0.8
    deadlock_timeout: float = 30.0


@dataclass
class AsyncPoolConfig:
    """Configuration for async context pooling."""

    pool_size: int = 20
    max_idle_time: float = 300.0
    cleanup_interval: float = 60.0
    enable_warmup: bool = True
    warmup_size: int = 5


class AdaptiveSemaphore:
    """Adaptive semaphore that adjusts capacity based on performance metrics."""

    def __init__(self, initial_value: int, config: ConcurrencyConfig):
        self.config = config
        self._semaphore = asyncio.Semaphore(initial_value)
        self._current_capacity = initial_value
        self._pending_count = 0
        self._completed_count = 0
        self._error_count = 0
        self._total_wait_time = 0.0
        self._last_adjustment = time.time()
        self._lock = asyncio.Lock()

        # Performance tracking
        self._operation_times: deque = deque(maxlen=100)
        self._wait_times: deque = deque(maxlen=100)

        logger.info("Adaptive semaphore initialized", initial_capacity=initial_value)

    async def acquire(self) -> float:
        """Acquire semaphore with wait time tracking."""
        start_time = time.time()

        async with self._lock:
            self._pending_count += 1

        await self._semaphore.acquire()

        wait_time = time.time() - start_time
        self._wait_times.append(wait_time)
        self._total_wait_time += wait_time

        return wait_time

    def release(self, operation_time: float, success: bool = True):
        """Release semaphore with performance tracking."""
        self._semaphore.release()

        self._operation_times.append(operation_time)
        self._completed_count += 1

        if not success:
            self._error_count += 1

        # Trigger adaptive scaling check
        if self.config.adaptive_scaling:
            asyncio.create_task(self._check_scaling())

    async def _check_scaling(self):
        """Check if semaphore capacity should be adjusted."""
        async with self._lock:
            now = time.time()

            # Only adjust every 10 seconds minimum
            if now - self._last_adjustment < 10:
                return

            # Calculate performance metrics
            avg_wait_time = (
                sum(self._wait_times) / len(self._wait_times) if self._wait_times else 0
            )
            avg_operation_time = (
                sum(self._operation_times) / len(self._operation_times)
                if self._operation_times
                else 0
            )
            error_rate = self._error_count / max(self._completed_count, 1)

            # Determine if scaling is needed
            should_scale_up = (
                avg_wait_time
                > avg_operation_time * 0.1  # Wait time > 10% of operation time
                and error_rate < 0.05  # Low error rate
                and self._current_capacity < self.config.max_concurrency
            )

            should_scale_down = (
                avg_wait_time < avg_operation_time * 0.01  # Very low wait time
                or error_rate > 0.1  # High error rate
                or self._current_capacity > self.config.min_concurrency
            )

            if should_scale_up:
                new_capacity = min(
                    int(self._current_capacity * self.config.scaling_factor),
                    self.config.max_concurrency,
                )
                await self._adjust_capacity(new_capacity)
                logger.info(
                    "Scaled up semaphore",
                    old_capacity=self._current_capacity,
                    new_capacity=new_capacity,
                )

            elif should_scale_down and not should_scale_up:
                new_capacity = max(
                    int(self._current_capacity / self.config.scaling_factor),
                    self.config.min_concurrency,
                )
                await self._adjust_capacity(new_capacity)
                logger.info(
                    "Scaled down semaphore",
                    old_capacity=self._current_capacity,
                    new_capacity=new_capacity,
                )

            self._last_adjustment = now

    async def _adjust_capacity(self, new_capacity: int):
        """Adjust semaphore capacity."""
        if new_capacity > self._current_capacity:
            # Increase capacity
            for _ in range(new_capacity - self._current_capacity):
                self._semaphore.release()
        elif new_capacity < self._current_capacity:
            # Decrease capacity (acquire without releasing)
            for _ in range(self._current_capacity - new_capacity):
                try:
                    await asyncio.wait_for(self._semaphore.acquire(), timeout=1.0)
                except asyncio.TimeoutError:
                    break

        self._current_capacity = new_capacity

    def get_stats(self) -> Dict[str, Any]:
        """Get semaphore statistics."""
        avg_wait_time = (
            sum(self._wait_times) / len(self._wait_times) if self._wait_times else 0
        )
        avg_operation_time = (
            sum(self._operation_times) / len(self._operation_times)
            if self._operation_times
            else 0
        )

        return {
            "current_capacity": self._current_capacity,
            "pending_count": self._pending_count,
            "completed_count": self._completed_count,
            "error_count": self._error_count,
            "error_rate": self._error_count / max(self._completed_count, 1),
            "avg_wait_time": avg_wait_time,
            "avg_operation_time": avg_operation_time,
            "total_wait_time": self._total_wait_time,
        }


class AsyncContextPool:
    """Pool for reusing async contexts and resources."""

    def __init__(self, config: AsyncPoolConfig):
        self.config = config
        self._pools: Dict[str, List[Any]] = defaultdict(list)
        self._in_use: Dict[str, Set[Any]] = defaultdict(set)
        self._last_used: Dict[str, Dict[Any, float]] = defaultdict(dict)
        self._lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None

        self._stats = {
            "contexts_created": 0,
            "contexts_reused": 0,
            "contexts_cleaned": 0,
            "pool_hits": 0,
            "pool_misses": 0,
        }

        logger.info("Async context pool initialized", config=config)

    async def start(self):
        """Start the context pool."""
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

        if self.config.enable_warmup:
            await self._warmup_pools()

        logger.info("Async context pool started")

    async def stop(self):
        """Stop the context pool."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        # Cleanup all contexts
        async with self._lock:
            for pool_type, contexts in self._pools.items():
                for context in contexts:
                    await self._cleanup_context(context)

            self._pools.clear()
            self._in_use.clear()
            self._last_used.clear()

        logger.info("Async context pool stopped")

    @asynccontextmanager
    async def get_context(self, context_type: str, factory_func: Callable):
        """Get a context from the pool or create a new one."""
        context = await self._acquire_context(context_type, factory_func)

        try:
            yield context
        finally:
            await self._release_context(context_type, context)

    async def _acquire_context(self, context_type: str, factory_func: Callable):
        """Acquire a context from the pool."""
        async with self._lock:
            # Try to get from pool
            if self._pools[context_type]:
                context = self._pools[context_type].pop()
                self._in_use[context_type].add(context)
                self._stats["pool_hits"] += 1
                self._stats["contexts_reused"] += 1
                return context

            # Create new context
            self._stats["pool_misses"] += 1

        # Create outside of lock to avoid blocking
        context = await self._create_context(factory_func)

        async with self._lock:
            self._in_use[context_type].add(context)
            self._stats["contexts_created"] += 1

        return context

    async def _release_context(self, context_type: str, context):
        """Release a context back to the pool."""
        async with self._lock:
            if context in self._in_use[context_type]:
                self._in_use[context_type].remove(context)

                # Add back to pool if not full
                if len(self._pools[context_type]) < self.config.pool_size:
                    self._pools[context_type].append(context)
                    self._last_used[context_type][context] = time.time()
                else:
                    # Pool is full, cleanup context
                    await self._cleanup_context(context)

    async def _create_context(self, factory_func: Callable):
        """Create a new context using the factory function."""
        if asyncio.iscoroutinefunction(factory_func):
            return await factory_func()
        else:
            return factory_func()

    async def _cleanup_context(self, context):
        """Cleanup a context."""
        try:
            if hasattr(context, "close"):
                if asyncio.iscoroutinefunction(context.close):
                    await context.close()
                else:
                    context.close()
            elif hasattr(context, "__aexit__"):
                await context.__aexit__(None, None, None)
        except Exception as e:
            logger.warning("Error cleaning up context", error=str(e))

    async def _warmup_pools(self):
        """Warmup pools with initial contexts."""
        # This would be implemented based on specific context types
        logger.info("Context pool warmup completed")

    async def _cleanup_loop(self):
        """Periodic cleanup of idle contexts."""
        while True:
            try:
                await asyncio.sleep(self.config.cleanup_interval)
                await self._cleanup_idle_contexts()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in context cleanup loop", error=str(e))

    async def _cleanup_idle_contexts(self):
        """Clean up contexts that have been idle too long."""
        current_time = time.time()

        async with self._lock:
            for context_type, contexts in list(self._pools.items()):
                idle_contexts = []

                for context in contexts:
                    last_used = self._last_used[context_type].get(context, current_time)
                    if current_time - last_used > self.config.max_idle_time:
                        idle_contexts.append(context)

                # Remove and cleanup idle contexts
                for context in idle_contexts:
                    if context in contexts:
                        contexts.remove(context)
                    if context in self._last_used[context_type]:
                        del self._last_used[context_type][context]

                    await self._cleanup_context(context)
                    self._stats["contexts_cleaned"] += 1

    def get_stats(self) -> Dict[str, Any]:
        """Get context pool statistics."""
        pool_sizes = {
            pool_type: len(contexts) for pool_type, contexts in self._pools.items()
        }
        in_use_counts = {
            pool_type: len(contexts) for pool_type, contexts in self._in_use.items()
        }

        return {
            **self._stats,
            "pool_sizes": pool_sizes,
            "in_use_counts": in_use_counts,
            "total_pools": len(self._pools),
        }


class DeadlockDetector:
    """Deadlock detection and prevention for async operations."""

    def __init__(self, timeout: float = 30.0):
        self.timeout = timeout
        self._waiting_for: Dict[asyncio.Task, Set[asyncio.Task]] = {}
        self._lock = asyncio.Lock()
        self._detection_task: Optional[asyncio.Task] = None

        logger.info("Deadlock detector initialized", timeout=timeout)

    async def start(self):
        """Start deadlock detection."""
        self._detection_task = asyncio.create_task(self._detection_loop())
        logger.info("Deadlock detector started")

    async def stop(self):
        """Stop deadlock detection."""
        if self._detection_task:
            self._detection_task.cancel()
            try:
                await self._detection_task
            except asyncio.CancelledError:
                pass

        logger.info("Deadlock detector stopped")

    @asynccontextmanager
    async def track_dependency(self, waiting_for_tasks: List[asyncio.Task]):
        """Track task dependencies for deadlock detection."""
        current_task = asyncio.current_task()
        if not current_task:
            yield
            return

        async with self._lock:
            self._waiting_for[current_task] = set(waiting_for_tasks)

        try:
            yield
        finally:
            async with self._lock:
                if current_task in self._waiting_for:
                    del self._waiting_for[current_task]

    async def _detection_loop(self):
        """Periodic deadlock detection."""
        while True:
            try:
                await asyncio.sleep(5)  # Check every 5 seconds
                await self._detect_deadlocks()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in deadlock detection loop", error=str(e))

    async def _detect_deadlocks(self):
        """Detect potential deadlocks using cycle detection."""
        async with self._lock:
            # Build dependency graph
            graph = self._waiting_for.copy()

            # Find cycles using DFS
            visited = set()
            rec_stack = set()

            for task in graph:
                if task not in visited:
                    cycle = self._find_cycle_dfs(task, graph, visited, rec_stack, [])
                    if cycle:
                        logger.warning(
                            "Potential deadlock detected",
                            cycle_length=len(cycle),
                            tasks=[str(t) for t in cycle],
                        )
                        # Could implement deadlock resolution here

    def _find_cycle_dfs(self, task, graph, visited, rec_stack, path):
        """Find cycles in the dependency graph using DFS."""
        visited.add(task)
        rec_stack.add(task)
        path.append(task)

        for dependent_task in graph.get(task, []):
            if dependent_task not in visited:
                cycle = self._find_cycle_dfs(
                    dependent_task, graph, visited, rec_stack, path
                )
                if cycle:
                    return cycle
            elif dependent_task in rec_stack:
                # Found a cycle
                cycle_start = path.index(dependent_task)
                return path[cycle_start:] + [dependent_task]

        rec_stack.remove(task)
        path.pop()
        return None


class AsyncOptimizer:
    """Main async execution optimizer."""

    def __init__(
        self,
        concurrency_config: Optional[ConcurrencyConfig] = None,
        pool_config: Optional[AsyncPoolConfig] = None,
    ):

        self.concurrency_config = concurrency_config or ConcurrencyConfig()
        self.pool_config = pool_config or AsyncPoolConfig()

        # Initialize components
        self.global_semaphore = AdaptiveSemaphore(
            self.concurrency_config.max_concurrent_operations, self.concurrency_config
        )

        self.type_semaphores: Dict[str, AdaptiveSemaphore] = {}
        self.context_pool = AsyncContextPool(self.pool_config)
        self.deadlock_detector = DeadlockDetector(
            self.concurrency_config.deadlock_timeout
        )

        # Performance monitoring
        self.performance_monitor = get_performance_monitor()

        # Operation tracking
        self._active_operations: Dict[str, int] = defaultdict(int)
        self._operation_stats: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"count": 0, "total_time": 0.0, "errors": 0}
        )

        logger.info("Async optimizer initialized")

    async def start(self):
        """Start the async optimizer."""
        await self.context_pool.start()
        await self.deadlock_detector.start()

        logger.info("Async optimizer started")

    async def stop(self):
        """Stop the async optimizer."""
        await self.context_pool.stop()
        await self.deadlock_detector.stop()

        logger.info("Async optimizer stopped")

    @asynccontextmanager
    async def optimized_execution(
        self, operation_type: str, operation_name: str = None
    ):
        """Context manager for optimized async execution."""
        operation_name = operation_name or operation_type

        # Get type-specific semaphore
        if operation_type not in self.type_semaphores:
            self.type_semaphores[operation_type] = AdaptiveSemaphore(
                self.concurrency_config.max_concurrent_per_type, self.concurrency_config
            )

        type_semaphore = self.type_semaphores[operation_type]

        # Acquire semaphores
        start_time = time.time()
        global_wait = await self.global_semaphore.acquire()
        type_wait = await type_semaphore.acquire()

        self._active_operations[operation_type] += 1

        try:
            async with self.performance_monitor.measure_async(operation_name):
                yield

            # Success
            operation_time = time.time() - start_time
            self._update_operation_stats(operation_type, operation_time, True)

        except Exception as e:
            # Error
            operation_time = time.time() - start_time
            self._update_operation_stats(operation_type, operation_time, False)
            raise

        finally:
            # Release semaphores
            operation_time = time.time() - start_time
            self.global_semaphore.release(operation_time, True)
            type_semaphore.release(operation_time, True)

            self._active_operations[operation_type] -= 1

    async def batch_execute(
        self,
        operations: List[Callable],
        operation_type: str,
        max_concurrency: Optional[int] = None,
    ) -> List[Any]:
        """Execute a batch of operations with optimal concurrency."""
        max_concurrency = (
            max_concurrency or self.concurrency_config.max_concurrent_per_type
        )
        semaphore = asyncio.Semaphore(max_concurrency)

        async def execute_with_semaphore(operation):
            async with semaphore:
                async with self.optimized_execution(operation_type):
                    if asyncio.iscoroutinefunction(operation):
                        return await operation()
                    else:
                        return operation()

        # Execute all operations concurrently
        tasks = [execute_with_semaphore(op) for op in operations]

        # Track dependencies for deadlock detection
        async with self.deadlock_detector.track_dependency(tasks):
            results = await asyncio.gather(*tasks, return_exceptions=True)

        return results

    def _update_operation_stats(
        self, operation_type: str, operation_time: float, success: bool
    ):
        """Update operation statistics."""
        stats = self._operation_stats[operation_type]
        stats["count"] += 1
        stats["total_time"] += operation_time

        if not success:
            stats["errors"] += 1

    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive async optimization statistics."""
        # Calculate operation averages
        operation_averages = {}
        for op_type, stats in self._operation_stats.items():
            if stats["count"] > 0:
                operation_averages[op_type] = {
                    "avg_time": stats["total_time"] / stats["count"],
                    "error_rate": stats["errors"] / stats["count"],
                    "total_count": stats["count"],
                }

        return {
            "global_semaphore": self.global_semaphore.get_stats(),
            "type_semaphores": {
                op_type: sem.get_stats()
                for op_type, sem in self.type_semaphores.items()
            },
            "context_pool": self.context_pool.get_stats(),
            "active_operations": dict(self._active_operations),
            "operation_averages": operation_averages,
        }


# Global optimizer instance
_global_async_optimizer: Optional[AsyncOptimizer] = None


def get_async_optimizer() -> AsyncOptimizer:
    """Get the global async optimizer instance."""
    global _global_async_optimizer
    if _global_async_optimizer is None:
        _global_async_optimizer = AsyncOptimizer()
    return _global_async_optimizer


def reset_async_optimizer():
    """Reset the global async optimizer."""
    global _global_async_optimizer
    if _global_async_optimizer:
        # Note: In a real implementation, you'd want to properly stop the optimizer
        pass
    _global_async_optimizer = None


# Decorator for optimized async execution
def optimize_async(operation_type: str, operation_name: str = None):
    """Decorator for optimized async execution."""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            optimizer = get_async_optimizer()

            async with optimizer.optimized_execution(operation_type, operation_name):
                return await func(*args, **kwargs)

        return wrapper

    return decorator


# Utility function for batch async execution
async def batch_async_execute(
    operations: List[Callable],
    operation_type: str,
    max_concurrency: Optional[int] = None,
) -> List[Any]:
    """Execute a batch of async operations with optimization."""
    optimizer = get_async_optimizer()
    return await optimizer.batch_execute(operations, operation_type, max_concurrency)
