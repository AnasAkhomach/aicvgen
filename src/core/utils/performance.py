"""Performance optimization utilities for the AI CV Generator.

This module provides comprehensive performance monitoring, optimization,
and profiling tools to enhance the application's efficiency.
"""

import asyncio
import functools
import gc
import json
import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import psutil

from src.config.logging_config import get_structured_logger
from src.constants.memory_constants import MemoryConstants
from src.constants.performance_constants import PerformanceConstants
from src.error_handling.exceptions import AicvgenError

logger = get_structured_logger(__name__)

# Global registry for performance monitors


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""

    operation_name: str
    start_time: float
    end_time: float
    duration: float
    memory_before: float
    memory_after: float
    memory_delta: float
    cpu_percent: float
    success: bool = True
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        """Duration in milliseconds."""
        return self.duration * 1000


class PerformanceMonitor:
    """Comprehensive performance monitoring system."""

    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.metrics_history: deque = deque(maxlen=max_history)
        self.operation_stats: Dict[str, List[float]] = defaultdict(list)
        self.lock = threading.Lock()
        self._start_time = time.time()

        # Performance thresholds (in seconds)
        self.thresholds = {
            "llm_call": PerformanceConstants.LLM_CALL_THRESHOLD,
            "agent_execution": PerformanceConstants.AGENT_EXECUTION_THRESHOLD,
            "file_processing": PerformanceConstants.FILE_PROCESSING_THRESHOLD,
            "database_operation": PerformanceConstants.DATABASE_OPERATION_THRESHOLD,
            "default": PerformanceConstants.DEFAULT_OPERATION_THRESHOLD,
        }

        logger.info("Performance monitor initialized", max_history=max_history)

    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024

    def get_cpu_percent(self) -> float:
        """Get current CPU usage percentage."""
        return psutil.cpu_percent(interval=0.1)

    @contextmanager
    def measure_sync(self, operation_name: str, **metadata):
        """Context manager for measuring synchronous operations."""
        start_time = time.time()
        memory_before = self.get_memory_usage()
        cpu_percent = self.get_cpu_percent()

        success = True
        error_message = None

        try:
            yield
        except (AicvgenError, psutil.Error) as e:
            success = False
            error_message = str(e)
            logger.error(
                f"Operation {operation_name} failed with known error: {e}",
                exc_info=True,
            )
            raise
        except (TypeError, ValueError, KeyError, AttributeError) as e:
            success = False
            error_message = str(e)
            logger.error(
                f"Operation {operation_name} failed with unexpected error: {e}",
                exc_info=True,
            )
            raise
        finally:
            end_time = time.time()
            memory_after = self.get_memory_usage()
            duration = end_time - start_time

            metrics = PerformanceMetrics(
                operation_name=operation_name,
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                memory_before=memory_before,
                memory_after=memory_after,
                memory_delta=memory_after - memory_before,
                cpu_percent=cpu_percent,
                success=success,
                error_message=error_message,
                metadata=metadata,
            )

            self._record_metrics(metrics)

    @asynccontextmanager
    async def measure_async(self, operation_name: str, **metadata):
        """Async context manager for measuring asynchronous operations."""
        start_time = time.time()
        memory_before = self.get_memory_usage()
        cpu_percent = self.get_cpu_percent()

        success = True
        error_message = None

        try:
            yield
        except (AicvgenError, psutil.Error, asyncio.CancelledError) as e:
            success = False
            error_message = str(e)
            logger.error(
                f"Async operation {operation_name} failed with known error: {e}",
                exc_info=True,
            )
            raise
        except (TypeError, ValueError, KeyError, AttributeError) as e:
            success = False
            error_message = str(e)
            logger.error(
                f"Async operation {operation_name} failed with unexpected error: {e}",
                exc_info=True,
            )
            raise
        finally:
            end_time = time.time()
            memory_after = self.get_memory_usage()
            duration = end_time - start_time

            metrics = PerformanceMetrics(
                operation_name=operation_name,
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                memory_before=memory_before,
                memory_after=memory_after,
                memory_delta=memory_after - memory_before,
                cpu_percent=cpu_percent,
                success=success,
                error_message=error_message,
                metadata=metadata,
            )

            self._record_metrics(metrics)

    def _record_metrics(self, metrics: PerformanceMetrics):
        """Record performance metrics."""
        with self.lock:
            self.metrics_history.append(metrics)
            self.operation_stats[metrics.operation_name].append(metrics.duration)

            # Check for performance issues
            threshold = self.thresholds.get(
                metrics.operation_name, self.thresholds["default"]
            )
            if metrics.duration > threshold:
                logger.warning(
                    f"Performance threshold exceeded for {metrics.operation_name}: duration {metrics.duration} > threshold {threshold}",
                    extra={
                        "operation": metrics.operation_name,
                        "duration": metrics.duration,
                        "threshold": threshold,
                        "memory_delta": metrics.memory_delta,
                    },
                )

            # Log successful operations at debug level
            if metrics.success:
                logger.debug(
                    f"Operation {metrics.operation_name} completed successfully",
                    extra={
                        "operation": metrics.operation_name,
                        "duration_ms": metrics.duration_ms,
                        "memory_delta_mb": round(metrics.memory_delta, 2),
                        "cpu_percent": metrics.cpu_percent,
                    },
                )
            else:
                logger.error(
                    f"Operation {metrics.operation_name} failed",
                    extra={
                        "operation": metrics.operation_name,
                        "duration_ms": metrics.duration_ms,
                        "error": metrics.error_message,
                    },
                )

    def get_operation_stats(self, operation_name: str) -> Dict[str, Any]:
        """Get statistics for a specific operation."""
        with self.lock:
            durations = self.operation_stats.get(operation_name, [])

            if not durations:
                return {"operation": operation_name, "count": 0}

            return {
                "operation": operation_name,
                "count": len(durations),
                "total_duration": sum(durations),
                "average_duration": sum(durations) / len(durations),
                "min_duration": min(durations),
                "max_duration": max(durations),
                "last_duration": durations[-1] if durations else 0,
            }

    def get_overall_stats(self) -> Dict[str, Any]:
        """Get overall performance statistics."""
        with self.lock:
            total_operations = len(self.metrics_history)
            if total_operations == 0:
                return {
                    "total_operations": 0,
                    "uptime_seconds": time.time() - self._start_time,
                }

            successful_ops = sum(1 for m in self.metrics_history if m.success)
            failed_ops = total_operations - successful_ops

            total_duration = sum(m.duration for m in self.metrics_history)
            avg_duration = total_duration / total_operations

            memory_deltas = [m.memory_delta for m in self.metrics_history]
            avg_memory_delta = sum(memory_deltas) / len(memory_deltas)

            return {
                "total_operations": total_operations,
                "successful_operations": successful_ops,
                "failed_operations": failed_ops,
                "success_rate_percent": (successful_ops / total_operations) * 100,
                "average_duration_seconds": avg_duration,
                "average_duration_ms": avg_duration * 1000,
                "total_duration_seconds": total_duration,
                "average_memory_delta_mb": avg_memory_delta,
                "uptime_seconds": time.time() - self._start_time,
                "current_memory_mb": self.get_memory_usage(),
                "current_cpu_percent": self.get_cpu_percent(),
            }

    def get_recent_metrics(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent performance metrics."""
        with self.lock:
            recent = list(self.metrics_history)[-limit:]
            return [
                {
                    "operation": m.operation_name,
                    "duration_ms": m.duration_ms,
                    "memory_delta_mb": round(m.memory_delta, 2),
                    "cpu_percent": m.cpu_percent,
                    "success": m.success,
                    "timestamp": datetime.fromtimestamp(m.start_time).isoformat(),
                    "metadata": m.metadata,
                }
                for m in recent
            ]

    def export_metrics(self, filepath: str):
        """Export metrics to a JSON file."""
        with self.lock:
            data = {
                "export_timestamp": datetime.now().isoformat(),
                "overall_stats": self.get_overall_stats(),
                "operation_stats": {
                    op: self.get_operation_stats(op)
                    for op in self.operation_stats.keys()
                },
                "recent_metrics": self.get_recent_metrics(100),
            }

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

            logger.info("Performance metrics exported", filepath=filepath)

    def clear_metrics(self):
        """Clear all stored metrics."""
        with self.lock:
            self.metrics_history.clear()
            self.operation_stats.clear()
            self._start_time = time.time()

            logger.info("Performance metrics cleared")


class MemoryOptimizer:
    """Memory optimization utilities."""

    def __init__(self):
        self.gc_threshold = MemoryConstants.GC_THRESHOLD_MB
        self.last_gc_time = time.time()
        self.gc_interval = MemoryConstants.GC_INTERVAL_SECONDS

        logger.info("Memory optimizer initialized")

    def get_memory_info(self) -> Dict[str, Any]:
        """Get detailed memory information."""
        process = psutil.Process()
        memory_info = process.memory_info()

        return {
            "rss_mb": memory_info.rss / 1024 / 1024,
            "vms_mb": memory_info.vms / 1024 / 1024,
            "percent": process.memory_percent(),
            "available_mb": psutil.virtual_memory().available / 1024 / 1024,
            "total_mb": psutil.virtual_memory().total / 1024 / 1024,
        }

    def should_run_gc(self) -> bool:
        """Determine if garbage collection should be run."""
        current_time = time.time()
        memory_info = self.get_memory_info()

        # Run GC if memory usage is high or enough time has passed
        return (
            memory_info["rss_mb"] > self.gc_threshold
            or current_time - self.last_gc_time > self.gc_interval
        )

    def optimize_memory(self) -> Dict[str, Any]:
        """Run memory optimization."""
        memory_before = self.get_memory_info()

        # Force garbage collection
        collected = gc.collect()

        memory_after = self.get_memory_info()
        self.last_gc_time = time.time()

        memory_freed = memory_before["rss_mb"] - memory_after["rss_mb"]

        result = {
            "objects_collected": collected,
            "memory_before_mb": memory_before["rss_mb"],
            "memory_after_mb": memory_after["rss_mb"],
            "memory_freed_mb": memory_freed,
            "timestamp": datetime.now().isoformat(),
        }

        logger.info(
            f"Memory optimization completed, freed {round(memory_freed, 2)} MB",
            extra={
                "objects_collected": collected,
                "memory_freed_mb": round(memory_freed, 2),
            },
        )

        return result

    def auto_optimize(self):
        """Automatically optimize memory if needed."""
        if self.should_run_gc():
            return self.optimize_memory()
        return None


class BatchProcessor:
    """Optimized batch processing utilities."""

    def __init__(
        self,
        batch_size: int = MemoryConstants.DEFAULT_BATCH_SIZE,
        max_concurrent: int = MemoryConstants.MAX_CONCURRENT_OPERATIONS,
    ):
        self.batch_size = batch_size
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)

        logger.info(
            f"Batch processor initialized with batch_size={batch_size}, max_concurrent={max_concurrent}"
        )

    async def process_batch_async(
        self,
        items: List[Any],
        processor_func: Callable,
        progress_callback: Optional[Callable] = None,
    ) -> List[Any]:
        """Process items in optimized batches."""
        results = []
        total_items = len(items)

        # Split items into batches
        batches = [
            items[i : i + self.batch_size]
            for i in range(0, len(items), self.batch_size)
        ]

        logger.info(
            f"Starting batch processing for {total_items} items in {len(batches)} batches"
        )

        for batch_idx, batch in enumerate(batches):
            async with self.semaphore:
                batch_results = await asyncio.gather(
                    *[processor_func(item) for item in batch], return_exceptions=True
                )

                results.extend(batch_results)

                if progress_callback:
                    progress = ((batch_idx + 1) / len(batches)) * 100
                    await progress_callback(progress, batch_idx + 1, len(batches))

                logger.debug(
                    f"Batch {batch_idx + 1}/{len(batches)} completed (size: {len(batch)})"
                )

        logger.info(
            f"Batch processing completed for {total_items} items, {len(results)} results produced"
        )

        return results


class _SingletonManager:
    """Manages singleton instances to avoid global statements."""

    def __init__(self):
        self._performance_monitor = None
        self._memory_optimizer = None
        self._batch_processor = None

    def get_performance_monitor(self) -> PerformanceMonitor:
        """Get performance monitor instance."""
        if self._performance_monitor is None:
            self._performance_monitor = PerformanceMonitor()
        return self._performance_monitor

    def get_memory_optimizer(self) -> MemoryOptimizer:
        """Get memory optimizer instance."""
        if self._memory_optimizer is None:
            self._memory_optimizer = MemoryOptimizer()
        return self._memory_optimizer

    def get_batch_processor(self) -> BatchProcessor:
        """Get batch processor instance."""
        if self._batch_processor is None:
            self._batch_processor = BatchProcessor()
        return self._batch_processor


# Module-level singleton manager
_singleton_manager = _SingletonManager()


def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor instance."""
    return _singleton_manager.get_performance_monitor()


def get_memory_optimizer() -> MemoryOptimizer:
    """Get global memory optimizer instance."""
    return _singleton_manager.get_memory_optimizer()


def get_batch_processor() -> BatchProcessor:
    """Get global batch processor instance."""
    return _singleton_manager.get_batch_processor()


# Decorator functions for easy performance monitoring
def monitor_performance(operation_name: str = None, **metadata):
    """Decorator for monitoring function performance."""

    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            name = operation_name or f"{func.__module__}.{func.__name__}"
            monitor = get_performance_monitor()

            async with monitor.measure_async(name, **metadata):
                return await func(*args, **kwargs)

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            name = operation_name or f"{func.__module__}.{func.__name__}"
            monitor = get_performance_monitor()

            with monitor.measure_sync(name, **metadata):
                return func(*args, **kwargs)

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def auto_memory_optimize(
    threshold_mb: float = MemoryConstants.AUTO_OPTIMIZE_THRESHOLD_MB,
):
    """Decorator for automatic memory optimization."""

    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            optimizer = get_memory_optimizer()

            try:
                result = await func(*args, **kwargs)
            finally:
                # Check memory after execution and optimize if needed
                memory_after = optimizer.get_memory_info()["rss_mb"]
                if memory_after > threshold_mb:
                    optimizer.auto_optimize()

            return result

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            optimizer = get_memory_optimizer()

            try:
                result = func(*args, **kwargs)
            finally:
                # Check memory after execution and optimize if needed
                memory_after = optimizer.get_memory_info()["rss_mb"]
                if memory_after > threshold_mb:
                    optimizer.auto_optimize()

            return result

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator
