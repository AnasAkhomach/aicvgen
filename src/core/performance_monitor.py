"""Performance monitoring and analysis for the CV generation system."""

import asyncio
import json
import os
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import psutil

from src.config.logging_config import get_structured_logger
from src.constants.performance_constants import PerformanceConstants
from src.core.performance_monitor.agent_lifecycle_manager import get_agent_lifecycle_manager

logger = get_structured_logger(__name__)


@dataclass
class PerformanceMetric:
    """Individual performance metric."""

    name: str
    value: float
    unit: str
    timestamp: datetime
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemSnapshot:
    """System resource snapshot."""

    timestamp: datetime
    cpu_percent: float
    memory_mb: float
    memory_percent: float
    active_threads: int
    open_files: int
    network_connections: int


@dataclass
class AgentPerformanceStats:
    """Performance statistics for agents."""

    agent_type: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    average_response_time: float
    min_response_time: float
    max_response_time: float
    active_instances: int
    pool_utilization: float
    last_activity: datetime


class PerformanceMonitor:
    """Monitors and analyzes system performance."""

    # Use centralized constants from PerformanceConstants
    # All hardcoded values moved to constants module

    def __init__(self, max_history_size: int = PerformanceConstants.DEFAULT_MAX_HISTORY_SIZE):
        self.max_history_size = max_history_size

        # Metrics storage
        self.metrics_history: deque = deque(maxlen=max_history_size)
        self.system_snapshots: deque = deque(maxlen=max_history_size)
        self.agent_stats: Dict[str, AgentPerformanceStats] = {}

        # Performance tracking
        self.operation_times: Dict[str, List[float]] = defaultdict(list)
        self.error_counts: Dict[str, int] = defaultdict(int)

        # Monitoring state
        self.monitoring_active = False
        self.monitoring_interval = PerformanceConstants.DEFAULT_MONITORING_INTERVAL_SECONDS
        self.monitoring_task: Optional[asyncio.Task] = None

        # Thresholds for alerts
        self.thresholds = {
            "cpu_percent": PerformanceConstants.CPU_THRESHOLD_PERCENT,
            "memory_percent": PerformanceConstants.MEMORY_THRESHOLD_PERCENT,
            "response_time_ms": PerformanceConstants.RESPONSE_TIME_THRESHOLD_MS,
            "error_rate_percent": PerformanceConstants.ERROR_RATE_THRESHOLD_PERCENT,
            "pool_utilization_percent": PerformanceConstants.POOL_UTILIZATION_THRESHOLD_PERCENT,
        }

        # Performance baselines
        self.baselines: Dict[str, float] = {}

    def start_monitoring(
        self, interval: int = PerformanceConstants.DEFAULT_MONITORING_INTERVAL_SECONDS
    ) -> None:
        """Start continuous performance monitoring."""
        if self.monitoring_active:
            logger.warning("Performance monitoring is already active")
            return

        self.monitoring_interval = interval
        self.monitoring_active = True

        # Start monitoring task
        try:
            loop = asyncio.get_running_loop()
            self.monitoring_task = loop.create_task(self._monitoring_loop())
        except RuntimeError:
            # No event loop running, monitoring will be manual
            logger.warning(
                "No running event loop found, performance monitoring disabled"
            )

        logger.info(f"Performance monitoring started with {interval}s interval")

    def stop_monitoring(self) -> None:
        """Stop performance monitoring."""
        self.monitoring_active = False

        if self.monitoring_task and not self.monitoring_task.done():
            self.monitoring_task.cancel()

        logger.info("Performance monitoring stopped")

    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                await self._collect_system_snapshot()
                await self._collect_agent_stats()
                await self._check_thresholds()

                await asyncio.sleep(self.monitoring_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in monitoring loop", exc_info=e)
                await asyncio.sleep(self.monitoring_interval)

    async def _collect_system_snapshot(self) -> None:
        """Collect system resource snapshot."""
        try:
            process = psutil.Process(os.getpid())

            snapshot = SystemSnapshot(
                timestamp=datetime.now(),
                cpu_percent=psutil.cpu_percent(interval=1),
                memory_mb=process.memory_info().rss / PerformanceConstants.MEMORY_CONVERSION_FACTOR,
                memory_percent=process.memory_percent(),
                active_threads=process.num_threads(),
                open_files=len(process.open_files()),
                network_connections=len(process.connections()),
            )

            self.system_snapshots.append(snapshot)

            # Record as metrics
            self.record_metric("cpu_percent", snapshot.cpu_percent, "%")
            self.record_metric("memory_mb", snapshot.memory_mb, "MB")
            self.record_metric("memory_percent", snapshot.memory_percent, "%")
            self.record_metric("active_threads", snapshot.active_threads, "count")

        except OSError as e:
            logger.warning("Failed to collect system snapshot", extra={"exc_info": e})

    async def _collect_agent_stats(self) -> None:
        """Collect agent performance statistics."""
        try:
            lifecycle_manager = get_agent_lifecycle_manager()
            stats = lifecycle_manager.get_metrics()

            for agent_id, agent_metrics in stats.get("agent_metrics", {}).items():
                agent_type = agent_id.split("_")[0]  # Extract agent type from agent_id
                total_requests = agent_metrics.get("usage_count", 0)
                successful_requests = (
                    total_requests  # Assuming all are successful for now
                )
                failed_requests = 0

                # These metrics are not directly available from AgentMetrics
                avg_response_time = 0.0
                min_response_time = 0.0
                max_response_time = 0.0
                active_instances = 0  # Not tracked by AgentMetrics
                pool_utilization = 0.0  # Not tracked by AgentMetrics

                self.agent_stats[agent_type] = AgentPerformanceStats(
                    agent_type=agent_type,
                    total_requests=total_requests,
                    successful_requests=successful_requests,
                    failed_requests=failed_requests,
                    average_response_time=avg_response_time,
                    min_response_time=min_response_time,
                    max_response_time=max_response_time,
                    active_instances=active_instances,
                    pool_utilization=pool_utilization,
                    last_activity=agent_metrics.get("last_used", datetime.now()),
                )

                # Record as metrics
                self.record_metric(
                    f"agent_{agent_type}_total_requests", total_requests, "count"
                )
                # Other metrics are not available from AgentMetrics currently

        except Exception as e:
            logger.warning("Failed to collect agent stats", extra={"exc_info": e})

    async def _check_thresholds(self) -> None:
        """Check performance thresholds and generate alerts."""
        try:
            if not self.system_snapshots:
                return

            latest_snapshot = self.system_snapshots[-1]
            alerts = []

            # Check system thresholds
            if latest_snapshot.cpu_percent > self.thresholds["cpu_percent"]:
                alerts.append(f"High CPU usage: {latest_snapshot.cpu_percent:.1f}%")

            if latest_snapshot.memory_percent > self.thresholds["memory_percent"]:
                alerts.append(
                    f"High memory usage: {latest_snapshot.memory_percent:.1f}%"
                )

            # Check agent thresholds
            for agent_type, stats in self.agent_stats.items():
                if stats.average_response_time > self.thresholds["response_time_ms"]:
                    alerts.append(
                        f"Slow response time for {agent_type}: "
                        f"{stats.average_response_time:.1f}ms"
                    )

                error_rate = (
                    (stats.failed_requests / stats.total_requests * 100)
                    if stats.total_requests > 0
                    else 0
                )
                if error_rate > self.thresholds["error_rate_percent"]:
                    alerts.append(
                        f"High error rate for {agent_type}: {error_rate:.1f}%"
                    )

                if stats.pool_utilization > self.thresholds["pool_utilization_percent"]:
                    alerts.append(
                        f"High pool utilization for {agent_type}: "
                        f"{stats.pool_utilization:.1f}%"
                    )

            # Log alerts
            for alert in alerts:
                logger.warning(f"Performance alert: {alert}")

        except Exception as e:
            logger.warning("Failed to check thresholds", extra={"exc_info": e})

    def record_metric(
        self,
        name: str,
        value: float,
        unit: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a performance metric."""
        metric = PerformanceMetric(
            name=name,
            value=value,
            unit=unit,
            timestamp=datetime.now(),
            context=context or {},
        )

        self.metrics_history.append(metric)

    def record_operation_time(self, operation: str, duration: float) -> None:
        """Record operation execution time."""
        self.operation_times[operation].append(duration)

        # Keep only recent measurements
        if len(self.operation_times[operation]) > PerformanceConstants.OPERATION_TIMES_MAX_SIZE:
            self.operation_times[operation] = self.operation_times[operation][
                -PerformanceConstants.OPERATION_TIMES_MAX_SIZE :
            ]

        self.record_metric(f"operation_{operation}_time", duration, "ms")

    def record_error(self, error_type: str) -> None:
        """Record an error occurrence."""
        self.error_counts[error_type] += 1
        self.record_metric(
            f"error_{error_type}_count", self.error_counts[error_type], "count"
        )

    def get_performance_summary(
        self, time_window: Optional[timedelta] = None
    ) -> Dict[str, Any]:
        """Get performance summary for the specified time window."""
        if time_window is None:
            time_window = timedelta(hours=1)

        cutoff_time = datetime.now() - time_window

        # Filter metrics by time window
        recent_metrics = [m for m in self.metrics_history if m.timestamp >= cutoff_time]
        recent_snapshots = [
            s for s in self.system_snapshots if s.timestamp >= cutoff_time
        ]

        if not recent_metrics:
            return {"error": "No metrics available for the specified time window"}

        # Calculate summary statistics
        summary = {
            "time_window": str(time_window),
            "metrics_count": len(recent_metrics),
            "system_performance": self._calculate_system_summary(recent_snapshots),
            "agent_performance": self._calculate_agent_summary(),
            "operation_performance": self._calculate_operation_summary(),
            "error_summary": dict(self.error_counts),
            "recommendations": self._generate_performance_recommendations(),
        }

        return summary

    def _calculate_system_summary(
        self, snapshots: List[SystemSnapshot]
    ) -> Dict[str, Any]:
        """Calculate system performance summary."""
        if not snapshots:
            return {}

        cpu_values = [s.cpu_percent for s in snapshots]
        memory_values = [s.memory_mb for s in snapshots]
        memory_percent_values = [s.memory_percent for s in snapshots]

        return {
            "cpu_usage": {
                "average": sum(cpu_values) / len(cpu_values),
                "min": min(cpu_values),
                "max": max(cpu_values),
                "current": cpu_values[-1] if cpu_values else 0,
            },
            "memory_usage_mb": {
                "average": sum(memory_values) / len(memory_values),
                "min": min(memory_values),
                "max": max(memory_values),
                "current": memory_values[-1] if memory_values else 0,
            },
            "memory_usage_percent": {
                "average": sum(memory_percent_values) / len(memory_percent_values),
                "min": min(memory_percent_values),
                "max": max(memory_percent_values),
                "current": memory_percent_values[-1] if memory_percent_values else 0,
            },
            "snapshots_analyzed": len(snapshots),
        }

    def _calculate_agent_summary(self) -> Dict[str, Any]:
        """Calculate agent performance summary."""
        if not self.agent_stats:
            return {}

        summary = {}

        for agent_type, stats in self.agent_stats.items():
            summary[agent_type] = {
                "total_requests": stats.total_requests,
                "success_rate": (
                    (stats.successful_requests / stats.total_requests * 100)
                    if stats.total_requests > 0
                    else 0
                ),
                "average_response_time_ms": stats.average_response_time,
                "pool_utilization_percent": stats.pool_utilization,
                "active_instances": stats.active_instances,
            }

        return summary

    def _calculate_operation_summary(self) -> Dict[str, Any]:
        """Calculate operation performance summary."""
        summary = {}

        for operation, times in self.operation_times.items():
            if times:
                summary[operation] = {
                    "average_time_ms": sum(times) / len(times),
                    "min_time_ms": min(times),
                    "max_time_ms": max(times),
                    "total_executions": len(times),
                }

        return summary

    def _generate_performance_recommendations(self) -> List[str]:
        """Generate performance optimization recommendations."""
        recommendations = []

        # Check recent system performance
        if self.system_snapshots:
            latest_snapshot = self.system_snapshots[-1]

            if latest_snapshot.cpu_percent > PerformanceConstants.RECOMMENDATION_CPU_THRESHOLD:
                recommendations.append(
                    "High CPU usage detected. Consider optimizing agent algorithms "
                    "or reducing concurrent operations."
                )

            if latest_snapshot.memory_percent > PerformanceConstants.RECOMMENDATION_MEMORY_THRESHOLD:
                recommendations.append(
                    "High memory usage detected. Consider implementing more "
                    "aggressive cleanup or reducing agent pool sizes."
                )

        # Check agent performance
        for agent_type, stats in self.agent_stats.items():
            if stats.average_response_time > PerformanceConstants.RECOMMENDATION_AGENT_RESPONSE_TIME_MS:
                recommendations.append(
                    f"Agent {agent_type} has slow response times. "
                    "Consider optimization or scaling."
                )

            if stats.pool_utilization > PerformanceConstants.RECOMMENDATION_POOL_UTILIZATION:
                recommendations.append(
                    f"Agent {agent_type} pool is highly utilized. "
                    "Consider increasing pool size."
                )

            error_rate = (
                (stats.failed_requests / stats.total_requests * 100)
                if stats.total_requests > 0
                else 0
            )
            if error_rate > PerformanceConstants.RECOMMENDATION_ERROR_RATE:
                recommendations.append(
                    f"Agent {agent_type} has high error rate ({error_rate:.1f}%). "
                    "Investigate error causes."
                )

        # Check operation performance
        for operation, times in self.operation_times.items():
            if times and len(times) > 10:
                avg_time = sum(times) / len(times)
                if avg_time > PerformanceConstants.RECOMMENDATION_OPERATION_AVG_TIME_MS:  # 2 seconds
                    recommendations.append(
                        f"Operation {operation} is slow (avg: {avg_time:.1f}ms). Consider optimization."
                    )

        if not recommendations:
            recommendations.append(
                "System performance looks good! No specific recommendations."
            )

        return recommendations

    def export_metrics(self, file_path: str, export_format: str = "json") -> None:
        """Export metrics to file."""
        try:
            data = {
                "export_timestamp": datetime.now().isoformat(),
                "metrics": [
                    {
                        "name": m.name,
                        "value": m.value,
                        "unit": m.unit,
                        "timestamp": m.timestamp.isoformat(),
                        "context": m.context,
                    }
                    for m in self.metrics_history
                ],
                "system_snapshots": [
                    {
                        "timestamp": s.timestamp.isoformat(),
                        "cpu_percent": s.cpu_percent,
                        "memory_mb": s.memory_mb,
                        "memory_percent": s.memory_percent,
                        "active_threads": s.active_threads,
                        "open_files": s.open_files,
                        "network_connections": s.network_connections,
                    }
                    for s in self.system_snapshots
                ],
                "agent_stats": {
                    agent_type: {
                        "agent_type": stats.agent_type,
                        "total_requests": stats.total_requests,
                        "successful_requests": stats.successful_requests,
                        "failed_requests": stats.failed_requests,
                        "average_response_time": stats.average_response_time,
                        "min_response_time": stats.min_response_time,
                        "max_response_time": stats.max_response_time,
                        "active_instances": stats.active_instances,
                        "pool_utilization": stats.pool_utilization,
                        "last_activity": stats.last_activity.isoformat(),
                    }
                    for agent_type, stats in self.agent_stats.items()
                },
            }

            if export_format.lower() == "json":
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
            else:
                raise ValueError(f"Unsupported export format: {export_format}")

            logger.info(f"Metrics exported to {file_path}")

        except Exception as e:
            logger.error("Failed to export metrics", extra={"exc_info": e})

    def set_baseline(self, metric_name: str, value: float) -> None:
        """Set performance baseline for comparison."""
        self.baselines[metric_name] = value
        logger.info(f"Baseline set for {metric_name}: {value}")

    def compare_to_baseline(self, metric_name: str) -> Optional[Dict[str, Any]]:
        """Compare current performance to baseline."""
        if metric_name not in self.baselines:
            return None

        # Get recent values for the metric
        recent_values = [
            m.value
            for m in self.metrics_history
            if m.name == metric_name
            and m.timestamp >= datetime.now() - timedelta(minutes=10)
        ]

        if not recent_values:
            return None

        current_avg = sum(recent_values) / len(recent_values)
        baseline = self.baselines[metric_name]

        improvement = (
            ((baseline - current_avg) / baseline * 100) if baseline != 0 else 0
        )

        return {
            "metric_name": metric_name,
            "baseline_value": baseline,
            "current_average": current_avg,
            "improvement_percent": improvement,
            "samples_analyzed": len(recent_values),
        }


# Context manager for operation timing
class OperationTimer:
    """Context manager for timing operations."""

    def __init__(self, monitor: PerformanceMonitor, operation_name: str):
        self.monitor = monitor
        self.operation_name = operation_name
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            duration = (time.time() - self.start_time) * 1000  # Convert to milliseconds
            self.monitor.record_operation_time(self.operation_name, duration)


# Global monitor instance
_global_monitor: Optional[PerformanceMonitor] = None
_monitor_lock = threading.Lock()


def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor."""
    global _global_monitor

    with _monitor_lock:
        if _global_monitor is None:
            _global_monitor = PerformanceMonitor()
        return _global_monitor


def reset_performance_monitor() -> None:
    """Reset the global monitor (mainly for testing)."""
    global _global_monitor

    with _monitor_lock:
        if _global_monitor and _global_monitor.monitoring_active:
            _global_monitor.stop_monitoring()
        _global_monitor = None


def time_operation(operation_name: str) -> OperationTimer:
    """Create an operation timer context manager."""
    monitor = get_performance_monitor()
    return OperationTimer(monitor, operation_name)
