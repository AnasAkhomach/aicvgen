"""Metrics-related constants for Prometheus monitoring.

This module contains constants used for metrics collection and monitoring
to eliminate hardcoded values and improve maintainability.
"""

from typing import Final


class MetricsConstants:
    """Constants for metrics collection and Prometheus monitoring."""

    # Workflow duration buckets (seconds)
    WORKFLOW_DURATION_BUCKETS: Final[tuple] = (1, 5, 10, 30, 60, 120, 300, 600)

    # LLM request duration buckets (seconds)
    LLM_REQUEST_DURATION_BUCKETS: Final[tuple] = (0.1, 0.5, 1, 2, 5, 10, 30)

    # Agent execution duration buckets (seconds)
    AGENT_EXECUTION_DURATION_BUCKETS: Final[tuple] = (0.1, 0.5, 1, 5, 10, 30, 60)

    # Memory usage buckets (MB)
    MEMORY_USAGE_BUCKETS: Final[tuple] = (100, 250, 500, 1000, 2000, 4000)

    # CPU usage buckets (percentage)
    CPU_USAGE_BUCKETS: Final[tuple] = (10, 25, 50, 75, 90, 95, 99)

    # Error rate buckets (percentage)
    ERROR_RATE_BUCKETS: Final[tuple] = (0.1, 0.5, 1, 2, 5, 10, 25)

    # Queue size buckets
    QUEUE_SIZE_BUCKETS: Final[tuple] = (1, 5, 10, 25, 50, 100, 250)

    # Response time buckets (milliseconds)
    RESPONSE_TIME_BUCKETS: Final[tuple] = (10, 50, 100, 250, 500, 1000, 2500, 5000)
