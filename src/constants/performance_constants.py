"""Performance-related constants for centralized configuration.

This module contains constants used for performance monitoring and optimization
to eliminate hardcoded values and improve maintainability.
"""

from typing import Final


class PerformanceConstants:
    """Constants for performance monitoring and optimization."""

    # Memory thresholds (in MB)
    MEMORY_WARNING_THRESHOLD: Final[int] = 512
    MEMORY_CRITICAL_THRESHOLD: Final[int] = 1024
    MEMORY_MAX_THRESHOLD: Final[int] = 2048

    # Performance monitoring intervals (seconds)
    MONITORING_INTERVAL: Final[float] = 5.0
    METRICS_COLLECTION_INTERVAL: Final[float] = 10.0
    HEALTH_CHECK_INTERVAL: Final[float] = 30.0
    
    # PerformanceMonitor specific constants
    DEFAULT_MAX_HISTORY_SIZE: Final[int] = 1000
    DEFAULT_MONITORING_INTERVAL_SECONDS: Final[int] = 30
    OPERATION_TIMES_MAX_SIZE: Final[int] = 100
    MEMORY_CONVERSION_FACTOR: Final[int] = 1024 * 1024
    MILLISECONDS_PER_SECOND: Final[int] = 1000
    
    # System resource alert thresholds
    CPU_THRESHOLD_PERCENT: Final[float] = 80.0
    MEMORY_THRESHOLD_PERCENT: Final[float] = 85.0
    RESPONSE_TIME_THRESHOLD_MS: Final[float] = 5000.0
    ERROR_RATE_THRESHOLD_PERCENT: Final[float] = 5.0
    POOL_UTILIZATION_THRESHOLD_PERCENT: Final[float] = 90.0
    
    # Recommendation thresholds (lower than alert thresholds)
    RECOMMENDATION_CPU_THRESHOLD: Final[int] = 70
    RECOMMENDATION_MEMORY_THRESHOLD: Final[int] = 75
    RECOMMENDATION_AGENT_RESPONSE_TIME_MS: Final[int] = 3000
    RECOMMENDATION_POOL_UTILIZATION: Final[int] = 80
    RECOMMENDATION_ERROR_RATE: Final[int] = 3
    RECOMMENDATION_OPERATION_AVG_TIME_MS: Final[int] = 2000
    RECOMMENDATION_RECENT_SAMPLES_MINUTES: Final[int] = 10

    # Batch processing limits
    DEFAULT_BATCH_SIZE: Final[int] = 10
    MAX_BATCH_SIZE: Final[int] = 100
    MIN_BATCH_SIZE: Final[int] = 1

    # Timeout configurations (seconds)
    OPERATION_TIMEOUT: Final[float] = 60.0
    BATCH_TIMEOUT: Final[float] = 300.0
    HEALTH_CHECK_TIMEOUT: Final[float] = 10.0

    # Resource limits
    MAX_CONCURRENT_OPERATIONS: Final[int] = 5
    MAX_QUEUE_SIZE: Final[int] = 100

    # Performance optimization thresholds
    CPU_USAGE_THRESHOLD: Final[float] = 80.0
    RESPONSE_TIME_THRESHOLD: Final[float] = 2.0
    ERROR_RATE_THRESHOLD: Final[float] = 0.05

    # Metrics buckets for histogram data
    LATENCY_BUCKETS: Final[tuple] = (0.1, 0.5, 1.0, 2.0, 5.0, 10.0)
    MEMORY_BUCKETS: Final[tuple] = (100, 250, 500, 1000, 2000, 4000)
    
    # Async optimization constants
    DEFAULT_MAX_CONCURRENT_OPERATIONS: Final[int] = 50
    DEFAULT_MAX_CONCURRENT_PER_TYPE: Final[int] = 10
    DEFAULT_SCALING_FACTOR: Final[float] = 1.2
    DEFAULT_DEADLOCK_TIMEOUT: Final[float] = 30.0
    DEFAULT_POOL_SIZE: Final[int] = 20
    DEFAULT_MAX_IDLE_TIME: Final[float] = 300.0
    DEFAULT_CLEANUP_INTERVAL: Final[float] = 60.0
    DEFAULT_WARMUP_SIZE: Final[int] = 5
    DEADLOCK_DETECTION_INTERVAL: Final[int] = 5
    
    # AdaptiveSemaphore constants
    MIN_ADJUSTMENT_INTERVAL: Final[int] = 10  # seconds
    WAIT_TIME_THRESHOLD: Final[float] = 0.1
    ERROR_RATE_THRESHOLD: Final[float] = 0.05
    HIGH_ERROR_RATE_THRESHOLD: Final[float] = 0.1
    PERFORMANCE_IMPROVEMENT_THRESHOLD: Final[float] = 0.01
    MAX_OPERATION_TIMES_HISTORY: Final[int] = 100
    MAX_WAIT_TIMES_HISTORY: Final[int] = 100
    
    # Scaling thresholds
    HIGH_LOAD_THRESHOLD: Final[float] = 0.8
    LOW_LOAD_THRESHOLD: Final[float] = 0.3
    CAPACITY_ADJUSTMENT_TIMEOUT: Final[float] = 1.0
    
    # Performance operation thresholds (seconds)
    LLM_CALL_THRESHOLD: Final[float] = 30.0
    AGENT_EXECUTION_THRESHOLD: Final[float] = 60.0
    FILE_PROCESSING_THRESHOLD: Final[float] = 10.0
    DATABASE_OPERATION_THRESHOLD: Final[float] = 5.0
    DEFAULT_OPERATION_THRESHOLD: Final[float] = 15.0
