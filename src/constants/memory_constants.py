"""Memory management constants for centralized configuration.

This module contains constants used across memory management operations
to eliminate hardcoded values and improve maintainability.
"""

from typing import Final


class MemoryConstants:
    """Constants for memory management and optimization."""

    # Memory thresholds (in MB)
    MEMORY_WARNING_THRESHOLD: Final[int] = 512
    MEMORY_CRITICAL_THRESHOLD: Final[int] = 1024
    MEMORY_LOW_THRESHOLD: Final[int] = 256
    MEMORY_OPTIMIZATION_THRESHOLD: Final[int] = 100

    # Garbage collection configuration
    GC_THRESHOLD_MB: Final[int] = 100
    GC_INTERVAL_SECONDS: Final[int] = 300  # 5 minutes
    GC_FORCE_THRESHOLD_MB: Final[int] = 200
    GC_AUTO_ENABLE: Final[bool] = True

    # Memory conversion factors
    BYTES_TO_MB: Final[int] = 1024 * 1024
    BYTES_TO_KB: Final[int] = 1024
    MB_TO_BYTES: Final[int] = 1024 * 1024
    KB_TO_BYTES: Final[int] = 1024

    # Cache memory estimation
    CACHE_ENTRY_ESTIMATE_KB: Final[float] = 0.1  # 100KB per entry
    CACHE_OVERHEAD_FACTOR: Final[float] = 1.2  # 20% overhead
    CACHE_MEMORY_LIMIT_MB: Final[int] = 500

    # Memory monitoring intervals
    MEMORY_CHECK_INTERVAL_SECONDS: Final[int] = 60
    MEMORY_LOG_INTERVAL_SECONDS: Final[int] = 300
    MEMORY_ALERT_INTERVAL_SECONDS: Final[int] = 30

    # Memory optimization settings
    AUTO_OPTIMIZE_ENABLED: Final[bool] = True
    OPTIMIZE_ON_HIGH_USAGE: Final[bool] = True
    OPTIMIZE_BATCH_SIZE: Final[int] = 1000
    OPTIMIZE_MAX_ITERATIONS: Final[int] = 3

    # Memory usage categories
    USAGE_LOW_PERCENTAGE: Final[float] = 50.0
    USAGE_MEDIUM_PERCENTAGE: Final[float] = 70.0
    USAGE_HIGH_PERCENTAGE: Final[float] = 85.0
    USAGE_CRITICAL_PERCENTAGE: Final[float] = 95.0

    # Memory allocation limits
    MAX_ALLOCATION_MB: Final[int] = 2048
    MAX_SINGLE_OBJECT_MB: Final[int] = 100
    MAX_BUFFER_SIZE_MB: Final[int] = 50

    # Memory pool configuration
    POOL_INITIAL_SIZE: Final[int] = 10
    POOL_MAX_SIZE: Final[int] = 100
    POOL_GROWTH_FACTOR: Final[float] = 1.5
    POOL_SHRINK_THRESHOLD: Final[float] = 0.3

    # Memory profiling
    PROFILING_ENABLED: Final[bool] = False
    PROFILING_SAMPLE_RATE: Final[float] = 0.1
    PROFILING_MAX_SAMPLES: Final[int] = 1000

    # Memory leak detection
    LEAK_DETECTION_ENABLED: Final[bool] = True
    LEAK_DETECTION_THRESHOLD_MB: Final[int] = 50
    LEAK_DETECTION_WINDOW_MINUTES: Final[int] = 30

    # Memory pressure handling
    PRESSURE_LOW_THRESHOLD: Final[float] = 0.6
    PRESSURE_MEDIUM_THRESHOLD: Final[float] = 0.8
    PRESSURE_HIGH_THRESHOLD: Final[float] = 0.9

    # Memory cleanup strategies
    CLEANUP_STRATEGY_AGGRESSIVE: Final[str] = "aggressive"
    CLEANUP_STRATEGY_CONSERVATIVE: Final[str] = "conservative"
    CLEANUP_STRATEGY_BALANCED: Final[str] = "balanced"
    DEFAULT_CLEANUP_STRATEGY: Final[str] = CLEANUP_STRATEGY_BALANCED

    # Memory reporting
    REPORT_MEMORY_STATS: Final[bool] = True
    REPORT_INTERVAL_MINUTES: Final[int] = 15
    REPORT_DETAILED_STATS: Final[bool] = False

    # Memory error thresholds
    OUT_OF_MEMORY_THRESHOLD: Final[float] = 0.98
    MEMORY_FRAGMENTATION_THRESHOLD: Final[float] = 0.7

    # Memory optimization decorators
    AUTO_MEMORY_OPTIMIZE_THRESHOLD: Final[int] = 100
    MEMORY_DECORATOR_ENABLED: Final[bool] = True

    # Memory metrics
    METRICS_COLLECTION_ENABLED: Final[bool] = True
    METRICS_RETENTION_HOURS: Final[int] = 24
    METRICS_AGGREGATION_INTERVAL: Final[int] = 60

    # Memory warning messages
    MSG_MEMORY_WARNING: Final[
        str
    ] = "Memory usage is high: {usage_mb}MB ({percentage}%)"
    MSG_MEMORY_CRITICAL: Final[
        str
    ] = "Critical memory usage: {usage_mb}MB ({percentage}%)"
    MSG_MEMORY_OPTIMIZED: Final[str] = "Memory optimized: freed {freed_mb}MB"
    MSG_GC_TRIGGERED: Final[
        str
    ] = "Garbage collection triggered: collected {objects} objects"
    MSG_MEMORY_LEAK_DETECTED: Final[
        str
    ] = "Potential memory leak detected: {growth_mb}MB growth"

    # Memory allocation strategies
    ALLOCATION_STRATEGY_EAGER: Final[str] = "eager"
    ALLOCATION_STRATEGY_LAZY: Final[str] = "lazy"
    ALLOCATION_STRATEGY_ADAPTIVE: Final[str] = "adaptive"
    DEFAULT_ALLOCATION_STRATEGY: Final[str] = ALLOCATION_STRATEGY_ADAPTIVE

    # Memory debugging
    DEBUG_MEMORY_ENABLED: Final[bool] = False
    DEBUG_TRACK_ALLOCATIONS: Final[bool] = False
    DEBUG_LOG_LARGE_ALLOCATIONS: Final[bool] = True
    DEBUG_LARGE_ALLOCATION_THRESHOLD: Final[int] = 10  # MB

    # Batch processing constants (memory-related)
    DEFAULT_BATCH_SIZE: Final[int] = 10
    MAX_CONCURRENT_OPERATIONS: Final[int] = 5

    # Auto-optimization threshold
    AUTO_OPTIMIZE_THRESHOLD_MB: Final[int] = 100
