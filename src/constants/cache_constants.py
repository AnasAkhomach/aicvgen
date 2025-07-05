"""Cache-related constants for centralized configuration.

This module contains constants used for caching operations to eliminate
hardcoded values and improve maintainability.
"""

from typing import Final


class CacheConstants:
    """Constants for cache operations and configuration."""

    # Cache TTL (Time To Live) in seconds
    DEFAULT_TTL: Final[int] = 3600  # 1 hour
    SHORT_TTL: Final[int] = 300  # 5 minutes
    LONG_TTL: Final[int] = 86400  # 24 hours
    PERMANENT_TTL: Final[int] = -1  # Never expires

    # Cache size limits
    DEFAULT_MAX_SIZE: Final[int] = 1000
    SMALL_CACHE_SIZE: Final[int] = 100
    LARGE_CACHE_SIZE: Final[int] = 10000

    # Cache thresholds
    EVICTION_THRESHOLD: Final[float] = 0.8  # 80% full
    WARNING_THRESHOLD: Final[float] = 0.7  # 70% full

    # Cache key prefixes
    LLM_CACHE_PREFIX: Final[str] = "llm:"
    AGENT_CACHE_PREFIX: Final[str] = "agent:"
    SESSION_CACHE_PREFIX: Final[str] = "session:"
    VECTOR_CACHE_PREFIX: Final[str] = "vector:"

    # Cache operation timeouts (seconds)
    GET_TIMEOUT: Final[float] = 1.0
    SET_TIMEOUT: Final[float] = 2.0
    DELETE_TIMEOUT: Final[float] = 1.0

    # Cache statistics intervals
    STATS_COLLECTION_INTERVAL: Final[int] = 60  # 1 minute
    CLEANUP_INTERVAL: Final[int] = 300  # 5 minutes

    # Cache hit ratio thresholds
    GOOD_HIT_RATIO: Final[float] = 0.8
    ACCEPTABLE_HIT_RATIO: Final[float] = 0.6
    POOR_HIT_RATIO: Final[float] = 0.4

    # Access pattern limits
    MAX_ACCESS_PATTERNS: Final[int] = 100
    MAX_PATTERN_HISTORY: Final[int] = 100

    # Memory conversion factors
    MEMORY_CONVERSION_FACTOR: Final[int] = 1024 * 1024  # MB to bytes

    # Time intervals (seconds)
    TEMPORAL_WINDOW_SECONDS: Final[float] = 3600.0  # 1 hour

    # Size thresholds (bytes)
    SMALL_ITEM_THRESHOLD: Final[int] = 10240  # 10KB
    MEDIUM_ITEM_THRESHOLD: Final[int] = 102400  # 100KB
    LARGE_ITEM_THRESHOLD: Final[int] = 1048576  # 1MB

    # Similarity and scoring thresholds
    SIMILARITY_THRESHOLD: Final[float] = 0.3
    TEMPORAL_WEIGHT: Final[float] = 0.6
    SPATIAL_WEIGHT: Final[float] = 0.4

    # Default cache configuration
    DEFAULT_MAX_MEMORY_MB: Final[int] = 500
    DEFAULT_MAX_PREDICTIONS: Final[int] = 1000
    DEFAULT_TTL_HOURS: Final[int] = 1
    DEFAULT_PRIORITY: Final[int] = 1

    # Cache level thresholds
    HIGH_PRIORITY_THRESHOLD: Final[int] = 4

    # Memory allocation percentages for cache levels
    L1_MEMORY_PERCENTAGE: Final[float] = 0.3  # 30% of total memory for L1
    L2_MEMORY_PERCENTAGE: Final[float] = 0.5  # 50% of total memory for L2
    L3_MEMORY_PERCENTAGE: Final[float] = 0.2  # 20% of total memory for L3

    # Maintenance intervals (seconds)
    MAINTENANCE_INTERVAL: Final[int] = 300  # 5 minutes
    WARMING_INTERVAL: Final[int] = 60  # 1 minute

    # Eviction configuration
    EVICTION_BATCH_SIZE_RATIO: Final[float] = 0.25  # Evict 25% of entries when needed

    # Exponential moving average alpha for access time
    ACCESS_TIME_EMA_ALPHA: Final[float] = 0.1
