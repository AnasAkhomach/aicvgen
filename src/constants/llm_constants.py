"""LLM-related constants for centralized configuration.

This module contains constants used for LLM operations to eliminate
hardcoded values and improve maintainability.
"""

from typing import Final


class LLMConstants:
    """Constants for LLM operations and configuration."""

    # Default LLM parameters
    DEFAULT_MAX_TOKENS: Final[int] = 1024
    DEFAULT_TEMPERATURE: Final[float] = 0.7
    DEFAULT_TOP_P: Final[float] = 1.0
    DEFAULT_TOP_K: Final[int] = 40

    # Token limits for different operations
    MAX_TOKENS_SUMMARY: Final[int] = 500
    MAX_TOKENS_ANALYSIS: Final[int] = 1500
    MAX_TOKENS_GENERATION: Final[int] = 2000
    MAX_TOKENS_CONVERSATION: Final[int] = 4000

    # Temperature settings for different use cases
    TEMPERATURE_CREATIVE: Final[float] = 0.9
    TEMPERATURE_BALANCED: Final[float] = 0.7
    TEMPERATURE_PRECISE: Final[float] = 0.3
    TEMPERATURE_DETERMINISTIC: Final[float] = 0.0

    # Model names
    DEFAULT_MODEL: Final[str] = "gemini-1.5-flash"
    FALLBACK_MODEL: Final[str] = "gemini-1.0-pro"

    # Request timeouts (seconds)
    DEFAULT_TIMEOUT: Final[float] = 30.0
    LONG_TIMEOUT: Final[float] = 120.0

    # Retry configuration
    MAX_RETRIES: Final[int] = 3
    RETRY_DELAY: Final[float] = 1.0
    EXPONENTIAL_BACKOFF_MULTIPLIER: Final[float] = 2.0

    # Token usage validation and limits
    MAX_TOKEN_USAGE_THRESHOLD: Final[int] = 10000
    MIN_TOKEN_USAGE: Final[int] = 0
    DEFAULT_TOKEN_ESTIMATE: Final[int] = 100
    TOKEN_PROCESSING_RATE: Final[float] = 1000.0  # tokens per second for estimation

    # Processing time defaults
    MIN_PROCESSING_TIME: Final[float] = 0.001  # 1ms minimum
    DEFAULT_PROCESSING_TIME_PER_TOKEN: Final[float] = 0.001  # 1ms per token

    # Rate limiting defaults
    DEFAULT_REQUESTS_PER_MINUTE: Final[int] = 30
    DEFAULT_TOKENS_PER_MINUTE: Final[int] = 50000
    RATE_LIMIT_WINDOW_SECONDS: Final[int] = 60

    # Backoff configuration
    BASE_BACKOFF_SECONDS: Final[float] = 1.0
    MAX_BACKOFF_SECONDS: Final[float] = 300.0
    BACKOFF_JITTER_ENABLED: Final[bool] = True

    # Response validation
    MIN_RESPONSE_LENGTH: Final[int] = 1
    MAX_RESPONSE_LENGTH: Final[int] = 100000

    # Cache configuration
    DEFAULT_CACHE_TTL_SECONDS: Final[int] = 3600  # 1 hour
    CACHE_KEY_MAX_LENGTH: Final[int] = 250

    # Error handling
    MAX_ERROR_MESSAGE_LENGTH: Final[int] = 1000
    FALLBACK_CONTENT_PREFIX: Final[str] = "[Fallback Content]"

    # Performance monitoring
    PERFORMANCE_LOG_INTERVAL: Final[int] = 100  # Log every 100 requests
    STATS_RESET_THRESHOLD: Final[int] = 10000  # Reset stats after 10k requests
