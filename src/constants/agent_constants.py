"""Agent-related constants for centralized configuration.

This module contains constants used across agent implementations to eliminate
hardcoded values and improve maintainability.
"""

from typing import Final
from .progress_constants import ProgressConstants


class AgentConstants:
    """Constants for agent operations and configuration."""

    # Progress tracking constants (imported from ProgressConstants)
    PROGRESS_COMPLETE: Final[int] = ProgressConstants.PROGRESS_COMPLETE
    PROGRESS_START: Final[int] = ProgressConstants.PROGRESS_START
    PROGRESS_HALFWAY: Final[int] = ProgressConstants.PROGRESS_HALFWAY
    
    # Common progress milestones
    PROGRESS_INPUT_VALIDATION: Final[int] = ProgressConstants.PROGRESS_INPUT_VALIDATION
    PROGRESS_PROCESSING: Final[int] = ProgressConstants.PROGRESS_MAIN_PROCESSING
    PROGRESS_NEAR_COMPLETE: Final[int] = ProgressConstants.PROGRESS_NEAR_COMPLETE
    
    # Agent-specific progress stages
    PROGRESS_PARSING_COMPLETE: Final[int] = ProgressConstants.PROGRESS_PARSING_COMPLETE
    PROGRESS_VECTOR_STORAGE: Final[int] = ProgressConstants.PROGRESS_VECTOR_STORAGE
    PROGRESS_SECTION_CHECKS: Final[int] = ProgressConstants.PROGRESS_SECTION_CHECKS
    PROGRESS_OVERALL_CHECKS: Final[int] = ProgressConstants.PROGRESS_OVERALL_CHECKS
    PROGRESS_PDF_GENERATION: Final[int] = ProgressConstants.PROGRESS_PDF_GENERATION
    PROGRESS_HTML_GENERATION: Final[int] = ProgressConstants.PROGRESS_HTML_GENERATION
    PROGRESS_CLEANING_COMPLETE: Final[int] = ProgressConstants.PROGRESS_CLEANING_COMPLETE
    PROGRESS_LLM_PARSING: Final[int] = ProgressConstants.PROGRESS_LLM_PARSING
    PROGRESS_MAIN_PROCESSING: Final[int] = ProgressConstants.PROGRESS_MAIN_PROCESSING
    PROGRESS_POST_PROCESSING: Final[int] = ProgressConstants.PROGRESS_POST_PROCESSING
    PROGRESS_PREPROCESSING: Final[int] = ProgressConstants.PROGRESS_PREPROCESSING

    # Quality assurance thresholds
    MIN_WORD_COUNT_EXECUTIVE_SUMMARY: Final[int] = 50
    EXECUTIVE_SUMMARY_WORD_COUNT_RANGE: Final[int] = 25
    MIN_WORD_COUNT_SECTION: Final[int] = 20

    # Analysis thresholds
    EXPERIENCE_RELEVANCE_THRESHOLD: Final[float] = 0.7
    SKILL_MATCH_THRESHOLD: Final[float] = 0.6
    CONTENT_QUALITY_THRESHOLD: Final[float] = 0.8

    # Agent execution timeouts (seconds)
    DEFAULT_AGENT_TIMEOUT: Final[float] = 300.0
    QUICK_AGENT_TIMEOUT: Final[float] = 60.0
    LONG_AGENT_TIMEOUT: Final[float] = 600.0

    # Retry configuration
    DEFAULT_MAX_RETRIES: Final[int] = 3
    RETRY_DELAY_SECONDS: Final[float] = 1.0

    # Content validation
    MIN_CONTENT_LENGTH: Final[int] = 10
    MAX_CONTENT_LENGTH: Final[int] = 10000

    # Confidence scores
    DEFAULT_CONFIDENCE_SCORE: Final[float] = 0.85
    HIGH_CONFIDENCE_SCORE: Final[float] = 0.9
    LOW_CONFIDENCE_SCORE: Final[float] = 0.6

    # Agent status codes
    STATUS_SUCCESS: Final[str] = "success"
    STATUS_FAILURE: Final[str] = "failure"
    STATUS_PARTIAL: Final[str] = "partial"
    STATUS_TIMEOUT: Final[str] = "timeout"
