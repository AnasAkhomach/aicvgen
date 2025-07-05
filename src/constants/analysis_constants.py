"""Analysis-related constants for CV and job matching.

This module contains constants used for CV analysis operations to eliminate
hardcoded values and improve maintainability.
"""

from typing import Final


class AnalysisConstants:
    """Constants for CV analysis and job matching operations."""

    # Experience relevance thresholds
    MIN_EXPERIENCE_RELEVANCE: Final[float] = 0.7
    GOOD_EXPERIENCE_RELEVANCE: Final[float] = 0.8
    EXCELLENT_EXPERIENCE_RELEVANCE: Final[float] = 0.9
    
    # Skill matching thresholds
    MIN_SKILL_MATCHES: Final[int] = 3
    GOOD_SKILL_MATCHES: Final[int] = 5
    EXCELLENT_SKILL_MATCHES: Final[int] = 8
    
    # Scoring weights and multipliers
    SKILL_SCORE_MULTIPLIER: Final[float] = 0.2
    GAP_PENALTY_MULTIPLIER: Final[float] = 0.1
    SCORE_WEIGHT_FACTOR: Final[float] = 0.5  # For averaging skill and experience scores
    
    # Match score bounds
    MIN_MATCH_SCORE: Final[float] = 0.0
    MAX_MATCH_SCORE: Final[float] = 1.0
    
    # Confidence score defaults
    DEFAULT_CONFIDENCE_SCORE: Final[float] = 0.85
    HIGH_CONFIDENCE_SCORE: Final[float] = 0.9
    LOW_CONFIDENCE_SCORE: Final[float] = 0.6