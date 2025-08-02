"""Quality Assurance constants for centralized configuration.

This module contains constants used across quality assurance operations
to eliminate hardcoded values and improve maintainability.
"""

from typing import Final


class QAConstants:
    """Constants for quality assurance operations and configuration."""

    # Executive Summary validation
    MIN_WORD_COUNT_EXECUTIVE_SUMMARY: Final[int] = 50
    MAX_WORD_COUNT_EXECUTIVE_SUMMARY: Final[int] = 150
    EXECUTIVE_SUMMARY_WORD_COUNT_RANGE: Final[int] = 25
    RECOMMENDED_EXECUTIVE_SUMMARY_MIN: Final[int] = 75
    RECOMMENDED_EXECUTIVE_SUMMARY_MAX: Final[int] = 125

    # Section validation
    MIN_WORD_COUNT_SECTION: Final[int] = 20
    MAX_WORD_COUNT_SECTION: Final[int] = 500
    MIN_ITEMS_PER_SECTION: Final[int] = 1
    MAX_ITEMS_PER_SECTION: Final[int] = 10

    # Content quality thresholds
    MIN_CONTENT_LENGTH: Final[int] = 10
    MAX_CONTENT_LENGTH: Final[int] = 1000
    MIN_DESCRIPTION_LENGTH: Final[int] = 20
    MAX_DESCRIPTION_LENGTH: Final[int] = 300

    # Experience validation
    MIN_EXPERIENCE_YEARS: Final[int] = 0
    MAX_EXPERIENCE_YEARS: Final[int] = 50
    MIN_JOB_DURATION_MONTHS: Final[int] = 1
    MAX_JOB_DURATION_MONTHS: Final[int] = 600  # 50 years

    # Skills validation
    MIN_SKILLS_COUNT: Final[int] = 3
    MAX_SKILLS_COUNT: Final[int] = 20
    MIN_SKILL_NAME_LENGTH: Final[int] = 2
    MAX_SKILL_NAME_LENGTH: Final[int] = 50

    # Education validation
    MIN_EDUCATION_ENTRIES: Final[int] = 1
    MAX_EDUCATION_ENTRIES: Final[int] = 5
    MIN_DEGREE_LENGTH: Final[int] = 5
    MAX_DEGREE_LENGTH: Final[int] = 100

    # Contact information validation
    MIN_EMAIL_LENGTH: Final[int] = 5
    MAX_EMAIL_LENGTH: Final[int] = 100
    MIN_PHONE_LENGTH: Final[int] = 10
    MAX_PHONE_LENGTH: Final[int] = 20
    MIN_NAME_LENGTH: Final[int] = 2
    MAX_NAME_LENGTH: Final[int] = 100

    # Quality score thresholds
    EXCELLENT_QUALITY_THRESHOLD: Final[float] = 0.9
    GOOD_QUALITY_THRESHOLD: Final[float] = 0.7
    ACCEPTABLE_QUALITY_THRESHOLD: Final[float] = 0.5
    POOR_QUALITY_THRESHOLD: Final[float] = 0.3

    # Content analysis patterns
    BULLET_POINT_PATTERNS: Final[list] = ["•", "*", "-", "◦"]
    FORBIDDEN_WORDS: Final[list] = ["TODO", "FIXME", "XXX", "PLACEHOLDER"]
    REQUIRED_SECTIONS: Final[list] = ["Executive Summary", "Experience", "Skills"]

    # Formatting validation
    MAX_LINE_LENGTH: Final[int] = 120
    MIN_PARAGRAPH_SENTENCES: Final[int] = 1
    MAX_PARAGRAPH_SENTENCES: Final[int] = 5

    # Date validation
    MIN_YEAR: Final[int] = 1950
    MAX_YEAR: Final[int] = 2030

    # Language and grammar
    MIN_READABILITY_SCORE: Final[float] = 30.0  # Flesch Reading Ease
    MAX_READABILITY_SCORE: Final[float] = 100.0
    MAX_PASSIVE_VOICE_PERCENTAGE: Final[float] = 20.0

    # Performance thresholds
    QA_PROCESSING_TIMEOUT_SECONDS: Final[int] = 30
    MAX_QA_CHECKS_PER_ITEM: Final[int] = 10

    # Error messages
    ERROR_MISSING_SECTION: Final[str] = "Required section '{section_name}' is missing"
    ERROR_WORD_COUNT_LOW: Final[
        str
    ] = "Word count too low: {actual} words (minimum: {minimum})"
    ERROR_WORD_COUNT_HIGH: Final[
        str
    ] = "Word count too high: {actual} words (maximum: {maximum})"
    ERROR_INVALID_FORMAT: Final[str] = "Invalid format detected in {field_name}"
    ERROR_FORBIDDEN_CONTENT: Final[str] = "Forbidden content detected: {content}"

    # Success messages
    SUCCESS_QUALITY_CHECK: Final[str] = "Quality check passed for {item_type}"
    SUCCESS_SECTION_VALIDATION: Final[
        str
    ] = "Section '{section_name}' validation passed"
    SUCCESS_OVERALL_VALIDATION: Final[
        str
    ] = "Overall CV validation completed successfully"
