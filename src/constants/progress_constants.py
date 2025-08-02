"""Progress tracking constants for centralized configuration.

This module contains constants used across progress tracking operations
to eliminate hardcoded values and improve consistency.
"""

from typing import Final


class ProgressConstants:
    """Constants for progress tracking and status reporting."""

    # Base progress values
    PROGRESS_START: Final[int] = 0
    PROGRESS_COMPLETE: Final[int] = 100
    PROGRESS_HALFWAY: Final[int] = 50
    PROGRESS_QUARTER: Final[int] = 25
    PROGRESS_THREE_QUARTERS: Final[int] = 75

    # Common progress milestones
    PROGRESS_INITIALIZATION: Final[int] = 5
    PROGRESS_INPUT_VALIDATION: Final[int] = 15
    PROGRESS_PREPROCESSING: Final[int] = 25
    PROGRESS_MAIN_PROCESSING: Final[int] = 60
    PROGRESS_POST_PROCESSING: Final[int] = 80
    PROGRESS_VALIDATION: Final[int] = 90
    PROGRESS_FINALIZATION: Final[int] = 95
    PROGRESS_NEAR_COMPLETE: Final[int] = 85

    # Agent-specific progress stages
    # Parsing Agent
    PROGRESS_PARSING_START: Final[int] = 10
    PROGRESS_PARSING_STRUCTURE: Final[int] = 30
    PROGRESS_PARSING_CONTENT: Final[int] = 50
    PROGRESS_PARSING_VALIDATION: Final[int] = 70
    PROGRESS_PARSING_COMPLETE: Final[int] = 65

    # Vector Storage Agent
    PROGRESS_VECTOR_INIT: Final[int] = 10
    PROGRESS_VECTOR_EMBEDDING: Final[int] = 40
    PROGRESS_VECTOR_INDEXING: Final[int] = 70
    PROGRESS_VECTOR_STORAGE: Final[int] = 85

    # Quality Assurance Agent
    PROGRESS_QA_INIT: Final[int] = 10
    PROGRESS_QA_SECTION_START: Final[int] = 20
    PROGRESS_SECTION_CHECKS: Final[int] = 65
    PROGRESS_OVERALL_CHECKS: Final[int] = 85
    PROGRESS_QA_COMPLETE: Final[int] = 95

    # Generation Agent
    PROGRESS_GENERATION_INIT: Final[int] = 10
    PROGRESS_CONTENT_GENERATION: Final[int] = 40
    PROGRESS_FORMATTING: Final[int] = 70
    PROGRESS_GENERATION_COMPLETE: Final[int] = 90

    # Output Agent
    PROGRESS_OUTPUT_INIT: Final[int] = 10
    PROGRESS_HTML_GENERATION: Final[int] = 40
    PROGRESS_PDF_GENERATION: Final[int] = 70
    PROGRESS_FILE_SAVING: Final[int] = 90
    PROGRESS_OUTPUT_COMPLETE: Final[int] = 95

    # Cleaning Agent
    PROGRESS_CLEANING_INIT: Final[int] = 10
    PROGRESS_CLEANING_ANALYSIS: Final[int] = 30
    PROGRESS_CLEANING_PROCESSING: Final[int] = 60
    PROGRESS_CLEANING_COMPLETE: Final[int] = 65

    # LLM Processing
    PROGRESS_LLM_INIT: Final[int] = 10
    PROGRESS_LLM_REQUEST: Final[int] = 30
    PROGRESS_LLM_PROCESSING: Final[int] = 60
    PROGRESS_LLM_PARSING: Final[int] = 80
    PROGRESS_LLM_COMPLETE: Final[int] = 90

    # Progress update intervals
    PROGRESS_UPDATE_INTERVAL_MS: Final[int] = 500
    PROGRESS_BATCH_UPDATE_SIZE: Final[int] = 5
    PROGRESS_THROTTLE_THRESHOLD: Final[int] = 2  # Minimum percentage change to report

    # Progress status messages
    MSG_STARTING: Final[str] = "Starting {operation}"
    MSG_PROCESSING: Final[str] = "Processing {operation}"
    MSG_COMPLETING: Final[str] = "Completing {operation}"
    MSG_FINISHED: Final[str] = "Finished {operation}"
    MSG_FAILED: Final[str] = "Failed {operation}: {error}"
    MSG_RETRYING: Final[str] = "Retrying {operation} (attempt {attempt}/{max_attempts})"

    # Progress tracking configuration
    ENABLE_PROGRESS_LOGGING: Final[bool] = True
    ENABLE_PROGRESS_PERSISTENCE: Final[bool] = False
    PROGRESS_LOG_LEVEL: Final[str] = "INFO"
    PROGRESS_PRECISION_DECIMALS: Final[int] = 1

    # Progress validation
    MIN_PROGRESS_VALUE: Final[int] = 0
    MAX_PROGRESS_VALUE: Final[int] = 100
    PROGRESS_STEP_VALIDATION: Final[bool] = True

    # Progress categories
    CATEGORY_INITIALIZATION: Final[str] = "initialization"
    CATEGORY_PROCESSING: Final[str] = "processing"
    CATEGORY_VALIDATION: Final[str] = "validation"
    CATEGORY_OUTPUT: Final[str] = "output"
    CATEGORY_CLEANUP: Final[str] = "cleanup"

    # Progress event types
    EVENT_PROGRESS_START: Final[str] = "progress_start"
    EVENT_PROGRESS_UPDATE: Final[str] = "progress_update"
    EVENT_PROGRESS_COMPLETE: Final[str] = "progress_complete"
    EVENT_PROGRESS_ERROR: Final[str] = "progress_error"
    EVENT_PROGRESS_RESET: Final[str] = "progress_reset"

    # Progress calculation helpers
    PROGRESS_WEIGHT_HIGH: Final[float] = 1.0
    PROGRESS_WEIGHT_MEDIUM: Final[float] = 0.7
    PROGRESS_WEIGHT_LOW: Final[float] = 0.3

    # Progress estimation
    ESTIMATED_DURATION_PARSING: Final[int] = 30  # seconds
    ESTIMATED_DURATION_GENERATION: Final[int] = 60  # seconds
    ESTIMATED_DURATION_QA: Final[int] = 20  # seconds
    ESTIMATED_DURATION_OUTPUT: Final[int] = 15  # seconds
    ESTIMATED_DURATION_TOTAL: Final[int] = 125  # seconds
