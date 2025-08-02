"""Configuration-related constants for centralized configuration.

This module contains constants used for application configuration
to eliminate hardcoded values and improve maintainability.
"""

from typing import Final, List


class ConfigConstants:
    """Constants for application configuration settings."""

    # LLM Settings
    DEFAULT_MODEL: Final[str] = "gemini-2.0-flash"
    DEFAULT_TEMPERATURE: Final[float] = 0.7
    DEFAULT_MAX_TOKENS: Final[int] = 4096
    ANALYSIS_TEMPERATURE: Final[float] = 0.5
    ANALYSIS_MAX_TOKENS: Final[int] = 2048
    CLEANING_MODEL: Final[str] = "llama-3.3-70b-versatile"

    # Agent Settings
    DEFAULT_SKILLS: Final[List[str]] = [
        "Problem Solving",
        "Team Collaboration",
        "Communication Skills",
        "Analytical Thinking",
        "Project Management",
        "Technical Documentation",
        "Quality Assurance",
        "Process Improvement",
        "Leadership",
        "Adaptability",
    ]
    MAX_SKILLS_TO_PARSE: Final[int] = 10
    MAX_BULLET_POINTS_PER_ROLE: Final[int] = 5
    MAX_BULLET_POINTS_PER_PROJECT: Final[int] = 3
    DEFAULT_COMPANY_NAME: Final[str] = "Unknown Company"
    DEFAULT_JOB_TITLE: Final[str] = "Unknown Title"

    # Rate Limiting
    DEFAULT_REQUESTS_PER_MINUTE: Final[int] = 30
    DEFAULT_TOKENS_PER_MINUTE: Final[int] = 60000

    # Retry Configuration
    DEFAULT_MAX_RETRIES: Final[int] = 3
    DEFAULT_RETRY_DELAY: Final[float] = 1.0
    DEFAULT_REQUEST_TIMEOUT: Final[int] = 60
    LLM_RETRY_MAX_ATTEMPTS: Final[int] = 5
    LLM_RETRY_MULTIPLIER: Final[int] = 1
    LLM_RETRY_MIN_WAIT: Final[int] = 2
    LLM_RETRY_MAX_WAIT: Final[int] = 60

    # Vector Database
    DEFAULT_PERSIST_DIRECTORY: Final[str] = "instance/vector_db"
    DEFAULT_COLLECTION_NAME: Final[str] = "cv_content"
    DEFAULT_EMBEDDING_MODEL: Final[str] = "all-MiniLM-L6-v2"
    DEFAULT_EMBEDDING_DIMENSION: Final[int] = 384
    DEFAULT_MAX_SEARCH_RESULTS: Final[int] = 10
    DEFAULT_SEARCH_LIMIT: Final[int] = 5
    DEFAULT_SIMILAR_CONTENT_LIMIT: Final[int] = 3
    DEFAULT_SESSION_LIST_LIMIT: Final[int] = 50
    DEFAULT_SIMILARITY_THRESHOLD: Final[float] = 0.7
    DEFAULT_VECTOR_STORE_TIMEOUT: Final[int] = 30
    DEFAULT_RECENT_EVENTS_LIMIT: Final[int] = 10

    # UI Configuration
    DEFAULT_PAGE_TITLE: Final[str] = "AI CV Generator"
    DEFAULT_PAGE_ICON: Final[str] = "📄"
    DEFAULT_LAYOUT: Final[str] = "wide"
    DEFAULT_SESSION_TIMEOUT: Final[int] = 3600  # 1 hour
    DEFAULT_AUTO_SAVE_INTERVAL: Final[int] = 30  # seconds
    DEFAULT_ITEMS_PER_PAGE: Final[int] = 5

    # Session Management
    DEFAULT_MAX_ACTIVE_SESSIONS: Final[int] = 100
    DEFAULT_CLEANUP_INTERVAL_MINUTES: Final[int] = 30

    # Output Configuration
    DEFAULT_PRIMARY_FORMAT: Final[str] = "pdf"
    DEFAULT_SUPPORTED_FORMATS: Final[List[str]] = ["pdf", "markdown", "html"]
    DEFAULT_PDF_TEMPLATE_PATH: Final[str] = "src/templates/cv_template.md"
    DEFAULT_PDF_OUTPUT_DIRECTORY: Final[str] = "instance/output"
    DEFAULT_MAX_SKILLS_COUNT: Final[int] = 10
    DEFAULT_MIN_SKILL_LENGTH: Final[int] = 2

    # Logging Configuration
    DEFAULT_LOG_LEVEL: Final[str] = "INFO"
    DEFAULT_LOG_DIRECTORY: Final[str] = "instance/logs"
    DEFAULT_MAIN_LOG_FILE: Final[str] = "app.log"
    DEFAULT_ERROR_LOG_FILE: Final[str] = "error.log"
    DEFAULT_LLM_LOG_FILE: Final[str] = "llm_calls.log"
    DEFAULT_MAX_LOG_SIZE_MB: Final[int] = 10
    DEFAULT_BACKUP_COUNT: Final[int] = 5
    DEFAULT_LOG_FORMAT: Final[
        str
    ] = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    DEFAULT_DATE_FORMAT: Final[str] = "%Y-%m-%d %H:%M:%S"

    # Application Metadata
    DEFAULT_APP_NAME: Final[str] = "AI CV Generator"
    DEFAULT_APP_VERSION: Final[str] = "1.0.0"
    DEFAULT_ENVIRONMENT: Final[str] = "development"

    # Paths Configuration
    DEFAULT_DATA_DIRECTORY: Final[str] = "data"
    DEFAULT_PROMPTS_DIRECTORY: Final[str] = "data/prompts"
    DEFAULT_SESSIONS_DIRECTORY: Final[str] = "instance/sessions"
    DEFAULT_LOGS_DIRECTORY: Final[str] = "instance/logs"
    DEFAULT_OUTPUT_DIRECTORY: Final[str] = "instance/output"
    DEFAULT_VECTOR_DB_DIRECTORY: Final[str] = "instance/vector_db"

    # Prompt Settings
    DEFAULT_PROMPTS_DIRECTORY_NAME: Final[str] = "data/prompts"
    CV_PARSER_PROMPT: Final[str] = "cv_parsing_prompt_v2.md"
    CV_ASSESSMENT_PROMPT: Final[str] = "cv_assessment_prompt.md"
    JOB_DESCRIPTION_PARSER_PROMPT: Final[str] = "job_description_parsing_prompt.md"
    RESUME_ROLE_WRITER_PROMPT: Final[str] = "resume_role_prompt.md"
    PROJECT_WRITER_PROMPT: Final[str] = "side_project_prompt.md"
    KEY_QUALIFICATIONS_WRITER_PROMPT: Final[str] = "key_qualifications_prompt.md"
    EXECUTIVE_SUMMARY_WRITER_PROMPT: Final[str] = "executive_summary_prompt.md"
    CV_ANALYSIS_PROMPT: Final[str] = "cv_analysis_prompt.md"
    CLEAN_BIG_6_PROMPT: Final[str] = "clean_big_6_prompt.md"
    CLEAN_JSON_OUTPUT_PROMPT: Final[str] = "clean_json_output_prompt.md"
    JOB_RESEARCH_ANALYSIS_PROMPT: Final[str] = "job_research_analysis_prompt.md"
