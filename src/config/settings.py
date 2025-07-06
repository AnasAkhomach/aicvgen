"""Configuration management for AI CV Generator.

This module provides centralized configuration management for the application,
including API keys, model settings, and other configuration parameters.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field
from config.shared_configs import PerformanceConfig, DatabaseConfig
from constants.config_constants import ConfigConstants
from src.utils.import_fallbacks import get_dotenv

# Try to import python-dotenv, but don't fail if it's not available
# Load environment variables with standardized fallback handling
# Note: Import moved to avoid circular dependency
def _load_environment_variables():
    """Load environment variables using standardized fallback handling."""
    try:
        load_dotenv, dotenv_available = get_dotenv()
        if dotenv_available:
            # Load environment variables from .env file
            env_path = Path(__file__).parent.parent.parent / ".env"
            if env_path.exists():
                load_dotenv(env_path)
                print(f"Loaded environment variables from {env_path}")
            else:
                print(f"Warning: .env file not found at {env_path}")
        else:
            print(
                "Warning: python-dotenv not available, environment variables must be set manually"
            )
    except ImportError:
        # Fallback if import_fallbacks is not available
        try:
            from dotenv import load_dotenv
            env_path = Path(__file__).parent.parent.parent / ".env"
            if env_path.exists():
                load_dotenv(env_path)
                print(f"Loaded environment variables from {env_path}")
        except ImportError:
            print("Warning: python-dotenv not available, environment variables must be set manually")

# Load environment variables at module level
_load_environment_variables()


class LLMSettings(BaseModel):
    """Configuration for Large Language Model services."""

    default_model: str = ConfigConstants.DEFAULT_MODEL
    default_temperature: float = ConfigConstants.DEFAULT_TEMPERATURE
    max_tokens: int = ConfigConstants.DEFAULT_MAX_TOKENS
    temperature_analysis: float = ConfigConstants.ANALYSIS_TEMPERATURE
    max_tokens_analysis: int = ConfigConstants.ANALYSIS_MAX_TOKENS


class PromptSettings(BaseModel):
    """Manages paths and names for prompt templates."""

    directory: str = ConfigConstants.DEFAULT_PROMPTS_DIRECTORY_NAME
    cv_parser: str = ConfigConstants.CV_PARSER_PROMPT
    cv_assessment: str = ConfigConstants.CV_ASSESSMENT_PROMPT
    job_description_parser: str = ConfigConstants.JOB_DESCRIPTION_PARSER_PROMPT
    resume_role_writer: str = ConfigConstants.RESUME_ROLE_WRITER_PROMPT
    project_writer: str = ConfigConstants.PROJECT_WRITER_PROMPT
    key_qualifications_writer: str = ConfigConstants.KEY_QUALIFICATIONS_WRITER_PROMPT
    executive_summary_writer: str = ConfigConstants.EXECUTIVE_SUMMARY_WRITER_PROMPT
    cv_analysis: str = ConfigConstants.CV_ANALYSIS_PROMPT
    clean_big_6: str = ConfigConstants.CLEAN_BIG_6_PROMPT
    clean_json_output: str = ConfigConstants.CLEAN_JSON_OUTPUT_PROMPT
    job_research_analysis: str = ConfigConstants.JOB_RESEARCH_ANALYSIS_PROMPT


class AgentSettings(BaseModel):
    """Settings for agent behavior and default values."""

    default_skills: list[str] = Field(
        default_factory=lambda: ConfigConstants.DEFAULT_SKILLS
    )
    max_skills_to_parse: int = ConfigConstants.MAX_SKILLS_TO_PARSE
    max_bullet_points_per_role: int = ConfigConstants.MAX_BULLET_POINTS_PER_ROLE
    max_bullet_points_per_project: int = ConfigConstants.MAX_BULLET_POINTS_PER_PROJECT
    default_company_name: str = ConfigConstants.DEFAULT_COMPANY_NAME
    default_job_title: str = ConfigConstants.DEFAULT_JOB_TITLE


@dataclass
class RateLimitingConfig:
    """Rate limiting settings for LLM API calls."""
    max_requests_per_minute: int = field(
        default_factory=lambda: int(os.getenv("LLM_REQUESTS_PER_MINUTE", str(ConfigConstants.DEFAULT_REQUESTS_PER_MINUTE)))
    )
    max_tokens_per_minute: int = field(
        default_factory=lambda: int(os.getenv("LLM_TOKENS_PER_MINUTE", str(ConfigConstants.DEFAULT_TOKENS_PER_MINUTE)))
    )

@dataclass
class RetryConfig:
    """Retry settings for LLM API calls."""
    max_retries: int = ConfigConstants.DEFAULT_MAX_RETRIES
    retry_delay: float = ConfigConstants.DEFAULT_RETRY_DELAY
    exponential_backoff: bool = True
    request_timeout: int = field(
        default_factory=lambda: int(os.getenv("REQUEST_TIMEOUT_SECONDS", str(ConfigConstants.DEFAULT_REQUEST_TIMEOUT)))
    )

@dataclass
class LLMConfig:
    """Configuration for LLM models and API settings."""

    # Gemini API Configuration
    gemini_api_key_primary: str = field(
        default_factory=lambda: os.getenv("GEMINI_API_KEY", "")
    )
    gemini_api_key_fallback: str = field(
        default_factory=lambda: os.getenv("GEMINI_API_KEY_FALLBACK", "")
    )

    # Model Configuration
    generation_model: str = "gemini-2.0-flash"
    cleaning_model: str = "llama-3.3-70b-versatile"

    # Composed configurations
    rate_limiting: RateLimitingConfig = field(default_factory=RateLimitingConfig)
    retry: RetryConfig = field(default_factory=RetryConfig)

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Note: API key validation is deferred to LLM service initialization
        # to prevent startup hangs. The LLM service will validate keys when needed.
        pass


@dataclass
class VectorDBConfig:
    """Configuration for vector database settings."""

    # ChromaDB Configuration
    persist_directory: str = field(
        default_factory=lambda: os.getenv("VECTOR_DB_PERSIST_DIR", ConfigConstants.DEFAULT_PERSIST_DIRECTORY)
    )
    collection_name: str = field(
        default_factory=lambda: os.getenv("VECTOR_DB_COLLECTION", ConfigConstants.DEFAULT_COLLECTION_NAME)
    )

    # Embedding Configuration
    embedding_model: str = field(
        default_factory=lambda: os.getenv("VECTOR_DB_EMBEDDING_MODEL", ConfigConstants.DEFAULT_EMBEDDING_MODEL)
    )
    embedding_dimension: int = field(
        default_factory=lambda: int(os.getenv("VECTOR_DB_EMBEDDING_DIM", str(ConfigConstants.DEFAULT_EMBEDDING_DIMENSION)))
    )

    # Search Configuration
    max_search_results: int = field(
        default_factory=lambda: int(os.getenv("VECTOR_DB_MAX_RESULTS", str(ConfigConstants.DEFAULT_MAX_SEARCH_RESULTS)))
    )
    similarity_threshold: float = field(
        default_factory=lambda: float(os.getenv("VECTOR_DB_SIMILARITY_THRESHOLD", str(ConfigConstants.DEFAULT_SIMILARITY_THRESHOLD)))
    )


@dataclass
class UIConfig:
    """Configuration for user interface settings."""

    # Streamlit Configuration
    page_title: str = field(
        default_factory=lambda: os.getenv("UI_PAGE_TITLE", ConfigConstants.DEFAULT_PAGE_TITLE)
    )
    page_icon: str = field(
        default_factory=lambda: os.getenv("UI_PAGE_ICON", ConfigConstants.DEFAULT_PAGE_ICON)
    )
    layout: str = field(
        default_factory=lambda: os.getenv("UI_LAYOUT", ConfigConstants.DEFAULT_LAYOUT)
    )

    # Session Configuration
    session_timeout_seconds: int = field(
        default_factory=lambda: int(os.getenv("SESSION_TIMEOUT_SECONDS", str(ConfigConstants.DEFAULT_SESSION_TIMEOUT)))
    )
    auto_save_interval_seconds: int = field(
        default_factory=lambda: int(os.getenv("UI_AUTO_SAVE_INTERVAL", str(ConfigConstants.DEFAULT_AUTO_SAVE_INTERVAL)))
    )

    # Display Configuration
    show_raw_llm_output: bool = field(
        default_factory=lambda: os.getenv("UI_SHOW_RAW_OUTPUT", "true").lower() == "true"
    )
    show_debug_information: bool = field(
        default_factory=lambda: os.getenv("UI_SHOW_DEBUG", "false").lower() == "true"
    )
    items_per_page: int = field(
        default_factory=lambda: int(os.getenv("UI_ITEMS_PER_PAGE", str(ConfigConstants.DEFAULT_ITEMS_PER_PAGE)))
    )


@dataclass
class SessionSettings:
    """Configuration for session management."""

    max_active_sessions: int = field(
        default_factory=lambda: int(os.getenv("MAX_ACTIVE_SESSIONS", str(ConfigConstants.DEFAULT_MAX_ACTIVE_SESSIONS)))
    )
    cleanup_interval_minutes: int = field(
        default_factory=lambda: int(os.getenv("SESSION_CLEANUP_INTERVAL_MINUTES", str(ConfigConstants.DEFAULT_CLEANUP_INTERVAL_MINUTES)))
    )


@dataclass
class OutputConfig:
    """Configuration for output generation settings."""

    # Output Formats
    primary_format: str = field(
        default_factory=lambda: os.getenv("OUTPUT_PRIMARY_FORMAT", ConfigConstants.DEFAULT_PRIMARY_FORMAT)
    )
    supported_formats: list = field(
        default_factory=lambda: os.getenv("OUTPUT_SUPPORTED_FORMATS", ",".join(ConfigConstants.DEFAULT_SUPPORTED_FORMATS)).split(",")
    )

    # PDF Configuration
    pdf_template_path: str = field(
        default_factory=lambda: os.getenv("OUTPUT_PDF_TEMPLATE_PATH", ConfigConstants.DEFAULT_PDF_TEMPLATE_PATH)
    )
    pdf_output_directory: str = field(
        default_factory=lambda: os.getenv("OUTPUT_PDF_DIRECTORY", ConfigConstants.DEFAULT_PDF_OUTPUT_DIRECTORY)
    )

    # Content Generation Constants
    max_skills_count: int = field(
        default_factory=lambda: int(os.getenv("OUTPUT_MAX_SKILLS_COUNT", str(ConfigConstants.DEFAULT_MAX_SKILLS_COUNT)))
    )
    max_bullet_points_per_role: int = field(
        default_factory=lambda: int(os.getenv("OUTPUT_MAX_BULLET_POINTS_ROLE", str(ConfigConstants.MAX_BULLET_POINTS_PER_ROLE)))
    )
    max_bullet_points_per_project: int = field(
        default_factory=lambda: int(os.getenv("OUTPUT_MAX_BULLET_POINTS_PROJECT", str(ConfigConstants.MAX_BULLET_POINTS_PER_PROJECT)))
    )
    min_skill_length: int = field(
        default_factory=lambda: int(os.getenv("OUTPUT_MIN_SKILL_LENGTH", str(ConfigConstants.DEFAULT_MIN_SKILL_LENGTH)))
    )


@dataclass
class LoggingConfig:
    """Configuration for logging settings."""

    # Log Levels
    log_level: str = field(
        default_factory=lambda: os.getenv("LOG_LEVEL", ConfigConstants.DEFAULT_LOG_LEVEL)
    )

    # Log Files
    log_directory: str = field(
        default_factory=lambda: os.getenv("LOG_DIRECTORY", ConfigConstants.DEFAULT_LOG_DIRECTORY)
    )
    main_log_file: str = field(
        default_factory=lambda: os.getenv("LOG_MAIN_FILE", ConfigConstants.DEFAULT_MAIN_LOG_FILE)
    )
    error_log_file: str = field(
        default_factory=lambda: os.getenv("LOG_ERROR_FILE", ConfigConstants.DEFAULT_ERROR_LOG_FILE)
    )
    llm_log_file: str = field(
        default_factory=lambda: os.getenv("LOG_LLM_FILE", ConfigConstants.DEFAULT_LLM_LOG_FILE)
    )

    # Log Rotation
    max_log_size_mb: int = field(
        default_factory=lambda: int(os.getenv("LOG_MAX_SIZE_MB", str(ConfigConstants.DEFAULT_MAX_LOG_SIZE_MB)))
    )
    backup_count: int = field(
        default_factory=lambda: int(os.getenv("LOG_BACKUP_COUNT", str(ConfigConstants.DEFAULT_BACKUP_COUNT)))
    )

    # Log Format
    log_format: str = field(
        default_factory=lambda: os.getenv("LOG_FORMAT", ConfigConstants.DEFAULT_LOG_FORMAT)
    )
    date_format: str = field(
        default_factory=lambda: os.getenv("LOG_DATE_FORMAT", ConfigConstants.DEFAULT_DATE_FORMAT)
    )

    # Performance Logging
    performance_logging: bool = field(
        default_factory=lambda: os.getenv("LOG_PERFORMANCE", "false").lower() == "true"
    )
    log_to_console: bool = field(
        default_factory=lambda: os.getenv("LOG_TO_CONSOLE", "true").lower() == "true"
    )


@dataclass
class ApplicationMetadataConfig:
    """Application metadata settings."""

    app_name: str = field(
        default_factory=lambda: os.getenv("APP_NAME", ConfigConstants.DEFAULT_APP_NAME)
    )
    app_version: str = field(
        default_factory=lambda: os.getenv("APP_VERSION", ConfigConstants.DEFAULT_APP_VERSION)
    )


@dataclass
class EnvironmentConfig:
    """Environment-specific settings."""

    environment: str = field(default_factory=lambda: os.getenv("ENVIRONMENT", ConfigConstants.DEFAULT_ENVIRONMENT))
    debug: bool = field(default_factory=lambda: os.getenv("DEBUG", "true").lower() == "true")


@dataclass
class PathsConfig:
    """Configuration for application paths."""

    project_root: Path = field(default_factory=lambda: Path.cwd())
    data_directory: str = field(
        default_factory=lambda: os.getenv("PATHS_DATA_DIRECTORY", ConfigConstants.DEFAULT_DATA_DIRECTORY)
    )
    prompts_directory: str = field(
        default_factory=lambda: os.getenv("PATHS_PROMPTS_DIRECTORY", ConfigConstants.DEFAULT_PROMPTS_DIRECTORY)
    )
    sessions_directory: str = field(
        default_factory=lambda: os.getenv("PATHS_SESSIONS_DIRECTORY", ConfigConstants.DEFAULT_SESSIONS_DIRECTORY)
    )
    logs_directory: str = field(
        default_factory=lambda: os.getenv("PATHS_LOGS_DIRECTORY", ConfigConstants.DEFAULT_LOGS_DIRECTORY)
    )
    output_directory: str = field(
        default_factory=lambda: os.getenv("PATHS_OUTPUT_DIRECTORY", ConfigConstants.DEFAULT_OUTPUT_DIRECTORY)
    )
    vector_db_directory: str = field(
        default_factory=lambda: os.getenv("PATHS_VECTOR_DB_DIRECTORY", ConfigConstants.DEFAULT_VECTOR_DB_DIRECTORY)
    )


# PerformanceConfig and DatabaseConfig are imported from environment.py

@dataclass
class AppConfig:
    """Main application configuration."""

    # Sub-configurations
    llm: LLMConfig = field(default_factory=LLMConfig)
    vector_db: VectorDBConfig = field(default_factory=VectorDBConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    session: SessionSettings = field(
        default_factory=lambda: SessionSettings(
            max_active_sessions=int(os.getenv("MAX_ACTIVE_SESSIONS", "100")),
            cleanup_interval_minutes=int(
                os.getenv("SESSION_CLEANUP_INTERVAL_MINUTES", "30")
            ),
        )
    )

    # NEW: Add structured configuration sections
    llm_settings: LLMSettings = field(default_factory=LLMSettings)
    prompts: PromptSettings = field(default_factory=PromptSettings)
    agent_settings: AgentSettings = field(default_factory=AgentSettings)
    metadata: ApplicationMetadataConfig = field(default_factory=ApplicationMetadataConfig)
    env: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)



    def __post_init__(self):
        """Initialize paths and create directories if needed."""
        # Define instance path and ensure it exists
        instance_path = self.paths.project_root / "instance"
        instance_path.mkdir(exist_ok=True)

        # Create all runtime directories under instance path
        (instance_path / "sessions").mkdir(parents=True, exist_ok=True)
        (instance_path / "logs").mkdir(exist_ok=True)
        (instance_path / "output").mkdir(parents=True, exist_ok=True)
        (instance_path / "vector_db").mkdir(parents=True, exist_ok=True)

        # Ensure source data directories exist (relative to project root)
        (self.paths.project_root / self.paths.data_directory).mkdir(exist_ok=True)
        (self.paths.project_root / self.paths.prompts_directory).mkdir(parents=True, exist_ok=True)

        # Environment-specific adjustments
        if self.env.environment == "development":
            self.logging.performance_logging = True
            self.ui.show_debug_information = True
            self.performance.enable_profiling = True
        elif self.env.environment == "testing":
            self.logging.log_to_console = False
            self.database.backup_enabled = False
            self.performance.enable_caching = False
        elif self.env.environment == "production":
            self.env.debug = False
            self.logging.log_level = "WARNING"
            self.ui.show_debug_information = False
            self.performance.enable_profiling = False

    def get_prompt_path(self, prompt_name: str) -> Path:
        """Get the full path to a prompt file."""
        return self.paths.project_root / self.paths.prompts_directory / f"{prompt_name}.md"

    def get_prompt_path_by_key(self, prompt_key: str) -> str:
        """
        Constructs the full path to a prompt file using a key from PromptSettings.
        """
        prompt_filename = getattr(self.prompts, prompt_key, None)
        if not prompt_filename:
            raise ValueError(f"Prompt key '{prompt_key}' not found in settings.")

        path = self.paths.project_root / self.paths.prompts_directory / prompt_filename
        if not path.exists():
            raise FileNotFoundError(f"Prompt file does not exist at path: {path}")
        return str(path)

    def get_session_path(self, session_id: str) -> Path:
        """Get the full path to a session directory."""
        return self.paths.project_root / self.paths.sessions_directory / session_id

    def get_output_path(self, filename: str) -> Path:
        """Get the full path to an output file."""
        return self.paths.project_root / self.output.pdf_output_directory / filename


# For backward compatibility: export Settings as AppConfig
Settings = AppConfig

# Global configuration instance
_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = AppConfig()
    return _config


def reload_config() -> AppConfig:
    """Reload the configuration from environment variables."""
    global _config
    _config = AppConfig()
    return _config


def update_config(**kwargs) -> None:
    """Update configuration values."""
    config = get_config()
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f"Unknown configuration key: {key}")
