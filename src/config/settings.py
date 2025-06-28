"""Configuration management for AI CV Generator.

This module provides centralized configuration management for the application,
including API keys, model settings, and other configuration parameters.
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
from pydantic import BaseModel, Field, DirectoryPath

# Try to import python-dotenv, but don't fail if it's not available
try:
    from dotenv import load_dotenv

    # Load .env file from project root
    env_path = Path(__file__).parent.parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f"Loaded environment variables from {env_path}")
    else:
        print(f"Warning: .env file not found at {env_path}")
except ImportError:
    print(
        "Warning: python-dotenv not available, environment variables must be set manually"
    )


class LLMSettings(BaseModel):
    """Configuration for Large Language Model services."""

    default_model: str = "gemini-2.0-flash"
    default_temperature: float = 0.7
    max_tokens: int = 4096
    temperature_analysis: float = 0.5
    max_tokens_analysis: int = 2048


class PromptSettings(BaseModel):
    """Manages paths and names for prompt templates."""

    directory: str = "data/prompts"
    cv_parser: str = "cv_parsing_prompt_v2.md"
    cv_assessment: str = "cv_assessment_prompt.md"
    job_description_parser: str = "job_description_parsing_prompt.md"
    resume_role_writer: str = "resume_role_prompt.md"
    project_writer: str = "side_project_prompt.md"
    key_qualifications_writer: str = "key_qualifications_prompt.md"
    executive_summary_writer: str = "executive_summary_prompt.md"
    cv_analysis: str = "cv_analysis_prompt.md"
    clean_big_6: str = "clean_big_6_prompt.md"
    clean_json_output: str = "clean_json_output_prompt.md"
    job_research_analysis: str = "job_research_analysis_prompt.md"


class AgentSettings(BaseModel):
    """Settings for agent behavior and default values."""

    default_skills: list[str] = Field(
        default_factory=lambda: [
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
    )
    max_skills_to_parse: int = 10
    max_bullet_points_per_role: int = 5
    max_bullet_points_per_project: int = 3
    default_company_name: str = "Unknown Company"
    default_job_title: str = "Unknown Title"


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
    generation_model: str = "deepseek-r1-distill-llama-70b"
    cleaning_model: str = "llama-3.3-70b-versatile"

    # Rate Limiting Configuration
    max_requests_per_minute: int = field(
        default_factory=lambda: int(os.getenv("LLM_REQUESTS_PER_MINUTE", "30"))
    )
    max_tokens_per_minute: int = field(
        default_factory=lambda: int(os.getenv("LLM_TOKENS_PER_MINUTE", "60000"))
    )

    # Retry Configuration
    max_retries: int = 3
    retry_delay: float = 1.0
    exponential_backoff: bool = True  # Timeout Configuration
    request_timeout: int = field(
        default_factory=lambda: int(os.getenv("REQUEST_TIMEOUT_SECONDS", "60"))
    )

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Note: API key validation is deferred to LLM service initialization
        # to prevent startup hangs. The LLM service will validate keys when needed.
        pass


@dataclass
class VectorDBConfig:
    """Configuration for vector database settings."""

    # ChromaDB Configuration
    persist_directory: str = "instance/vector_db"
    collection_name: str = "cv_content"

    # Embedding Configuration
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384

    # Search Configuration
    max_search_results: int = 10
    similarity_threshold: float = 0.7


@dataclass
class UIConfig:
    """Configuration for user interface settings."""

    # Streamlit Configuration
    page_title: str = "AI CV Generator"
    page_icon: str = "📄"
    layout: str = "wide"

    # Session Configuration
    session_timeout_seconds: int = field(
        default_factory=lambda: int(os.getenv("SESSION_TIMEOUT_SECONDS", "3600"))
    )
    auto_save_interval_seconds: int = 30

    # Display Configuration
    show_raw_llm_output: bool = True
    items_per_page: int = 5


@dataclass
class OutputConfig:
    """Configuration for output generation settings."""

    # Output Formats
    primary_format: str = "pdf"
    supported_formats: list = field(default_factory=lambda: ["pdf", "markdown", "html"])

    # PDF Configuration
    pdf_template_path: str = "src/templates/cv_template.md"
    pdf_output_directory: str = "instance/output"

    # Content Generation Constants
    max_skills_count: int = 10
    max_bullet_points_per_role: int = 5
    max_bullet_points_per_project: int = 3
    min_skill_length: int = 2


@dataclass
class LoggingConfig:
    """Configuration for logging settings."""

    # Log Levels
    log_level: str = "INFO"

    # Log Files
    log_directory: str = "instance/logs"
    main_log_file: str = "app.log"
    error_log_file: str = "error.log"
    llm_log_file: str = "llm_calls.log"

    # Log Rotation
    max_log_size_mb: int = 10
    backup_count: int = 5

    # Log Format
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"


@dataclass
class AppConfig:
    """Main application configuration."""

    # Sub-configurations
    llm: LLMConfig = field(default_factory=LLMConfig)
    vector_db: VectorDBConfig = field(default_factory=VectorDBConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # NEW: Add structured configuration sections
    llm_settings: LLMSettings = field(default_factory=LLMSettings)
    prompts: PromptSettings = field(default_factory=PromptSettings)
    agent_settings: AgentSettings = field(default_factory=AgentSettings)

    # Application Metadata
    app_name: str = "AI CV Generator"
    app_version: str = "1.0.0"

    # Environment
    environment: str = field(
        default_factory=lambda: os.getenv("ENVIRONMENT", "development")
    )
    debug: bool = field(
        default_factory=lambda: os.getenv("DEBUG_MODE", "false").lower() == "true"
    )

    # Paths
    project_root: Path = field(
        default_factory=lambda: Path(__file__).parent.parent.parent
    )
    data_directory: Path = field(default_factory=lambda: Path("data"))
    prompts_directory: Path = field(default_factory=lambda: Path("data/prompts"))
    sessions_directory: Path = field(default_factory=lambda: Path("instance/sessions"))

    def __post_init__(self):
        """Initialize paths and create directories if needed."""
        # Define instance path and ensure it exists
        instance_path = self.project_root / "instance"
        instance_path.mkdir(exist_ok=True)

        # Create all runtime directories under instance path
        (instance_path / "sessions").mkdir(parents=True, exist_ok=True)
        (instance_path / "logs").mkdir(exist_ok=True)
        (instance_path / "output").mkdir(parents=True, exist_ok=True)
        (instance_path / "vector_db").mkdir(parents=True, exist_ok=True)

        # Ensure source data directories exist (relative to project root)
        (self.project_root / self.data_directory).mkdir(exist_ok=True)
        (self.project_root / self.prompts_directory).mkdir(parents=True, exist_ok=True)

    def get_prompt_path(self, prompt_name: str) -> Path:
        """Get the full path to a prompt file."""
        return self.project_root / self.prompts_directory / f"{prompt_name}.md"

    def get_prompt_path_by_key(self, prompt_key: str) -> str:
        """
        Constructs the full path to a prompt file using a key from PromptSettings.
        """
        prompt_filename = getattr(self.prompts, prompt_key, None)
        if not prompt_filename:
            raise ValueError(f"Prompt key '{prompt_key}' not found in settings.")

        path = self.project_root / self.prompts_directory / prompt_filename
        if not path.exists():
            raise FileNotFoundError(f"Prompt file does not exist at path: {path}")
        return str(path)

    def get_session_path(self, session_id: str) -> Path:
        """Get the full path to a session directory."""
        return self.project_root / self.sessions_directory / session_id

    def get_output_path(self, filename: str) -> Path:
        """Get the full path to an output file."""
        return self.project_root / self.output.pdf_output_directory / filename


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
