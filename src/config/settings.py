"""Configuration management for AI CV Generator.

This module provides centralized configuration management for the application,
including API keys, model settings, and other configuration parameters.
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path

# Try to import python-dotenv, but don't fail if it's not available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


@dataclass
class LLMConfig:
    """Configuration for LLM models and API settings."""
    
    # API Configuration
    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    groq_base_url: str = "https://api.groq.com/openai/v1"
    
    # Gemini API Configuration
    gemini_api_key_primary: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    gemini_api_key_fallback: str = "AIzaSyAvaPMYEVKCSKOfSf4wPIDYIjYAG4QC8us"
    
    # Model Configuration
    generation_model: str = "deepseek-r1-distill-llama-70b"
    cleaning_model: str = "llama-3.3-70b-versatile"
    
    # Rate Limiting Configuration
    max_requests_per_minute: int = 30
    max_tokens_per_minute: int = 6000
    
    # Retry Configuration
    max_retries: int = 3
    retry_delay: float = 1.0
    exponential_backoff: bool = True
    
    # Timeout Configuration
    request_timeout: int = 60
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # Check if at least one API key is available (primary or fallback)
        if not self.groq_api_key and not self.gemini_api_key_primary and not self.gemini_api_key_fallback:
            raise ValueError(
                "At least one API key is required. "
                "Please set GROQ_API_KEY or GEMINI_API_KEY environment variable, "
                "or ensure fallback key is configured."
            )


@dataclass
class VectorDBConfig:
    """Configuration for vector database settings."""
    
    # ChromaDB Configuration
    persist_directory: str = "data/vector_db"
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
    session_timeout_minutes: int = 60
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
    pdf_output_directory: str = "data/output"
    
    # Content Configuration
    max_skills_count: int = 10  # "Big 10" instead of "Big 6"
    max_bullet_points_per_role: int = 5
    max_bullet_points_per_project: int = 3


@dataclass
class LoggingConfig:
    """Configuration for logging settings."""
    
    # Log Levels
    log_level: str = "INFO"
    
    # Log Files
    log_directory: str = "logs"
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
    
    # Application Metadata
    app_name: str = "AI CV Generator"
    app_version: str = "1.0.0"
    
    # Environment
    environment: str = field(default_factory=lambda: os.getenv("ENVIRONMENT", "development"))
    debug: bool = field(default_factory=lambda: os.getenv("DEBUG", "False").lower() == "true")
    
    # Paths
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent)
    data_directory: Path = field(default_factory=lambda: Path("data"))
    prompts_directory: Path = field(default_factory=lambda: Path("data/prompts"))
    sessions_directory: Path = field(default_factory=lambda: Path("data/sessions"))
    
    def __post_init__(self):
        """Initialize paths and create directories if needed."""
        # Ensure data directories exist
        self.data_directory.mkdir(exist_ok=True)
        self.sessions_directory.mkdir(parents=True, exist_ok=True)
        
        # Ensure log directory exists
        log_dir = Path(self.logging.log_directory)
        log_dir.mkdir(exist_ok=True)
        
        # Ensure output directory exists
        output_dir = Path(self.output.pdf_output_directory)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    def get_prompt_path(self, prompt_name: str) -> Path:
        """Get the full path to a prompt file."""
        return self.prompts_directory / f"{prompt_name}.md"
    
    def get_session_path(self, session_id: str) -> Path:
        """Get the full path to a session directory."""
        return self.sessions_directory / session_id
    
    def get_output_path(self, filename: str) -> Path:
        """Get the full path to an output file."""
        return Path(self.output.pdf_output_directory) / filename


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