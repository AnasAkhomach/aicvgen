#!/usr/bin/env python3
"""
Environment Configuration Module

This module provides environment-specific configurations for development, testing, and production.
It handles environment variable loading, validation, and provides typed configuration objects.
"""

import os
import logging
from enum import Enum
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pathlib import Path

# Try to import python-dotenv, but don't fail if it's not available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class Environment(Enum):
    """Application environment types."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"


@dataclass
class DatabaseConfig:
    """Database configuration settings."""
    vector_db_path: str = field(default_factory=lambda: "data/enhanced_vector_db")
    session_db_path: str = field(default_factory=lambda: "data/sessions")
    backup_enabled: bool = True
    backup_interval_hours: int = 24
    max_backups: int = 7


@dataclass
class LoggingConfig:
    """Logging configuration settings."""
    level: str = "INFO"
    log_to_file: bool = True
    log_to_console: bool = True
    max_file_size_mb: int = 10
    backup_count: int = 5
    structured_logging: bool = True
    performance_logging: bool = False
    
    def get_log_level(self) -> int:
        """Convert string log level to logging constant."""
        return getattr(logging, self.level.upper(), logging.INFO)


@dataclass
class SecurityConfig:
    """Security configuration settings."""
    api_key_validation: bool = True
    rate_limiting_enabled: bool = True
    session_timeout_seconds: int = field(default_factory=lambda: int(os.getenv("SESSION_TIMEOUT_SECONDS", "3600")))
    max_file_upload_size_mb: int = 10
    allowed_file_types: List[str] = field(default_factory=lambda: [".txt", ".md", ".pdf", ".docx"])


# UIConfig is imported from settings.py to avoid duplication
from .settings import UIConfig


@dataclass
class PerformanceConfig:
    """Performance configuration settings."""
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600
    max_concurrent_requests: int = 10
    request_timeout_seconds: int = field(default_factory=lambda: int(os.getenv("REQUEST_TIMEOUT_SECONDS", "60")))
    enable_profiling: bool = False
    memory_limit_mb: Optional[int] = None


@dataclass
class AppConfig:
    """Main application configuration."""
    environment: Environment
    debug: bool
    testing: bool
    
    # Sub-configurations
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    
    # Paths
    project_root: Path = field(default_factory=lambda: Path.cwd())
    data_dir: Path = field(default_factory=lambda: Path("data"))
    logs_dir: Path = field(default_factory=lambda: Path("logs"))
    templates_dir: Path = field(default_factory=lambda: Path("src/templates"))
    
    def __post_init__(self):
        """Post-initialization setup."""
        # Ensure directories exist
        self.data_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        
        # Environment-specific adjustments
        if self.environment == Environment.DEVELOPMENT:
            self.logging.performance_logging = True
            self.ui.show_debug_info = True
            self.performance.enable_profiling = True
        elif self.environment == Environment.TESTING:
            self.logging.log_to_console = False
            self.database.backup_enabled = False
            self.performance.enable_caching = False
        elif self.environment == Environment.PRODUCTION:
            self.debug = False
            self.logging.level = "WARNING"
            self.ui.show_debug_info = False
            self.performance.enable_profiling = False


def get_environment() -> Environment:
    """Get the current environment from environment variables."""
    env_name = os.getenv("ENVIRONMENT", "development").lower()
    try:
        return Environment(env_name)
    except ValueError:
        return Environment.DEVELOPMENT


def load_config() -> AppConfig:
    """Load configuration based on the current environment."""
    env = get_environment()
    debug = os.getenv("DEBUG_MODE", "false").lower() == "true"
    testing = os.getenv("TESTING", "false").lower() == "true"
    
    config = AppConfig(
        environment=env,
        debug=debug,
        testing=testing
    )
    
    # Override with environment variables if present
    _override_from_env(config)
    
    return config


def _override_from_env(config: AppConfig) -> None:
    """Override configuration values from environment variables."""
    # Logging overrides
    if log_level := os.getenv("LOG_LEVEL"):
        config.logging.level = log_level
    
    if log_to_file := os.getenv("LOG_TO_FILE"):
        config.logging.log_to_file = log_to_file.lower() == "true"
    
    # Performance overrides
    if cache_enabled := os.getenv("ENABLE_CACHING"):
        config.performance.enable_caching = cache_enabled.lower() == "true"
    
    if timeout := os.getenv("REQUEST_TIMEOUT"):
        try:
            config.performance.request_timeout_seconds = int(timeout)
        except ValueError:
            pass
    
    # Security overrides
    if session_timeout := os.getenv("SESSION_TIMEOUT_SECONDS"):
        try:
            config.security.session_timeout_seconds = int(session_timeout)
        except ValueError:
            pass


# Global configuration instance
config = load_config()