#!/usr/bin/env python3
"""
Environment Configuration Module

This module provides environment-specific configurations for development, testing, and production.
It handles environment variable loading, validation, and provides typed configuration objects.
"""

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List
from config.settings import UIConfig, EnvironmentConfig
from config.shared_configs import PerformanceConfig, DatabaseConfig as SharedDatabaseConfig

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


# DatabaseConfig is imported from shared_configs as SharedDatabaseConfig
# Use SharedDatabaseConfig for the shared configuration
DatabaseConfig = SharedDatabaseConfig


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
    session_timeout_seconds: int = field(
        default_factory=lambda: int(os.getenv("SESSION_TIMEOUT_SECONDS", "3600"))
    )
    max_file_upload_size_mb: int = 10
    allowed_file_types: List[str] = field(
        default_factory=lambda: [".txt", ".md", ".pdf", ".docx"]
    )


# UIConfig and EnvironmentConfig are imported from settings.py to avoid duplication


# PerformanceConfig is imported from shared_configs


@dataclass
class PathsConfig:
    """Configuration for application paths."""

    project_root: Path = field(default_factory=lambda: Path.cwd())
    data_dir: Path = field(default_factory=lambda: Path("data"))
    logs_dir: Path = field(default_factory=lambda: Path("logs"))
    templates_dir: Path = field(default_factory=lambda: Path("src/templates"))

    def __post_init__(self):
        """Ensure directories exist."""
        self.data_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)


@dataclass
class AppConfig:
    """Main application configuration."""

    # Sub-configurations
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    env: EnvironmentConfig = field(default_factory=EnvironmentConfig)

    def __post_init__(self):
        """Post-initialization setup."""
        # Environment-specific adjustments
        if self.env.environment == Environment.DEVELOPMENT:
            self.logging.performance_logging = True
            self.ui.show_debug_information = True
            self.performance.enable_profiling = True
        elif self.env.environment == Environment.TESTING:
            self.logging.log_to_console = False
            self.database.backup_enabled = False
            self.performance.enable_caching = False
        elif self.env.environment == Environment.PRODUCTION:
            self.env.debug = False
            self.logging.level = "WARNING"
            self.ui.show_debug_information = False
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
    config = AppConfig()

    # Override with environment variables if present
    _override_from_env(config)

    return config


def _override_from_env(config: AppConfig) -> None:
    """Override configuration values from environment variables."""
    # Logging overrides
    if log_level := os.getenv("LOG_LEVEL"):
        config.logging.log_level = log_level

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
