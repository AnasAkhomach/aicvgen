#!/usr/bin/env python3
"""
Shared Configuration Classes

This module contains configuration classes that are shared between different
configuration modules to avoid circular imports.
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PerformanceConfig:
    """Performance configuration settings."""

    enable_caching: bool = True
    cache_ttl_seconds: int = 3600
    max_concurrent_requests: int = 10
    request_timeout_seconds: int = field(
        default_factory=lambda: int(os.getenv("REQUEST_TIMEOUT_SECONDS", "60"))
    )
    enable_profiling: bool = False
    memory_limit_mb: Optional[int] = None


@dataclass
class DatabaseConfig:
    """Database configuration settings."""

    vector_db_path: str = field(default_factory=lambda: "data/enhanced_vector_db")
    session_db_path: str = field(default_factory=lambda: "data/sessions")
    backup_enabled: bool = True
    backup_interval_hours: int = 24
    max_backups: int = 7