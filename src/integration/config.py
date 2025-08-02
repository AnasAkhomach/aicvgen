"""Configuration classes for CV system integration."""

from dataclasses import dataclass
from datetime import timedelta
from enum import Enum
from typing import Optional


class IntegrationMode(Enum):
    """Integration modes for different use cases."""

    DEVELOPMENT = "development"
    PRODUCTION = "production"
    TESTING = "testing"
    DEMO = "demo"


@dataclass
class EnhancedCVConfig:
    """Configuration for enhanced CV system integration."""

    mode: IntegrationMode
    enable_vector_db: bool = True
    enable_orchestration: bool = True
    enable_templates: bool = True
    enable_specialized_agents: bool = True
    vector_db_path: Optional[str] = None
    template_cache_size: int = 100
    orchestration_timeout: timedelta = timedelta(minutes=30)
    max_concurrent_agents: int = 5
    enable_performance_monitoring: bool = True
    enable_error_recovery: bool = True
    debug_mode: bool = False
    api_key: Optional[str] = None
    enable_caching: bool = True
    enable_monitoring: bool = True

    def to_dict(self):
        """Convert config to dictionary with JSON-serializable values."""
        result = {}
        for field_name, field_value in self.__dict__.items():
            if isinstance(field_value, IntegrationMode):
                result[field_name] = field_value.value
            elif isinstance(field_value, timedelta):
                result[field_name] = field_value.total_seconds()
            else:
                result[field_name] = field_value
        return result

    @classmethod
    def from_dict(cls, data: dict):
        """Create config from dictionary."""
        # Convert mode back to enum
        if "mode" in data and isinstance(data["mode"], str):
            data["mode"] = IntegrationMode(data["mode"])
        # Convert timeout back to timedelta
        if "orchestration_timeout" in data and isinstance(
            data["orchestration_timeout"], (int, float)
        ):
            data["orchestration_timeout"] = timedelta(
                seconds=data["orchestration_timeout"]
            )
        return cls(**data)
