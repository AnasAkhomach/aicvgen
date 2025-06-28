"""Application Startup Service

This module provides a formal startup sequence and service initialization
for the AI CV Generator application. It ensures proper initialization order,
error handling, and service dependency management.
"""

import os
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
import threading

from ..config.logging_config import setup_logging, get_structured_logger
from ..config.settings import Settings
from ..error_handling.exceptions import ConfigurationError, ServiceInitializationError
from ..core.container import get_container


logger = get_structured_logger(__name__)


@dataclass
class ServiceStatus:
    """Status of a service during initialization."""

    name: str
    initialized: bool
    initialization_time: float
    error: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)


@dataclass
class StartupResult:
    """Result of application startup process."""

    success: bool
    total_time: float
    services: Dict[str, ServiceStatus]
    errors: List[str]
    timestamp: datetime


class ApplicationStartup:
    """Manages application startup sequence and service initialization."""

    def __init__(self):
        self.services: Dict[str, ServiceStatus] = {}
        self.startup_time = None
        self.is_initialized = False
        self.container = get_container()
        self._shutdown_hook_registered = False

    def initialize_application(
        self, user_api_key: str = "", session_id: Optional[str] = None
    ) -> StartupResult:
        """
        Ensures all core services are wired and ready.
        With the new DI container, initialization is mostly declarative.
        This method can be used to perform any explicit startup actions
        or validations if needed.
        """
        logger.info("Initializing application services...")
        start_time = time.time()
        errors: List[str] = []
        service_statuses: Dict[str, ServiceStatus] = {}

        try:
            # With dependency-injector, services are lazily instantiated.
            # We can "ping" a few critical services to ensure they are configured correctly.

            # Validate settings
            self.container.config()
            service_statuses["settings"] = ServiceStatus(
                name="settings",
                initialized=True,
                initialization_time=0.0,
                dependencies=[],
            )

            # Validate LLM service
            self.container.llm_service()
            service_statuses["llm_service"] = ServiceStatus(
                name="llm_service",
                initialized=True,
                initialization_time=0.0,
                dependencies=["settings", "llm_client"],
            )

            # Validate Vector Store
            self.container.vector_store_service()
            service_statuses["vector_store_service"] = ServiceStatus(
                name="vector_store_service",
                initialized=True,
                initialization_time=0.0,
                dependencies=["settings"],
            )

            self.is_initialized = True
            logger.info("Application services initialized successfully.")

        except Exception as e:
            logger.error("Failed to initialize application services", error=str(e))
            errors.append(str(e))
            self.is_initialized = False

        total_time = time.time() - start_time
        return StartupResult(
            success=self.is_initialized,
            total_time=total_time,
            services=service_statuses,
            errors=errors,
            timestamp=datetime.now(),
        )

    def get_service(self, service_name: str, session_id: Optional[str] = None) -> Any:
        """Retrieve a service from the container."""
        if not self.is_initialized:
            raise ServiceInitializationError(
                "Application not initialized. Call initialize_services() first."
            )

        try:
            # The container from dependency-injector uses attribute access
            if hasattr(self.container, service_name):
                return getattr(self.container, service_name)()
            else:
                raise AttributeError(
                    f"Service '{service_name}' not found in container."
                )
        except Exception as e:
            logger.error(f"Failed to retrieve service '{service_name}'", error=str(e))
            raise ServiceInitializationError(
                f"Could not get service: {service_name}"
            ) from e

    def shutdown(self):
        """Perform graceful shutdown of services."""
        logger.info("Shutting down application services...")
        # dependency-injector containers have a shutdown_resources method
        self.container.shutdown_resources()
        self.is_initialized = False
        logger.info("Application shutdown complete.")

    def validate_application(self) -> List[str]:
        """
        Validate that all critical services are properly initialized and configured.
        Returns a list of validation errors, empty if all validations pass.
        """
        errors = []

        if not self.is_initialized:
            errors.append("Application not initialized")
            return errors

        try:
            # Validate that critical services are accessible
            self.container.config()
            self.container.llm_service()
            self.container.vector_store_service()

            logger.info("Application validation completed successfully")
        except Exception as e:
            logger.error("Application validation failed", error=str(e))
            errors.append(f"Service validation failed: {str(e)}")

        return errors

    def shutdown_application(self):
        """Alias for shutdown() method to match main.py expectations."""
        self.shutdown()


_startup_lock = threading.Lock()
_startup_instance: Optional[ApplicationStartup] = None


def get_application_startup() -> ApplicationStartup:
    """Returns a singleton instance of the ApplicationStartup class."""
    global _startup_instance
    if _startup_instance is None:
        with _startup_lock:
            if _startup_instance is None:
                _startup_instance = ApplicationStartup()
    return _startup_instance


def get_startup_manager() -> ApplicationStartup:
    """Alias for get_application_startup() to match main.py expectations."""
    return get_application_startup()
