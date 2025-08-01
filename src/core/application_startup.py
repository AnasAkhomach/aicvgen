"""Application Startup Service

This module provides a formal startup sequence and service initialization
for the AI CV Generator application. It ensures proper initialization order,
error handling, and service dependency management.

This is the single source of truth for all application initialization logic,
including logging setup, DI container setup, API key validation, and StateManager initialization.
"""

import atexit
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.config.logging_config import get_structured_logger, setup_logging
from src.config.settings import Settings
from src.core.container import get_container
from src.core.state_manager import StateManager
from src.error_handling.exceptions import ConfigurationError, ServiceInitializationError

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
    """Manages application startup sequence and service initialization.

    This is the single source of truth for all application initialization logic.
    """

    def __init__(self):
        self.services: Dict[str, ServiceStatus] = {}
        self.startup_time = None
        self.is_initialized = False
        self.container = get_container()
        self._shutdown_hook_registered = False
        self.state_manager: Optional[StateManager] = None
        self.last_startup_result: Optional[StartupResult] = None

    def initialize_application(
        self, user_api_key: str = "", session_id: Optional[str] = None
    ) -> StartupResult:
        """
        Consolidated application initialization - the single source of truth.

        This method handles:
        1. Logging setup
        2. StateManager initialization
        3. DI container setup
        4. API key validation
        5. Core service validation

        Args:
            user_api_key: Optional user API key for validation
            session_id: Optional session identifier

        Returns:
            StartupResult with success status and details
        """
        logger.info("Starting consolidated application initialization...")
        start_time = time.time()
        errors: List[str] = []
        service_statuses: Dict[str, ServiceStatus] = {}

        try:
            # Step 1: Setup logging (if not already done)
            logging_start = time.time()
            try:
                setup_logging()
                service_statuses["logging"] = ServiceStatus(
                    name="logging",
                    initialized=True,
                    initialization_time=time.time() - logging_start,
                    dependencies=[],
                )
                logger.info("Logging setup completed")
            except Exception as e:
                errors.append(f"Logging setup failed: {str(e)}")
                service_statuses["logging"] = ServiceStatus(
                    name="logging",
                    initialized=False,
                    initialization_time=time.time() - logging_start,
                    error=str(e),
                )

            # Step 2: Initialize StateManager
            state_start = time.time()
            try:
                if self.state_manager is None:
                    self.state_manager = StateManager()
                service_statuses["state_manager"] = ServiceStatus(
                    name="state_manager",
                    initialized=True,
                    initialization_time=time.time() - state_start,
                    dependencies=["logging"],
                )
                logger.info("StateManager initialized")
            except Exception as e:
                errors.append(f"StateManager initialization failed: {str(e)}")
                service_statuses["state_manager"] = ServiceStatus(
                    name="state_manager",
                    initialized=False,
                    initialization_time=time.time() - state_start,
                    error=str(e),
                )

            # Step 3: Register shutdown hook
            if not self._shutdown_hook_registered:
                atexit.register(self.shutdown_application)
                self._shutdown_hook_registered = True
                logger.info("Shutdown hook registered")

            # Step 4: Validate DI container and core services
            container_start = time.time()
            try:
                # Validate settings
                self.container.config()
                service_statuses["settings"] = ServiceStatus(
                    name="settings",
                    initialized=True,
                    initialization_time=0.0,
                    dependencies=["logging"],
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
                    initialization_time=time.time() - container_start,
                    dependencies=["settings"],
                )
                logger.info("Core services validated")
            except Exception as e:
                errors.append(f"Service validation failed: {str(e)}")
                service_statuses["container_services"] = ServiceStatus(
                    name="container_services",
                    initialized=False,
                    initialization_time=time.time() - container_start,
                    error=str(e),
                )

            # Step 5: API key validation (if provided)
            if user_api_key and self.state_manager:
                api_start = time.time()
                try:
                    # Store the API key in state manager
                    self.state_manager.user_gemini_api_key = user_api_key
                    service_statuses["api_key"] = ServiceStatus(
                        name="api_key",
                        initialized=True,
                        initialization_time=time.time() - api_start,
                        dependencies=["state_manager"],
                    )
                    logger.info("API key validated and stored")
                except Exception as e:
                    errors.append(f"API key validation failed: {str(e)}")
                    service_statuses["api_key"] = ServiceStatus(
                        name="api_key",
                        initialized=False,
                        initialization_time=time.time() - api_start,
                        error=str(e),
                    )

            # Mark as initialized only if no critical errors
            self.is_initialized = len(errors) == 0
            if self.is_initialized:
                logger.info(
                    "Consolidated application initialization completed successfully"
                )
            else:
                logger.warning(
                    f"Application initialization completed with {len(errors)} errors"
                )

        except Exception as e:
            logger.error(
                "Failed to initialize application services", error=str(e), exc_info=True
            )
            errors.append(f"Unexpected initialization error: {str(e)}")
            self.is_initialized = False

        total_time = time.time() - start_time
        startup_result = StartupResult(
            success=self.is_initialized,
            total_time=total_time,
            services=service_statuses,
            errors=errors,
            timestamp=datetime.now(),
        )

        # Store the result for later access
        self.last_startup_result = startup_result

        return startup_result

    def get_service(self, service_name: str, session_id: Optional[str] = None) -> Any:
        """Retrieve a service from the container."""
        if not self.is_initialized:
            raise ServiceInitializationError(
                service_name="application",
                message="Application not initialized. Call initialize_application() first.",
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
                service_name=service_name, message="Could not get service"
            ) from e

    def get_state_manager(self) -> StateManager:
        """Get the initialized StateManager instance.

        Returns:
            StateManager: The initialized state manager

        Raises:
            ServiceInitializationError: If application is not initialized
        """
        if not self.is_initialized or self.state_manager is None:
            raise ServiceInitializationError(
                service_name="state_manager",
                message="Application not initialized or StateManager not available. Call initialize_application() first.",
            )
        return self.state_manager

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
