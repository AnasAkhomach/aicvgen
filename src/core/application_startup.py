"""Application Startup Service

This module provides a formal startup sequence and service initialization
for the AI CV Generator application. It ensures proper initialization order,
error handling, and service dependency management.
"""

import os
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime
import threading

from ..config.logging_config import setup_logging, get_logger
from ..config.environment import load_config
from ..config.settings import Settings
from ..services.llm_service import EnhancedLLMService
from ..services.vector_store_service import get_vector_store_service
from ..services.session_manager import get_session_manager
from ..utils.exceptions import ConfigurationError, ServiceInitializationError
from ..core.dependency_injection import (
    configure_container,
    DependencyContainer,
)
from ..utils.streamlit_utils import configure_page


logger = get_logger(__name__)


@dataclass
class ServiceStatus:
    """Status of a service during initialization."""

    name: str
    initialized: bool
    initialization_time: float
    error: Optional[str] = None
    dependencies: List[str] = None


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

    def initialize_application(self, user_api_key: str = "") -> StartupResult:
        """Initialize the application with proper startup sequence.

        Args:
            user_api_key: User-provided API key for LLM service (unused)

        Returns:
            StartupResult: Result of the startup process
        """
        _ = user_api_key  # Mark as intentionally unused
        start_time = time.time()
        errors = []

        logger.info("Starting application initialization")

        try:
            # Get container and register services
            self.container = DependencyContainer()

            # Check if settings is registered, if not register core services
            try:
                self.container.get(Settings, "settings")
                logger.debug("Settings already registered")
            except ValueError:
                logger.info("Settings not registered, configuring container")
                configure_container(self.container)

            # Phase 1: Core Infrastructure
            self._initialize_logging()
            self._initialize_environment()
            self._ensure_directories()

            # Phase 2: External Services
            logger.info("Starting LLM service initialization...")
            self._initialize_llm_service()
            logger.info("LLM service initialization complete")

            logger.info("Starting vector store initialization...")
            self._initialize_vector_store()
            logger.info("Vector store initialization complete")

            # Phase 3: Application Services
            logger.info("Starting session manager initialization...")
            self._initialize_session_manager()
            logger.info("Session manager initialization complete")

            # Phase 4: Streamlit Configuration
            logger.info("Starting Streamlit configuration...")
            self._configure_streamlit()
            logger.info("Streamlit configuration complete")

            self.is_initialized = True
            total_time = time.time() - start_time
            self.startup_time = total_time

            logger.info("Application initialization completed in %.2fs", total_time)

            return StartupResult(
                success=True,
                total_time=total_time,
                services=self.services.copy(),
                errors=errors,
                timestamp=datetime.now(),
            )

        except (ServiceInitializationError, ConfigurationError) as e:
            total_time = time.time() - start_time
            error_msg = f"Application initialization failed: {e}"
            errors.append(error_msg)
            logger.error(error_msg, exc_info=True)
            raise
        except Exception as e:
            total_time = time.time() - start_time
            error_msg = (
                f"An unexpected error occurred during application initialization: {e}"
            )
            errors.append(error_msg)
            logger.error(error_msg, exc_info=True)
            return StartupResult(
                success=False,
                total_time=total_time,
                services=self.services.copy(),
                errors=errors,
                timestamp=datetime.now(),
            )

    def _initialize_logging(self):
        """Initialize logging system."""
        start_time = time.time()
        try:
            setup_logging()
            self.services["logging"] = ServiceStatus(
                name="logging",
                initialized=True,
                initialization_time=time.time() - start_time,
            )
            logger.info("Logging system initialized")
        except Exception as e:
            self.services["logging"] = ServiceStatus(
                name="logging",
                initialized=False,
                initialization_time=time.time() - start_time,
                error=str(e),
            )
            raise ServiceInitializationError(
                f"Failed to initialize logging: {e}"
            ) from e

    def _initialize_environment(self):
        """Initialize environment configuration."""
        start_time = time.time()
        try:
            load_config()
            self.services["environment"] = ServiceStatus(
                name="environment",
                initialized=True,
                initialization_time=time.time() - start_time,
            )
            logger.info("Environment configuration loaded")
        except Exception as e:
            self.services["environment"] = ServiceStatus(
                name="environment",
                initialized=False,
                initialization_time=time.time() - start_time,
                error=str(e),
            )
            raise ConfigurationError(
                f"Failed to load environment configuration: {e}"
            ) from e

    def _ensure_directories(self):
        """Ensure required directories exist."""
        start_time = time.time()
        try:
            directories = [
                "data/sessions",
                "data/output",
                "data/cache",
                "logs",
                "data/vector_db",
            ]

            for directory in directories:
                os.makedirs(directory, exist_ok=True)

            self.services["directories"] = ServiceStatus(
                name="directories",
                initialized=True,
                initialization_time=time.time() - start_time,
            )
            logger.info("Required directories ensured")
        except Exception as e:
            self.services["directories"] = ServiceStatus(
                name="directories",
                initialized=False,
                initialization_time=time.time() - start_time,
                error=str(e),
            )
            raise ServiceInitializationError(
                f"Failed to create directories: {e}"
            ) from e

    def _initialize_llm_service(self):
        """Initialize LLM service using the DI container."""
        start_time = time.time()

        # Check if LLM service initialization should be skipped
        if os.getenv("SKIP_LLM_SERVICE", "false").lower() == "true":
            logger.warning(
                "Skipping LLM service initialization (SKIP_LLM_SERVICE=true)"
            )
            self.services["llm_service"] = ServiceStatus(
                name="llm_service",
                initialized=False,
                initialization_time=time.time() - start_time,
                error="Skipped via SKIP_LLM_SERVICE environment variable",
            )
            return

        try:
            logger.info("Getting DI container...")
            container = self.container
            logger.info("Calling container.get() for EnhancedLLMService...")
            # The factory registered in initialize_application will be called here
            llm_service = container.get(EnhancedLLMService, "EnhancedLLMService")
            logger.info("EnhancedLLMService obtained from container")

            if not llm_service:
                raise ServiceInitializationError(
                    "LLM service could not be created from DI container."
                )

            self.services["llm_service"] = ServiceStatus(
                name="llm_service",
                initialized=True,
                initialization_time=time.time() - start_time,
                dependencies=[
                    "environment",
                    "settings",
                    "RateLimiter",
                    "AdvancedCache",
                ],
            )
            logger.info("LLM service initialized via DI container")
        except Exception as e:
            self.services["llm_service"] = ServiceStatus(
                name="llm_service",
                initialized=False,
                initialization_time=time.time() - start_time,
                error=str(e),
            )
            raise ServiceInitializationError(
                f"Failed to initialize LLM service from DI container: {e}"
            ) from e

    def _initialize_vector_store(self):
        """Initialize vector store service."""
        start_time = time.time()

        # Check if vector store initialization should be skipped
        if os.getenv("SKIP_VECTOR_STORE", "false").lower() == "true":
            logger.warning(
                "Skipping vector store initialization (SKIP_VECTOR_STORE=true)"
            )
            self.services["vector_store"] = ServiceStatus(
                name="vector_store",
                initialized=False,
                initialization_time=time.time() - start_time,
                error="Skipped via SKIP_VECTOR_STORE environment variable",
                dependencies=["directories"],
            )
            return

        try:
            get_vector_store_service()
            self.services["vector_store"] = ServiceStatus(
                name="vector_store",
                initialized=True,
                initialization_time=time.time() - start_time,
                dependencies=["directories"],
            )
            logger.info("Vector store service initialized")
        except Exception as e:
            self.services["vector_store"] = ServiceStatus(
                name="vector_store",
                initialized=False,
                initialization_time=time.time() - start_time,
                error=str(e),
                dependencies=["directories"],
            )
            raise ServiceInitializationError(
                f"Failed to initialize vector store: {e}"
            ) from e

    def _initialize_session_manager(self):
        """Initialize session manager service."""
        start_time = time.time()
        try:
            get_session_manager()
            self.services["session_manager"] = ServiceStatus(
                name="session_manager",
                initialized=True,
                initialization_time=time.time() - start_time,
                dependencies=["directories"],
            )
            logger.info("Session manager initialized")
        except Exception as e:
            self.services["session_manager"] = ServiceStatus(
                name="session_manager",
                initialized=False,
                initialization_time=time.time() - start_time,
                error=str(e),
                dependencies=["directories"],
            )
            # Session manager is not critical, log warning but continue
            logger.warning("Session manager initialization failed: %s", e)

    def _configure_streamlit(self):
        """Configure Streamlit page settings."""
        start_time = time.time()
        try:
            configure_page()

            self.services["streamlit"] = ServiceStatus(
                name="streamlit",
                initialized=True,
                initialization_time=time.time() - start_time,
            )
            logger.info("Streamlit configuration applied")
        except Exception as e:
            self.services["streamlit"] = ServiceStatus(
                name="streamlit",
                initialized=False,
                initialization_time=time.time() - start_time,
                error=str(e),
            )
            # Streamlit config failure is not critical
            logger.warning("Streamlit configuration failed: %s", e)

    def get_startup_status(self) -> Dict[str, Any]:
        """Get current startup status and metrics."""
        return {
            "initialized": self.is_initialized,
            "startup_time": self.startup_time,
            "services": {
                name: {
                    "initialized": status.initialized,
                    "initialization_time": status.initialization_time,
                    "error": status.error,
                    "dependencies": status.dependencies or [],
                }
                for name, status in self.services.items()
            },
            "total_services": len(self.services),
            "successful_services": sum(
                1 for s in self.services.values() if s.initialized
            ),
            "failed_services": sum(
                1 for s in self.services.values() if not s.initialized
            ),
        }

    def validate_services(self) -> List[str]:
        """Validate that all critical services are properly initialized.

        Returns:
            List of validation errors
        """
        errors = []

        critical_services = ["logging", "environment", "llm_service", "vector_store"]

        for service_name in critical_services:
            if service_name not in self.services:
                errors.append(f"Critical service '{service_name}' not found")
            elif not self.services[service_name].initialized:
                error_msg = self.services[service_name].error or "Unknown error"
                errors.append(
                    f"Critical service '{service_name}' failed to initialize: {error_msg}"
                )

        return errors


# Global startup manager instance
_startup_manager: Optional[ApplicationStartup] = None
_startup_manager_lock = threading.Lock()


def get_startup_manager() -> ApplicationStartup:
    """Get the global startup manager instance in a thread-safe manner."""
    # Use a function attribute instead of global
    if not hasattr(get_startup_manager, "_startup_manager"):
        with _startup_manager_lock:
            if not hasattr(get_startup_manager, "_startup_manager"):
                get_startup_manager._startup_manager = ApplicationStartup()
    return get_startup_manager._startup_manager


def initialize_application(user_api_key: str = "") -> StartupResult:
    """Initialize the application using the startup manager.

    Args:
        user_api_key: User-provided API key for LLM service

    Returns:
        StartupResult: Result of the startup process
    """
    startup_manager = get_startup_manager()
    return startup_manager.initialize_application(user_api_key)


def get_application_status() -> Dict[str, Any]:
    """Get current application startup status."""
    startup_manager = get_startup_manager()
    return startup_manager.get_startup_status()


def validate_application() -> List[str]:
    """Validate that the application is properly initialized.

    Returns:
        List of validation errors
    """
    startup_manager = get_startup_manager()
    return startup_manager.validate_services()
