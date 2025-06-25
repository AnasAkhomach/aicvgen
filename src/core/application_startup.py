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

import google.generativeai as genai

from ..config.logging_config import setup_logging, get_structured_logger
from ..config.settings import Settings, get_config as settings_factory
from ..services.llm_service import (
    EnhancedLLMService,
    get_advanced_cache,
    AdvancedCache,
)
from ..services.vector_store_service import get_vector_store_service, VectorStoreService
from ..services.session_manager import get_session_manager, SessionManager
from ..services.error_recovery import ErrorRecoveryService
from ..utils.exceptions import ConfigurationError, ServiceInitializationError
from ..core.dependency_injection import get_container, DependencyContainer
from ..agents.parser_agent import ParserAgent
from ..agents.research_agent import ResearchAgent
from ..agents.quality_assurance_agent import QualityAssuranceAgent
from ..agents.enhanced_content_writer import (
    EnhancedContentWriterAgent as WriterAgent,
)
from ..orchestration.cv_workflow_graph import CVWorkflowGraph
from ..services.llm_client import LLMClient
from ..services.llm_retry_handler import LLMRetryHandler
from ..services.progress_tracker import ProgressTracker
from ..templates.content_templates import ContentTemplateManager


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


def create_llm_client(settings: Settings) -> LLMClient:
    """Factory function to create an LLMClient instance."""
    # Configure the generative AI model with the API key from settings
    if not settings.llm.gemini_api_key_primary:
        raise ConfigurationError("Gemini API key is not configured.")
    genai.configure(api_key=settings.llm.gemini_api_key_primary)

    # Create the generative model instance
    llm_model = genai.GenerativeModel(settings.llm_settings.default_model)

    # Create and return the LLMClient
    return LLMClient(llm_model=llm_model)


class ApplicationStartup:
    """Manages application startup sequence and service initialization."""

    def __init__(self):
        self.services: Dict[str, ServiceStatus] = {}
        self.startup_time = None
        self.is_initialized = False
        self.container = get_container()
        self._shutdown_hook_registered = False

    def _register_dependencies(self):
        """Register all application dependencies with the container."""
        try:
            self.container.get_by_name("settings")
            logger.debug("Dependencies already registered. Skipping.")
            return
        except ValueError:
            logger.info("Registering application dependencies.")

        # Configuration
        self.container.register_singleton(
            "settings", Settings, factory=settings_factory
        )

        # Core Services
        self.container.register_singleton(
            "llm_client",
            LLMClient,
            factory=create_llm_client,
            dependencies=["settings"],
        )
        self.container.register_singleton(
            "llm_retry_handler",
            LLMRetryHandler,
            dependencies=["llm_client"],
        )
        self.container.register_singleton(
            "cache", AdvancedCache, factory=get_advanced_cache
        )
        self.container.register_singleton(
            "llm_service",
            EnhancedLLMService,
            dependencies=["settings", "llm_client", "llm_retry_handler", "cache"],
        )
        self.container.register_singleton(
            "vector_store_service",
            VectorStoreService,
            factory=get_vector_store_service,
        )
        # Alias for compatibility with ResearchAgent constructor
        self.container.register_singleton(
            "vector_db",
            VectorStoreService,
            factory=get_vector_store_service,
        )
        self.container.register_singleton(
            "session_manager",
            SessionManager,
            factory=get_session_manager,
        )
        self.container.register_singleton("template_manager", ContentTemplateManager)

        # Create ErrorRecoveryService with proper logger
        def create_error_recovery_service():
            from ..config.logging_config import get_structured_logger

            return ErrorRecoveryService(logger=get_structured_logger("error_recovery"))

        self.container.register_singleton(
            "error_recovery_service",
            ErrorRecoveryService,
            factory=create_error_recovery_service,
        )

        # Session-scoped services
        self.container.register_session("progress_tracker", ProgressTracker)

        # Agents (Session-scoped)
        self.container.register_session(
            "parser_agent",
            ParserAgent,
            dependencies=[
                "llm_service",
                "vector_store_service",
                "progress_tracker",
                "settings",
                "template_manager",
            ],
        )
        self.container.register_session(
            "research_agent",
            ResearchAgent,
            dependencies=[
                "llm_service",
                "progress_tracker",
                "vector_db",
                "settings",
                "template_manager",
            ],
        )
        self.container.register_session(
            "writer_agent",
            WriterAgent,
            dependencies=[
                "llm_service",
                "progress_tracker",
                "parser_agent",
                "settings",
            ],
        )
        self.container.register_session(
            "qa_agent",
            QualityAssuranceAgent,
            dependencies=[
                "llm_service",
                "progress_tracker",
                "template_manager",
            ],
        )

        # Orchestration (Session-scoped)
        self.container.register_session(
            "cv_workflow_graph",
            CVWorkflowGraph,
            dependencies=[
                "parser_agent",
                "research_agent",
                "writer_agent",
                "session_manager",
                "settings",
            ],
        )
        logger.info("All application dependencies registered successfully.")

    def initialize_application(self, user_api_key: str = "") -> StartupResult:
        """Initialize the application with proper startup sequence."""
        _ = user_api_key  # Mark as intentionally unused
        start_time = time.time()
        errors = []

        logger.info("Starting application initialization")

        try:
            # Phase 0: Register Dependencies
            self._register_dependencies()

            # Phase 1: Core Infrastructure
            setup_logging()
            logger.info("Service 'logging' initialized successfully.")

            # Trigger initialization by getting services from the container
            self.container.get_by_name("settings")
            logger.info("Service 'environment' (settings) initialized successfully.")

            self._ensure_directories()

            # Phase 2: External Services
            self.container.get_by_name("llm_service")
            logger.info("Service 'llm_service' initialized successfully.")

            self.container.get_by_name("vector_store_service")
            logger.info("Service 'vector_store_service' initialized successfully.")

            # Phase 3: Session Management
            self.container.get_by_name("session_manager")
            logger.info("Service 'session_manager' initialized successfully.")

            self.is_initialized = True
            logger.info("Application initialization successful")

        except (ConfigurationError, ServiceInitializationError) as e:
            logger.error(f"Application initialization failed: {e}", exc_info=True)
            errors.append(str(e))
            self.is_initialized = False
        except Exception as e:
            logger.error(
                f"An unexpected error occurred during startup: {e}", exc_info=True
            )
            errors.append(f"An unexpected error occurred: {e}")
            self.is_initialized = False

        end_time = time.time()
        self.startup_time = end_time - start_time

        result = StartupResult(
            success=self.is_initialized,
            total_time=self.startup_time,
            services=self.services,  # This is now sparsely populated, but kept for compatibility
            errors=errors,
            timestamp=datetime.now(),
        )

        logger.info(f"Application startup finished in {self.startup_time:.2f} seconds.")
        return result

    def _ensure_directories(self):
        """Ensure all necessary directories exist."""
        start_service_time = time.time()
        try:
            settings: Settings = self.container.get_by_name("settings")

            dirs_to_check = [
                settings.logging.log_directory,
                settings.data_directory,
                settings.vector_db.persist_directory,
                settings.prompts_directory,
                settings.output.pdf_output_directory,
                settings.sessions_directory,
            ]

            for directory in dirs_to_check:
                try:
                    # Path objects can be used directly with os.makedirs
                    if not os.path.exists(directory):
                        os.makedirs(directory)
                        logger.info(f"Created directory: {directory}")
                except OSError as e:
                    raise ConfigurationError(
                        f"Failed to create directory {directory}: {e}"
                    )

            status = ServiceStatus(
                name="directories",
                initialized=True,
                initialization_time=time.time() - start_service_time,
            )
            self.services["directories"] = status
            logger.info("Service 'directories' initialized successfully.")
        except Exception as e:
            logger.error(
                f"Failed to initialize service 'directories': {e}", exc_info=True
            )
            status = ServiceStatus(
                name="directories",
                initialized=False,
                initialization_time=time.time() - start_service_time,
                error=str(e),
            )
            self.services["directories"] = status
            raise ServiceInitializationError(
                f"Failed to initialize service 'directories': {e}"
            )

    def validate_application(self) -> List[str]:
        """
        Validates that critical services are initialized and ready.
        Returns a list of validation errors, or an empty list if successful.
        """
        errors = []
        if not self.is_initialized:
            errors.append("Application initialization flag is not set.")

        critical_services = [
            "settings",
            "llm_service",
            "vector_store_service",
            "session_manager",
        ]

        for service_name in critical_services:
            try:
                self.container.get_by_name(service_name)
            except Exception as e:
                error_msg = f"Critical service '{service_name}' failed validation: {e}"
                logger.error(error_msg)
                errors.append(error_msg)

        return errors

    def shutdown_application(self):
        """Gracefully shut down all registered services."""
        logger.info("Starting application shutdown sequence.")

        # List of services that are known to require a clean shutdown.
        shutdown_order = ["vector_store_service"]

        for service_name in shutdown_order:
            try:
                # This assumes the container will not re-create the service if it exists.
                service_instance = self.container.get_by_name(service_name)
                if hasattr(service_instance, "shutdown") and callable(
                    service_instance.shutdown
                ):
                    logger.info(f"Shutting down service: {service_name}")
                    service_instance.shutdown()
            except ValueError:
                # This can happen if the service was never initialized, which is fine.
                logger.debug(
                    f"Service '{service_name}' was not found in container, skipping shutdown."
                )
            except Exception as e:
                logger.error(
                    f"Error shutting down service '{service_name}': {e}", exc_info=True
                )

        logger.info("Application shutdown sequence complete.")


# Singleton instance of the startup manager
_startup_manager_instance = None
_startup_lock = threading.Lock()


def get_startup_manager() -> ApplicationStartup:
    """Get the singleton instance of the application startup manager."""
    global _startup_manager_instance
    if _startup_manager_instance is None:
        with _startup_lock:
            if _startup_manager_instance is None:
                _startup_manager_instance = ApplicationStartup()
    return _startup_manager_instance
