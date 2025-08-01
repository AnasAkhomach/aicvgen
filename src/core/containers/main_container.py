"""Module for the dependency injection container."""

import threading
from pathlib import Path
from typing import Optional

from dependency_injector import containers, providers

# Agent imports removed - all agents now created through AgentFactory
from src.config.logging_config import get_logger, get_structured_logger
from src.config.settings import get_config
from src.core.factories import AgentFactory
from src.core.facades.cv_generation_facade import CvGenerationFacade
from src.core.facades.cv_template_manager_facade import CVTemplateManagerFacade
from src.core.facades.cv_vector_store_facade import CVVectorStoreFacade
from src.core.managers.workflow_manager import WorkflowManager
from src.services.cv_template_loader_service import CVTemplateLoaderService
from src.services.error_recovery import ErrorRecoveryService
from src.services.llm.gemini_client import GeminiClient
from src.services.llm.llm_client_interface import LLMClientInterface
from src.services.llm_api_key_manager import LLMApiKeyManager
from src.services.llm_cv_parser_service import LLMCVParserService

from src.services.llm_service import EnhancedLLMService
from src.services.progress_tracker import ProgressTracker
from src.services.session_manager import SessionManager
from src.services.vector_store_service import VectorStoreService
from src.core.state_manager import StateManager

# Workflow graph creation is now handled directly in WorkflowManager
# from src.orchestration.graphs.main_graph import create_cv_workflow_graph_with_di
from src.templates.content_templates import ContentTemplateManager

logger = get_logger(__name__)


def validate_prompts_directory(config_prompts_dir: str) -> str:
    """Validate prompts directory path and provide graceful fallback.
    Args:
        config_prompts_dir: The prompts directory path from configuration
    Returns:
        str: Validated prompts directory path
    Raises:
        RuntimeError: If no valid prompts directory can be found
    """
    # Convert to Path object for easier manipulation
    prompts_path = Path(config_prompts_dir)

    # Check if the configured path exists
    if prompts_path.exists() and prompts_path.is_dir():
        logger.info(
            "Prompts directory validated",
            extra={"directory": str(prompts_path.resolve())},
        )
        return str(prompts_path)

    # Try fallback paths
    fallback_paths = [
        Path("data/prompts"),
        Path("./data/prompts"),
        Path.cwd() / "data" / "prompts",
    ]

    for fallback_path in fallback_paths:
        if fallback_path.exists() and fallback_path.is_dir():
            logger.warning(
                "Using fallback prompts directory",
                extra={
                    "configured_path": config_prompts_dir,
                    "fallback_path": str(fallback_path.resolve()),
                },
            )
            return str(fallback_path)

    # Create the configured directory if it doesn't exist
    try:
        prompts_path.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Created missing prompts directory",
            extra={"directory": str(prompts_path.resolve())},
        )
        return str(prompts_path)
    except (OSError, PermissionError) as e:
        logger.error(
            "Failed to create prompts directory",
            extra={"directory": config_prompts_dir, "error": str(e)},
        )
        raise RuntimeError(
            f"Cannot access or create prompts directory: {config_prompts_dir}. "
            f"Please ensure the directory exists and is accessible."
        ) from e


# Removed _get_current_session_id helper function as part of REM-CORE-001 refactoring


def _get_agent_settings_dict() -> dict:
    """Helper function to get agent settings as a dictionary.

    This function is used to provide agent settings to agent providers
    without circular reference issues.
    """
    config = get_config()
    return config.agent_settings.model_dump()


class Container(
    containers.DeclarativeContainer
):  # pylint: disable=c-extension-no-member
    """Dependency injection container for the application.

    This class enforces singleton behavior and should not be instantiated directly.
    Use get_container() function instead.
    """

    _singleton_key = object()  # Private key for singleton creation

    def __new__(cls, *args, singleton_key=None, **kwargs):
        """Prevent direct instantiation unless proper singleton key is provided."""
        if singleton_key is not cls._singleton_key:
            raise RuntimeError(
                "Container cannot be instantiated directly. "
                "Use get_container() function instead."
            )
        return object.__new__(cls)

    config = providers.Singleton(get_config)  # pylint: disable=c-extension-no-member

    template_manager = providers.Singleton(  # pylint: disable=c-extension-no-member
        ContentTemplateManager,
        prompt_directory=providers.Callable(
            validate_prompts_directory, config.provided.paths.prompts_directory
        ),  # pylint: disable=c-extension-no-member
    )

    # LLM Service Stack with Direct Class Instantiation
    # Register LLMClientInterface with GeminiClient implementation
    llm_client = providers.Singleton(  # pylint: disable=c-extension-no-member
        GeminiClient,
        api_key=config.provided.llm.gemini_api_key_primary,
        model_name=config.provided.llm_settings.default_model,
    )

    # Lazy initialization for interdependent services
    llm_api_key_manager = providers.Singleton(
        LLMApiKeyManager,
        settings=config,
        llm_client=llm_client,
        user_api_key=providers.Object(None),
    )

    llm_service = providers.Singleton(  # pylint: disable=c-extension-no-member
        EnhancedLLMService,
        settings=config,
        llm_client=llm_client,
        api_key_manager=llm_api_key_manager,
    )

    vector_store_service = providers.Singleton(  # pylint: disable=c-extension-no-member
        VectorStoreService,
        vector_config=config.provided.vector_db,
        logger=providers.Callable(get_structured_logger, "vector_store_service"),
    )

    progress_tracker = providers.Singleton(  # pylint: disable=c-extension-no-member
        ProgressTracker,
        logger=providers.Callable(get_structured_logger, "progress_tracker"),
    )

    # Session Manager Service
    session_manager = providers.Singleton(  # pylint: disable=c-extension-no-member
        SessionManager,
        settings=config,
        logger=providers.Callable(get_structured_logger, "session_manager"),
    )

    # State Manager Service
    state_manager = providers.Singleton(  # pylint: disable=c-extension-no-member
        StateManager,
    )

    # Error Recovery Service
    error_recovery_service = (
        providers.Singleton(  # pylint: disable=c-extension-no-member
            ErrorRecoveryService,
            logger=providers.Callable(get_structured_logger, "error_recovery"),
        )
    )

    # CV Template Loader Service
    cv_template_loader_service = (
        providers.Singleton(  # pylint: disable=c-extension-no-member
            CVTemplateLoaderService,
        )
    )

    # LLM CV Parser Service
    llm_cv_parser_service = providers.Factory(  # pylint: disable=c-extension-no-member
        LLMCVParserService,
        llm_service=llm_service,
        settings=config,
        template_manager=template_manager,
    )

    # Agent Factory with SessionManager integration
    agent_factory = providers.Singleton(  # pylint: disable=c-extension-no-member
        AgentFactory,
        llm_service=llm_service,
        template_manager=template_manager,
        vector_store_service=vector_store_service,
        llm_cv_parser_service=llm_cv_parser_service,
        session_id_provider=session_manager.provided.get_current_session_id,
    )

    # Agent Providers - session_id passed as runtime argument
    cv_analyzer_agent = providers.Factory(  # pylint: disable=c-extension-no-member
        agent_factory.provided.create_cv_analyzer_agent,
    )

    key_qualifications_writer_agent = providers.Factory(
        agent_factory.provided.create_key_qualifications_writer_agent,
    )

    professional_experience_writer_agent = providers.Factory(
        agent_factory.provided.create_professional_experience_writer_agent,
    )

    projects_writer_agent = providers.Factory(
        agent_factory.provided.create_projects_writer_agent,
    )

    executive_summary_writer_agent = providers.Factory(
        agent_factory.provided.create_executive_summary_writer_agent,
    )

    # Cleaning Agent
    cleaning_agent = providers.Factory(
        agent_factory.provided.create_cleaning_agent,
    )

    quality_assurance_agent = providers.Factory(
        agent_factory.provided.create_quality_assurance_agent,
    )

    formatter_agent = providers.Factory(
        agent_factory.provided.create_formatter_agent,
    )

    research_agent = providers.Factory(
        agent_factory.provided.create_research_agent,
    )

    job_description_parser_agent = providers.Factory(
        agent_factory.provided.create_job_description_parser_agent,
    )

    user_cv_parser_agent = providers.Factory(
        agent_factory.provided.create_user_cv_parser_agent,
    )

    # Updater Agent Providers
    key_qualifications_updater_agent = providers.Factory(
        agent_factory.provided.create_key_qualifications_updater_agent,
    )

    professional_experience_updater_agent = providers.Factory(
        agent_factory.provided.create_professional_experience_updater_agent,
    )

    projects_updater_agent = providers.Factory(
        agent_factory.provided.create_projects_updater_agent,
    )

    executive_summary_updater_agent = providers.Factory(
        agent_factory.provided.create_executive_summary_updater_agent,
    )

    # WorkflowManager Singleton - now uses DI container directly
    workflow_manager = providers.Singleton(
        WorkflowManager,
        cv_template_loader_service=cv_template_loader_service,
        session_manager=session_manager,
        container=providers.Callable(lambda: ContainerSingleton.get_instance()),
    )

    # Facade providers
    cv_template_manager_facade = providers.Singleton(
        CVTemplateManagerFacade,
        template_manager=template_manager,
    )

    cv_vector_store_facade = providers.Singleton(
        CVVectorStoreFacade,
        vector_db=vector_store_service,
    )

    cv_generation_facade = providers.Singleton(
        CvGenerationFacade,
        workflow_manager=workflow_manager,
        user_cv_parser_agent=user_cv_parser_agent,
        template_facade=cv_template_manager_facade,
        vector_store_facade=cv_vector_store_facade,
    )


class ContainerSingleton:
    """Thread-safe singleton for the DI container."""

    _instance: Optional["Container"] = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "Container":
        """Get the singleton instance of the container."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    # Create Container instance with singleton key
                    cls._instance = Container(singleton_key=Container._singleton_key)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (mainly for testing)."""
        with cls._lock:
            cls._instance = None


def get_container() -> Container:
    """Returns the singleton instance of the DI container."""
    return ContainerSingleton.get_instance()
