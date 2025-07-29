"""Module for the dependency injection container."""

import threading
from pathlib import Path
from typing import Optional

from dependency_injector import containers, providers

from src.agents.cleaning_agent import CleaningAgent
from src.agents.cv_analyzer_agent import CVAnalyzerAgent
from src.agents.executive_summary_updater_agent import ExecutiveSummaryUpdaterAgent
from src.agents.executive_summary_writer_agent import ExecutiveSummaryWriterAgent
from src.agents.formatter_agent import FormatterAgent
from src.agents.job_description_parser_agent import JobDescriptionParserAgent
from src.agents.key_qualifications_updater_agent import KeyQualificationsUpdaterAgent
from src.agents.key_qualifications_writer_agent import KeyQualificationsWriterAgent
from src.agents.professional_experience_updater_agent import (
    ProfessionalExperienceUpdaterAgent,
)
from src.agents.professional_experience_writer_agent import (
    ProfessionalExperienceWriterAgent,
)
from src.agents.projects_updater_agent import ProjectsUpdaterAgent
from src.agents.projects_writer_agent import ProjectsWriterAgent
from src.agents.quality_assurance_agent import QualityAssuranceAgent
from src.agents.research_agent import ResearchAgent
from src.agents.user_cv_parser_agent import UserCVParserAgent
from src.config.logging_config import get_logger, get_structured_logger
from src.config.settings import get_config
from src.core.factories import AgentFactory
from src.core.managers.workflow_manager import WorkflowManager
from src.services.cv_template_loader_service import CVTemplateLoaderService
from src.services.error_recovery import ErrorRecoveryService
from src.services.llm.gemini_client import GeminiClient
from src.services.llm.llm_client_interface import LLMClientInterface
from src.services.llm_api_key_manager import LLMApiKeyManager
from src.services.llm_caching_service import LLMCachingService
from src.services.llm_cv_parser_service import LLMCVParserService
from src.services.llm_retry_handler import LLMRetryHandler
from src.services.llm_retry_service import LLMRetryService
from src.services.llm_service import EnhancedLLMService
from src.services.progress_tracker import ProgressTracker
from src.services.rate_limiter import RateLimiter
from src.services.session_manager import SessionManager
from src.services.vector_store_service import VectorStoreService

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

    llm_retry_handler = providers.Singleton(  # pylint: disable=c-extension-no-member
        LLMRetryHandler,
        llm_client=llm_client,
    )

    advanced_cache = providers.Singleton(
        LLMCachingService,
        max_size=1000,
        default_ttl_hours=24,
        persist_file=None,
    )

    rate_limiter = providers.Singleton(
        RateLimiter,
        logger=providers.Callable(get_structured_logger, "rate_limiter"),
        config=config,
    )

    # Lazy initialization for interdependent services
    llm_api_key_manager = providers.Singleton(
        LLMApiKeyManager,
        settings=config,
        llm_client=llm_client,
        user_api_key=providers.Object(None),
    )

    llm_retry_service = providers.Singleton(
        LLMRetryService,
        llm_retry_handler=llm_retry_handler,
        api_key_manager=llm_api_key_manager,
        rate_limiter=rate_limiter,
        timeout=config.provided.llm.retry.request_timeout,
        model_name=config.provided.llm_settings.default_model,
    )

    llm_service = providers.Singleton(  # pylint: disable=c-extension-no-member
        EnhancedLLMService,
        settings=config,
        caching_service=advanced_cache,
        api_key_manager=llm_api_key_manager,
        retry_service=llm_retry_service,
        rate_limiter=rate_limiter,
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
        session_id_provider=session_manager.provided.get_current_session_id,
    )

    # Agent Providers - session_id passed as runtime argument
    cv_analyzer_agent = providers.Factory(  # pylint: disable=c-extension-no-member
        CVAnalyzerAgent,
        llm_service=llm_service,
        settings=providers.Callable(
            _get_agent_settings_dict
        ),  # pylint: disable=c-extension-no-member
    )

    key_qualifications_writer_agent = providers.Factory(
        agent_factory.provided.create_key_qualifications_writer_agent,
        settings=providers.Callable(_get_agent_settings_dict),
    )

    professional_experience_writer_agent = providers.Factory(
        ProfessionalExperienceWriterAgent,
        llm_service=llm_service,
        template_manager=template_manager,
        settings=providers.Callable(_get_agent_settings_dict),
    )

    projects_writer_agent = providers.Factory(
        ProjectsWriterAgent,
        llm_service=llm_service,
        template_manager=template_manager,
        settings=providers.Callable(_get_agent_settings_dict),
    )

    executive_summary_writer_agent = providers.Factory(
        ExecutiveSummaryWriterAgent,
        llm_service=llm_service,
        template_manager=template_manager,
        settings=providers.Callable(_get_agent_settings_dict),
    )

    cleaning_agent = providers.Factory(  # pylint: disable=c-extension-no-member
        CleaningAgent,
        llm_service=llm_service,
        template_manager=template_manager,
        settings=providers.Callable(
            _get_agent_settings_dict
        ),  # pylint: disable=c-extension-no-member
    )

    quality_assurance_agent = providers.Factory(
        QualityAssuranceAgent,
        llm_service=llm_service,
        template_manager=template_manager,
        settings=providers.Callable(
            _get_agent_settings_dict
        ),  # pylint: disable=c-extension-no-member
    )

    formatter_agent = providers.Factory(  # pylint: disable=c-extension-no-member
        FormatterAgent,
        template_manager=template_manager,
        settings=providers.Callable(
            _get_agent_settings_dict
        ),  # pylint: disable=c-extension-no-member
    )

    research_agent = providers.Factory(  # pylint: disable=c-extension-no-member
        ResearchAgent,
        llm_service=llm_service,
        vector_store_service=vector_store_service,
        template_manager=template_manager,
        settings=providers.Callable(
            _get_agent_settings_dict
        ),  # pylint: disable=c-extension-no-member
    )

    job_description_parser_agent = providers.Factory(
        JobDescriptionParserAgent,
        llm_service=llm_service,
        llm_cv_parser_service=llm_cv_parser_service,
        template_manager=template_manager,
        settings=providers.Callable(_get_agent_settings_dict),
    )

    user_cv_parser_agent = providers.Factory(
        UserCVParserAgent,
        llm_service=llm_service,
        vector_store_service=vector_store_service,
        template_manager=template_manager,
        settings=providers.Callable(_get_agent_settings_dict),
    )

    # Updater Agent Providers
    key_qualifications_updater_agent = providers.Factory(
        KeyQualificationsUpdaterAgent,
    )

    professional_experience_updater_agent = providers.Factory(
        ProfessionalExperienceUpdaterAgent,
    )

    projects_updater_agent = providers.Factory(
        ProjectsUpdaterAgent,
    )

    executive_summary_updater_agent = providers.Factory(
        ExecutiveSummaryUpdaterAgent,
    )

    # WorkflowManager Singleton - now uses DI container directly
    workflow_manager = providers.Singleton(
        WorkflowManager,
        container=providers.Self(),
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
