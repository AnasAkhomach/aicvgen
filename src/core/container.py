"""Module for the dependency injection container."""

import threading
from pathlib import Path
from typing import Optional

from dependency_injector import containers, providers

from src.agents.cleaning_agent import CleaningAgent
from src.agents.cv_analyzer_agent import CVAnalyzerAgent
from src.agents.executive_summary_writer_agent import ExecutiveSummaryWriterAgent
from src.agents.formatter_agent import FormatterAgent
from src.agents.job_description_parser_agent import JobDescriptionParserAgent
from src.agents.key_qualifications_writer_agent import KeyQualificationsWriterAgent
from src.agents.professional_experience_writer_agent import ProfessionalExperienceWriterAgent
from src.agents.projects_writer_agent import ProjectsWriterAgent
from src.agents.quality_assurance_agent import QualityAssuranceAgent
from src.agents.research_agent import ResearchAgent
from src.agents.user_cv_parser_agent import UserCVParserAgent
from src.config.logging_config import get_logger
from src.config.settings import get_config
from src.core.factories import AgentFactory, ServiceFactory, create_configured_llm_model
from src.core.workflow_manager import WorkflowManager
from src.orchestration.cv_workflow_graph import CVWorkflowGraph
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
            extra={"directory": str(prompts_path.resolve())}
        )
        return str(prompts_path)

    # Try fallback paths
    fallback_paths = [
        Path("data/prompts"),
        Path("./data/prompts"),
        Path.cwd() / "data" / "prompts"
    ]

    for fallback_path in fallback_paths:
        if fallback_path.exists() and fallback_path.is_dir():
            logger.warning(
                "Using fallback prompts directory",
                extra={
                    "configured_path": config_prompts_dir,
                    "fallback_path": str(fallback_path.resolve())
                }
            )
            return str(fallback_path)

    # Create the configured directory if it doesn't exist
    try:
        prompts_path.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Created missing prompts directory",
            extra={"directory": str(prompts_path.resolve())}
        )
        return str(prompts_path)
    except (OSError, PermissionError) as e:
        logger.error(
            "Failed to create prompts directory",
            extra={
                "directory": config_prompts_dir,
                "error": str(e)
            }
        )
        raise RuntimeError(
            f"Cannot access or create prompts directory: {config_prompts_dir}. "
            f"Please ensure the directory exists and is accessible."
        ) from e


def _get_current_session_id() -> str:
    """Helper function to get current session ID from Container.

    This function is used to avoid circular reference issues when
    defining providers within the Container class.
    """
    return Container.get_current_session_id()


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
    _current_session_id = "default"  # Default session ID

    def __new__(cls, *args, singleton_key=None, **kwargs):
        """Prevent direct instantiation unless proper singleton key is provided."""
        if singleton_key is not cls._singleton_key:
            raise RuntimeError(
                "Container cannot be instantiated directly. "
                "Use get_container() function instead."
            )
        return object.__new__(cls)

    @classmethod
    def set_session_id(cls, session_id: str) -> None:
        """Set the current session ID for all agent providers.

        Args:
            session_id: The session ID to use for agent instantiation
        """
        cls._current_session_id = session_id

    @classmethod
    def get_current_session_id(cls) -> str:
        """Get the current session ID.

        Returns:
            The current session ID
        """
        return cls._current_session_id

    config = providers.Singleton(get_config)  # pylint: disable=c-extension-no-member

    template_manager = providers.Singleton(  # pylint: disable=c-extension-no-member
        ContentTemplateManager,
        prompt_directory=providers.Callable(
            validate_prompts_directory, config.provided.paths.prompts_directory
        ),  # pylint: disable=c-extension-no-member
    )

    # LLM Service Stack with Lazy Initialization
    llm_model = providers.Singleton(  # pylint: disable=c-extension-no-member
        create_configured_llm_model,
        api_key=config.provided.llm.gemini_api_key_primary,
        model_name=config.provided.llm_settings.default_model,
    )

    llm_client = providers.Singleton(  # pylint: disable=c-extension-no-member
        ServiceFactory.create_llm_client,
        llm_model=llm_model,
    )

    llm_retry_handler = providers.Singleton(  # pylint: disable=c-extension-no-member
        ServiceFactory.create_llm_retry_handler,
        llm_client=llm_client,
    )

    advanced_cache = providers.Singleton(ServiceFactory.get_caching_service)

    rate_limiter = providers.Singleton(ServiceFactory.get_rate_limiter)

    # Lazy initialization for interdependent services
    llm_api_key_manager = providers.Singleton(
        ServiceFactory.create_llm_api_key_manager_lazy,
        settings=config,
        llm_client=llm_client,
        user_api_key=providers.Object(None),
    )

    llm_retry_service = providers.Singleton(
        ServiceFactory.create_llm_retry_service_lazy,
        llm_retry_handler=llm_retry_handler,
        api_key_manager=llm_api_key_manager,
        rate_limiter=rate_limiter,
        timeout=config.provided.llm.retry.request_timeout,
        model_name=config.provided.llm_settings.default_model,
    )

    llm_service = providers.Singleton(  # pylint: disable=c-extension-no-member
        ServiceFactory.create_enhanced_llm_service_lazy,
        settings=config,
        caching_service=advanced_cache,
        api_key_manager=llm_api_key_manager,
        retry_service=llm_retry_service,
        rate_limiter=rate_limiter,
    )

    vector_store_service = providers.Singleton(  # pylint: disable=c-extension-no-member
        ServiceFactory.create_vector_store_service, vector_config=config.provided.vector_db
    )

    progress_tracker = providers.Singleton(  # pylint: disable=c-extension-no-member
        ServiceFactory.create_progress_tracker
    )

    # Agent Factory
    agent_factory = providers.Singleton(  # pylint: disable=c-extension-no-member
        AgentFactory,
        llm_service=llm_service,
        template_manager=template_manager,
        vector_store_service=vector_store_service,
        session_id_provider=providers.Callable(_get_current_session_id)
    )

    # Agent Providers
    cv_analyzer_agent = providers.Factory(  # pylint: disable=c-extension-no-member
        CVAnalyzerAgent,
        llm_service=llm_service,
        session_id=providers.Callable(_get_current_session_id),  # pylint: disable=c-extension-no-member
    )

    key_qualifications_writer_agent = providers.Factory(
        KeyQualificationsWriterAgent,
        llm_service=llm_service,
        template_manager=template_manager,
        settings=providers.Callable(_get_agent_settings_dict),
        session_id=providers.Callable(_get_current_session_id),
    )

    professional_experience_writer_agent = providers.Factory(
        ProfessionalExperienceWriterAgent,
        llm_service=llm_service,
        template_manager=template_manager,
        settings=providers.Callable(_get_agent_settings_dict),
        session_id=providers.Callable(_get_current_session_id),
    )

    projects_writer_agent = providers.Factory(
        ProjectsWriterAgent,
        llm_service=llm_service,
        template_manager=template_manager,
        settings=providers.Callable(_get_agent_settings_dict),
        session_id=providers.Callable(_get_current_session_id),
    )

    executive_summary_writer_agent = providers.Factory(
        ExecutiveSummaryWriterAgent,
        llm_service=llm_service,
        template_manager=template_manager,
        settings=providers.Callable(_get_agent_settings_dict),
        session_id=providers.Callable(_get_current_session_id),
    )

    cleaning_agent = providers.Factory(  # pylint: disable=c-extension-no-member
        CleaningAgent,
        llm_service=llm_service,
        template_manager=template_manager,
        settings=providers.Callable(_get_agent_settings_dict),  # pylint: disable=c-extension-no-member
        session_id=providers.Callable(_get_current_session_id),  # pylint: disable=c-extension-no-member
    )

    quality_assurance_agent = providers.Factory(
        QualityAssuranceAgent,
        llm_service=llm_service,
        template_manager=template_manager,
        settings=providers.Callable(_get_agent_settings_dict),  # pylint: disable=c-extension-no-member
        session_id=providers.Callable(_get_current_session_id),  # pylint: disable=c-extension-no-member
    )

    formatter_agent = providers.Factory(  # pylint: disable=c-extension-no-member
        FormatterAgent,
        template_manager=template_manager,
        settings=providers.Callable(_get_agent_settings_dict),  # pylint: disable=c-extension-no-member
        session_id=providers.Callable(_get_current_session_id),  # pylint: disable=c-extension-no-member
    )

    research_agent = providers.Factory(  # pylint: disable=c-extension-no-member
        ResearchAgent,
        llm_service=llm_service,
        vector_store_service=vector_store_service,
        template_manager=template_manager,
        settings=providers.Callable(_get_agent_settings_dict),  # pylint: disable=c-extension-no-member
        session_id=providers.Callable(_get_current_session_id),  # pylint: disable=c-extension-no-member
    )

    job_description_parser_agent = providers.Factory(
        JobDescriptionParserAgent,
        llm_service=llm_service,
        template_manager=template_manager,
        settings=providers.Callable(_get_agent_settings_dict),
        session_id=providers.Callable(_get_current_session_id)
    )

    user_cv_parser_agent = providers.Factory(
        UserCVParserAgent,
        llm_service=llm_service,
        vector_store_service=vector_store_service,
        template_manager=template_manager,
        settings=providers.Callable(_get_agent_settings_dict),
        session_id=providers.Callable(_get_current_session_id)
    )

    # CVWorkflowGraph Factory
    cv_workflow_graph = providers.Factory(
        CVWorkflowGraph,
        session_id=providers.Callable(_get_current_session_id),
        job_description_parser_agent=job_description_parser_agent,
        user_cv_parser_agent=user_cv_parser_agent,
        research_agent=research_agent,
        cv_analyzer_agent=cv_analyzer_agent,
        key_qualifications_writer_agent=key_qualifications_writer_agent,
        professional_experience_writer_agent=professional_experience_writer_agent,
        projects_writer_agent=projects_writer_agent,
        executive_summary_writer_agent=executive_summary_writer_agent,
        qa_agent=quality_assurance_agent,
        formatter_agent=formatter_agent,
    )

    # WorkflowManager Singleton
    workflow_manager = providers.Singleton(
        WorkflowManager,
        cv_workflow_graph=cv_workflow_graph,
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
