"""Module for the dependency injection container."""

import threading
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
from src.config.settings import get_config
from src.core.factories import AgentFactory, ServiceFactory, create_configured_llm_model
from src.orchestration.cv_workflow_graph import CVWorkflowGraph
from src.templates.content_templates import ContentTemplateManager


class Container(
    containers.DeclarativeContainer
):  # pylint: disable=c-extension-no-member
    """Dependency injection container for the application.

    This class enforces singleton behavior and should not be instantiated directly.
    Use get_container() function instead.
    """

    _creation_allowed = False

    def __new__(cls, *args, **kwargs):
        """Prevent direct instantiation unless explicitly allowed."""
        if not cls._creation_allowed:
            raise RuntimeError(
                "Container cannot be instantiated directly. "
                "Use get_container() function instead."
            )
        return object.__new__(cls)

    config = providers.Singleton(get_config)  # pylint: disable=c-extension-no-member

    template_manager = providers.Singleton(  # pylint: disable=c-extension-no-member
        ContentTemplateManager,
        prompt_directory=providers.Callable(
            str, config.provided.paths.prompts_directory
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
        ServiceFactory.create_vector_store_service, settings=config.provided
    )

    progress_tracker = providers.Factory(  # pylint: disable=c-extension-no-member
        ServiceFactory.create_progress_tracker
    )

    # Agent Factory
    agent_factory = providers.Singleton(  # pylint: disable=c-extension-no-member
        AgentFactory,
        llm_service=llm_service,
        template_manager=template_manager,
        vector_store_service=vector_store_service
    )

    # Agent Providers
    cv_analyzer_agent = providers.Factory(  # pylint: disable=c-extension-no-member
        CVAnalyzerAgent,
        llm_service=llm_service,
        session_id=providers.Object("default"),  # pylint: disable=c-extension-no-member
    )

    key_qualifications_writer_agent = providers.Factory(
        KeyQualificationsWriterAgent,
        llm_service=llm_service,
        template_manager=template_manager,
        settings=providers.Object({}),
        session_id=providers.Object("default"),
    )

    professional_experience_writer_agent = providers.Factory(
        ProfessionalExperienceWriterAgent,
        llm_service=llm_service,
        template_manager=template_manager,
        settings=providers.Object({}),
        session_id=providers.Object("default"),
    )

    projects_writer_agent = providers.Factory(
        ProjectsWriterAgent,
        llm_service=llm_service,
        template_manager=template_manager,
        settings=providers.Object({}),
        session_id=providers.Object("default"),
    )

    executive_summary_writer_agent = providers.Factory(
        ExecutiveSummaryWriterAgent,
        llm_service=llm_service,
        template_manager=template_manager,
        settings=providers.Object({}),
        session_id=providers.Object("default"),
    )

    cleaning_agent = providers.Factory(  # pylint: disable=c-extension-no-member
        CleaningAgent,
        llm_service=llm_service,
        template_manager=template_manager,
        settings=providers.Object({}),  # pylint: disable=c-extension-no-member
        session_id=providers.Object("default"),  # pylint: disable=c-extension-no-member
    )

    quality_assurance_agent = providers.Factory(
        QualityAssuranceAgent,
        llm_service=llm_service,
        template_manager=template_manager,
        settings=providers.Object({}),  # pylint: disable=c-extension-no-member
        session_id=providers.Object("default"),  # pylint: disable=c-extension-no-member
    )

    formatter_agent = providers.Factory(  # pylint: disable=c-extension-no-member
        FormatterAgent,
        template_manager=template_manager,
        settings=providers.Object({}),  # pylint: disable=c-extension-no-member
        session_id=providers.Object("default"),  # pylint: disable=c-extension-no-member
    )

    research_agent = providers.Factory(  # pylint: disable=c-extension-no-member
        ResearchAgent,
        llm_service=llm_service,
        vector_store_service=vector_store_service,
        template_manager=template_manager,
        settings=providers.Object({}),  # pylint: disable=c-extension-no-member
        session_id=providers.Object("default"),  # pylint: disable=c-extension-no-member
    )

    job_description_parser_agent = providers.Factory(
        JobDescriptionParserAgent,
        llm_service=llm_service,
        template_manager=template_manager,
        settings=providers.Object({}),
        session_id=providers.Object("default")
    )

    user_cv_parser_agent = providers.Factory(
        UserCVParserAgent,
        llm_service=llm_service,
        vector_store_service=vector_store_service,
        template_manager=template_manager,
        settings=providers.Object({}),
        session_id=providers.Object("default")
    )

    # CVWorkflowGraph Factory
    cv_workflow_graph = providers.Factory(
        CVWorkflowGraph,
        session_id=providers.Object("default"),
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
                    # Temporarily allow Container creation
                    Container._creation_allowed = True
                    try:
                        cls._instance = Container()
                    finally:
                        Container._creation_allowed = False
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (mainly for testing)."""
        with cls._lock:
            cls._instance = None


def get_container() -> Container:
    """Returns the singleton instance of the DI container."""
    return ContainerSingleton.get_instance()
