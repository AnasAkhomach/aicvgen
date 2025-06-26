"""Module for the dependency injection container."""

from dependency_injector import containers, providers
import google.generativeai as genai

from src.config.settings import get_config
from src.services.llm_service import EnhancedLLMService, get_advanced_cache
from src.services.llm_client import LLMClient
from src.services.llm_retry_handler import LLMRetryHandler
from src.agents.parser_agent import ParserAgent
from src.agents.cv_analyzer_agent import CVAnalyzerAgent
from src.agents.enhanced_content_writer import EnhancedContentWriterAgent
from src.agents.cleaning_agent import CleaningAgent
from src.agents.quality_assurance_agent import QualityAssuranceAgent
from src.agents.formatter_agent import FormatterAgent
from src.agents.research_agent import ResearchAgent
from src.agents.specialized_agents import EnhancedParserAgent
from src.services.vector_store_service import VectorStoreService
from src.services.progress_tracker import ProgressTracker
from src.templates.content_templates import ContentTemplateManager


class Container(containers.DeclarativeContainer):
    """Dependency injection container for the application."""

    config = providers.Singleton(get_config)

    # Configure Google Generative AI
    # This is a one-time setup
    genai.configure(api_key=config.provided.llm.gemini_api_key_primary)

    template_manager = providers.Singleton(
        ContentTemplateManager,
        prompt_directory=config.provided.paths.prompts,
    )

    # LLM Service Stack
    llm_model = providers.Singleton(
        genai.GenerativeModel,
        model_name=config.provided.llm_settings.default_model,
    )

    llm_client = providers.Singleton(
        LLMClient,
        llm_model=llm_model,
    )

    llm_retry_handler = providers.Singleton(
        LLMRetryHandler,
        llm_client=llm_client,
    )

    advanced_cache = providers.Singleton(get_advanced_cache)

    llm_service = providers.Singleton(
        EnhancedLLMService,
        settings=config,
        llm_client=llm_client,
        llm_retry_handler=llm_retry_handler,
        cache=advanced_cache,
        timeout=config.provided.llm.timeout,
    )

    vector_store_service = providers.Singleton(
        VectorStoreService, settings=config.provided
    )

    progress_tracker = providers.Factory(
        ProgressTracker,
        enabled=config.provided.progress_tracker.enabled,
        log_interval=config.provided.progress_tracker.log_interval,
        max_history=config.provided.progress_tracker.max_history,
    )

    # Agent Providers
    parser_agent = providers.Factory(
        ParserAgent,
        llm_service=llm_service,
        vector_store_service=vector_store_service,
        template_manager=template_manager,
        settings=config.provided.agent_settings.parser,
    )

    cv_analyzer_agent = providers.Factory(
        CVAnalyzerAgent,
        llm_service=llm_service,
        template_manager=template_manager,
        settings=config.provided.agent_settings.cv_analyzer,
    )

    enhanced_content_writer_agent = providers.Factory(
        EnhancedContentWriterAgent,
        llm_service=llm_service,
        template_manager=template_manager,
        settings=config.provided.agent_settings.content_writer,
    )

    cleaning_agent = providers.Factory(
        CleaningAgent,
        llm_service=llm_service,
        template_manager=template_manager,
        settings=config.provided.agent_settings.cleaning,
    )

    quality_assurance_agent = providers.Factory(
        QualityAssuranceAgent,
        llm_service=llm_service,
        template_manager=template_manager,
        settings=config.provided.agent_settings.qa,
    )

    formatter_agent = providers.Factory(
        FormatterAgent,
        settings=config.provided.agent_settings.formatter,
        llm_service=llm_service,
        template_manager=template_manager,
    )

    research_agent = providers.Factory(
        ResearchAgent,
        llm_service=llm_service,
        vector_store_service=vector_store_service,
        template_manager=template_manager,
        settings=config.provided.agent_settings.research,
    )

    enhanced_parser_agent = providers.Factory(
        EnhancedParserAgent,
        parser_agent=parser_agent,
    )


_container: Container | None = None


def get_container() -> Container:
    """Returns the singleton instance of the container."""
    global _container
    if _container is None:
        _container = Container()
    return _container
