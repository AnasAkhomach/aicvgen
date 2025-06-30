"""Module for the dependency injection container."""

import threading
from typing import Optional

from dependency_injector import (
    containers,
    providers,
)  # pylint: disable=c-extension-no-member
import google.generativeai as genai

from ..config.settings import get_config
from ..services.llm_service import EnhancedLLMService
from ..services.llm_caching_service import get_llm_caching_service
from ..services.llm_client import LLMClient
from ..services.llm_retry_handler import LLMRetryHandler
from ..services.llm_api_key_manager import LLMApiKeyManager
from ..services.llm_retry_service import LLMRetryService
from ..services.rate_limiter import get_rate_limiter
from ..agents.parser_agent import ParserAgent
from ..agents.cv_analyzer_agent import CVAnalyzerAgent
from ..agents.enhanced_content_writer import EnhancedContentWriterAgent
from ..agents.cleaning_agent import CleaningAgent
from ..agents.quality_assurance_agent import QualityAssuranceAgent
from ..agents.formatter_agent import FormatterAgent
from ..agents.research_agent import ResearchAgent
from ..services.vector_store_service import VectorStoreService
from ..services.progress_tracker import ProgressTracker
from ..templates.content_templates import ContentTemplateManager


def create_configured_llm_model(api_key: str, model_name: str) -> genai.GenerativeModel:
    """Create a GenerativeModel with proper API key configuration."""
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name=model_name)


class Container(
    containers.DeclarativeContainer
):  # pylint: disable=c-extension-no-member
    """Dependency injection container for the application."""

    config = providers.Singleton(get_config)  # pylint: disable=c-extension-no-member

    template_manager = providers.Singleton(  # pylint: disable=c-extension-no-member
        ContentTemplateManager,
        prompt_directory=providers.Callable(
            str, config.provided.prompts_directory
        ),  # pylint: disable=c-extension-no-member
    )

    # LLM Service Stack
    llm_model = providers.Singleton(  # pylint: disable=c-extension-no-member
        create_configured_llm_model,
        api_key=config.provided.llm.gemini_api_key_primary,
        model_name=config.provided.llm_settings.default_model,
    )

    llm_client = providers.Singleton(  # pylint: disable=c-extension-no-member
        LLMClient,
        llm_model=llm_model,
    )

    llm_retry_handler = providers.Singleton(  # pylint: disable=c-extension-no-member
        LLMRetryHandler,
        llm_client=llm_client,
    )

    advanced_cache = providers.Singleton(get_llm_caching_service)

    llm_api_key_manager = providers.Singleton(
        LLMApiKeyManager,
        settings=config,
        llm_client=llm_client,
        user_api_key=providers.Object(None),
    )

    rate_limiter = providers.Singleton(get_rate_limiter)

    llm_retry_service = providers.Singleton(
        LLMRetryService,
        llm_retry_handler=llm_retry_handler,
        api_key_manager=llm_api_key_manager,
        rate_limiter=rate_limiter,
        timeout=config.provided.llm.request_timeout,
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
        VectorStoreService, settings=config.provided
    )

    progress_tracker = providers.Factory(  # pylint: disable=c-extension-no-member
        ProgressTracker
    )

    # Agent Providers
    parser_agent = providers.Factory(  # pylint: disable=c-extension-no-member
        ParserAgent,
        llm_service=llm_service,
        vector_store_service=vector_store_service,
        template_manager=template_manager,
        settings=providers.Object({}),  # pylint: disable=c-extension-no-member
        session_id=providers.Object("default"),  # pylint: disable=c-extension-no-member
    )

    cv_analyzer_agent = providers.Factory(  # pylint: disable=c-extension-no-member
        CVAnalyzerAgent,
        llm_service=llm_service,
        session_id=providers.Object("default"),  # pylint: disable=c-extension-no-member
    )

    enhanced_content_writer_agent = (
        providers.Factory(  # pylint: disable=c-extension-no-member
            EnhancedContentWriterAgent,
            llm_service=llm_service,
            template_manager=template_manager,
            settings=providers.Object({}),  # pylint: disable=c-extension-no-member
            session_id=providers.Object(
                "default"
            ),  # pylint: disable=c-extension-no-member
        )
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
        settings=providers.Object({}),  # pylint: disable=c-extension-no-member
        template_manager=template_manager,
        session_id=providers.Object("default"),  # pylint: disable=c-extension-no-member
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
                    cls._instance = Container()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (mainly for testing)."""
        with cls._lock:
            cls._instance = None


def get_container() -> Container:
    """Returns the singleton instance of the DI container."""
    return ContainerSingleton.get_instance()
