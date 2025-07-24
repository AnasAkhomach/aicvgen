"""Factory module for creating service instances."""


from typing import Any, Optional

import google.generativeai as genai
from src.models.vector_store_config_interface import VectorStoreConfigInterface

from src.config.logging_config import get_structured_logger
from src.error_handling.exceptions import ServiceInitializationError
from src.services.llm_api_key_manager import LLMApiKeyManager

from src.services.llm_client import LLMClient
from src.services.llm_retry_handler import LLMRetryHandler
from src.services.llm_retry_service import LLMRetryService
from src.services.llm_service import EnhancedLLMService
from src.services.progress_tracker import ProgressTracker
from src.services.rate_limiter import get_rate_limiter
from src.services.vector_store_service import VectorStoreService

logger = get_structured_logger("service_factory")


def create_configured_llm_model(api_key: str, model_name: str) -> genai.GenerativeModel:
    """Create a GenerativeModel with proper API key configuration."""
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name=model_name)


class ServiceFactory:
    """Factory for creating service instances with dependency injection."""

    @staticmethod
    def create_llm_client(llm_model: genai.GenerativeModel) -> LLMClient:
        """Create an LLM client instance."""
        return LLMClient(llm_model=llm_model)

    @staticmethod
    def create_llm_retry_handler(llm_client: LLMClient) -> LLMRetryHandler:
        """Create an LLM retry handler instance."""
        return LLMRetryHandler(llm_client=llm_client)

    @staticmethod
    def create_llm_api_key_manager(
        settings: Any,
        llm_client: LLMClient,
        user_api_key: Optional[str] = None
    ) -> LLMApiKeyManager:
        """Create an LLM API key manager instance."""
        return LLMApiKeyManager(
            settings=settings,
            llm_client=llm_client,
            user_api_key=user_api_key
        )

    @staticmethod
    def create_llm_retry_service(
        llm_retry_handler: LLMRetryHandler,
        api_key_manager: LLMApiKeyManager,
        rate_limiter: Any,
        timeout: int,
        model_name: str
    ) -> LLMRetryService:
        """Create an LLM retry service instance."""
        return LLMRetryService(
            llm_retry_handler=llm_retry_handler,
            api_key_manager=api_key_manager,
            rate_limiter=rate_limiter,
            timeout=timeout,
            model_name=model_name
        )

    @staticmethod
    def create_enhanced_llm_service(
        settings: Any,
        caching_service: Any,
        api_key_manager: LLMApiKeyManager,
        retry_service: LLMRetryService,
        rate_limiter: Any
    ) -> EnhancedLLMService:
        """Create an enhanced LLM service instance."""
        return EnhancedLLMService(
            settings=settings,
            caching_service=caching_service,
            api_key_manager=api_key_manager,
            retry_service=retry_service,
            rate_limiter=rate_limiter
        )

    @staticmethod
    def create_vector_store_service(vector_config: VectorStoreConfigInterface) -> VectorStoreService:
        """Create a vector store service instance."""
        return VectorStoreService(vector_config=vector_config)

    @staticmethod
    def create_progress_tracker() -> ProgressTracker:
        """Create a progress tracker instance."""
        return ProgressTracker()



    # Lazy initialization methods for interdependent services
    @staticmethod
    def create_llm_api_key_manager_lazy(
        settings: Any,
        llm_client: LLMClient,
        user_api_key: Optional[str] = None
    ) -> LLMApiKeyManager:
        """Create an LLM API key manager with lazy initialization and validation."""
        try:
            logger.info("Creating LLM API key manager with lazy initialization")

            # Validate dependencies before creation
            if not settings:
                raise ServiceInitializationError(
                    "llm_api_key_manager",
                    "Settings dependency is None or invalid"
                )

            if not llm_client:
                raise ServiceInitializationError(
                    "llm_api_key_manager",
                    "LLM client dependency is None or invalid"
                )

            # Create the service with validated dependencies
            manager = LLMApiKeyManager(
                settings=settings,
                llm_client=llm_client,
                user_api_key=user_api_key
            )

            logger.info("LLM API key manager created successfully")
            return manager

        except Exception as e:
            logger.error(f"Failed to create LLM API key manager: {e}")
            raise ServiceInitializationError(
                "llm_api_key_manager",
                f"Initialization failed: {str(e)}"
            ) from e

    @staticmethod
    def create_llm_retry_service_lazy(
        llm_retry_handler: LLMRetryHandler,
        api_key_manager: LLMApiKeyManager,
        rate_limiter: Any,
        timeout: int,
        model_name: str
    ) -> LLMRetryService:
        """Create an LLM retry service with lazy initialization and validation."""
        try:
            logger.info("Creating LLM retry service with lazy initialization")

            # Validate dependencies before creation
            if not llm_retry_handler:
                raise ServiceInitializationError(
                    "llm_retry_service",
                    "LLM retry handler dependency is None or invalid"
                )

            if not api_key_manager:
                raise ServiceInitializationError(
                    "llm_retry_service",
                    "API key manager dependency is None or invalid"
                )

            if timeout <= 0:
                raise ServiceInitializationError(
                    "llm_retry_service",
                    f"Invalid timeout value: {timeout}"
                )

            if not model_name or not model_name.strip():
                raise ServiceInitializationError(
                    "llm_retry_service",
                    "Model name is empty or invalid"
                )

            # Create the service with validated dependencies
            service = LLMRetryService(
                llm_retry_handler=llm_retry_handler,
                api_key_manager=api_key_manager,
                rate_limiter=rate_limiter,
                timeout=timeout,
                model_name=model_name
            )

            logger.info("LLM retry service created successfully")
            return service

        except Exception as e:
            logger.error(f"Failed to create LLM retry service: {e}")
            raise ServiceInitializationError(
                "llm_retry_service",
                f"Initialization failed: {str(e)}"
            ) from e

    @staticmethod
    def create_enhanced_llm_service_lazy(
        settings: Any,
        caching_service: Any,
        api_key_manager: LLMApiKeyManager,
        retry_service: LLMRetryService,
        rate_limiter: Any
    ) -> EnhancedLLMService:
        """Create an enhanced LLM service with lazy initialization and validation."""
        try:
            logger.info("Creating enhanced LLM service with lazy initialization")

            # Validate dependencies before creation
            if not settings:
                raise ServiceInitializationError(
                    "enhanced_llm_service",
                    "Settings dependency is None or invalid"
                )

            if not caching_service:
                raise ServiceInitializationError(
                    "enhanced_llm_service",
                    "Caching service dependency is None or invalid"
                )

            if not api_key_manager:
                raise ServiceInitializationError(
                    "enhanced_llm_service",
                    "API key manager dependency is None or invalid"
                )

            if not retry_service:
                raise ServiceInitializationError(
                    "enhanced_llm_service",
                    "Retry service dependency is None or invalid"
                )

            # Create the service with validated dependencies
            service = EnhancedLLMService(
                settings=settings,
                caching_service=caching_service,
                api_key_manager=api_key_manager,
                retry_service=retry_service,
                rate_limiter=rate_limiter
            )

            logger.info("Enhanced LLM service created successfully")
            return service

        except Exception as e:
            logger.error(f"Failed to create enhanced LLM service: {e}")
            raise ServiceInitializationError(
                "enhanced_llm_service",
                f"Initialization failed: {str(e)}"
            ) from e