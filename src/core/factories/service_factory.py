"""Factory module for creating service instances."""

from typing import Any, Optional

from src.models.vector_store_config_interface import VectorStoreConfigInterface

from src.config.logging_config import get_structured_logger
from src.error_handling.exceptions import ServiceInitializationError
from src.services.llm_api_key_manager import LLMApiKeyManager

from src.services.llm.llm_client_interface import LLMClientInterface
from src.services.llm.gemini_client import GeminiClient

from src.services.llm_service import EnhancedLLMService
from src.services.progress_tracker import ProgressTracker
from src.services.vector_store_service import VectorStoreService

logger = get_structured_logger("service_factory")


class ServiceFactory:
    """Factory for creating service instances with dependency injection."""

    @staticmethod
    def create_llm_client(api_key: str, model_name: str) -> LLMClientInterface:
        """Create an LLM client instance."""
        return GeminiClient(api_key=api_key, model_name=model_name)

    @staticmethod
    def create_llm_api_key_manager(
        settings: Any,
        llm_client: LLMClientInterface,
        user_api_key: Optional[str] = None,
    ) -> LLMApiKeyManager:
        """Create an LLM API key manager instance."""
        return LLMApiKeyManager(
            settings=settings, llm_client=llm_client, user_api_key=user_api_key
        )

    @staticmethod
    def create_enhanced_llm_service(
        settings: Any,
        llm_client: LLMClientInterface,
        api_key_manager: LLMApiKeyManager,
    ) -> EnhancedLLMService:
        """Create an enhanced LLM service instance."""
        return EnhancedLLMService(
            settings=settings,
            llm_client=llm_client,
            api_key_manager=api_key_manager,
        )

    @staticmethod
    def create_vector_store_service(
        vector_config: VectorStoreConfigInterface,
    ) -> VectorStoreService:
        """Create a vector store service instance."""
        return VectorStoreService(vector_config=vector_config)

    @staticmethod
    def create_progress_tracker() -> ProgressTracker:
        """Create a progress tracker instance."""
        return ProgressTracker(logger=logger)

    # Lazy initialization methods for interdependent services
    @staticmethod
    def create_llm_api_key_manager_lazy(
        settings: Any,
        llm_client: LLMClientInterface,
        user_api_key: Optional[str] = None,
    ) -> LLMApiKeyManager:
        """Create an LLM API key manager with lazy initialization and validation."""
        try:
            logger.info("Creating LLM API key manager with lazy initialization")

            # Validate dependencies before creation
            if not settings:
                raise ServiceInitializationError(
                    "llm_api_key_manager", "Settings dependency is None or invalid"
                )

            if not llm_client:
                raise ServiceInitializationError(
                    "llm_api_key_manager", "LLM client dependency is None or invalid"
                )

            # Create the service with validated dependencies
            manager = LLMApiKeyManager(
                settings=settings, llm_client=llm_client, user_api_key=user_api_key
            )

            logger.info("LLM API key manager created successfully")
            return manager

        except Exception as e:
            logger.error(f"Failed to create LLM API key manager: {e}")
            raise ServiceInitializationError(
                "llm_api_key_manager", f"Initialization failed: {str(e)}"
            ) from e

    @staticmethod
    def create_enhanced_llm_service_lazy(
        settings: Any,
        llm_client: LLMClientInterface,
        api_key_manager: LLMApiKeyManager,
    ) -> EnhancedLLMService:
        """Create an enhanced LLM service with lazy initialization and validation."""
        try:
            logger.info("Creating enhanced LLM service with lazy initialization")

            # Validate dependencies before creation
            if not settings:
                raise ServiceInitializationError(
                    "enhanced_llm_service", "Settings dependency is None or invalid"
                )

            if not llm_client:
                raise ServiceInitializationError(
                    "enhanced_llm_service", "LLM client dependency is None or invalid"
                )

            if not api_key_manager:
                raise ServiceInitializationError(
                    "enhanced_llm_service",
                    "API key manager dependency is None or invalid",
                )

            # Create the service with validated dependencies
            service = EnhancedLLMService(
                settings=settings,
                llm_client=llm_client,
                api_key_manager=api_key_manager,
            )

            logger.info("Enhanced LLM service created successfully")
            return service

        except Exception as e:
            logger.error(f"Failed to create enhanced LLM service: {e}")
            raise ServiceInitializationError(
                "enhanced_llm_service", f"Initialization failed: {str(e)}"
            ) from e
