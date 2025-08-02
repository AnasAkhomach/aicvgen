import asyncio
import logging
from typing import Optional

from src.utils.import_fallbacks import get_google_exceptions
from src.config.logging_config import get_structured_logger
from src.error_handling.exceptions import ConfigurationError
from src.models.llm_service_models import LLMApiKeyInfo
from src.services.llm.llm_client_interface import LLMClientInterface

# Import Google API exceptions with standardized fallback handling
google_exceptions, _ = get_google_exceptions()

logger = get_structured_logger("llm_api_key_manager")


class LLMApiKeyManager:
    """
    Manages API key validation, switching, and fallback logic.
    Handles all API key-related operations for LLM services.
    """

    def __init__(
        self,
        settings,
        llm_client: LLMClientInterface,
        user_api_key: Optional[str] = None,
    ):
        """
        Initialize the API key manager.

        Args:
            settings: Injected settings/config dependency
            llm_client: Injected LLMClientInterface instance
            user_api_key: Optional user-provided API key (takes priority)
        """
        self.settings = settings
        self.llm_client = llm_client
        self.user_api_key = user_api_key

        # API key management state
        self.active_api_key = self._determine_active_api_key(user_api_key)
        self.fallback_api_key = self.settings.llm.gemini_api_key_fallback
        self.using_fallback = False
        self.using_user_key = bool(user_api_key)

        logger.info(
            "LLM API key manager initialized",
            using_user_key=self.using_user_key,
            using_fallback_key=self.using_fallback,
            has_fallback_key=bool(self.fallback_api_key),
        )

    def _determine_active_api_key(self, user_api_key: Optional[str]) -> str:
        """
        Determine the active API key based on priority: user > primary > fallback.

        Args:
            user_api_key: Optional user-provided API key

        Returns:
            str: The API key to use

        Raises:
            ConfigurationError: If no valid API key is found
        """
        # Priority order: user-provided > primary > fallback
        if user_api_key:
            return user_api_key
        if self.settings.llm.gemini_api_key_primary:
            return self.settings.llm.gemini_api_key_primary
        if self.settings.llm.gemini_api_key_fallback:
            return self.settings.llm.gemini_api_key_fallback
        raise ConfigurationError(
            "CRITICAL: Gemini API key is not configured. "
            "Please set the GEMINI_API_KEY in your .env file or provide it in the UI. "
            "Application cannot start without a valid API key."
        )

    async def validate_api_key(self) -> bool:
        """
        Validate the current API key by making a lightweight API call.

        This method performs a simple, low-cost API call (listing models)
        to verify that the API key is valid and the service is accessible.

        Returns:
            bool: True if the API key is valid, False otherwise.
        """
        try:
            # Make a lightweight API call to validate the key
            models = await self.llm_client.list_models()

            # If we get here without exception, the key is valid
            logger.info(
                "API key validation successful",
                models_count=len(models),
                using_user_key=self.using_user_key,
                using_fallback=self.using_fallback,
            )
            return True
        except (
            google_exceptions.GoogleAPICallError if google_exceptions else (),
            ConfigurationError,
        ) as e:
            # Log the validation failure with full details
            logger.warning(
                "API key validation failed",
                error_type=type(e).__name__,
                error_message=str(e),
                using_user_key=self.using_user_key,
                using_fallback=self.using_fallback,
            )
            logging.error("Gemini API key validation exception: %s", e, exc_info=True)
            return False

    async def ensure_api_key_valid(self):
        """
        Explicitly validate the API key. Raises ConfigurationError if invalid.
        Call this after construction in async context.
        """
        if not await self.validate_api_key():
            raise ConfigurationError(
                "Gemini API key validation failed. Please check your GEMINI_API_KEY or GEMINI_API_KEY_FALLBACK. "
                "Application cannot start without a valid key."
            )

    async def switch_to_fallback_key(self) -> bool:
        """
        Switch to fallback API key when rate limits or errors are encountered.

        Returns:
            bool: True if successfully switched to fallback, False otherwise
        """
        if not self.using_fallback and self.fallback_api_key:
            logger.warning(
                "Switching to fallback API key due to rate limit or error",
                current_key_type="primary" if not self.using_user_key else "user",
                fallback_available=True,
            )

            try:
                # Reconfigure the client with the fallback key
                await asyncio.to_thread(
                    self.llm_client.reconfigure, api_key=self.fallback_api_key
                )
                self.active_api_key = self.fallback_api_key
                self.using_fallback = True

            except (ValueError, ConnectionError) as e:
                logger.error(
                    "Failed to switch to fallback API key",
                    error=str(e),
                    error_type=type(e).__name__,
                )
                return False

            logger.info("Successfully switched to fallback API key")
            return True
        else:
            logger.error(
                "Cannot switch to fallback key",
                already_using_fallback=self.using_fallback,
                fallback_available=bool(self.fallback_api_key),
            )
            return False

    def get_current_api_key_info(self) -> LLMApiKeyInfo:
        """
        Get information about the currently active API key.

        Returns:
            LLMApiKeyInfo containing API key status information
        """
        return LLMApiKeyInfo(
            using_user_key=self.using_user_key,
            using_fallback=self.using_fallback,
            has_fallback_available=bool(self.fallback_api_key),
            key_source=(
                "user"
                if self.using_user_key
                else ("fallback" if self.using_fallback else "primary")
            ),
        )

    def get_active_api_key(self) -> str:
        """Get the currently active API key."""
        return self.active_api_key

    def is_using_fallback(self) -> bool:
        """Check if currently using fallback API key."""
        return self.using_fallback

    def is_using_user_key(self) -> bool:
        """Check if currently using user-provided API key."""
        return self.using_user_key

    def has_fallback_available(self) -> bool:
        """Check if fallback API key is available."""
        return bool(self.fallback_api_key) and not self.using_fallback
