import unittest
from unittest.mock import Mock, patch
import pytest

from src.core.dependency_injection import (
    DependencyContainer,
    get_container,
    reset_container,
    build_llm_service,
    register_core_services,
    EnhancedLLMService,
    LifecycleScope,
    DependencyMetadata,
)
from src.config.settings import Settings


class TestDependencyInjection(unittest.TestCase):

    def setUp(self):
        # Reset the container before each test to ensure isolation
        reset_container()

    def tearDown(self):
        # Clean up after each test
        reset_container()

    def test_singleton_llm_service_resolution(self):
        """
        Verify that resolving EnhancedLLMService twice returns the same instance.
        """
        # Arrange
        container = get_container()

        # Mock dependencies
        mock_settings = Mock(spec=Settings)
        mock_settings.llm_settings.default_model = "gemini-pro"
        mock_settings.llm_settings.default_timeout = 60
        mock_settings.cache_settings.persist_file = None

        container.register_singleton(
            name="settings", dependency_type=Settings, factory=lambda: mock_settings
        )
        register_core_services(container)

        # Register the EnhancedLLMService as a singleton
        container.register_singleton(
            "EnhancedLLMService",
            EnhancedLLMService,
            factory=lambda: build_llm_service(container),
        )

        # Act
        llm_service_1 = container.get(EnhancedLLMService, "EnhancedLLMService")
        llm_service_2 = container.get(EnhancedLLMService, "EnhancedLLMService")

        # Assert
        self.assertIs(llm_service_1, llm_service_2)
        self.assertIsInstance(llm_service_1, EnhancedLLMService)

    def test_build_llm_service_with_user_api_key(self):
        """
        Test that the build_llm_service factory correctly passes the user_api_key.
        """
        # Arrange
        container = get_container()
        user_api_key = "test_user_key"

        # Mock dependencies
        mock_settings = Mock(spec=Settings)
        mock_settings.llm_settings.default_model = "gemini-pro"
        mock_settings.llm_settings.default_timeout = 60
        mock_settings.cache_settings.persist_file = None
        # Mock the fallback key to ensure the user key is prioritized
        mock_settings.llm.gemini_api_key_fallback = "fallback_key"

        container.register_singleton(
            name="settings", dependency_type=Settings, factory=lambda: mock_settings
        )
        register_core_services(container)

        # Act
        with patch("google.generativeai.GenerativeModel") as mock_genai_model:
            llm_service = build_llm_service(container, user_api_key=user_api_key)

            # Assert
            self.assertEqual(llm_service.user_api_key, user_api_key)
            self.assertTrue(llm_service.using_user_key)
            self.assertEqual(llm_service.active_api_key, user_api_key)


if __name__ == "__main__":
    unittest.main()
