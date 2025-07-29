"""Integration tests for LLM client interface with DI container."""

from unittest.mock import Mock, patch

import pytest

from src.core.container import ContainerSingleton, get_container
from src.services.llm.gemini_client import GeminiClient
from src.services.llm.llm_client_interface import LLMClientInterface


class TestLLMClientInterfaceIntegration:
    """Integration tests for LLM client interface with dependency injection."""

    @pytest.fixture
    def container(self):
        """Get a fresh container instance for testing."""
        ContainerSingleton.reset_instance()
        return get_container()

    def test_container_provides_llm_client_interface(self, container):
        """Test that container provides LLMClientInterface instance."""
        # Act & Assert
        with patch(
            "src.services.llm.gemini_client.GeminiClient.__init__", return_value=None
        ):
            llm_client = container.llm_client()
            assert llm_client is not None
            assert isinstance(llm_client, GeminiClient)

    def test_llm_api_key_manager_uses_interface(self, container):
        """Test that LLMApiKeyManager works with LLMClientInterface."""
        # Act & Assert
        with patch(
            "src.services.llm.gemini_client.GeminiClient.__init__", return_value=None
        ), patch(
            "src.services.llm_api_key_manager.LLMApiKeyManager.__init__",
            return_value=None,
        ) as mock_init:
            api_key_manager = container.llm_api_key_manager()
            assert api_key_manager is not None

            # Verify that LLMApiKeyManager was initialized with LLMClientInterface
            mock_init.assert_called_once()
            call_kwargs = mock_init.call_args.kwargs
            assert "llm_client" in call_kwargs
            assert isinstance(call_kwargs["llm_client"], GeminiClient)

    def test_llm_retry_handler_uses_interface(self, container):
        """Test that LLMRetryHandler works with LLMClientInterface."""
        # Act & Assert
        with patch(
            "src.services.llm.gemini_client.GeminiClient.__init__", return_value=None
        ), patch(
            "src.services.llm_retry_handler.LLMRetryHandler.__init__", return_value=None
        ) as mock_init:
            retry_handler = container.llm_retry_handler()
            assert retry_handler is not None

            # Verify that LLMRetryHandler was initialized with LLMClientInterface
            mock_init.assert_called_once()
            call_kwargs = mock_init.call_args.kwargs
            assert "llm_client" in call_kwargs
            assert isinstance(call_kwargs["llm_client"], GeminiClient)

    def test_container_singleton_behavior(self, container):
        """Test that container provides same LLMClientInterface instance."""
        # Act & Assert
        with patch(
            "src.services.llm.gemini_client.GeminiClient.__init__", return_value=None
        ):
            # Get LLM client multiple times
            client1 = container.llm_client()
            client2 = container.llm_client()

            # Should be the same instance (singleton behavior)
            assert client1 is client2
            assert isinstance(client1, GeminiClient)
            assert isinstance(client2, GeminiClient)

    def test_gemini_client_configuration(self, container):
        """Test that GeminiClient is properly configured through container."""
        # Act & Assert
        with patch(
            "src.services.llm.gemini_client.GeminiClient.__init__", return_value=None
        ):
            llm_client = container.llm_client()

            # Verify that we get a GeminiClient instance
            assert llm_client is not None
            assert isinstance(llm_client, GeminiClient)

            # Verify that the client has the expected interface methods
            assert hasattr(llm_client, "generate_content")
            assert hasattr(llm_client, "list_models")
            assert hasattr(llm_client, "reconfigure")

    def test_interface_methods_available(self, container):
        """Test that all interface methods are available through container-provided instance."""
        # Act & Assert
        with patch(
            "src.services.llm.gemini_client.GeminiClient.__init__", return_value=None
        ):
            llm_client = container.llm_client()

            # Test that interface methods are available
            assert hasattr(llm_client, "generate_content")
            assert hasattr(llm_client, "list_models")
            assert hasattr(llm_client, "reconfigure")

            # Test that methods are callable
            assert callable(getattr(llm_client, "generate_content"))
            assert callable(getattr(llm_client, "list_models"))
            assert callable(getattr(llm_client, "reconfigure"))
