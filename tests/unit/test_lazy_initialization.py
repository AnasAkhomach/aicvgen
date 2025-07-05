"""Tests for lazy initialization and dependency validation in ServiceFactory."""

import pytest
from unittest.mock import Mock, patch

from src.core.factories.service_factory import ServiceFactory
from src.error_handling.exceptions import ServiceInitializationError
from src.services.llm_api_key_manager import LLMApiKeyManager
from src.services.llm_retry_service import LLMRetryService
from src.services.llm_service import EnhancedLLMService
from src.services.llm_client import LLMClient
from src.services.llm_retry_handler import LLMRetryHandler


class TestLazyInitialization:
    """Test cases for lazy initialization methods."""

    def test_create_llm_api_key_manager_lazy_success(self):
        """Test successful creation of LLM API key manager with lazy initialization."""
        # Arrange
        mock_settings = Mock()
        mock_llm_client = Mock(spec=LLMClient)
        user_api_key = "test-api-key"
        
        with patch('src.core.factories.service_factory.LLMApiKeyManager') as mock_manager_class:
            mock_manager = Mock()
            mock_manager_class.return_value = mock_manager
            
            # Act
            result = ServiceFactory.create_llm_api_key_manager_lazy(
                settings=mock_settings,
                llm_client=mock_llm_client,
                user_api_key=user_api_key
            )
            
            # Assert
            assert result == mock_manager
            mock_manager_class.assert_called_once_with(
                settings=mock_settings,
                llm_client=mock_llm_client,
                user_api_key=user_api_key
            )

    def test_create_llm_api_key_manager_lazy_invalid_settings(self):
        """Test failure when settings dependency is invalid."""
        # Arrange
        mock_llm_client = Mock(spec=LLMClient)
        
        # Act & Assert
        with pytest.raises(ServiceInitializationError) as exc_info:
            ServiceFactory.create_llm_api_key_manager_lazy(
                settings=None,
                llm_client=mock_llm_client
            )
        
        assert "Settings dependency is None or invalid" in str(exc_info.value)
        assert exc_info.value.context.additional_data['service_name'] == "llm_api_key_manager"

    def test_create_llm_api_key_manager_lazy_invalid_client(self):
        """Test failure when LLM client dependency is invalid."""
        # Arrange
        mock_settings = Mock()
        
        # Act & Assert
        with pytest.raises(ServiceInitializationError) as exc_info:
            ServiceFactory.create_llm_api_key_manager_lazy(
                settings=mock_settings,
                llm_client=None
            )
        
        assert "LLM client dependency is None or invalid" in str(exc_info.value)
        assert exc_info.value.context.additional_data['service_name'] == "llm_api_key_manager"

    def test_create_llm_retry_service_lazy_success(self):
        """Test successful creation of LLM retry service with lazy initialization."""
        # Arrange
        mock_retry_handler = Mock(spec=LLMRetryHandler)
        mock_api_key_manager = Mock(spec=LLMApiKeyManager)
        mock_rate_limiter = Mock()
        timeout = 30
        model_name = "test-model"
        
        with patch('src.core.factories.service_factory.LLMRetryService') as mock_service_class:
            mock_service = Mock()
            mock_service_class.return_value = mock_service
            
            # Act
            result = ServiceFactory.create_llm_retry_service_lazy(
                llm_retry_handler=mock_retry_handler,
                api_key_manager=mock_api_key_manager,
                rate_limiter=mock_rate_limiter,
                timeout=timeout,
                model_name=model_name
            )
            
            # Assert
            assert result == mock_service
            mock_service_class.assert_called_once_with(
                llm_retry_handler=mock_retry_handler,
                api_key_manager=mock_api_key_manager,
                rate_limiter=mock_rate_limiter,
                timeout=timeout,
                model_name=model_name
            )

    def test_create_llm_retry_service_lazy_invalid_timeout(self):
        """Test failure when timeout value is invalid."""
        # Arrange
        mock_retry_handler = Mock(spec=LLMRetryHandler)
        mock_api_key_manager = Mock(spec=LLMApiKeyManager)
        mock_rate_limiter = Mock()
        
        # Act & Assert
        with pytest.raises(ServiceInitializationError) as exc_info:
            ServiceFactory.create_llm_retry_service_lazy(
                llm_retry_handler=mock_retry_handler,
                api_key_manager=mock_api_key_manager,
                rate_limiter=mock_rate_limiter,
                timeout=0,
                model_name="test-model"
            )
        
        assert "Invalid timeout value: 0" in str(exc_info.value)
        assert exc_info.value.context.additional_data['service_name'] == "llm_retry_service"

    def test_create_llm_retry_service_lazy_empty_model_name(self):
        """Test failure when model name is empty."""
        # Arrange
        mock_retry_handler = Mock(spec=LLMRetryHandler)
        mock_api_key_manager = Mock(spec=LLMApiKeyManager)
        mock_rate_limiter = Mock()
        
        # Act & Assert
        with pytest.raises(ServiceInitializationError) as exc_info:
            ServiceFactory.create_llm_retry_service_lazy(
                llm_retry_handler=mock_retry_handler,
                api_key_manager=mock_api_key_manager,
                rate_limiter=mock_rate_limiter,
                timeout=30,
                model_name=""
            )
        
        assert "Model name is empty or invalid" in str(exc_info.value)
        assert exc_info.value.context.additional_data['service_name'] == "llm_retry_service"

    def test_create_enhanced_llm_service_lazy_success(self):
        """Test successful creation of enhanced LLM service with lazy initialization."""
        # Arrange
        mock_settings = Mock()
        mock_caching_service = Mock()
        mock_api_key_manager = Mock(spec=LLMApiKeyManager)
        mock_retry_service = Mock(spec=LLMRetryService)
        mock_rate_limiter = Mock()
        
        with patch('src.core.factories.service_factory.EnhancedLLMService') as mock_service_class:
            mock_service = Mock()
            mock_service_class.return_value = mock_service
            
            # Act
            result = ServiceFactory.create_enhanced_llm_service_lazy(
                settings=mock_settings,
                caching_service=mock_caching_service,
                api_key_manager=mock_api_key_manager,
                retry_service=mock_retry_service,
                rate_limiter=mock_rate_limiter
            )
            
            # Assert
            assert result == mock_service
            mock_service_class.assert_called_once_with(
                settings=mock_settings,
                caching_service=mock_caching_service,
                api_key_manager=mock_api_key_manager,
                retry_service=mock_retry_service,
                rate_limiter=mock_rate_limiter
            )

    def test_create_enhanced_llm_service_lazy_invalid_dependencies(self):
        """Test failure when dependencies are invalid."""
        # Test invalid settings
        with pytest.raises(ServiceInitializationError) as exc_info:
            ServiceFactory.create_enhanced_llm_service_lazy(
                settings=None,
                caching_service=Mock(),
                api_key_manager=Mock(spec=LLMApiKeyManager),
                retry_service=Mock(spec=LLMRetryService),
                rate_limiter=Mock()
            )
        
        assert "Settings dependency is None or invalid" in str(exc_info.value)
        assert exc_info.value.context.additional_data['service_name'] == "enhanced_llm_service"
        
        # Test invalid caching service
        with pytest.raises(ServiceInitializationError) as exc_info:
            ServiceFactory.create_enhanced_llm_service_lazy(
                settings=Mock(),
                caching_service=None,
                api_key_manager=Mock(spec=LLMApiKeyManager),
                retry_service=Mock(spec=LLMRetryService),
                rate_limiter=Mock()
            )
        
        assert "Caching service dependency is None or invalid" in str(exc_info.value)
        assert exc_info.value.context.additional_data['service_name'] == "enhanced_llm_service"

    def test_lazy_initialization_exception_handling(self):
        """Test that exceptions during service creation are properly handled."""
        # Arrange
        mock_settings = Mock()
        mock_llm_client = Mock(spec=LLMClient)
        
        with patch('src.core.factories.service_factory.LLMApiKeyManager') as mock_manager_class:
            mock_manager_class.side_effect = Exception("Service creation failed")
            
            # Act & Assert
            with pytest.raises(ServiceInitializationError) as exc_info:
                ServiceFactory.create_llm_api_key_manager_lazy(
                    settings=mock_settings,
                    llm_client=mock_llm_client
                )
            
            assert "Initialization failed: Service creation failed" in str(exc_info.value)
            assert exc_info.value.context.additional_data['service_name'] == "llm_api_key_manager"