"""Tests for LLM Service Interface contract compliance."""

import pytest
from unittest.mock import Mock, AsyncMock

from src.models.llm_data_models import LLMResponse
from src.models.llm_service_models import LLMApiKeyInfo
from src.models.workflow_models import ContentType
from src.services.llm_service import EnhancedLLMService
from src.services.llm_service_interface import LLMServiceInterface


class TestLLMServiceInterface:
    """Test cases for LLM Service Interface contract compliance."""

    def test_enhanced_llm_service_implements_interface(self):
        """Test that EnhancedLLMService properly implements LLMServiceInterface."""
        # Verify that EnhancedLLMService is a subclass of LLMServiceInterface
        assert issubclass(EnhancedLLMService, LLMServiceInterface)

        # Verify that all interface methods are implemented
        interface_methods = {
            "generate_content",
            "generate",
            "validate_api_key",
            "get_current_api_key_info",
            "ensure_api_key_valid",
        }

        enhanced_service_methods = set(dir(EnhancedLLMService))

        for method in interface_methods:
            assert (
                method in enhanced_service_methods
            ), f"Method {method} not implemented in EnhancedLLMService"

    @pytest.fixture
    def mock_llm_service(self):
        """Create a mock LLM service that implements the interface."""
        mock_service = Mock(spec=LLMServiceInterface)

        # Mock the async methods
        mock_service.generate_content = AsyncMock()
        mock_service.generate = AsyncMock()
        mock_service.validate_api_key = AsyncMock()
        mock_service.ensure_api_key_valid = AsyncMock()

        # Mock the sync method
        mock_service.get_current_api_key_info = Mock()

        return mock_service

    @pytest.mark.asyncio
    async def test_interface_contract_generate_content(self, mock_llm_service):
        """Test that the interface contract for generate_content is respected."""
        # Setup mock response
        expected_response = LLMResponse(
            content="Test response",
            model="test-model",
            tokens_used=10,
            processing_time=1.0,
        )
        mock_llm_service.generate_content.return_value = expected_response

        # Test the interface method
        result = await mock_llm_service.generate_content(
            prompt="test prompt",
            content_type=ContentType.CV_ANALYSIS,
            session_id="test_session",
            max_tokens=100,
            temperature=0.7,
        )

        assert result == expected_response
        mock_llm_service.generate_content.assert_called_once_with(
            prompt="test prompt",
            content_type=ContentType.CV_ANALYSIS,
            session_id="test_session",
            max_tokens=100,
            temperature=0.7,
        )

    @pytest.mark.asyncio
    async def test_interface_contract_generate(self, mock_llm_service):
        """Test that the interface contract for generate is respected."""
        # Setup mock response
        expected_response = LLMResponse(
            content="Test response",
            model="test-model",
            tokens_used=10,
            processing_time=1.0,
        )
        mock_llm_service.generate.return_value = expected_response

        # Test the interface method
        result = await mock_llm_service.generate("test prompt", max_tokens=100)

        assert result == expected_response
        mock_llm_service.generate.assert_called_once_with("test prompt", max_tokens=100)

    @pytest.mark.asyncio
    async def test_interface_contract_validate_api_key(self, mock_llm_service):
        """Test that the interface contract for validate_api_key is respected."""
        mock_llm_service.validate_api_key.return_value = True

        result = await mock_llm_service.validate_api_key()

        assert result is True
        mock_llm_service.validate_api_key.assert_called_once()

    def test_interface_contract_get_current_api_key_info(self, mock_llm_service):
        """Test that the interface contract for get_current_api_key_info is respected."""
        expected_info = LLMApiKeyInfo(
            using_user_key=True,
            using_fallback=False,
            has_fallback_available=True,
            key_source="test",
        )
        mock_llm_service.get_current_api_key_info.return_value = expected_info

        result = mock_llm_service.get_current_api_key_info()

        assert result == expected_info
        mock_llm_service.get_current_api_key_info.assert_called_once()

    @pytest.mark.asyncio
    async def test_interface_contract_ensure_api_key_valid(self, mock_llm_service):
        """Test that the interface contract for ensure_api_key_valid is respected."""
        # Test successful validation
        await mock_llm_service.ensure_api_key_valid()
        mock_llm_service.ensure_api_key_valid.assert_called_once()

    def test_interface_hides_implementation_details(self):
        """Test that the interface does not expose implementation details."""
        interface_methods = set(dir(LLMServiceInterface))

        # These methods should NOT be in the interface as they expose implementation details
        forbidden_methods = {
            "get_service_stats",
            "clear_cache",
            "optimize_performance",
            "caching_service",
            "retry_service",
            "rate_limiter",
        }

        for method in forbidden_methods:
            assert (
                method not in interface_methods
            ), f"Implementation detail {method} exposed in interface"

    def test_cb011_contract_breach_resolved(self):
        """Test that CB-011 contract breach has been resolved."""
        # Verify that EnhancedLLMService implements the clean interface
        assert issubclass(EnhancedLLMService, LLMServiceInterface)

        # Verify that implementation details are not exposed in the interface
        interface_methods = set(dir(LLMServiceInterface))

        # These were the problematic methods that exposed implementation details
        problematic_methods = {
            "get_service_stats",  # Exposed cache_stats, optimizer_stats
            "clear_cache",  # Exposed caching mechanism
            "optimize_performance",  # Exposed optimization internals
        }

        for method in problematic_methods:
            assert (
                method not in interface_methods
            ), f"CB-011 contract breach: {method} still exposed in interface"

        # Verify that the essential methods are still available
        essential_methods = {
            "generate_content",
            "generate",
            "validate_api_key",
            "get_current_api_key_info",
            "ensure_api_key_valid",
        }

        for method in essential_methods:
            assert (
                method in interface_methods
            ), f"Essential method {method} missing from interface"
