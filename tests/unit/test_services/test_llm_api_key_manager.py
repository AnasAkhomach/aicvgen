import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from src.services.llm_api_key_manager import LLMApiKeyManager
from src.models.llm_service_models import LLMApiKeyInfo
from src.error_handling.exceptions import ConfigurationError


class TestLLMApiKeyManager:
    """Test cases for LLMApiKeyManager."""

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings."""
        settings = MagicMock()
        settings.llm.gemini_api_key_primary = "primary_key"
        settings.llm.gemini_api_key_fallback = "fallback_key"
        return settings

    @pytest.fixture
    def mock_llm_client(self):
        """Create mock LLM client."""
        client = AsyncMock()
        client.list_models = AsyncMock(
            return_value=[{"name": "model1"}, {"name": "model2"}]
        )
        client.reconfigure = MagicMock()
        return client

    @pytest.fixture
    def api_key_manager(self, mock_settings, mock_llm_client):
        """Create an LLMApiKeyManager instance for testing."""
        return LLMApiKeyManager(settings=mock_settings, llm_client=mock_llm_client)

    def test_initialization_with_primary_key(self, mock_settings, mock_llm_client):
        """Test initialization with primary API key."""
        manager = LLMApiKeyManager(settings=mock_settings, llm_client=mock_llm_client)

        assert manager.active_api_key == "primary_key"
        assert not manager.using_fallback
        assert not manager.using_user_key
        assert manager.fallback_api_key == "fallback_key"

    def test_initialization_with_user_key(self, mock_settings, mock_llm_client):
        """Test initialization with user-provided API key."""
        manager = LLMApiKeyManager(
            settings=mock_settings, llm_client=mock_llm_client, user_api_key="user_key"
        )

        assert manager.active_api_key == "user_key"
        assert not manager.using_fallback
        assert manager.using_user_key
        assert manager.user_api_key == "user_key"

    def test_initialization_no_keys_raises_error(self, mock_llm_client):
        """Test initialization fails when no API keys are configured."""
        settings = MagicMock()
        settings.llm.gemini_api_key_primary = None
        settings.llm.gemini_api_key_fallback = None

        with pytest.raises(
            ConfigurationError, match="CRITICAL: Gemini API key is not configured"
        ):
            LLMApiKeyManager(settings=settings, llm_client=mock_llm_client)

    def test_determine_active_api_key_priority(self, mock_settings, mock_llm_client):
        """Test API key priority: user > primary > fallback."""
        # User key has highest priority
        manager = LLMApiKeyManager(
            settings=mock_settings, llm_client=mock_llm_client, user_api_key="user_key"
        )
        assert manager.active_api_key == "user_key"

        # Primary key when no user key
        manager = LLMApiKeyManager(settings=mock_settings, llm_client=mock_llm_client)
        assert manager.active_api_key == "primary_key"

        # Fallback when no primary
        mock_settings.llm.gemini_api_key_primary = None
        manager = LLMApiKeyManager(settings=mock_settings, llm_client=mock_llm_client)
        assert manager.active_api_key == "fallback_key"

    @pytest.mark.asyncio
    async def test_validate_api_key_success(self, api_key_manager):
        """Test successful API key validation."""
        result = await api_key_manager.validate_api_key()
        assert result is True
        api_key_manager.llm_client.list_models.assert_called_once()

    @pytest.mark.asyncio
    async def test_validate_api_key_failure(self, api_key_manager):
        """Test API key validation failure."""
        from src.error_handling.exceptions import ConfigurationError

        api_key_manager.llm_client.list_models.side_effect = ConfigurationError(
            "Invalid key"
        )

        result = await api_key_manager.validate_api_key()
        assert result is False

    @pytest.mark.asyncio
    async def test_ensure_api_key_valid_success(self, api_key_manager):
        """Test ensure_api_key_valid with valid key."""
        # Should not raise exception
        await api_key_manager.ensure_api_key_valid()

    @pytest.mark.asyncio
    async def test_ensure_api_key_valid_failure(self, api_key_manager):
        """Test ensure_api_key_valid with invalid key."""
        from src.error_handling.exceptions import ConfigurationError

        api_key_manager.llm_client.list_models.side_effect = ConfigurationError(
            "Invalid key"
        )

        with pytest.raises(
            ConfigurationError, match="Gemini API key validation failed"
        ):
            await api_key_manager.ensure_api_key_valid()

    @pytest.mark.asyncio
    async def test_switch_to_fallback_key_success(self, api_key_manager):
        """Test successful switch to fallback key."""
        assert not api_key_manager.using_fallback

        result = await api_key_manager.switch_to_fallback_key()

        assert result is True
        assert api_key_manager.using_fallback is True
        assert api_key_manager.active_api_key == "fallback_key"
        api_key_manager.llm_client.reconfigure.assert_called_once_with(
            api_key="fallback_key"
        )

    @pytest.mark.asyncio
    async def test_switch_to_fallback_key_already_using(self, api_key_manager):
        """Test switch to fallback when already using fallback."""
        api_key_manager.using_fallback = True

        result = await api_key_manager.switch_to_fallback_key()

        assert result is False
        api_key_manager.llm_client.reconfigure.assert_not_called()

    @pytest.mark.asyncio
    async def test_switch_to_fallback_key_no_fallback_available(
        self, mock_settings, mock_llm_client
    ):
        """Test switch to fallback when no fallback key is available."""
        mock_settings.llm.gemini_api_key_fallback = None
        manager = LLMApiKeyManager(settings=mock_settings, llm_client=mock_llm_client)

        result = await manager.switch_to_fallback_key()

        assert result is False
        manager.llm_client.reconfigure.assert_not_called()

    @pytest.mark.asyncio
    async def test_switch_to_fallback_key_reconfigure_failure(self, api_key_manager):
        """Test switch to fallback when reconfigure fails."""
        api_key_manager.llm_client.reconfigure.side_effect = ValueError(
            "Reconfigure failed"
        )

        result = await api_key_manager.switch_to_fallback_key()

        assert result is False
        assert not api_key_manager.using_fallback

    def test_get_current_api_key_info_primary(self, api_key_manager):
        """Test get_current_api_key_info with primary key."""
        info = api_key_manager.get_current_api_key_info()

        assert isinstance(info, LLMApiKeyInfo)
        assert not info.using_user_key
        assert not info.using_fallback
        assert info.has_fallback_available is True
        assert info.key_source == "primary"

    def test_get_current_api_key_info_user_key(self, mock_settings, mock_llm_client):
        """Test get_current_api_key_info with user key."""
        manager = LLMApiKeyManager(
            settings=mock_settings, llm_client=mock_llm_client, user_api_key="user_key"
        )

        info = manager.get_current_api_key_info()

        assert info.using_user_key is True
        assert not info.using_fallback
        assert info.key_source == "user"

    def test_get_current_api_key_info_fallback(self, api_key_manager):
        """Test get_current_api_key_info with fallback key."""
        api_key_manager.using_fallback = True
        api_key_manager.active_api_key = "fallback_key"

        info = api_key_manager.get_current_api_key_info()

        assert not info.using_user_key
        assert info.using_fallback is True
        assert info.key_source == "fallback"

    def test_get_active_api_key(self, api_key_manager):
        """Test get_active_api_key method."""
        assert api_key_manager.get_active_api_key() == "primary_key"

    def test_is_using_fallback(self, api_key_manager):
        """Test is_using_fallback method."""
        assert api_key_manager.is_using_fallback() is False

        api_key_manager.using_fallback = True
        assert api_key_manager.is_using_fallback() is True

    def test_is_using_user_key(self, api_key_manager):
        """Test is_using_user_key method."""
        assert api_key_manager.is_using_user_key() is False

        api_key_manager.using_user_key = True
        assert api_key_manager.is_using_user_key() is True

    def test_has_fallback_available(self, api_key_manager):
        """Test has_fallback_available method."""
        assert api_key_manager.has_fallback_available() is True

        api_key_manager.using_fallback = True
        assert api_key_manager.has_fallback_available() is False

        api_key_manager.fallback_api_key = None
        assert api_key_manager.has_fallback_available() is False
