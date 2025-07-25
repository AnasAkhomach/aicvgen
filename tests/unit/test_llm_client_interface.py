"""Tests for LLM client interface and implementations."""

import pytest
from unittest.mock import Mock, patch, AsyncMock

from src.services.llm.llm_client_interface import LLMClientInterface
from src.services.llm.gemini_client import GeminiClient


class TestLLMClientInterface:
    """Test cases for LLMClientInterface abstract class."""

    def test_interface_cannot_be_instantiated(self):
        """Test that LLMClientInterface cannot be instantiated directly."""
        with pytest.raises(TypeError):
            LLMClientInterface()


class TestGeminiClient:
    """Test cases for GeminiClient implementation."""

    def test_initialization_success(self):
        """Test successful initialization of GeminiClient."""
        # Arrange
        api_key = "test-api-key"
        model_name = "gemini-pro"
        
        # Act
        client = GeminiClient(api_key=api_key, model_name=model_name)
        
        # Assert
        assert client.get_model_name() == model_name
        assert client.is_initialized() is True

    def test_initialization_empty_api_key(self):
        """Test initialization failure with empty API key."""
        with pytest.raises(ValueError, match="API key cannot be empty"):
            GeminiClient(api_key="", model_name="gemini-pro")

    def test_initialization_empty_model_name(self):
        """Test initialization failure with empty model name."""
        with pytest.raises(ValueError, match="Model name cannot be empty"):
            GeminiClient(api_key="test-key", model_name="")

    @pytest.mark.asyncio
    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    async def test_generate_content_success(self, mock_model_class, mock_configure):
        """Test successful content generation."""
        # Arrange
        api_key = "test-api-key"
        model_name = "gemini-pro"
        prompt = "Test prompt"
        
        mock_model = Mock()
        mock_model.generate_content_async = AsyncMock(return_value="Generated content")
        mock_model_class.return_value = mock_model
        
        client = GeminiClient(api_key=api_key, model_name=model_name)
        
        # Act
        result = await client.generate_content(prompt)
        
        # Assert
        assert result == "Generated content"
        mock_configure.assert_called_with(api_key=api_key)
        mock_model_class.assert_called_with(model_name)
        mock_model.generate_content_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_content_empty_prompt(self):
        """Test generate_content with empty prompt."""
        # Arrange
        client = GeminiClient(api_key="test-key", model_name="gemini-pro")
        
        # Act & Assert
        with pytest.raises(ValueError, match="Prompt cannot be empty"):
            await client.generate_content("")

    @pytest.mark.asyncio
    @patch('google.generativeai.configure')
    @patch('google.generativeai.list_models')
    async def test_list_models_success(self, mock_list_models, mock_configure):
        """Test successful model listing."""
        # Arrange
        api_key = "test-api-key"
        model_name = "gemini-pro"
        mock_models = [Mock(name="model1"), Mock(name="model2")]
        mock_list_models.return_value = iter(mock_models)
        
        client = GeminiClient(api_key=api_key, model_name=model_name)
        
        # Act
        result = await client.list_models()
        
        # Assert
        assert len(result) == 2
        assert result == mock_models
        mock_configure.assert_called_with(api_key=api_key)

    def test_reconfigure_success(self):
        """Test successful reconfiguration with new API key."""
        # Arrange
        client = GeminiClient(api_key="old-key", model_name="gemini-pro")
        new_api_key = "new-api-key"
        
        with patch('google.generativeai.configure') as mock_configure:
            # Act
            client.reconfigure(new_api_key)
            
            # Assert
            mock_configure.assert_called_with(api_key=new_api_key)
            assert client._api_key == new_api_key

    def test_reconfigure_empty_api_key(self):
        """Test reconfigure with empty API key."""
        # Arrange
        client = GeminiClient(api_key="test-key", model_name="gemini-pro")
        
        # Act & Assert
        with pytest.raises(ValueError, match="API key cannot be empty"):
            client.reconfigure("")

    def test_implements_interface(self):
        """Test that GeminiClient properly implements LLMClientInterface."""
        # Arrange
        client = GeminiClient(api_key="test-key", model_name="gemini-pro")
        
        # Assert
        assert isinstance(client, LLMClientInterface)
        
        # Verify all interface methods are implemented
        assert hasattr(client, 'generate_content')
        assert hasattr(client, 'get_model_name')
        assert hasattr(client, 'is_initialized')
        assert hasattr(client, 'list_models')
        assert hasattr(client, 'reconfigure')