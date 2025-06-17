"""Unit tests for the shared utility method _generate_and_parse_json in agent_base.py."""

import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch

from src.agents.agent_base import EnhancedAgentBase, AgentExecutionContext, AgentResult
from src.core.state_manager import AgentIO
from src.models.data_models import ContentType


class TestAgentBase(EnhancedAgentBase):
    """Test implementation of EnhancedAgentBase for testing purposes."""

    def __init__(self):
        super().__init__(
            name="test_agent",
            description="Test agent for unit testing",
            input_schema=AgentIO(data_type="dict", description="Test input"),
            output_schema=AgentIO(data_type="dict", description="Test output"),
            content_type=ContentType.EXPERIENCE,
        )

    async def run_async(
        self, input_data, context: AgentExecutionContext
    ) -> AgentResult:
        """Test implementation of run_async."""
        return AgentResult(success=True, output_data={"test": "data"})

    async def run_as_node(self, state) -> dict:
        """Test implementation of run_as_node."""
        return {"test_field": "test_value"}


class TestGenerateAndParseJsonUtility:
    """Test the _generate_and_parse_json shared utility method."""

    @pytest.fixture
    def test_agent(self):
        """Create a test agent instance."""
        return TestAgentBase()

    @pytest.mark.asyncio
    async def test_successful_json_generation_and_parsing(self, test_agent):
        """Test successful LLM generation and JSON parsing."""
        mock_response = '{"key": "value", "number": 42}'

        with patch("src.services.llm_service.get_llm_service") as mock_get_service:
            mock_service = AsyncMock()
            mock_service.generate_async.return_value = mock_response
            mock_get_service.return_value = mock_service

            result = await test_agent._generate_and_parse_json(
                prompt="Test prompt", model_name="test-model", temperature=0.5
            )

            assert result == {"key": "value", "number": 42}
            mock_service.generate_async.assert_called_once_with(
                prompt="Test prompt", model="test-model", temperature=0.5
            )

    @pytest.mark.asyncio
    async def test_json_extraction_from_markdown(self, test_agent):
        """Test JSON extraction from markdown code blocks."""
        mock_response = '```json\n{"extracted": true}\n```'

        with patch("src.services.llm_service.get_llm_service") as mock_get_service:
            mock_service = AsyncMock()
            mock_service.generate_async.return_value = mock_response
            mock_get_service.return_value = mock_service

            result = await test_agent._generate_and_parse_json("Test prompt")

            assert result == {"extracted": True}

    @pytest.mark.asyncio
    async def test_json_extraction_with_surrounding_text(self, test_agent):
        """Test JSON extraction when surrounded by other text."""
        mock_response = (
            'Here is the JSON response: {"data": "extracted"} and some more text.'
        )

        with patch("src.services.llm_service.get_llm_service") as mock_get_service:
            mock_service = AsyncMock()
            mock_service.generate_async.return_value = mock_response
            mock_get_service.return_value = mock_service

            result = await test_agent._generate_and_parse_json("Test prompt")

            assert result == {"data": "extracted"}

    @pytest.mark.asyncio
    async def test_json_parsing_error_handling(self, test_agent):
        """Test handling of JSON parsing errors."""
        mock_response = "Invalid JSON: {key: value}"

        with patch("src.services.llm_service.get_llm_service") as mock_get_service:
            mock_service = AsyncMock()
            mock_service.generate_async.return_value = mock_response
            mock_get_service.return_value = mock_service

            with pytest.raises(Exception) as exc_info:
                await test_agent._generate_and_parse_json("Test prompt")

            assert "Failed to parse JSON response" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_llm_service_error_handling(self, test_agent):
        """Test handling of LLM service errors."""
        with patch("src.services.llm_service.get_llm_service") as mock_get_service:
            mock_service = AsyncMock()
            mock_service.generate_async.side_effect = Exception("LLM service error")
            mock_get_service.return_value = mock_service

            with pytest.raises(Exception) as exc_info:
                await test_agent._generate_and_parse_json("Test prompt")

            assert "LLM service error" in str(exc_info.value)

    def test_extract_json_from_response_markdown_removal(self, test_agent):
        """Test _extract_json_from_response removes markdown correctly."""
        response = '```json\n{"test": "data"}\n```'
        result = test_agent._extract_json_from_response(response)
        assert result == '{"test": "data"}'

    def test_extract_json_from_response_boundary_detection(self, test_agent):
        """Test _extract_json_from_response detects JSON boundaries."""
        response = 'Some text before {"key": "value"} some text after'
        result = test_agent._extract_json_from_response(response)
        assert result == '{"key": "value"}'

    def test_extract_json_from_response_no_boundaries(self, test_agent):
        """Test _extract_json_from_response when no clear boundaries exist."""
        response = "No JSON here at all"
        result = test_agent._extract_json_from_response(response)
        assert result == "No JSON here at all"

    def test_extract_json_from_response_whitespace_handling(self, test_agent):
        """Test _extract_json_from_response handles whitespace correctly."""
        response = '   \n  {"clean": "data"}  \n  '
        result = test_agent._extract_json_from_response(response)
        assert result == '{"clean": "data"}'

    @pytest.mark.asyncio
    async def test_default_parameters(self, test_agent):
        """Test that default parameters are used correctly."""
        mock_response = '{"default": "test"}'

        with patch("src.services.llm_service.get_llm_service") as mock_get_service:
            mock_service = AsyncMock()
            mock_service.generate_async.return_value = mock_response
            mock_get_service.return_value = mock_service

            result = await test_agent._generate_and_parse_json("Test prompt")

            # Verify default parameters were used
            mock_service.generate_async.assert_called_once_with(
                prompt="Test prompt",
                model="gemini-2.0-flash",  # Default model
                temperature=0.7,  # Default temperature
            )
            assert result == {"default": "test"}

    @pytest.mark.asyncio
    async def test_custom_parameters(self, test_agent):
        """Test that custom parameters are passed correctly."""
        mock_response = '{"custom": "test"}'

        with patch("src.services.llm_service.get_llm_service") as mock_get_service:
            mock_service = AsyncMock()
            mock_service.generate_async.return_value = mock_response
            mock_get_service.return_value = mock_service

            result = await test_agent._generate_and_parse_json(
                prompt="Custom prompt", model_name="custom-model", temperature=0.9
            )

            # Verify custom parameters were used
            mock_service.generate_async.assert_called_once_with(
                prompt="Custom prompt", model="custom-model", temperature=0.9
            )
            assert result == {"custom": "test"}

    @pytest.mark.asyncio
    async def test_logging_on_json_parse_error(self, test_agent):
        """Test that JSON parsing errors are logged correctly."""
        mock_response = "Invalid JSON"

        with patch("src.services.llm_service.get_llm_service") as mock_get_service:
            mock_service = AsyncMock()
            mock_service.generate_async.return_value = mock_response
            mock_get_service.return_value = mock_service

            # Mock the logger
            with patch.object(test_agent.logger, "error") as mock_logger:
                with pytest.raises(Exception):
                    await test_agent._generate_and_parse_json("Test prompt")

                # Verify error was logged
                mock_logger.assert_called()
                # Check if the error message is in the first positional argument
                call_args = mock_logger.call_args[0]
                assert "JSON parsing failed for agent test_agent" in str(call_args)

    @pytest.mark.asyncio
    async def test_logging_on_llm_error(self, test_agent):
        """Test that LLM generation errors are logged correctly."""
        with patch("src.services.llm_service.get_llm_service") as mock_get_service:
            mock_service = AsyncMock()
            mock_service.generate_async.side_effect = Exception("LLM error")
            mock_get_service.return_value = mock_service

            # Mock the logger
            with patch.object(test_agent.logger, "error") as mock_logger:
                with pytest.raises(Exception):
                    await test_agent._generate_and_parse_json("Test prompt")

                # Verify error was logged
                mock_logger.assert_called()
                # Check if the error message is in the first positional argument
                call_args = mock_logger.call_args[0]
                assert "LLM generation failed for agent test_agent" in str(call_args)
