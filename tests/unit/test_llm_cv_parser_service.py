import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pytest
from unittest.mock import AsyncMock, MagicMock
import json

from src.services.llm_cv_parser_service import LLMCVParserService
from src.utils.exceptions import LLMResponseParsingError
from src.services.llm_service import LLMResponse

# Test data
VALID_JSON_STRING = '{"key": "value"}'
JSON_IN_MARKDOWN = f"```json\n{VALID_JSON_STRING}\n```"
MALFORMED_JSON_STRING = '{"key": "value"'
EMPTY_STRING = ""


@pytest.fixture
def mock_llm_service():
    """Fixture for a mocked EnhancedLLMService."""
    mock = AsyncMock()
    # Configure the 'generate' method to be an async mock
    mock.generate = AsyncMock()
    return mock


@pytest.fixture
def mock_template_manager():
    """Fixture for a mocked ContentTemplateManager."""
    return MagicMock()


@pytest.fixture
def mock_settings():
    """Fixture for a mocked Settings object."""
    return MagicMock()


@pytest.fixture
def parser_service(mock_llm_service, mock_settings, mock_template_manager):
    """Fixture for the LLMCVParserService with mocked dependencies."""
    return LLMCVParserService(
        llm_service=mock_llm_service,
        settings=mock_settings,
        template_manager=mock_template_manager,
    )


@pytest.mark.asyncio
async def test_parse_json_valid(parser_service, mock_llm_service):
    """Test case 1: Mock response is valid JSON."""
    mock_llm_service.generate.return_value = LLMResponse(
        content=VALID_JSON_STRING, status_code=200
    )

    result = await parser_service._generate_and_parse_json("prompt", "sid", "tid")

    assert result == json.loads(VALID_JSON_STRING)
    mock_llm_service.generate.assert_awaited_once()


@pytest.mark.asyncio
async def test_parse_json_in_markdown(parser_service, mock_llm_service):
    """Test case 2: Mock response is JSON wrapped in ```json ... ```."""
    mock_llm_service.generate.return_value = LLMResponse(
        content=JSON_IN_MARKDOWN, status_code=200
    )

    result = await parser_service._generate_and_parse_json("prompt", "sid", "tid")

    assert result == json.loads(VALID_JSON_STRING)
    mock_llm_service.generate.assert_awaited_once()


@pytest.mark.asyncio
async def test_parse_json_empty_string_raises_error(parser_service, mock_llm_service):
    """Test case 3: Mock response is an empty string. Assert LLMResponseParsingError is raised."""
    mock_llm_service.generate.return_value = LLMResponse(
        content=EMPTY_STRING, status_code=200
    )

    with pytest.raises(
        LLMResponseParsingError, match="Received empty or non-string response from LLM."
    ):
        await parser_service._generate_and_parse_json("prompt", "sid", "tid")

    mock_llm_service.generate.assert_awaited_once()


@pytest.mark.asyncio
async def test_parse_json_malformed_raises_error(parser_service, mock_llm_service):
    """Test case 4: Mock response is malformed JSON. Assert LLMResponseParsingError is raised."""
    mock_llm_service.generate.return_value = LLMResponse(
        content=MALFORMED_JSON_STRING, status_code=200
    )

    with pytest.raises(
        LLMResponseParsingError, match="Could not parse JSON from LLM response"
    ):
        await parser_service._generate_and_parse_json("prompt", "sid", "tid")

    mock_llm_service.generate.assert_awaited_once()


@pytest.mark.asyncio
async def test_parse_json_no_json_found_raises_error(parser_service, mock_llm_service):
    """Test case: No JSON object found in the response."""
    mock_llm_service.generate.return_value = LLMResponse(
        content="This is a string without any json.", status_code=200
    )

    with pytest.raises(
        LLMResponseParsingError, match="No valid JSON object found in the LLM response."
    ):
        await parser_service._generate_and_parse_json("prompt", "sid", "tid")

    mock_llm_service.generate.assert_awaited_once()
