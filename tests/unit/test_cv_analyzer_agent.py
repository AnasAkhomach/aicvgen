"""Unit tests for the CVAnalyzerAgent."""

import pytest
from unittest.mock import MagicMock, AsyncMock

# Module-level constants for reuse
MOCK_SETTINGS = MagicMock()
MOCK_LLM_SERVICE = MagicMock()
MOCK_PROGRESS_TRACKER = MagicMock()
MOCK_TEMPLATE_MANAGER = MagicMock()
MOCK_CONTEXT = MagicMock()

# Test data
VALID_CV_TEXT = "Experienced Python Developer with 5 years of experience."
VALID_LLM_RESPONSE_CONTENT = """
Here is the JSON:
```json
{
    "summary": "An experienced software developer.",
    "key_skills": ["Python", "FastAPI", "Docker"],
    "experience": [
        {
            "role": "Senior Developer",
            "company": "Tech Corp",
            "duration": "2020-2024",
            "responsibilities": ["Developed APIs", "Mentored junior developers"]
        }
    ]
}
```
"""
MALFORMED_LLM_RESPONSE_CONTENT = "This is not a valid JSON."
EMPTY_INPUT_DATA = {"user_cv": {"raw_text": ""}}
VALID_INPUT_DATA = {
    "user_cv": {"raw_text": VALID_CV_TEXT},
    "job_description": "Some job description",
}


@pytest.fixture
def cv_analyzer_agent():
    """Fixture to create a CVAnalyzerAgent instance with mocked dependencies."""
    from src.agents.cv_analyzer_agent import CVAnalyzerAgent

    # Reset mocks before each test to ensure isolation
    MOCK_LLM_SERVICE.reset_mock()
    MOCK_TEMPLATE_MANAGER.reset_mock()

    # Mock the template manager to return a simple formatted string
    mock_template = MagicMock()
    mock_template.format.return_value = "Formatted prompt"
    MOCK_TEMPLATE_MANAGER.get_template.return_value = mock_template

    return CVAnalyzerAgent(
        {
            "llm_service": MOCK_LLM_SERVICE,
            "settings": MOCK_SETTINGS,
            "progress_tracker": MOCK_PROGRESS_TRACKER,
            "template_manager": MOCK_TEMPLATE_MANAGER,
        }
    )


@pytest.mark.asyncio
async def test_analyze_cv_success(cv_analyzer_agent):
    """
    Tests the happy path where the LLM returns a valid, parsable JSON response.
    """
    # Arrange
    # Configure the async mock for the LLM service
    mock_llm_response = MagicMock()
    mock_llm_response.content = VALID_LLM_RESPONSE_CONTENT
    MOCK_LLM_SERVICE.invoke_async = AsyncMock(return_value=mock_llm_response)

    # Act
    result = await cv_analyzer_agent.analyze_cv(VALID_INPUT_DATA, MOCK_CONTEXT)

    # Assert
    assert result is not None
    assert result.analysis_results is not None
    assert result.analysis_results.summary == "An experienced software developer."
    assert "Python" in result.analysis_results.key_skills
    MOCK_LLM_SERVICE.invoke_async.assert_awaited_once()
    MOCK_TEMPLATE_MANAGER.get_template.assert_called_once_with("cv_analysis_prompt")


@pytest.mark.asyncio
async def test_run_async_success_path(cv_analyzer_agent):
    """
    Tests the full run_async method on a successful analysis.
    """
    # Arrange
    mock_llm_response = MagicMock()
    mock_llm_response.content = VALID_LLM_RESPONSE_CONTENT
    MOCK_LLM_SERVICE.invoke_async = AsyncMock(return_value=mock_llm_response)

    # Act
    agent_result = await cv_analyzer_agent.run_async(VALID_INPUT_DATA, MOCK_CONTEXT)

    # Assert
    assert agent_result.success is True
    assert agent_result.output_data is not None
    assert (
        agent_result.output_data.analysis_results.summary
        == "An experienced software developer."
    )
    assert agent_result.error_message is None


@pytest.mark.asyncio
async def test_run_async_llm_parsing_error_raises(cv_analyzer_agent):
    """
    Tests that run_async raises LLMResponseParsingError on LLM parsing error.
    """
    from src.utils.exceptions import LLMResponseParsingError

    mock_llm_response = MagicMock()
    mock_llm_response.content = MALFORMED_LLM_RESPONSE_CONTENT
    MOCK_LLM_SERVICE.invoke_async = AsyncMock(return_value=mock_llm_response)

    with pytest.raises(LLMResponseParsingError):
        await cv_analyzer_agent.run_async(VALID_INPUT_DATA, MOCK_CONTEXT)


@pytest.mark.asyncio
async def test_run_async_empty_input_error_raises(cv_analyzer_agent):
    """
    Tests that run_async raises ValueError from empty input.
    """
    MOCK_LLM_SERVICE.invoke_async = AsyncMock()
    with pytest.raises(ValueError):
        await cv_analyzer_agent.run_async(EMPTY_INPUT_DATA, MOCK_CONTEXT)
    MOCK_LLM_SERVICE.invoke_async.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_async_template_loading_error_raises(cv_analyzer_agent):
    """
    Tests that run_async raises AgentExecutionError if the prompt template cannot be loaded.
    """
    from src.utils.exceptions import AgentExecutionError

    MOCK_TEMPLATE_MANAGER.get_template.side_effect = FileNotFoundError(
        "Template not found"
    )
    MOCK_LLM_SERVICE.invoke_async = AsyncMock()
    with pytest.raises(AgentExecutionError):
        await cv_analyzer_agent.run_async(VALID_INPUT_DATA, MOCK_CONTEXT)
    MOCK_LLM_SERVICE.invoke_async.assert_not_awaited()
