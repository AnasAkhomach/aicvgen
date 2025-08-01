"""Unit test for KeyQualificationsWriterAgent with pure LCEL implementation."""

import pytest
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.agents.key_qualifications_writer_agent import KeyQualificationsWriterAgent
from src.models.data_models import StructuredCV, JobDescriptionData
from src.models.cv_models import Section, Item, ItemType, ItemStatus
from src.models.agent_output_models import KeyQualificationsLLMOutput
from uuid import uuid4


@pytest.fixture
def mock_config():
    """Fixture for mock configuration."""
    config = MagicMock()
    config.llm.gemini_api_key_primary = "test_api_key_primary"
    config.llm.gemini_api_key_fallback = "test_api_key_fallback"
    config.llm_settings.default_model = "gemini-1.5-flash"
    return config


@pytest.fixture
def sample_structured_cv():
    """Fixture for a sample StructuredCV with Key Qualifications section."""
    return StructuredCV(
        sections=[
            Section(
                name="Key Qualifications",
                items=[
                    Item(
                        content="",
                        item_type=ItemType.KEY_QUALIFICATION,
                        status=ItemStatus.PENDING,
                    )
                ],
            ),
            Section(
                name="Professional Summary",
                items=[
                    Item(
                        content="A seasoned professional with expertise in software development.",
                        item_type=ItemType.BULLET_POINT,
                        status=ItemStatus.COMPLETED,
                    )
                ],
            ),
            Section(
                name="Professional Experience",
                items=[
                    Item(
                        content="Senior Software Engineer at Tech Corp\nDeveloped scalable applications",
                        item_type=ItemType.BULLET_POINT,
                        status=ItemStatus.COMPLETED,
                    )
                ],
            ),
        ]
    )


@pytest.fixture
def sample_job_description_data():
    """Fixture for sample JobDescriptionData."""
    return JobDescriptionData(
        job_title="Software Engineer",
        company_name="Tech Corp",
        raw_text="We are looking for a skilled Software Engineer with experience in Python, machine learning, and cloud technologies. The ideal candidate should have strong problem-solving skills and experience with agile development.",
    )


@pytest.fixture
def sample_agent_input(sample_job_description_data):
    """Fixture for agent input data."""
    return {
        "main_job_description_raw": sample_job_description_data.raw_text,
        "my_talents": "A seasoned professional with expertise in software development. Senior Software Engineer at Tech Corp with experience in developing scalable applications.",
        "session_id": "test_session",
    }


@pytest.mark.asyncio
async def test_key_qualifications_writer_agent_success(sample_agent_input):
    """Test successful generation of key qualifications using Gold Standard LCEL."""
    # Mock the LLM output
    mock_llm_output = KeyQualificationsLLMOutput(
        qualifications=[
            "Python Development",
            "Machine Learning",
            "Cloud Technologies",
            "Agile Development",
            "Problem Solving",
        ]
    )

    # Create mock components for Gold Standard pattern
    mock_llm = AsyncMock()
    mock_prompt = MagicMock()
    mock_parser = MagicMock()
    mock_settings = MagicMock()

    # Create agent with Gold Standard pattern
    agent = KeyQualificationsWriterAgent(
        llm=mock_llm,
        prompt=mock_prompt,
        parser=mock_parser,
        settings=mock_settings,
        session_id="test_session",
    )

    # Mock the chain execution
    agent.chain = AsyncMock()
    agent.chain.ainvoke.return_value = mock_llm_output

    # Execute the agent
    result = await agent._execute(**sample_agent_input)

    # Verify results
    assert isinstance(result, dict)
    assert "generated_key_qualifications" in result
    assert result["generated_key_qualifications"] == mock_llm_output.qualifications

    # Verify chain was called with correct input
    agent.chain.ainvoke.assert_called_once()
    call_args = agent.chain.ainvoke.call_args[0][0]
    assert "main_job_description_raw" in call_args
    assert "my_talents" in call_args


@pytest.mark.asyncio
async def test_key_qualifications_writer_agent_success_with_different_talents():
    """Test successful generation with different talent input."""
    # Mock the LLM output
    mock_llm_output = KeyQualificationsLLMOutput(
        qualifications=[
            "Test qualification 1",
            "Test qualification 2",
            "Test qualification 3",
        ]
    )

    # Create mock components for Gold Standard pattern
    mock_llm = AsyncMock()
    mock_prompt = MagicMock()
    mock_parser = MagicMock()
    mock_settings = MagicMock()

    # Create agent with Gold Standard pattern
    agent = KeyQualificationsWriterAgent(
        llm=mock_llm,
        prompt=mock_prompt,
        parser=mock_parser,
        settings=mock_settings,
        session_id="test_session",
    )

    # Mock the chain with async support
    agent.chain = AsyncMock()
    agent.chain.ainvoke = AsyncMock(return_value=mock_llm_output)

    # Execute the agent
    result = await agent._execute(
        main_job_description_raw="Software Engineer position requiring Python skills",
        my_talents="A professional summary.",
        session_id="test_session",
    )

    # Verify successful generation
    assert isinstance(result, dict)
    assert "generated_key_qualifications" in result
    assert result["generated_key_qualifications"] == mock_llm_output.qualifications


@pytest.mark.asyncio
async def test_key_qualifications_writer_agent_llm_failure(sample_agent_input):
    """Test agent failure when chain fails."""
    # Create mock components for Gold Standard pattern
    mock_llm = AsyncMock()
    mock_prompt = MagicMock()
    mock_parser = MagicMock()
    mock_settings = MagicMock()

    # Create agent with Gold Standard pattern
    agent = KeyQualificationsWriterAgent(
        llm=mock_llm,
        prompt=mock_prompt,
        parser=mock_parser,
        settings=mock_settings,
        session_id="test_session",
    )

    # Mock the chain to raise an exception
    agent.chain = AsyncMock()
    agent.chain.ainvoke = AsyncMock(side_effect=Exception("LLM failure"))

    # Execute the agent and expect an exception
    with pytest.raises(Exception) as exc_info:
        await agent._execute(**sample_agent_input)

    # Verify the exception is raised
    assert "LLM failure" in str(exc_info.value)


@pytest.mark.asyncio
async def test_key_qualifications_writer_agent_empty_qualifications(sample_agent_input):
    """Test agent behavior when LLM returns empty qualifications."""
    # Create mock components for Gold Standard pattern
    mock_llm = AsyncMock()
    mock_prompt = MagicMock()
    mock_parser = MagicMock()
    mock_settings = MagicMock()

    # Create agent with Gold Standard pattern
    agent = KeyQualificationsWriterAgent(
        llm=mock_llm,
        prompt=mock_prompt,
        parser=mock_parser,
        settings=mock_settings,
        session_id="test_session",
    )

    # Mock the chain to return None (simulating LLM failure)
    agent.chain = AsyncMock()
    agent.chain.ainvoke = AsyncMock(return_value=None)

    # Execute the agent and expect an exception
    with pytest.raises(Exception) as exc_info:
        await agent._execute(**sample_agent_input)

    # Verify the exception is raised
    assert "No qualifications generated" in str(exc_info.value)


def test_key_qualifications_agent_input_validation():
    """Test the KeyQualificationsAgentInput Pydantic model validation."""
    from src.agents.key_qualifications_writer_agent import KeyQualificationsAgentInput

    # Test valid input
    valid_data = {
        "main_job_description_raw": "Software Engineer position",
        "my_talents": "Python, Machine Learning",
        "session_id": "test_session",
    }

    input_model = KeyQualificationsAgentInput(**valid_data)
    assert input_model.main_job_description_raw == "Software Engineer position"
    assert input_model.my_talents == "Python, Machine Learning"
    assert input_model.session_id == "test_session"

    # Test invalid input (missing required fields)
    with pytest.raises(Exception):  # Pydantic validation error
        KeyQualificationsAgentInput(main_job_description_raw="test")
