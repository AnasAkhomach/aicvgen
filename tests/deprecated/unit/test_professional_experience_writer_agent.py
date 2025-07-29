"""Unit test for ProfessionalExperienceWriterAgent."""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from uuid import UUID, uuid4

from src.agents.professional_experience_writer_agent import (
    ProfessionalExperienceWriterAgent,
)
from src.models.agent_models import AgentResult
from src.models.cv_models import Item, ItemStatus, ItemType, Section
from src.models.data_models import ContentType, JobDescriptionData, StructuredCV
from src.models.llm_service_models import LLMServiceResponse
from src.orchestration.state import AgentState


@pytest.fixture
def mock_llm_service():
    """Fixture for a mock EnhancedLLMService."""
    mock = AsyncMock()
    mock.generate_content.return_value = LLMServiceResponse(
        content="Generated professional experience content."
    )
    return mock


@pytest.fixture
def mock_template_manager():
    """Fixture for a mock ContentTemplateManager."""
    mock = MagicMock()
    mock.get_template_by_type.return_value = "Job Description: {job_description}\nExperience Item: {experience_item}\nKey Qualifications: {key_qualifications}\nResearch Findings: {research_findings}"
    mock.format_template.return_value = "Job Description: test job data\nExperience Item: test experience\nKey Qualifications: Strong leadership skills\nResearch Findings: test research"
    return mock


@pytest.fixture
def mock_settings():
    """Fixture for mock settings."""
    return {
        "max_tokens_content_generation": 1024,
        "temperature_content_generation": 0.7,
    }


@pytest.fixture
def sample_structured_cv():
    """Fixture for a sample StructuredCV with professional experience and key qualifications."""
    return StructuredCV(
        sections=[
            Section(
                name="Key Qualifications",
                items=[
                    Item(
                        id=UUID("11111111-1111-1111-1111-111111111111"),
                        content="Strong leadership skills",
                        item_type=ItemType.KEY_QUALIFICATION,
                    )
                ],
            ),
            Section(
                name="Professional Experience",
                items=[
                    Item(
                        id=UUID("12345678-1234-5678-9012-123456789012"),
                        content="Original experience 1",
                        item_type=ItemType.EXPERIENCE_ROLE_TITLE,
                    ),
                    Item(
                        id=uuid4(),
                        content="Original experience 2",
                        item_type=ItemType.EXPERIENCE_ROLE_TITLE,
                    ),
                ],
            ),
        ]
    )


@pytest.fixture
def sample_job_description_data():
    """Fixture for sample JobDescriptionData."""
    return JobDescriptionData(
        job_title="Senior Software Engineer",
        company_name="InnovateX",
        raw_text="Seeking a senior engineer with leadership skills.",
    )


@pytest.mark.asyncio
async def test_professional_experience_writer_agent_success(
    mock_llm_service,
    mock_template_manager,
    mock_settings,
    sample_structured_cv,
    sample_job_description_data,
):
    """Test successful generation of professional experience content."""
    agent = ProfessionalExperienceWriterAgent(
        llm_service=mock_llm_service,
        template_manager=mock_template_manager,
        settings=mock_settings,
        session_id="test_session",
    )

    initial_state = AgentState(
        session_id="test_session",
        structured_cv=sample_structured_cv,
        job_description_data=sample_job_description_data,
        current_item_id="12345678-1234-5678-9012-123456789012",
        research_findings={"company_culture": "fast-paced"},
        cv_text="mock cv text",
    )

    result = await agent.run_as_node(initial_state)

    assert isinstance(result, dict)
    assert "structured_cv" in result
    assert "current_item_id" in result
    assert "error_messages" not in result

    updated_cv = result["structured_cv"]
    exp_section = next(
        s for s in updated_cv.sections if s.name == "Professional Experience"
    )
    updated_item = next(
        item
        for item in exp_section.items
        if str(item.id) == "12345678-1234-5678-9012-123456789012"
    )

    assert updated_item.content == "Generated professional experience content."
    assert updated_item.status == ItemStatus.GENERATED

    mock_template_manager.get_template_by_type.assert_called_once_with(
        ContentType.EXPERIENCE
    )
    mock_llm_service.generate_content.assert_called_once()
    call_args = mock_llm_service.generate_content.call_args[1]
    assert "Job Description:" in call_args["prompt"]
    assert "Experience Item:" in call_args["prompt"]
    assert "Key Qualifications: Strong leadership skills" in call_args["prompt"]
    assert "Research Findings:" in call_args["prompt"]
    assert call_args["max_tokens"] == 1024
    assert call_args["temperature"] == 0.7


@pytest.mark.asyncio
async def test_professional_experience_writer_agent_missing_inputs(
    mock_llm_service,
    mock_template_manager,
    mock_settings,
    sample_structured_cv,
    sample_job_description_data,
):
    """Test agent failure when required inputs are missing."""
    agent = ProfessionalExperienceWriterAgent(
        llm_service=mock_llm_service,
        template_manager=mock_template_manager,
        settings=mock_settings,
        session_id="test_session",
    )

    # Test missing job_description_data
    initial_state_missing_jd = AgentState(
        session_id="test_session",
        structured_cv=sample_structured_cv,
        current_item_id="exp1",
        cv_text="mock cv text",
    )
    result = await agent.run_as_node(initial_state_missing_jd)
    assert isinstance(result, dict)
    assert "error_messages" in result
    assert "Input validation failed" in result["error_messages"][0]
    assert "job_description_data" in result["error_messages"][0]

    # Test missing current_item_id
    initial_state_missing_item_id = AgentState(
        session_id="test_session",
        structured_cv=sample_structured_cv,
        job_description_data=sample_job_description_data,
        cv_text="mock cv text",
    )
    result = await agent.run_as_node(initial_state_missing_item_id)
    assert isinstance(result, dict)
    assert "error_messages" in result
    assert (
        "Missing or invalid 'current_item_id' in input_data."
        in result["error_messages"][0]
    )


@pytest.mark.asyncio
async def test_professional_experience_writer_agent_llm_failure(
    mock_llm_service,
    mock_template_manager,
    mock_settings,
    sample_structured_cv,
    sample_job_description_data,
):
    """Test agent failure when LLM does not generate content."""
    mock_llm_service.generate_content.return_value = LLMServiceResponse(content=None)

    agent = ProfessionalExperienceWriterAgent(
        llm_service=mock_llm_service,
        template_manager=mock_template_manager,
        settings=mock_settings,
        session_id="test_session",
    )

    initial_state = AgentState(
        session_id="test_session",
        structured_cv=sample_structured_cv,
        job_description_data=sample_job_description_data,
        current_item_id="12345678-1234-5678-9012-123456789012",
        research_findings={"company_culture": "fast-paced"},
        cv_text="mock cv text",
    )

    result = await agent.run_as_node(initial_state)

    assert isinstance(result, dict)
    assert "error_messages" in result
    assert (
        "Agent 'ProfessionalExperienceWriter' failed: LLM failed to generate valid Professional Experience content."
        in result["error_messages"][0]
    )


@pytest.mark.asyncio
async def test_professional_experience_writer_agent_item_not_found(
    mock_llm_service,
    mock_template_manager,
    mock_settings,
    sample_structured_cv,
    sample_job_description_data,
):
    """Test agent failure when the specified item_id is not found or is wrong type."""
    agent = ProfessionalExperienceWriterAgent(
        llm_service=mock_llm_service,
        template_manager=mock_template_manager,
        settings=mock_settings,
        session_id="test_session",
    )

    initial_state = AgentState(
        session_id="test_session",
        structured_cv=sample_structured_cv,
        job_description_data=sample_job_description_data,
        current_item_id="87654321-4321-8765-2109-876543210987",
        research_findings={"company_culture": "fast-paced"},
        cv_text="mock cv text",
    )

    result = await agent.run_as_node(initial_state)

    assert isinstance(result, dict)
    assert "error_messages" in result
    assert (
        "Item with ID '87654321-4321-8765-2109-876543210987' not found or is not a professional experience item."
        in result["error_messages"][0]
    )

    # Test with an item of wrong type
    initial_state_wrong_type = AgentState(
        session_id="test_session",
        structured_cv=sample_structured_cv,
        job_description_data=sample_job_description_data,
        current_item_id="11111111-1111-1111-1111-111111111111",  # This is a Key Qualification item
        research_findings={"company_culture": "fast-paced"},
        cv_text="mock cv text",
    )
    result = await agent.run_as_node(initial_state_wrong_type)
    assert isinstance(result, dict)
    assert "error_messages" in result
    assert (
        "Item with ID '11111111-1111-1111-1111-111111111111' not found or is not a professional experience item."
        in result["error_messages"][0]
    )


@pytest.mark.asyncio
async def test_professional_experience_writer_agent_template_not_found(
    mock_llm_service,
    mock_template_manager,
    mock_settings,
    sample_structured_cv,
    sample_job_description_data,
):
    """Test agent failure when template is not found."""
    mock_template_manager.get_template_by_type.return_value = None

    agent = ProfessionalExperienceWriterAgent(
        llm_service=mock_llm_service,
        template_manager=mock_template_manager,
        settings=mock_settings,
        session_id="test_session",
    )

    initial_state = AgentState(
        session_id="test_session",
        structured_cv=sample_structured_cv,
        job_description_data=sample_job_description_data,
        current_item_id="12345678-1234-5678-9012-123456789012",
        research_findings={"company_culture": "fast-paced"},
        cv_text="mock cv text",
    )

    result = await agent.run_as_node(initial_state)

    assert isinstance(result, dict)
    assert "error_messages" in result
    assert (
        f"No prompt template found for type {ContentType.EXPERIENCE}"
        in result["error_messages"][0]
    )
