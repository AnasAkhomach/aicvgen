"""Unit test for ExecutiveSummaryWriterAgent."""

import pytest
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.agents.executive_summary_writer_agent import ExecutiveSummaryWriterAgent
from src.models.agent_models import AgentResult
from src.models.data_models import StructuredCV, JobDescriptionData, ContentType
from src.models.llm_service_models import LLMServiceResponse
from src.models.cv_models import Section, Item, ItemType, ItemStatus
from uuid import uuid4, UUID

# Fixed UUIDs for consistent testing
EXEC_SUMMARY_ID = UUID("12345678-1234-5678-9abc-123456789abc")
KEY_QUAL_ID = UUID("12345678-1234-5678-9abc-123456789abd")
PROF_EXP_ID = UUID("12345678-1234-5678-9abc-123456789abe")
PROJ_EXP_ID = UUID("12345678-1234-5678-9abc-123456789abf")

@pytest.fixture
def mock_llm_service():
    """Fixture for a mock EnhancedLLMService."""
    mock = AsyncMock()
    mock.generate_content.return_value = LLMServiceResponse(content="Generated executive summary.")
    return mock

@pytest.fixture
def mock_template_manager():
    """Fixture for a mock ContentTemplateManager."""
    mock = MagicMock()
    mock.get_template_by_type.return_value = (
        "Job Description: {job_description}\n" +
        "Key Qualifications: {key_qualifications}\n" +
        "Professional Experience: {professional_experience}\n" +
        "Projects: {projects}\n" +
        "Research Findings: {research_findings}"
    )
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
    """Fixture for a sample StructuredCV with all relevant sections."""
    return StructuredCV(
        sections=[
            Section(name="Executive Summary", items=[Item(id=EXEC_SUMMARY_ID, content="Original summary", item_type=ItemType.EXECUTIVE_SUMMARY_PARA)]),
            Section(name="Key Qualifications", items=[
                Item(id=KEY_QUAL_ID, content="Strategic thinker", item_type=ItemType.KEY_QUALIFICATION)
            ]),
            Section(name="Professional Experience", items=[
                Item(id=PROF_EXP_ID, content="Led cross-functional teams.", item_type=ItemType.EXPERIENCE_ROLE_TITLE)
            ]),
            Section(name="Project Experience", items=[
                Item(id=PROJ_EXP_ID, content="Delivered complex projects.", item_type=ItemType.PROJECT_DESCRIPTION_BULLET)
            ])
        ]
    )

@pytest.fixture
def sample_job_description_data():
    """Fixture for sample JobDescriptionData."""
    return JobDescriptionData(
        job_title="CTO",
        company_name="Global Innovations",
        raw_text="Seeking a visionary leader."
    )

@pytest.mark.asyncio
async def test_executive_summary_writer_agent_success(
    mock_llm_service, mock_template_manager, mock_settings, sample_structured_cv, sample_job_description_data
):
    """Test successful generation of executive summary content."""
    agent = ExecutiveSummaryWriterAgent(
        llm_service=mock_llm_service,
        template_manager=mock_template_manager,
        settings=mock_settings,
        session_id="test_session",
    )

    initial_state = {
        "session_id": "test_session",
        "structured_cv": sample_structured_cv,
        "job_description_data": sample_job_description_data,
        "research_findings": {"company_vision": "future-proof"},
        "cv_text": "mock cv text"
    }

    result = await agent.run_as_node(initial_state)

    assert isinstance(result, dict)
    assert result.get("error_messages", []) == []  # No errors should occur
    
    # Check that the executive summary was updated in the structured_cv
    updated_cv = result["structured_cv"]
    summary_section = next(s for s in updated_cv.sections if s.name == "Executive Summary")
    assert len(summary_section.items) == 1
    assert summary_section.items[0].content == "Generated executive summary."
    assert summary_section.items[0].status == ItemStatus.GENERATED

    mock_template_manager.get_template_by_type.assert_called_once_with(ContentType.EXECUTIVE_SUMMARY)
    mock_llm_service.generate_content.assert_called_once()
    call_args = mock_llm_service.generate_content.call_args[1]
    assert "Job Description:" in call_args["prompt"]
    assert "Key Qualifications: Strategic thinker" in call_args["prompt"]
    assert "Professional Experience: Led cross-functional teams." in call_args["prompt"]
    assert "Projects: Delivered complex projects." in call_args["prompt"]
    assert "Research Findings:" in call_args["prompt"]
    assert call_args["max_tokens"] == 1024
    assert call_args["temperature"] == 0.7

@pytest.mark.asyncio
async def test_executive_summary_writer_agent_missing_inputs(
    mock_llm_service, mock_template_manager, mock_settings, sample_structured_cv
):
    """Test agent failure when required inputs are missing."""
    agent = ExecutiveSummaryWriterAgent(
        llm_service=mock_llm_service,
        template_manager=mock_template_manager,
        settings=mock_settings,
        session_id="test_session",
    )

    # Test missing job_description_data
    initial_state_missing_jd = {
        "session_id": "test_session",
        "structured_cv": sample_structured_cv,
        "cv_text": "mock cv text"
    }
    result = await agent.run_as_node(initial_state_missing_jd)
    assert len(result.get("error_messages", [])) > 0
    assert "Missing or invalid 'job_description_data' in input_data." in result["error_messages"][0]

@pytest.mark.asyncio
async def test_executive_summary_writer_agent_llm_failure(
    mock_llm_service, mock_template_manager, mock_settings, sample_structured_cv, sample_job_description_data
):
    """Test agent failure when LLM does not generate content."""
    mock_llm_service.generate_content.return_value = LLMServiceResponse(content=None)

    agent = ExecutiveSummaryWriterAgent(
        llm_service=mock_llm_service,
        template_manager=mock_template_manager,
        settings=mock_settings,
        session_id="test_session",
    )

    initial_state = {
        "session_id": "test_session",
        "structured_cv": sample_structured_cv,
        "job_description_data": sample_job_description_data,
        "research_findings": {"company_vision": "future-proof"},
        "cv_text": "mock cv text"
    }

    result = await agent.run_as_node(initial_state)

    assert isinstance(result, dict)
    assert len(result.get("error_messages", [])) > 0
    assert "LLM failed to generate valid Executive Summary content." in result["error_messages"][0]

@pytest.mark.asyncio
async def test_executive_summary_writer_agent_no_summary_section(
    mock_llm_service, mock_template_manager, mock_settings, sample_job_description_data
):
    """Test agent failure when Executive Summary section is missing in CV."""
    # Create a structured_cv without an "Executive Summary" section
    cv_without_summary_section = StructuredCV(
        sections=[
            Section(name="Education", items=[]),
        ]
    )

    agent = ExecutiveSummaryWriterAgent(
        llm_service=mock_llm_service,
        template_manager=mock_template_manager,
        settings=mock_settings,
        session_id="test_session",
    )

    initial_state = {
        "session_id": "test_session",
        "structured_cv": cv_without_summary_section,
        "job_description_data": sample_job_description_data,
        "research_findings": {"company_vision": "future-proof"},
        "cv_text": "mock cv text"
    }

    result = await agent.run_as_node(initial_state)

    assert isinstance(result, dict)
    assert len(result.get("error_messages", [])) > 0
    assert "Executive Summary section not found in structured_cv." in result["error_messages"][0]

@pytest.mark.asyncio
async def test_executive_summary_writer_agent_template_not_found(
    mock_llm_service, mock_template_manager, mock_settings, sample_structured_cv, sample_job_description_data
):
    """Test agent failure when template is not found."""
    mock_template_manager.get_template_by_type.return_value = None

    agent = ExecutiveSummaryWriterAgent(
        llm_service=mock_llm_service,
        template_manager=mock_template_manager,
        settings=mock_settings,
        session_id="test_session",
    )

    initial_state = {
        "session_id": "test_session",
        "structured_cv": sample_structured_cv,
        "job_description_data": sample_job_description_data,
        "research_findings": {"company_vision": "future-proof"},
        "cv_text": "mock cv text"
    }

    result = await agent.run_as_node(initial_state)

    assert isinstance(result, dict)
    assert len(result.get("error_messages", [])) > 0
    assert f"No prompt template found for type {ContentType.EXECUTIVE_SUMMARY}" in result["error_messages"][0]
