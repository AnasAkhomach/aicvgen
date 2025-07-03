"""Unit test for KeyQualificationsWriterAgent."""

import pytest
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.agents.key_qualifications_writer_agent import KeyQualificationsWriterAgent
from src.models.agent_models import AgentResult
from src.models.data_models import StructuredCV, JobDescriptionData, ContentType
from src.models.llm_service_models import LLMServiceResponse
from src.models.cv_models import Section, Item, ItemType, ItemStatus
from uuid import uuid4
from src.orchestration.state import AgentState

@pytest.fixture
def mock_llm_service():
    """Fixture for a mock EnhancedLLMService."""
    mock = AsyncMock()
    mock.generate_content.return_value = LLMServiceResponse(content="- Qualification 1\n- Qualification 2")
    return mock

@pytest.fixture
def mock_template_manager():
    """Fixture for a mock ContentTemplateManager."""
    mock = MagicMock()
    mock.get_template_by_type.return_value = "Job Description: {job_description}\nCV Summary: {cv_summary}\nResearch Findings: {research_findings}"
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
    """Fixture for a sample StructuredCV."""
    return StructuredCV(
        sections=[
            Section(name="Key Qualifications", items=[]),
            Section(name="Executive Summary", items=[Item(id=uuid4(), content="A seasoned professional.", item_type=ItemType.EXECUTIVE_SUMMARY_PARA)])
        ]
    )

@pytest.fixture
def sample_job_description_data():
    """Fixture for sample JobDescriptionData."""
    return JobDescriptionData(
        job_title="Software Engineer",
        company_name="Tech Corp",
        raw_text="Job description details."
    )

@pytest.mark.asyncio
async def test_key_qualifications_writer_agent_success(
    mock_llm_service, mock_template_manager, mock_settings, sample_structured_cv, sample_job_description_data
):
    """Test successful generation of key qualifications."""
    agent = KeyQualificationsWriterAgent(
        llm_service=mock_llm_service,
        template_manager=mock_template_manager,
        settings=mock_settings,
        session_id="test_session",
    )

    initial_state = AgentState(
        session_id="test_session",
        structured_cv=sample_structured_cv,
        job_description_data=sample_job_description_data,
        research_findings={"company_values": "innovation"},
        cv_text="mock cv text"
    )

    result = await agent.run_as_node(initial_state)
    print(f"DEBUG: result = {result}")
    print(f"DEBUG: result type = {type(result)}")
    print(f"DEBUG: result.get('success') = {result.get('success') if isinstance(result, dict) else 'Not a dict'}")

    assert isinstance(result, dict)
    assert not result.get("error_messages")

    updated_cv = result["structured_cv"]
    qual_section = next(s for s in updated_cv.sections if s.name == "Key Qualifications")
    assert len(qual_section.items) == 2
    assert qual_section.items[0].content == "Qualification 1"
    assert qual_section.items[1].content == "Qualification 2"
    assert qual_section.items[0].status == ItemStatus.GENERATED
    assert qual_section.items[0].item_type == ItemType.KEY_QUALIFICATION

    mock_template_manager.get_template_by_type.assert_called_once_with(ContentType.QUALIFICATION)
    mock_llm_service.generate_content.assert_called_once()
    call_args = mock_llm_service.generate_content.call_args[1]
    assert "Job Description:" in call_args["prompt"]
    assert "CV Summary: A seasoned professional." in call_args["prompt"]
    assert "Research Findings:" in call_args["prompt"]
    assert call_args["max_tokens"] == 1024
    assert call_args["temperature"] == 0.7

@pytest.mark.asyncio
async def test_key_qualifications_writer_agent_missing_structured_cv(
    mock_llm_service, mock_template_manager, mock_settings, sample_job_description_data
):
    """Test agent failure when structured_cv is missing."""
    agent = KeyQualificationsWriterAgent(
        llm_service=mock_llm_service,
        template_manager=mock_template_manager,
        settings=mock_settings,
        session_id="test_session",
    )

    # Create an empty structured_cv to test missing data validation
    empty_structured_cv = StructuredCV(sections=[])
    
    initial_state = AgentState(
        session_id="test_session",
        structured_cv=empty_structured_cv,
        cv_text="Sample CV text",
        job_description_data=sample_job_description_data,
        research_findings={"company_values": "innovation"}
    )

    result = await agent.run_as_node(initial_state)

    assert isinstance(result, dict)
    assert result.get("error_messages")
    assert "Key Qualifications section not found in structured_cv." in result["error_messages"][0]

@pytest.mark.asyncio
async def test_key_qualifications_writer_agent_llm_failure(
    mock_llm_service, mock_template_manager, mock_settings, sample_structured_cv, sample_job_description_data
):
    """Test agent failure when LLM does not generate content."""
    mock_llm_service.generate_content.return_value = LLMServiceResponse(content=None)

    agent = KeyQualificationsWriterAgent(
        llm_service=mock_llm_service,
        template_manager=mock_template_manager,
        settings=mock_settings,
        session_id="test_session",
    )

    initial_state = AgentState(
        session_id="test_session",
        structured_cv=sample_structured_cv,
        job_description_data=sample_job_description_data,
        research_findings={"company_values": "innovation"},
        cv_text="mock cv text"
    )

    result = await agent.run_as_node(initial_state)
    
    assert isinstance(result, dict)
    assert result.get("error_messages")
    error_msg = result["error_messages"][0]
    assert "LLM failed" in error_msg or "generate" in error_msg or "Key Qualifications" in error_msg

@pytest.mark.asyncio
async def test_key_qualifications_writer_agent_no_qual_section(
    mock_llm_service, mock_template_manager, mock_settings, sample_job_description_data
):
    """Test agent failure when Key Qualifications section is missing in CV."""
    # Create a structured_cv without a "Key Qualifications" section
    cv_without_qual_section = StructuredCV(
        sections=[
            Section(name="Education", items=[]),
        ]
    )

    agent = KeyQualificationsWriterAgent(
        llm_service=mock_llm_service,
        template_manager=mock_template_manager,
        settings=mock_settings,
        session_id="test_session",
    )

    initial_state = AgentState(
        session_id="test_session",
        structured_cv=cv_without_qual_section,
        cv_text="Sample CV text",
        job_description_data=sample_job_description_data,
        research_findings={"company_values": "innovation"}
    )

    result = await agent.run_as_node(initial_state)

    assert isinstance(result, dict)
    assert result.get("error_messages")
    assert "Key Qualifications section not found in structured_cv." in result["error_messages"][0]

@pytest.mark.asyncio
async def test_key_qualifications_writer_agent_empty_llm_response(
    mock_llm_service, mock_template_manager, mock_settings, sample_structured_cv, sample_job_description_data
):
    """Test agent handling of empty LLM response content."""
    mock_llm_service.generate_content.return_value = LLMServiceResponse(content="")

    agent = KeyQualificationsWriterAgent(
        llm_service=mock_llm_service,
        template_manager=mock_template_manager,
        settings=mock_settings,
        session_id="test_session",
    )

    initial_state = AgentState(
        session_id="test_session",
        structured_cv=sample_structured_cv,
        job_description_data=sample_job_description_data,
        research_findings={"company_values": "innovation"},
        cv_text="mock cv text"
    )

    result = await agent.run_as_node(initial_state)

    assert isinstance(result, dict)
    assert result.get("error_messages")
    assert "LLM failed to generate valid" in result["error_messages"][0]
    assert "Key Qualifications" in result["error_messages"][0]

@pytest.mark.asyncio
async def test_key_qualifications_writer_agent_template_not_found(
    mock_llm_service, mock_template_manager, mock_settings, sample_structured_cv, sample_job_description_data
):
    """Test agent failure when template is not found."""
    mock_template_manager.get_template_by_type.return_value = None

    agent = KeyQualificationsWriterAgent(
        llm_service=mock_llm_service,
        template_manager=mock_template_manager,
        settings=mock_settings,
        session_id="test_session",
    )

    initial_state = AgentState(
        session_id="test_session",
        structured_cv=sample_structured_cv,
        job_description_data=sample_job_description_data,
        research_findings={"company_values": "innovation"},
        cv_text="mock cv text"
    )

    result = await agent.run_as_node(initial_state)

    assert isinstance(result, dict)
    assert result.get("error_messages")
    error_msg = result["error_messages"][0]
    assert "template" in error_msg or "QUALIFICATION" in error_msg or "prompt" in error_msg


# You would repeat similar tests for ProfessionalExperienceWriterAgent, ProjectsWriterAgent, and ExecutiveSummaryWriterAgent
# For brevity, only KeyQualificationsWriterAgent is fully implemented here.
