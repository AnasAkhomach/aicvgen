"""Unit test for ProjectsWriterAgent."""

import pytest
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.agents.projects_writer_agent import ProjectsWriterAgent
from src.models.agent_models import AgentResult
from src.models.data_models import StructuredCV, JobDescriptionData, ContentType
from src.models.llm_service_models import LLMServiceResponse
from src.models.cv_models import Section, Item, ItemType, ItemStatus
from src.orchestration.state import AgentState

from uuid import uuid4, UUID

@pytest.fixture
def mock_llm_service():
    """Fixture for a mock EnhancedLLMService."""
    mock = AsyncMock()
    mock.generate_content.return_value = LLMServiceResponse(content="Generated project content.")
    return mock

@pytest.fixture
def mock_template_manager():
    """Fixture for a mock ContentTemplateManager."""
    mock = MagicMock()
    mock.get_template_by_type.return_value = (
        "Job Description: {job_description}\n" +
        "Project Item: {project_item}\n" +
        "Key Qualifications: {key_qualifications}\n" +
        "Professional Experience: {professional_experience}\n" +
        "Research Findings: {research_findings}"
    )
    # Mock format_template to return a formatted string
    mock.format_template.return_value = (
        "Job Description: {\"job_title\": \"Full Stack Developer\", \"company_name\": \"InnovateTech\", \"raw_text\": \"Seeking developer with strong project delivery.\"}\n" +
        "Project Item: {'id': UUID('12345678-1234-5678-9012-123456789012'), 'content': 'Original project 1', 'item_type': <ItemType.PROJECT_DESCRIPTION_BULLET: 'project_description_bullet'>, 'status': <ItemStatus.ORIGINAL: 'original'>}\n" +
        "Key Qualifications: Problem-solving expert\n" +
        "Professional Experience: Developed scalable solutions.\n" +
        "Research Findings: {'tech_stack': 'Python, React'}"
    )
    return mock

@pytest.fixture
def mock_settings():
    """Fixture for mock settings."""
    return {
        "max_tokens_content_generation": 1024,
        "temperature_content_generation": 0.7,
    }

# Define fixed UUIDs for testing
PROJ1_ID = UUID("12345678-1234-5678-9012-123456789012")
PROJ2_ID = UUID("12345678-1234-5678-9012-123456789013")
QUAL1_ID = UUID("12345678-1234-5678-9012-123456789014")
EXP1_ID = UUID("12345678-1234-5678-9012-123456789015")

@pytest.fixture
def sample_structured_cv():
    """Fixture for a sample StructuredCV with projects, experience, and qualifications."""
    return StructuredCV(
        sections=[
            Section(name="Key Qualifications", items=[
                Item(id=QUAL1_ID, content="Problem-solving expert", item_type=ItemType.KEY_QUALIFICATION)
            ]),
            Section(name="Professional Experience", items=[
                Item(id=EXP1_ID, content="Developed scalable solutions.", item_type=ItemType.EXPERIENCE_ROLE_TITLE)
            ]),
            Section(name="Project Experience", items=[
                Item(id=PROJ1_ID, content="Original project 1", item_type=ItemType.PROJECT_DESCRIPTION_BULLET),
                Item(id=PROJ2_ID, content="Original project 2", item_type=ItemType.PROJECT_DESCRIPTION_BULLET)
            ])
        ]
    )

@pytest.fixture
def sample_job_description_data():
    """Fixture for sample JobDescriptionData."""
    return JobDescriptionData(
        job_title="Full Stack Developer",
        company_name="InnovateTech",
        raw_text="Seeking developer with strong project delivery."
    )

@pytest.mark.asyncio
async def test_projects_writer_agent_success(
    mock_llm_service, mock_template_manager, mock_settings, sample_structured_cv, sample_job_description_data
):
    """Test successful generation of project experience content."""
    agent = ProjectsWriterAgent(
        llm_service=mock_llm_service,
        template_manager=mock_template_manager,
        settings=mock_settings,
        session_id="test_session",
    )

    initial_state = AgentState(
        session_id="test_session",
        structured_cv=sample_structured_cv,
        job_description_data=sample_job_description_data,
        current_item_id=str(PROJ1_ID),
        research_findings={"tech_stack": "Python, React"},
        cv_text="mock cv text"
    )

    result = await agent.run_as_node(initial_state)

    assert isinstance(result, AgentState)
    assert not result.error_messages

    updated_cv = result.structured_cv
    proj_section = next(s for s in updated_cv.sections if s.name == "Project Experience")
    updated_item = next(item for item in proj_section.items if item.id == PROJ1_ID)

    assert updated_item.content == "Generated project content."
    assert updated_item.status == ItemStatus.GENERATED

    mock_template_manager.get_template_by_type.assert_called_once_with(ContentType.PROJECT)
    mock_llm_service.generate_content.assert_called_once()
    call_args = mock_llm_service.generate_content.call_args[1]
    assert "Job Description:" in call_args["prompt"]
    assert "Project Item:" in call_args["prompt"]
    assert "Key Qualifications: Problem-solving expert" in call_args["prompt"]
    assert "Professional Experience: Developed scalable solutions." in call_args["prompt"]
    assert "Research Findings:" in call_args["prompt"]
    assert call_args["max_tokens"] == 1024
    assert call_args["temperature"] == 0.7

@pytest.mark.asyncio
async def test_projects_writer_agent_missing_inputs(
    mock_llm_service, mock_template_manager, mock_settings, sample_structured_cv, sample_job_description_data
):
    """Test agent failure when required inputs are missing."""
    agent = ProjectsWriterAgent(
        llm_service=mock_llm_service,
        template_manager=mock_template_manager,
        settings=mock_settings,
        session_id="test_session",
    )

    # Test missing job_description_data
    initial_state_missing_jd = AgentState(
        session_id="test_session",
        structured_cv=sample_structured_cv,
        current_item_id=str(PROJ1_ID),
        cv_text="mock cv text"
    )
    result = await agent.run_as_node(initial_state_missing_jd)
    assert result.error_messages
    assert "Input extraction failed:" in result.error_messages[0]
    assert "job_description_data" in result.error_messages[0]

    # Test missing structured_cv - this will cause a Pydantic validation error during AgentState creation
    # So we'll test with a minimal AgentState that has structured_cv but missing job_description_data
    from src.models.cv_models import StructuredCV as EmptyCV
    minimal_cv = EmptyCV(sections=[])
    initial_state_missing_cv = AgentState(
        session_id="test_session",
        structured_cv=minimal_cv,
        current_item_id=str(PROJ1_ID),
        cv_text="mock cv text"
    )
    result = await agent.run_as_node(initial_state_missing_cv)
    assert result.error_messages
    assert "Input extraction failed:" in result.error_messages[0]
    assert "job_description_data" in result.error_messages[0]

@pytest.mark.asyncio
async def test_projects_writer_agent_llm_failure(
    mock_llm_service, mock_template_manager, mock_settings, sample_structured_cv, sample_job_description_data
):
    """Test agent failure when LLM does not generate content."""
    mock_llm_service.generate_content.return_value = LLMServiceResponse(content=None)

    agent = ProjectsWriterAgent(
        llm_service=mock_llm_service,
        template_manager=mock_template_manager,
        settings=mock_settings,
        session_id="test_session",
    )

    initial_state = AgentState(
        session_id="test_session",
        structured_cv=sample_structured_cv,
        job_description_data=sample_job_description_data,
        current_item_id=str(PROJ1_ID),
        research_findings={"tech_stack": "Python, React"},
        cv_text="mock cv text"
    )

    result = await agent.run_as_node(initial_state)

    assert isinstance(result, AgentState)
    assert result.error_messages
    assert "LLM failed to generate valid Projects content." in result.error_messages[0]

@pytest.mark.asyncio
async def test_projects_writer_agent_item_not_found(
    mock_llm_service, mock_template_manager, mock_settings, sample_structured_cv, sample_job_description_data
):
    """Test agent failure when the specified item_id is not found or is wrong type."""
    agent = ProjectsWriterAgent(
        llm_service=mock_llm_service,
        template_manager=mock_template_manager,
        settings=mock_settings,
        session_id="test_session",
    )

    initial_state = AgentState(
        session_id="test_session",
        structured_cv=sample_structured_cv,
        job_description_data=sample_job_description_data,
        current_item_id="non_existent_id",
        research_findings={"tech_stack": "Python, React"},
        cv_text="mock cv text"
    )

    result = await agent.run_as_node(initial_state)

    assert isinstance(result, AgentState)
    assert result.error_messages
    assert "Item with ID 'non_existent_id' not found or is not a project experience item." in result.error_messages[0]

    # Test with an item of wrong type
    initial_state_wrong_type = AgentState(
        session_id="test_session",
        structured_cv=sample_structured_cv,
        job_description_data=sample_job_description_data,
        current_item_id="kq1",  # This is a Key Qualification item
        research_findings={"tech_stack": "Python, React"},
        cv_text="mock cv text"
    )
    result = await agent.run_as_node(initial_state_wrong_type)
    assert isinstance(result, AgentState)
    assert result.error_messages
    assert "Item with ID 'kq1' not found or is not a project experience item." in result.error_messages[0]

@pytest.mark.asyncio
async def test_projects_writer_agent_template_not_found(
    mock_llm_service, mock_template_manager, mock_settings, sample_structured_cv, sample_job_description_data
):
    """Test agent failure when template is not found."""
    mock_template_manager.get_template_by_type.return_value = None

    agent = ProjectsWriterAgent(
        llm_service=mock_llm_service,
        template_manager=mock_template_manager,
        settings=mock_settings,
        session_id="test_session",
    )

    initial_state = AgentState(
        session_id="test_session",
        structured_cv=sample_structured_cv,
        job_description_data=sample_job_description_data,
        current_item_id=str(PROJ1_ID),
        research_findings={"tech_stack": "Python, React"},
        cv_text="mock cv text"
    )

    result = await agent.run_as_node(initial_state)

    assert isinstance(result, AgentState)
    assert result.error_messages
    assert f"No prompt template found for type {ContentType.PROJECT}" in result.error_messages[0]
