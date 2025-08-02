"""Unit test for ProjectsWriterAgent."""

import pytest
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.agents.projects_writer_agent import ProjectsWriterAgent
from src.agents.agent_base import AgentBase
from src.models.agent_models import AgentResult
from src.models.data_models import StructuredCV, JobDescriptionData, ContentType
from src.models.llm_service_models import LLMServiceResponse
from src.models.cv_models import Section, Item, ItemType, ItemStatus
from src.orchestration.state import AgentState

from uuid import uuid4, UUID


@pytest.fixture
def mock_llm():
    """Fixture for a mock BaseLanguageModel."""
    mock = AsyncMock()
    mock.ainvoke = AsyncMock(return_value="mocked llm response")
    return mock


@pytest.fixture
def mock_prompt():
    """Fixture for a mock ChatPromptTemplate."""
    mock = MagicMock()
    return mock


@pytest.fixture
def mock_parser():
    """Fixture for a mock BaseOutputParser."""
    from src.models.agent_output_models import ProjectLLMOutput

    mock = MagicMock()
    mock.parse = MagicMock(
        return_value=ProjectLLMOutput(
            project_description="Mock project description",
            technologies_used=["Python", "Django"],
            achievements=["Improved performance by 50%"],
            bullet_points=[
                "Developed scalable web application",
                "Implemented CI/CD pipeline",
            ],
        )
    )
    mock.get_format_instructions.return_value = "Format as JSON"
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
            Section(
                name="Key Qualifications",
                items=[
                    Item(
                        id=QUAL1_ID,
                        content="Problem-solving expert",
                        item_type=ItemType.KEY_QUALIFICATION,
                    )
                ],
            ),
            Section(
                name="Professional Experience",
                items=[
                    Item(
                        id=EXP1_ID,
                        content="Developed scalable solutions.",
                        item_type=ItemType.EXPERIENCE_ROLE_TITLE,
                    )
                ],
            ),
            Section(
                name="Project Experience",
                items=[
                    Item(
                        id=PROJ1_ID,
                        content="Original project 1",
                        item_type=ItemType.PROJECT_DESCRIPTION_BULLET,
                    ),
                    Item(
                        id=PROJ2_ID,
                        content="Original project 2",
                        item_type=ItemType.PROJECT_DESCRIPTION_BULLET,
                    ),
                ],
            ),
        ]
    )


@pytest.fixture
def sample_job_description_data():
    """Fixture for sample JobDescriptionData."""
    return JobDescriptionData(
        job_title="Full Stack Developer",
        company_name="InnovateTech",
        raw_text="Seeking developer with strong project delivery.",
    )


@pytest.mark.asyncio
async def test_projects_writer_agent_success(
    mock_llm, mock_prompt, mock_parser, mock_settings
):
    """Test successful generation of project experience content using LCEL."""
    from src.models.agent_output_models import ProjectLLMOutput
    from unittest.mock import patch

    # Mock the LCEL chain result
    mock_chain_result = ProjectLLMOutput(
        project_description="Mock project description",
        technologies_used=["Python", "Django"],
        achievements=["Improved performance by 50%"],
        bullet_points=[
            "Developed scalable web application",
            "Implemented CI/CD pipeline",
        ],
    )

    # Create agent first
    agent = ProjectsWriterAgent(
        llm=mock_llm,
        prompt=mock_prompt,
        parser=mock_parser,
        settings=mock_settings,
        session_id="test_session",
    )

    # Mock the chain's ainvoke method directly
    with patch.object(agent.chain, "ainvoke", new_callable=AsyncMock) as mock_ainvoke:
        mock_ainvoke.return_value = mock_chain_result

        # Test the _execute method directly with required parameters
        result = await agent._execute(
            job_description="Seeking developer with strong project delivery.",
            project_item={"id": str(PROJ1_ID), "content": "Original project 1"},
            key_qualifications="Problem-solving expert",
            professional_experience="Developed scalable solutions.",
            research_findings={"tech_stack": "Python, React"},
            template_content="Mock template content",
            format_instructions="Format as JSON",
        )

        assert isinstance(result, dict)
        assert "generated_projects" in result
        assert isinstance(result["generated_projects"], ProjectLLMOutput)

        generated_data = result["generated_projects"]
        assert generated_data.project_description == "Mock project description"
        assert generated_data.technologies_used == ["Python", "Django"]
        assert generated_data.achievements == ["Improved performance by 50%"]
        assert generated_data.bullet_points == [
            "Developed scalable web application",
            "Implemented CI/CD pipeline",
        ]

        mock_ainvoke.assert_called_once()


@pytest.mark.asyncio
async def test_projects_writer_agent_missing_inputs(
    mock_llm, mock_prompt, mock_parser, mock_settings
):
    """Test agent failure when required inputs are missing."""
    agent = ProjectsWriterAgent(
        llm=mock_llm,
        prompt=mock_prompt,
        parser=mock_parser,
        settings=mock_settings,
        session_id="test_session",
    )

    # Test missing required field - should raise validation error
    with pytest.raises(Exception):  # Pydantic validation error
        await agent._execute(
            # Missing job_description
            project_item={"id": str(PROJ1_ID), "content": "Original project 1"},
            key_qualifications="Problem-solving expert",
            professional_experience="Developed scalable solutions.",
            template_content="Mock template content",
            format_instructions="Format as JSON",
        )


@pytest.mark.asyncio
async def test_projects_writer_agent_llm_failure(
    mock_llm, mock_prompt, mock_parser, mock_settings
):
    """Test agent failure when LCEL chain fails."""
    from unittest.mock import patch

    # Create agent first
    agent = ProjectsWriterAgent(
        llm=mock_llm,
        prompt=mock_prompt,
        parser=mock_parser,
        settings=mock_settings,
        session_id="test_session",
    )

    # Mock the chain's ainvoke method to raise an exception
    with patch.object(agent.chain, "ainvoke", new_callable=AsyncMock) as mock_ainvoke:
        mock_ainvoke.side_effect = Exception("LCEL chain failed")

        # Test that the _execute method propagates the exception
        with pytest.raises(Exception) as exc_info:
            await agent._execute(
                job_description="Seeking developer with strong project delivery.",
                project_item={"id": str(PROJ1_ID), "content": "Original project 1"},
                key_qualifications="Problem-solving expert",
                professional_experience="Developed scalable solutions.",
                research_findings={"tech_stack": "Python, React"},
                template_content="Mock template content",
                format_instructions="Format as JSON",
            )

        assert "LCEL chain failed" in str(exc_info.value)


# Note: The new LCEL-based ProjectsWriterAgent works differently:
# - It doesn't search for items by ID in the CV
# - It doesn't use template managers
# - It receives all required data as direct parameters
# Therefore, the item_not_found and template_not_found tests are no longer applicable.


def test_projects_writer_agent_inheritance():
    """Test that ProjectsWriterAgent properly inherits from AgentBase (REM-AGENT-003)."""
    from unittest.mock import AsyncMock, MagicMock
    from langchain_core.language_models import BaseLanguageModel
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import BaseOutputParser

    # Verify inheritance
    assert issubclass(ProjectsWriterAgent, AgentBase)

    # Create mock objects for the new constructor signature
    mock_llm = AsyncMock(spec=BaseLanguageModel)
    mock_prompt = MagicMock(spec=ChatPromptTemplate)
    mock_parser = MagicMock(spec=BaseOutputParser)
    mock_settings = {}

    agent = ProjectsWriterAgent(
        llm=mock_llm,
        prompt=mock_prompt,
        parser=mock_parser,
        settings=mock_settings,
        session_id="test_session",
    )

    assert isinstance(agent, AgentBase)

    # Verify that required AgentBase methods are available
    assert hasattr(agent, "run")
    assert hasattr(agent, "run_as_node")
    assert hasattr(agent, "_execute")
    assert hasattr(agent, "set_progress_tracker")
    assert hasattr(agent, "update_progress")

    # Verify that _execute is async
    import inspect

    assert inspect.iscoroutinefunction(agent._execute)
