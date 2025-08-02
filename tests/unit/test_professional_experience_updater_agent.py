"""Unit tests for ProfessionalExperienceUpdaterAgent."""

import pytest
from unittest.mock import Mock, patch
from src.agents.professional_experience_updater_agent import (
    ProfessionalExperienceUpdaterAgent,
)
from src.models.cv_models import StructuredCV, Section, Item
from src.models.agent_input_models import ProfessionalExperienceUpdaterAgentInput
from src.models.agent_output_models import ProfessionalExperienceUpdaterAgentOutput
from src.models.cv_models import ItemStatus
from src.error_handling.exceptions import AgentExecutionError


class TestProfessionalExperienceUpdaterAgent:
    """Test cases for ProfessionalExperienceUpdaterAgent."""

    @pytest.fixture
    def mock_structured_cv(self):
        """Create a mock structured CV with existing professional experience section."""
        return StructuredCV(
            sections=[
                Section(
                    name="Professional Experience",
                    items=[
                        Item(
                            content="Old professional experience content",
                            status=ItemStatus.INITIAL,
                        )
                    ],
                ),
                Section(name="Education", items=[]),
            ]
        )

    @pytest.fixture
    def generated_experience_data(self):
        """Create mock generated professional experience data."""
        return "Senior Data Analyst at TechCorp (2020-2023)\n• Led data analysis projects\n\nData Analyst at StartupXYZ (2018-2020)\n• Developed analytics dashboards"

    @pytest.fixture
    def agent_input(self, mock_structured_cv, generated_experience_data):
        """Create agent input with structured CV and generated data."""
        return ProfessionalExperienceUpdaterAgentInput(
            structured_cv=mock_structured_cv,
            generated_professional_experience=generated_experience_data,
        )

    @pytest.fixture
    def updater_agent(self):
        """Create ProfessionalExperienceUpdaterAgent instance."""
        return ProfessionalExperienceUpdaterAgent(
            session_id="test_session", name="test_professional_experience_updater"
        )

    def test_agent_initialization(self, updater_agent):
        """Test agent initializes correctly."""
        assert updater_agent.name == "test_professional_experience_updater"
        assert updater_agent.session_id == "test_session"

    def test_input_validation_success(self, updater_agent, agent_input):
        """Test successful input validation."""
        # Should not raise any exception
        updater_agent._validate_inputs(agent_input.model_dump())

    def test_input_validation_missing_structured_cv(
        self, updater_agent, generated_experience_data
    ):
        """Test input validation fails when structured_cv is missing."""
        invalid_input = {
            "structured_cv": None,
            "generated_professional_experience": generated_experience_data,
        }

        with pytest.raises(
            AgentExecutionError, match="Missing required input: structured_cv"
        ):
            updater_agent._validate_inputs(invalid_input)

    def test_input_validation_missing_generated_data(
        self, updater_agent, mock_structured_cv
    ):
        """Test input validation fails when generated data is missing."""
        invalid_input = {
            "structured_cv": mock_structured_cv,
            "generated_professional_experience": None,
        }

        with pytest.raises(
            AgentExecutionError,
            match="Missing required input: generated_professional_experience",
        ):
            updater_agent._validate_inputs(invalid_input)

    def test_input_validation_empty_generated_data(
        self, updater_agent, mock_structured_cv
    ):
        """Test input validation fails when generated data is empty."""
        invalid_input = {
            "structured_cv": mock_structured_cv,
            "generated_professional_experience": "",
        }

        with pytest.raises(
            AgentExecutionError,
            match="generated_professional_experience cannot be empty",
        ):
            updater_agent._validate_inputs(invalid_input)

    # NOTE: The following tests are commented out because the methods they test
    # don't exist in the actual agent implementation

    # def test_find_professional_experience_section_exists(self, updater_agent, mock_structured_cv):
    #     """Test finding existing professional experience section."""
    #     section = updater_agent._find_professional_experience_section(mock_structured_cv)
    #     assert section is not None
    #     assert section.name == "Professional Experience"

    # def test_find_professional_experience_section_not_exists(self, updater_agent):
    #     """Test finding professional experience section when it doesn't exist."""
    #     cv_without_experience = StructuredCV(
    #         sections=[
    #             Section(name="Education", items=[])
    #         ]
    #     )
    #
    #     section = updater_agent._find_professional_experience_section(cv_without_experience)
    #     assert section is None

    # def test_create_professional_experience_section(self, updater_agent):
    #     """Test creating new professional experience section."""
    #     cv_without_experience = StructuredCV(sections=[])
    #
    #     section = updater_agent._create_professional_experience_section(cv_without_experience)
    #
    #     assert section.name == "Professional Experience"
    #     assert len(section.items) == 0
    #     assert len(cv_without_experience.sections) == 1
    #     assert cv_without_experience.sections[0] == section

    # def test_update_section_with_generated_data(self, updater_agent, generated_experience_data):
    #     """Test updating section with generated professional experience data."""
    #     section = Section(title="Professional Experience", items=[])
    #
    #     updater_agent._update_section_with_generated_data(section, generated_experience_data)
    #
    #     assert len(section.items) == 2
    #
    #     # Check first item
    #     assert section.items[0].content == "Senior Data Analyst at TechCorp (2020-2023)\n• Led data analysis projects"
    #     assert section.items[0].status == ItemStatus.GENERATED
    #     assert section.items[0].metadata == {"company": "TechCorp", "role": "Senior Data Analyst"}
    #
    #     # Check second item
    #     assert section.items[1].content == "Data Analyst at StartupXYZ (2018-2020)\n• Developed analytics dashboards"
    #     assert section.items[1].status == ItemStatus.GENERATED
    #     assert section.items[1].metadata == {"company": "StartupXYZ", "role": "Data Analyst"}

    @pytest.mark.asyncio
    async def test_execute_with_existing_section(
        self, updater_agent, agent_input, generated_experience_data
    ):
        """Test execute method with existing professional experience section."""
        result = await updater_agent.run(**agent_input.model_dump())

        assert "error_messages" not in result
        assert "structured_cv" in result

        # Find the professional experience section
        experience_section = None
        for section in result["structured_cv"].sections:
            if section.name == "Professional Experience":
                experience_section = section
                break

        assert experience_section is not None
        assert len(experience_section.items) == 2  # Old item + new item

        # Verify the new item has GENERATED status and correct content
        new_item = experience_section.items[-1]  # Last item should be the new one
        assert new_item.status == ItemStatus.GENERATED
        assert generated_experience_data in new_item.content

    @pytest.mark.asyncio
    async def test_execute_without_existing_section(
        self, updater_agent, generated_experience_data
    ):
        """Test execute method when professional experience section doesn't exist."""
        cv_without_experience = StructuredCV(
            sections=[Section(name="Education", items=[])]
        )

        agent_input = ProfessionalExperienceUpdaterAgentInput(
            structured_cv=cv_without_experience,
            generated_professional_experience=generated_experience_data,
        )

        result = await updater_agent.run(**agent_input.model_dump())

        # Should return an error since section doesn't exist
        assert "error_messages" in result
        assert (
            "Professional Experience section not found" in result["error_messages"][0]
        )

    @pytest.mark.asyncio
    async def test_execute_with_empty_generated_data(
        self, updater_agent, mock_structured_cv
    ):
        """Test execute method with empty generated data."""
        agent_input = ProfessionalExperienceUpdaterAgentInput(
            structured_cv=mock_structured_cv, generated_professional_experience=""
        )

        # This should fail validation and return error messages
        result = await updater_agent.run(**agent_input.model_dump())
        assert "error_messages" in result
        assert (
            "generated_professional_experience cannot be empty"
            in result["error_messages"][0]
        )

    @patch("src.agents.professional_experience_updater_agent.logger")
    @pytest.mark.asyncio
    async def test_logging_during_execution(
        self, mock_logger, updater_agent, agent_input
    ):
        """Test that appropriate logging occurs during execution."""
        await updater_agent.run(**agent_input.model_dump())

        # Verify logging calls
        mock_logger.info.assert_called()
        assert any(
            "Updated Professional Experience section with generated content"
            in str(call)
            for call in mock_logger.info.call_args_list
        )
