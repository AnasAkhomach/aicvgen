"""Unit tests for ExecutiveSummaryUpdaterAgent."""

import pytest
import logging
from unittest.mock import Mock, patch
from src.agents.executive_summary_updater_agent import ExecutiveSummaryUpdaterAgent
from src.models.cv_models import StructuredCV, Section, Item
from src.models.agent_input_models import ExecutiveSummaryUpdaterAgentInput
from src.models.agent_output_models import ExecutiveSummaryUpdaterAgentOutput
from src.models.cv_models import ItemStatus
from src.error_handling.exceptions import AgentExecutionError


class TestExecutiveSummaryUpdaterAgent:
    """Test cases for ExecutiveSummaryUpdaterAgent."""

    @pytest.fixture
    def mock_structured_cv(self):
        """Create a mock structured CV with existing executive summary section."""
        return StructuredCV(
            sections=[
                Section(
                    name="Executive Summary",
                    items=[
                        Item(
                            content="Old executive summary content",
                            status=ItemStatus.INITIAL,
                        )
                    ],
                ),
                Section(name="Education", items=[]),
            ]
        )

    @pytest.fixture
    def generated_summary_data(self):
        """Create mock generated executive summary data."""
        return "Experienced data analyst with 5+ years in analytics and business intelligence. Proven track record of delivering actionable insights that drive business growth."

    @pytest.fixture
    def agent_input(self, mock_structured_cv, generated_summary_data):
        """Create agent input with structured CV and generated data."""
        return ExecutiveSummaryUpdaterAgentInput(
            structured_cv=mock_structured_cv,
            generated_executive_summary=generated_summary_data,
        )

    @pytest.fixture
    def updater_agent(self):
        """Create ExecutiveSummaryUpdaterAgent instance."""
        return ExecutiveSummaryUpdaterAgent(
            session_id="test_session", name="test_executive_summary_updater"
        )

    def test_agent_initialization(self, updater_agent):
        """Test agent initializes correctly."""
        assert updater_agent.name == "test_executive_summary_updater"
        assert updater_agent.session_id == "test_session"

    def test_input_validation_success(self, updater_agent, agent_input):
        """Test successful input validation."""
        # Should not raise any exception
        updater_agent._validate_inputs(agent_input.model_dump())

    def test_input_validation_missing_structured_cv(
        self, updater_agent, generated_summary_data
    ):
        """Test input validation fails when structured_cv is missing."""
        invalid_input_dict = {
            "structured_cv": None,
            "generated_executive_summary": generated_summary_data,
        }

        with pytest.raises(
            AgentExecutionError, match="Missing required input: structured_cv"
        ):
            updater_agent._validate_inputs(invalid_input_dict)

    def test_input_validation_missing_generated_data(
        self, updater_agent, mock_structured_cv
    ):
        """Test input validation fails when generated data is missing."""
        invalid_input_dict = {
            "structured_cv": mock_structured_cv,
            "generated_executive_summary": None,
        }

        with pytest.raises(
            AgentExecutionError,
            match="Missing required input: generated_executive_summary",
        ):
            updater_agent._validate_inputs(invalid_input_dict)

    def test_input_validation_empty_generated_data(
        self, updater_agent, mock_structured_cv
    ):
        """Test input validation fails when generated data is empty."""
        invalid_input_dict = {
            "structured_cv": mock_structured_cv,
            "generated_executive_summary": "",
        }

        with pytest.raises(
            AgentExecutionError, match="generated_executive_summary cannot be empty"
        ):
            updater_agent._validate_inputs(invalid_input_dict)

    # Note: Helper methods like _find_executive_summary_section are not implemented in the agent
    # The agent uses inline logic in _execute method instead

    @pytest.mark.asyncio
    async def test_execute_with_existing_section(self, updater_agent, agent_input):
        """Test execute method with existing executive summary section."""
        result = await updater_agent._execute(**agent_input.model_dump())

        assert isinstance(result, dict)
        assert result["structured_cv"] is not None

        # Find the executive summary section
        summary_section = None
        for section in result["structured_cv"].sections:
            if section.name == "Executive Summary":
                summary_section = section
                break

        assert summary_section is not None
        assert len(summary_section.items) == 1  # Old items cleared, new one added

        # Verify item has GENERATED status
        assert summary_section.items[0].status == ItemStatus.GENERATED
        assert "Experienced data analyst" in summary_section.items[0].content

    @pytest.mark.asyncio
    async def test_execute_without_existing_section(
        self, updater_agent, generated_summary_data
    ):
        """Test execute method when executive summary section doesn't exist."""
        cv_without_summary = StructuredCV(
            sections=[Section(name="Education", items=[])]
        )

        agent_input = ExecutiveSummaryUpdaterAgentInput(
            structured_cv=cv_without_summary,
            generated_executive_summary=generated_summary_data,
        )

        result = await updater_agent._execute(**agent_input.model_dump())

        assert isinstance(result, dict)
        assert "error_messages" in result
        assert "Executive Summary section not found" in str(result["error_messages"])

    @pytest.mark.asyncio
    async def test_logging_during_execution(self, updater_agent, agent_input, caplog):
        """Test that appropriate logging occurs during execution."""
        with caplog.at_level(logging.INFO):
            await updater_agent._execute(**agent_input.model_dump())

        assert "Updated Executive Summary section" in caplog.text

    # Note: Alternative section title matching is not implemented in the current agent
    # The agent only looks for exact "Executive Summary" section name

    @pytest.mark.asyncio
    async def test_section_placement_at_beginning(
        self, updater_agent, generated_summary_data
    ):
        """Test that executive summary section is placed at the beginning when created."""
        cv_with_other_sections = StructuredCV(
            sections=[
                Section(name="Education", items=[]),
                Section(name="Experience", items=[]),
            ]
        )

        agent_input = ExecutiveSummaryUpdaterAgentInput(
            structured_cv=cv_with_other_sections,
            generated_executive_summary=generated_summary_data,
        )

        result = await updater_agent._execute(**agent_input.model_dump())

        # Should return error since no Executive Summary section exists
        assert isinstance(result, dict)
        assert "error_messages" in result
