"""Test for jd_parser_node serialization fix."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from src.orchestration.nodes.parsing_nodes import jd_parser_node
from src.models.cv_models import JobDescriptionData
from src.orchestration.state import create_global_state


class TestJDParserNodeSerializationFix:
    """Test jd_parser_node handles both Pydantic objects and string representations."""

    @pytest.fixture
    def mock_agent(self):
        """Create a mock JobDescriptionParserAgent."""
        agent = AsyncMock()
        agent.run.return_value = {
            "job_description_data": {
                "job_title": "Software Engineer",
                "company_name": "TechCorp",
                "skills": ["Python", "Django"],
            }
        }
        return agent

    @pytest.mark.asyncio
    async def test_jd_parser_node_with_pydantic_object(self, mock_agent):
        """Test jd_parser_node works with proper JobDescriptionData object."""
        # Arrange
        job_data = JobDescriptionData(
            raw_text="Software Engineer position at TechCorp",
            job_title="Software Engineer",
            company_name="TechCorp",
        )

        state = create_global_state(session_id="test-session", cv_text="Sample CV")
        state["job_description_data"] = job_data

        # Act
        result = await jd_parser_node(state, agent=mock_agent)

        # Assert
        mock_agent.run.assert_called_once_with(
            input_data={"raw_text": "Software Engineer position at TechCorp"}
        )
        assert result["last_executed_node"] == "JD_PARSER"
        assert "parsed_jd" in result

    @pytest.mark.asyncio
    async def test_jd_parser_node_with_string_representation(self, mock_agent):
        """Test jd_parser_node works with string representation of JobDescriptionData."""
        # Arrange - simulate string representation from serialization issue
        job_data_str = "raw_text='Software Engineer position at TechCorp' job_title='Software Engineer' company_name='TechCorp' skills=[] experience_level=None responsibilities=[] industry_terms=[] company_values=[] error=None"

        state = create_global_state(session_id="test-session", cv_text="Sample CV")
        state["job_description_data"] = job_data_str

        # Act
        result = await jd_parser_node(state, agent=mock_agent)

        # Assert
        mock_agent.run.assert_called_once_with(
            input_data={"raw_text": "Software Engineer position at TechCorp"}
        )
        assert result["last_executed_node"] == "JD_PARSER"
        assert "parsed_jd" in result

    @pytest.mark.asyncio
    async def test_jd_parser_node_with_string_no_quotes(self, mock_agent):
        """Test jd_parser_node works with string representation without quotes."""
        # Arrange - simulate different string format
        job_data_str = "raw_text=TestJobDescription job_title=Engineer"

        state = create_global_state(session_id="test-session", cv_text="Sample CV")
        state["job_description_data"] = job_data_str

        # Act
        result = await jd_parser_node(state, agent=mock_agent)

        # Assert
        mock_agent.run.assert_called_once_with(
            input_data={"raw_text": "TestJobDescription"}
        )
        assert result["last_executed_node"] == "JD_PARSER"

    @pytest.mark.asyncio
    async def test_jd_parser_node_with_fallback_string(self, mock_agent):
        """Test jd_parser_node falls back to entire string if no raw_text found."""
        # Arrange - simulate string without raw_text pattern
        job_data_str = "Some random job description text"

        state = create_global_state(session_id="test-session", cv_text="Sample CV")
        state["job_description_data"] = job_data_str

        # Act
        result = await jd_parser_node(state, agent=mock_agent)

        # Assert
        mock_agent.run.assert_called_once_with(
            input_data={"raw_text": "Some random job description text"}
        )
        assert result["last_executed_node"] == "JD_PARSER"

    @pytest.mark.asyncio
    async def test_jd_parser_node_with_none_job_data(self, mock_agent):
        """Test jd_parser_node works when job_description_data is None."""
        # Arrange
        state = create_global_state(session_id="test-session", cv_text="Sample CV")
        state["job_description_data"] = None

        # Act
        result = await jd_parser_node(state, agent=mock_agent)

        # Assert
        mock_agent.run.assert_called_once_with(input_data={"raw_text": ""})
        assert result["last_executed_node"] == "JD_PARSER"

    @pytest.mark.asyncio
    async def test_jd_parser_node_handles_agent_error(self, mock_agent):
        """Test jd_parser_node handles agent execution errors gracefully."""
        # Arrange
        mock_agent.run.side_effect = Exception("Agent failed")

        job_data = JobDescriptionData(
            raw_text="Test job description", job_title="Engineer"
        )

        state = create_global_state(session_id="test-session", cv_text="Sample CV")
        state["job_description_data"] = job_data

        # Act
        result = await jd_parser_node(state, agent=mock_agent)

        # Assert
        assert result["last_executed_node"] == "JD_PARSER"
        assert "error_messages" in result
        assert len(result["error_messages"]) > 0
        assert "JD Parser failed: Agent failed" in result["error_messages"]
