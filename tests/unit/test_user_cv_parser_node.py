"""Unit tests for user_cv_parser_node."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from src.orchestration.nodes.parsing_nodes import user_cv_parser_node
from src.models.cv_models import StructuredCV, Section, Item
from src.orchestration.state import create_global_state


class TestUserCVParserNode:
    """Test user_cv_parser_node functionality."""

    @pytest.fixture
    def mock_agent(self):
        """Create a mock UserCVParserAgent."""
        agent = AsyncMock()

        # Mock structured CV with sections
        mock_section = Section(name="Professional Experience", items=[])
        mock_structured_cv = StructuredCV(sections=[mock_section])

        agent.run.return_value = mock_structured_cv
        return agent

    @pytest.mark.asyncio
    async def test_user_cv_parser_node_success(self, mock_agent):
        """Test user_cv_parser_node with successful CV parsing."""
        # Arrange
        cv_text = (
            "John Doe\nSoftware Engineer\nExperience: 5 years in Python development"
        )

        state = create_global_state(session_id="test-session", cv_text=cv_text)

        # Act
        result = await user_cv_parser_node(state, agent=mock_agent)

        # Assert
        mock_agent.run.assert_called_once_with(cv_text)
        assert result["last_executed_node"] == "USER_CV_PARSER"
        assert "structured_cv" in result
        assert result["structured_cv"] is not None
        assert len(result["structured_cv"].sections) == 1
        assert result["structured_cv"].sections[0].name == "Professional Experience"

    @pytest.mark.asyncio
    async def test_user_cv_parser_node_with_empty_cv_text(self, mock_agent):
        """Test user_cv_parser_node with empty CV text."""
        # Arrange
        state = create_global_state(session_id="test-session", cv_text="")

        # Act
        result = await user_cv_parser_node(state, agent=mock_agent)

        # Assert
        mock_agent.run.assert_called_once_with("")
        assert result["last_executed_node"] == "USER_CV_PARSER"
        assert "structured_cv" in result

    @pytest.mark.asyncio
    async def test_user_cv_parser_node_with_none_cv_text(self, mock_agent):
        """Test user_cv_parser_node when cv_text is None."""
        # Arrange
        state = create_global_state(session_id="test-session", cv_text=None)

        # Act
        result = await user_cv_parser_node(state, agent=mock_agent)

        # Assert
        mock_agent.run.assert_called_once_with("")
        assert result["last_executed_node"] == "USER_CV_PARSER"
        assert "structured_cv" in result

    @pytest.mark.asyncio
    async def test_user_cv_parser_node_handles_agent_error(self, mock_agent):
        """Test user_cv_parser_node handles agent execution errors gracefully."""
        # Arrange
        mock_agent.run.side_effect = Exception("CV parsing failed")

        state = create_global_state(
            session_id="test-session", cv_text="Test CV content"
        )

        # Act
        result = await user_cv_parser_node(state, agent=mock_agent)

        # Assert
        assert result["last_executed_node"] == "USER_CV_PARSER"
        assert "error_messages" in result
        assert len(result["error_messages"]) > 0
        assert "CV Parser failed: CV parsing failed" in result["error_messages"]
        # structured_cv should remain None when parsing fails
        assert result.get("structured_cv") is None

    @pytest.mark.asyncio
    async def test_user_cv_parser_node_preserves_existing_state(self, mock_agent):
        """Test user_cv_parser_node preserves existing state fields."""
        # Arrange
        cv_text = "Sample CV content"
        existing_jd_data = {"job_title": "Software Engineer"}

        state = create_global_state(session_id="test-session", cv_text=cv_text)
        state["parsed_jd"] = existing_jd_data
        state["custom_field"] = "custom_value"

        # Act
        result = await user_cv_parser_node(state, agent=mock_agent)

        # Assert
        assert result["last_executed_node"] == "USER_CV_PARSER"
        assert result["parsed_jd"] == existing_jd_data
        assert result["custom_field"] == "custom_value"
        assert result["session_id"] == "test-session"
        assert result["cv_text"] == cv_text

    @pytest.mark.asyncio
    async def test_user_cv_parser_node_with_complex_structured_cv(self, mock_agent):
        """Test user_cv_parser_node with complex structured CV output."""
        # Arrange
        cv_text = "Complex CV with multiple sections"

        # Create complex mock structured CV
        experience_items = [
            Item(content="Senior Developer at TechCorp"),
            Item(content="Junior Developer at StartupInc"),
        ]
        skills_items = [Item(content="Python"), Item(content="JavaScript")]

        mock_sections = [
            Section(name="Professional Experience", items=experience_items),
            Section(name="Technical Skills", items=skills_items),
        ]
        mock_structured_cv = StructuredCV(sections=mock_sections)

        mock_agent.run.return_value = mock_structured_cv

        state = create_global_state(session_id="test-session", cv_text=cv_text)

        # Act
        result = await user_cv_parser_node(state, agent=mock_agent)

        # Assert
        mock_agent.run.assert_called_once_with(cv_text)
        assert result["last_executed_node"] == "USER_CV_PARSER"
        assert "structured_cv" in result

        structured_cv = result["structured_cv"]
        assert len(structured_cv.sections) == 2
        assert structured_cv.sections[0].name == "Professional Experience"
        assert len(structured_cv.sections[0].items) == 2
        assert structured_cv.sections[1].name == "Technical Skills"
        assert len(structured_cv.sections[1].items) == 2
