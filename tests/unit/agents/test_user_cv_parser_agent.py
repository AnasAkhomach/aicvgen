"""Unit tests for UserCVParserAgent."""

from unittest.mock import AsyncMock, Mock
import pytest

import sys
import os

# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.agents.user_cv_parser_agent import UserCVParserAgent
from src.error_handling.exceptions import AgentExecutionError
from src.models.cv_models import StructuredCV
from src.services.llm_cv_parser_service import LLMCVParserService
from src.constants.agent_constants import AgentConstants


class TestUserCVParserAgent:
    """Test suite for UserCVParserAgent."""

    @pytest.fixture
    def mock_llm_cv_parser_service(self):
        """Create a mock LLMCVParserService."""
        service = AsyncMock()
        return service

    @pytest.fixture
    def mock_structured_cv(self):
        """Create a mock StructuredCV for testing."""
        mock_cv = Mock(spec=StructuredCV)
        mock_cv.sections = []
        return mock_cv

    @pytest.fixture
    def user_cv_parser_agent(self, mock_llm_cv_parser_service):
        """Create a UserCVParserAgent instance for testing."""
        return UserCVParserAgent(
            parser_service=mock_llm_cv_parser_service, session_id="test_session"
        )

    @pytest.fixture
    def sample_cv_text(self):
        """Sample CV text for testing."""
        return """
        John Doe
        Software Engineer

        Experience:
        - Senior Developer at Tech Corp (2020-2023)
        - Junior Developer at StartupCo (2018-2020)

        Education:
        - BS Computer Science, University of Tech (2018)

        Skills:
        - Python, JavaScript, React
        """

    def test_agent_initialization(self, user_cv_parser_agent):
        """Test that the agent initializes correctly."""
        assert user_cv_parser_agent.name == "UserCVParserAgent"
        assert user_cv_parser_agent.session_id == "test_session"
        assert hasattr(user_cv_parser_agent, "_parser_service")

    @pytest.mark.asyncio
    async def test_run_empty_text(self, user_cv_parser_agent):
        """Test running with empty CV text."""
        with pytest.raises(AgentExecutionError, match="Cannot parse empty CV text"):
            await user_cv_parser_agent.run("")

    @pytest.mark.asyncio
    async def test_run_whitespace_only(self, user_cv_parser_agent):
        """Test running with whitespace-only CV text."""
        with pytest.raises(AgentExecutionError, match="Cannot parse empty CV text"):
            await user_cv_parser_agent.run("   \n\t   ")

    @pytest.mark.asyncio
    async def test_run_successful(
        self, user_cv_parser_agent, sample_cv_text, mock_structured_cv
    ):
        """Test successful CV parsing."""
        # Mock the LLM CV parser service
        user_cv_parser_agent._parser_service.parse_cv_to_structured_cv = AsyncMock(
            return_value=mock_structured_cv
        )

        result = await user_cv_parser_agent.run(sample_cv_text)

        # Verify the result is a StructuredCV
        assert result == mock_structured_cv
        user_cv_parser_agent._parser_service.parse_cv_to_structured_cv.assert_called_once_with(
            cv_text=sample_cv_text, session_id="test_session"
        )

    @pytest.mark.asyncio
    async def test_execute_successful(
        self, user_cv_parser_agent, sample_cv_text, mock_structured_cv
    ):
        """Test successful execution of the agent via execute method."""
        # Mock the LLM CV parser service
        user_cv_parser_agent._parser_service.parse_cv_to_structured_cv = AsyncMock(
            return_value=mock_structured_cv
        )

        result = await user_cv_parser_agent._execute(cv_text=sample_cv_text)

        # Verify the result contains a StructuredCV
        assert "structured_cv" in result
        assert result["structured_cv"] == mock_structured_cv
        user_cv_parser_agent._parser_service.parse_cv_to_structured_cv.assert_called_once_with(
            cv_text=sample_cv_text, session_id="test_session"
        )

    @pytest.mark.asyncio
    async def test_execute_missing_cv_text(self, user_cv_parser_agent):
        """Test _execute method with missing cv_text parameter."""
        with pytest.raises(
            AgentExecutionError, match="cv_text is required for parsing"
        ):
            await user_cv_parser_agent._execute()

    @pytest.mark.asyncio
    async def test_run_service_error(self, user_cv_parser_agent, sample_cv_text):
        """Test handling of service errors during parsing."""
        # Mock the service to raise an exception
        user_cv_parser_agent._parser_service.parse_cv_to_structured_cv = AsyncMock(
            side_effect=Exception("Service error")
        )

        with pytest.raises(Exception, match="Service error"):
            await user_cv_parser_agent.run(sample_cv_text)

    @pytest.mark.asyncio
    async def test_progress_tracking(
        self, user_cv_parser_agent, sample_cv_text, mock_structured_cv
    ):
        """Test that progress tracking is called during execution."""
        # Mock the LLM CV parser service
        user_cv_parser_agent._parser_service.parse_cv_to_structured_cv = AsyncMock(
            return_value=mock_structured_cv
        )

        # Mock the update_progress method
        user_cv_parser_agent.update_progress = Mock()

        await user_cv_parser_agent.run(sample_cv_text)

        # Verify progress tracking was called
        user_cv_parser_agent.update_progress.assert_any_call(
            AgentConstants.PROGRESS_MAIN_PROCESSING, "Parsing CV"
        )
        user_cv_parser_agent.update_progress.assert_any_call(
            AgentConstants.PROGRESS_COMPLETE, "Parsing completed"
        )
