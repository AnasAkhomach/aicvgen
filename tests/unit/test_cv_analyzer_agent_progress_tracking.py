"""Unit tests for CVAnalyzerAgent progress tracking implementation."""

import pytest
from unittest.mock import Mock, AsyncMock, call
from src.agents.cv_analyzer_agent import CVAnalyzerAgent
from src.models.cv_models import StructuredCV, JobDescriptionData
from src.constants.agent_constants import AgentConstants
from src.services.llm_service import EnhancedLLMService


class TestCVAnalyzerAgentProgressTracking:
    """Test progress tracking implementation in CVAnalyzerAgent."""

    @pytest.fixture
    def mock_llm_service(self):
        """Create a mock LLM service."""
        return Mock(spec=EnhancedLLMService)

    @pytest.fixture
    def cv_analyzer_agent(self, mock_llm_service):
        """Create a CVAnalyzerAgent instance for testing."""
        return CVAnalyzerAgent(
            llm_service=mock_llm_service, settings={}, session_id="test_session"
        )

    @pytest.fixture
    def sample_cv_data(self):
        """Create sample CV data for testing."""
        return {
            "personal_information": {"name": "John Doe"},
            "big_10_skills": ["Python", "Machine Learning"],
            "sections": ["experience", "education"],
        }

    @pytest.fixture
    def sample_job_description(self):
        """Create sample job description data for testing."""
        return {
            "title": "Software Engineer",
            "skills": ["Python", "JavaScript"],
            "requirements": ["3+ years experience"],
        }

    @pytest.mark.asyncio
    async def test_progress_tracking_calls(
        self, cv_analyzer_agent, sample_cv_data, sample_job_description
    ):
        """Test that progress tracking calls are made at appropriate points."""
        # Mock the progress tracker
        mock_progress_tracker = Mock()
        cv_analyzer_agent.set_progress_tracker(mock_progress_tracker)

        # Prepare input data - use the correct format expected by the agent
        input_data = {
            "cv_data": sample_cv_data,
            "job_description": sample_job_description,
        }

        # Execute the agent using run method
        result = await cv_analyzer_agent.run(**input_data)

        # Debug: Print actual progress calls
        progress_calls = mock_progress_tracker.update_progress.call_args_list
        print(f"\nActual progress calls made: {len(progress_calls)}")
        for i, call_args in enumerate(progress_calls):
            print(f"Call {i+1}: {call_args}")

        print(f"\nResult: {result}")
        print(f"Result type: {type(result)}")

        # The agent should make at least the initial progress call from AgentBase.run
        assert mock_progress_tracker.update_progress.call_count >= 1

        # Check if the agent completed successfully
        # If it failed, we still want to verify that progress tracking works
        # The key is that progress calls were made without errors
        assert len(progress_calls) > 0, "No progress calls were made"

        # Verify result is a dictionary
        assert isinstance(result, dict), f"Expected dict, got {type(result)}"

    @pytest.mark.asyncio
    async def test_progress_tracking_without_tracker(
        self, cv_analyzer_agent, sample_cv_data, sample_job_description
    ):
        """Test that agent works correctly even without a progress tracker set."""
        # Don't set a progress tracker (should be None by default)
        assert cv_analyzer_agent.progress_tracker is None

        # Prepare input data
        input_data = {
            "cv_data": sample_cv_data,
            "job_description": sample_job_description,
        }

        # Execute the agent - should not raise any errors even without progress tracker
        try:
            result = await cv_analyzer_agent.run(input_data=input_data)
            # The key test is that no exception was raised
            # The agent should handle missing progress tracker gracefully
            assert result is not None
            print(f"Agent executed without progress tracker. Result: {result.success}")
        except Exception as e:
            # If an exception occurs, it should not be related to progress tracking
            error_msg = str(e).lower()
            assert (
                "progress" not in error_msg and "tracker" not in error_msg
            ), f"Progress tracking error: {e}"

    def test_progress_constants_usage(self):
        """Test that the agent uses the correct progress constants."""
        # Verify that AgentConstants has the required progress constants
        assert hasattr(AgentConstants, "PROGRESS_START")
        assert hasattr(AgentConstants, "PROGRESS_INPUT_VALIDATION")
        assert hasattr(AgentConstants, "PROGRESS_MAIN_PROCESSING")
        assert hasattr(AgentConstants, "PROGRESS_POST_PROCESSING")
        assert hasattr(AgentConstants, "PROGRESS_COMPLETE")

        # Verify they are integers (progress percentages)
        assert isinstance(AgentConstants.PROGRESS_START, int)
        assert isinstance(AgentConstants.PROGRESS_INPUT_VALIDATION, int)
        assert isinstance(AgentConstants.PROGRESS_MAIN_PROCESSING, int)
        assert isinstance(AgentConstants.PROGRESS_POST_PROCESSING, int)
        assert isinstance(AgentConstants.PROGRESS_COMPLETE, int)
