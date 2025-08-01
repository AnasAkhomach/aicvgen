"""Unit tests for CvGenerationFacade.

This module tests the CvGenerationFacade class that encapsulates
WorkflowManager complexity for the UI layer.
"""

from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from src.core.facades.cv_generation_facade import CvGenerationFacade
from src.core.facades.cv_template_manager_facade import CVTemplateManagerFacade
from src.core.facades.cv_vector_store_facade import CVVectorStoreFacade
from src.core.managers.workflow_manager import WorkflowManager
from src.models.cv_models import JobDescriptionData, StructuredCV
from src.models.workflow_models import (
    ContentType,
    UserAction,
    UserFeedback,
    WorkflowType,
)
from src.orchestration.state import GlobalState


class TestCvGenerationFacade:
    """Test cases for CvGenerationFacade."""

    @pytest.fixture
    def mock_workflow_manager(self):
        """Create a mock WorkflowManager for testing."""
        mock_manager = Mock()
        mock_manager.create_new_workflow = Mock(return_value="test-session-123")
        mock_manager.get_workflow_status = Mock(
            return_value={"workflow_status": "INITIALIZATION"}
        )
        mock_manager.trigger_workflow_step = AsyncMock(
            return_value={"workflow_status": "PROCESSING"}
        )
        mock_manager.send_feedback = Mock(return_value=True)
        mock_manager.cleanup_workflow = Mock(return_value=True)
        return mock_manager

    @pytest.fixture
    def mock_template_facade(self):
        """Create a mock template facade."""
        mock = AsyncMock(spec=CVTemplateManagerFacade)
        mock.get_template.return_value = "test_template"
        mock.format_template.return_value = "formatted_template"
        mock.list_templates.return_value = ["template1", "template2"]
        return mock

    @pytest.fixture
    def mock_vector_store_facade(self):
        """Create a mock vector store facade."""
        mock = AsyncMock(spec=CVVectorStoreFacade)
        mock.store_content.return_value = "stored_id"
        mock.search_content.return_value = ["result1", "result2"]
        mock.find_similar_content.return_value = ["similar1", "similar2"]
        return mock

    @pytest.fixture
    def mock_user_cv_parser_agent(self):
        """Create a mock UserCVParserAgent for testing."""
        mock_agent = Mock()
        mock_structured_cv = Mock(spec=StructuredCV)
        mock_structured_cv.sections = []
        mock_agent.run.return_value = mock_structured_cv
        return mock_agent

    @pytest.fixture
    def facade(self, mock_workflow_manager, mock_user_cv_parser_agent):
        """Create a CvGenerationFacade instance for testing."""
        return CvGenerationFacade(
            workflow_manager=mock_workflow_manager,
            user_cv_parser_agent=mock_user_cv_parser_agent,
        )

    @pytest.fixture
    def facade_with_facades(
        self,
        mock_workflow_manager,
        mock_user_cv_parser_agent,
        mock_template_facade,
        mock_vector_store_facade,
    ):
        """Create a CvGenerationFacade instance with all facades."""
        return CvGenerationFacade(
            workflow_manager=mock_workflow_manager,
            user_cv_parser_agent=mock_user_cv_parser_agent,
            template_facade=mock_template_facade,
            vector_store_facade=mock_vector_store_facade,
        )

    @pytest.mark.asyncio
    async def test_generate_cv_success(self, facade, mock_workflow_manager):
        """Test successful CV generation workflow creation."""
        cv_text = "Sample CV content"
        jd_text = "Sample job description"

        session_id, initial_state = await facade.generate_cv(cv_text, jd_text)

        assert session_id == "test-session-123"
        assert initial_state["workflow_status"] == "INITIALIZATION"

        mock_workflow_manager.create_new_workflow.assert_called_once_with(
            cv_text=cv_text,
            jd_text=jd_text,
            session_id=None,
            workflow_type=WorkflowType.JOB_TAILORED_CV,
        )
        mock_workflow_manager.get_workflow_status.assert_called_once_with(
            "test-session-123"
        )

    @pytest.mark.asyncio
    async def test_generate_cv_with_custom_workflow_type(
        self, facade, mock_workflow_manager
    ):
        """Test CV generation with custom workflow type."""
        cv_text = "Sample CV content"
        jd_text = "Sample job description"
        workflow_type = WorkflowType.CV_OPTIMIZATION
        session_id = "custom-session-456"

        result_session_id, initial_state = await facade.generate_cv(
            cv_text, jd_text, workflow_type, session_id
        )

        assert result_session_id == "test-session-123"
        mock_workflow_manager.create_new_workflow.assert_called_once_with(
            cv_text=cv_text,
            jd_text=jd_text,
            session_id=session_id,
            workflow_type=workflow_type,
        )

    @pytest.mark.asyncio
    async def test_generate_cv_empty_cv_text(self, facade):
        """Test CV generation with empty CV text raises ValueError."""
        with pytest.raises(ValueError, match="CV text cannot be empty"):
            await facade.generate_cv("", "Sample job description")

    @pytest.mark.asyncio
    async def test_generate_cv_empty_jd_text(self, facade):
        """Test CV generation with empty job description raises ValueError."""
        with pytest.raises(ValueError, match="Job description text cannot be empty"):
            await facade.generate_cv("Sample CV", "")

    @pytest.mark.asyncio
    async def test_generate_cv_workflow_manager_failure(
        self, facade, mock_workflow_manager
    ):
        """Test CV generation when WorkflowManager fails."""
        mock_workflow_manager.create_new_workflow.side_effect = RuntimeError(
            "Workflow creation failed"
        )

        with pytest.raises(RuntimeError, match="Workflow creation failed"):
            await facade.generate_cv("Sample CV", "Sample JD")

    @pytest.mark.asyncio
    async def test_generate_cv_no_initial_state(self, facade, mock_workflow_manager):
        """Test CV generation when initial state retrieval fails."""
        mock_workflow_manager.get_workflow_status.return_value = None

        with pytest.raises(
            RuntimeError, match="Failed to retrieve initial workflow state"
        ):
            await facade.generate_cv("Sample CV", "Sample JD")

    @pytest.mark.asyncio
    async def test_execute_workflow_step_success(self, facade, mock_workflow_manager):
        """Test successful workflow step execution."""
        session_id = "test-session-123"
        current_state = {"workflow_status": "PROCESSING"}
        updated_state = {"workflow_status": "AWAITING_FEEDBACK"}

        mock_workflow_manager.get_workflow_status.return_value = current_state
        mock_workflow_manager.trigger_workflow_step.return_value = updated_state

        result = await facade.execute_workflow_step(session_id)

        assert result == updated_state
        mock_workflow_manager.get_workflow_status.assert_called_once_with(session_id)
        mock_workflow_manager.trigger_workflow_step.assert_called_once_with(
            session_id=session_id, agent_state=current_state
        )

    @pytest.mark.asyncio
    async def test_execute_workflow_step_no_session(
        self, facade, mock_workflow_manager
    ):
        """Test workflow step execution with non-existent session."""
        mock_workflow_manager.get_workflow_status.return_value = None

        with pytest.raises(ValueError, match="No workflow found for session"):
            await facade.execute_workflow_step("non-existent-session")

    @pytest.mark.asyncio
    async def test_execute_workflow_step_execution_failure(
        self, facade, mock_workflow_manager
    ):
        """Test workflow step execution failure."""
        session_id = "test-session-123"
        current_state = {"workflow_status": "PROCESSING"}

        mock_workflow_manager.get_workflow_status.return_value = current_state
        mock_workflow_manager.trigger_workflow_step.side_effect = RuntimeError(
            "Execution failed"
        )

        with pytest.raises(RuntimeError, match="Execution failed"):
            await facade.execute_workflow_step(session_id)

    def test_get_workflow_status_success(self, facade, mock_workflow_manager):
        """Test successful workflow status retrieval."""
        session_id = "test-session-123"
        expected_state = {"workflow_status": "PROCESSING", "trace_id": "trace-456"}

        mock_workflow_manager.get_workflow_status.return_value = expected_state

        result = facade.get_workflow_status(session_id)

        assert result == expected_state
        mock_workflow_manager.get_workflow_status.assert_called_once_with(session_id)

    def test_get_workflow_status_not_found(self, facade, mock_workflow_manager):
        """Test workflow status retrieval when session not found."""
        mock_workflow_manager.get_workflow_status.return_value = None

        result = facade.get_workflow_status("non-existent-session")

        assert result is None

    def test_get_workflow_status_exception(self, facade, mock_workflow_manager):
        """Test workflow status retrieval with exception handling."""
        mock_workflow_manager.get_workflow_status.side_effect = Exception(
            "Database error"
        )

        result = facade.get_workflow_status("test-session-123")

        assert result is None

    def test_submit_user_feedback_success(self, facade, mock_workflow_manager):
        """Test successful user feedback submission."""
        session_id = "test-session-123"
        feedback = UserFeedback(
            action=UserAction.APPROVE,
            item_id="section-1",
            feedback_text="Looks good",
            rating=5,
        )

        result = facade.submit_user_feedback(session_id, feedback)

        assert result is True
        mock_workflow_manager.send_feedback.assert_called_once_with(
            session_id, feedback
        )

    def test_submit_user_feedback_failure(self, facade, mock_workflow_manager):
        """Test user feedback submission failure."""
        mock_workflow_manager.send_feedback.return_value = False

        feedback = UserFeedback(action=UserAction.REGENERATE, item_id="section-1")

        result = facade.submit_user_feedback("test-session-123", feedback)

        assert result is False

    def test_submit_user_feedback_exception(self, facade, mock_workflow_manager):
        """Test user feedback submission with exception handling."""
        mock_workflow_manager.send_feedback.side_effect = Exception("Network error")

        feedback = UserFeedback(action=UserAction.APPROVE, item_id="section-1")

        result = facade.submit_user_feedback("test-session-123", feedback)

        assert result is False

    def test_cleanup_workflow_success(self, facade, mock_workflow_manager):
        """Test successful workflow cleanup."""
        session_id = "test-session-123"

        result = facade.cleanup_workflow(session_id)

        assert result is True
        mock_workflow_manager.cleanup_workflow.assert_called_once_with(session_id)

    def test_cleanup_workflow_failure(self, facade, mock_workflow_manager):
        """Test workflow cleanup failure."""
        session_id = "test-session-123"
        mock_workflow_manager.cleanup_workflow.side_effect = Exception("Cleanup failed")

        result = facade.cleanup_workflow(session_id)

        assert result is False
        mock_workflow_manager.cleanup_workflow.assert_called_once_with(session_id)

    # Template Management Tests
    def test_get_template_success(self, facade_with_facades, mock_template_facade):
        """Test successful template retrieval."""
        template_name = "modern_cv"

        result = facade_with_facades.get_template(template_name)

        assert result == "test_template"
        mock_template_facade.get_template.assert_called_once_with(template_name, None)

    def test_get_template_no_facade(self, facade):
        """Test template retrieval without template facade."""
        result = facade.get_template("template_name")

        assert result is None

    def test_format_template_success(self, facade_with_facades, mock_template_facade):
        """Test successful template formatting."""
        template_name = "modern_cv"
        data = {"name": "John Doe", "skills": ["Python", "AI"]}
        mock_template_facade.format_template.return_value = "formatted_template"

        result = facade_with_facades.format_template(template_name, data)

        assert result == "formatted_template"
        mock_template_facade.format_template.assert_called_once_with(
            template_name, data, None
        )

    def test_format_template_no_facade(self, facade):
        """Test template formatting without template facade."""
        result = facade.format_template("template_name", {})

        assert result is None

    def test_list_templates_success(self, facade_with_facades, mock_template_facade):
        """Test successful template listing."""
        result = facade_with_facades.list_templates()

        assert result == ["template1", "template2"]
        mock_template_facade.list_templates.assert_called_once()

    def test_list_templates_no_facade(self, facade):
        """Test template listing without template facade."""
        result = facade.list_templates()

        assert result == []

    # Vector Store Tests
    @pytest.mark.asyncio
    async def test_store_content_success(
        self, facade_with_facades, mock_vector_store_facade
    ):
        """Test successful content storage."""
        content = "Test CV content"
        content_type = ContentType.CV_ANALYSIS
        metadata = {"user_id": "123"}

        result = await facade_with_facades.store_content(
            content, content_type, metadata
        )

        assert result == "stored_id"
        mock_vector_store_facade.store_content.assert_called_once_with(
            content, content_type, metadata
        )

    @pytest.mark.asyncio
    async def test_store_content_no_facade(self, facade):
        """Test content storage without vector store facade."""
        result = await facade.store_content("content", ContentType.CV_ANALYSIS, {})

        assert result is None

    @pytest.mark.asyncio
    async def test_search_content_success(
        self, facade_with_facades, mock_vector_store_facade
    ):
        """Test successful content search."""
        query = "Python developer"
        content_type = ContentType.CV_ANALYSIS
        mock_vector_store_facade.search_content = AsyncMock(
            return_value=[{"content": "match1"}, {"content": "match2"}]
        )

        result = await facade_with_facades.search_content(query, content_type, limit=5)

        assert result == [{"content": "match1"}, {"content": "match2"}]
        mock_vector_store_facade.search_content.assert_called_once_with(
            query, content_type, 5
        )

    @pytest.mark.asyncio
    async def test_search_content_no_facade(self, facade):
        """Test content search without vector store facade."""
        result = await facade.search_content("query", ContentType.CV_ANALYSIS)

        assert result == []

    @pytest.mark.asyncio
    async def test_find_similar_content_success(
        self, facade_with_facades, mock_vector_store_facade
    ):
        """Test successful similar content finding."""
        content = "Python developer with 5 years experience"
        content_type = ContentType.CV_ANALYSIS
        mock_vector_store_facade.find_similar_content = AsyncMock(
            return_value=[{"content": "similar1"}, {"content": "similar2"}]
        )

        result = await facade_with_facades.find_similar_content(
            content, content_type, limit=3
        )

        assert result == [{"content": "similar1"}, {"content": "similar2"}]
        mock_vector_store_facade.find_similar_content.assert_called_once_with(
            content, content_type, 3
        )

    @pytest.mark.asyncio
    async def test_find_similar_content_no_facade(self, facade):
        """Test similar content finding without vector store facade."""
        result = await facade.find_similar_content("content_id")

        assert result == []

    # High-level Workflow Tests
    @pytest.mark.asyncio
    @patch("src.core.facades.cv_generation_facade.create_global_state")
    async def test_generate_basic_cv_success(
        self, mock_create_state, facade, mock_workflow_manager
    ):
        """Test successful basic CV generation."""
        personal_info = {"name": "John Doe", "email": "john@example.com"}
        experience = [{"company": "Tech Corp", "role": "Developer"}]
        education = [{"degree": "BS Computer Science", "school": "University"}]
        mock_state = Mock(spec=GlobalState)
        mock_create_state.return_value = mock_state
        mock_workflow_manager.execute_workflow = AsyncMock(
            return_value={"status": "completed"}
        )

        result = await facade.generate_basic_cv(personal_info, experience, education)

        assert result == {"status": "completed"}
        mock_workflow_manager.execute_workflow.assert_called_once_with(
            WorkflowType.BASIC_CV_GENERATION, mock_state, None
        )

    @pytest.mark.asyncio
    @patch("src.core.facades.cv_generation_facade.create_global_state")
    async def test_generate_job_tailored_cv_success(
        self, mock_create_state, facade, mock_workflow_manager
    ):
        """Test successful job-tailored CV generation."""
        personal_info = {"name": "John Doe", "email": "john@example.com"}
        experience = [{"company": "Tech Corp", "role": "Developer"}]
        job_description = "Python Developer role with 3+ years experience"
        mock_state = Mock(spec=GlobalState)
        mock_create_state.return_value = mock_state
        mock_workflow_manager.execute_workflow = AsyncMock(
            return_value={"status": "completed"}
        )

        result = await facade.generate_job_tailored_cv(
            personal_info, experience, job_description
        )

        assert result == {"status": "completed"}
        mock_workflow_manager.execute_workflow.assert_called_once_with(
            WorkflowType.JOB_TAILORED_CV, mock_state, None
        )

    @pytest.mark.asyncio
    @patch("src.core.facades.cv_generation_facade.create_global_state")
    async def test_optimize_cv_success(
        self, mock_create_state, facade, mock_workflow_manager
    ):
        """Test successful CV optimization."""
        existing_cv = {"personal_info": {"name": "John Doe"}, "experience": []}
        mock_state = Mock(spec=GlobalState)
        mock_create_state.return_value = mock_state
        mock_workflow_manager.execute_workflow = AsyncMock(
            return_value={"status": "optimized"}
        )

        result = await facade.optimize_cv(existing_cv)

        assert result == {"status": "optimized"}
        mock_workflow_manager.execute_workflow.assert_called_once_with(
            WorkflowType.CV_OPTIMIZATION, mock_state, None
        )

    @pytest.mark.asyncio
    @patch("src.core.facades.cv_generation_facade.create_global_state")
    async def test_check_cv_quality_success(
        self, mock_create_state, facade, mock_workflow_manager
    ):
        """Test successful CV quality check."""
        cv_data = {"personal_info": {"name": "John Doe"}, "experience": []}
        mock_state = Mock(spec=GlobalState)
        mock_create_state.return_value = mock_state
        mock_workflow_manager.execute_workflow = AsyncMock(
            return_value={"quality_score": 85}
        )

        result = await facade.check_cv_quality(cv_data)

        assert result == {"quality_score": 85}
        mock_workflow_manager.execute_workflow.assert_called_once_with(
            WorkflowType.QUALITY_ASSURANCE, mock_state, None
        )

    def test_cleanup_workflow_exception(self, facade, mock_workflow_manager):
        """Test workflow cleanup with exception handling."""
        mock_workflow_manager.cleanup_workflow.side_effect = Exception(
            "File system error"
        )

        result = facade.cleanup_workflow("test-session-123")

        assert result is False

    @patch("src.core.facades.cv_generation_facade.logger")
    def test_logging_integration(self, mock_logger, facade, mock_workflow_manager):
        """Test that facade properly logs operations."""
        session_id = "test-session-123"

        # Test successful status retrieval logging
        mock_workflow_manager.get_workflow_status.return_value = {
            "workflow_status": "PROCESSING"
        }
        facade.get_workflow_status(session_id)

        # Verify debug logging was called
        mock_logger.debug.assert_called()

        # Test warning logging for not found
        mock_workflow_manager.get_workflow_status.return_value = None
        facade.get_workflow_status(session_id)

        # Verify warning logging was called
        mock_logger.warning.assert_called()

    # Tests for TICKET REM-P2-01 required methods
    def test_start_cv_generation_success(
        self, facade, mock_workflow_manager, mock_user_cv_parser_agent
    ):
        """Test successful CV generation start using new method."""
        cv_content = "Sample CV content"
        job_description = "Sample job description"
        user_api_key = "test-api-key"

        result = facade.start_cv_generation(cv_content, job_description, user_api_key)

        assert result == "test-session-123"
        # Verify workflow creation is called with raw CV text (parsing now handled by workflow node)
        mock_workflow_manager.create_new_workflow.assert_called_once_with(
            cv_text=cv_content, jd_text=job_description
        )
        # Verify workflow step is triggered (via trigger_workflow_step in background thread)
        mock_workflow_manager.get_workflow_status.assert_called_once_with(
            "test-session-123"
        )

    def test_start_cv_generation_empty_cv_content(self, facade):
        """Test start_cv_generation with empty CV content."""
        with pytest.raises(ValueError, match="CV content cannot be empty"):
            facade.start_cv_generation("", "Sample job description")

    def test_start_cv_generation_empty_job_description(self, facade):
        """Test start_cv_generation with empty job description."""
        with pytest.raises(ValueError, match="Job description cannot be empty"):
            facade.start_cv_generation("Sample CV", "")

    def test_start_cv_generation_workflow_manager_failure(
        self, facade, mock_workflow_manager
    ):
        """Test start_cv_generation when WorkflowManager fails."""
        mock_workflow_manager.create_new_workflow.side_effect = RuntimeError(
            "Workflow creation failed"
        )

        with pytest.raises(RuntimeError, match="Workflow creation failed"):
            facade.start_cv_generation("Sample CV", "Sample JD")

    def test_get_workflow_state_success(self, facade, mock_workflow_manager):
        """Test successful workflow state retrieval using new method."""
        session_id = "test-session-123"
        expected_state = {"workflow_status": "PROCESSING", "trace_id": "trace-456"}

        mock_workflow_manager.get_workflow_status.return_value = expected_state

        result = facade.get_workflow_state(session_id)

        assert result == expected_state
        mock_workflow_manager.get_workflow_status.assert_called_once_with(session_id)

    def test_get_workflow_state_not_found(self, facade, mock_workflow_manager):
        """Test workflow state retrieval when session not found using new method."""
        mock_workflow_manager.get_workflow_status.return_value = None

        result = facade.get_workflow_state("non-existent-session")

        assert result is None

    def test_provide_user_feedback_success(self, facade, mock_workflow_manager):
        """Test successful user feedback provision using new method."""
        session_id = "test-session-123"
        feedback = UserFeedback(
            action=UserAction.APPROVE,
            item_id="section-1",
            feedback_text="Looks good",
            rating=5,
        )

        mock_workflow_manager.send_feedback.return_value = True

        # Should not raise any exception
        facade.provide_user_feedback(session_id, feedback)

        mock_workflow_manager.send_feedback.assert_called_once_with(
            session_id, feedback
        )

    def test_provide_user_feedback_failure(self, facade, mock_workflow_manager):
        """Test user feedback provision failure using new method."""
        mock_workflow_manager.send_feedback.return_value = False

        feedback = UserFeedback(action=UserAction.REGENERATE, item_id="section-1")

        with pytest.raises(RuntimeError, match="Failed to submit user feedback"):
            facade.provide_user_feedback("test-session-123", feedback)

    def test_provide_user_feedback_exception(self, facade, mock_workflow_manager):
        """Test user feedback provision with exception handling using new method."""
        mock_workflow_manager.send_feedback.side_effect = Exception("Network error")

        feedback = UserFeedback(action=UserAction.APPROVE, item_id="section-1")

        with pytest.raises(RuntimeError, match="Failed to provide user feedback"):
            facade.provide_user_feedback("test-session-123", feedback)
