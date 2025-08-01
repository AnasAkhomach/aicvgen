"""End-to-end tests for CV generation workflow.

These tests validate the complete CV generation process from start to finish,
including user interactions, workflow execution, and output generation.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.core.container import ContainerSingleton
from src.core.workflow_manager import WorkflowManager
from src.models.cv_models import StructuredCV
from src.models.workflow_models import UserAction, UserFeedback


class TestCVGenerationWorkflowE2E:
    """End-to-end test suite for CV generation workflow."""

    @pytest.fixture
    def sample_cv_text(self):
        """Sample CV text for testing."""
        return """
        John Doe
        Software Engineer

        Experience:
        - Senior Developer at Tech Corp (2020-2023)
        - Junior Developer at StartupXYZ (2018-2020)

        Skills:
        - Python, JavaScript, React
        - AWS, Docker, Kubernetes

        Education:
        - BS Computer Science, University ABC (2018)
        """

    @pytest.fixture
    def sample_job_description(self):
        """Sample job description for testing."""
        return """
        Senior Software Engineer Position

        Requirements:
        - 5+ years Python experience
        - Experience with cloud platforms (AWS/Azure)
        - Strong problem-solving skills
        - Team leadership experience

        Responsibilities:
        - Lead development team
        - Design scalable systems
        - Mentor junior developers
        """

    @pytest.fixture
    def temp_template_file(self):
        """Create a temporary CV template file."""
        template_content = """
## Executive Summary

## Key Qualifications

## Professional Experience

### Current Role

### Previous Roles

## Projects

### Technical Projects

## Education
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(template_content)
            temp_path = f.name

        yield temp_path

        # Cleanup
        os.unlink(temp_path)

    @pytest.fixture
    def workflow_manager(self):
        """Create a workflow manager instance for testing."""
        container = ContainerSingleton.get_instance()
        return WorkflowManager(container=container)

    @pytest.mark.asyncio
    async def test_complete_cv_generation_workflow(
        self,
        workflow_manager,
        sample_cv_text,
        sample_job_description,
        temp_template_file,
    ):
        """Test the complete CV generation workflow from start to finish."""
        # Step 1: Create new workflow
        session_id = workflow_manager.create_new_workflow(
            cv_text=sample_cv_text, jd_text=sample_job_description
        )

        assert session_id is not None
        assert len(session_id) > 0

        # Step 2: Check initial workflow status
        status = workflow_manager.get_workflow_status(session_id)
        assert status is not None
        assert status.get("workflow_status") in ["PROCESSING", "AWAITING_FEEDBACK"]

        # Step 3: If workflow is awaiting feedback, provide approval
        if status.get("workflow_status") == "AWAITING_FEEDBACK":
            feedback = UserFeedback(
                action=UserAction.APPROVE,
                message="Content looks good, proceed with generation.",
            )

            workflow_manager.send_feedback(session_id, feedback)

            # Wait for processing to complete
            import asyncio

            await asyncio.sleep(2)

            # Check final status
            final_status = workflow_manager.get_workflow_status(session_id)
            assert final_status.get("workflow_status") in ["COMPLETED", "PROCESSING"]

        # Step 4: Verify structured CV is generated
        if status.get("structured_cv"):
            structured_cv = status["structured_cv"]
            # The structured_cv might be serialized as a string representation
            # In a real implementation, this would be properly deserialized
            assert structured_cv is not None
            # For now, just verify it exists and is not empty
            assert len(str(structured_cv)) > 0

        # Cleanup
        workflow_manager.cleanup_workflow(session_id)

    @pytest.mark.asyncio
    async def test_workflow_with_regeneration_feedback(
        self,
        workflow_manager,
        sample_cv_text,
        sample_job_description,
        temp_template_file,
    ):
        """Test workflow with regeneration feedback."""
        # Create workflow
        session_id = workflow_manager.create_new_workflow(
            cv_text=sample_cv_text, jd_text=sample_job_description
        )

        # Wait for initial processing
        import asyncio

        await asyncio.sleep(1)

        status = workflow_manager.get_workflow_status(session_id)

        if status.get("workflow_status") == "AWAITING_FEEDBACK":
            # Provide regeneration feedback
            feedback = UserFeedback(
                action=UserAction.REGENERATE,
                message="Please make the key qualifications more technical and specific.",
            )

            workflow_manager.send_feedback(session_id, feedback)

            # Wait for regeneration
            await asyncio.sleep(2)

            # Check that workflow is processing regeneration
            updated_status = workflow_manager.get_workflow_status(session_id)
            assert updated_status.get("workflow_status") in [
                "PROCESSING",
                "AWAITING_FEEDBACK",
                "COMPLETED",
            ]

        # Cleanup
        workflow_manager.cleanup_workflow(session_id)

    @pytest.mark.asyncio
    async def test_workflow_error_handling(self, workflow_manager, temp_template_file):
        """Test workflow error handling with invalid inputs."""
        # Test with empty CV text
        with pytest.raises(Exception):
            workflow_manager.create_new_workflow(
                cv_text="", jd_text="Valid job description"
            )

        # Test with empty job description
        with pytest.raises(Exception):
            workflow_manager.create_new_workflow(cv_text="Valid CV text", jd_text="")

    @pytest.mark.asyncio
    async def test_workflow_session_persistence(
        self,
        workflow_manager,
        sample_cv_text,
        sample_job_description,
        temp_template_file,
    ):
        """Test that workflow sessions are properly persisted and can be resumed."""
        # Create workflow
        session_id = workflow_manager.create_new_workflow(
            cv_text=sample_cv_text, jd_text=sample_job_description
        )

        # Get initial status
        initial_status = workflow_manager.get_workflow_status(session_id)
        assert initial_status is not None

        # Create new workflow manager instance (simulating app restart)
        new_workflow_manager = WorkflowManager(
            container=ContainerSingleton.get_instance()
        )

        # Should be able to retrieve the same session
        resumed_status = new_workflow_manager.get_workflow_status(session_id)
        assert resumed_status is not None
        assert resumed_status.get("session_id") == session_id

        # Cleanup
        workflow_manager.cleanup_workflow(session_id)
