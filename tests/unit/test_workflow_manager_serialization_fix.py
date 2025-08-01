"""Tests for workflow_manager serialization fix.

This test verifies that JobDescriptionData objects are properly serialized
and deserialized when saving/loading workflow state.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from src.core.managers.workflow_manager import (
    WorkflowManager,
    _serialize_for_json,
    _deserialize_from_json,
)
from src.models.cv_models import JobDescriptionData
from src.orchestration.state import GlobalState


class TestWorkflowManagerSerializationFix:
    """Test cases for workflow manager serialization fix."""

    @pytest.fixture
    def workflow_manager(self):
        """Create a WorkflowManager instance for testing."""
        from unittest.mock import Mock

        # Mock the required dependencies
        mock_cv_template_loader = Mock()
        mock_session_manager = Mock()
        mock_container = Mock()

        return WorkflowManager(
            cv_template_loader_service=mock_cv_template_loader,
            session_manager=mock_session_manager,
            container=mock_container,
        )

    @pytest.fixture
    def sample_state_with_job_data(self):
        """Create a sample GlobalState with JobDescriptionData for testing."""
        job_data = JobDescriptionData(
            raw_text="Software Engineer position at TechCorp",
            job_title="Software Engineer",
            company_name="TechCorp",
            skills=["Python", "Django", "PostgreSQL"],
        )

        # Create a GlobalState as a dictionary since it's a TypedDict
        state = {
            "session_id": "test-session",
            "trace_id": "test-trace",
            "structured_cv": None,
            "job_description_data": job_data,
            "cv_text": "Sample CV content",
            "current_section_key": None,
            "current_section_index": None,
        }
        return state

    def test_serialize_for_json_handles_job_description_data(
        self, workflow_manager, sample_state_with_job_data
    ):
        """Test that _serialize_for_json properly handles JobDescriptionData objects."""
        # Test serialization
        serialized = _serialize_for_json(sample_state_with_job_data)

        # Assert
        assert isinstance(serialized, dict)
        assert "job_description_data" in serialized

        # JobDescriptionData should be serialized as a dict with __pydantic_model__ marker
        job_data_serialized = serialized["job_description_data"]
        assert isinstance(job_data_serialized, dict)
        assert (
            job_data_serialized["__pydantic_model__"]
            == "src.models.cv_models.JobDescriptionData"
        )
        assert "data" in job_data_serialized

        data = job_data_serialized["data"]
        assert data["raw_text"] == "Software Engineer position at TechCorp"
        assert data["job_title"] == "Software Engineer"
        assert data["company_name"] == "TechCorp"
        assert data["skills"] == ["Python", "Django", "PostgreSQL"]

    def test_deserialize_from_json_reconstructs_job_description_data(
        self, workflow_manager
    ):
        """Test that _deserialize_from_json properly reconstructs JobDescriptionData objects."""
        # Arrange - simulate serialized data
        serialized_data = {
            "session_id": "test-session",
            "job_description_data": {
                "__pydantic_model__": "src.models.cv_models.JobDescriptionData",
                "data": {
                    "raw_text": "Software Engineer position at TechCorp",
                    "job_title": "Software Engineer",
                    "company_name": "TechCorp",
                    "skills": ["Python", "Django", "PostgreSQL"],
                },
            },
            "cv_text": "Sample CV content",
        }

        # Act
        deserialized = _deserialize_from_json(serialized_data)

        # Assert
        assert isinstance(deserialized, dict)
        assert "job_description_data" in deserialized

        # JobDescriptionData should be reconstructed as proper object
        job_data = deserialized["job_description_data"]
        assert isinstance(job_data, JobDescriptionData)
        assert job_data.raw_text == "Software Engineer position at TechCorp"
        assert job_data.job_title == "Software Engineer"
        assert job_data.company_name == "TechCorp"
        assert job_data.skills == ["Python", "Django", "PostgreSQL"]

    def test_round_trip_serialization_preserves_job_description_data(
        self, workflow_manager, sample_state_with_job_data
    ):
        """Test that serialization and deserialization preserves JobDescriptionData correctly."""
        # Act - serialize then deserialize
        serialized = _serialize_for_json(sample_state_with_job_data)
        deserialized = _deserialize_from_json(serialized)

        # Assert
        original_job_data = sample_state_with_job_data["job_description_data"]
        restored_job_data = deserialized["job_description_data"]

        assert isinstance(restored_job_data, JobDescriptionData)
        assert restored_job_data.raw_text == original_job_data.raw_text
        assert restored_job_data.job_title == original_job_data.job_title
        assert restored_job_data.company_name == original_job_data.company_name
        assert restored_job_data.skills == original_job_data.skills

    def test_save_and_load_workflow_state_integration(
        self, workflow_manager, sample_state_with_job_data
    ):
        """Test integration of save and load workflow state with JobDescriptionData."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Arrange
            workflow_manager.sessions_dir = Path(temp_dir)
            session_id = "test-session"

            # Act - save state
            workflow_manager._save_state(session_id, sample_state_with_job_data)

            # Act - load state
            loaded_status = workflow_manager.get_workflow_status(session_id)

            # Assert
            assert loaded_status is not None
            assert "job_description_data" in loaded_status

            # JobDescriptionData should be properly reconstructed
            job_data = loaded_status["job_description_data"]
            assert isinstance(job_data, JobDescriptionData)
            assert job_data.raw_text == "Software Engineer position at TechCorp"
            assert job_data.job_title == "Software Engineer"
