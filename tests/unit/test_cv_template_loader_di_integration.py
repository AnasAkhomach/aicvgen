"""Integration tests for CVTemplateLoaderService with Dependency Injection.

Tests the integration of CVTemplateLoaderService with the DI container
and WorkflowManager to ensure proper dependency injection functionality.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.core.container import get_container
from src.services.cv_template_loader_service import CVTemplateLoaderService
from src.models.cv_models import StructuredCV


class TestCVTemplateLoaderDIIntegration:
    """Test suite for CVTemplateLoaderService DI integration."""

    def test_container_provides_cv_template_loader_service(self):
        """Test that the DI container provides CVTemplateLoaderService as singleton."""
        container = get_container()

        # Get the service from container
        service1 = container.cv_template_loader_service()
        service2 = container.cv_template_loader_service()

        # Verify it's the same instance (singleton)
        assert service1 is service2
        assert isinstance(service1, CVTemplateLoaderService)

    def test_workflow_manager_receives_cv_template_loader_service(self):
        """Test that WorkflowManager receives CVTemplateLoaderService via DI."""
        container = get_container()

        # Create WorkflowManager via container (as done in the actual code)
        workflow_manager = container.workflow_manager()

        # Verify WorkflowManager has the service
        assert hasattr(workflow_manager, "cv_template_loader_service")
        assert isinstance(
            workflow_manager.cv_template_loader_service, CVTemplateLoaderService
        )

        # Verify it's the same instance as from container
        container_service = container.cv_template_loader_service()
        assert workflow_manager.cv_template_loader_service is container_service

    def test_workflow_manager_uses_injected_service(self):
        """Test that WorkflowManager uses the injected service instead of class methods."""
        # Get container and create workflow manager via container
        container = get_container()
        workflow_manager = container.workflow_manager()

        # Create a real StructuredCV instance instead of MagicMock to avoid JSON serialization issues
        test_structured_cv = StructuredCV.create_empty()

        # Mock the sessions directory and file operations to avoid file system interactions
        with tempfile.TemporaryDirectory() as temp_sessions_dir:
            workflow_manager.sessions_dir = Path(temp_sessions_dir)

            with patch.object(
                workflow_manager.cv_template_loader_service,
                "load_from_markdown",
                return_value=test_structured_cv,
            ) as mock_load:
                # Create a new workflow
                session_id = workflow_manager.create_new_workflow(
                    cv_text="Test CV content", jd_text="Test JD content"
                )

                # Verify the injected service method was called
                mock_load.assert_called_once_with(
                    "src/templates/default_cv_template.md"
                )
                assert session_id is not None

    def test_service_instance_methods_work_correctly(self):
        """Test that the refactored instance methods work correctly."""
        markdown_content = """
## Test Section
### Test Subsection
Some content here.
"""

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, encoding="utf-8"
        ) as f:
            f.write(markdown_content)
            temp_path = f.name

        try:
            # Get service from container
            container = get_container()
            service = container.cv_template_loader_service()

            # Test the instance method
            result = service.load_from_markdown(temp_path)

            # Verify the result
            assert isinstance(result, StructuredCV)
            assert len(result.sections) == 1
            assert result.sections[0].name == "Test Section"
            assert len(result.sections[0].subsections) == 1
            assert result.sections[0].subsections[0].name == "Test Subsection"

        finally:
            Path(temp_path).unlink()

    def test_multiple_workflow_managers_share_same_service(self):
        """Test that multiple WorkflowManager instances share the same service singleton."""
        container = get_container()

        # Create multiple workflow managers via container
        wm1 = container.workflow_manager()
        wm2 = container.workflow_manager()

        # Verify they share the same service instance
        assert wm1.cv_template_loader_service is wm2.cv_template_loader_service
        assert isinstance(wm1.cv_template_loader_service, CVTemplateLoaderService)
        assert isinstance(wm2.cv_template_loader_service, CVTemplateLoaderService)
