"""Test for content template metadata fix.

Verifies that the content_type metadata in prompt files matches
the ContentType enum members for proper template loading.
"""

import pytest
from pathlib import Path
from src.templates.content_templates import ContentTemplateManager
from src.models.workflow_models import ContentType


class TestContentTemplateMetadataFix:
    """Test that content template metadata is correctly aligned with ContentType enum."""

    def test_professional_experience_prompt_content_type(self):
        """Test that professional_experience_prompt.md has correct content_type metadata."""
        prompt_file = Path("data/prompts/professional_experience_prompt.md")
        assert prompt_file.exists(), f"Prompt file not found: {prompt_file}"

        # Read the file content
        content = prompt_file.read_text(encoding="utf-8")

        # Check that the content_type is set to EXPERIENCE
        assert (
            "content_type: EXPERIENCE" in content
        ), "content_type should be EXPERIENCE"

        # Verify EXPERIENCE is a valid ContentType enum member
        assert hasattr(
            ContentType, "EXPERIENCE"
        ), "EXPERIENCE should be a ContentType enum member"
        assert ContentType.EXPERIENCE.value == "experience"

    def test_projects_prompt_content_type(self):
        """Test that projects_prompt.md has correct content_type metadata."""
        prompt_file = Path("data/prompts/projects_prompt.md")
        assert prompt_file.exists(), f"Prompt file not found: {prompt_file}"

        # Read the file content
        content = prompt_file.read_text(encoding="utf-8")

        # Check that the content_type is set to PROJECT
        assert "content_type: PROJECT" in content, "content_type should be PROJECT"

        # Verify PROJECT is a valid ContentType enum member
        assert hasattr(
            ContentType, "PROJECT"
        ), "PROJECT should be a ContentType enum member"
        assert ContentType.PROJECT.value == "project"

    def test_content_template_manager_can_load_templates(self):
        """Test that ContentTemplateManager can load templates with corrected metadata."""
        # Initialize the template manager (templates are loaded automatically)
        template_manager = ContentTemplateManager(prompt_directory="data/prompts")

        # Verify that templates with EXPERIENCE content_type can be retrieved
        experience_templates = template_manager.get_templates_by_type(
            ContentType.EXPERIENCE
        )
        assert (
            len(experience_templates) > 0
        ), "Should find at least one EXPERIENCE template"

        # Verify that templates with PROJECT content_type can be retrieved
        project_templates = template_manager.get_templates_by_type(ContentType.PROJECT)
        assert len(project_templates) > 0, "Should find at least one PROJECT template"

    def test_template_metadata_parsing(self):
        """Test that template metadata is correctly parsed from frontmatter."""
        template_manager = ContentTemplateManager(prompt_directory="data/prompts")

        # Find the professional experience template
        experience_templates = template_manager.get_templates_by_type(
            ContentType.EXPERIENCE
        )
        professional_exp_template = None
        for template in experience_templates:
            if (
                "resume_role" in template.name
                or "professional_experience" in template.name
            ):
                professional_exp_template = template
                break

        assert (
            professional_exp_template is not None
        ), "Should find professional experience template"
        assert professional_exp_template.content_type == ContentType.EXPERIENCE

        # Find the projects template
        project_templates = template_manager.get_templates_by_type(ContentType.PROJECT)
        projects_template = None
        for template in project_templates:
            if "side_project" in template.name or "projects" in template.name:
                projects_template = template
                break

        assert projects_template is not None, "Should find projects template"
        assert projects_template.content_type == ContentType.PROJECT

    def test_no_old_content_types_remain(self):
        """Test that old content_type values are no longer present in prompt files."""
        # Check professional_experience_prompt.md
        prof_exp_file = Path("data/prompts/professional_experience_prompt.md")
        prof_exp_content = prof_exp_file.read_text(encoding="utf-8")
        assert (
            "content_type: role_generation" not in prof_exp_content
        ), "Old content_type 'role_generation' should be removed"

        # Check projects_prompt.md
        projects_file = Path("data/prompts/projects_prompt.md")
        projects_content = projects_file.read_text(encoding="utf-8")
        assert (
            "content_type: project_generation" not in projects_content
        ), "Old content_type 'project_generation' should be removed"

    def test_content_type_enum_completeness(self):
        """Test that ContentType enum contains the required values."""
        # Verify that the enum has the values we're using
        required_content_types = ["EXPERIENCE", "PROJECT"]

        for content_type_name in required_content_types:
            assert hasattr(
                ContentType, content_type_name
            ), f"ContentType enum should have {content_type_name} member"

            # Verify the enum value is a string
            enum_member = getattr(ContentType, content_type_name)
            assert isinstance(
                enum_member.value, str
            ), f"ContentType.{content_type_name}.value should be a string"
