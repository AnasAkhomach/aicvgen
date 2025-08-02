"""Unit tests for CVTemplateLoaderService.

Tests the CV template loading functionality including:
- Valid Markdown parsing
- Section and subsection extraction
- Error handling for malformed files
- Pydantic validation
"""

import pytest
import tempfile
import uuid
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

from src.services.cv_template_loader_service import CVTemplateLoaderService
from src.models.cv_models import StructuredCV, Section, Subsection


class TestCVTemplateLoaderService:
    """Test suite for CVTemplateLoaderService."""

    @pytest.fixture
    def service(self):
        """Create a CVTemplateLoaderService instance for testing."""
        return CVTemplateLoaderService()

    def test_load_from_markdown_valid_template(self):
        """Test loading a valid Markdown template with sections and subsections."""
        markdown_content = """
# CV Template

## Personal Information
Some personal info content here.

### Contact Details
Email and phone details.

### Address
Home address information.

## Experience
Work experience section.

### Software Engineer
Details about software engineering role.

### Project Manager
Details about project management role.

## Education
Education background.

### University Degree
Bachelor's degree information.
"""

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, encoding="utf-8"
        ) as f:
            f.write(markdown_content)
            temp_path = f.name

        try:
            # Load the template
            service = CVTemplateLoaderService()
            result = service.load_from_markdown(temp_path)

            # Verify it's a StructuredCV instance
            assert isinstance(result, StructuredCV)

            # Verify sections
            assert len(result.sections) == 3
            section_names = [section.name for section in result.sections]
            assert "Personal Information" in section_names
            assert "Experience" in section_names
            assert "Education" in section_names

            # Verify subsections in Personal Information
            personal_section = next(
                s for s in result.sections if s.name == "Personal Information"
            )
            assert len(personal_section.subsections) == 2
            subsection_names = [sub.name for sub in personal_section.subsections]
            assert "Contact Details" in subsection_names
            assert "Address" in subsection_names

            # Verify subsections in Experience
            experience_section = next(
                s for s in result.sections if s.name == "Experience"
            )
            assert len(experience_section.subsections) == 2
            exp_subsection_names = [sub.name for sub in experience_section.subsections]
            assert "Software Engineer" in exp_subsection_names
            assert "Project Manager" in exp_subsection_names

            # Verify subsections in Education
            education_section = next(
                s for s in result.sections if s.name == "Education"
            )
            assert len(education_section.subsections) == 1
            assert education_section.subsections[0].name == "University Degree"

            # Verify all sections have empty items lists
            for section in result.sections:
                assert section.items == []
                for subsection in section.subsections:
                    assert subsection.items == []

            # Verify metadata
            assert result.metadata.created_by == "cv_template_loader"
            assert str(temp_path) in result.metadata.source_file

        finally:
            Path(temp_path).unlink()

    def test_load_from_markdown_sections_only(self):
        """Test loading a template with only sections, no subsections."""
        markdown_content = """
## Skills
Technical skills content.

## Projects
Project portfolio content.
"""

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, encoding="utf-8"
        ) as f:
            f.write(markdown_content)
            temp_path = f.name

        try:
            service = CVTemplateLoaderService()
            result = service.load_from_markdown(temp_path)

            assert isinstance(result, StructuredCV)
            assert len(result.sections) == 2

            # Verify sections have no subsections
            for section in result.sections:
                assert len(section.subsections) == 0
                assert section.items == []

        finally:
            Path(temp_path).unlink()

    def test_load_from_markdown_file_not_found(self):
        """Test error handling when template file doesn't exist."""
        non_existent_path = "/path/that/does/not/exist.md"

        with pytest.raises(FileNotFoundError) as exc_info:
            service = CVTemplateLoaderService()
            service.load_from_markdown(non_existent_path)

        assert "Template file not found" in str(exc_info.value)
        assert non_existent_path in str(exc_info.value)

    def test_load_from_markdown_empty_file(self):
        """Test error handling for empty template files."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, encoding="utf-8"
        ) as f:
            f.write("   \n\t  \n  ")  # Only whitespace
            temp_path = f.name

        try:
            with pytest.raises(ValueError) as exc_info:
                service = CVTemplateLoaderService()
                service.load_from_markdown(temp_path)

            assert "Template file is empty" in str(exc_info.value)

        finally:
            Path(temp_path).unlink()

    def test_load_from_markdown_no_sections(self):
        """Test error handling when no valid sections are found."""
        markdown_content = """
# This is just a title

Some content without proper sections.

### This is a subsection without a parent section

More content.
"""

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, encoding="utf-8"
        ) as f:
            f.write(markdown_content)
            temp_path = f.name

        try:
            with pytest.raises(ValueError) as exc_info:
                service = CVTemplateLoaderService()
                service.load_from_markdown(temp_path)

            assert "No valid sections found" in str(exc_info.value)

        finally:
            Path(temp_path).unlink()

    def test_load_from_markdown_unicode_decode_error(self):
        """Test error handling for files with encoding issues."""
        # Create a file with invalid UTF-8 content
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".md", delete=False) as f:
            f.write(b"\xff\xfe## Invalid UTF-8 Section\n")
            temp_path = f.name

        try:
            with pytest.raises(ValueError) as exc_info:
                service = CVTemplateLoaderService()
                service.load_from_markdown(temp_path)

            assert "Failed to read template file as UTF-8" in str(exc_info.value)

        finally:
            Path(temp_path).unlink()

    def test_parse_sections_with_various_spacing(self):
        """Test section parsing with different spacing patterns."""
        markdown_content = """
##Personal Information
##  Experience
##   Education
"""

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, encoding="utf-8"
        ) as f:
            f.write(markdown_content)
            temp_path = f.name

        try:
            service = CVTemplateLoaderService()
            result = service.load_from_markdown(temp_path)

            assert len(result.sections) == 3
            section_names = [section.name for section in result.sections]
            assert "Personal Information" in section_names
            assert "Experience" in section_names
            assert "Education" in section_names

        finally:
            Path(temp_path).unlink()

    def test_parse_subsections_with_various_spacing(self):
        """Test subsection parsing with different spacing patterns."""
        markdown_content = """
## Main Section

###Contact
###  Skills
###   Projects
"""

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, encoding="utf-8"
        ) as f:
            f.write(markdown_content)
            temp_path = f.name

        try:
            service = CVTemplateLoaderService()
            result = service.load_from_markdown(temp_path)

            assert len(result.sections) == 1
            section = result.sections[0]
            assert len(section.subsections) == 3

            subsection_names = [sub.name for sub in section.subsections]
            assert "Contact" in subsection_names
            assert "Skills" in subsection_names
            assert "Projects" in subsection_names

        finally:
            Path(temp_path).unlink()

    def test_service_is_stateless(self):
        """Test that the service is stateless and can be called multiple times."""
        markdown_content = """
## Section 1
### Subsection 1

## Section 2
### Subsection 2
"""

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, encoding="utf-8"
        ) as f:
            f.write(markdown_content)
            temp_path = f.name

        try:
            # Call the service multiple times
            service = CVTemplateLoaderService()
            result1 = service.load_from_markdown(temp_path)
            result2 = service.load_from_markdown(temp_path)

            # Results should be equivalent but different instances
            assert result1 is not result2
            assert len(result1.sections) == len(result2.sections)
            assert result1.sections[0].name == result2.sections[0].name

            # IDs should be different (new instances)
            assert result1.id != result2.id
            assert result1.sections[0].id != result2.sections[0].id

        finally:
            Path(temp_path).unlink()

    @patch("src.services.cv_template_loader_service.StructuredCV")
    def test_pydantic_validation_error(self, mock_structured_cv):
        """Test handling of Pydantic validation errors."""
        # Mock StructuredCV to raise ValidationError
        from pydantic import ValidationError

        mock_structured_cv.side_effect = ValidationError.from_exception_data(
            "StructuredCV",
            [{"type": "missing", "loc": ("test_field",), "msg": "Field required"}],
        )

        markdown_content = """
## Valid Section
Content here.
"""

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, encoding="utf-8"
        ) as f:
            f.write(markdown_content)
            temp_path = f.name

        try:
            with pytest.raises(ValueError) as exc_info:
                service = CVTemplateLoaderService()
                service.load_from_markdown(temp_path)

            assert "Failed to create valid StructuredCV" in str(exc_info.value)

        finally:
            Path(temp_path).unlink()

    def test_regex_patterns(self):
        """Test the regex patterns used for parsing."""
        # Test section pattern
        service = CVTemplateLoaderService()
        section_pattern = service.SECTION_PATTERN

        # Should match
        assert section_pattern.search("## Valid Section")
        assert section_pattern.search("##Another Section")
        assert section_pattern.search("##  Spaced Section  ")

        # Should not match
        assert not section_pattern.search("# Single hash")
        assert not section_pattern.search("### Triple hash")
        assert not section_pattern.search("Text ## in middle")

        # Test subsection pattern
        subsection_pattern = service.SUBSECTION_PATTERN

        # Should match
        assert subsection_pattern.search("### Valid Subsection")
        assert subsection_pattern.search("###Another Subsection")
        assert subsection_pattern.search("###  Spaced Subsection  ")

        # Should not match
        assert not subsection_pattern.search("## Double hash")
        assert not subsection_pattern.search("#### Quad hash")
        assert not subsection_pattern.search("Text ### in middle")

    def test_metadata_creation(self):
        """Test that metadata is properly created for all objects."""
        markdown_content = """
## Test Section
### Test Subsection
"""

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, encoding="utf-8"
        ) as f:
            f.write(markdown_content)
            temp_path = f.name

        try:
            service = CVTemplateLoaderService()
            result = service.load_from_markdown(temp_path)

            # Check CV metadata
            assert result.metadata.created_by == "cv_template_loader"
            assert result.metadata.source_file == str(Path(temp_path))
            assert result.metadata.template_version == "1.0"

            # Check section metadata
            section = result.sections[0]
            assert section.metadata.created_by == "cv_template_loader"
            assert section.metadata.section_type == "template_section"

            # Check subsection metadata
            subsection = section.subsections[0]
            assert subsection.metadata.created_by == "cv_template_loader"
            assert subsection.metadata.subsection_type == "template_subsection"

        finally:
            Path(temp_path).unlink()
