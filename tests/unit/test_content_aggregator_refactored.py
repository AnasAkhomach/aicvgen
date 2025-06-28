"""Unit tests for the refactored ContentAggregator class.

This module tests the consolidated ContentAggregator that works directly with
StructuredCV instead of the deprecated ContentData model.
"""

import pytest
from unittest.mock import Mock, MagicMock
from uuid import uuid4

from src.core.content_aggregator import ContentAggregator
from src.models.data_models import (
    StructuredCV,
    Section,
    Item,
    MetadataModel,
    ItemStatus,
)


class TestContentAggregator:
    """Test cases for the ContentAggregator class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.aggregator = ContentAggregator()

    def test_initialization(self):
        """Test that ContentAggregator initializes correctly."""
        assert self.aggregator is not None
        assert hasattr(self.aggregator, "section_map")
        assert "executive_summary" in self.aggregator.section_map
        assert "experience" in self.aggregator.section_map
        assert "skills" in self.aggregator.section_map

    def test_aggregate_results_empty_list(self):
        """Test aggregation with empty task results list."""
        result = self.aggregator.aggregate_results([])

        assert isinstance(result, StructuredCV)
        assert len(result.sections) == 0

    def test_aggregate_results_with_state_manager(self):
        """Test aggregation using existing StructuredCV from state manager."""
        # Create a mock state manager with existing StructuredCV
        mock_state_manager = Mock()
        existing_cv = StructuredCV()
        existing_cv.sections.append(
            Section(name="Existing Section", items=[Item(content="Existing content")])
        )
        mock_state_manager.get_structured_cv.return_value = existing_cv

        result = self.aggregator.aggregate_results([], mock_state_manager)

        assert isinstance(result, StructuredCV)
        assert len(result.sections) == 1
        assert result.sections[0].name == "Existing Section"

    def test_process_content_writer_result(self):
        """Test processing content writer agent results."""
        task_result = {
            "agent_type": "content_writer",
            "content": {
                "content": "This is a professional summary",
                "content_type": "executive_summary",
            },
        }

        structured_cv = StructuredCV()
        success = self.aggregator._process_content_writer_result(
            task_result, structured_cv
        )

        assert success is True
        assert len(structured_cv.sections) == 1
        assert structured_cv.sections[0].name == "Executive Summary"
        assert len(structured_cv.sections[0].items) == 1
        assert (
            structured_cv.sections[0].items[0].content
            == "This is a professional summary"
        )

    def test_process_content_writer_result_string_content(self):
        """Test processing content writer results with direct string content."""
        task_result = {
            "agent_type": "content_writer",
            "content": "Experienced professional with 5+ years in Python",
        }

        structured_cv = StructuredCV()
        success = self.aggregator._process_content_writer_result(
            task_result, structured_cv
        )

        assert success is True
        assert len(structured_cv.sections) == 1
        # Should infer as executive summary due to "experienced professional"
        assert structured_cv.sections[0].name == "Executive Summary"

    def test_process_generic_result(self):
        """Test processing generic agent results."""
        task_result = {
            "agent_type": "other",
            "content": {
                "experience": [
                    "Software Engineer at TechCorp",
                    "Developer at StartupInc",
                ]
            },
        }

        structured_cv = StructuredCV()
        success = self.aggregator._process_generic_result(task_result, structured_cv)

        assert success is True
        assert len(structured_cv.sections) == 1
        assert structured_cv.sections[0].name == "Professional Experience"
        assert len(structured_cv.sections[0].items) == 2

    def test_add_content_to_section_string(self):
        """Test adding string content to a section."""
        structured_cv = StructuredCV()
        success = self.aggregator._add_content_to_section(
            structured_cv, "Technical Skills", "Python, JavaScript, React"
        )

        assert success is True
        assert len(structured_cv.sections) == 1
        assert structured_cv.sections[0].name == "Technical Skills"
        assert structured_cv.sections[0].items[0].content == "Python, JavaScript, React"

    def test_add_content_to_section_list(self):
        """Test adding list content to a section."""
        structured_cv = StructuredCV()
        content_list = ["Python", "JavaScript", "React"]
        success = self.aggregator._add_content_to_section(
            structured_cv, "Technical Skills", content_list
        )

        assert success is True
        assert len(structured_cv.sections) == 1
        assert len(structured_cv.sections[0].items) == 3
        assert structured_cv.sections[0].items[0].content == "Python"

    def test_add_content_to_section_dict(self):
        """Test adding dictionary content to a section."""
        structured_cv = StructuredCV()
        content_dict = {
            "primary_skills": "Python, Django, FastAPI",
            "secondary_skills": "JavaScript, React, Vue.js",
        }
        success = self.aggregator._add_content_to_section(
            structured_cv, "Technical Skills", content_dict
        )

        assert success is True
        assert len(structured_cv.sections) == 1
        assert len(structured_cv.sections[0].items) == 2

    def test_find_or_create_section_existing(self):
        """Test finding an existing section."""
        structured_cv = StructuredCV()
        existing_section = Section(name="Technical Skills")
        structured_cv.sections.append(existing_section)

        found_section = self.aggregator._find_or_create_section(
            structured_cv, "technical skills"  # Case insensitive
        )

        assert found_section is existing_section
        assert len(structured_cv.sections) == 1  # No new section created

    def test_find_or_create_section_new(self):
        """Test creating a new section."""
        structured_cv = StructuredCV()

        new_section = self.aggregator._find_or_create_section(structured_cv, "Projects")

        assert new_section.name == "Projects"
        assert len(structured_cv.sections) == 1
        assert structured_cv.sections[0] is new_section

    def test_infer_and_assign_content_summary(self):
        """Test content inference for summary content."""
        structured_cv = StructuredCV()
        content = (
            "Experienced professional with strong background in software development"
        )

        success = self.aggregator._infer_and_assign_content(content, structured_cv)

        assert success is True
        assert len(structured_cv.sections) == 1
        assert structured_cv.sections[0].name == "Executive Summary"

    def test_infer_and_assign_content_experience(self):
        """Test content inference for experience content."""
        structured_cv = StructuredCV()
        content = "Worked as Senior Software Engineer developing scalable applications"

        success = self.aggregator._infer_and_assign_content(content, structured_cv)

        assert success is True
        assert len(structured_cv.sections) == 1
        assert structured_cv.sections[0].name == "Professional Experience"

    def test_infer_and_assign_content_skills(self):
        """Test content inference for skills content."""
        structured_cv = StructuredCV()
        content = "Proficient in Python, Java, and various database technologies"

        success = self.aggregator._infer_and_assign_content(content, structured_cv)

        assert success is True
        assert len(structured_cv.sections) == 1
        assert structured_cv.sections[0].name == "Technical Skills"

    def test_populate_big_10_skills(self):
        """Test populating Big 10 skills from state manager."""
        # Create mock state manager with Big 10 skills
        mock_state_manager = Mock()
        state_cv = StructuredCV()
        state_cv.big_10_skills = [
            "Python",
            "Machine Learning",
            "AWS",
            "Docker",
            "Kubernetes",
        ]
        mock_state_manager.get_structured_cv.return_value = state_cv

        structured_cv = StructuredCV()
        self.aggregator._populate_big_10_skills(structured_cv, mock_state_manager)

        assert structured_cv.big_10_skills == [
            "Python",
            "Machine Learning",
            "AWS",
            "Docker",
            "Kubernetes",
        ]

        # Check that Key Qualifications section was created
        qualifications_section = None
        for section in structured_cv.sections:
            if section.name == "Key Qualifications":
                qualifications_section = section
                break

        assert qualifications_section is not None
        assert len(qualifications_section.items) == 5
        assert qualifications_section.items[0].content == "• Python"

    def test_validate_structured_cv_valid(self):
        """Test validation of valid StructuredCV."""
        structured_cv = StructuredCV()
        section = Section(name="Professional Experience")
        section.items.append(Item(content="Software Engineer at TechCorp"))
        structured_cv.sections.append(section)

        assert self.aggregator.validate_structured_cv(structured_cv) is True

    def test_validate_structured_cv_empty(self):
        """Test validation of empty StructuredCV."""
        structured_cv = StructuredCV()

        assert self.aggregator.validate_structured_cv(structured_cv) is False

    def test_validate_structured_cv_no_content(self):
        """Test validation of StructuredCV with sections but no content."""
        structured_cv = StructuredCV()
        section = Section(name="Professional Experience")
        section.items.append(Item(content=""))  # Empty content
        structured_cv.sections.append(section)

        assert self.aggregator.validate_structured_cv(structured_cv) is False

    def test_aggregate_results_integration(self):
        """Integration test for complete aggregation workflow."""
        task_results = [
            {
                "agent_type": "content_writer",
                "content": {
                    "content": "Experienced software engineer with expertise in full-stack development",
                    "content_type": "executive_summary",
                },
            },
            {
                "agent_type": "content_writer",
                "content": {
                    "content": [
                        "Software Engineer at TechCorp (2020-2023)",
                        "Developer at StartupInc (2018-2020)",
                    ],
                    "content_type": "experience",
                },
            },
            {
                "agent_type": "other",
                "content": {"skills": "Python, JavaScript, React, Node.js, PostgreSQL"},
            },
        ]

        # Mock state manager with Big 10 skills
        mock_state_manager = Mock()
        state_cv = StructuredCV()
        state_cv.big_10_skills = ["Python", "JavaScript", "React"]
        mock_state_manager.get_structured_cv.return_value = state_cv

        result = self.aggregator.aggregate_results(task_results, mock_state_manager)

        assert isinstance(result, StructuredCV)
        assert (
            len(result.sections) >= 3
        )  # Executive Summary, Experience, Skills, Key Qualifications

        # Verify content was properly aggregated
        section_names = [section.name for section in result.sections]
        assert "Executive Summary" in section_names
        assert "Professional Experience" in section_names
        assert "Technical Skills" in section_names
        assert "Key Qualifications" in section_names

        # Verify Big 10 skills were populated
        assert result.big_10_skills == ["Python", "JavaScript", "React"]
