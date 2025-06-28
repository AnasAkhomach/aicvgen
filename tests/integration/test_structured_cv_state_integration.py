"""Integration tests for StructuredCV state management.

This module tests the canonical StructuredCV data model integration
with the state management system after consolidating competing data models.
"""

import pytest
from uuid import uuid4

from src.models.data_models import (
    StructuredCV,
    Section,
    Item,
    MetadataModel,
    ItemStatus,
    JobDescriptionData,
)
from src.orchestration.state import AgentState


class TestStructuredCVStateIntegration:
    """Integration tests for StructuredCV state management."""

    def test_structured_cv_creation(self):
        """Test creating a basic StructuredCV structure."""
        cv = StructuredCV()

        assert cv.id is not None
        assert isinstance(cv.sections, list)
        assert len(cv.sections) == 0
        assert isinstance(cv.metadata, MetadataModel)
        assert isinstance(cv.big_10_skills, list)

    def test_structured_cv_with_sections(self):
        """Test StructuredCV with multiple sections and items."""
        cv = StructuredCV()

        # Add Executive Summary section
        summary_section = Section(name="Executive Summary", content_type="DYNAMIC")
        summary_section.items.append(
            Item(
                content="Experienced software engineer with 8+ years of experience",
                status=ItemStatus.GENERATED,
            )
        )
        cv.sections.append(summary_section)

        # Add Professional Experience section
        exp_section = Section(name="Professional Experience", content_type="DYNAMIC")
        exp_section.items.extend(
            [
                Item(
                    content="Senior Software Engineer at TechCorp (2020-2023)",
                    status=ItemStatus.GENERATED,
                ),
                Item(
                    content="Software Developer at StartupInc (2018-2020)",
                    status=ItemStatus.GENERATED,
                ),
            ]
        )
        cv.sections.append(exp_section)

        # Verify structure
        assert len(cv.sections) == 2
        assert cv.sections[0].name == "Executive Summary"
        assert cv.sections[1].name == "Professional Experience"
        assert len(cv.sections[1].items) == 2

    def test_structured_cv_metadata_extra(self):
        """Test StructuredCV metadata extra fields functionality."""
        cv = StructuredCV()

        # Test setting extra data
        cv.metadata.extra["personal_info"] = {
            "name": "John Doe",
            "email": "john.doe@example.com",
            "phone": "+1-555-0123",
        }
        cv.metadata.extra["original_cv_text"] = "Original CV text content here"

        # Verify extra data was stored correctly
        assert "personal_info" in cv.metadata.extra
        assert cv.metadata.extra["personal_info"]["name"] == "John Doe"
        assert cv.metadata.extra["original_cv_text"] == "Original CV text content here"

    def test_structured_cv_big_10_skills(self):
        """Test StructuredCV Big 10 skills functionality."""
        cv = StructuredCV()

        # Set Big 10 skills
        cv.big_10_skills = [
            "Python",
            "Machine Learning",
            "AWS",
            "Docker",
            "Kubernetes",
            "React",
            "Node.js",
            "PostgreSQL",
            "Git",
            "Agile Methodologies",
        ]

        assert len(cv.big_10_skills) == 10
        assert "Python" in cv.big_10_skills
        assert "Machine Learning" in cv.big_10_skills

    def test_structured_cv_find_item_by_id(self):
        """Test finding items by ID in StructuredCV."""
        cv = StructuredCV()

        # Create section with items
        section = Section(name="Professional Experience")
        item1 = Item(content="Senior Engineer role", status=ItemStatus.GENERATED)
        item2 = Item(content="Junior Developer role", status=ItemStatus.GENERATED)
        section.items.extend([item1, item2])
        cv.sections.append(section)

        # Test finding by actual item ID
        found_item, found_section, found_subsection = cv.find_item_by_id(str(item1.id))

        assert found_item is item1
        assert found_section is section
        assert found_subsection is None

    def test_structured_cv_create_empty_with_job_data(self):
        """Test creating empty StructuredCV with job data."""
        job_data = JobDescriptionData(
            raw_text="Software Engineer position at TechCorp",
            job_title="Senior Software Engineer",
            company_name="TechCorp",
            skills=["Python", "Django", "PostgreSQL"],
        )

        cv = StructuredCV.create_empty("Original CV text here", job_data)

        assert isinstance(cv, StructuredCV)
        assert cv.metadata.extra["original_cv_text"] == "Original CV text here"
        assert "job_description" in cv.metadata.extra
        assert (
            cv.metadata.extra["job_description"]["job_title"]
            == "Senior Software Engineer"
        )

    def test_structured_cv_create_empty_without_job_data(self):
        """Test creating empty StructuredCV without job data."""
        cv = StructuredCV.create_empty("Original CV text here")

        assert isinstance(cv, StructuredCV)
        assert cv.metadata.extra["original_cv_text"] == "Original CV text here"
        assert cv.metadata.extra["job_description"] == {}

    def test_agent_state_structured_cv_integration(self):
        """Test AgentState integration with StructuredCV."""
        # Create StructuredCV with content
        cv = StructuredCV()
        summary_section = Section(name="Executive Summary")
        summary_section.items.append(
            Item(
                content="Senior software engineer with expertise in Python and cloud technologies"
            )
        )
        cv.sections.append(summary_section)

        # Create JobDescriptionData
        job_data = JobDescriptionData(
            raw_text="Looking for a Python developer",
            job_title="Python Developer",
            skills=["Python", "AWS", "Docker"],
        )

        # Create AgentState with StructuredCV
        state = AgentState(
            structured_cv=cv,
            job_description_data=job_data,
            cv_text="Original CV text content",
        )

        assert state.structured_cv is cv
        assert state.job_description_data is job_data
        assert state.cv_text == "Original CV text content"
        assert state.session_id is None  # Optional field
        assert isinstance(state.trace_id, str)

    def test_structured_cv_serialization(self):
        """Test StructuredCV serialization and deserialization."""
        cv = StructuredCV()
        cv.big_10_skills = ["Python", "AWS", "Docker"]
        cv.metadata.extra["test_data"] = "test_value"

        # Add section with items
        section = Section(name="Skills")
        section.items.append(Item(content="Expert in Python programming"))
        cv.sections.append(section)

        # Test serialization
        cv_dict = cv.model_dump()

        assert isinstance(cv_dict, dict)
        assert "sections" in cv_dict
        assert "metadata" in cv_dict
        assert "big_10_skills" in cv_dict
        assert cv_dict["big_10_skills"] == ["Python", "AWS", "Docker"]

        # Test deserialization
        cv_restored = StructuredCV.model_validate(cv_dict)

        assert cv_restored.big_10_skills == cv.big_10_skills
        assert cv_restored.metadata.extra["test_data"] == "test_value"
        assert len(cv_restored.sections) == 1
        assert cv_restored.sections[0].name == "Skills"

    def test_structured_cv_state_persistence_cycle(self):
        """Test complete state persistence cycle with StructuredCV."""
        # Create initial StructuredCV
        original_cv = StructuredCV()
        original_cv.big_10_skills = ["Python", "Django", "PostgreSQL", "AWS", "Docker"]
        original_cv.metadata.extra["personal_info"] = {
            "name": "Jane Developer",
            "email": "jane@example.com",
        }

        # Add content sections
        sections_data = [
            ("Executive Summary", ["Experienced full-stack developer"]),
            (
                "Professional Experience",
                ["Senior Developer at TechCorp", "Developer at StartupInc"],
            ),
            (
                "Technical Skills",
                ["Python, Django, PostgreSQL", "AWS, Docker, Kubernetes"],
            ),
        ]

        for section_name, items_content in sections_data:
            section = Section(name=section_name)
            for content in items_content:
                section.items.append(Item(content=content, status=ItemStatus.GENERATED))
            original_cv.sections.append(section)

        # Create AgentState
        job_data = JobDescriptionData(
            raw_text="Senior Python Developer position",
            job_title="Senior Python Developer",
            skills=["Python", "Django", "AWS"],
        )

        original_state = AgentState(
            structured_cv=original_cv,
            job_description_data=job_data,
            cv_text="Original CV text",
        )

        # Simulate persistence cycle (serialize -> deserialize)
        state_dict = original_state.model_dump()
        restored_state = AgentState.model_validate(state_dict)

        # Verify complete restoration
        assert restored_state.structured_cv.big_10_skills == original_cv.big_10_skills
        assert restored_state.structured_cv.metadata.extra == original_cv.metadata.extra
        assert len(restored_state.structured_cv.sections) == len(original_cv.sections)

        # Verify section content
        for orig_section, rest_section in zip(
            original_cv.sections, restored_state.structured_cv.sections
        ):
            assert orig_section.name == rest_section.name
            assert len(orig_section.items) == len(rest_section.items)
            for orig_item, rest_item in zip(orig_section.items, rest_section.items):
                assert orig_item.content == rest_item.content
                assert orig_item.status == rest_item.status

        # Verify job description data
        assert restored_state.job_description_data.job_title == job_data.job_title
        assert restored_state.job_description_data.skills == job_data.skills
        assert restored_state.cv_text == "Original CV text"

    def test_structured_cv_data_consistency(self):
        """Test data consistency across StructuredCV operations."""
        cv = StructuredCV()

        # Test that modifications maintain consistency
        cv.big_10_skills = ["Skill1", "Skill2", "Skill3"]
        cv.metadata.extra["source"] = "test"

        # Add sections
        for i in range(3):
            section = Section(name=f"Section {i}")
            for j in range(2):
                section.items.append(
                    Item(content=f"Content {i}-{j}", status=ItemStatus.GENERATED)
                )
            cv.sections.append(section)

        # Verify data integrity
        assert len(cv.big_10_skills) == 3
        assert len(cv.sections) == 3
        assert all(len(section.items) == 2 for section in cv.sections)
        assert cv.metadata.extra["source"] == "test"

        # Test item ID uniqueness
        all_item_ids = []
        for section in cv.sections:
            for item in section.items:
                all_item_ids.append(str(item.id))

        assert len(all_item_ids) == len(set(all_item_ids))  # All IDs should be unique
