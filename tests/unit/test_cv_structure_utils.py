"""
Unit tests for cv_structure_utils.py
"""

import pytest
from src.agents.cv_structure_utils import (
    create_empty_cv_structure,
    determine_section_content_type,
    determine_item_type,
    add_section_items,
    add_section_subsections,
)
from src.models.data_models import (
    JobDescriptionData,
    Section,
    Subsection,
    ItemType,
    ItemStatus,
)


def test_create_empty_cv_structure_with_dict():
    job_data = {"skills": ["Python", "AI", "Data Science"]}
    cv = create_empty_cv_structure(job_data)
    assert cv.metadata.extra["job_description"]["skills"] == [
        "Python",
        "AI",
        "Data Science",
    ]
    assert any(s.name == "Key Qualifications" for s in cv.sections)
    kq_section = next(s for s in cv.sections if s.name == "Key Qualifications")
    assert len(kq_section.items) == 3
    assert all(i.content in ["Python", "AI", "Data Science"] for i in kq_section.items)


def test_create_empty_cv_structure_with_none():
    cv = create_empty_cv_structure(None)
    assert cv.metadata.extra["job_description"] == {}
    assert any(s.name == "Executive Summary" for s in cv.sections)


def test_determine_section_content_type():
    assert determine_section_content_type("Executive Summary") == "DYNAMIC"
    assert determine_section_content_type("Education") == "STATIC"


def test_determine_item_type():
    assert determine_item_type("Key Qualifications") == ItemType.KEY_QUALIFICATION
    assert determine_item_type("Executive Summary") == ItemType.EXECUTIVE_SUMMARY_PARA
    assert determine_item_type("Education") == ItemType.EDUCATION_ENTRY
    assert determine_item_type("Certifications") == ItemType.CERTIFICATION_ENTRY
    assert determine_item_type("Languages") == ItemType.LANGUAGE_ENTRY
    assert determine_item_type("Other Section") == ItemType.BULLET_POINT


def test_add_section_items_and_subsections():
    class DummyParsedSection:
        def __init__(self):
            self.name = "Key Qualifications"
            self.items = ["Python", "AI"]
            self.subsections = []

    section = Section(name="Key Qualifications", content_type="DYNAMIC", order=1)
    parsed_section = DummyParsedSection()
    add_section_items(section, parsed_section, "DYNAMIC")
    assert len(section.items) == 2
    assert all(i.status == ItemStatus.INITIAL for i in section.items)
    # Test add_section_subsections (no subsections)
    add_section_subsections(section, parsed_section, "DYNAMIC")
    assert len(section.subsections) == 0

    # Now with a subsection
    class DummySubsection:
        def __init__(self):
            self.name = "Subsection 1"
            self.items = ["Subitem 1", "Subitem 2"]

    parsed_section.subsections = [DummySubsection()]
    add_section_subsections(section, parsed_section, "DYNAMIC")
    assert len(section.subsections) == 1
    assert section.subsections[0].name == "Subsection 1"
    assert len(section.subsections[0].items) == 2
