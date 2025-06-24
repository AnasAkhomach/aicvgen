import pytest
from src.models.data_models import (
    StructuredCV,
    JobDescriptionData,
    ItemStatus,
    ItemType,
    Section,
    Subsection,
)


def test_create_empty_cv_structure():
    """
    Tests the creation of an empty CV structure.
    """
    job_data = JobDescriptionData(
        raw_text="Software Engineer job description",
        skills=["Python", "FastAPI", "Docker"],
    )

    structured_cv = StructuredCV.create_empty(job_data)

    assert isinstance(structured_cv, StructuredCV)
    assert structured_cv.metadata.extra["start_from_scratch"] is True
    assert structured_cv.metadata.extra["job_description"] == job_data.model_dump()

    section_names = [section.name for section in structured_cv.sections]
    expected_sections = [
        "Executive Summary",
        "Key Qualifications",
        "Professional Experience",
        "Project Experience",
        "Education",
        "Certifications",
        "Languages",
    ]
    assert section_names == expected_sections

    # Check Executive Summary
    executive_summary_section = structured_cv.get_section_by_name("Executive Summary")
    assert executive_summary_section is not None
    assert len(executive_summary_section.items) == 1
    assert (
        executive_summary_section.items[0].item_type == ItemType.EXECUTIVE_SUMMARY_PARA
    )
    assert executive_summary_section.items[0].status == ItemStatus.TO_REGENERATE

    # Check Key Qualifications
    key_qualifications_section = structured_cv.get_section_by_name("Key Qualifications")
    assert key_qualifications_section is not None
    assert len(key_qualifications_section.items) == 3
    for item in key_qualifications_section.items:
        assert item.item_type == ItemType.KEY_QUALIFICATION
        assert item.status == ItemStatus.TO_REGENERATE

    # Check Professional Experience
    professional_experience_section = structured_cv.get_section_by_name(
        "Professional Experience"
    )
    assert professional_experience_section is not None
    assert len(professional_experience_section.subsections) == 1
    subsection = professional_experience_section.subsections[0]
    assert subsection.name == "Position Title at Company Name"
    assert len(subsection.items) == 3
    for item in subsection.items:
        assert item.item_type == ItemType.BULLET_POINT
        assert item.status == ItemStatus.TO_REGENERATE

    # Check Project Experience
    project_experience_section = structured_cv.get_section_by_name("Project Experience")
    assert project_experience_section is not None
    assert len(project_experience_section.subsections) == 1
    subsection = project_experience_section.subsections[0]
    assert subsection.name == "Project Name"
    assert len(subsection.items) == 2
    for item in subsection.items:
        assert item.item_type == ItemType.BULLET_POINT
        assert item.status == ItemStatus.TO_REGENERATE
