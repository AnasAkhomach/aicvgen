"""
Unit tests for cv_conversion_utils.py
"""

from src.agents.cv_conversion_utils import (
    convert_parsing_result_to_structured_cv,
    create_structured_cv_with_metadata,
    add_contact_info_to_metadata,
    convert_sections_to_structured_cv,
)
from src.models.data_models import (
    StructuredCV,
    CVParsingResult,
    JobDescriptionData,
    CVParsingPersonalInfo,
)


def test_create_structured_cv_with_metadata():
    job_data = JobDescriptionData(
        raw_text="foo",
        skills=["Python"],
        responsibilities=[],
        company_values=[],
        industry_terms=[],
        experience_level="Mid",
    )
    cv = create_structured_cv_with_metadata("CV TEXT", job_data)
    assert cv.metadata.extra["job_description"]["skills"] == ["Python"]
    assert cv.metadata.extra["original_cv_text"] == "CV TEXT"


def test_add_contact_info_to_metadata():
    cv = StructuredCV(sections=[])
    parsing_result = CVParsingResult(
        personal_info=CVParsingPersonalInfo(
            name="Jane Doe",
            email="jane@example.com",
            phone="555-1234",
            linkedin="linkedin.com/in/jane",
            github="github.com/jane",
            location="NYC",
        ),
        sections=[],
    )
    add_contact_info_to_metadata(cv, parsing_result)
    assert cv.metadata.extra["name"] == "Jane Doe"
    assert cv.metadata.extra["email"] == "jane@example.com"
    assert cv.metadata.extra["phone"] == "555-1234"
    assert cv.metadata.extra["linkedin"] == "linkedin.com/in/jane"
    assert cv.metadata.extra["github"] == "github.com/jane"
    assert cv.metadata.extra["location"] == "NYC"


def test_convert_parsing_result_to_structured_cv():
    job_data = JobDescriptionData(
        raw_text="foo",
        skills=["Python"],
        responsibilities=[],
        company_values=[],
        industry_terms=[],
        experience_level="Mid",
    )
    parsing_result = CVParsingResult(
        personal_info=CVParsingPersonalInfo(
            name="Jane Doe",
            email="jane@example.com",
            phone="555-1234",
            linkedin="linkedin.com/in/jane",
            github="github.com/jane",
            location="NYC",
        ),
        sections=[],
    )
    cv = convert_parsing_result_to_structured_cv(parsing_result, "CV TEXT", job_data)
    assert isinstance(cv, StructuredCV)
    assert cv.metadata.extra["name"] == "Jane Doe"
    assert cv.metadata.extra["job_description"]["skills"] == ["Python"]
    assert cv.metadata.extra["original_cv_text"] == "CV TEXT"
