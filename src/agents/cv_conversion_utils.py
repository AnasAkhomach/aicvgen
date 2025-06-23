"""
cv_conversion_utils.py
Utility functions for converting LLM parsing results to StructuredCV and handling metadata.
"""

from src.models.data_models import (
    StructuredCV,
    CVParsingResult,
    JobDescriptionData,
    Section,
)
from typing import Any
import logging

logger = logging.getLogger(__name__)


def create_structured_cv_with_metadata(
    cv_text: str, job_data: JobDescriptionData
) -> StructuredCV:
    structured_cv = StructuredCV(sections=[])
    if hasattr(job_data, "model_dump"):
        structured_cv.metadata.extra["job_description"] = job_data.model_dump()
    elif isinstance(job_data, dict):
        structured_cv.metadata.extra["job_description"] = job_data
    else:
        structured_cv.metadata.extra["job_description"] = {}
    structured_cv.metadata.extra["original_cv_text"] = cv_text
    return structured_cv


def add_contact_info_to_metadata(
    structured_cv: StructuredCV, parsing_result: CVParsingResult
) -> None:
    try:
        personal_info = parsing_result.personal_info
        structured_cv.metadata.extra.update(
            {
                "name": personal_info.name,
                "email": personal_info.email,
                "phone": personal_info.phone,
                "linkedin": personal_info.linkedin,
                "github": personal_info.github,
                "location": personal_info.location,
            }
        )
    except Exception as e:
        structured_cv.metadata.extra["parsing_error"] = str(e)
        logger.error(f"Error accessing personal info: {e}")
        logger.error(f"Parsing result type: {type(parsing_result)}")
        logger.error(f"Parsing result: {parsing_result}")


def convert_sections_to_structured_cv(
    structured_cv: StructuredCV, parsing_result: CVParsingResult
) -> None:
    section_order = 0
    for parsed_section in parsing_result.sections:
        content_type = determine_section_content_type(parsed_section.name)
        section = Section(
            name=parsed_section.name, content_type=content_type, order=section_order
        )
        section_order += 1
        add_section_items(section, parsed_section, content_type)
        add_section_subsections(section, parsed_section, content_type)
        structured_cv.sections.append(section)


# Import helpers from cv_structure_utils
from src.agents.cv_structure_utils import (
    determine_section_content_type,
    add_section_items,
    add_section_subsections,
)


def convert_parsing_result_to_structured_cv(
    parsing_result: CVParsingResult,
    cv_text: str,
    job_data: JobDescriptionData,
) -> StructuredCV:
    try:
        structured_cv = create_structured_cv_with_metadata(cv_text, job_data)
        add_contact_info_to_metadata(structured_cv, parsing_result)
        convert_sections_to_structured_cv(structured_cv, parsing_result)
        return structured_cv
    except Exception as e:
        logger.error(f"Error converting parsing result to StructuredCV: {str(e)}")
        structured_cv = create_structured_cv_with_metadata(cv_text, job_data)
        structured_cv.metadata.extra["error"] = str(e)
        return structured_cv
