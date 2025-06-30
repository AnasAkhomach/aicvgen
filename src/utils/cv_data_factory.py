"""
This module provides factory functions for creating and manipulating CV data structures.
"""

from typing import Any, Dict, Optional

from ..models.cv_models import (
    Item,
    ItemStatus,
    JobDescriptionData,
    StructuredCV,
    Section,
    Subsection,
    ItemType,
)
from ..models.agent_output_models import ParserAgentOutput
from ..error_handling.exceptions import DataConversionError


def create_empty_cv_structure(
    cv_text: str = "", job_data: Optional[JobDescriptionData] = None
) -> StructuredCV:
    """
    Initialize a StructuredCV object with metadata using the static method on the model.
    """
    return StructuredCV.create_empty(job_data=job_data, cv_text=cv_text)


def determine_section_content_type(section_name: str) -> str:
    """Determines if a section's content is dynamic or static (string values)."""
    dynamic_sections = [
        "executive summary",
        "key qualifications",
        "professional experience",
        "project experience",
    ]
    return "DYNAMIC" if section_name.lower() in dynamic_sections else "STATIC"


def determine_item_type(section_name: str) -> ItemType:
    """Determines the item type as an ItemType enum based on the section name."""
    section_lower = section_name.lower()
    if "qualification" in section_lower or "skill" in section_lower:
        return ItemType.KEY_QUALIFICATION
    if "executive" in section_lower or "summary" in section_lower:
        return ItemType.EXECUTIVE_SUMMARY_PARA
    if "project" in section_lower:
        return ItemType.PROJECT_DESCRIPTION_BULLET
    if "experience" in section_lower:
        return ItemType.EXPERIENCE_ROLE_TITLE
    if "education" in section_lower:
        return ItemType.EDUCATION_ENTRY
    if "certification" in section_lower:
        return ItemType.CERTIFICATION_ENTRY
    if "language" in section_lower:
        return ItemType.LANGUAGE_ENTRY
    return ItemType.BULLET_POINT


def convert_parser_output_to_structured_cv(
    parsing_result: Dict[str, Any],
    cv_text: str,
    job_data: JobDescriptionData,
) -> StructuredCV:
    """
    Converts the structured output from an LLM parsing service into a StructuredCV object.
    """
    structured_cv = create_empty_cv_structure(cv_text, job_data)

    try:
        parsed_data = ParserAgentOutput(**parsing_result)

        sections = []
        for section_data in parsed_data.sections:
            section = Section(
                id=section_data.id,
                title=section_data.title,
                content_type=determine_section_content_type(section_data.title),
                status=ItemStatus.GENERATED,
                subsections=[],
            )

            subsections = []
            for subsection_data in section_data.subsections:
                subsection = Subsection(
                    id=subsection_data.id,
                    title=subsection_data.title,
                    status=ItemStatus.GENERATED,
                    items=[],
                )

                items = []
                for item_data in subsection_data.items:
                    item = Item(
                        id=item_data.id,
                        content=item_data.content,
                        item_type=determine_item_type(section_data.title),
                        status=ItemStatus.GENERATED,
                    )
                    items.append(item)
                subsection.items = items
                subsections.append(subsection)
            section.subsections = subsections
            sections.append(section)
        structured_cv.sections = sections

    except (DataConversionError, TypeError, KeyError) as e:
        raise DataConversionError(
            f"Failed to convert parser output to StructuredCV: {e}"
        ) from e


def get_item_by_id(cv_data: StructuredCV, item_id: str) -> Optional[Dict[str, Any]]:
    """Finds an item (section, subsection, or item) in the CV by its ID."""
    for section in cv_data.sections:
        if str(section.id) == item_id:
            return section.dict()
        for subsection in section.subsections:
            if str(subsection.id) == item_id:
                return subsection.dict()
            for item in subsection.items:
                if str(item.id) == item_id:
                    return item.dict()
    return None


def update_item_by_id(
    cv_data: StructuredCV, item_id: str, new_data: Dict[str, Any]
) -> StructuredCV:
    """Updates an item in the CV by its ID."""
    for section in cv_data.sections:
        if str(section.id) == item_id:
            for key, value in new_data.items():
                setattr(section, key, value)
            return cv_data
        for subsection in section.subsections:
            if str(subsection.id) == item_id:
                for key, value in new_data.items():
                    setattr(subsection, key, value)
                return cv_data
            for item in subsection.items:
                if str(item.id) == item_id:
                    for key, value in new_data.items():
                        setattr(item, key, value)
                    return cv_data
    return cv_data
