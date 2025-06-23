"""
cv_structure_utils.py
Utility functions for creating and transforming CV sections, items, and subsections.
"""

from typing import Any, Dict, List, Optional
from ..models.data_models import (
    StructuredCV,
    Section,
    Subsection,
    Item,
    ItemType,
    ItemStatus,
    JobDescriptionData,
)


def create_empty_cv_structure(job_data: Optional[JobDescriptionData]) -> StructuredCV:
    """
    Creates an empty CV structure for the "Start from Scratch" option.
    Args:
        job_data: The parsed job description data.
    Returns:
        A StructuredCV object with empty sections.
    """
    structured_cv = StructuredCV()
    # Add metadata - handle both dict and JobDescriptionData object types
    if job_data:
        if hasattr(job_data, "to_dict"):
            structured_cv.metadata.extra["job_description"] = job_data.to_dict()
        elif isinstance(job_data, dict):
            structured_cv.metadata.extra["job_description"] = job_data
        else:
            structured_cv.metadata.extra["job_description"] = {}
    else:
        structured_cv.metadata.extra["job_description"] = {}
    structured_cv.metadata.extra["start_from_scratch"] = True

    # Create standard CV sections with proper order
    sections = [
        {"name": "Executive Summary", "type": "DYNAMIC", "order": 0},
        {"name": "Key Qualifications", "type": "DYNAMIC", "order": 1},
        {"name": "Professional Experience", "type": "DYNAMIC", "order": 2},
        {"name": "Project Experience", "type": "DYNAMIC", "order": 3},
        {"name": "Education", "type": "STATIC", "order": 4},
        {"name": "Certifications", "type": "STATIC", "order": 5},
        {"name": "Languages", "type": "STATIC", "order": 6},
    ]

    for section_info in sections:
        section = Section(
            name=section_info["name"],
            content_type=section_info["type"],
            order=section_info["order"],
            items=[],
            subsections=[],
        )
        if section.name == "Executive Summary":
            section.items.append(  # pylint: disable=no-member
                Item(
                    content="",
                    status=ItemStatus.TO_REGENERATE,
                    item_type=ItemType.EXECUTIVE_SUMMARY_PARA,
                )
            )
        if section.name == "Key Qualifications":
            skills = None
            if job_data:
                if hasattr(job_data, "skills"):
                    skills = job_data.skills
                elif isinstance(job_data, dict) and "skills" in job_data:
                    skills = job_data["skills"]
            if skills:
                for skill in skills[:8]:
                    section.items.append(  # pylint: disable=no-member
                        Item(
                            content=skill,
                            status=ItemStatus.TO_REGENERATE,
                            item_type=ItemType.KEY_QUALIFICATION,
                        )
                    )
            else:
                for i in range(6):
                    section.items.append(  # pylint: disable=no-member
                        Item(
                            content=f"Key qualification {i+1}",
                            status=ItemStatus.TO_REGENERATE,
                            item_type=ItemType.KEY_QUALIFICATION,
                        )
                    )
        if section.name == "Professional Experience":
            subsection = Subsection(name="Position Title at Company Name", items=[])
            for _ in range(3):
                subsection.items.append(  # pylint: disable=no-member
                    Item(
                        content="",
                        status=ItemStatus.TO_REGENERATE,
                        item_type=ItemType.BULLET_POINT,
                    )
                )
            section.subsections.append(subsection)  # pylint: disable=no-member
        if section.name == "Project Experience":
            subsection = Subsection(name="Project Name", items=[])
            for _ in range(2):
                subsection.items.append(  # pylint: disable=no-member
                    Item(
                        content="",
                        status=ItemStatus.TO_REGENERATE,
                        item_type=ItemType.BULLET_POINT,
                    )
                )
            section.subsections.append(subsection)  # pylint: disable=no-member
        structured_cv.sections.append(section)  # pylint: disable=no-member
    return structured_cv


def determine_section_content_type(section_name: str) -> str:
    dynamic_sections = [
        "Executive Summary",
        "Key Qualifications",
        "Professional Experience",
        "Project Experience",
    ]
    return (
        "DYNAMIC"
        if any(section_name.lower() == s.lower() for s in dynamic_sections)
        else "STATIC"
    )


def determine_item_type(section_name: str) -> ItemType:
    section_lower = section_name.lower()
    if "qualification" in section_lower or "skill" in section_lower:
        return ItemType.KEY_QUALIFICATION
    elif "executive" in section_lower or "summary" in section_lower:
        return ItemType.EXECUTIVE_SUMMARY_PARA
    elif "education" in section_lower:
        return ItemType.EDUCATION_ENTRY
    elif "certification" in section_lower:
        return ItemType.CERTIFICATION_ENTRY
    elif "language" in section_lower:
        return ItemType.LANGUAGE_ENTRY
    else:
        return ItemType.BULLET_POINT


def add_section_items(section: Section, parsed_section: Any, content_type: str) -> None:
    for item_content in parsed_section.items:
        if item_content.strip():
            item_type = determine_item_type(parsed_section.name)
            status = (
                ItemStatus.INITIAL if content_type == "DYNAMIC" else ItemStatus.STATIC
            )
            item = Item(
                content=item_content.strip(), status=status, item_type=item_type
            )
            section.items.append(item)  # pylint: disable=no-member


def add_section_subsections(
    section: Section, parsed_section: Any, content_type: str
) -> None:
    for parsed_subsection in getattr(parsed_section, "subsections", []):
        subsection = Subsection(name=parsed_subsection.name, items=[])
        for item_content in parsed_subsection.items:
            if item_content.strip():
                item_type = determine_item_type(parsed_section.name)
                status = (
                    ItemStatus.INITIAL
                    if content_type == "DYNAMIC"
                    else ItemStatus.STATIC
                )
                item = Item(
                    content=item_content.strip(), status=status, item_type=item_type
                )
                subsection.items.append(item)  # pylint: disable=no-member
        section.subsections.append(subsection)  # pylint: disable=no-member
