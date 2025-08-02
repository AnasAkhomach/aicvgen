from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class ItemStatus(str, Enum):
    """Enumeration for the status of a content item."""

    INITIAL = "initial"
    GENERATED = "generated"
    USER_MODIFIED = "user_modified"
    USER_ACCEPTED = "user_accepted"
    TO_REGENERATE = "to_regenerate"
    GENERATION_FAILED = "generation_failed"
    GENERATED_FALLBACK = "generated_fallback"
    STATIC = "static"
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RATE_LIMITED = "rate_limited"


class ItemType(str, Enum):
    """Enumeration for the type of a content item."""

    BULLET_POINT = "bullet_point"
    KEY_QUALIFICATION = "key_qualification"
    EXECUTIVE_SUMMARY_PARA = "executive_summary_para"
    EXPERIENCE_ROLE_TITLE = "experience_role_title"
    PROJECT_DESCRIPTION_BULLET = "project_description_bullet"
    EDUCATION_ENTRY = "education_entry"
    CERTIFICATION_ENTRY = "certification_entry"
    LANGUAGE_ENTRY = "language_entry"
    SUMMARY_PARAGRAPH = "summary_paragraph"


class MetadataModel(BaseModel):
    # Extend this model as needed for common metadata fields
    item_id: Optional[str] = None  # Added item_id for test and runtime compatibility
    company: Optional[str] = None
    position: Optional[str] = None
    location: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    status: Optional[ItemStatus] = None  # Added status field for item processing

    # Template loader specific fields
    created_by: Optional[str] = None
    source_file: Optional[str] = None
    template_version: Optional[str] = None
    section_type: Optional[str] = None
    subsection_type: Optional[str] = None

    extra: Dict[str, Any] = Field(
        default_factory=dict
    )  # Added extra field for arbitrary data

    def update_status(self, status: ItemStatus, error_message: Optional[str] = None):
        self.status = status
        if error_message is not None:
            self.extra["error_message"] = error_message  # pylint: disable=no-member


class Item(BaseModel):
    """A granular piece of content within the CV (e.g., a bullet point)."""

    id: UUID = Field(default_factory=uuid4)
    content: str
    status: ItemStatus = ItemStatus.INITIAL
    item_type: ItemType = ItemType.BULLET_POINT
    raw_llm_output: Optional[str] = Field(
        default=None,
        description="Raw LLM output for this item for transparency and debugging",
    )  # REQ-FUNC-UI-6
    confidence_score: Optional[float] = None
    metadata: MetadataModel = Field(default_factory=MetadataModel)
    user_feedback: Optional[str] = None


# Content Item Classes for Processing
class ContentItem(BaseModel):
    """Base class for content items that can be processed."""

    id: UUID = Field(default_factory=uuid4)
    content_type: str
    original_content: str
    generated_content: Optional[str] = None
    status: ItemStatus = ItemStatus.INITIAL
    priority: int = 0
    metadata: MetadataModel = Field(default_factory=MetadataModel)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    @property
    def is_ready_for_processing(self) -> bool:
        """Check if item is ready for processing."""

        return self.status in [ItemStatus.INITIAL, ItemStatus.TO_REGENERATE]


class QualificationItem(ContentItem):
    """Represents a key qualification item for processing."""

    content_type: str = Field(default="qualification")
    skill_category: Optional[str] = None
    relevance_score: Optional[float] = None


class ExperienceItem(ContentItem):
    """Represents a work experience item for processing."""

    content_type: str = Field(default="experience")
    company: Optional[str] = None
    role: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None


class ContentWriterJobData(BaseModel):
    """Job data for the content writer agent."""

    title: Optional[str] = None
    raw_text: Optional[str] = None
    description: Optional[str] = None
    company: Optional[str] = None
    skills: Optional[List[str]] = Field(default_factory=list)
    responsibilities: Optional[List[str]] = Field(default_factory=list)
    industry_terms: Optional[List[str]] = Field(default_factory=list)
    company_values: Optional[List[str]] = Field(default_factory=list)


class Subsection(BaseModel):
    """A subsection within a main section of the CV (e.g., a specific job)."""

    id: UUID = Field(default_factory=uuid4)
    name: str  # e.g., "Senior Software Engineer @ TechCorp Inc."
    items: List[Item] = Field(default_factory=list)
    metadata: MetadataModel = Field(
        default_factory=MetadataModel
    )  # e.g., dates, company, location


class Section(BaseModel):
    """A major section of the CV (e.g., "Professional Experience")."""

    id: UUID = Field(default_factory=uuid4)
    name: str
    content_type: str = "DYNAMIC"  # DYNAMIC or STATIC
    subsections: List[Subsection] = Field(default_factory=list)
    items: List[Item] = Field(default_factory=list)  # For sections without subsections
    order: int = 0
    status: ItemStatus = ItemStatus.INITIAL
    metadata: MetadataModel = Field(default_factory=MetadataModel)


class StructuredCV(BaseModel):
    """The main data model representing the entire CV structure."""

    id: UUID = Field(default_factory=uuid4)
    sections: List[Section] = Field(default_factory=list)
    metadata: MetadataModel = Field(default_factory=MetadataModel)

    # New fields for "Big 10" Skills (Task 3.2)
    big_10_skills: List[str] = Field(
        default_factory=list,
        description="Top 10 most relevant skills extracted from CV and job description",
    )
    # big_10_skills_raw_output: Optional[str] = Field(
    #     default=None,
    #     description="Raw LLM output for Big 10 skills generation for transparency",
    # )

    def find_item_by_id(
        self, item_id: str
    ) -> tuple[Optional[Item], Optional[Section], Optional[Subsection]]:
        """Find an item by its ID and return the item along with its parent section and subsection."""

        # Searches by both the actual item.id and the metadata.item_id as fallback.

        # Returns:
        #     tuple: (item, section, subsection) where subsection is None if the item is directly in a section
        for section in self.sections:
            # Check items directly in the section
            for item in section.items:
                # Primary search: by actual item.id
                if str(item.id) == item_id:
                    return item, section, None
                # Fallback search: by metadata.item_id
                if (
                    hasattr(item, "metadata")
                    and item.metadata
                    and hasattr(item.metadata, "item_id")
                    and str(item.metadata.item_id) == item_id
                ):
                    return item, section, None

            # Check items in subsections
            for subsection in section.subsections:
                for item in subsection.items:
                    # Primary search: by actual item.id
                    if str(item.id) == item_id:
                        return item, section, subsection
                    # Fallback search: by metadata.item_id
                    if (
                        hasattr(item, "metadata")
                        and item.metadata
                        and hasattr(item.metadata, "item_id")
                        and str(item.metadata.item_id) == item_id
                    ):
                        return item, section, subsection

        return None, None, None

    def get_section_by_name(self, name: str) -> Optional[Section]:
        """Find a section by its name."""

        for section in self.sections:
            if section.name == name:
                return section
        return None

    def find_section_by_id(self, section_id: str) -> Optional[Section]:
        """Find a section by its ID."""

        for section in self.sections:
            if str(section.id) == section_id:
                return section
        return None

    def update_item_content(self, item_id: str, new_content: str) -> bool:
        """Update the content of a specific item by its ID."""

        item, _, _ = self.find_item_by_id(item_id)
        if item:
            item.content = new_content
            return True
        return False

    def update_item_status(self, item_id: str, new_status: ItemStatus) -> bool:
        """Update the status of a specific item by its ID."""

        item, _, _ = self.find_item_by_id(item_id)
        if item:
            item.status = new_status
            return True
        return False

    def get_items_by_status(self, status: ItemStatus) -> List[Item]:
        """Get all items that match a specific status."""

        items = []
        for section in self.sections:
            for item in section.items:
                if item.status == status:
                    items.append(item)
            for subsection in section.subsections:
                for item in subsection.items:
                    if item.status == status:
                        items.append(item)
        return items

    def to_raw_text(self) -> str:
        """Converts the structured CV back into a raw text string."""
        lines = []
        for section in self.sections:
            lines.append(f"\n## {section.name}\n")
            for item in section.items:
                lines.append(item.content)
            for subsection in section.subsections:
                lines.append(f"\n### {subsection.name}\n")
                for item in subsection.items:
                    lines.append(f"- {item.content}")
        return "\n".join(lines)

    def ensure_required_sections(self) -> None:
        """
        Ensures all required sections exist in the CV structure.
        Creates missing sections with default empty structure.
        """
        required_sections = [
            "Executive Summary",
            "Key Qualifications",
            "Professional Experience",
            "Project Experience",
            "Education",
        ]

        existing_section_names = {section.name for section in self.sections}

        for section_name in required_sections:
            if section_name not in existing_section_names:
                section = Section(
                    id=uuid4(),
                    name=section_name,
                    content_type="DYNAMIC",
                    order=len(self.sections),
                    status=ItemStatus.INITIAL,
                    subsections=[],
                    items=[],
                )
                self.sections.append(section)  # pylint: disable=no-member

    @staticmethod
    def create_empty(
        cv_text: str = "", job_data: Optional["JobDescriptionData"] = None
    ) -> "StructuredCV":
        """
        Creates an empty CV structure with pre-initialized standard sections.
        """

        # Pre-initialize standard CV sections
        standard_sections = [
            "Executive Summary",
            "Key Qualifications",
            "Professional Experience",
            "Project Experience",
            "Education",
        ]

        sections = []
        for i, section_name in enumerate(standard_sections):
            section = Section(
                id=uuid4(),
                name=section_name,
                content_type="DYNAMIC",
                order=i,
                status=ItemStatus.INITIAL,
                subsections=[],
                items=[],
            )
            sections.append(section)

        structured_cv = StructuredCV(sections=sections)
        if job_data:
            structured_cv.metadata.extra[
                "job_description"
            ] = job_data.model_dump()  # pylint: disable=no-member
        else:
            structured_cv.metadata.extra[
                "job_description"
            ] = {}  # pylint: disable=no-member
        structured_cv.metadata.extra[
            "original_cv_text"
        ] = cv_text  # pylint: disable=no-member
        return structured_cv


class JobDescriptionData(BaseModel):
    """A structured representation of a parsed job description."""

    raw_text: str
    job_title: Optional[str] = None
    company_name: Optional[str] = None
    main_job_description_raw: Optional[str] = None
    skills: List[str] = Field(default_factory=list)
    experience_level: Optional[str] = None
    responsibilities: List[str] = Field(default_factory=list)
    industry_terms: List[str] = Field(default_factory=list)
    company_values: List[str] = Field(default_factory=list)
    error: Optional[str] = None


class SkillEntry(BaseModel):
    """Data model for individual skill entries."""

    name: str
    category: Optional[str] = None
    proficiency_level: Optional[str] = None
    years_experience: Optional[int] = None
    is_primary: bool = False
    metadata: MetadataModel = Field(default_factory=MetadataModel)


class ExperienceEntry(BaseModel):
    """Data model for individual experience entries."""

    company: str
    position: str
    duration: str
    location: Optional[str] = None
    description: str
    responsibilities: List[str] = Field(default_factory=list)
    achievements: List[str] = Field(default_factory=list)
    technologies: List[str] = Field(default_factory=list)
    metadata: MetadataModel = Field(default_factory=MetadataModel)
