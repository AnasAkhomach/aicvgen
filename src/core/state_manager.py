from typing import TypedDict, List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
import enum
import json
import uuid
import os
import logging
import time
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    filename="debug.log",
    filemode="a",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class WorkflowStage(TypedDict):
    """
    Represents a stage in the workflow.
    """

    stage_name: str
    description: str
    is_completed: bool


class AgentIO(TypedDict):
    """
    Represents the input and output of an agent.
    """

    input: Dict[str, Any]  # Input data.
    output: Any  # Output data.
    description: str  # Description of what the agent does.


class JobDescriptionData:
    """
    Represents the parsed data of a job description.
    """

    def __init__(
        self,
        raw_text,
        skills,
        experience_level,
        responsibilities,
        industry_terms,
        company_values,
        error=None,
    ):
        self.raw_text = raw_text
        self.skills = skills
        self.experience_level = experience_level
        self.responsibilities = responsibilities
        self.industry_terms = industry_terms
        self.company_values = company_values
        self.error = error  # Optional error field to store error messages

    def __str__(self):
        return f"JobDescriptionData(raw_text='{self.raw_text}', skills={self.skills}, experience_level='{self.experience_level}', responsibilities={self.responsibilities}, industry_terms={self.industry_terms}, company_values={self.company_values})"

    def to_dict(self):
        """
        Convert JobDescriptionData to a dictionary.

        Returns:
            dict: A dictionary representation of this JobDescriptionData object
        """
        return {
            "raw_text": self.raw_text,
            "skills": self.skills,
            "experience_level": self.experience_level,
            "responsibilities": self.responsibilities,
            "industry_terms": self.industry_terms,
            "company_values": self.company_values,
            "error": self.error,
        }


@dataclass
class VectorStoreConfig:
    dimension: int
    index_type: str  # IndexFlatL2, IndexIVFFlat


@dataclass
class ContentPiece:
    """
    Represents a single piece of generated content (e.g., a bullet point, a summary paragraph).
    """

    content: str
    section_type: str  # e.g., "summary", "experience", "skill", "project"
    piece_id: str  # Unique identifier for the piece
    status: str = (
        "pending_generation"  # e.g., "pending_generation", "pending_review", "approved", "rejected"
    )
    feedback: str = ""  # Feedback from review
    revision_count: int = 0


class CVData(TypedDict):
    """
    Represents the data extracted from the user's CV.
    Now includes fields for extracted raw text and lists of content pieces for different sections.
    """

    raw_text: str  # The original text of the CV
    summary: str  # Original summary text from CV, if any
    experiences: List[str]  # Original experiences list from CV, if any
    skills: List[str]  # Original skills list from CV, if any
    education: List[str]  # Original education list from CV, if any
    projects: List[str]  # Original projects list from CV, if any


@dataclass
class SkillEntry:
    """
    Represents a skill entry.
    """

    text: str


@dataclass
class ExperienceEntry:
    """
    Represents an experience entry.
    """

    text: str


class ContentData(Dict):
    """
    Represents the assembled tailored CV content for rendering.
    """

    def __init__(
        self,
        summary="",
        experience_bullets=None,
        skills_section="",
        projects=None,
        other_content=None,
        name="",
        email="",
        phone="",
        linkedin="",
        github="",
        education=None,
        certifications=None,
        languages=None,
    ):
        super().__init__()
        self["name"] = name
        self["email"] = email
        self["phone"] = phone
        self["linkedin"] = linkedin
        self["github"] = github
        self["summary"] = summary
        experience_bullets = experience_bullets if experience_bullets is not None else []
        self["experience_bullets"] = experience_bullets
        self["skills_section"] = skills_section
        projects = projects if projects is not None else []
        self["projects"] = projects
        education = education if education is not None else []
        self["education"] = education
        certifications = certifications if certifications is not None else []
        self["certifications"] = certifications
        languages = languages if languages is not None else []
        self["languages"] = languages
        other_content = other_content if other_content is not None else {}
        self["other_content"] = other_content

    @property
    def name(self):
        return self.get("name")

    @property
    def email(self):
        return self.get("email")

    @property
    def phone(self):
        return self.get("phone")

    @property
    def linkedin(self):
        return self.get("linkedin")

    @property
    def github(self):
        return self.get("github")

    @property
    def summary(self):
        return self.get("summary")

    @property
    def experience_bullets(self):
        return self.get("experience_bullets")

    @property
    def skills_section(self):
        return self.get("skills_section")

    @property
    def projects(self):
        return self.get("projects")

    @property
    def education(self):
        return self.get("education")

    @property
    def certifications(self):
        return self.get("certifications")

    @property
    def languages(self):
        return self.get("languages")

    @property
    def other_content(self):
        return self.get("other_content")


class WorkflowState(TypedDict):
    """
    Represents the overall state of the CV tailoring workflow.
    Now tracks content pieces and their review status for iterative refinement.
    """

    job_description: Dict  # Parsed job description data
    user_cv: CVData  # Extracted user CV data
    extracted_skills: Dict  # Deprecate or refine - skills are now in user_cv
    generated_content: Dict  # Map of piece_id -> ContentPiece
    content_pieces: List[ContentPiece]  # List of all content pieces
    current_content_piece_id: Optional[
        str
    ]  # ID of the content piece currently being processed/reviewed
    content_generation_plan: List[Dict[str, Any]]  # A plan of content pieces to generate/review
    plan_index: int  # To track progress through the generation plan
    formatted_cv_text: str  # Final formatted text before rendering
    rendered_cv: str  # Final rendered output
    feedback: List[Dict[str, Any]]  # List of feedback entries for transparency
    revision_history: List[Dict[str, Any]]  # List of revision history entries
    current_stage: WorkflowStage  # Current stage of the overall workflow
    workflow_id: str  # Workflow ID
    relevant_experiences: List[str]  # Relevant experiences from vector search
    research_results: Dict[str, Any]  # Research results
    review_status: str  # e.g., "pending", "approved", "rejected"
    review_feedback: str  # Store feedback comments for the current piece


class ItemStatus(enum.Enum):
    """
    Enumeration for the status of an item in the CV.
    """

    INITIAL = "initial"  # Parsed from raw input
    GENERATED = "generated"  # Generated by ContentWriter
    USER_EDITED = "user_edited"  # Modified directly by user in UI
    TO_REGENERATE = "to_regenerate"  # Marked for regeneration by user
    ACCEPTED = "accepted"  # Approved by user
    STATIC = "static"  # Content from base CV, not tailored by AI

    def __str__(self):
        return self.value  # Allows easy string conversion


class ItemType(enum.Enum):
    """
    Enumeration for the type of an item in the CV.
    """

    BULLET_POINT = "bullet_point"
    KEY_QUAL = "key_qual"
    SUMMARY_PARAGRAPH = "summary_paragraph"
    SECTION_TITLE = "section_title"
    SUBSECTION_TITLE = "subsection_title"
    EDUCATION_ENTRY = "education_entry"
    CERTIFICATION_ENTRY = "certification_entry"
    LANGUAGE_ENTRY = "language_entry"

    def __str__(self):
        return self.value


@dataclass
class Item:
    """
    Represents a granular piece of content in the CV (e.g., a bullet point, a key qualification).
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))  # Unique ID for granular access
    content: str = ""
    status: ItemStatus = ItemStatus.INITIAL
    item_type: ItemType = ItemType.BULLET_POINT
    metadata: Dict[str, Any] = field(
        default_factory=dict
    )  # e.g., relevance_score, source_chunk_id, original_line_number
    user_feedback: Optional[str] = None  # Optional user comment from UI

    def to_dict(self):
        """Helper for serialization"""
        return {
            "id": self.id,
            "content": self.content,
            "status": (self.status.value if isinstance(self.status, enum.Enum) else self.status),
            "item_type": (
                self.item_type.value if isinstance(self.item_type, enum.Enum) else self.item_type
            ),
            "metadata": self.metadata,
            "user_feedback": self.user_feedback,
        }

    @classmethod
    def from_dict(cls, data):
        """Create an Item from a dictionary"""
        # Convert string status/item_type to enum if needed
        status = data.get("status", "initial")
        if isinstance(status, str):
            try:
                status = ItemStatus(status)
            except ValueError:
                status = ItemStatus.INITIAL

        item_type = data.get("item_type", "bullet_point")
        if isinstance(item_type, str):
            try:
                item_type = ItemType(item_type)
            except ValueError:
                item_type = ItemType.BULLET_POINT

        return cls(
            id=data.get("id", str(uuid.uuid4())),
            content=data.get("content", ""),
            status=status,
            item_type=item_type,
            metadata=data.get("metadata", {}),
            user_feedback=data.get("user_feedback"),
        )


@dataclass
class Subsection:
    """
    Represents a subsection within a section (e.g., a specific job role within Professional Experience).
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""  # e.g., "Software Engineer at XYZ Inc."
    items: List[Item] = field(default_factory=list)  # e.g., bullet points under this role/project
    metadata: Dict[str, Any] = field(default_factory=dict)  # e.g., dates, company, location
    raw_text: str = ""  # Original text snippet from parsing

    def to_dict(self):
        """Helper for serialization"""
        return {
            "id": self.id,
            "name": self.name,
            "items": [item.to_dict() for item in self.items],
            "metadata": self.metadata,
            "raw_text": self.raw_text,
        }

    @classmethod
    def from_dict(cls, data):
        """Create a Subsection from a dictionary"""
        items = []
        for item_data in data.get("items", []):
            items.append(Item.from_dict(item_data))

        return cls(
            id=data.get("id", str(uuid.uuid4())),
            name=data.get("name", ""),
            items=items,
            metadata=data.get("metadata", {}),
            raw_text=data.get("raw_text", ""),
        )


@dataclass
class Section:
    """
    Represents a major section in the CV (e.g., Professional Experience, Key Qualifications).
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""  # e.g., "Professional Experience", "Key Qualifications"
    content_type: str = "DYNAMIC"  # or "STATIC"
    subsections: List[Subsection] = field(
        default_factory=list
    )  # For sections like Experience, Projects
    items: List[Item] = field(
        default_factory=list
    )  # For sections like Key Quals, Education, Languages
    raw_text: str = ""  # Original text snippet from parsing
    order: int = 0  # For maintaining section order from the template
    status: ItemStatus = ItemStatus.INITIAL  # Status of the entire section
    user_feedback: Optional[str] = None  # Optional user comment for the section

    def to_dict(self):
        """Helper for serialization"""
        return {
            "id": self.id,
            "name": self.name,
            "content_type": self.content_type,
            "subsections": [subsection.to_dict() for subsection in self.subsections],
            "items": [item.to_dict() for item in self.items],
            "raw_text": self.raw_text,
            "order": self.order,
            "status": (self.status.value if isinstance(self.status, enum.Enum) else self.status),
            "user_feedback": self.user_feedback,
        }

    @classmethod
    def from_dict(cls, data):
        """Create a Section from a dictionary"""
        subsections = []
        for subsection_data in data.get("subsections", []):
            subsections.append(Subsection.from_dict(subsection_data))

        items = []
        for item_data in data.get("items", []):
            items.append(Item.from_dict(item_data))

        # Convert string status to enum if needed
        status = data.get("status", "initial")
        if isinstance(status, str):
            try:
                status = ItemStatus(status)
            except ValueError:
                status = ItemStatus.INITIAL

        return cls(
            id=data.get("id", str(uuid.uuid4())),
            name=data.get("name", ""),
            content_type=data.get("content_type", "DYNAMIC"),
            subsections=subsections,
            items=items,
            raw_text=data.get("raw_text", ""),
            order=data.get("order", 0),
            status=status,
            user_feedback=data.get("user_feedback"),
        )


class EnumEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, enum.Enum):
            return obj.value
        return super().default(obj)


@dataclass
class StructuredCV:
    """
    The main data model representing a CV with a hierarchical structure.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))  # Session ID / CV ID
    sections: List[Section] = field(default_factory=list)
    metadata: Dict[str, Any] = field(
        default_factory=dict
    )  # e.g., original file paths, timestamp, main_jd_text, similar_jd_texts

    def to_dict(self):
        """Convert the StructuredCV to a dictionary for JSON serialization"""
        return {
            "id": self.id,
            "sections": [section.to_dict() for section in self.sections],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data):
        """Create a StructuredCV from a dictionary"""
        sections = []
        for section_data in data.get("sections", []):
            sections.append(Section.from_dict(section_data))

        return cls(
            id=data.get("id", str(uuid.uuid4())),
            sections=sections,
            metadata=data.get("metadata", {}),
        )

    def save(self, directory="data/sessions"):
        """Save the StructuredCV to a JSON file"""
        # Create directory if it doesn't exist
        os.makedirs(f"{directory}/{self.id}", exist_ok=True)

        # Save to file
        with open(f"{directory}/{self.id}/state.json", "w") as f:
            json.dump(self.to_dict(), f, indent=2, cls=EnumEncoder)

        return f"{directory}/{self.id}/state.json"

    @classmethod
    def load(cls, session_id, directory="data/sessions"):
        """Load a StructuredCV from a JSON file"""
        try:
            with open(f"{directory}/{session_id}/state.json", "r") as f:
                data = json.load(f)
                return cls.from_dict(data)
        except FileNotFoundError:
            logger.error(f"Session file not found: {directory}/{session_id}/state.json")
            return None
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in session file: {directory}/{session_id}/state.json")
            return None

    def get_section_by_name(self, name):
        """Get a section by name"""
        for section in self.sections:
            if section.name.lower() == name.lower():
                return section
        return None

    def find_item_by_id(self, item_id):
        """Find an item by its ID"""
        for section in self.sections:
            # Check items directly in the section
            for item in section.items:
                if item.id == item_id:
                    return item, section, None

            # Check items in subsections
            for subsection in section.subsections:
                for item in subsection.items:
                    if item.id == item_id:
                        return item, section, subsection

        return None, None, None

    def update_item_content(self, item_id, new_content):
        """Update the content of an item"""
        item, _, _ = self.find_item_by_id(item_id)
        if item:
            item.content = new_content
            return True
        return False

    def update_item_status(self, item_id, new_status):
        """Update the status of an item"""
        item, _, _ = self.find_item_by_id(item_id)
        if item:
            if isinstance(new_status, str):
                try:
                    new_status = ItemStatus(new_status)
                except ValueError:
                    new_status = ItemStatus.INITIAL

            item.status = new_status
            return True
        return False

    def get_items_by_status(self, status):
        """Get all items with a specific status"""
        if isinstance(status, str):
            try:
                status = ItemStatus(status)
            except ValueError:
                status = ItemStatus.INITIAL

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

    def find_section_by_id(self, section_id):
        """Find a section by its ID"""
        for section in self.sections:
            if section.id == section_id:
                return section
        return None

    def update_section_content(self, section_id, new_content_structure):
        """
        Update the content of a section

        Args:
            section_id: ID of the section
            new_content_structure: Dictionary with updated content

        Returns:
            bool: Success or failure
        """
        if self._structured_cv:
            success = self._structured_cv.update_section_content(section_id, new_content_structure)
            if success:
                self.save_state()
            return success
        return False

    def update_section_status(self, section_id, new_status):
        """
        Update the status of an entire section

        Args:
            section_id: ID of the section
            new_status: New status to set

        Returns:
            bool: Success or failure
        """
        if self._structured_cv:
            section = self._structured_cv.find_section_by_id(section_id)
            if section:
                old_status = section.status
                success = self._structured_cv.update_section_status(section_id, new_status)
                if success:
                    self._log_section_state_change(section_id, old_status, new_status)
                    self.save_state()
                return success
        return False

    def _log_section_state_change(self, section_id, old_status, new_status):
        """Log a change in section status for debugging and analytics"""
        section = self._structured_cv.find_section_by_id(section_id)
        if section:
            section_name = section.name
            logger.info(
                f"Section status change: {section_id} ({section_name}) from {old_status} to {new_status}"
            )
        else:
            logger.info(
                f"Section status change: {section_id} (unknown) from {old_status} to {new_status}"
            )

    def get_sections_by_status(self, status):
        """Get all sections with a specific status"""
        if self._structured_cv:
            return self._structured_cv.get_sections_by_status(status)
        return []

    def get_diagnostics(self) -> Dict[str, Any]:
        """
        Get diagnostic information about the current state.

        Returns:
            A dictionary containing diagnostic information.
        """
        if not self._structured_cv:
            return {"status": "No CV data loaded"}

        sections = self._structured_cv.sections
        items_by_status = {}
        total_items = 0

        for section in sections:
            for item in section.items:
                status = str(item.status)
                items_by_status[status] = items_by_status.get(status, 0) + 1
                total_items += 1
            for subsection in section.subsections:
                for item in subsection.items:
                    status = str(item.status)
                    items_by_status[status] = items_by_status.get(status, 0) + 1
                    total_items += 1

        return {
            "session_id": self.session_id,
            "last_save": self._last_save_time,
            "total_sections": len(sections),
            "total_items": total_items,
            "items_by_status": items_by_status,
            "state_changes": len(self._state_changes),
        }

    def to_content_data(self):
        """
        Convert the StructuredCV to ContentData format.
        
        Returns:
            ContentData instance containing the CV content.
        """
        # Initialize content dictionary with default values
        content = {
            "summary": "",
            "experience_bullets": [],
            "skills_section": "",
            "projects": [],
            "education": [],
            "certifications": [],
            "languages": [],
            "name": self.metadata.get("name", ""),
            "email": self.metadata.get("email", ""),
            "phone": self.metadata.get("phone", ""),
            "linkedin": self.metadata.get("linkedin", ""),
            "github": self.metadata.get("github", "")
        }
        
        # Extract content from sections
        for section in self.sections:
            section_name = section.name.lower()
            
            if section_name in ["summary", "executive summary", "professional summary"]:
                # Get summary content from items
                summary_items = []
                for item in section.items:
                    if isinstance(item.content, str):
                        summary_items.append(item.content)
                    elif isinstance(item.content, dict) and "text" in item.content:
                        summary_items.append(item.content["text"])
                content["summary"] = "\n".join(summary_items)
                
            elif section_name in ["experience", "work experience", "professional experience"]:
                # Get experience bullets
                experience_bullets = []
                for item in section.items:
                    if isinstance(item.content, str):
                        experience_bullets.append(item.content)
                    elif isinstance(item.content, dict):
                        if "bullets" in item.content:
                            experience_bullets.extend(item.content["bullets"])
                        elif "text" in item.content:
                            experience_bullets.append(item.content["text"])
                content["experience_bullets"] = experience_bullets
                
            elif section_name in ["skills", "technical skills", "core competencies"]:
                # Get skills section content
                skills_items = []
                for item in section.items:
                    if isinstance(item.content, str):
                        skills_items.append(item.content)
                    elif isinstance(item.content, dict) and "text" in item.content:
                        skills_items.append(item.content["text"])
                content["skills_section"] = "\n".join(skills_items)
                
            elif section_name in ["projects", "key projects", "notable projects"]:
                # Get projects
                projects = []
                for item in section.items:
                    if isinstance(item.content, dict):
                        projects.append(item.content)
                    elif isinstance(item.content, str):
                        projects.append({"title": item.name or "Project", "description": item.content})
                content["projects"] = projects
                
            elif section_name in ["education", "academic background"]:
                # Get education
                education = []
                for item in section.items:
                    if isinstance(item.content, dict):
                        education.append(item.content)
                    elif isinstance(item.content, str):
                        education.append({"degree": item.name or "Degree", "description": item.content})
                content["education"] = education
                
            elif section_name in ["certifications", "certificates"]:
                # Get certifications
                certifications = []
                for item in section.items:
                    if isinstance(item.content, dict):
                        certifications.append(item.content)
                    elif isinstance(item.content, str):
                        certifications.append({"name": item.name or "Certification", "description": item.content})
                content["certifications"] = certifications
                
            elif section_name in ["languages"]:
                # Get languages
                languages = []
                for item in section.items:
                    if isinstance(item.content, dict):
                        languages.append(item.content)
                    elif isinstance(item.content, str):
                        languages.append({"language": item.name or "Language", "proficiency": item.content})
                content["languages"] = languages
        
        # Create and return ContentData instance
        from src.core.state_manager import ContentData
        return ContentData(**content)

    @property
    def cv_data(self):
        """
        Get the CV data in a format compatible with the main application.
        
        Returns:
            Dictionary containing CV data or None if no CV exists.
        """
        if not self._structured_cv:
            return None
        
        # Convert StructuredCV to a dictionary format expected by main.py
        content_data = self.convert_to_content_data()
        if content_data:
            return {
                "content": dict(content_data),
                "sections": [{
                    "id": section.id,
                    "name": section.name,
                    "status": str(section.status)
                } for section in self._structured_cv.sections]
            }
        return None

    def update_cv_data(self, new_content):
        """
        Update the CV data with new content.
        
        Args:
            new_content: The new content to update the CV with.
            
        Returns:
            bool: True if update was successful, False otherwise.
        """
        try:
            if isinstance(new_content, dict):
                # If we don't have a structured CV yet, create one
                if not self._structured_cv:
                    self.create_new_cv()
                
                # Handle different content formats
                if "content" in new_content:
                    # Format: {"content": {...}}
                    content_dict = new_content["content"]
                    # Filter to only include valid ContentData arguments
                    filtered_content = self._filter_content_data_args(content_dict)
                    content_data = ContentData(**filtered_content)
                    # Actually populate the structured CV with the content
                    self._populate_structured_cv_from_content(content_data)
                    logger.info(f"Updated CV data with wrapped content")
                    self.save_state()
                    return True
                else:
                    # Format: direct content dictionary
                    # This is the actual CV content structure being passed
                    # Filter to only include valid ContentData arguments
                    filtered_content = self._filter_content_data_args(new_content)
                    content_data = ContentData(**filtered_content)
                    # Actually populate the structured CV with the content
                    self._populate_structured_cv_from_content(content_data)
                    logger.info(f"Updated CV data with direct content structure")
                    # Store the content in the structured CV
                    # The new_content is already the CV content structure
                    self.save_state()
                    return True
                    
            logger.warning(f"Invalid content format for CV update: {type(new_content)}")
            return False
            
        except Exception as e:
            logger.error(f"Failed to update CV data: {str(e)}")
            logger.error(f"Content keys: {list(new_content.keys()) if isinstance(new_content, dict) else 'Not a dict'}")
            return False

    def _filter_content_data_args(self, content_dict: dict) -> dict:
        """
        Filter content dictionary to only include valid ContentData arguments.
        
        Args:
            content_dict: Dictionary containing content data
            
        Returns:
            Filtered dictionary with only valid ContentData arguments
        """
        # Valid ContentData constructor arguments
        valid_args = {
            'summary', 'experience_bullets', 'skills_section', 'projects',
            'other_content', 'name', 'email', 'phone', 'linkedin', 'github',
            'education', 'certifications', 'languages'
        }
        
        # Filter the content to only include valid arguments
        filtered = {k: v for k, v in content_dict.items() if k in valid_args}
        
        # Log what was filtered out for debugging
        filtered_out = {k: v for k, v in content_dict.items() if k not in valid_args}
        if filtered_out:
            logger.info(f"Filtered out non-ContentData arguments: {list(filtered_out.keys())}")
        
        return filtered

    def _populate_structured_cv_from_content(self, content_data: ContentData):
        """
        Populate the StructuredCV with content from ContentData.
        
        Args:
            content_data: ContentData instance containing the CV content
        """
        if not self._structured_cv:
            logger.error("Cannot populate: No StructuredCV instance exists")
            return
            
        # Clear existing sections
        self._structured_cv.sections = []
        
        # Add Executive Summary section
        if content_data.get("summary"):
            summary_section = Section(
                id="executive_summary",
                name="Executive Summary",
                status=ItemStatus.GENERATED
            )
            summary_item = Item(
                id="summary_item",
                content=content_data.get("summary", ""),
                status=ItemStatus.GENERATED
            )
            summary_section.items.append(summary_item)
            self._structured_cv.sections.append(summary_section)
            
        # Add Professional Experience section
        if content_data.get("experience_bullets"):
            experience_section = Section(
                id="professional_experience",
                name="Professional Experience",
                status=ItemStatus.GENERATED
            )
            
            for i, exp in enumerate(content_data.get("experience_bullets", [])):
                if isinstance(exp, dict):
                    # Structured experience entry
                    exp_subsection = Subsection(
                        id=f"experience_{i}",
                        name=exp.get("position", f"Experience {i+1}"),
                        metadata={
                            "company": exp.get("company", ""),
                            "location": exp.get("location", ""),
                            "period": exp.get("period", "")
                        }
                    )
                    
                    # Add bullets as items
                    for j, bullet in enumerate(exp.get("bullets", [])):
                        bullet_item = Item(
                            id=f"experience_{i}_bullet_{j}",
                            content=bullet,
                            status=ItemStatus.GENERATED
                        )
                        exp_subsection.items.append(bullet_item)
                    
                    experience_section.subsections.append(exp_subsection)
                else:
                    # Simple string experience
                    exp_item = Item(
                        id=f"experience_item_{i}",
                        content=str(exp),
                        status=ItemStatus.GENERATED
                    )
                    experience_section.items.append(exp_item)
                    
            self._structured_cv.sections.append(experience_section)
            
        # Add Key Qualifications/Skills section
        if content_data.get("skills_section"):
            skills_section = Section(
                id="key_qualifications",
                name="Key Qualifications",
                status=ItemStatus.GENERATED
            )
            skills_item = Item(
                id="skills_item",
                content=content_data.get("skills_section", ""),
                status=ItemStatus.GENERATED
            )
            skills_section.items.append(skills_item)
            self._structured_cv.sections.append(skills_section)
            
        # Add Projects section
        if content_data.get("projects"):
            projects_section = Section(
                id="project_experience",
                name="Project Experience",
                status=ItemStatus.GENERATED
            )
            
            for i, project in enumerate(content_data.get("projects", [])):
                if isinstance(project, dict):
                    project_subsection = Subsection(
                        id=f"project_{i}",
                        name=project.get("name", f"Project {i+1}"),
                        metadata={
                            "description": project.get("description", ""),
                            "technologies": project.get("technologies", [])
                        }
                    )
                    
                    # Add project bullets as items
                    for j, bullet in enumerate(project.get("bullets", [])):
                        bullet_item = Item(
                            id=f"project_{i}_bullet_{j}",
                            content=bullet,
                            status=ItemStatus.GENERATED
                        )
                        project_subsection.items.append(bullet_item)
                    
                    projects_section.subsections.append(project_subsection)
                else:
                    # Simple string project
                    project_item = Item(
                        id=f"project_item_{i}",
                        content=str(project),
                        status=ItemStatus.GENERATED
                    )
                    projects_section.items.append(project_item)
                    
            self._structured_cv.sections.append(projects_section)
            
        # Add Education section
        if content_data.get("education"):
            education_section = Section(
                id="education",
                name="Education",
                status=ItemStatus.GENERATED
            )
            
            for i, edu in enumerate(content_data.get("education", [])):
                if isinstance(edu, dict):
                    edu_subsection = Subsection(
                        id=f"education_{i}",
                        name=edu.get("degree", f"Education {i+1}"),
                        metadata={
                            "institution": edu.get("institution", ""),
                            "location": edu.get("location", ""),
                            "period": edu.get("period", "")
                        }
                    )
                    
                    # Add education details as items
                    for j, detail in enumerate(edu.get("details", [])):
                        detail_item = Item(
                            id=f"education_{i}_detail_{j}",
                            content=detail,
                            status=ItemStatus.GENERATED
                        )
                        edu_subsection.items.append(detail_item)
                    
                    education_section.subsections.append(edu_subsection)
                else:
                    # Simple string education
                    edu_item = Item(
                        id=f"education_item_{i}",
                        content=str(edu),
                        status=ItemStatus.GENERATED
                    )
                    education_section.items.append(edu_item)
                    
            self._structured_cv.sections.append(education_section)
            
        # Update personal info in metadata
        if not self._structured_cv.metadata:
            self._structured_cv.metadata = {}
            
        self._structured_cv.metadata.update({
            "name": content_data.get("name", ""),
            "email": content_data.get("email", ""),
            "phone": content_data.get("phone", ""),
            "linkedin": content_data.get("linkedin", ""),
            "github": content_data.get("github", "")
        })
        
        logger.info(f"Populated StructuredCV with {len(self._structured_cv.sections)} sections")

class StateManager:
    """
    Manages the state of the CV tailoring process.
    """

    def __init__(self, session_id: Optional[str] = None):
        """
        Initialize the StateManager.

        Args:
            session_id: Optional ID for the session. If not provided, a new one will be generated.
        """
        self.session_id = session_id or str(uuid.uuid4())
        self._structured_cv = None
        self._last_save_time = None
        self._state_changes = []  # Track state transitions
        logger.info(f"Initialized StateManager with session ID: {session_id}")

    def create_new_cv(self, metadata=None):
        """
        Create a new StructuredCV.

        Args:
            metadata: Optional metadata for the CV.

        Returns:
            The new StructuredCV instance.
        """
        self._structured_cv = StructuredCV(id=self.session_id, metadata=metadata or {})
        return self._structured_cv

    def load_state(self, session_id=None):
        """
        Load a StructuredCV from a saved state.

        Args:
            session_id: The ID of the session to load. If not provided, uses the instance's session_id.

        Returns:
            The loaded StructuredCV instance, or None if loading failed.
        """
        try:
            start_time = time.time()
            session_id = session_id or self.session_id
            state_file = f"data/sessions/{session_id}/state.json"
            if not os.path.exists(state_file):
                logger.warning(f"State file not found: {state_file}")
                return None

            with open(state_file, "r") as f:
                data = json.load(f)
                self._structured_cv = StructuredCV.from_dict(data)

            log_file = f"data/sessions/{session_id}/state_changes.json"
            if os.path.exists(log_file):
                with open(log_file, "r") as f:
                    self._state_changes = json.load(f)

            duration = time.time() - start_time
            logger.info(f"State loaded from {state_file} in {duration:.2f}s")
            return self._structured_cv
        except Exception as e:
            logger.error(f"Failed to load state: {str(e)}")
            return None

    def save_state(self):
        """
        Save the current StructuredCV state.

        Returns:
            The path to the saved file, or None if saving failed.
        """
        try:
            start_time = time.time()
            if not self._structured_cv:
                logger.warning("No CV data to save")
                return None

            os.makedirs(f"data/sessions/{self._structured_cv.id}", exist_ok=True)

            state_file = f"data/sessions/{self._structured_cv.id}/state.json"
            with open(state_file, "w") as f:
                json.dump(self._structured_cv.to_dict(), f, indent=2, cls=EnumEncoder)

            log_file = f"data/sessions/{self._structured_cv.id}/state_changes.json"
            with open(log_file, "w") as f:
                json.dump(self._state_changes, f, indent=2)

            duration = time.time() - start_time
            self._last_save_time = datetime.now().isoformat()
            logger.info(f"State saved to {state_file} in {duration:.2f}s")
            return state_file
        except Exception as e:
            logger.error(f"Failed to save state: {str(e)}")
            return None

    def get_structured_cv(self):
        """
        Get the current StructuredCV instance.

        Returns:
            The current StructuredCV instance, or None if it doesn't exist.
        """
        return self._structured_cv

    def update_item_content(self, item_id, new_content):
        """
        Update the content of an item.

        Args:
            item_id: The ID of the item to update.
            new_content: The new content for the item.

        Returns:
            True if the update was successful, False otherwise.
        """
        if not self._structured_cv:
            logger.error("Cannot update item: No StructuredCV instance exists.")
            return False

        return self._structured_cv.update_item_content(item_id, new_content)

    def _log_state_change(self, item_id: str, old_status: str, new_status: str):
        """
        Log a state change for an item.

        Args:
            item_id: The ID of the item.
            old_status: The previous status of the item.
            new_status: The new status of the item.
        """
        timestamp = datetime.now().isoformat()
        change = {
            "timestamp": timestamp,
            "item_id": item_id,
            "old_status": old_status,
            "new_status": new_status,
        }
        self._state_changes.append(change)
        logger.info(f"State change: Item {item_id} transitioned from {old_status} to {new_status}")

    def update_item_status(self, item_id: str, new_status: str) -> bool:
        """
        Update the status of an item.

        Args:
            item_id: The ID of the item to update.
            new_status: The new status for the item.

        Returns:
            True if the update was successful, False otherwise.
        """
        try:
            item = self.get_item(item_id)
            if item:
                old_status = str(item.status)
                item.status = new_status
                self._log_state_change(item_id, old_status, str(new_status))
                return True
            logger.warning(f"Failed to update status: Item {item_id} not found")
            return False
        except Exception as e:
            logger.error(f"Error updating item status: {str(e)}")
            return False

    def get_item(self, item_id):
        """
        Get an item by its ID.

        Args:
            item_id: The ID of the item to get.

        Returns:
            The item, or None if it wasn't found.
        """
        if not self._structured_cv:
            logger.error("Cannot get item: No StructuredCV instance exists.")
            return None

        item, _, _ = self._structured_cv.find_item_by_id(item_id)
        return item

    def get_items_by_status(self, status):
        """
        Get all items with a specific status.

        Args:
            status: The status to filter by.

        Returns:
            A list of items with the specified status.
        """
        if self._structured_cv:
            return self._structured_cv.get_items_by_status(status)
        return []

    def find_section_by_id(self, section_id):
        """Find a section by its ID"""
        if self._structured_cv:
            return self._structured_cv.find_section_by_id(section_id)
        return None

    def convert_to_content_data(self):
        """
        Convert the current StructuredCV to ContentData format.

        Returns:
            ContentData instance or None if no CV exists.
        """
        if not self._structured_cv:
            return None
        
        return self._structured_cv.to_content_data()

    @property
    def cv_data(self):
        """
        Get the CV data in a format compatible with the main application.
        
        Returns:
            Dictionary containing CV data or None if no CV exists.
        """
        if not self._structured_cv:
            return None
        
        # Convert StructuredCV to a dictionary format expected by main.py
        content_data = self.convert_to_content_data()
        if content_data:
            return {
                "content": dict(content_data),
                "sections": [{
                    "id": section.id,
                    "name": section.name,
                    "status": str(section.status)
                } for section in self._structured_cv.sections]
            }
        return None

    def update_cv_data(self, new_content):
        """
        Update the CV data with new content.
        
        Args:
            new_content: The new content to update the CV with.
            
        Returns:
            bool: True if update was successful, False otherwise.
        """
        try:
            if isinstance(new_content, dict):
                # If we don't have a structured CV yet, create one
                if not self._structured_cv:
                    self.create_new_cv()
                
                # Handle different content formats
                if "content" in new_content:
                    # Format: {"content": {...}}
                    content_dict = new_content["content"]
                    # Filter to only include valid ContentData arguments
                    filtered_content = self._filter_content_data_args(content_dict)
                    content_data = ContentData(**filtered_content)
                    # Actually populate the structured CV with the content
                    self._populate_structured_cv_from_content(content_data)
                    logger.info(f"Updated CV data with wrapped content")
                    self.save_state()
                    return True
                else:
                    # Format: direct content dictionary
                    # This is the actual CV content structure being passed
                    # Filter to only include valid ContentData arguments
                    filtered_content = self._filter_content_data_args(new_content)
                    content_data = ContentData(**filtered_content)
                    # Actually populate the structured CV with the content
                    self._populate_structured_cv_from_content(content_data)
                    logger.info(f"Updated CV data with direct content structure")
                    # Store the content in the structured CV
                    # The new_content is already the CV content structure
                    self.save_state()
                    return True
                    
            logger.warning(f"Invalid content format for CV update: {type(new_content)}")
            return False
            
        except Exception as e:
            logger.error(f"Failed to update CV data: {str(e)}")
            logger.error(f"Content keys: {list(new_content.keys()) if isinstance(new_content, dict) else 'Not a dict'}")
            return False

    def _filter_content_data_args(self, content_dict: dict) -> dict:
        """
        Filter content dictionary to only include valid ContentData arguments.
        
        Args:
            content_dict: Dictionary containing content data
            
        Returns:
            Filtered dictionary with only valid ContentData arguments
        """
        # Valid ContentData constructor arguments
        valid_args = {
            'summary', 'experience_bullets', 'skills_section', 'projects',
            'other_content', 'name', 'email', 'phone', 'linkedin', 'github',
            'education', 'certifications', 'languages'
        }
        
        # Filter the content to only include valid arguments
        filtered = {k: v for k, v in content_dict.items() if k in valid_args}
        
        # Log what was filtered out for debugging
        filtered_out = {k: v for k, v in content_dict.items() if k not in valid_args}
        if filtered_out:
            logger.info(f"Filtered out non-ContentData arguments: {list(filtered_out.keys())}")
        
        return filtered

    def _populate_structured_cv_from_content(self, content_data: ContentData):
        """
        Populate the StructuredCV with content from ContentData.
        
        Args:
            content_data: ContentData instance containing the CV content
        """
        if not self._structured_cv:
            logger.error("Cannot populate: No StructuredCV instance exists")
            return
            
        # Clear existing sections
        self._structured_cv.sections = []
        
        # Add Executive Summary section
        if content_data.get("summary"):
            summary_section = Section(
                id="executive_summary",
                name="Executive Summary",
                status=ItemStatus.GENERATED
            )
            summary_item = Item(
                id="summary_item",
                content=content_data.get("summary", ""),
                status=ItemStatus.GENERATED
            )
            summary_section.items.append(summary_item)
            self._structured_cv.sections.append(summary_section)
            
        # Add Professional Experience section
        if content_data.get("experience_bullets"):
            experience_section = Section(
                id="professional_experience",
                name="Professional Experience",
                status=ItemStatus.GENERATED
            )
            
            for i, exp in enumerate(content_data.get("experience_bullets", [])):
                if isinstance(exp, dict):
                    # Structured experience entry
                    exp_subsection = Subsection(
                        id=f"experience_{i}",
                        name=exp.get("position", f"Experience {i+1}"),
                        metadata={
                            "company": exp.get("company", ""),
                            "location": exp.get("location", ""),
                            "period": exp.get("period", "")
                        }
                    )
                    
                    # Add bullets as items
                    for j, bullet in enumerate(exp.get("bullets", [])):
                        bullet_item = Item(
                            id=f"experience_{i}_bullet_{j}",
                            content=bullet,
                            status=ItemStatus.GENERATED
                        )
                        exp_subsection.items.append(bullet_item)
                    
                    experience_section.subsections.append(exp_subsection)
                else:
                    # Simple string experience
                    exp_item = Item(
                        id=f"experience_item_{i}",
                        content=str(exp),
                        status=ItemStatus.GENERATED
                    )
                    experience_section.items.append(exp_item)
                    
            self._structured_cv.sections.append(experience_section)
            
        # Add Key Qualifications/Skills section
        if content_data.get("skills_section"):
            skills_section = Section(
                id="key_qualifications",
                name="Key Qualifications",
                status=ItemStatus.GENERATED
            )
            skills_item = Item(
                id="skills_item",
                content=content_data.get("skills_section", ""),
                status=ItemStatus.GENERATED
            )
            skills_section.items.append(skills_item)
            self._structured_cv.sections.append(skills_section)
            
        # Add Projects section
        if content_data.get("projects"):
            projects_section = Section(
                id="project_experience",
                name="Project Experience",
                status=ItemStatus.GENERATED
            )
            
            for i, project in enumerate(content_data.get("projects", [])):
                if isinstance(project, dict):
                    project_subsection = Subsection(
                        id=f"project_{i}",
                        name=project.get("name", f"Project {i+1}"),
                        metadata={
                            "description": project.get("description", ""),
                            "technologies": project.get("technologies", [])
                        }
                    )
                    
                    # Add project bullets as items
                    for j, bullet in enumerate(project.get("bullets", [])):
                        bullet_item = Item(
                            id=f"project_{i}_bullet_{j}",
                            content=bullet,
                            status=ItemStatus.GENERATED
                        )
                        project_subsection.items.append(bullet_item)
                    
                    projects_section.subsections.append(project_subsection)
                else:
                    # Simple string project
                    project_item = Item(
                        id=f"project_item_{i}",
                        content=str(project),
                        status=ItemStatus.GENERATED
                    )
                    projects_section.items.append(project_item)
                    
            self._structured_cv.sections.append(projects_section)
            
        # Add Education section
        if content_data.get("education"):
            education_section = Section(
                id="education",
                name="Education",
                status=ItemStatus.GENERATED
            )
            
            for i, edu in enumerate(content_data.get("education", [])):
                if isinstance(edu, dict):
                    edu_subsection = Subsection(
                        id=f"education_{i}",
                        name=edu.get("degree", f"Education {i+1}"),
                        metadata={
                            "institution": edu.get("institution", ""),
                            "location": edu.get("location", ""),
                            "period": edu.get("period", "")
                        }
                    )
                    
                    # Add education details as items
                    for j, detail in enumerate(edu.get("details", [])):
                        detail_item = Item(
                            id=f"education_{i}_detail_{j}",
                            content=detail,
                            status=ItemStatus.GENERATED
                        )
                        edu_subsection.items.append(detail_item)
                    
                    education_section.subsections.append(edu_subsection)
                else:
                    # Simple string education
                    edu_item = Item(
                        id=f"education_item_{i}",
                        content=str(edu),
                        status=ItemStatus.GENERATED
                    )
                    education_section.items.append(edu_item)
                    
            self._structured_cv.sections.append(education_section)
            
        # Update personal info in metadata
        if not self._structured_cv.metadata:
            self._structured_cv.metadata = {}
            
        self._structured_cv.metadata.update({
            "name": content_data.get("name", ""),
            "email": content_data.get("email", ""),
            "phone": content_data.get("phone", ""),
            "linkedin": content_data.get("linkedin", ""),
            "github": content_data.get("github", "")
        })
        
        logger.info(f"Populated StructuredCV with {len(self._structured_cv.sections)} sections")
