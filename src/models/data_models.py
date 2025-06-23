"""Core Pydantic models for the AI CV Generator.

This module defines the strict data contracts for the application's primary
data structures, such as the StructuredCV and JobDescriptionData. These models
ensure data consistency, validation, and clarity across all components, from
parsing and generation to state management and API serialization.
"""

from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from dataclasses import dataclass
from enum import Enum
from uuid import UUID, uuid4


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


class ProcessingStatus(str, Enum):
    """Enumeration for processing status - backward compatibility alias for ItemStatus."""

    INITIAL = "initial"
    GENERATED = "generated"
    USER_MODIFIED = "user_modified"
    USER_ACCEPTED = "user_accepted"
    TO_REGENERATE = "to_regenerate"
    GENERATION_FAILED = "generation_failed"
    GENERATED_FALLBACK = "generated_fallback"
    STATIC = "static"

    # Additional processing-specific statuses
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


class WorkflowStage(str, Enum):
    """Enumeration for workflow stages."""

    INITIALIZATION = "initialization"
    CV_PARSING = "cv_parsing"
    JOB_ANALYSIS = "job_analysis"
    CONTENT_GENERATION = "content_generation"
    REVIEW = "review"
    COMPLETED = "completed"
    FAILED = "failed"


class ContentType(str, Enum):
    """Enumeration for content types."""

    QUALIFICATION = "qualification"
    EXPERIENCE = "experience"
    EXPERIENCE_ITEM = "experience_item"
    PROJECT = "project"
    PROJECT_ITEM = "project_item"
    EXECUTIVE_SUMMARY = "executive_summary"
    SKILL = "skill"
    SKILLS = "skills"
    ACHIEVEMENT = "achievement"
    EDUCATION = "education"
    PROJECTS = "projects"
    ANALYSIS = "analysis"
    QUALITY_CHECK = "quality_check"
    OPTIMIZATION = "optimization"

    CV_ANALYSIS = "cv_analysis"
    CV_PARSING = "cv_parsing"
    ACHIEVEMENTS = "achievements"


class UserAction(str, Enum):
    """Enumeration for user actions in the UI."""

    ACCEPT = "accept"
    REGENERATE = "regenerate"


class UserFeedback(BaseModel):
    """User feedback for item review."""

    action: UserAction
    item_id: str
    feedback_text: Optional[str] = None
    rating: Optional[int] = Field(None, ge=1, le=5)
    timestamp: datetime = Field(default_factory=datetime.now)


class WorkflowType(Enum):
    """Types of predefined workflows."""

    BASIC_CV_GENERATION = "basic_cv_generation"
    JOB_TAILORED_CV = "job_tailored_cv"
    CV_OPTIMIZATION = "cv_optimization"
    QUALITY_ASSURANCE = "quality_assurance"
    COMPREHENSIVE_CV = "comprehensive_cv"
    QUICK_UPDATE = "quick_update"
    MULTI_LANGUAGE_CV = "multi_language_cv"
    INDUSTRY_SPECIFIC = "industry_specific"


class MetadataModel(BaseModel):
    # Extend this model as needed for common metadata fields
    item_id: Optional[str] = None  # Added item_id for test and runtime compatibility
    company: Optional[str] = None
    position: Optional[str] = None
    location: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    status: Optional[ProcessingStatus] = None  # Added status field for item processing
    processing_time_seconds: Optional[float] = None  # For timing info
    tokens_used: Optional[int] = None  # For LLM usage info
    # Add more fields as required by the application
    extra: Optional[dict] = Field(default_factory=dict)

    def update_status(
        self, status: ProcessingStatus, error_message: Optional[str] = None
    ):
        self.status = status
        if error_message is not None:
            self.extra["error_message"] = error_message


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

    content_type: str = Field(default="experience_item")
    company: str = ""
    position: str = ""
    duration: str = ""
    responsibilities: List[str] = Field(default_factory=list)
    achievements: List[str] = Field(default_factory=list)
    technologies: List[str] = Field(default_factory=list)


class ProjectItem(ContentItem):
    """Represents a project item for processing."""

    content_type: str = Field(default="project_item")
    name: str = ""
    description: str = ""
    technologies: List[str] = Field(default_factory=list)
    achievements: List[str] = Field(default_factory=list)
    url: Optional[str] = None


class Subsection(BaseModel):
    """A subsection within a section (e.g., a specific job role)."""

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
    big_10_skills_raw_output: Optional[str] = Field(
        default=None,
        description="Raw LLM output for Big 10 skills generation for transparency",
    )

    def find_item_by_id(
        self, item_id: str
    ) -> tuple[Optional[Item], Optional[Section], Optional[Subsection]]:
        """Find an item by its ID and return the item along with its parent section and subsection.

        Searches by both the actual item.id and the metadata.item_id as fallback.

        Returns:
            tuple: (item, section, subsection) where subsection is None if the item is directly in a section
        """
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

    def to_content_data(self) -> Dict[str, Any]:
        """Convert StructuredCV to ContentData format for backward compatibility."""
        # Ensure self.metadata is a MetadataModel instance
        meta = (
            self.metadata
            if isinstance(self.metadata, MetadataModel)
            else MetadataModel()
        )
        content_data = {
            "personal_info": getattr(meta, "extra", {}).get("personal_info", {}),
            "executive_summary": [],
            "professional_experience": [],
            "key_qualifications": [],
            "projects": [],
            "education": [],
        }

        for section in self.sections:
            section_name = section.name.lower().replace(" ", "_")

            if "executive" in section_name or "summary" in section_name:
                content_data["executive_summary"] = [
                    item.content for item in section.items
                ]
            elif "experience" in section_name or "employment" in section_name:
                content_data["professional_experience"] = [
                    item.content for item in section.items
                ]
            elif "qualification" in section_name or "skill" in section_name:
                content_data["key_qualifications"] = [
                    item.content for item in section.items
                ]
            elif "project" in section_name:
                content_data["projects"] = [item.content for item in section.items]
            elif "education" in section_name:
                content_data["education"] = [item.content for item in section.items]

        return content_data

    def update_from_content(self, content_data: Dict[str, Any]) -> bool:
        """Update StructuredCV from ContentData format."""
        try:
            # Validate input is a dictionary
            if not isinstance(content_data, dict):
                return False

            # Update personal info in metadata.extra
            if "personal_info" in content_data:
                meta = (
                    self.metadata
                    if isinstance(self.metadata, MetadataModel)
                    else MetadataModel()
                )
                meta.extra["personal_info"] = content_data["personal_info"]
                self.metadata = meta

            # Clear existing sections
            self.sections.clear()  # pylint: disable=no-member

            # Recreate sections from content_data
            section_mappings = {
                "executive_summary": "Executive Summary",
                "professional_experience": "Professional Experience",
                "key_qualifications": "Key Qualifications",
                "projects": "Projects",
                "education": "Education",
            }

            for content_key, section_name in section_mappings.items():
                if content_key in content_data and content_data[content_key]:
                    section = Section(
                        name=section_name,
                        items=[
                            Item(content=content, status=ItemStatus.INITIAL)
                            for content in content_data[content_key]
                            if content  # Skip empty content
                        ],
                    )
                    if section.items:  # Only add section if it has items
                        self.sections.append(section)  # pylint: disable=no-member

            return True
        except Exception as e:
            # Optionally log e
            return False


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


# Missing models that are imported by state_manager.py
class ContentData(BaseModel):
    """Data model for CV content used in template rendering and state management."""

    summary: Optional[str] = None
    experience_bullets: Optional[List[str]] = Field(default_factory=list)
    skills_section: Optional[str] = None
    projects: Optional[List[str]] = Field(default_factory=list)
    other_content: Optional[str] = None
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    linkedin: Optional[str] = None
    github: Optional[str] = None
    education: Optional[List[str]] = Field(default_factory=list)
    certifications: Optional[List[str]] = Field(default_factory=list)
    languages: Optional[List[str]] = Field(default_factory=list)


class CVData(BaseModel):
    """Data model for CV data used in analysis and processing."""

    raw_text: str
    parsed_sections: Dict[str, Any] = Field(default_factory=dict)
    skills: List[str] = Field(default_factory=list)
    experience: List[str] = Field(default_factory=list)
    education: List[str] = Field(default_factory=list)
    projects: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


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


class WorkflowState(BaseModel):
    """Data model for tracking workflow state."""

    current_stage: WorkflowStage = WorkflowStage.INITIALIZATION
    completed_stages: List[WorkflowStage] = Field(default_factory=list)
    failed_stages: List[WorkflowStage] = Field(default_factory=list)
    session_id: str = Field(default_factory=lambda: str(uuid4()))
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    metadata: MetadataModel = Field(default_factory=MetadataModel)

    def advance_to_stage(self, stage: WorkflowStage):
        """Advance workflow to a new stage."""
        # Ensure completed_stages is a list, not FieldInfo
        if not isinstance(self.completed_stages, list):
            self.completed_stages = []
        if self.current_stage not in self.completed_stages:
            self.completed_stages.append(self.current_stage)
        self.current_stage = stage
        self.updated_at = datetime.now()

    def mark_stage_failed(self, stage: WorkflowStage):
        """Mark a stage as failed."""
        # Ensure failed_stages is a list, not FieldInfo
        if not isinstance(self.failed_stages, list):
            self.failed_stages = []
        if stage not in self.failed_stages:
            self.failed_stages.append(stage)
        self.updated_at = datetime.now()


class AgentIO(BaseModel):
    """Data model for agent input/output schema definition."""

    description: str
    required_fields: List[str] = Field(default_factory=list)
    optional_fields: List[str] = Field(default_factory=list)
    input: Optional[Dict[str, Any]] = Field(default_factory=dict)
    output: Optional[Dict[str, Any]] = Field(default_factory=dict)
    metadata: MetadataModel = Field(default_factory=MetadataModel)


class VectorStoreConfig(BaseModel):
    """Configuration for vector store database."""

    collection_name: str = "cv_content"
    persist_directory: str = "data/vector_store"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    dimension: int = 768
    index_type: str = "IndexFlatL2"


# Temporary model for CV parsing LLM output
class CVParsingPersonalInfo(BaseModel):
    """Personal information extracted from CV parsing."""

    name: str
    email: str
    phone: str
    linkedin: Optional[str] = None
    github: Optional[str] = None
    location: Optional[str] = None


class CVParsingSubsection(BaseModel):
    """Subsection structure for CV parsing output."""

    name: str
    items: List[str] = Field(default_factory=list)


class CVParsingSection(BaseModel):
    """Section structure for CV parsing output."""

    name: str
    items: List[str] = Field(default_factory=list)
    subsections: List[CVParsingSubsection] = Field(default_factory=list)


class CVParsingResult(BaseModel):
    """Complete CV parsing result from LLM."""

    personal_info: CVParsingPersonalInfo
    sections: List[CVParsingSection]


# API Model backward compatibility aliases
class PersonalInfo(BaseModel):
    """Personal information model."""

    name: str
    email: str
    phone: Optional[str] = None
    address: Optional[str] = None
    linkedin: Optional[str] = None
    website: Optional[str] = None
    summary: Optional[str] = None


class Experience(BaseModel):
    """Work experience model."""

    title: str
    company: str
    location: Optional[str] = None
    start_date: str
    end_date: Optional[str] = None
    current: bool = False
    description: Optional[str] = None
    achievements: Optional[List[str]] = None
    technologies: Optional[List[str]] = None


class Education(BaseModel):
    """Education model."""

    degree: str
    institution: str
    location: Optional[str] = None
    graduation_date: Optional[str] = None
    gpa: Optional[str] = None
    honors: Optional[List[str]] = None
    relevant_coursework: Optional[List[str]] = None


class Project(BaseModel):
    """Project model."""

    name: str
    description: str
    technologies: Optional[List[str]] = None
    url: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    achievements: Optional[List[str]] = None


class Skill(BaseModel):
    """Skill model."""

    name: str
    level: Optional[str] = None
    category: Optional[str] = None
    years_experience: Optional[int] = None


class Certification(BaseModel):
    """Certification model."""

    name: str
    issuer: str
    date_obtained: Optional[str] = None
    expiry_date: Optional[str] = None
    credential_id: Optional[str] = None
    url: Optional[str] = None


class Language(BaseModel):
    """Language model."""

    name: str
    proficiency: Optional[str] = None
    native: bool = False


@dataclass
class RateLimitState:
    """State tracking for rate limiting per model."""

    model: str
    requests_made: int = 0
    requests_limit: int = 60
    tokens_made: int = 0
    tokens_limit: int = 1000000
    window_start: datetime = None
    window_duration: timedelta = None
    last_request_time: Optional[datetime] = None
    backoff_until: Optional[datetime] = None
    consecutive_failures: int = 0

    @property
    def requests_per_minute(self) -> int:
        """Alias for requests_made for backward compatibility."""
        return self.requests_made

    @property
    def tokens_per_minute(self) -> int:
        """Alias for tokens_made for backward compatibility."""
        return self.tokens_made

    def __post_init__(self):
        """Initialize default values after dataclass creation."""
        if self.window_start is None:
            self.window_start = datetime.now()
        if self.window_duration is None:
            self.window_duration = timedelta(minutes=1)

    def is_rate_limited(self) -> bool:
        """Check if currently rate limited."""
        now = datetime.now()

        # Check if in backoff period
        if self.backoff_until and now < self.backoff_until:
            return True

        # Check if window has reset
        if now - self.window_start > self.window_duration:
            self.requests_made = 0
            self.window_start = now

        return self.requests_made >= self.requests_limit

    def record_request(self, tokens_used: int = 0):
        """Record a new request."""
        now = datetime.now()

        # Reset window if needed
        if now - self.window_start > self.window_duration:
            self.requests_made = 0
            self.tokens_made = 0
            self.window_start = now

        self.requests_made += 1
        self.tokens_made += tokens_used
        self.last_request_time = now

    def record_failure(self):
        """Record a request failure."""
        self.consecutive_failures += 1
        # Exponential backoff
        backoff_seconds = min(300, 2**self.consecutive_failures)  # Max 5 minutes
        self.backoff_until = datetime.now() + timedelta(seconds=backoff_seconds)

    def record_success(self):
        """Record a successful request."""
        self.consecutive_failures = 0
        self.backoff_until = None

    def can_make_request(self, estimated_tokens: int = 0) -> bool:
        """Check if a request can be made given current rate limits."""
        if self.is_rate_limited():
            return False
        return True


# Structured Logging Data Models
@dataclass
class AgentExecutionLog:
    """Structured log entry for agent execution tracking."""

    timestamp: str
    agent_name: str
    session_id: str
    item_id: Optional[str]
    content_type: Optional[str]
    execution_phase: str  # 'start', 'success', 'error', 'retry'
    processing_time_seconds: Optional[float] = None
    confidence_score: Optional[float] = None
    retry_count: int = 0
    error_message: Optional[str] = None
    input_data_type: Optional[str] = None
    output_data_size: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class AgentDecisionLog:
    """Structured log entry for agent decision tracking."""

    timestamp: str
    agent_name: str
    session_id: str
    item_id: Optional[str]
    decision_type: str  # 'validation', 'processing', 'fallback', 'enhancement'
    decision_details: str
    confidence_score: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class AgentPerformanceLog:
    """Structured log entry for agent performance metrics."""

    timestamp: str
    agent_name: str
    session_id: str
    metric_type: str  # 'execution_time', 'success_rate', 'error_rate', 'throughput'
    metric_value: float
    time_window: str  # 'session', 'hour', 'day'
    metadata: Optional[Dict[str, Any]] = None


class LLMResponse(BaseModel):
    """Structured response from LLM calls."""

    content: str
    tokens_used: int = 0
    processing_time: float = 0.0
    model_used: str = ""
    success: bool = True
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ErrorFallbackModel(BaseModel):
    error: str
