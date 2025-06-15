"""Core Pydantic models for the AI CV Generator.

This module defines the strict data contracts for the application's primary
data structures, such as the StructuredCV and JobDescriptionData. These models
ensure data consistency, validation, and clarity across all components, from
parsing and generation to state management and API serialization.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from dataclasses import dataclass
from enum import Enum
from dataclasses import field
from uuid import UUID, uuid4
from pydantic import HttpUrl


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


class Item(BaseModel):
    """A granular piece of content within the CV (e.g., a bullet point)."""
    id: UUID = Field(default_factory=uuid4)
    content: str
    status: ItemStatus = ItemStatus.INITIAL
    item_type: ItemType = ItemType.BULLET_POINT
    raw_llm_output: Optional[str] = Field(default=None, description="Raw LLM output for this item for transparency and debugging")  # REQ-FUNC-UI-6
    confidence_score: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
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
    metadata: Dict[str, Any] = Field(default_factory=dict)
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
    metadata: Dict[str, Any] = Field(default_factory=dict)  # e.g., dates, company, location


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
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # New fields for "Big 10" Skills (Task 3.2)
    big_10_skills: List[str] = Field(default_factory=list, description="Top 10 most relevant skills extracted from CV and job description")
    big_10_skills_raw_output: Optional[str] = Field(default=None, description="Raw LLM output for Big 10 skills generation for transparency")
    
    def find_item_by_id(self, item_id: str) -> tuple[Optional[Item], Optional[Section], Optional[Subsection]]:
        """Find an item by its ID and return the item along with its parent section and subsection.
        
        Returns:
            tuple: (item, section, subsection) where subsection is None if the item is directly in a section
        """
        for section in self.sections:
            # Check items directly in the section
            for item in section.items:
                if str(item.id) == item_id:
                    return item, section, None
            
            # Check items in subsections
            for subsection in section.subsections:
                for item in subsection.items:
                    if str(item.id) == item_id:
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
        content_data = {
            "personal_info": self.metadata.get("personal_info", {}),
            "executive_summary": [],
            "professional_experience": [],
            "key_qualifications": [],
            "projects": [],
            "education": []
        }
        
        for section in self.sections:
            section_name = section.name.lower().replace(" ", "_")
            
            if "executive" in section_name or "summary" in section_name:
                content_data["executive_summary"] = [item.content for item in section.items]
            elif "experience" in section_name or "employment" in section_name:
                content_data["professional_experience"] = [item.content for item in section.items]
            elif "qualification" in section_name or "skill" in section_name:
                content_data["key_qualifications"] = [item.content for item in section.items]
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
            
            # Update personal info in metadata
            if "personal_info" in content_data:
                self.metadata["personal_info"] = content_data["personal_info"]
            
            # Clear existing sections
            self.sections.clear()
            
            # Recreate sections from content_data
            section_mappings = {
                "executive_summary": "Executive Summary",
                "professional_experience": "Professional Experience", 
                "key_qualifications": "Key Qualifications",
                "projects": "Projects",
                "education": "Education"
            }
            
            for content_key, section_name in section_mappings.items():
                if content_key in content_data and content_data[content_key]:
                    section = Section(
                        name=section_name,
                        items=[
                            Item(content=content, status=ItemStatus.INITIAL)
                            for content in content_data[content_key]
                            if content  # Skip empty content
                        ]
                    )
                    if section.items:  # Only add section if it has items
                        self.sections.append(section)
            
            return True
        except Exception:
            return False


class JobDescriptionData(BaseModel):
    """A structured representation of a parsed job description."""
    raw_text: str
    skills: List[str] = Field(default_factory=list)
    experience_level: Optional[str] = None
    responsibilities: List[str] = Field(default_factory=list)
    industry_terms: List[str] = Field(default_factory=list)
    company_values: List[str] = Field(default_factory=list)
    error: Optional[str] = None



# Legacy models for backward compatibility during transition
# These will be removed once all components are updated to use the new models

class ProcessingStatus(Enum):
    """Status of processing for individual items or sections."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RATE_LIMITED = "rate_limited"


@dataclass
class ProcessingMetadata:
    """Metadata for processing items."""
    item_id: str
    status: 'ProcessingStatus'
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

@dataclass
class ProcessingQueue:
    """Queue for managing content processing."""
    pending_items: List[Item] = field(default_factory=list)
    in_progress_items: List[Item] = field(default_factory=list)
    completed_items: List[Item] = field(default_factory=list)
    failed_items: List[Item] = field(default_factory=list)
    
    def add_item(self, item: Item):
        """Add an item to the pending queue."""
        self.pending_items.append(item)
    
    def get_next_item(self) -> Optional[Item]:
        """Get the next item ready for processing."""
        # Sort by priority (higher first) and creation time
        ready_items = [
            item for item in self.pending_items 
            if item.is_ready_for_processing
        ]
        
        if not ready_items:
            return None
        
        # Sort by priority (descending) then by creation time (ascending)
        ready_items.sort(
            key=lambda x: (-x.priority, x.metadata.created_at)
        )
        
        item = ready_items[0]
        self.pending_items.remove(item)
        self.in_progress_items.append(item)
        item.metadata.update_status(ProcessingStatus.IN_PROGRESS)
        
        return item
    
    def complete_item(self, item: Item, generated_content: str):
        """Mark an item as completed."""
        item.content = generated_content
        item.status = ItemStatus.GENERATED
        
        if item in self.in_progress_items:
            self.in_progress_items.remove(item)
        self.completed_items.append(item)
    
    def fail_item(self, item: Item, error: str):
        """Mark an item as failed."""
        item.status = ItemStatus.FAILED
        item.metadata["error"] = error
        
        if item in self.in_progress_items:
            self.in_progress_items.remove(item)
        self.failed_items.append(item)
    
    def rate_limit_item(self, item: Item):
        """Mark an item as rate limited and return to pending."""
        item.status = ItemStatus.RATE_LIMITED
        item.metadata["rate_limit_hits"] = item.metadata.get("rate_limit_hits", 0) + 1
        
        if item in self.in_progress_items:
            self.in_progress_items.remove(item)
        self.pending_items.append(item)
    
    @property
    def total_items(self) -> int:
        """Total number of items in all queues."""
        return (
            len(self.pending_items) + 
            len(self.in_progress_items) + 
            len(self.completed_items) + 
            len(self.failed_items)
        )
    
    @property
    def completion_percentage(self) -> float:
        """Percentage of items completed."""
        if self.total_items == 0:
            return 0.0
        return (len(self.completed_items) / self.total_items) * 100


@dataclass
class CVGenerationState:
    """Complete state for CV generation workflow."""
    session_id: str = field(default_factory=lambda: str(uuid4()))
    current_stage: WorkflowStage = WorkflowStage.INITIALIZATION
    job_description: Optional[JobDescriptionData] = None
    
    # Individual processing queues
    qualification_queue: ProcessingQueue = field(default_factory=ProcessingQueue)
    experience_queue: ProcessingQueue = field(default_factory=ProcessingQueue)
    project_queue: ProcessingQueue = field(default_factory=ProcessingQueue)
    
    # Generated content
    key_qualifications: List[Item] = field(default_factory=list)
    professional_experiences: List[Item] = field(default_factory=list)
    side_projects: List[Item] = field(default_factory=list)
    executive_summary: Optional[Item] = None
    
    # Workflow metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    total_processing_time: float = 0.0
    total_llm_calls: int = 0
    total_tokens_used: int = 0
    total_rate_limit_hits: int = 0
    
    # Configuration
    target_qualifications_count: int = 10
    max_retry_attempts: int = 3
    rate_limit_backoff_seconds: float = 60.0
    
    def update_stage(self, new_stage: WorkflowStage):
        """Update the current workflow stage."""
        self.current_stage = new_stage
        self.updated_at = datetime.now()
    
    def add_qualification_items(self, items: List[str]):
        """Add qualification items to the processing queue."""
        for i, item_text in enumerate(items):
            qualification = Item(
                content=item_text,
                item_type=ItemType.KEY_QUALIFICATION,
                metadata={"priority": len(items) - i}  # Earlier items have higher priority
            )
            self.qualification_queue.add_item(qualification)
    
    def add_experience_items(self, experiences: List[Dict[str, Any]]):
        """Add experience items to the processing queue."""
        for i, exp_data in enumerate(experiences):
            experience = ExperienceItem(
                content_type=ContentType.EXPERIENCE_ITEM,
                original_content=exp_data.get('description', ''),
                company=exp_data.get('company', ''),
                position=exp_data.get('position', ''),
                duration=exp_data.get('duration', ''),
                responsibilities=exp_data.get('responsibilities', []),
                achievements=exp_data.get('achievements', []),
                technologies=exp_data.get('technologies', []),
                priority=len(experiences) - i
            )
            self.experience_queue.add_item(experience)
    
    def add_project_items(self, projects: List[Dict[str, Any]]):
        """Add project items to the processing queue."""
        for i, proj_data in enumerate(projects):
            project = ProjectItem(
                content_type=ContentType.PROJECT_ITEM,
                original_content=proj_data.get('description', ''),
                name=proj_data.get('name', ''),
                description=proj_data.get('description', ''),
                technologies=proj_data.get('technologies', []),
                achievements=proj_data.get('achievements', []),
                url=proj_data.get('url'),
                priority=len(projects) - i
            )
            self.project_queue.add_item(project)
    
    @property
    def overall_progress(self) -> Dict[str, Any]:
        """Get overall progress statistics."""
        total_items = (
            self.qualification_queue.total_items +
            self.experience_queue.total_items +
            self.project_queue.total_items
        )
        
        completed_items = (
            len(self.qualification_queue.completed_items) +
            len(self.experience_queue.completed_items) +
            len(self.project_queue.completed_items)
        )
        
        return {
            "total_items": total_items,
            "completed_items": completed_items,
            "completion_percentage": (completed_items / total_items * 100) if total_items > 0 else 0,
            "current_stage": self.current_stage.value,
            "qualifications_progress": self.qualification_queue.completion_percentage,
            "experience_progress": self.experience_queue.completion_percentage,
            "projects_progress": self.project_queue.completion_percentage,
            "total_processing_time": self.total_processing_time,
            "total_llm_calls": self.total_llm_calls,
            "total_tokens_used": self.total_tokens_used,
            "rate_limit_hits": self.total_rate_limit_hits
        }
    
    @property
    def is_complete(self) -> bool:
        """Check if all processing is complete."""
        return (
            self.current_stage == WorkflowStage.COMPLETED and
            self.qualification_queue.completion_percentage == 100 and
            self.experience_queue.completion_percentage == 100 and
            self.project_queue.completion_percentage == 100 and
            self.executive_summary is not None and
            self.executive_summary.metadata.status == ProcessingStatus.COMPLETED
        )
    
    def get_final_cv(self) -> Dict[str, Any]:
        """Get the final CV content as a dictionary."""
        return {
            "executive_summary": self.executive_summary.generated_content if self.executive_summary else "",
            "key_qualifications": [q.generated_content for q in self.key_qualifications if q.generated_content],
            "professional_experiences": [{
                "company": exp.company,
                "position": exp.position,
                "duration": exp.duration,
                "content": exp.generated_content
            } for exp in self.professional_experiences if exp.generated_content],
            "side_projects": [{
                "name": proj.name,
                "description": proj.description,
                "content": proj.generated_content
            } for proj in self.side_projects if proj.generated_content],
            "metadata": {
                "session_id": self.session_id,
                "created_at": self.created_at.isoformat(),
                "total_tokens_used": self.total_tokens_used,
                "total_llm_calls": self.total_llm_calls
            }
        }


@dataclass
class RateLimitState:
    """State for tracking rate limits across different models."""
    model_name: str
    requests_per_minute: int = 0
    tokens_per_minute: int = 0
    window_start: datetime = field(default_factory=datetime.now)
    last_request_time: datetime = field(default_factory=datetime.now)
    consecutive_failures: int = 0
    backoff_until: Optional[datetime] = None
    max_requests_per_minute: int = 30
    max_tokens_per_minute: int = 50000
    
    def can_make_request(self, estimated_tokens: int = 0) -> bool:
        """Check if a request can be made given current rate limits."""
        now = datetime.now()
        
        # Check if we're in a backoff period
        if self.backoff_until and now < self.backoff_until:
            return False
        
        # Reset window if it's been more than a minute
        if (now - self.window_start).total_seconds() >= 60:
            self.requests_per_minute = 0
            self.tokens_per_minute = 0
            self.window_start = now
        
        # Check rate limits using configured limits
        return (
            self.requests_per_minute < self.max_requests_per_minute and
            (self.tokens_per_minute + estimated_tokens) < self.max_tokens_per_minute
        )
    
    def record_request(self, tokens_used: int, success: bool):
        """Record a request and update rate limit state."""
        now = datetime.now()
        
        # Reset window if needed
        if (now - self.window_start).total_seconds() >= 60:
            self.requests_per_minute = 0
            self.tokens_per_minute = 0
            self.window_start = now
        
        self.requests_per_minute += 1
        self.tokens_per_minute += tokens_used
        self.last_request_time = now
        
        if success:
            self.consecutive_failures = 0
            self.backoff_until = None
        else:
            self.consecutive_failures += 1
            # Exponential backoff
            backoff_seconds = min(300, 30 * (2 ** self.consecutive_failures))
            self.backoff_until = now + datetime.timedelta(seconds=backoff_seconds)


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
    metadata: Dict[str, Any] = Field(default_factory=dict)


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
    metadata: Dict[str, Any] = Field(default_factory=dict)


class WorkflowState(BaseModel):
    """Data model for tracking workflow state."""
    current_stage: WorkflowStage = WorkflowStage.INITIALIZATION
    completed_stages: List[WorkflowStage] = Field(default_factory=list)
    failed_stages: List[WorkflowStage] = Field(default_factory=list)
    session_id: str = Field(default_factory=lambda: str(uuid4()))
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def advance_to_stage(self, stage: WorkflowStage):
        """Advance workflow to a new stage."""
        if self.current_stage not in self.completed_stages:
            self.completed_stages.append(self.current_stage)
        self.current_stage = stage
        self.updated_at = datetime.now()
    
    def mark_stage_failed(self, stage: WorkflowStage):
        """Mark a stage as failed."""
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
    metadata: Dict[str, Any] = Field(default_factory=dict)


class VectorStoreConfig(BaseModel):
    """Configuration for vector store database."""
    collection_name: str = "cv_content"
    persist_directory: str = "data/vector_store"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    dimension: int = 768
    index_type: str = "IndexFlatL2"