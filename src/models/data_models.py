"""Enhanced data models for the AI CV Generator MVP.

This module defines the core data structures for managing CV generation workflow,
with support for individual item processing to mitigate LLM rate limits.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from uuid import uuid4


class ProcessingStatus(Enum):
    """Status of processing for individual items or sections."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RATE_LIMITED = "rate_limited"


class WorkflowStage(Enum):
    """Stages in the CV generation workflow."""
    INITIALIZATION = "initialization"
    JOB_PARSING = "job_parsing"
    QUALIFICATION_GENERATION = "qualification_generation"
    EXPERIENCE_PROCESSING = "experience_processing"
    PROJECT_PROCESSING = "project_processing"
    SUMMARY_GENERATION = "summary_generation"
    CV_ASSEMBLY = "cv_assembly"
    PDF_GENERATION = "pdf_generation"
    COMPLETED = "completed"
    ERROR = "error"


class ContentType(Enum):
    """Types of content that can be generated."""
    QUALIFICATION = "qualification"
    EXPERIENCE = "experience"
    EXPERIENCE_ITEM = "experience_item"
    PROJECT = "project"
    PROJECT_ITEM = "project_item"
    PROJECTS = "projects"
    EXECUTIVE_SUMMARY = "executive_summary"
    SKILL = "skill"
    ACHIEVEMENT = "achievement"
    EDUCATION = "education"
    SKILLS = "skills"
    ANALYSIS = "analysis"
    QUALITY_CHECK = "quality_check"
    OPTIMIZATION = "optimization"


@dataclass
class ProcessingMetadata:
    """Metadata for tracking processing of individual items."""
    item_id: str = field(default_factory=lambda: str(uuid4()))
    status: ProcessingStatus = ProcessingStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    processing_attempts: int = 0
    last_error: Optional[str] = None
    processing_time_seconds: float = 0.0
    llm_calls_made: int = 0
    tokens_used: int = 0
    rate_limit_hits: int = 0

    def update_status(self, new_status: ProcessingStatus, error: Optional[str] = None):
        """Update the processing status and metadata."""
        self.status = new_status
        self.updated_at = datetime.now()
        if error:
            self.last_error = error
            self.processing_attempts += 1


@dataclass
class ContentItem:
    """Individual content item with processing metadata."""
    content_type: ContentType
    original_content: str
    generated_content: Optional[str] = None
    metadata: ProcessingMetadata = field(default_factory=ProcessingMetadata)
    context: Dict[str, Any] = field(default_factory=dict)
    priority: int = 1  # Higher number = higher priority
    dependencies: List[str] = field(default_factory=list)  # Item IDs this depends on

    @property
    def is_ready_for_processing(self) -> bool:
        """Check if item is ready for processing based on dependencies."""
        return self.metadata.status == ProcessingStatus.PENDING

    @property
    def needs_retry(self) -> bool:
        """Check if item needs retry due to failure or rate limiting."""
        return self.metadata.status in [ProcessingStatus.FAILED, ProcessingStatus.RATE_LIMITED]


@dataclass
class ExperienceItem(ContentItem):
    """Professional experience item."""
    company: str = ""
    position: str = ""
    duration: str = ""
    responsibilities: List[str] = field(default_factory=list)
    achievements: List[str] = field(default_factory=list)
    technologies: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if self.content_type != ContentType.EXPERIENCE_ITEM:
            self.content_type = ContentType.EXPERIENCE_ITEM


@dataclass
class ProjectItem(ContentItem):
    """Side project item."""
    name: str = ""
    description: str = ""
    technologies: List[str] = field(default_factory=list)
    achievements: List[str] = field(default_factory=list)
    url: Optional[str] = None
    
    def __post_init__(self):
        if self.content_type != ContentType.PROJECT_ITEM:
            self.content_type = ContentType.PROJECT_ITEM


@dataclass
class QualificationItem(ContentItem):
    """Key qualification item."""
    skill_category: str = ""
    proficiency_level: str = ""
    years_experience: Optional[int] = None
    
    def __post_init__(self):
        if self.content_type != ContentType.QUALIFICATION:
            self.content_type = ContentType.QUALIFICATION


@dataclass
class JobDescriptionData:
    """Parsed job description data."""
    raw_text: str
    company_name: str = ""
    position_title: str = ""
    required_skills: List[str] = field(default_factory=list)
    preferred_skills: List[str] = field(default_factory=list)
    responsibilities: List[str] = field(default_factory=list)
    qualifications: List[str] = field(default_factory=list)
    company_culture: List[str] = field(default_factory=list)
    benefits: List[str] = field(default_factory=list)
    salary_range: Optional[str] = None
    location: Optional[str] = None
    remote_policy: Optional[str] = None
    parsed_at: datetime = field(default_factory=datetime.now)


@dataclass
class ProcessingQueue:
    """Queue for managing individual item processing."""
    pending_items: List[ContentItem] = field(default_factory=list)
    in_progress_items: List[ContentItem] = field(default_factory=list)
    completed_items: List[ContentItem] = field(default_factory=list)
    failed_items: List[ContentItem] = field(default_factory=list)
    
    def add_item(self, item: ContentItem):
        """Add an item to the pending queue."""
        self.pending_items.append(item)
    
    def get_next_item(self) -> Optional[ContentItem]:
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
    
    def complete_item(self, item: ContentItem, generated_content: str):
        """Mark an item as completed."""
        item.generated_content = generated_content
        item.metadata.update_status(ProcessingStatus.COMPLETED)
        
        if item in self.in_progress_items:
            self.in_progress_items.remove(item)
        self.completed_items.append(item)
    
    def fail_item(self, item: ContentItem, error: str):
        """Mark an item as failed."""
        item.metadata.update_status(ProcessingStatus.FAILED, error)
        
        if item in self.in_progress_items:
            self.in_progress_items.remove(item)
        self.failed_items.append(item)
    
    def rate_limit_item(self, item: ContentItem):
        """Mark an item as rate limited and return to pending."""
        item.metadata.update_status(ProcessingStatus.RATE_LIMITED)
        item.metadata.rate_limit_hits += 1
        
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
    key_qualifications: List[QualificationItem] = field(default_factory=list)
    professional_experiences: List[ExperienceItem] = field(default_factory=list)
    side_projects: List[ProjectItem] = field(default_factory=list)
    executive_summary: Optional[ContentItem] = None
    
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
            qualification = QualificationItem(
                content_type=ContentType.QUALIFICATION,
                original_content=item_text,
                priority=len(items) - i  # Earlier items have higher priority
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
        
        # Check rate limits (conservative estimates)
        # These should be configured based on actual API limits
        max_requests_per_minute = 30  # Conservative estimate
        max_tokens_per_minute = 50000  # Conservative estimate
        
        return (
            self.requests_per_minute < max_requests_per_minute and
            (self.tokens_per_minute + estimated_tokens) < max_tokens_per_minute
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