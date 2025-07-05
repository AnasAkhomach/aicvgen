from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import  uuid4

from pydantic import BaseModel, Field

from src.models.cv_models import MetadataModel


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
    JOB_ANALYSIS = "job_analysis"
    JSON = "json"
    SKILL_EXTRACTION = "skill_extraction"
    JSON_CLEANING = "json_cleaning"
    CV_ASSESSMENT = "cv_assessment"
    SKILL_GENERATION = "skill_generation"
    ROLE_GENERATION = "role_generation"
    PROJECT_GENERATION = "project_generation"


class UserAction(str, Enum):
    """Enumeration for user actions in the UI."""

    ACCEPT = "accept"
    REGENERATE = "regenerate"
    APPROVE = "approve"


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
