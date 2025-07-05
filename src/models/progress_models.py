from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, Optional

from src.models.workflow_models import WorkflowStage


class ProgressEventType(Enum):
    """Types of progress events."""

    WORKFLOW_STARTED = "workflow_started"
    STAGE_CHANGED = "stage_changed"
    ITEM_STARTED = "item_started"
    ITEM_COMPLETED = "item_completed"
    ITEM_FAILED = "item_failed"
    ITEM_RATE_LIMITED = "item_rate_limited"
    BATCH_COMPLETED = "batch_completed"
    WORKFLOW_COMPLETED = "workflow_completed"
    ERROR_OCCURRED = "error_occurred"


@dataclass
class ProgressEvent:
    """Progress event data structure."""

    event_type: ProgressEventType
    timestamp: datetime
    session_id: str
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "session_id": self.session_id,
            "data": self.data,
        }


@dataclass
class ProgressMetrics:
    """Progress metrics for a workflow session."""

    session_id: str
    started_at: datetime
    current_stage: WorkflowStage

    # Item counts
    total_items: int = 0
    completed_items: int = 0
    failed_items: int = 0
    rate_limited_items: int = 0
    in_progress_items: int = 0

    # Stage-specific progress
    qualifications_total: int = 0
    qualifications_completed: int = 0
    experiences_total: int = 0
    experiences_completed: int = 0
    projects_total: int = 0
    projects_completed: int = 0

    # Performance metrics
    total_processing_time: float = 0.0
    average_item_time: float = 0.0
    estimated_completion_time: Optional[datetime] = None

    # Rate limiting metrics
    total_rate_limit_hits: int = 0
    total_retries: int = 0

    # LLM usage metrics
    total_llm_calls: int = 0
    total_tokens_used: int = 0

    def update_from_state(self, state: Any):
        """Update metrics from workflow state."""
        self.current_stage = state.current_stage

        # Update item counts
        self.qualifications_total = state.qualification_queue.total_items
        self.qualifications_completed = len(state.qualification_queue.completed_items)

        self.experiences_total = state.experience_queue.total_items
        self.experiences_completed = len(state.experience_queue.completed_items)

        self.projects_total = state.project_queue.total_items
        self.projects_completed = len(state.project_queue.completed_items)

        self.total_items = (
            self.qualifications_total + self.experiences_total + self.projects_total
        )

        self.completed_items = (
            self.qualifications_completed
            + self.experiences_completed
            + self.projects_completed
        )

        self.failed_items = (
            len(state.qualification_queue.failed_items)
            + len(state.experience_queue.failed_items)
            + len(state.project_queue.failed_items)
        )

        self.in_progress_items = (
            len(state.qualification_queue.in_progress_items)
            + len(state.experience_queue.in_progress_items)
            + len(state.project_queue.in_progress_items)
        )

        # Update performance metrics
        self.total_processing_time = state.total_processing_time
        self.total_llm_calls = state.total_llm_calls
        self.total_tokens_used = state.total_tokens_used
        self.total_rate_limit_hits = state.total_rate_limit_hits

        # Calculate average processing time
        if self.completed_items > 0:
            self.average_item_time = self.total_processing_time / self.completed_items

        # Estimate completion time
        if self.completed_items > 0 and self.average_item_time > 0:
            remaining_items = self.total_items - self.completed_items
            estimated_remaining_time = remaining_items * self.average_item_time
            self.estimated_completion_time = datetime.now() + timedelta(
                seconds=estimated_remaining_time
            )

    @property
    def completion_percentage(self) -> float:
        """Calculate overall completion percentage."""
        if self.total_items == 0:
            return 0.0
        return (self.completed_items / self.total_items) * 100

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        processed_items = self.completed_items + self.failed_items
        if processed_items == 0:
            return 0.0
        return (self.completed_items / processed_items) * 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "session_id": self.session_id,
            "started_at": self.started_at.isoformat(),
            "current_stage": self.current_stage.value,
            "total_items": self.total_items,
            "completed_items": self.completed_items,
            "failed_items": self.failed_items,
            "rate_limited_items": self.rate_limited_items,
            "in_progress_items": self.in_progress_items,
            "completion_percentage": self.completion_percentage,
            "success_rate": self.success_rate,
            "qualifications": {
                "total": self.qualifications_total,
                "completed": self.qualifications_completed,
                "percentage": (
                    (self.qualifications_completed / self.qualifications_total * 100)
                    if self.qualifications_total > 0
                    else 0
                ),
            },
            "experiences": {
                "total": self.experiences_total,
                "completed": self.experiences_completed,
                "percentage": (
                    (self.experiences_completed / self.experiences_total * 100)
                    if self.experiences_total > 0
                    else 0
                ),
            },
            "projects": {
                "total": self.projects_total,
                "completed": self.projects_completed,
                "percentage": (
                    (self.projects_completed / self.projects_total * 100)
                    if self.projects_total > 0
                    else 0
                ),
            },
            "performance": {
                "total_processing_time": self.total_processing_time,
                "average_item_time": self.average_item_time,
                "estimated_completion_time": (
                    self.estimated_completion_time.isoformat()
                    if self.estimated_completion_time
                    else None
                ),
                "total_rate_limit_hits": self.total_rate_limit_hits,
                "total_retries": self.total_retries,
                "total_llm_calls": self.total_llm_calls,
                "total_tokens_used": self.total_tokens_used,
            },
        }
