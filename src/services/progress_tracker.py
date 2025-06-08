"""Progress tracking service for CV generation workflow.

This module provides real-time progress tracking and reporting capabilities
for the individual item processing workflow.
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum

from ..config.logging_config import get_structured_logger
from ..models.data_models import (
    CVGenerationState, WorkflowStage, ProcessingStatus, ContentType
)


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
            "data": self.data
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
    
    def update_from_state(self, state: CVGenerationState):
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
            self.qualifications_total + 
            self.experiences_total + 
            self.projects_total
        )
        
        self.completed_items = (
            self.qualifications_completed + 
            self.experiences_completed + 
            self.projects_completed
        )
        
        self.failed_items = (
            len(state.qualification_queue.failed_items) +
            len(state.experience_queue.failed_items) +
            len(state.project_queue.failed_items)
        )
        
        self.in_progress_items = (
            len(state.qualification_queue.in_progress_items) +
            len(state.experience_queue.in_progress_items) +
            len(state.project_queue.in_progress_items)
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
            self.estimated_completion_time = datetime.now() + timedelta(seconds=estimated_remaining_time)
    
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
                "percentage": (self.qualifications_completed / self.qualifications_total * 100) if self.qualifications_total > 0 else 0
            },
            "experiences": {
                "total": self.experiences_total,
                "completed": self.experiences_completed,
                "percentage": (self.experiences_completed / self.experiences_total * 100) if self.experiences_total > 0 else 0
            },
            "projects": {
                "total": self.projects_total,
                "completed": self.projects_completed,
                "percentage": (self.projects_completed / self.projects_total * 100) if self.projects_total > 0 else 0
            },
            "performance": {
                "total_processing_time": self.total_processing_time,
                "average_item_time": self.average_item_time,
                "estimated_completion_time": self.estimated_completion_time.isoformat() if self.estimated_completion_time else None,
                "total_rate_limit_hits": self.total_rate_limit_hits,
                "total_retries": self.total_retries,
                "total_llm_calls": self.total_llm_calls,
                "total_tokens_used": self.total_tokens_used
            }
        }


class ProgressTracker:
    """Progress tracker for CV generation workflow."""
    
    def __init__(self):
        self.logger = get_structured_logger("progress_tracker")
        
        # Event storage
        self.events: Dict[str, List[ProgressEvent]] = defaultdict(list)
        self.metrics: Dict[str, ProgressMetrics] = {}
        
        # Subscribers for real-time updates
        self.subscribers: Dict[str, Set[Callable]] = defaultdict(set)
        
        # Event history limits
        self.max_events_per_session = 1000
    
    def start_tracking(self, session_id: str, state: CVGenerationState):
        """Start tracking progress for a session."""
        
        # Initialize metrics
        self.metrics[session_id] = ProgressMetrics(
            session_id=session_id,
            started_at=datetime.now(),
            current_stage=state.current_stage
        )
        
        # Update from initial state
        self.metrics[session_id].update_from_state(state)
        
        # Record start event
        self._record_event(
            session_id,
            ProgressEventType.WORKFLOW_STARTED,
            {
                "total_items": self.metrics[session_id].total_items,
                "initial_stage": state.current_stage.value
            }
        )
        
        self.logger.info(
            "Started progress tracking",
            session_id=session_id,
            total_items=self.metrics[session_id].total_items
        )
    
    def update_progress(self, session_id: str, state: CVGenerationState):
        """Update progress from workflow state."""
        
        if session_id not in self.metrics:
            self.start_tracking(session_id, state)
            return
        
        old_stage = self.metrics[session_id].current_stage
        self.metrics[session_id].update_from_state(state)
        
        # Check for stage change
        if old_stage != state.current_stage:
            self._record_event(
                session_id,
                ProgressEventType.STAGE_CHANGED,
                {
                    "old_stage": old_stage.value,
                    "new_stage": state.current_stage.value,
                    "completion_percentage": self.metrics[session_id].completion_percentage
                }
            )
    
    def record_item_started(self, session_id: str, item_id: str, item_type: ContentType):
        """Record that an item has started processing."""
        self._record_event(
            session_id,
            ProgressEventType.ITEM_STARTED,
            {
                "item_id": item_id,
                "item_type": item_type.value
            }
        )
    
    def record_item_completed(
        self, 
        session_id: str, 
        item_id: str, 
        item_type: ContentType,
        processing_time: float,
        tokens_used: int = 0
    ):
        """Record that an item has completed processing."""
        self._record_event(
            session_id,
            ProgressEventType.ITEM_COMPLETED,
            {
                "item_id": item_id,
                "item_type": item_type.value,
                "processing_time": processing_time,
                "tokens_used": tokens_used
            }
        )
    
    def record_item_failed(
        self, 
        session_id: str, 
        item_id: str, 
        item_type: ContentType,
        error: str,
        retry_count: int = 0
    ):
        """Record that an item has failed processing."""
        self._record_event(
            session_id,
            ProgressEventType.ITEM_FAILED,
            {
                "item_id": item_id,
                "item_type": item_type.value,
                "error": error,
                "retry_count": retry_count
            }
        )
    
    def record_item_rate_limited(
        self, 
        session_id: str, 
        item_id: str, 
        item_type: ContentType,
        retry_after: float
    ):
        """Record that an item hit rate limits."""
        self._record_event(
            session_id,
            ProgressEventType.ITEM_RATE_LIMITED,
            {
                "item_id": item_id,
                "item_type": item_type.value,
                "retry_after": retry_after
            }
        )
    
    def record_batch_completed(
        self, 
        session_id: str, 
        batch_size: int,
        successful_items: int,
        failed_items: int,
        processing_time: float
    ):
        """Record that a batch has completed processing."""
        self._record_event(
            session_id,
            ProgressEventType.BATCH_COMPLETED,
            {
                "batch_size": batch_size,
                "successful_items": successful_items,
                "failed_items": failed_items,
                "processing_time": processing_time
            }
        )
    
    def record_workflow_completed(self, session_id: str, final_metrics: Dict[str, Any]):
        """Record that the workflow has completed."""
        self._record_event(
            session_id,
            ProgressEventType.WORKFLOW_COMPLETED,
            final_metrics
        )
    
    def record_error(self, session_id: str, error: str, context: Dict[str, Any] = None):
        """Record an error event."""
        self._record_event(
            session_id,
            ProgressEventType.ERROR_OCCURRED,
            {
                "error": error,
                "context": context or {}
            }
        )
    
    def _record_event(self, session_id: str, event_type: ProgressEventType, data: Dict[str, Any]):
        """Record a progress event."""
        event = ProgressEvent(
            event_type=event_type,
            timestamp=datetime.now(),
            session_id=session_id,
            data=data
        )
        
        # Add to event history
        self.events[session_id].append(event)
        
        # Limit event history size
        if len(self.events[session_id]) > self.max_events_per_session:
            self.events[session_id] = self.events[session_id][-self.max_events_per_session:]
        
        # Notify subscribers
        self._notify_subscribers(session_id, event)
        
        # Log the event
        self.logger.info(
            f"Progress event: {event_type.value}",
            session_id=session_id,
            event_data=data
        )
    
    def _notify_subscribers(self, session_id: str, event: ProgressEvent):
        """Notify all subscribers of a progress event."""
        for callback in self.subscribers[session_id]:
            try:
                if asyncio.iscoroutinefunction(callback):
                    # Schedule async callback
                    asyncio.create_task(callback(event))
                else:
                    callback(event)
            except Exception as e:
                self.logger.error(
                    f"Error in progress subscriber callback: {e}",
                    session_id=session_id
                )
    
    def subscribe(self, session_id: str, callback: Callable[[ProgressEvent], None]):
        """Subscribe to progress events for a session."""
        self.subscribers[session_id].add(callback)
        
        self.logger.info(
            "Added progress subscriber",
            session_id=session_id,
            total_subscribers=len(self.subscribers[session_id])
        )
    
    def unsubscribe(self, session_id: str, callback: Callable[[ProgressEvent], None]):
        """Unsubscribe from progress events for a session."""
        self.subscribers[session_id].discard(callback)
        
        # Clean up empty subscriber sets
        if not self.subscribers[session_id]:
            del self.subscribers[session_id]
    
    def get_metrics(self, session_id: str) -> Optional[ProgressMetrics]:
        """Get current metrics for a session."""
        return self.metrics.get(session_id)
    
    def get_events(
        self, 
        session_id: str, 
        event_types: Optional[List[ProgressEventType]] = None,
        limit: Optional[int] = None
    ) -> List[ProgressEvent]:
        """Get events for a session."""
        events = self.events.get(session_id, [])
        
        # Filter by event types if specified
        if event_types:
            events = [e for e in events if e.event_type in event_types]
        
        # Apply limit
        if limit:
            events = events[-limit:]
        
        return events
    
    def get_progress_summary(self, session_id: str) -> Dict[str, Any]:
        """Get a comprehensive progress summary."""
        metrics = self.get_metrics(session_id)
        if not metrics:
            return {"error": "Session not found"}
        
        recent_events = self.get_events(session_id, limit=10)
        
        return {
            "metrics": metrics.to_dict(),
            "recent_events": [event.to_dict() for event in recent_events],
            "is_active": session_id in self.subscribers and len(self.subscribers[session_id]) > 0
        }
    
    def cleanup_session(self, session_id: str):
        """Clean up tracking data for a completed session."""
        # Keep metrics but remove events and subscribers
        if session_id in self.events:
            del self.events[session_id]
        
        if session_id in self.subscribers:
            del self.subscribers[session_id]
        
        self.logger.info("Cleaned up session tracking data", session_id=session_id)
    
    def get_active_sessions(self) -> List[str]:
        """Get list of active session IDs."""
        return list(self.metrics.keys())
    
    def export_session_data(self, session_id: str) -> Dict[str, Any]:
        """Export all tracking data for a session."""
        metrics = self.get_metrics(session_id)
        events = self.get_events(session_id)
        
        return {
            "session_id": session_id,
            "metrics": metrics.to_dict() if metrics else None,
            "events": [event.to_dict() for event in events],
            "exported_at": datetime.now().isoformat()
        }


# Global progress tracker instance
_global_progress_tracker: Optional[ProgressTracker] = None


def get_progress_tracker() -> ProgressTracker:
    """Get the global progress tracker instance."""
    global _global_progress_tracker
    if _global_progress_tracker is None:
        _global_progress_tracker = ProgressTracker()
    return _global_progress_tracker


def reset_progress_tracker():
    """Reset the global progress tracker (useful for testing)."""
    global _global_progress_tracker
    _global_progress_tracker = None