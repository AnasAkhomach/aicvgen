"""Progress tracking service for CV generation workflow.

This module provides real-time progress tracking and reporting capabilities
for the individual item processing workflow.
"""

import asyncio
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set

import logging
from src.models.progress_models import (ProgressEvent, ProgressEventType, ProgressMetrics)
from src.models.workflow_models import ContentType
from src.orchestration.state import GlobalState
from src.constants.config_constants import ConfigConstants


class SessionTracker:
    """Manages progress tracking data for a single session."""

    def __init__(self, session_id: str, initial_state: GlobalState):
        self.session_id = session_id
        self.events: List[ProgressEvent] = []
        self.metrics: ProgressMetrics = ProgressMetrics(
            session_id=session_id,
            started_at=datetime.now(),
            current_stage=initial_state.get("current_stage"),
        )
        self.subscribers: Set[Callable] = set()
        self.max_events_per_session = 1000  # Default limit, can be configured

    def update_from_state(self, state: GlobalState):
        """Update metrics from workflow state."""
        self.metrics.update_from_state(state)

    def record_event(self, event_type: ProgressEventType, data: Dict[str, Any]):
        """Record a progress event for this session."""
        event = ProgressEvent(
            event_type=event_type,
            timestamp=datetime.now(),
            session_id=self.session_id,
            data=data,
        )
        self.events.append(event)
        # Limit event history size
        if len(self.events) > self.max_events_per_session:
            self.events = self.events[-self.max_events_per_session :]
        return event

    def notify_subscribers(self, event: ProgressEvent, logger):
        """Notify all subscribers of a progress event for this session."""
        for callback in list(self.subscribers):  # Iterate over a copy to allow
            # modification during iteration
            try:
                if asyncio.iscoroutinefunction(callback):
                    asyncio.create_task(callback(event))
                else:
                    callback(event)
            except (TypeError, ValueError, AttributeError) as e:
                logger.error(
                    f"Error in progress subscriber callback: {e}",
                    session_id=self.session_id,
                )

    def subscribe(self, callback: Callable[[ProgressEvent], None]):
        """Subscribe to progress events for this session."""
        self.subscribers.add(callback)

    def unsubscribe(self, callback: Callable[[ProgressEvent], None]):
        """Unsubscribe from progress events for this session."""
        self.subscribers.discard(callback)

    def get_events(
        self,
        event_types: Optional[List[ProgressEventType]] = None,
        limit: Optional[int] = None,
    ) -> List[ProgressEvent]:
        """Get events for this session."""
        events = self.events

        # Filter by event types if specified
        if event_types:
            events = [e for e in events if e.event_type in event_types]

        # Apply limit
        if limit:
            events = events[-limit:]

        return events

    def get_progress_summary(self) -> Dict[str, Any]:
        """Get a comprehensive progress summary for this session."""
        recent_events = self.get_events(limit=ConfigConstants.DEFAULT_RECENT_EVENTS_LIMIT)

        return {
            "metrics": self.metrics.to_dict(),
            "recent_events": [event.to_dict() for event in recent_events],
            "is_active": len(self.subscribers) > 0,
        }

    def export_data(self) -> Dict[str, Any]:
        """Export all tracking data for this session."""
        return {
            "session_id": self.session_id,
            "metrics": self.metrics.to_dict(),
            "events": [event.to_dict() for event in self.events],
            "exported_at": datetime.now().isoformat(),
        }


class ProgressTracker:
    """Centralized progress tracker for managing multiple workflow sessions."""

    def __init__(self, logger: logging.Logger):
        """Initialize ProgressTracker with injected dependencies.
        
        Args:
            logger: Logger instance for tracking operations.
        """
        self.logger = logger
        self.sessions: Dict[str, SessionTracker] = {}

    def start_tracking(self, session_id: str, state: GlobalState):
        """Start tracking progress for a new session or resume an existing one."""
        if session_id not in self.sessions:
            self.sessions[session_id] = SessionTracker(session_id, state)
            self.logger.info(
                "Initialized new session tracker",
                session_id=session_id,
                initial_stage=state.get("current_stage").value,
            )
        else:
            self.logger.info(
                "Resuming existing session tracker",
                session_id=session_id,
                current_stage=state.get("current_stage").value,
            )

        # Update from initial state and record start event
        session_tracker = self.sessions[session_id]
        session_tracker.update_from_state(state)
        event = session_tracker.record_event(
            ProgressEventType.WORKFLOW_STARTED,
            {
                "total_items": session_tracker.metrics.total_items,
                "initial_stage": state.get("current_stage").value,
            },
        )
        session_tracker.notify_subscribers(event, self.logger)

        self.logger.info(
            "Started progress tracking",
            session_id=session_id,
            total_items=session_tracker.metrics.total_items,
        )

    def update_progress(self, session_id: str, state: GlobalState):
        """Update progress from workflow state for a given session."""
        session_tracker = self.sessions.get(session_id)
        if not session_tracker:
            self.logger.warning(
                "Attempted to update progress for unknown session",
                session_id=session_id,
            )
            return

        old_stage = session_tracker.metrics.current_stage
        session_tracker.update_from_state(state)

        # Check for stage change
        if old_stage != state.get("current_stage"):
            event = session_tracker.record_event(
                ProgressEventType.STAGE_CHANGED,
                {
                    "old_stage": old_stage.value,
                    "new_stage": state.get("current_stage").value,
                    "completion_percentage": session_tracker.metrics.completion_percentage,
                },
            )
            session_tracker.notify_subscribers(event, self.logger)

    def record_item_started(
        self, session_id: str, item_id: str, item_type: ContentType
    ):
        """Record that an item has started processing for a session."""
        session_tracker = self.sessions.get(session_id)
        if not session_tracker:
            return
        event = session_tracker.record_event(
            ProgressEventType.ITEM_STARTED,
            {"item_id": item_id, "item_type": item_type.value},
        )
        session_tracker.notify_subscribers(event, self.logger)

    def record_item_completed(self, session_id: str, item_data: Dict[str, Any]):
        """Record that an item has completed processing for a session."""
        session_tracker = self.sessions.get(session_id)
        if not session_tracker:
            return
        event = session_tracker.record_event(
            ProgressEventType.ITEM_COMPLETED,
            item_data,
        )
        session_tracker.notify_subscribers(event, self.logger)

    def record_item_failed(self, session_id: str, item_data: Dict[str, Any]):
        """Record that an item has failed processing for a session."""
        session_tracker = self.sessions.get(session_id)
        if not session_tracker:
            return
        event = session_tracker.record_event(
            ProgressEventType.ITEM_FAILED,
            item_data,
        )
        session_tracker.notify_subscribers(event, self.logger)

    def record_item_rate_limited(
        self, session_id: str, item_id: str, item_type: ContentType, retry_after: float
    ):
        """Record that an item hit rate limits for a session."""
        session_tracker = self.sessions.get(session_id)
        if not session_tracker:
            return
        event = session_tracker.record_event(
            ProgressEventType.ITEM_RATE_LIMITED,
            {
                "item_id": item_id,
                "item_type": item_type.value,
                "retry_after": retry_after,
            },
        )
        session_tracker.notify_subscribers(event, self.logger)

    def record_batch_completed(self, session_id: str, batch_data: Dict[str, Any]):
        """Record that a batch has completed processing for a session."""
        session_tracker = self.sessions.get(session_id)
        if not session_tracker:
            return
        event = session_tracker.record_event(
            ProgressEventType.BATCH_COMPLETED,
            batch_data,
        )
        session_tracker.notify_subscribers(event, self.logger)

    def record_workflow_completed(self, session_id: str, final_metrics: Dict[str, Any]):
        """Record that the workflow has completed for a session."""
        session_tracker = self.sessions.get(session_id)
        if not session_tracker:
            return
        event = session_tracker.record_event(
            ProgressEventType.WORKFLOW_COMPLETED, final_metrics
        )
        session_tracker.notify_subscribers(event, self.logger)

    def record_error(self, session_id: str, error: str, context: Dict[str, Any] = None):
        """Record an error event for a session."""
        session_tracker = self.sessions.get(session_id)
        if not session_tracker:
            return
        event = session_tracker.record_event(
            ProgressEventType.ERROR_OCCURRED,
            {"error": error, "context": context or {}},
        )
        session_tracker.notify_subscribers(event, self.logger)

    def subscribe(self, session_id: str, callback: Callable[[ProgressEvent], None]):
        """Subscribe to progress events for a session."""
        session_tracker = self.sessions.get(session_id)
        if not session_tracker:
            self.logger.warning(
                "Attempted to subscribe to unknown session", session_id=session_id
            )
            return
        session_tracker.subscribe(callback)
        self.logger.info(
            "Added progress subscriber",
            session_id=session_id,
            total_subscribers=len(session_tracker.subscribers),
        )

    def unsubscribe(self, session_id: str, callback: Callable[[ProgressEvent], None]):
        """Unsubscribe from progress events for a session."""
        session_tracker = self.sessions.get(session_id)
        if not session_tracker:
            return
        session_tracker.unsubscribe(callback)
        if not session_tracker.subscribers:
            self.logger.info(
                "No more subscribers for session, cleaning up", session_id=session_id
            )
            # Optionally remove the session_tracker if no more subscribers
            # del self.sessions[session_id]

    def get_metrics(self, session_id: str) -> Optional[ProgressMetrics]:
        """Get current metrics for a session."""
        session_tracker = self.sessions.get(session_id)
        return session_tracker.metrics if session_tracker else None

    def get_events(
        self,
        session_id: str,
        event_types: Optional[List[ProgressEventType]] = None,
        limit: Optional[int] = None,
    ) -> List[ProgressEvent]:
        """Get events for a session."""
        session_tracker = self.sessions.get(session_id)
        if not session_tracker:
            return []
        return session_tracker.get_events(event_types, limit)

    def get_progress_summary(self, session_id: str) -> Dict[str, Any]:
        """Get a comprehensive progress summary."""
        session_tracker = self.sessions.get(session_id)
        if not session_tracker:
            return {"error": "Session not found"}
        return session_tracker.get_progress_summary()

    def cleanup_session(self, session_id: str):
        """Clean up tracking data for a completed session."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            self.logger.info("Cleaned up session tracking data", session_id=session_id)

    def get_active_sessions(self) -> List[str]:
        """Get list of active session IDs."""
        return list(self.sessions.keys())

    def export_session_data(self, session_id: str) -> Dict[str, Any]:
        """Export all tracking data for a session."""
        session_tracker = self.sessions.get(session_id)
        if not session_tracker:
            return {"error": "Session not found"}
        return session_tracker.export_data()


# Note: Global progress tracker functions removed to enforce dependency injection.
# ProgressTracker should be obtained through the DI container.
