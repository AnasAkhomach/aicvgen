"""Session management service for CV generation workflow.

This module provides session management, state persistence, and user session
tracking capabilities for the individual item processing workflow.
"""

import asyncio
import json
import logging
import pickle
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config.settings import AppConfig
from src.constants.config_constants import ConfigConstants
from src.models.vector_store_and_session_models import SessionInfoModel
from src.models.workflow_models import WorkflowStage, WorkflowState


class SessionStatus(Enum):
    """Status of a user session."""

    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


@dataclass
class SessionInfo:
    """Information about a user session."""

    session_id: str
    user_id: Optional[str]
    status: SessionStatus
    created_at: datetime
    updated_at: datetime
    expires_at: datetime

    # Workflow information
    current_stage: WorkflowStage
    total_items: int = 0
    completed_items: int = 0
    failed_items: int = 0

    # Session metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Performance metrics
    processing_time: float = 0.0
    llm_calls: int = 0
    tokens_used: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "current_stage": self.current_stage.value,
            "total_items": self.total_items,
            "completed_items": self.completed_items,
            "failed_items": self.failed_items,
            "metadata": self.metadata,
            "processing_time": self.processing_time,
            "llm_calls": self.llm_calls,
            "tokens_used": self.tokens_used,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionInfo":
        """Create from dictionary."""
        return cls(
            session_id=data["session_id"],
            user_id=data.get("user_id"),
            status=SessionStatus(data["status"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            expires_at=datetime.fromisoformat(data["expires_at"]),
            current_stage=WorkflowStage(data["current_stage"]),
            total_items=data.get("total_items", 0),
            completed_items=data.get("completed_items", 0),
            failed_items=data.get("failed_items", 0),
            metadata=data.get("metadata", {}),
            processing_time=data.get("processing_time", 0.0),
            llm_calls=data.get("llm_calls", 0),
            tokens_used=data.get("tokens_used", 0),
        )


class SessionManager:
    """Manager for user sessions and state persistence."""

    def __init__(
        self,
        settings: AppConfig,
        logger: logging.Logger,
    ):
        """Initializes the SessionManager.

        Args:
            settings: Application configuration.
            logger: Logger instance.
        """
        self.logger = logger
        self.settings = settings

        # Storage path is now determined by the application configuration.
        self.storage_path = Path(self.settings.paths.sessions_directory)
        # The startup service ensures this directory exists, but we double-check.
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # In-memory session tracking
        self.active_sessions: Dict[str, SessionInfo] = {}
        self.session_states: Dict[str, WorkflowState] = {}

        # Thread safety
        self._lock = threading.RLock()

        # Session configuration from settings
        self.session_timeout = timedelta(
            seconds=self.settings.ui.session_timeout_seconds
        )
        self.max_active_sessions = self.settings.session.max_active_sessions
        self.cleanup_interval = timedelta(
            minutes=self.settings.session.cleanup_interval_minutes
        )

        # Background cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
        self._start_cleanup_task()

    def _start_cleanup_task(self):
        """Start background cleanup task."""
        try:
            loop = asyncio.get_running_loop()
            self._cleanup_task = loop.create_task(self._periodic_cleanup())
        except RuntimeError:
            # No event loop running, cleanup will be manual
            pass

    async def _periodic_cleanup(self):
        """Periodic cleanup of expired sessions."""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval.total_seconds())
                self.cleanup_expired_sessions()
            except asyncio.CancelledError:
                break
            except (TypeError, ValueError, KeyError, AttributeError) as e:
                self.logger.error("Error in periodic cleanup", error=str(e))

    def _initialize_session_state(
        self,
        session_id: str,
        user_id: Optional[str],
        now: datetime,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Centralized initialization for session info and workflow state."""
        session_info = SessionInfo(
            session_id=session_id,
            user_id=user_id,
            status=SessionStatus.ACTIVE,
            created_at=now,
            updated_at=now,
            expires_at=now + self.session_timeout,
            current_stage=WorkflowStage.INITIALIZATION,
            metadata=metadata or {},
        )
        initial_state = WorkflowState(
            session_id=session_id,
            current_stage=WorkflowStage.INITIALIZATION,
            created_at=now,
        )
        return session_info, initial_state

    @staticmethod
    def generate_session_id() -> str:
        """Generate a unique session ID."""
        return str(uuid.uuid4())

    def create_session(
        self, user_id: Optional[str] = None, metadata: Dict[str, Any] = None
    ) -> str:
        """Create a new session."""

        with self._lock:
            # Check session limits
            if len(self.active_sessions) >= self.max_active_sessions:
                self.cleanup_expired_sessions()
                if len(self.active_sessions) >= self.max_active_sessions:
                    raise RuntimeError("Maximum number of active sessions reached")

            # Generate session ID
            session_id = self.generate_session_id()
            now = datetime.now()

            # Centralized initialization
            session_info, initial_state = self._initialize_session_state(
                session_id, user_id, now, metadata
            )

            # Store in memory
            self.active_sessions[session_id] = session_info
            self.session_states[session_id] = initial_state

            # Persist to storage
            self._save_session(session_id)

            self.logger.info(
                "Created new session", session_id=session_id, user_id=user_id
            )

            return session_id

    def get_session(self, session_id: str) -> Optional[SessionInfo]:
        """Get session information."""
        with self._lock:
            # Check in-memory first
            if session_id in self.active_sessions:
                session_info = self.active_sessions[session_id]

                # Check if expired
                if datetime.now() > session_info.expires_at:
                    self._expire_session(session_id)
                    return None

                return session_info

            # Try to load from storage
            return self._load_session(session_id)

    def get_session_state(self, session_id: str) -> Optional[WorkflowState]:
        """Get session state."""
        with self._lock:
            # Check if session exists and is valid
            session_info = self.get_session(session_id)
            if not session_info:
                return None

            # Return in-memory state if available
            if session_id in self.session_states:
                return self.session_states[session_id]

            # Try to load state from storage
            return self._load_session_state(session_id)

    def update_session_state(self, session_id: str, state: WorkflowState):
        """Update session state."""
        with self._lock:
            # Check if session exists
            session_info = self.get_session(session_id)
            if not session_info:
                raise ValueError(f"Session {session_id} not found")

            # Update session info from state
            session_info.current_stage = state.current_stage
            session_info.updated_at = datetime.now()
            session_info.processing_time = state.total_processing_time
            session_info.llm_calls = state.total_llm_calls
            session_info.tokens_used = state.total_tokens_used

            # Calculate item counts
            session_info.total_items = (
                state.qualification_queue.total_items
                + state.experience_queue.total_items
                + state.project_queue.total_items
            )

            session_info.completed_items = (
                len(state.qualification_queue.completed_items)
                + len(state.experience_queue.completed_items)
                + len(state.project_queue.completed_items)
            )

            session_info.failed_items = (
                len(state.qualification_queue.failed_items)
                + len(state.experience_queue.failed_items)
                + len(state.project_queue.failed_items)
            )

            # Update in-memory state
            self.session_states[session_id] = state

            # Persist to storage
            self._save_session(session_id)

            self.logger.debug(
                "Updated session state",
                session_id=session_id,
                stage=state.current_stage.value,
                completed_items=session_info.completed_items,
                total_items=session_info.total_items,
            )

    def update_session_status(self, session_id: str, status: SessionStatus):
        """Update session status."""
        with self._lock:
            session_info = self.get_session(session_id)
            if not session_info:
                raise ValueError(f"Session {session_id} not found")

            old_status = session_info.status
            session_info.status = status
            session_info.updated_at = datetime.now()

            # If completing or failing, remove from active sessions
            if status in [
                SessionStatus.COMPLETED,
                SessionStatus.FAILED,
                SessionStatus.CANCELLED,
            ]:
                if session_id in self.active_sessions:
                    del self.active_sessions[session_id]
                if session_id in self.session_states:
                    del self.session_states[session_id]

            # Persist changes
            self._save_session(session_id)

            self.logger.info(
                "Updated session status",
                session_id=session_id,
                old_status=old_status.value,
                new_status=status.value,
            )

    def extend_session(self, session_id: str, extension: timedelta = None):
        """Extend session expiration time."""
        with self._lock:
            session_info = self.get_session(session_id)
            if not session_info:
                raise ValueError(f"Session {session_id} not found")

            extension = extension or self.session_timeout
            session_info.expires_at = datetime.now() + extension
            session_info.updated_at = datetime.now()

            # Persist changes
            self._save_session(session_id)

            self.logger.info(
                "Extended session",
                session_id=session_id,
                new_expiry=session_info.expires_at.isoformat(),
            )

    def pause_session(self, session_id: str):
        """Pause a session."""
        self.update_session_status(session_id, SessionStatus.PAUSED)

    def resume_session(self, session_id: str):
        """Resume a paused session."""
        with self._lock:
            session_info = self.get_session(session_id)
            if not session_info:
                raise ValueError(f"Session {session_id} not found")

            if session_info.status != SessionStatus.PAUSED:
                raise ValueError(f"Session {session_id} is not paused")

            # Extend expiration and resume
            session_info.expires_at = datetime.now() + self.session_timeout
            self.update_session_status(session_id, SessionStatus.ACTIVE)

            # Reload state into memory
            state = self._load_session_state(session_id)
            if state:
                self.session_states[session_id] = state

    def cancel_session(self, session_id: str):
        """Cancel a session."""
        self.update_session_status(session_id, SessionStatus.CANCELLED)

    def complete_session(self, session_id: str):
        """Mark a session as completed."""
        self.update_session_status(session_id, SessionStatus.COMPLETED)

    def get_current_session_id(self) -> str:
        """Get the current active session ID or create a new one.

        This method provides centralized session ID management for the application.
        It returns the most recently created active session or creates a new one if none exists.

        Returns:
            str: The current session ID
        """
        with self._lock:
            # Return the most recent active session if available
            if self.active_sessions:
                # Get the most recently created session
                latest_session = max(
                    self.active_sessions.values(), key=lambda s: s.created_at
                )
                return latest_session.session_id

            # Create a new session if none exists
            return self.create_session()

    def fail_session(self, session_id: str, error: str = None):
        """Mark a session as failed."""
        with self._lock:
            session_info = self.get_session(session_id)
            if session_info and error:
                session_info.metadata["failure_reason"] = error

            self.update_session_status(session_id, SessionStatus.FAILED)

    def cleanup_expired_sessions(self):
        """Clean up expired sessions."""
        with self._lock:
            now = datetime.now()
            expired_sessions = []

            for session_id, session_info in list(self.active_sessions.items()):
                if now > session_info.expires_at:
                    expired_sessions.append(session_id)

            for session_id in expired_sessions:
                self._expire_session(session_id)

            if expired_sessions:
                self.logger.info(
                    f"Cleaned up {len(expired_sessions)} expired sessions",
                    expired_session_ids=expired_sessions,
                )

    def _expire_session(self, session_id: str):
        """Mark a session as expired and clean up."""
        if session_id in self.active_sessions:
            session_info = self.active_sessions[session_id]
            session_info.status = SessionStatus.EXPIRED
            session_info.updated_at = datetime.now()

            # Save final state
            self._save_session(session_id)

            # Remove from active tracking
            del self.active_sessions[session_id]
            if session_id in self.session_states:
                del self.session_states[session_id]

            self.logger.info(f"Session expired: {session_id}")

    def _save_session(self, session_id: str):
        """Save session to persistent storage."""
        try:
            session_info = self.active_sessions.get(session_id)
            if not session_info:
                return

            # Save session info
            info_path = self.storage_path / f"{session_id}_info.json"
            with open(info_path, "w") as f:
                json.dump(session_info.to_dict(), f, indent=2)

            # Save session state if available
            if session_id in self.session_states:
                state_path = self.storage_path / f"{session_id}_state.pkl"
                with open(state_path, "wb") as f:
                    pickle.dump(self.session_states[session_id], f)

        except (
            IOError,
            OSError,
            json.JSONDecodeError,
            pickle.PicklingError,
            TypeError,
        ) as e:
            self.logger.error(
                f"Failed to save session {session_id}: {e}", session_id=session_id
            )

    def _load_session(self, session_id: str) -> Optional[SessionInfo]:
        """Load session from persistent storage."""
        try:
            info_path = self.storage_path / f"{session_id}_info.json"
            if not info_path.exists():
                return None

            with open(info_path, "r") as f:
                data = json.load(f)

            session_info = SessionInfo.from_dict(data)

            # Check if expired
            if datetime.now() > session_info.expires_at:
                return None

            # Add to active sessions if still active
            if session_info.status == SessionStatus.ACTIVE:
                self.active_sessions[session_id] = session_info

            return session_info

        except (IOError, OSError, json.JSONDecodeError, KeyError, TypeError) as e:
            self.logger.error(
                f"Failed to load session {session_id}: {e}", session_id=session_id
            )
            return None

    def _load_session_state(self, session_id: str) -> Optional[WorkflowState]:
        """Load session state from persistent storage."""
        try:
            state_path = self.storage_path / f"{session_id}_state.pkl"
            if not state_path.exists():
                return None

            with open(state_path, "rb") as f:
                state = pickle.load(f)

            # Add to in-memory states if not already there
            with self._lock:
                self.session_states[session_id] = state
            return state

        except (IOError, OSError, pickle.UnpicklingError) as e:
            self.logger.error(
                f"Failed to load session state {session_id}: {e}",
                session_id=session_id,
            )
            return None

    def list_sessions(
        self,
        user_id: Optional[str] = None,
        status: Optional[SessionStatus] = None,
        limit: int = ConfigConstants.DEFAULT_SESSION_LIST_LIMIT,
    ) -> List[SessionInfo]:
        """List sessions with optional filtering."""
        sessions = []

        # Check active sessions
        for session_info in self.active_sessions.values():
            if user_id and session_info.user_id != user_id:
                continue
            if status and session_info.status != status:
                continue
            sessions.append(session_info)

        # Check stored sessions if needed
        if len(sessions) < limit:
            for info_file in self.storage_path.glob("*_info.json"):
                if len(sessions) >= limit:
                    break

                session_id = info_file.stem.replace("_info", "")
                if session_id in self.active_sessions:
                    continue  # Already included

                session_info = self._load_session(session_id)
                if not session_info:
                    continue

                if user_id and session_info.user_id != user_id:
                    continue
                if status and session_info.status != status:
                    continue

                sessions.append(session_info)

        # Sort by creation time (newest first)
        sessions.sort(key=lambda s: s.created_at, reverse=True)

        return sessions[:limit]

    def get_session_summary(self) -> dict:
        """Get a summary of all active sessions."""
        summary = [
            SessionInfoModel(**session.to_dict())
            for session in self.active_sessions.values()
        ]
        return {"sessions": [s.dict() for s in summary]}

    @contextmanager
    def session_context(self, session_id: str):
        """Context manager for session operations."""
        session_info = self.get_session(session_id)
        if not session_info:
            raise ValueError(f"Session {session_id} not found")

        try:
            yield session_info
        finally:
            # Auto-save on exit            if session_id in self.active_sessions:
            self._save_session(session_id)

    def get_next_item_for_generation(self, session_id: str) -> Optional[str]:
        """Get the next item ID that needs content generation."""
        # For now, return None to indicate no more items to process
        # This method should be implemented based on the actual queue logic
        # when the content generation workflow is fully defined
        return None

    def shutdown(self):
        """Shutdown the session manager."""
        # Cancel cleanup task
        if self._cleanup_task:
            self._cleanup_task.cancel()

        # Save all active sessions
        with self._lock:
            for session_id in list(self.active_sessions.keys()):
                self._save_session(session_id)

        self.logger.info("Session manager shutdown complete")


# Note: Global session manager functions removed to enforce dependency injection.
# SessionManager should be obtained through the DI container.
