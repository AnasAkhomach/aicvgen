"""Base classes for agents."""

from abc import ABC, abstractmethod
from typing import Any

from ..config.logging_config import get_structured_logger
from ..models.agent_models import AgentResult
from ..services.progress_tracker import ProgressTracker


class AgentBase(ABC):
    """Abstract base class for all agents."""

    def __init__(self, name: str, description: str, session_id: str):
        """Initializes the agent."""
        self.name = name
        self.description = description
        self.session_id = session_id
        self.logger = get_structured_logger(f"{name}:{session_id}")
        self.progress_tracker: ProgressTracker | None = None
        self.logger.info(
            f"Agent '{self.name}' initialized for session '{self.session_id}'."
        )

    def set_progress_tracker(self, progress_tracker: ProgressTracker):
        """Sets the progress tracker for the agent."""
        self.progress_tracker = progress_tracker

    def update_progress(self, progress: int, message: str):
        """Updates the progress of the agent's task."""
        if self.progress_tracker:
            self.progress_tracker.update_progress(self.name, progress, message)
        self.logger.info(f"Progress for {self.name}: {progress}% - {message}")

    @abstractmethod
    async def run(self, **kwargs: Any) -> AgentResult:
        """
        Runs the agent's task.

        Args:
            **kwargs: The input data for the agent, passed as keyword arguments.

        Returns:
            The result of the agent's task as an AgentResult object.
        """
        pass
