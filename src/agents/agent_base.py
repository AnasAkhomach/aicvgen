"""Base classes for agents."""

from abc import ABC, abstractmethod
from typing import Any

from ..config.logging_config import get_structured_logger
from ..error_handling.exceptions import AgentExecutionError
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

    async def run(self, **kwargs: Any) -> AgentResult:
        """
        Template method for agent execution with standardized validation and error handling.

        Args:
            **kwargs: The input data for the agent, passed as keyword arguments.

        Returns:
            The result of the agent's task as an AgentResult object.
        """
        self.update_progress(0, f"Starting {self.name} execution.")
        input_data = kwargs.get("input_data", {})

        try:
            self._validate_inputs(input_data)
            self.update_progress(20, "Input validation passed.")
            return await self._execute(**kwargs)
        except AgentExecutionError as e:
            self.logger.error(
                f"Agent execution error in {self.name}: {e}", exc_info=True
            )
            return AgentResult.failure(agent_name=self.name, error_message=str(e))
        except (AttributeError, TypeError, ValueError, KeyError) as e:
            self.logger.error(
                f"An unexpected error occurred in {self.name}: {e}", exc_info=True
            )
            return AgentResult.failure(
                agent_name=self.name, error_message=f"An unexpected error occurred: {e}"
            )

    @abstractmethod
    def _validate_inputs(self, input_data: dict) -> None:
        """Hook for subclasses to implement specific input validation."""
        raise NotImplementedError

    @abstractmethod
    async def _execute(self, **kwargs: Any) -> AgentResult:
        """Hook for subclasses to implement the core agent logic."""
        raise NotImplementedError
