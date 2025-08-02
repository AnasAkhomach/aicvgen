"""Base classes for agents."""

from abc import ABC, abstractmethod
from typing import Any

from src.config.logging_config import get_structured_logger
from src.error_handling.exceptions import AgentExecutionError
from src.models.agent_input_models import extract_agent_inputs

from src.orchestration.state import GlobalState
from src.services.progress_tracker import ProgressTracker


class AgentBase(ABC):
    """Abstract base class for all agents."""

    def __init__(
        self,
        name: str,
        description: str,
        session_id: str,
        settings: dict[str, Any] | None = None,
    ):
        """Initializes the agent."""
        self.name = name
        self.description = description
        self.session_id = session_id
        self.settings = settings or {}
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

    async def run(self, **kwargs: Any) -> dict[str, Any]:
        """
        Template method for agent execution with standardized validation and error handling.

        Args:
            **kwargs: The input data for the agent, passed as keyword arguments.

        Returns:
            A dictionary containing the agent's output data or error information.
        """
        self.update_progress(0, f"Starting {self.name} execution.")
        input_data = kwargs  # Use kwargs directly as input_data

        try:
            return await self._execute(**kwargs)
        except AgentExecutionError as e:
            self.logger.error(
                f"Agent execution error in {self.name}: {e}", exc_info=True
            )
            return {"error_messages": [str(e)]}
        except (AttributeError, TypeError, ValueError, KeyError) as e:
            self.logger.error(
                f"An unexpected error occurred in {self.name}: {e}", exc_info=True
            )
            return {"error_messages": [f"An unexpected error occurred: {e}"]}

    async def run_as_node(self, state: GlobalState) -> dict[str, Any]:
        """
        Executes the agent as a LangGraph node, returning a dictionary for state updates.
        This method extracts necessary inputs from the GlobalState using explicit
        input mapping, runs the agent, and returns the results or errors as a dictionary.
        """
        self.session_id = state.get(
            "session_id", ""
        )  # Ensure agent uses current session ID from state
        self.logger = get_structured_logger(
            f"{self.name}:{self.session_id}"
        )  # Update logger with current session ID

        try:
            # Extract agent-specific inputs using explicit input mapping
            # This reduces coupling between agents and the global state
            agent_input_kwargs = extract_agent_inputs(self.name, state)
        except ValueError as e:
            # If input extraction fails, log error and return error dictionary
            self.logger.error(f"Input extraction failed for {self.name}: {e}")
            error_messages = list(state.get("error_messages", []))
            error_messages.append(f"Input extraction failed: {e}")
            return {"error_messages": error_messages}

        # Call the agent's core run method with validated inputs
        # The run method now returns a dictionary directly
        return await self.run(**agent_input_kwargs)

    @abstractmethod
    async def _execute(self, **kwargs: Any) -> dict[str, Any]:
        """Hook for subclasses to implement the core agent logic."""
        raise NotImplementedError
