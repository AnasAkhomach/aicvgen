"""Base classes for agents."""

from abc import ABC, abstractmethod
from typing import Any

from src.config.logging_config import get_structured_logger
from src.error_handling.exceptions import AgentExecutionError
from src.models.agent_input_models import extract_agent_inputs
from src.models.agent_models import AgentResult
from src.orchestration.state import AgentState
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

    async def run(self, **kwargs: Any) -> AgentResult:
        """
        Template method for agent execution with standardized validation and error handling.

        Args:
            **kwargs: The input data for the agent, passed as keyword arguments.

        Returns:
            The result of the agent's task as an AgentResult object.
        """
        self.update_progress(0, f"Starting {self.name} execution.")
        input_data = kwargs  # Use kwargs directly as input_data

        try:
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

    async def run_as_node(self, state: AgentState) -> AgentState:
        """
        Executes the agent as a LangGraph node, updating the state.
        This method extracts necessary inputs from the AgentState using explicit
        input mapping, runs the agent, and updates the AgentState with the results or errors.
        """
        self.session_id = (
            state.session_id
        )  # Ensure agent uses current session ID from state
        self.logger = get_structured_logger(
            f"{self.name}:{self.session_id}"
        )  # Update logger with current session ID

        try:
            # Extract agent-specific inputs using explicit input mapping
            # This reduces coupling between agents and the global state
            agent_input_kwargs = extract_agent_inputs(self.name, state)
        except ValueError as e:
            # If input extraction fails, log error and return state with error
            self.logger.error(f"Input extraction failed for {self.name}: {e}")
            error_messages = state.error_messages + [f"Input extraction failed: {e}"]
            return state.model_copy(update={"error_messages": error_messages})

        # Call the agent's core run method with validated inputs
        agent_result = await self.run(**agent_input_kwargs)

        # Update the AgentState based on the agent's result
        if agent_result.was_successful():
            # Assuming output_data is a Pydantic model that can be merged into AgentState
            if agent_result.output_data:
                # Handle specific field mappings for different agent output types
                # For EnhancedContentWriterOutput, we need to preserve the StructuredCV object
                if hasattr(agent_result.output_data, "updated_structured_cv"):
                    # Directly use the StructuredCV object without converting to dict
                    return state.model_copy(
                        update={
                            "structured_cv": agent_result.output_data.updated_structured_cv
                        }
                    )
                else:
                    # For other output types, use the standard dict conversion
                    updated_state_data = agent_result.output_data.model_dump(
                        exclude_unset=True
                    )
                    return state.model_copy(update=updated_state_data)
            return state  # No output data, return original state
        else:
            # If agent failed, add error message to state
            error_messages = state.error_messages + [agent_result.get_error_message()]
            return state.model_copy(update={"error_messages": error_messages})

    @abstractmethod
    async def _execute(self, **kwargs: Any) -> AgentResult:
        """Hook for subclasses to implement the core agent logic."""
        raise NotImplementedError
