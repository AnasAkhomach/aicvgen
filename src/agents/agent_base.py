"""Enhanced agent base module with Phase 1 infrastructure integration."""

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional, TYPE_CHECKING, Union
from pydantic import BaseModel, model_validator

from ..config.logging_config import get_structured_logger
from ..core.async_optimizer import optimize_async
from ..models.data_models import (
    AgentExecutionLog,
    AgentDecisionLog,
    AgentIO,
)
from ..services.progress_tracker import ProgressTracker

if TYPE_CHECKING:
    from ..orchestration.state import AgentState


@dataclass
class AgentExecutionContext:
    """Context information for agent execution."""

    session_id: str
    item_id: Optional[str] = None
    content_type: Optional[str] = None
    retry_count: int = 0
    metadata: Dict[str, Any] = None
    input_data: Optional[Dict[str, Any]] = None
    processing_options: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.input_data is None:
            self.input_data = {}
        if self.processing_options is None:
            self.processing_options = {}


class AgentResult(BaseModel):
    """Structured result from agent execution. Enforces Pydantic model output_data."""

    success: bool
    output_data: Union[BaseModel, Dict[str, BaseModel]]
    confidence_score: float = 1.0
    processing_time: float = 0.0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None

    @model_validator(mode="before")
    @classmethod
    def check_output_data_is_pydantic(cls, values):
        od = values.get("output_data")
        if od is None:
            raise ValueError("output_data must not be None")
        # Allow single Pydantic model
        if isinstance(od, BaseModel):
            return values
        # Allow dict of Pydantic models
        if isinstance(od, dict):
            if all(isinstance(v, BaseModel) for v in od.values()):
                return values
            raise TypeError("All values in output_data dict must be Pydantic models")
        raise TypeError(
            "output_data must be a Pydantic model or dict of Pydantic models"
        )

    class Config:
        arbitrary_types_allowed = True


class EnhancedAgentBase(ABC):
    """Enhanced abstract base class for all agents with Phase 1 infrastructure integration."""

    def __init__(
        self,
        name: str,
        description: str,
        input_schema: AgentIO,
        output_schema: AgentIO,
        progress_tracker: ProgressTracker,
        content_type: Optional[str] = None,
    ):
        """Initializes the EnhancedAgentBase with required dependencies.

        Args:
            name: The name of the agent.
            description: The description of the agent.
            input_schema: The input schema of the agent.
            output_schema: The output schema of the agent.
            progress_tracker: Progress tracker service dependency.
            content_type: The type of content this agent processes.
        """
        self.name = name
        self.description = description
        self.input_schema = input_schema
        self.output_schema = output_schema
        self.content_type = (
            content_type  # Required service dependencies (constructor injection)
        )
        self.progress_tracker = progress_tracker

        # Performance tracking
        self.execution_count = 0
        self.total_processing_time = 0.0
        self.success_count = 0
        self.error_count = 0

        self.logger = get_structured_logger(__name__)
        if not self.logger or getattr(self.logger, "info", None) is None:
            import logging

            self.logger = logging.getLogger(self.__class__.__name__)

        self.logger.info(
            "Agent initialized",
            agent_name=name,
            content_type=content_type.value if content_type else None,
            description=description,
        )

    @abstractmethod
    async def run_async(
        self, input_data: Any, context: AgentExecutionContext
    ) -> AgentResult:
        """Abstract async method to be implemented by each agent."""
        raise NotImplementedError

    async def execute_with_context(
        self, input_data: Any, context: AgentExecutionContext
    ) -> AgentResult:
        """
        Execute the agent with basic context tracking.

        Note: Retry logic and error recovery have been moved to the orchestration layer
        as part of Task A-04. Agents now fail fast and let the workflow handle recovery.

        Args:
            input_data: The input data for the agent
            context: Execution context with session info

        Returns:
            AgentResult with execution details
        """
        start_time = datetime.now()
        self.execution_count += 1

        # Log execution start with structured logging
        start_log = AgentExecutionLog(
            timestamp=start_time.isoformat(),
            agent_name=self.name,
            session_id=(
                getattr(context, "trace_id", "unknown") if context else "unknown"
            ),
            item_id=getattr(context, "current_item_id", None) if context else None,
            content_type=context.content_type.value if context.content_type else None,
            execution_phase="start",
            retry_count=context.retry_count,
            input_data_type=type(input_data).__name__,
            metadata={
                "agent_description": self.description,
                "execution_count": self.execution_count,
            },
        )
        self.logger.log_agent_execution(start_log)

        # Update progress
        if context.item_id:
            self.progress_tracker.record_item_started(
                context.session_id,
                context.item_id,
                context.content_type or self.content_type,
            )

        try:
            # Execute the agent
            result = await self.run_async(input_data, context)

            processing_time = (datetime.now() - start_time).total_seconds()
            result.processing_time = processing_time
            self.total_processing_time += processing_time

            if result.success:
                self.success_count += 1

                # Log successful execution with structured logging
                success_log = AgentExecutionLog(
                    timestamp=datetime.now().isoformat(),
                    agent_name=self.name,
                    session_id=context.session_id,
                    item_id=context.item_id,
                    content_type=(
                        context.content_type.value if context.content_type else None
                    ),
                    execution_phase="success",
                    processing_time_seconds=processing_time,
                    confidence_score=result.confidence_score,
                    retry_count=context.retry_count,
                    output_data_size=(
                        len(str(result.output_data)) if result.output_data else 0
                    ),
                    metadata={
                        "success_count": self.success_count,
                        "total_processing_time": self.total_processing_time,
                    },
                )
                self.logger.log_agent_execution(success_log)

                # Update progress
                if context.item_id:
                    await self.progress_tracker.record_item_completion(
                        context.session_id,
                        context.item_id,
                        context.content_type or self.content_type,
                        True,
                    )

                return result

            # If agent returned a failure result, raise an exception for orchestration to handle
            error_msg = result.error_message or "Agent execution failed"
            raise RuntimeError(error_msg)

        except Exception as e:
            self.error_count += 1
            processing_time = (datetime.now() - start_time).total_seconds()

            # Log failure with structured logging
            error_log = AgentExecutionLog(
                timestamp=datetime.now().isoformat(),
                agent_name=self.name,
                session_id=context.session_id,
                item_id=context.item_id,
                content_type=(
                    context.content_type.value if context.content_type else None
                ),
                execution_phase="error",
                processing_time_seconds=processing_time,
                retry_count=context.retry_count,
                error_message=str(e),
                metadata={
                    "error_count": self.error_count,
                    "total_processing_time": self.total_processing_time,
                },
            )
            self.logger.log_agent_execution(error_log)

            # Update progress with failure
            if context.item_id:
                self.progress_tracker.record_item_failed(
                    context.session_id,
                    context.item_id,
                    context.content_type or self.content_type,
                    str(e),
                )

            # Fail fast - let the orchestration layer handle retry/recovery
            from ..utils.exceptions import AgentExecutionError

            raise AgentExecutionError(agent_name=self.name, message=str(e)) from e

    def log_decision(
        self,
        message: str,
        context: Optional[AgentExecutionContext] = None,
        decision_type: str = "processing",
        confidence_score: Optional[float] = None,
    ):
        """
        Logs a decision or action taken by the agent with structured logging.

        Args:
            message: The message to log.
            context: Optional execution context for additional metadata.
            decision_type: Type of decision being made.
            confidence_score: Optional confidence score for the decision.
        """
        decision_log = AgentDecisionLog(
            timestamp=datetime.now().isoformat(),
            agent_name=self.name,
            session_id=(
                getattr(context, "trace_id", "unknown") if context else "unknown"
            ),
            item_id=getattr(context, "current_item_id", None) if context else None,
            decision_type=decision_type,
            decision_details=message,
            confidence_score=confidence_score,
            metadata={
                "content_type": (
                    getattr(context, "content_type", None).value
                    if context and getattr(context, "content_type", None) is not None
                    else None
                ),
                "retry_count": getattr(context, "retry_count", 0) if context else 0,
                "execution_count": self.execution_count,
                "success_rate": self.success_count / max(self.execution_count, 1),
            },
        )

        self.logger.log_agent_decision(decision_log)

    def generate_explanation(
        self,
        input_data: Any,
        _output_data: Any,
        _context: Optional[AgentExecutionContext] = None,
    ) -> str:
        """
        Generates an explanation for the agent's decision-making process.

        Args:
            input_data: The input data provided to the agent.
            output_data: The output data generated by the agent.
            context: Optional execution context.

        Returns:
            A string explanation of the decision-making process.
        """
        explanation = (
            f"Agent '{self.name}' processed the input data and generated the following output:\n"
            f"Input: {str(input_data)[:200]}{'...' if len(str(input_data)) > 200 else ''}\n"
            f"Output: {str(_output_data)[:200]}{'...' if len(str(_output_data)) > 200 else ''}\n"
            f"Description: {self.description}\n"
        )

        if _context:
            explanation += (
                f"Session ID: {_context.session_id}\n"
                f"Item ID: {_context.item_id}\n"
                f"Content Type: {_context.content_type.value if _context.content_type else 'Unknown'}\n"
                f"Retry Count: {_context.retry_count}\n"
            )

        return explanation

    def get_confidence_score(self, _output_data: Any) -> float:
        """
        Returns a confidence score for the agent's output.

        Args:
            output_data: The output data generated by the agent.

        Returns:
            A float representing the confidence score (0.0 to 1.0).
        """
        # Placeholder implementation; override in subclasses for specific logic
        return 1.0

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get agent performance statistics."""
        return {
            "agent_name": self.name,
            "execution_count": self.execution_count,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "success_rate": self.success_count / max(self.execution_count, 1),
            "total_processing_time": self.total_processing_time,
            "average_processing_time": self.total_processing_time
            / max(self.execution_count, 1),
            "content_type": self.content_type.value if self.content_type else None,
        }

    def reset_stats(self):
        """Reset performance statistics."""
        self.execution_count = 0
        self.total_processing_time = 0.0
        self.success_count = 0
        self.error_count = 0

        self.logger.info("Agent statistics reset", agent_name=self.name)

    async def _generate_and_parse_json(
        self, prompt: str, session_id: str, trace_id: str
    ) -> Dict[str, Any]:
        """
        Generates content from the LLM and robustly parses the JSON output.
        Handles common LLM response formats like markdown code blocks.

        This method consolidates the common pattern of:
        1. Sending a prompt to an LLM
        2. Receiving a response
        3. Parsing the response as JSON
        4. Handling errors and retries

        Args:
            prompt: The prompt to send to the LLM
            session_id: Session identifier for tracking
            trace_id: Trace identifier for debugging

        Returns:
            Dict[str, Any]: Parsed JSON response from the LLM

        Raises:
            AgentError: If LLM call fails or JSON parsing fails after retries
        """
        # Use the agent's LLM service if available, otherwise get default
        llm_service = getattr(self, "llm", None)
        if llm_service is None:
            raise ValueError(
                f"LLM service unavailable: No llm_service provided to agent {self.name}"
            )

        try:
            # Generate response from LLM
            llm_response = await llm_service.generate_content(
                prompt=prompt, session_id=session_id, trace_id=trace_id
            )

        except Exception as e:
            self.logger.error(
                "LLM generation failed for agent %s: %s", self.name, str(e)
            )
            self.logger.debug("Prompt preview: %s", prompt[:200])
            raise

        # Check if LLMResponse indicates failure
        if not llm_response.success:
            from ..utils.exceptions import (
                AgentExecutionError,
            )  # pylint: disable=import-outside-toplevel

            raise AgentExecutionError(
                self.name, f"LLM generation failed: {llm_response.error_message}"
            )

        raw_text = llm_response.content

        # Check for empty or invalid content
        if not raw_text or raw_text.strip() == "":
            self.logger.error("Empty response received for agent %s", self.name)
            self.logger.debug("Response preview: %s", str(llm_response)[:200])
            raise ValueError(
                "Received empty response from LLM"
            )  # Regex to find JSON within ```json ... ``` code blocks
        json_match = re.search(r"```json\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Fallback for raw JSON or JSON embedded in text
            start_index = raw_text.find("{")
            end_index = raw_text.rfind("}") + 1
            if start_index == -1 or end_index == 0:
                raise ValueError("No valid JSON object found in the LLM response.")
            json_str = raw_text[start_index:end_index]

        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            self.logger.error("Failed to decode JSON from LLM response: %s", e)
            self.logger.debug("Malformed JSON string: %s", json_str)
            raise ValueError(
                f"Could not parse JSON from LLM response. Error: {e}"
            ) from e

    @abstractmethod
    @optimize_async("agent_execution", "base_agent")
    async def run_as_node(self, state: "AgentState") -> Dict[str, Any]:
        """
        Executes the agent's logic as a node within the LangGraph.

        This method must be implemented by all concrete agent classes. It takes
        the current workflow state and returns a dictionary containing only the
        slice of the state that has been modified.

        Args:
            state (AgentState): The current state of the LangGraph workflow.

        Returns:
            Dict[str, Any]: A dictionary with keys matching AgentState fields
                            that have been updated.
        """
        raise NotImplementedError("Subclasses must implement run_as_node method")
