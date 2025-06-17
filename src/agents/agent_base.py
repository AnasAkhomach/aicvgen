"""Enhanced agent base module with Phase 1 infrastructure integration."""

import asyncio
import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional, TYPE_CHECKING

from src.config.logging_config import get_structured_logger
from src.core.async_optimizer import optimize_async
from src.core.state_manager import AgentIO
from src.models.data_models import AgentExecutionLog, AgentDecisionLog
from src.models.data_models import ContentType
from src.services.error_recovery import get_error_recovery_service
from src.services.llm_service import get_llm_service
from src.services.progress_tracker import get_progress_tracker
from src.services.session_manager import get_session_manager

if TYPE_CHECKING:
    from src.orchestration.state import AgentState


@dataclass
class AgentExecutionContext:
    """Context information for agent execution."""

    session_id: str
    item_id: Optional[str] = None
    content_type: Optional[ContentType] = None
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


@dataclass
class AgentResult:
    """Structured result from agent execution."""

    success: bool
    output_data: Any
    confidence_score: float = 1.0
    processing_time: float = 0.0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class EnhancedAgentBase(ABC):
    """Enhanced abstract base class for all agents with Phase 1 infrastructure integration."""

    def __init__(
        self,
        name: str,
        description: str,
        input_schema: AgentIO,
        output_schema: AgentIO,
        content_type: Optional[ContentType] = None,
    ):
        """Initializes the EnhancedAgentBase with the given attributes.

        Args:
            name: The name of the agent.
            description: The description of the agent.
            input_schema: The input schema of the agent.
            output_schema: The output schema of the agent.
            content_type: The type of content this agent processes.
        """
        self.name = name
        self.description = description
        self.input_schema = input_schema
        self.output_schema = output_schema
        self.content_type = content_type

        # Enhanced services
        self.logger = get_structured_logger(f"agent_{name.lower()}")
        self.error_recovery = get_error_recovery_service()
        self.progress_tracker = get_progress_tracker()
        self.session_manager = get_session_manager()

        # Performance tracking
        self.execution_count = 0
        self.total_processing_time = 0.0
        self.success_count = 0
        self.error_count = 0

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

    # Legacy run method removed - use run_as_node for LangGraph integration

    async def execute_with_context(
        self, input_data: Any, context: AgentExecutionContext, max_retries: int = 3
    ) -> AgentResult:
        """
        Execute the agent with full context and error handling.

        Args:
            input_data: The input data for the agent
            context: Execution context with session info
            max_retries: Maximum number of retries on failure

        Returns:
            AgentResult with execution details
        """
        start_time = datetime.now()
        self.execution_count += 1

        # Log execution start with structured logging
        start_log = AgentExecutionLog(
            timestamp=start_time.isoformat(),
            agent_name=self.name,
            session_id=context.session_id,
            item_id=context.item_id,
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
            await self.progress_tracker.record_item_start(
                context.session_id,
                context.item_id,
                context.content_type or self.content_type,
            )

        retry_count = 0
        while retry_count <= max_retries:
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
                        retry_count=retry_count,
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

                error_msg = result.error_message or "Agent execution failed"
                raise RuntimeError(error_msg)

            except (RuntimeError, ValueError, TypeError, AttributeError) as e:
                self.error_count += 1
                processing_time = (datetime.now() - start_time).total_seconds()

                # Handle error with recovery service
                recovery_action = await self.error_recovery.handle_error(
                    e,
                    context.item_id or "unknown",
                    context.content_type or self.content_type,
                    context.session_id,
                    retry_count,
                    {
                        "agent_name": self.name,
                        "input_data_type": type(input_data).__name__,
                    },
                )

                # Check if we should retry
                if (
                    recovery_action.strategy.value == "retry"
                    and retry_count < max_retries
                ):
                    retry_count += 1
                    context.retry_count = retry_count

                    # Log retry with structured logging
                    retry_log = AgentExecutionLog(
                        timestamp=datetime.now().isoformat(),
                        agent_name=self.name,
                        session_id=context.session_id,
                        item_id=context.item_id,
                        content_type=(
                            context.content_type.value if context.content_type else None
                        ),
                        execution_phase="retry",
                        processing_time_seconds=processing_time,
                        retry_count=retry_count,
                        error_message=str(e),
                        metadata={
                            "delay_seconds": recovery_action.delay_seconds,
                            "recovery_strategy": recovery_action.strategy.value,
                            "error_count": self.error_count,
                        },
                    )
                    self.logger.log_agent_execution(retry_log)

                    if recovery_action.delay_seconds > 0:
                        await asyncio.sleep(recovery_action.delay_seconds)
                    continue

                # Log final failure with structured logging
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
                    retry_count=retry_count,
                    error_message=str(e),
                    metadata={
                        "recovery_strategy": recovery_action.strategy.value,
                        "fallback_used": bool(recovery_action.fallback_content),
                        "error_count": self.error_count,
                        "total_processing_time": self.total_processing_time,
                    },
                )
                self.logger.log_agent_execution(error_log)

                # Update progress with failure
                if context.item_id:
                    await self.progress_tracker.record_item_completion(
                        context.session_id,
                        context.item_id,
                        context.content_type or self.content_type,
                        False,
                    )

                # Return error result with fallback content if available
                return AgentResult(
                    success=False,
                    output_data=recovery_action.fallback_content
                    or f"Failed to process with {self.name}: {str(e)}",
                    confidence_score=0.0,
                    processing_time=processing_time,
                    error_message=str(e),
                    metadata={
                        "retry_count": retry_count,
                        "fallback_used": bool(recovery_action.fallback_content),
                        "agent_name": self.name,
                    },
                )

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
            session_id=context.session_id if context else "unknown",
            item_id=context.item_id if context else None,
            decision_type=decision_type,
            decision_details=message,
            confidence_score=confidence_score,
            metadata={
                "content_type": (
                    context.content_type.value
                    if context and context.content_type
                    else None
                ),
                "retry_count": context.retry_count if context else 0,
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
        self,
        prompt: str,
        model_name: str = "gemini-2.0-flash",
        temperature: float = 0.7,
    ) -> Dict[str, Any]:
        """
        Shared utility method for LLM interaction with JSON parsing.

        This method consolidates the common pattern of:
        1. Sending a prompt to an LLM
        2. Receiving a response
        3. Parsing the response as JSON
        4. Handling errors and retries

        Args:
            prompt: The prompt to send to the LLM
            model_name: The model to use (default: gemini-2.0-flash)
            temperature: The temperature setting for generation (default: 0.7)

        Returns:
            Dict[str, Any]: Parsed JSON response from the LLM

        Raises:
            Exception: If LLM call fails or JSON parsing fails after retries
        """
        llm_service = get_llm_service()

        try:
            # Generate response from LLM
            response = await llm_service.generate_async(
                prompt=prompt, model=model_name, temperature=temperature
            )

        except Exception as e:
            self.logger.error(
                f"LLM generation failed for agent {self.name}",
                error=str(e),
                model=model_name,
                prompt_preview=prompt[:200],
            )
            raise

        # Clean and extract JSON from response
        cleaned_response = self._extract_json_from_response(response)

        # Parse JSON
        try:
            parsed_json = json.loads(cleaned_response)
            return parsed_json
        except json.JSONDecodeError as e:
            self.logger.error(
                f"JSON parsing failed for agent {self.name}",
                error=str(e),
                response_preview=cleaned_response[:200],
            )
            raise ValueError(f"Failed to parse JSON response: {str(e)}") from e

    def _extract_json_from_response(self, response: str) -> str:
        """
        Extract JSON content from LLM response, handling common formatting issues.

        Args:
            response: Raw response from LLM

        Returns:
            str: Cleaned JSON string
        """
        # Remove markdown code blocks
        response = re.sub(r"```json\s*", "", response)
        response = re.sub(r"```\s*$", "", response)

        # Remove leading/trailing whitespace
        response = response.strip()

        # Try to find JSON object boundaries
        json_start = response.find("{")
        json_end = response.rfind("}") + 1

        if json_start != -1 and json_end > json_start:
            return response[json_start:json_end]

        # If no clear JSON boundaries, return cleaned response
        return response

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


# Legacy AgentBase class removed - all agents now use EnhancedAgentBase
