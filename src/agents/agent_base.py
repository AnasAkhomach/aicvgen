"""Enhanced agent base module with Phase 1 infrastructure integration."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
import logging
import asyncio
from datetime import datetime
from dataclasses import dataclass

from src.core.state_manager import AgentIO
from src.config.logging_config import get_structured_logger
from src.models.data_models import ContentType, ProcessingStatus
from src.services.error_recovery import get_error_recovery_service
from src.services.progress_tracker import get_progress_tracker
from src.services.session_manager import get_session_manager

# Import for type hints - using TYPE_CHECKING to avoid circular imports
from typing import TYPE_CHECKING
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
        content_type: Optional[ContentType] = None
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
            description=description
        )

    @abstractmethod
    async def run_async(self, input_data: Any, context: AgentExecutionContext) -> AgentResult:
        """Abstract async method to be implemented by each agent."""
        raise NotImplementedError
    
    @abstractmethod
    def run(self, input_data: Any) -> Any:
        """Legacy abstract method for backward compatibility."""
        raise NotImplementedError
    
    async def execute_with_context(
        self, 
        input_data: Any, 
        context: AgentExecutionContext,
        max_retries: int = 3
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
        
        # Log execution start
        self.logger.info(
            "Agent execution started",
            agent_name=self.name,
            session_id=context.session_id,
            item_id=context.item_id,
            content_type=context.content_type.value if context.content_type else None,
            retry_count=context.retry_count
        )
        
        # Update progress
        if context.item_id:
            await self.progress_tracker.record_item_start(
                context.session_id, context.item_id, context.content_type or self.content_type
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
                    
                    # Log successful execution
                    self.logger.info(
                        "Agent execution completed successfully",
                        agent_name=self.name,
                        session_id=context.session_id,
                        item_id=context.item_id,
                        processing_time=processing_time,
                        confidence_score=result.confidence_score,
                        retry_count=retry_count
                    )
                    
                    # Update progress
                    if context.item_id:
                        await self.progress_tracker.record_item_completion(
                            context.session_id, context.item_id, 
                            context.content_type or self.content_type, True
                        )
                    
                    return result
                else:
                    raise Exception(result.error_message or "Agent execution failed")
                    
            except Exception as e:
                self.error_count += 1
                processing_time = (datetime.now() - start_time).total_seconds()
                
                # Handle error with recovery service
                recovery_action = await self.error_recovery.handle_error(
                    e, context.item_id or "unknown", 
                    context.content_type or self.content_type, 
                    context.session_id, retry_count, 
                    {"agent_name": self.name, "input_data_type": type(input_data).__name__}
                )
                
                # Check if we should retry
                if recovery_action.strategy.value == "retry" and retry_count < max_retries:
                    retry_count += 1
                    context.retry_count = retry_count
                    
                    self.logger.warning(
                        "Agent execution failed, retrying",
                        agent_name=self.name,
                        session_id=context.session_id,
                        item_id=context.item_id,
                        error=str(e),
                        retry_count=retry_count,
                        delay_seconds=recovery_action.delay_seconds
                    )
                    
                    if recovery_action.delay_seconds > 0:
                        await asyncio.sleep(recovery_action.delay_seconds)
                    continue
                
                # Log final failure
                self.logger.error(
                    "Agent execution failed after retries",
                    agent_name=self.name,
                    session_id=context.session_id,
                    item_id=context.item_id,
                    error=str(e),
                    retry_count=retry_count,
                    processing_time=processing_time
                )
                
                # Update progress with failure
                if context.item_id:
                    await self.progress_tracker.record_item_completion(
                        context.session_id, context.item_id, 
                        context.content_type or self.content_type, False
                    )
                
                # Return error result with fallback content if available
                return AgentResult(
                    success=False,
                    output_data=recovery_action.fallback_content or f"Failed to process with {self.name}: {str(e)}",
                    confidence_score=0.0,
                    processing_time=processing_time,
                    error_message=str(e),
                    metadata={
                        "retry_count": retry_count,
                        "fallback_used": bool(recovery_action.fallback_content),
                        "agent_name": self.name
                    }
                )
    
    def log_decision(self, message: str, context: Optional[AgentExecutionContext] = None):
        """
        Logs a decision or action taken by the agent with structured logging.

        Args:
            message: The message to log.
            context: Optional execution context for additional metadata.
        """
        self.logger.info(
            message,
            agent_name=self.name,
            session_id=context.session_id if context else None,
            item_id=context.item_id if context else None
        )

    def generate_explanation(self, input_data: Any, output_data: Any, context: Optional[AgentExecutionContext] = None) -> str:
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
            f"Output: {str(output_data)[:200]}{'...' if len(str(output_data)) > 200 else ''}\n"
            f"Description: {self.description}\n"
        )
        
        if context:
            explanation += (
                f"Session ID: {context.session_id}\n"
                f"Item ID: {context.item_id}\n"
                f"Content Type: {context.content_type.value if context.content_type else 'Unknown'}\n"
                f"Retry Count: {context.retry_count}\n"
            )
        
        return explanation

    def get_confidence_score(self, output_data: Any) -> float:
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
            "average_processing_time": self.total_processing_time / max(self.execution_count, 1),
            "content_type": self.content_type.value if self.content_type else None
        }
    
    def reset_stats(self):
        """Reset performance statistics."""
        self.execution_count = 0
        self.total_processing_time = 0.0
        self.success_count = 0
        self.error_count = 0
        
        self.logger.info("Agent statistics reset", agent_name=self.name)
    
    async def run_as_node(self, state: "AgentState") -> dict:
        """
        Standard LangGraph node interface for all agents.
        This method should be overridden by subclasses to provide
        agent-specific LangGraph integration logic.
        
        Args:
            state: The current workflow state (AgentState)
            
        Returns:
            dict: Updates to be applied to the state
        """
        raise NotImplementedError(
            f"Agent {self.name} must implement run_as_node method for LangGraph compatibility"
        )


# Legacy compatibility
class AgentBase(EnhancedAgentBase):
    """Legacy AgentBase class for backward compatibility."""
    
    def __init__(self, name: str, description: str, input_schema: AgentIO, output_schema: AgentIO):
        super().__init__(name, description, input_schema, output_schema)
    
    async def run_async(self, input_data: Any, context: AgentExecutionContext) -> AgentResult:
        """Default async implementation that calls the legacy run method."""
        try:
            output = self.run(input_data)
            confidence = self.get_confidence_score(output)
            
            return AgentResult(
                success=True,
                output_data=output,
                confidence_score=confidence,
                metadata={"legacy_execution": True}
            )
        except Exception as e:
            return AgentResult(
                success=False,
                output_data=None,
                confidence_score=0.0,
                error_message=str(e),
                metadata={"legacy_execution": True}
            )
