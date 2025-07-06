"""Models related to agent execution context, results, and base schemas."""

from dataclasses import dataclass, field
from typing import Any, Dict, Generic, Optional, TypeVar, Union

from pydantic import BaseModel, Field, model_validator, ConfigDict
from src.models.validation_schemas import validate_agent_result_output_data

T_Model = TypeVar("T_Model", bound=Union[BaseModel, Dict[str, BaseModel]])


@dataclass
class AgentExecutionContext:
    """Context information for agent execution."""

    session_id: str
    item_id: Optional[str] = None
    content_type: Optional[str] = None
    retry_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    input_data: Optional[Dict[str, Any]] = field(default_factory=dict)
    processing_options: Optional[Dict[str, Any]] = field(default_factory=dict)


class AgentResult(BaseModel, Generic[T_Model]):
    """Structured result from agent execution. Enforces Pydantic model output_data."""

    success: bool
    output_data: Optional[T_Model] = None
    confidence_score: float = 1.0
    processing_time: float = 0.0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def check_output_data(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that output_data is a Pydantic model or a dict of them if success is True."""
        return validate_agent_result_output_data(values)

    @classmethod
    def create_success(
        cls,
        agent_name: str,
        output_data: T_Model,
        message: str = "Operation successful.",
        **kwargs,
    ) -> "AgentResult[T_Model]":
        """Create a success result."""
        return cls(
            success=True,
            output_data=output_data,
            metadata={"agent_name": agent_name, "message": message, **kwargs},
        )

    @classmethod
    def create_failure(
        cls, agent_name: str, error_message: str, **kwargs
    ) -> "AgentResult[T_Model]":
        """Create a failure result."""
        return cls(
            success=False,
            output_data=None,
            error_message=error_message,
            metadata={"agent_name": agent_name, **kwargs},
        )

    def was_successful(self) -> bool:
        """Check if the agent execution was successful."""
        return self.success

    def get_error_message(self) -> Optional[str]:
        """Get the error message if the execution failed."""
        return self.error_message

    model_config = ConfigDict(arbitrary_types_allowed=True)
