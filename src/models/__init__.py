"""Models module for the aicvgen application."""

from src.models.data_models import (
    VectorStoreConfig,
    PersonalInfo,
    ItemStatus,
    ProcessingStatus,
    ItemType,
    WorkflowState,
    WorkflowStage,
)

# Import validation functions and schemas
from src.models.validation_schemas import (
    validate_agent_input,
    validate_agent_output,
    LLMJobDescriptionOutput,
    LLMRoleGenerationOutput,
    LLMProjectGenerationOutput,
    LLMSummaryOutput,
    LLMQualificationsOutput,
)
from src.models.agent_output_models import ResearchFindings

__all__ = [
    "VectorStoreConfig",
    "PersonalInfo",
    "ItemStatus",
    "ProcessingStatus",
    "ItemType",
    "WorkflowState",
    "WorkflowStage",
    "validate_agent_input",
    "validate_agent_output",
    "LLMJobDescriptionOutput",
    "LLMRoleGenerationOutput",
    "LLMProjectGenerationOutput",
    "LLMSummaryOutput",
    "LLMQualificationsOutput",
    "ResearchFindings",
]
