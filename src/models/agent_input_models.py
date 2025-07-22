"""Agent-specific input models for explicit input mapping.

This module defines Pydantic input models for each agent to reduce coupling
between agents and the global AgentState. Each model explicitly declares
the required inputs for its corresponding agent.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from pydantic import BaseModel, Field

from src.models.agent_output_models import ResearchFindings
from src.models.cv_models import JobDescriptionData, StructuredCV


class CVAnalyzerAgentInput(BaseModel):
    """Input model for CVAnalyzerAgent."""

    cv_data: StructuredCV = Field(description="The structured CV data to analyze")
    job_description: JobDescriptionData = Field(description="The job description data")
    session_id: str = Field(description="Session identifier")


class ExecutiveSummaryWriterAgentInput(BaseModel):
    """Input model for ExecutiveSummaryWriterAgent."""

    structured_cv: StructuredCV = Field(description="The structured CV data")
    job_description_data: JobDescriptionData = Field(
        description="The job description data"
    )
    research_findings: Optional[ResearchFindings] = Field(
        default=None, description="Research findings if available"
    )
    session_id: str = Field(description="Session identifier")


class ProfessionalExperienceWriterAgentInput(BaseModel):
    """Input model for ProfessionalExperienceWriterAgent."""

    structured_cv: StructuredCV = Field(description="The structured CV data")
    job_description_data: JobDescriptionData = Field(
        description="The job description data"
    )
    current_item_id: Optional[str] = Field(
        default=None, description="ID of the current item being processed"
    )
    research_findings: Optional[ResearchFindings] = Field(
        default=None, description="Research findings if available"
    )
    session_id: str = Field(description="Session identifier")


class KeyQualificationsWriterAgentInput(BaseModel):
    """Input model for KeyQualificationsWriterAgent."""

    structured_cv: StructuredCV = Field(description="The structured CV data")
    job_description_data: JobDescriptionData = Field(
        description="The job description data"
    )
    current_item_id: Optional[str] = Field(
        default=None, description="ID of the current item being processed"
    )
    research_findings: Optional[ResearchFindings] = Field(
        default=None, description="Research findings if available"
    )
    session_id: str = Field(description="Session identifier")


class ProjectsWriterAgentInput(BaseModel):
    """Input model for ProjectsWriterAgent."""

    structured_cv: StructuredCV = Field(description="The structured CV data")
    job_description_data: JobDescriptionData = Field(
        description="The job description data"
    )
    current_item_id: Optional[str] = Field(
        default=None, description="ID of the current item being processed"
    )
    research_findings: Optional[ResearchFindings] = Field(
        default=None, description="Research findings if available"
    )
    session_id: str = Field(description="Session identifier")


class KeyQualificationsUpdaterAgentInput(BaseModel):
    """Input model for KeyQualificationsUpdaterAgent."""

    structured_cv: StructuredCV = Field(description="The structured CV data")
    generated_key_qualifications: List[str] = Field(
        description="List of generated key qualifications to update the CV with"
    )
    session_id: str = Field(description="Session identifier")


class ResearchAgentInput(BaseModel):
    """Input model for ResearchAgent."""

    job_description_data: JobDescriptionData = Field(
        description="The job description data"
    )
    structured_cv: StructuredCV = Field(description="The structured CV data")
    session_id: str = Field(description="Session identifier")


class FormatterAgentInput(BaseModel):
    """Input model for FormatterAgent."""

    structured_cv: StructuredCV = Field(description="The structured CV data")
    session_id: str = Field(description="Session identifier")


class QualityAssuranceAgentInput(BaseModel):
    """Input model for QualityAssuranceAgent."""

    structured_cv: StructuredCV = Field(description="The structured CV data")
    job_description_data: JobDescriptionData = Field(
        description="The job description data"
    )
    session_id: str = Field(description="Session identifier")


class CleaningAgentInput(BaseModel):
    """Input model for CleaningAgent."""

    raw_data: Any = Field(description="Raw data to be cleaned")
    data_type: str = Field(description="Type of data being cleaned")
    session_id: str = Field(description="Session identifier")


class UserCVParserAgentInput(BaseModel):
    """Input model for UserCVParserAgent."""

    cv_text: str = Field(description="Raw CV text to parse")
    session_id: str = Field(description="Session identifier")


class JobDescriptionParserAgentInput(BaseModel):
    """Input model for JobDescriptionParserAgent."""

    job_description_text: str = Field(description="Raw job description text to parse")
    session_id: str = Field(description="Session identifier")


# Agent input mapping registry
AGENT_INPUT_MODELS = {
    "CVAnalyzerAgent": CVAnalyzerAgentInput,
    "ExecutiveSummaryWriter": ExecutiveSummaryWriterAgentInput,
    "ProfessionalExperienceWriter": ProfessionalExperienceWriterAgentInput,
    "KeyQualificationsWriter": KeyQualificationsWriterAgentInput,
    "KeyQualificationsUpdaterAgent": KeyQualificationsUpdaterAgentInput,
    "ProjectsWriter": ProjectsWriterAgentInput,
    "ResearchAgent": ResearchAgentInput,
    "FormatterAgent": FormatterAgentInput,
    "QualityAssuranceAgent": QualityAssuranceAgentInput,
    "CleaningAgent": CleaningAgentInput,
    "UserCVParserAgent": UserCVParserAgentInput,
    "JobDescriptionParserAgent": JobDescriptionParserAgentInput,
}


def get_agent_input_model(agent_name: str) -> Optional[BaseModel]:
    """Get the input model class for a specific agent.

    Args:
        agent_name: Name of the agent

    Returns:
        The Pydantic model class for the agent's inputs, or None if not found
    """
    return AGENT_INPUT_MODELS.get(agent_name)


def extract_agent_inputs(agent_name: str, state: "AgentState") -> Dict[str, Any]:
    """Extract and validate agent-specific inputs from AgentState.

    Args:
        agent_name: Name of the agent
        state: The AgentState object

    Returns:
        Dictionary of validated inputs for the agent

    Raises:
        ValueError: If agent input model not found or validation fails
    """
    input_model_class = get_agent_input_model(agent_name)
    if not input_model_class:
        raise ValueError(f"No input model found for agent: {agent_name}")

    # Extract state data as dict
    state_data = state.model_dump()

    # Map common state fields to agent inputs
    agent_inputs = {
        "session_id": state.session_id,
    }

    # Add agent-specific field mappings
    model_fields = input_model_class.model_fields

    if "structured_cv" in model_fields:
        agent_inputs["structured_cv"] = state.structured_cv

    if "job_description_data" in model_fields:
        agent_inputs["job_description_data"] = state.job_description_data

    if "cv_data" in model_fields:
        agent_inputs["cv_data"] = state.structured_cv

    if "job_description" in model_fields:
        agent_inputs["job_description"] = state.job_description_data

    if "cv_text" in model_fields:
        agent_inputs["cv_text"] = state.cv_text

    if "current_item_id" in model_fields:
        agent_inputs["current_item_id"] = state.current_item_id

    if "research_findings" in model_fields:
        agent_inputs["research_findings"] = getattr(state, "research_findings", None)

    if "user_feedback" in model_fields:
        agent_inputs["user_feedback"] = getattr(state, "user_feedback", None)

    if "generated_key_qualifications" in model_fields:
        agent_inputs["generated_key_qualifications"] = getattr(state, "generated_key_qualifications", None)

    # For cleaning agent, handle raw_data and data_type specially
    if agent_name == "CleaningAgent":
        agent_inputs["raw_data"] = getattr(state, "raw_data", None)
        agent_inputs["data_type"] = getattr(state, "data_type", "unknown")

    # For job description parser
    if agent_name == "JobDescriptionParserAgent":
        agent_inputs["job_description_text"] = getattr(
            state, "job_description_text", ""
        )

    # Validate inputs using the agent's input model
    try:
        validated_inputs = input_model_class(**agent_inputs)
        return validated_inputs.model_dump()
    except Exception as e:
        raise ValueError(f"Input validation failed for {agent_name}: {e}")
