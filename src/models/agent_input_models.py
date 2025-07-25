"""Agent-specific input models for explicit input mapping.

This module defines Pydantic input models for each agent to reduce coupling
between agents and the global GlobalState. Each model explicitly declares
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
    """Input model for ExecutiveSummaryWriterAgent following Gold Standard LCEL pattern."""

    job_description: str = Field(description="The job description text")
    key_qualifications: str = Field(description="Extracted key qualifications from CV")
    professional_experience: str = Field(description="Extracted professional experience from CV")
    projects: str = Field(description="Extracted projects from CV")
    research_findings: Optional[dict] = Field(default=None, description="Research findings if available")


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

    main_job_description_raw: str = Field(
        description="Raw job description text"
    )
    my_talents: str = Field(
        description="Summary of candidate's talents and experience"
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


class ProfessionalExperienceUpdaterAgentInput(BaseModel):
    """Input model for ProfessionalExperienceUpdaterAgent."""

    structured_cv: StructuredCV = Field(description="The structured CV data")
    generated_professional_experience: str = Field(
        description="Generated professional experience content to update the CV with"
    )


class ProjectsUpdaterAgentInput(BaseModel):
    """Input model for ProjectsUpdaterAgent."""

    structured_cv: StructuredCV = Field(description="The structured CV data")
    generated_projects: str = Field(
        description="Generated projects content to update the CV with"
    )


class ExecutiveSummaryUpdaterAgentInput(BaseModel):
    """Input model for ExecutiveSummaryUpdaterAgent."""

    structured_cv: StructuredCV = Field(description="The structured CV data")
    generated_executive_summary: str = Field(
        description="Generated executive summary content to update the CV with"
    )


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
    "ProfessionalExperienceUpdaterAgent": ProfessionalExperienceUpdaterAgentInput,
    "ProjectsUpdaterAgent": ProjectsUpdaterAgentInput,
    "ExecutiveSummaryUpdaterAgent": ExecutiveSummaryUpdaterAgentInput,
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


def extract_agent_inputs(agent_name: str, state: "GlobalState") -> Dict[str, Any]:
    """Extract and validate agent-specific inputs from GlobalState.

    Args:
        agent_name: Name of the agent
        state: The GlobalState object

    Returns:
        Dictionary of validated inputs for the agent

    Raises:
        ValueError: If agent input model not found or validation fails
    """
    input_model_class = get_agent_input_model(agent_name)
    if not input_model_class:
        raise ValueError(f"No input model found for agent: {agent_name}")

    # Get model fields first
    model_fields = input_model_class.model_fields

    # Map common state fields to agent inputs
    agent_inputs = {}
    
    # Only add session_id if the model has this field
    if "session_id" in model_fields:
        agent_inputs["session_id"] = state.get("session_id")

    # Add agent-specific field mappings

    if "structured_cv" in model_fields:
        agent_inputs["structured_cv"] = state.get("structured_cv")

    if "job_description_data" in model_fields:
        agent_inputs["job_description_data"] = state.get("job_description_data")

    if "cv_data" in model_fields:
        agent_inputs["cv_data"] = state.get("structured_cv")

    if "job_description" in model_fields:
        # For ExecutiveSummaryWriter, map job_description as string
        if agent_name == "ExecutiveSummaryWriter":
            job_desc_data = state.get("job_description_data")
            agent_inputs["job_description"] = job_desc_data.raw_text if job_desc_data else ""
        else:
            agent_inputs["job_description"] = state.get("job_description_data")

    if "cv_text" in model_fields:
        agent_inputs["cv_text"] = state.get("cv_text")

    if "current_item_id" in model_fields:
        agent_inputs["current_item_id"] = state.get("current_item_id")

    if "research_findings" in model_fields:
        agent_inputs["research_findings"] = state.get("research_findings")

    if "user_feedback" in model_fields:
        agent_inputs["user_feedback"] = state.get("user_feedback")

    if "generated_key_qualifications" in model_fields:
        agent_inputs["generated_key_qualifications"] = state.get("generated_key_qualifications", [])

    if "generated_professional_experience" in model_fields:
        agent_inputs["generated_professional_experience"] = state.get("generated_professional_experience", "")

    if "generated_projects" in model_fields:
        agent_inputs["generated_projects"] = state.get("generated_projects", "")

    if "generated_executive_summary" in model_fields:
        agent_inputs["generated_executive_summary"] = state.get("generated_executive_summary", "")

    if "raw_data" in model_fields:
        agent_inputs["raw_data"] = state.get("raw_data", "")

    if "data_type" in model_fields:
        agent_inputs["data_type"] = state.get("data_type", "unknown")

    if "job_description_text" in model_fields:
        agent_inputs["job_description_text"] = state.get("job_description_text", "")

    # Special handling for ExecutiveSummaryWriter string fields
    if agent_name == "ExecutiveSummaryWriter":
        structured_cv = state.get("structured_cv")
        if structured_cv and "key_qualifications" in model_fields:
            # Extract content from Key Qualifications section
            key_qual_section = next((s for s in structured_cv.sections if s.name == "Key Qualifications"), None)
            agent_inputs["key_qualifications"] = "\n".join([item.content for item in key_qual_section.items]) if key_qual_section else ""
        if structured_cv and "professional_experience" in model_fields:
            # Extract content from Professional Experience section
            prof_exp_section = next((s for s in structured_cv.sections if s.name == "Professional Experience"), None)
            agent_inputs["professional_experience"] = "\n".join([item.content for item in prof_exp_section.items]) if prof_exp_section else ""
        if structured_cv and "projects" in model_fields:
            # Extract content from Project Experience section
            projects_section = next((s for s in structured_cv.sections if s.name == "Project Experience"), None)
            agent_inputs["projects"] = "\n".join([item.content for item in projects_section.items]) if projects_section else ""

    # Special handling for KeyQualificationsWriter
    if agent_name == "KeyQualificationsWriter":
        if "main_job_description_raw" in model_fields:
            job_desc_data = state.get("job_description_data")
            agent_inputs["main_job_description_raw"] = job_desc_data.raw_text if job_desc_data else ""
        if "my_talents" in model_fields:
            structured_cv = state.get("structured_cv")
            if structured_cv:
                # Extract talents from Professional Experience and Key Qualifications sections
                talents = []
                for section in structured_cv.sections:
                    if section.name in ["Professional Experience", "Key Qualifications"]:
                        talents.extend([item.content for item in section.items])
                agent_inputs["my_talents"] = "\n".join(talents)



    # Validate inputs using the agent's input model
    try:
        validated_inputs = input_model_class(**agent_inputs)
        return validated_inputs.model_dump()
    except Exception as e:
        raise ValueError(f"Input validation failed for {agent_name}: {e}")
