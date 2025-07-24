"""Validation schemas and functions for agent inputs and outputs.

This module provides validation functions and error handling for agent inputs and outputs.
"""

import logging
from typing import TYPE_CHECKING, Any, List, Optional

from pydantic import BaseModel, Field
from src.models.cv_models import JobDescriptionData, StructuredCV


if TYPE_CHECKING:
    from src.models.agent_output_models import ResearchFindings
    from src.orchestration.state import GlobalState

logger = logging.getLogger(__name__)


# LLM Output Validation Schemas for JSON-based parsing
class LLMJobDescriptionOutput(BaseModel):
    """Schema for validating the JSON output from the job description parsing LLM call."""

    skills: List[str] = Field(
        ..., description="List of key skills and technologies mentioned."
    )
    experience_level: str = Field(
        ..., description="Required experience level (e.g., Senior, Mid-Level)."
    )
    responsibilities: List[str] = Field(
        ..., description="List of key job responsibilities."
    )
    industry_terms: List[str] = Field(
        ..., description="List of industry-specific terms or jargon."
    )
    company_values: List[str] = Field(
        ..., description="List of company values or cultural keywords."
    )


class LLMRoleGenerationOutput(BaseModel):
    """Schema for validating the JSON output for generating a single resume role."""

    organization_description: Optional[str] = Field(
        description="A brief description of the company."
    )
    role_description: Optional[str] = Field(
        description="A brief description of the role's main purpose."
    )
    bullet_points: List[str] = Field(
        ...,
        description="A list of 3-5 generated resume bullet points tailored to the job description.",
    )


class LLMProjectGenerationOutput(BaseModel):
    """Schema for validating the JSON output for generating project content."""

    project_description: Optional[str] = Field(
        description="A brief description of the project."
    )
    technologies_used: List[str] = Field(
        ..., description="List of technologies and tools used in the project."
    )
    achievements: List[str] = Field(
        ..., description="List of key achievements and outcomes from the project."
    )
    bullet_points: List[str] = Field(
        ..., description="A list of 3-5 generated project bullet points."
    )


class LLMSummaryOutput(BaseModel):
    """Schema for validating the JSON output for generating executive summary."""

    summary_text: str = Field(..., description="The generated executive summary text.")
    key_strengths: List[str] = Field(
        ..., description="List of key professional strengths highlighted."
    )
    career_focus: str = Field(..., description="Primary career focus or objective.")


class LLMQualificationsOutput(BaseModel):
    """Schema for validating the JSON output for generating key qualifications."""

    qualifications: List[str] = Field(
        ..., description="List of key qualifications and competencies."
    )
    technical_skills: List[str] = Field(
        ..., description="List of technical skills to highlight."
    )
    soft_skills: List[str] = Field(
        ..., description="List of soft skills and interpersonal abilities."
    )


# Input schemas for agent validation
class ParserAgentInput(BaseModel):
    cv_text: str
    job_description_data: JobDescriptionData


class ContentWriterAgentInput(BaseModel):
    structured_cv: StructuredCV
    research_findings: Optional["ResearchFindings"] = None
    current_item_id: Optional[str] = None


class ResearchAgentInput(BaseModel):
    job_description_data: JobDescriptionData
    structured_cv: StructuredCV


class QualityAssuranceAgentInput(BaseModel):
    structured_cv: StructuredCV
    current_item_id: Optional[str] = None


class FormatterAgentInput(BaseModel):
    """Input schema for the Formatter Agent."""

    structured_cv: StructuredCV
    job_description_data: Optional[JobDescriptionData] = None


class CVAnalyzerAgentInput(BaseModel):
    """Input schema for the CV Analyzer Agent."""

    cv_text: str
    job_description_data: JobDescriptionData


# CleaningAgentInput moved to agent_input_models.py to avoid circular imports


# validate_agent_input function removed - use ValidatorFactory.validate_agent_input directly


def validate_agent_result_output_data(values: dict) -> dict:
    """Validate that output_data is a Pydantic model or a dict of them if success is True."""
    is_success = values.get("success")
    output_data = values.get("output_data")

    if is_success:
        if output_data is None:
            raise ValueError("output_data must not be None for successful results")

        if not isinstance(output_data, (BaseModel, dict)):
            raise TypeError(
                "output_data must be a Pydantic model or a dictionary of Pydantic models"
            )

        if isinstance(output_data, dict) and not all(
            isinstance(v, BaseModel) for v in output_data.values()
        ):
            raise TypeError(
                "All values in the output_data dictionary must be Pydantic models"
            )

    return values


def validate_agent_output(output_data: Any, required_fields: List[str] = None) -> bool:
    """Validate agent output data.

    Args:
        output_data: The output data to validate
        required_fields: Optional list of required fields for dict outputs

    Returns:
        bool: True if validation passes, False otherwise
    """
    try:
        if required_fields and isinstance(output_data, dict):
            missing_fields = [
                field for field in required_fields if field not in output_data
            ]
            if missing_fields:
                logger.warning("Missing required fields", missing_fields=missing_fields)
                return False
        return True
    except (TypeError, AttributeError) as e:
        logger.error("Output validation error", error=str(e))
        return False
