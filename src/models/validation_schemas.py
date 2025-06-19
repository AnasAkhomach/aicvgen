"""Validation schemas and functions for agent inputs and outputs.

This module provides validation functions and error handling for agent inputs and outputs."""

from typing import Any, List, Optional
from pydantic import BaseModel, Field, ValidationError
import logging

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
        ..., description="A list of 3-5 generated resume bullet points tailored to the job description."
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
    key_strengths: List[str] = Field(..., description="List of key professional strengths highlighted.")
    career_focus: str = Field(..., description="Primary career focus or objective.")


class LLMQualificationsOutput(BaseModel):
    """Schema for validating the JSON output for generating key qualifications."""
    qualifications: List[str] = Field(..., description="List of key qualifications and competencies.")
    technical_skills: List[str] = Field(..., description="List of technical skills to highlight.")
    soft_skills: List[str] = Field(..., description="List of soft skills and interpersonal abilities.")


def validate_agent_input(agent_type: str, input_data: Any) -> Any:
    """Validate agent input data and return validated model.

    Args:
        agent_type: The type of agent (e.g., 'research', 'qa')
        input_data: The input data to validate

    Returns:
        Any: Validated input data (returns original data if no specific validation)
    """
    try:
        # For now, return the input data as-is since we don't have specific
        # validation schemas for each agent type
        # NOTE: Specific validation schemas can be implemented per agent type as needed
        return input_data
    except ValueError as e:
        logger.error("Validation error for %s: %s", agent_type, e)
        raise ValueError(f"Input validation failed for {agent_type}: {e}") from e


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
                logger.warning("Missing required fields: %s", missing_fields)
                return False
        return True
    except (TypeError, AttributeError) as e:
        logger.error("Output validation error: %s", e)
        return False
