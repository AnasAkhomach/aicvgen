"""Pydantic output models for agent results.

This module defines the output data models that agents return in their
AgentResult.output_data field. These models ensure type safety and data
validation for agent outputs according to Task C-03.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

from .data_models import JobDescriptionData, StructuredCV
from .quality_assurance_agent_models import QualityAssuranceResult


class ParserAgentOutput(BaseModel):
    """Output model for ParserAgent run_async method."""

    job_description_data: Optional[JobDescriptionData] = Field(
        default=None, description="Parsed job description data"
    )
    structured_cv: Optional[StructuredCV] = Field(
        default=None, description="Parsed CV data structure"
    )


class EnhancedContentWriterOutput(BaseModel):
    """Output model for EnhancedContentWriterAgent run_async method."""

    content_data: Optional[Dict[str, Any]] = Field(
        default=None, description="Generated content data"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Content generation metadata"
    )


class CleaningAgentOutput(BaseModel):
    """Output model for CleaningAgent run_async method."""

    cleaned_cv: Optional[StructuredCV] = Field(
        default=None, description="Cleaned and standardized CV data"
    )
    modifications_made: List[str] = Field(
        default_factory=list,
        description="List of modifications performed during cleaning",
    )


class CVAnalyzerAgentOutput(BaseModel):
    """Output model for CVAnalyzerAgent run_async method."""

    analysis_results: Optional[Dict[str, Any]] = Field(
        default=None, description="CV analysis results"
    )
    recommendations: List[str] = Field(
        default_factory=list, description="Analysis recommendations"
    )
    compatibility_score: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="CV-job compatibility score"
    )


class QualityAssuranceAgentOutput(BaseModel):
    """Output model for QualityAssuranceAgent run_async method."""

    quality_check_results: Optional[QualityAssuranceResult] = Field(
        default=None, description="Quality assessment results"
    )
    passed_checks: List[str] = Field(
        default_factory=list, description="List of passed quality checks"
    )
    failed_checks: List[str] = Field(
        default_factory=list, description="List of failed quality checks"
    )


class FormatterAgentOutput(BaseModel):
    """Output model for FormatterAgent run_async method."""

    formatted_content: Optional[str] = Field(
        default=None, description="Formatted CV content"
    )
    output_path: Optional[str] = Field(
        default=None, description="Path to the generated output file"
    )
    format_type: Optional[str] = Field(
        default=None, description="Type of formatting applied"
    )


class ResearchAgentOutput(BaseModel):
    """Output model for ResearchAgent run_async method."""

    research_findings: Optional[Dict[str, Any]] = Field(
        default=None, description="Research findings and insights"
    )
    sources: List[str] = Field(
        default_factory=list, description="List of research sources"
    )
    confidence_level: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Confidence level of research findings",
    )
