"""Pydantic output models for agent results.

This module defines the output data models that agents return in their
AgentResult.output_data field. These models ensure type safety and data
validation for agent outputs according to Task C-03.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator
from enum import Enum
from datetime import datetime

from src.models.data_models import JobDescriptionData, StructuredCV, BasicCVInfo


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

    updated_structured_cv: StructuredCV = Field(
        description="The full CV data structure with the enhanced content updated."
    )
    item_id: str = Field(description="The ID of the item that was enhanced.")
    generated_content: str = Field(description="The newly generated content.")


class CleaningAgentOutput(BaseModel):
    """Output model for CleaningAgent run_async method."""

    cleaned_data: Any = Field(
        description="The cleaned data, which can be a list, dict, or string."
    )
    modifications_made: List[str] = Field(
        default_factory=list,
        description="List of modifications performed during cleaning",
    )
    raw_output: Optional[str] = Field(
        default=None, description="The original raw output that was cleaned."
    )
    output_type: Optional[str] = Field(
        default=None,
        description="The type of output that was cleaned (e.g., 'skills_list').",
    )


class CVAnalysisResult(BaseModel):
    summary: Optional[str] = None
    key_skills: List[str] = Field(default_factory=list)
    skill_matches: List[str] = Field(default_factory=list)
    experience_relevance: float = 0.0
    gaps_identified: List[str] = Field(default_factory=list)
    strengths: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    match_score: float = 0.0
    analysis_timestamp: Optional[str] = None


class CVAnalyzerAgentOutput(BaseModel):
    """Output model for CVAnalyzerAgent run_async method."""

    analysis_results: Optional[CVAnalysisResult] = Field(
        default=None, description="CV analysis results"
    )
    extracted_data: Optional[BasicCVInfo] = Field(
        default=None, description="Fallback basic CV extraction if LLM fails"
    )
    recommendations: List[str] = Field(
        default_factory=list, description="Analysis recommendations"
    )
    compatibility_score: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="CV-job compatibility score"
    )


class ItemQualityResultModel(BaseModel):
    """Model for individual item quality check results."""

    item_id: str
    passed: bool
    issues: List[str]
    suggestions: Optional[List[str]] = None


class SectionQualityResultModel(BaseModel):
    """Model for section-level quality check results."""

    section_name: str
    passed: bool
    issues: List[str]
    item_checks: Optional[List[ItemQualityResultModel]] = Field(default_factory=list)


class OverallQualityCheckResultModel(BaseModel):
    """Model for overall CV quality check results."""

    check_name: str
    passed: bool
    details: Optional[str] = None


class QualityAssuranceAgentOutput(BaseModel):
    """Output model for QualityAssuranceAgent run_async method."""

    section_results: List[SectionQualityResultModel] = Field(
        default_factory=list, description="List of section-level quality results"
    )
    overall_passed: bool = Field(
        default=False, description="True if all sections passed quality checks"
    )
    recommendations: List[str] = Field(
        default_factory=list, description="Suggestions for improving the CV"
    )


class FormatterAgentOutput(BaseModel):
    """Output model for FormatterAgent run method."""

    output_path: str = Field(description="The absolute path to the generated CV file.")


# Research Models
class ResearchStatus(str, Enum):
    """Status of research analysis."""

    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    PENDING = "pending"


class CompanyInsight(BaseModel):
    """Company-specific research insight."""

    company_name: str
    industry: Optional[str] = None
    size: Optional[str] = None
    culture: Optional[str] = None
    recent_news: List[str] = Field(default_factory=list)
    key_values: List[str] = Field(default_factory=list)
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)


class IndustryInsight(BaseModel):
    """Industry-specific research insight."""

    industry_name: str
    trends: List[str] = Field(default_factory=list)
    key_skills: List[str] = Field(default_factory=list)
    growth_areas: List[str] = Field(default_factory=list)
    challenges: List[str] = Field(default_factory=list)
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)


class RoleInsight(BaseModel):
    """Role-specific research insight."""

    role_title: str
    required_skills: List[str] = Field(default_factory=list)
    preferred_qualifications: List[str] = Field(default_factory=list)
    responsibilities: List[str] = Field(default_factory=list)
    career_progression: List[str] = Field(default_factory=list)
    salary_range: Optional[str] = None
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)


class ResearchMetadataModel(BaseModel):
    """Model for research metadata."""

    source: Optional[str] = None
    analyst: Optional[str] = None
    notes: Optional[str] = None
    extra: Optional[dict] = Field(default_factory=dict)


class ResearchFindings(BaseModel):
    """Model for the findings of the research agent."""

    status: ResearchStatus = ResearchStatus.PENDING
    research_timestamp: datetime = Field(default_factory=datetime.now)

    # Core insights
    company_insights: Optional[CompanyInsight] = None
    industry_insights: Optional[IndustryInsight] = None
    role_insights: Optional[RoleInsight] = None

    # Analysis results
    key_terms: List[str] = Field(default_factory=list)
    skill_gaps: List[str] = Field(default_factory=list)
    enhancement_suggestions: List[str] = Field(default_factory=list)

    # Metadata
    research_sources: List[str] = Field(default_factory=list)
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    processing_time_seconds: Optional[float] = None
    error_message: Optional[str] = None

    # Legacy compatibility
    metadata: ResearchMetadataModel = Field(default_factory=ResearchMetadataModel)

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {datetime: lambda v: v.isoformat()}

    def to_dict(self) -> dict:
        """Convert to dictionary for backward compatibility."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict) -> "ResearchFindings":
        """Create from dictionary for backward compatibility."""
        return cls(**data)

    @classmethod
    def create_empty(cls) -> "ResearchFindings":
        """Create empty research findings."""
        return cls(status=ResearchStatus.PENDING)

    @classmethod
    def create_failed(cls, error_message: str) -> "ResearchFindings":
        """Create failed research findings."""
        return cls(status=ResearchStatus.FAILED, error_message=error_message)


class ResearchAgentOutput(BaseModel):
    """Output model for ResearchAgent run_async method."""

    research_findings: Optional[ResearchFindings] = Field(
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

    @field_validator("research_findings", mode="before")
    @classmethod
    def validate_research_findings(cls, v):
        """Convert dict to ResearchFindings if needed."""
        if v is None:
            return v
        if isinstance(v, dict):
            return ResearchFindings.from_dict(v)
        if isinstance(v, ResearchFindings):
            return v
        raise ValueError(
            f"research_findings must be a dict or ResearchFindings instance, got {type(v)}"
        )
