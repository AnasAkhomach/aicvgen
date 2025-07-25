"""Pydantic output models for agent results.

This module defines the output data models that agents can use to structure
their dictionary return values. These models ensure type safety and data
validation for agent outputs according to Task C-03.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from src.models.cv_models import JobDescriptionData, StructuredCV
from src.models.llm_data_models import BasicCVInfo


class ExecutiveSummaryLLMOutput(BaseModel):
    """Pydantic model for structured LLM output of executive summary."""
    
    executive_summary: str = Field(
        description="Generated executive summary content for the CV",
        min_length=100,
        max_length=2000
    )
    
    @field_validator('executive_summary')
    @classmethod
    def validate_executive_summary(cls, v):
        """Validate that executive summary content is meaningful."""
        if not v or not v.strip():
            raise ValueError("Executive summary content cannot be empty")
        if len(v.strip()) < 100:
            raise ValueError("Executive summary content must be at least 100 characters long")
        if len(v.strip()) > 2000:
            raise ValueError("Executive summary content must not exceed 2000 characters")
        return v.strip()


class KeyQualificationsLLMOutput(BaseModel):
    """Pydantic model for structured LLM output of key qualifications."""
    
    qualifications: List[str] = Field(
        description="List of key qualifications tailored to the job description",
        min_items=3,
        max_items=8
    )
    
    @field_validator('qualifications')
    @classmethod
    def validate_qualifications(cls, v):
        """Validate that qualifications are not empty and properly formatted."""
        if not v:
            raise ValueError("At least one qualification must be provided")
        
        cleaned_qualifications = []
        for qual in v:
            # Clean up common prefixes and ensure proper formatting
            cleaned = qual.strip().lstrip("- •*").strip()
            if cleaned and len(cleaned) > 5:  # Minimum meaningful length
                cleaned_qualifications.append(cleaned)
        
        if not cleaned_qualifications:
            raise ValueError("No valid qualifications found after cleaning")
            
        return cleaned_qualifications


class ProfessionalExperienceLLMOutput(BaseModel):
    """Model for LLM output when generating professional experience content."""

    professional_experience: str = Field(
        description="Generated professional experience content for the CV",
        min_length=50,
    )

    @field_validator("professional_experience")
    @classmethod
    def validate_experience_content(cls, v):
        """Validate that professional experience content is meaningful."""
        if not v or not v.strip():
            raise ValueError("Professional experience content cannot be empty")
        if len(v.strip()) < 50:
            raise ValueError("Professional experience content must be at least 50 characters long")
        return v.strip()


class ProjectLLMOutput(BaseModel):
    """Pydantic model for structured LLM output of project content."""
    
    project_description: Optional[str] = Field(
        default=None,
        description="A brief description of the project"
    )
    technologies_used: List[str] = Field(
        default_factory=list,
        description="List of technologies and tools used in the project"
    )
    achievements: List[str] = Field(
        default_factory=list,
        description="List of key achievements and outcomes from the project"
    )
    bullet_points: List[str] = Field(
        description="A list of 3-5 generated project bullet points",
        min_items=1,
        max_items=8
    )
    
    @field_validator('bullet_points')
    @classmethod
    def validate_bullet_points(cls, v):
        """Validate that bullet points are not empty and properly formatted."""
        if not v:
            raise ValueError("At least one bullet point must be provided")
        
        cleaned_bullets = []
        for bullet in v:
            # Clean up common prefixes and ensure proper formatting
            cleaned = bullet.strip().lstrip("- •*").strip()
            if cleaned and len(cleaned) > 10:  # Minimum meaningful length
                cleaned_bullets.append(cleaned)
        
        if not cleaned_bullets:
            raise ValueError("No valid bullet points found after cleaning")
            
        return cleaned_bullets


class KeyQualificationsAgentOutput(BaseModel):
    """Output model for KeyQualificationsAgent run_async method."""

    updated_structured_cv: StructuredCV = Field(
        description="The full CV data structure with the key qualifications section updated."
    )
    generated_qualifications: List[str] = Field(
        default_factory=list,
        description="The newly generated key qualifications."
    )
    qualification_count: int = Field(
        default=0,
        description="Number of qualifications generated."
    )

    class Config:
        """Pydantic configuration for proper JSON serialization."""

        arbitrary_types_allowed = True
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KeyQualificationsAgentOutput":
        """Create from dictionary for deserialization."""
        return cls(**data)


class ProfessionalExperienceWriterOutput(BaseModel):
    """Output model for ProfessionalExperienceWriterAgent run_async method."""

    professional_experience: str = Field(
        description="Generated professional experience content"
    )
    structured_cv: Optional[StructuredCV] = Field(
        default=None, description="Updated structured CV with professional experience"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Additional metadata about the generation"
    )

    class Config:
        """Pydantic configuration for proper JSON serialization."""

        arbitrary_types_allowed = True
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProfessionalExperienceWriterOutput":
        """Create from dictionary for deserialization."""
        return cls(**data)


class ParserAgentOutput(BaseModel):
    """Output model for ParserAgent run_async method."""

    job_description_data: Optional[JobDescriptionData] = Field(
        default=None, description="Parsed job description data"
    )
    structured_cv: Optional[StructuredCV] = Field(
        default=None, description="Parsed CV data structure"
    )

    class Config:
        """Pydantic configuration for proper JSON serialization."""

        arbitrary_types_allowed = True
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ParserAgentOutput":
        """Create from dictionary for deserialization."""
        return cls(**data)


class EnhancedContentWriterOutput(BaseModel):
    """Output model for EnhancedContentWriterAgent run_async method."""

    updated_structured_cv: StructuredCV = Field(
        description="The full CV data structure with the enhanced content updated."
    )
    item_id: str = Field(description="The ID of the item that was enhanced.")
    generated_content: str = Field(description="The newly generated content.")

    class Config:
        """Pydantic configuration for proper JSON serialization."""

        arbitrary_types_allowed = True
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EnhancedContentWriterOutput":
        """Create from dictionary for deserialization."""
        return cls(**data)


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

    class Config:
        """Pydantic configuration for proper JSON serialization."""

        arbitrary_types_allowed = True
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CleaningAgentOutput":
        """Create from dictionary for deserialization."""
        return cls(**data)


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

    class Config:
        """Pydantic configuration for proper JSON serialization."""

        arbitrary_types_allowed = True
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CVAnalysisResult":
        """Create from dictionary for deserialization."""
        return cls(**data)


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

    class Config:
        """Pydantic configuration for proper JSON serialization."""

        arbitrary_types_allowed = True
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CVAnalyzerAgentOutput":
        """Create from dictionary for deserialization."""
        return cls(**data)


class ItemQualityResultModel(BaseModel):
    """Model for individual item quality check results."""

    item_id: str
    passed: bool
    issues: List[str]
    suggestions: Optional[List[str]] = None

    class Config:
        """Pydantic configuration for proper JSON serialization."""

        arbitrary_types_allowed = True
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ItemQualityResultModel":
        """Create from dictionary for deserialization."""
        return cls(**data)


class SectionQualityResultModel(BaseModel):
    """Model for section-level quality check results."""

    section_name: str
    passed: bool
    issues: List[str]
    item_checks: Optional[List[ItemQualityResultModel]] = Field(default_factory=list)

    class Config:
        """Pydantic configuration for proper JSON serialization."""

        arbitrary_types_allowed = True
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SectionQualityResultModel":
        """Create from dictionary for deserialization."""
        return cls(**data)


class OverallQualityCheckResultModel(BaseModel):
    """Model for overall CV quality check results."""

    check_name: str
    passed: bool
    details: Optional[str] = None

    class Config:
        """Pydantic configuration for proper JSON serialization."""

        arbitrary_types_allowed = True
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OverallQualityCheckResultModel":
        """Create from dictionary for deserialization."""
        return cls(**data)


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

    class Config:
        """Pydantic configuration for proper JSON serialization."""

        arbitrary_types_allowed = True
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QualityAssuranceAgentOutput":
        """Create from dictionary for deserialization."""
        return cls(**data)


class FormatterAgentOutput(BaseModel):
    """Output model for FormatterAgent run method."""

    output_path: str = Field(description="The absolute path to the generated CV file.")

    class Config:
        """Pydantic configuration for proper JSON serialization."""

        arbitrary_types_allowed = True
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FormatterAgentOutput":
        """Create from dictionary for deserialization."""
        return cls(**data)


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

    class Config:
        """Pydantic configuration for proper JSON serialization."""

        arbitrary_types_allowed = True
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CompanyInsight":
        """Create from dictionary for deserialization."""
        return cls(**data)


class IndustryInsight(BaseModel):
    """Industry-specific research insight."""

    industry_name: str
    trends: List[str] = Field(default_factory=list)
    key_skills: List[str] = Field(default_factory=list)
    growth_areas: List[str] = Field(default_factory=list)
    challenges: List[str] = Field(default_factory=list)
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)

    class Config:
        """Pydantic configuration for proper JSON serialization."""

        arbitrary_types_allowed = True
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IndustryInsight":
        """Create from dictionary for deserialization."""
        return cls(**data)


class RoleInsight(BaseModel):
    """Role-specific research insight."""

    role_title: str
    required_skills: List[str] = Field(default_factory=list)
    preferred_qualifications: List[str] = Field(default_factory=list)
    responsibilities: List[str] = Field(default_factory=list)
    career_progression: List[str] = Field(default_factory=list)
    salary_range: Optional[str] = None
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)

    class Config:
        """Pydantic configuration for proper JSON serialization."""

        arbitrary_types_allowed = True
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RoleInsight":
        """Create from dictionary for deserialization."""
        return cls(**data)


class ResearchMetadataModel(BaseModel):
    """Model for research metadata."""

    source: Optional[str] = None
    analyst: Optional[str] = None
    notes: Optional[str] = None
    extra: Optional[Dict[str, Any]] = Field(default_factory=dict)

    class Config:
        """Pydantic configuration for proper JSON serialization."""

        arbitrary_types_allowed = True
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResearchMetadataModel":
        """Create from dictionary for deserialization."""
        return cls(**data)


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

    @field_validator("metadata", mode="before")
    @classmethod
    def validate_metadata(cls, v):
        """Convert dict to ResearchMetadataModel if needed."""
        if v is None:
            return ResearchMetadataModel()
        if isinstance(v, dict):
            return ResearchMetadataModel.from_dict(v)
        if isinstance(v, ResearchMetadataModel):
            return v
        raise ValueError(
            f"metadata must be a dict or ResearchMetadataModel instance, got {type(v)}"
        )

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

    class Config:
        """Pydantic configuration for proper JSON serialization."""

        arbitrary_types_allowed = True
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResearchAgentOutput":
        """Create from dictionary for deserialization."""
        return cls(**data)

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


class ProfessionalExperienceUpdaterAgentOutput(BaseModel):
    """Output model for ProfessionalExperienceUpdaterAgent."""

    updated_structured_cv: StructuredCV = Field(
        description="The structured CV with updated professional experience section"
    )

    class Config:
        """Pydantic configuration for proper JSON serialization."""
        arbitrary_types_allowed = True
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProfessionalExperienceUpdaterAgentOutput":
        """Create from dictionary for deserialization."""
        return cls(**data)


class ProjectsUpdaterAgentOutput(BaseModel):
    """Output model for ProjectsUpdaterAgent."""

    updated_structured_cv: StructuredCV = Field(
        description="The structured CV with updated projects section"
    )

    class Config:
        """Pydantic configuration for proper JSON serialization."""
        arbitrary_types_allowed = True
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProjectsUpdaterAgentOutput":
        """Create from dictionary for deserialization."""
        return cls(**data)


class ExecutiveSummaryUpdaterAgentOutput(BaseModel):
    """Output model for ExecutiveSummaryUpdaterAgent."""

    updated_structured_cv: StructuredCV = Field(
        description="The structured CV with updated executive summary section"
    )

    class Config:
        """Pydantic configuration for proper JSON serialization."""
        arbitrary_types_allowed = True
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExecutiveSummaryUpdaterAgentOutput":
        """Create from dictionary for deserialization."""
        return cls(**data)
