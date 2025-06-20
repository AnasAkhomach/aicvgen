"""Research-specific Pydantic models for type safety."""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


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


class ResearchFindings(BaseModel):
    """Comprehensive research findings from ResearchAgent."""
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
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        arbitrary_types_allowed = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for backward compatibility."""
        return self.model_dump()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResearchFindings":
        """Create from dictionary for backward compatibility."""
        return cls(**data)
    
    @classmethod
    def create_empty(cls) -> "ResearchFindings":
        """Create empty research findings."""
        return cls(status=ResearchStatus.PENDING)
    
    @classmethod
    def create_failed(cls, error_message: str) -> "ResearchFindings":
        """Create failed research findings."""
        return cls(
            status=ResearchStatus.FAILED,
            error_message=error_message
        )