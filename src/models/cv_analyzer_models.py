from typing import Optional
from pydantic import BaseModel, Field


class BasicCVInfo(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    linkedin: Optional[str] = None
    summary: Optional[str] = None
    # Add more fields as needed


class CVAnalyzerNodeResult(BaseModel):
    cv_analysis_results: Optional[dict] = Field(default_factory=dict)
    cv_analyzer_success: bool = False
    cv_analyzer_confidence: float = 0.0
    cv_analyzer_error: Optional[str] = None
