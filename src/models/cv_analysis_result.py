from typing import List, Optional
from pydantic import BaseModel, Field


class CVAnalysisResult(BaseModel):
    skill_matches: List[str] = Field(default_factory=list)
    experience_relevance: float = 0.0
    gaps_identified: List[str] = Field(default_factory=list)
    strengths: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    match_score: float = 0.0
    analysis_timestamp: Optional[str] = None
