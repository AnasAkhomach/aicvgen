from typing import Optional
from pydantic import BaseModel, Field
from .research_models import ResearchFindings


class ResearchAgentNodeResult(BaseModel):
    research_findings: Optional[ResearchFindings] = None
    error_messages: Optional[list] = Field(default_factory=list)
