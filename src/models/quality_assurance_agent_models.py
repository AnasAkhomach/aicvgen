from pydantic import BaseModel
from typing import List, Dict, Optional


class KeyTerms(BaseModel):
    skills: List[str] = []
    responsibilities: List[str] = []
    industry_terms: List[str] = []
    company_values: List[str] = []


class SectionQualityResultModel(BaseModel):
    section_name: str
    passed: bool
    issues: List[str]
    item_checks: Optional[List[dict]] = (
        None  # To be replaced with ItemQualityResultModel if defined
    )


class ItemQualityResultModel(BaseModel):
    item_id: str
    passed: bool
    issues: List[str]
    suggestions: Optional[List[str]] = None


class OverallQualityCheckResultModel(BaseModel):
    check_name: str
    passed: bool
    details: Optional[str] = None


# DEPRECATED: Use QualityAssuranceAgentOutput from agent_output_models.py instead.
# class QualityAssuranceResult(BaseModel):
#     section_results: List[SectionQualityResultModel]
#     overall_checks: List[OverallQualityCheckResultModel]
