from pydantic import BaseModel
from typing import List, Dict, Optional, Any


class ContentWriterJobData(BaseModel):
    title: Optional[str] = None
    raw_text: Optional[str] = None
    description: Optional[str] = None
    company: Optional[str] = None
    skills: Optional[List[str]] = []
    responsibilities: Optional[List[str]] = []
    industry_terms: Optional[List[str]] = []
    company_values: Optional[List[str]] = []


class ContentWriterContentItem(BaseModel):
    id: str
    content: Optional[str] = None
    raw_llm_output: Optional[str] = None
    # Add more fields as needed


class ContentWriterGenerationContext(BaseModel):
    my_talents: Optional[List[str]] = []
    # Add more fields as needed


class ContentWriterResult(BaseModel):
    structured_cv: Any
    error_messages: Optional[List[str]] = []
