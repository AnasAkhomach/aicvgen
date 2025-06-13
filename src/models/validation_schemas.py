"""Validation schemas for the AI CV Generator.

This module provides Pydantic validation schemas for API requests and responses.
Currently serves as a placeholder for future API development.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


# Placeholder API validation schemas
# These will be expanded when building the REST API

class CVGenerationRequestSchema(BaseModel):
    """Schema for CV generation API requests."""
    cv_text: str = Field(..., description="Raw CV text")
    job_description: str = Field(..., description="Job description text")
    options: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Generation options")


class CVGenerationResponseSchema(BaseModel):
    """Schema for CV generation API responses."""
    session_id: str = Field(..., description="Session identifier")
    status: str = Field(..., description="Generation status")
    structured_cv: Optional[Dict[str, Any]] = Field(None, description="Generated structured CV")
    errors: Optional[List[str]] = Field(None, description="Any errors that occurred")