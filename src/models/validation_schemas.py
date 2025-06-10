"""Pydantic validation schemas for AI CV Generator.

This module provides comprehensive data validation schemas using Pydantic
to ensure data integrity at component boundaries and catch type mismatches
early in the pipeline.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from uuid import UUID

from pydantic import BaseModel, Field, field_validator, model_validator, HttpUrl
from pydantic_core import ValidationError as PydanticValidationError


class ProcessingStatusSchema(str, Enum):
    """Validation schema for processing status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RATE_LIMITED = "rate_limited"


class ContentTypeSchema(str, Enum):
    """Validation schema for content types."""
    QUALIFICATION = "qualification"
    EXPERIENCE = "experience"
    EXPERIENCE_ITEM = "experience_item"
    PROJECT = "project"
    PROJECT_ITEM = "project_item"
    PROJECTS = "projects"
    EXECUTIVE_SUMMARY = "executive_summary"
    SKILL = "skill"
    ACHIEVEMENT = "achievement"
    EDUCATION = "education"
    SKILLS = "skills"
    ANALYSIS = "analysis"
    QUALITY_CHECK = "quality_check"
    OPTIMIZATION = "optimization"


class ProcessingMetadataSchema(BaseModel):
    """Validation schema for processing metadata."""
    item_id: str = Field(..., min_length=1, description="Unique identifier for the item")
    status: ProcessingStatusSchema = ProcessingStatusSchema.PENDING
    created_at: datetime
    updated_at: datetime
    processing_attempts: int = Field(ge=0, description="Number of processing attempts")
    last_error: Optional[str] = Field(None, max_length=1000)
    processing_time_seconds: float = Field(ge=0.0, description="Processing time in seconds")
    llm_calls_made: int = Field(ge=0, description="Number of LLM API calls made")
    tokens_used: int = Field(ge=0, description="Total tokens used")
    rate_limit_hits: int = Field(ge=0, description="Number of rate limit hits")

    model_config = {"use_enum_values": True}


class ContentItemSchema(BaseModel):
    """Validation schema for content items."""
    content_type: ContentTypeSchema
    original_content: str = Field(..., min_length=1, description="Original content text")
    generated_content: Optional[str] = Field(None, description="Generated content text")
    metadata: ProcessingMetadataSchema
    context: Dict[str, Any] = Field(default_factory=dict)
    priority: int = Field(1, ge=1, le=10, description="Processing priority (1-10)")
    dependencies: List[str] = Field(default_factory=list, description="Item IDs this depends on")

    @field_validator('generated_content')
    @classmethod
    def validate_generated_content(cls, v, info):
        """Validate generated content based on processing status."""
        if info.data and 'metadata' in info.data:
            status = info.data['metadata'].status
            if status == ProcessingStatusSchema.COMPLETED and not v:
                raise ValueError("Generated content is required when status is COMPLETED")
        return v

    model_config = {"use_enum_values": True}


class PersonalInfoSchema(BaseModel):
    """Validation schema for personal information."""
    name: str = Field(..., min_length=2, max_length=100, description="Full name")
    email: str = Field(..., pattern=r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', description="Email address")
    phone: Optional[str] = Field(None, pattern=r'^[\+]?[1-9][\d\s\-\(\)]{7,15}$', description="Phone number")
    address: Optional[str] = Field(None, max_length=200, description="Physical address")
    linkedin: Optional[HttpUrl] = Field(None, description="LinkedIn profile URL")
    website: Optional[HttpUrl] = Field(None, description="Personal website URL")
    summary: Optional[str] = Field(None, max_length=500, description="Professional summary")


class ExperienceSchema(BaseModel):
    """Validation schema for work experience."""
    title: str = Field(..., min_length=2, max_length=100, description="Job title")
    company: str = Field(..., min_length=2, max_length=100, description="Company name")
    location: Optional[str] = Field(None, max_length=100, description="Work location")
    start_date: str = Field(..., description="Start date (YYYY-MM or YYYY-MM-DD format)")
    end_date: Optional[str] = Field(None, description="End date (YYYY-MM or YYYY-MM-DD format)")
    current: bool = Field(False, description="Currently working in this position")
    description: Optional[str] = Field(None, max_length=2000, description="Job description")
    achievements: Optional[List[str]] = Field(None, description="Key achievements")
    technologies: Optional[List[str]] = Field(None, description="Technologies used")

    @field_validator('achievements')
    @classmethod
    def validate_achievements(cls, v):
        """Validate achievements list."""
        if v is not None:
            if len(v) > 10:
                raise ValueError("Maximum 10 achievements allowed")
            for achievement in v:
                if len(achievement) > 200:
                    raise ValueError("Each achievement must be 200 characters or less")
        return v

    @field_validator('technologies')
    @classmethod
    def validate_technologies(cls, v):
        """Validate technologies list."""
        if v is not None:
            if len(v) > 20:
                raise ValueError("Maximum 20 technologies allowed")
            for tech in v:
                if len(tech) > 50:
                    raise ValueError("Each technology must be 50 characters or less")
        return v

    @model_validator(mode='after')
    def validate_dates(self):
        """Validate date consistency."""
        current = self.current
        end_date = self.end_date
        
        if current and end_date:
            raise ValueError("Cannot have end_date when current is True")
        if not current and not end_date:
            raise ValueError("end_date is required when current is False")
        
        return self


class ProjectSchema(BaseModel):
    """Validation schema for projects."""
    name: str = Field(..., min_length=2, max_length=100, description="Project name")
    description: str = Field(..., min_length=10, max_length=1000, description="Project description")
    technologies: Optional[List[str]] = Field(None, description="Technologies used")
    url: Optional[HttpUrl] = Field(None, description="Project URL")
    github_url: Optional[HttpUrl] = Field(None, description="GitHub repository URL")
    start_date: Optional[str] = Field(None, description="Project start date")
    end_date: Optional[str] = Field(None, description="Project end date")
    status: Optional[str] = Field(None, description="Project status")

    @field_validator('technologies')
    @classmethod
    def validate_technologies(cls, v):
        """Validate technologies list."""
        if v is not None:
            if len(v) > 15:
                raise ValueError("Maximum 15 technologies allowed")
            for tech in v:
                if len(tech) > 50:
                    raise ValueError("Each technology must be 50 characters or less")
        return v


class JobDescriptionSchema(BaseModel):
    """Validation schema for job descriptions."""
    raw_text: str = Field(..., min_length=50, description="Raw job description text")
    title: Optional[str] = Field(None, max_length=100, description="Job title")
    company: Optional[str] = Field(None, max_length=100, description="Company name")
    location: Optional[str] = Field(None, max_length=100, description="Job location")
    skills: Optional[List[str]] = Field(None, description="Required skills")
    experience_level: Optional[str] = Field(None, description="Required experience level")
    responsibilities: Optional[List[str]] = Field(None, description="Job responsibilities")
    industry_terms: Optional[List[str]] = Field(None, description="Industry-specific terms")
    company_values: Optional[List[str]] = Field(None, description="Company values")

    @field_validator('skills')
    @classmethod
    def validate_skills(cls, v):
        """Validate skills list."""
        if v is not None:
            if len(v) > 30:
                raise ValueError("Maximum 30 skills allowed")
            for skill in v:
                if len(skill) > 100:
                    raise ValueError("Each skill must be 100 characters or less")
        return v


class AgentInputSchema(BaseModel):
    """Base validation schema for agent inputs."""
    session_id: Optional[str] = Field(None, description="Session identifier")
    user_id: Optional[str] = Field(None, description="User identifier")
    timestamp: datetime = Field(default_factory=datetime.now)
    context: Dict[str, Any] = Field(default_factory=dict)


class ParserAgentInputSchema(AgentInputSchema):
    """Validation schema for Parser Agent inputs."""
    job_description: Union[str, JobDescriptionSchema] = Field(..., description="Job description data")
    cv_text: str = Field(..., min_length=100, description="CV text to parse")
    start_from_scratch: bool = Field(False, description="Whether to start from scratch")

    @field_validator('job_description')
    @classmethod
    def validate_job_description(cls, v):
        """Validate job description input."""
        if isinstance(v, str):
            if len(v) < 50:
                raise ValueError("Job description text must be at least 50 characters")
        return v


class CVAnalyzerInputSchema(AgentInputSchema):
    """Validation schema for CV Analyzer Agent inputs."""
    cv_data: Union[str, Dict[str, Any]] = Field(..., description="CV data to analyze")
    analysis_type: Optional[str] = Field("comprehensive", description="Type of analysis to perform")


class ContentWriterInputSchema(AgentInputSchema):
    """Validation schema for Content Writer Agent inputs."""
    structured_cv_data: Dict[str, Any] = Field(..., description="Structured CV data")
    items_to_regenerate: Optional[List[str]] = Field(None, description="Item IDs to regenerate")
    job_description_data: Union[str, Dict[str, Any]] = Field(..., description="Job description data")
    research_results: Optional[Dict[str, Any]] = Field(None, description="Research results")


class AgentResultSchema(BaseModel):
    """Validation schema for agent results."""
    success: bool = Field(..., description="Whether the operation was successful")
    data: Optional[Dict[str, Any]] = Field(None, description="Result data")
    error: Optional[str] = Field(None, description="Error message if failed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    processing_time: Optional[float] = Field(None, ge=0.0, description="Processing time in seconds")
    tokens_used: Optional[int] = Field(None, ge=0, description="Tokens used")

    @model_validator(mode='after')
    def validate_result_consistency(self):
        """Validate result consistency."""
        success = self.success
        error = self.error
        data = self.data
        
        if not success and not error:
            raise ValueError("Error message is required when success is False")
        if success and error:
            raise ValueError("Error message should not be present when success is True")
        if success and not data:
            raise ValueError("Data is required when success is True")
        
        return self


class AgentExecutionContextSchema(BaseModel):
    """Validation schema for agent execution context."""
    agent_type: str = Field(..., min_length=1, description="Type of agent")
    input_data: Dict[str, Any] = Field(..., description="Input data for the agent")
    context: Dict[str, Any] = Field(default_factory=dict, description="Execution context")
    session_id: Optional[str] = Field(None, description="Session identifier")
    user_id: Optional[str] = Field(None, description="User identifier")
    timeout_seconds: Optional[int] = Field(None, ge=1, le=300, description="Execution timeout")


class ValidationError(Exception):
    """Custom exception for validation errors."""
    
    def __init__(self, message: str, field: Optional[str] = None, value: Any = None):
        self.message = message
        self.field = field
        self.value = value
        super().__init__(self.message)


def validate_data(data: Any, schema_class: BaseModel) -> BaseModel:
    """Validate data against a Pydantic schema.
    
    Args:
        data: Data to validate
        schema_class: Pydantic schema class to validate against
        
    Returns:
        Validated data as schema instance
        
    Raises:
        ValidationError: If validation fails
    """
    try:
        if isinstance(data, dict):
            return schema_class(**data)
        elif isinstance(data, schema_class):
            return data
        else:
            return schema_class.model_validate(data)
    except Exception as e:
        raise ValidationError(f"Validation failed: {str(e)}") from e


def validate_agent_input(agent_type: str, input_data: Any) -> BaseModel:
    """Validate agent input data based on agent type.
    
    Args:
        agent_type: Type of agent
        input_data: Input data to validate
        
    Returns:
        Validated input data
        
    Raises:
        ValidationError: If validation fails
    """
    schema_mapping = {
        'parser': ParserAgentInputSchema,
        'cv_analyzer': CVAnalyzerInputSchema,
        'content_writer': ContentWriterInputSchema,
        'enhanced_content_writer': ContentWriterInputSchema,
    }
    
    schema_class = schema_mapping.get(agent_type.lower())
    if not schema_class:
        # Use base schema for unknown agent types
        schema_class = AgentInputSchema
    
    return validate_data(input_data, schema_class)