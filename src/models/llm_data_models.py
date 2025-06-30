from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from pydantic import BaseModel, Field

from .cv_models import MetadataModel


class VectorStoreConfig(BaseModel):
    """Configuration for vector store database."""

    collection_name: str = "cv_content"
    persist_directory: str = "data/vector_store"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    dimension: int = 768
    index_type: str = "IndexFlatL2"


# Temporary model for CV parsing LLM output
class CVParsingPersonalInfo(BaseModel):
    """Personal information extracted from CV parsing."""

    name: str
    email: str
    phone: str
    linkedin: Optional[str] = None
    github: Optional[str] = None
    location: Optional[str] = None


class CVParsingSubsection(BaseModel):
    """Subsection structure for CV parsing output."""

    name: str
    items: List[str] = Field(default_factory=list)


class CVParsingSection(BaseModel):
    """Section structure for CV parsing output."""

    name: str
    items: List[str] = Field(default_factory=list)
    subsections: List[CVParsingSubsection] = Field(default_factory=list)


class CVParsingResult(BaseModel):
    """Complete CV parsing result from LLM."""

    personal_info: CVParsingPersonalInfo
    sections: List[CVParsingSection]


# API Model backward compatibility aliases
class PersonalInfo(BaseModel):
    """Personal information model."""

    name: str
    email: str
    phone: Optional[str] = None
    address: Optional[str] = None
    linkedin: Optional[str] = None
    website: Optional[str] = None
    summary: Optional[str] = None


class Experience(BaseModel):
    """Work experience model."""

    title: str
    company: str
    location: Optional[str] = None
    start_date: str
    end_date: Optional[str] = None
    current: bool = False
    description: Optional[str] = None
    achievements: Optional[List[str]] = None
    technologies: Optional[List[str]] = None


class Education(BaseModel):
    """Education model."""

    degree: str
    institution: str
    location: Optional[str] = None
    graduation_date: Optional[str] = None
    gpa: Optional[str] = None
    honors: Optional[List[str]] = None
    relevant_coursework: Optional[List[str]] = None


class Project(BaseModel):
    """Project model."""

    name: str
    description: str
    technologies: Optional[List[str]] = None
    url: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    achievements: Optional[List[str]] = None


class Skill(BaseModel):
    """Skill model."""

    name: str
    level: Optional[str] = None
    category: Optional[str] = None
    years_experience: Optional[int] = None


class Certification(BaseModel):
    """Certification model."""

    name: str
    issuer: str
    date_obtained: Optional[str] = None
    expiry_date: Optional[str] = None
    credential_id: Optional[str] = None
    url: Optional[str] = None


class Language(BaseModel):
    """Language model."""

    name: str
    proficiency: Optional[str] = None
    native: bool = False


class BasicCVInfo(BaseModel):
    """Basic CV information extracted as a fallback."""

    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    linkedin: Optional[str] = None
    summary: Optional[str] = None


@dataclass
class RateLimitState:
    """State tracking for rate limiting per model."""

    model: str
    requests_made: int = 0
    requests_limit: int = 60
    tokens_made: int = 0
    tokens_limit: int = 1000000
    window_start: datetime = None
    window_duration: timedelta = None
    last_request_time: Optional[datetime] = None
    backoff_until: Optional[datetime] = None
    consecutive_failures: int = 0

    def __post_init__(self):
        """Initialize default values after dataclass creation."""
        if self.window_start is None:
            self.window_start = datetime.now()
        if self.window_duration is None:
            self.window_duration = timedelta(minutes=1)

    def is_rate_limited(self) -> bool:
        """Check if currently rate limited."""
        now = datetime.now()

        # Check if in backoff period
        if self.backoff_until and now < self.backoff_until:
            return True

        # Check if window has reset
        if now - self.window_start > self.window_duration:
            self.requests_made = 0
            self.window_start = now

        return self.requests_made >= self.requests_limit

    def record_request(self, tokens_used: int = 0):
        """Record a new request."""
        now = datetime.now()

        # Reset window if needed
        if now - self.window_start > self.window_duration:
            self.requests_made = 0
            self.tokens_made = 0
            self.window_start = now

        self.requests_made += 1
        self.tokens_made += tokens_used
        self.last_request_time = now

    def record_failure(self):
        """Record a request failure."""
        self.consecutive_failures += 1
        # Exponential backoff
        backoff_seconds = min(300, 2**self.consecutive_failures)  # Max 5 minutes
        self.backoff_until = datetime.now() + timedelta(seconds=backoff_seconds)

    def record_success(self):
        """Record a successful request."""
        self.consecutive_failures = 0
        self.backoff_until = None

    def can_make_request(self, estimated_tokens: int = 0) -> bool:
        """Check if a request can be made given current rate limits."""
        if self.is_rate_limited():
            return False
        return True


# Structured Logging Data Models
@dataclass
class AgentExecutionLog:
    """Structured log entry for agent execution tracking."""

    timestamp: str
    agent_name: str
    session_id: str
    item_id: Optional[str]
    content_type: Optional[str]
    execution_phase: str  # 'start', 'success', 'error', 'retry'
    processing_time_seconds: Optional[float] = None
    confidence_score: Optional[float] = None
    retry_count: int = 0
    error_message: Optional[str] = None
    input_data_type: Optional[str] = None
    output_data_size: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class AgentDecisionLog:
    """Structured log entry for agent decision tracking."""

    timestamp: str
    agent_name: str
    session_id: str
    item_id: Optional[str]
    decision_type: str  # 'validation', 'processing', 'fallback', 'enhancement'
    decision_details: str
    confidence_score: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class AgentPerformanceLog:
    """Structured log entry for agent performance metrics."""

    timestamp: str
    agent_name: str
    session_id: str
    metric_type: str  # 'execution_time', 'success_rate', 'error_rate', 'throughput'
    metric_value: float
    time_window: str  # 'session', 'hour', 'day'
    metadata: Optional[Dict[str, Any]] = None


class LLMResponse(BaseModel):
    """Structured response from LLM calls."""

    content: str
    tokens_used: int = 0
    processing_time: float = 0.0
    model_used: str = ""
    success: bool = True
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ErrorFallbackModel(BaseModel):
    error: str


@dataclass
class RateLimitLog:
    """Structured log for rate limiting events."""

    timestamp: str
    model: str
    requests_in_window: int
    tokens_in_window: int
    window_start: str
    window_end: str
    limit_exceeded: bool
    wait_time_seconds: Optional[float] = None


class KeyTerms(BaseModel):
    """Model for key terms extracted from a job description."""

    skills: List[str] = Field(default_factory=list)
    responsibilities: List[str] = Field(default_factory=list)
    industry_terms: List[str] = Field(default_factory=list)
    company_values: List[str] = Field(default_factory=list)
