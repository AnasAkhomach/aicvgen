"""Quality assurance-specific Pydantic models for type safety."""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


class QualityStatus(str, Enum):
    """Status of quality check."""
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    PENDING = "pending"
    SKIPPED = "skipped"


class QualityCheckType(str, Enum):
    """Type of quality check performed."""
    CONTENT_LENGTH = "content_length"
    KEYWORD_RELEVANCE = "keyword_relevance"
    GRAMMAR_CHECK = "grammar_check"
    FORMATTING = "formatting"
    COMPLETENESS = "completeness"
    CONSISTENCY = "consistency"
    RELEVANCE = "relevance"
    PROFESSIONAL_TONE = "professional_tone"


class QualityCheck(BaseModel):
    """Individual quality check result."""
    check_type: QualityCheckType
    status: QualityStatus
    message: str
    score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    suggestions: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ItemQualityResult(BaseModel):
    """Quality check result for a single item."""
    item_id: str
    section: str
    subsection: Optional[str] = None
    content_preview: str = Field(default="", description="First 50 chars of content")
    overall_status: QualityStatus = QualityStatus.PENDING
    checks: List[QualityCheck] = Field(default_factory=list)
    overall_score: float = Field(default=0.0, ge=0.0, le=1.0)
    processing_time_seconds: Optional[float] = None
    timestamp: datetime = Field(default_factory=datetime.now)

    def add_check(self, check: QualityCheck) -> None:
        """Add a quality check result."""
        self.checks.append(check)
        self._update_overall_status()
    
    def _update_overall_status(self) -> None:
        """Update overall status based on individual checks."""
        if not self.checks:
            self.overall_status = QualityStatus.PENDING
            return
        
        statuses = [check.status for check in self.checks]
        if QualityStatus.FAIL in statuses:
            self.overall_status = QualityStatus.FAIL
        elif QualityStatus.WARNING in statuses:
            self.overall_status = QualityStatus.WARNING
        else:
            self.overall_status = QualityStatus.PASS
        
        # Calculate overall score
        scores = [check.score for check in self.checks if check.score is not None]
        if scores:
            self.overall_score = sum(scores) / len(scores)


class SectionQualityResult(BaseModel):
    """Quality check result for a section."""
    section_name: str
    items: List[ItemQualityResult] = Field(default_factory=list)
    overall_status: QualityStatus = QualityStatus.PENDING
    overall_score: float = Field(default=0.0, ge=0.0, le=1.0)
    total_items: int = 0
    passed_items: int = 0
    failed_items: int = 0
    warning_items: int = 0
    
    def add_item_result(self, item_result: ItemQualityResult) -> None:
        """Add an item quality result."""
        self.items.append(item_result)
        self._update_section_stats()
    
    def _update_section_stats(self) -> None:
        """Update section-level statistics."""
        self.total_items = len(self.items)
        self.passed_items = sum(1 for item in self.items if item.overall_status == QualityStatus.PASS)
        self.failed_items = sum(1 for item in self.items if item.overall_status == QualityStatus.FAIL)
        self.warning_items = sum(1 for item in self.items if item.overall_status == QualityStatus.WARNING)
        
        # Calculate overall section status
        if self.failed_items > 0:
            self.overall_status = QualityStatus.FAIL
        elif self.warning_items > 0:
            self.overall_status = QualityStatus.WARNING
        elif self.passed_items > 0:
            self.overall_status = QualityStatus.PASS
        
        # Calculate overall score
        if self.items:
            self.overall_score = sum(item.overall_score for item in self.items) / len(self.items)


class QualityCheckResults(BaseModel):
    """Comprehensive quality check results from QualityAssuranceAgent."""
    session_id: Optional[str] = None
    trace_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    
    # Section results
    sections: List[SectionQualityResult] = Field(default_factory=list)
    
    # Overall statistics
    overall_status: QualityStatus = QualityStatus.PENDING
    overall_score: float = Field(default=0.0, ge=0.0, le=1.0)
    total_sections: int = 0
    total_items: int = 0
    passed_items: int = 0
    failed_items: int = 0
    warning_items: int = 0
    
    # Processing metadata
    processing_time_seconds: Optional[float] = None
    key_terms_used: List[str] = Field(default_factory=list)
    quality_criteria: Dict[str, Any] = Field(default_factory=dict)
    error_message: Optional[str] = None
    
    # Legacy compatibility
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        arbitrary_types_allowed = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    def add_section_result(self, section_result: SectionQualityResult) -> None:
        """Add a section quality result."""
        self.sections.append(section_result)
        self._update_overall_stats()
    
    def _update_overall_stats(self) -> None:
        """Update overall statistics."""
        self.total_sections = len(self.sections)
        self.total_items = sum(section.total_items for section in self.sections)
        self.passed_items = sum(section.passed_items for section in self.sections)
        self.failed_items = sum(section.failed_items for section in self.sections)
        self.warning_items = sum(section.warning_items for section in self.sections)
        
        # Calculate overall status
        if self.failed_items > 0:
            self.overall_status = QualityStatus.FAIL
        elif self.warning_items > 0:
            self.overall_status = QualityStatus.WARNING
        elif self.passed_items > 0:
            self.overall_status = QualityStatus.PASS
        
        # Calculate overall score
        if self.sections:
            self.overall_score = sum(section.overall_score for section in self.sections) / len(self.sections)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for backward compatibility."""
        return self.model_dump()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QualityCheckResults":
        """Create from dictionary for backward compatibility."""
        return cls(**data)
    
    @classmethod
    def create_empty(cls) -> "QualityCheckResults":
        """Create empty quality check results."""
        return cls()
    
    @classmethod
    def create_failed(cls, error_message: str) -> "QualityCheckResults":
        """Create failed quality check results."""
        return cls(
            overall_status=QualityStatus.FAIL,
            error_message=error_message
        )