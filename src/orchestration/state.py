"""Defines the centralized state model for the LangGraph-based orchestration.

NOTE: AgentState is the single source of truth for workflow execution. It is created from UI input at the start and archived at the end. No UI or persistence logic should modify AgentState during workflow execution.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
import uuid

from ..models.data_models import JobDescriptionData, StructuredCV, UserFeedback
from ..models.agent_output_models import (
    CVAnalysisResult,
    QualityAssuranceAgentOutput,
    ResearchFindings,
)


class AgentState(BaseModel):
    """
    Represents the complete, centralized state of the CV generation workflow
    for LangGraph orchestration.
    """

    # Observability
    session_id: Optional[str] = None
    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    # Core Data Models
    structured_cv: StructuredCV
    job_description_data: Optional[JobDescriptionData] = None
    cv_text: str  # The raw text of the user's CV

    # Workflow Control & Granular Processing
    # The key of the section currently being processed (e.g., "professional_experience")
    current_section_key: Optional[str] = None
    # A queue of item IDs (subsections) for the current section to be processed one by one.
    items_to_process_queue: List[str] = Field(default_factory=list)
    # The ID of the specific role, project, or item currently being processed by an agent.
    current_item_id: Optional[str] = None
    # Flag to indicate if this is the first pass or a user-driven regeneration.
    is_initial_generation: bool = True

    # Content Generation Queue for Explicit Loop Processing
    # Queue of item IDs that need content generation/optimization, supporting both batch and single-item processing
    content_generation_queue: List[str] = Field(default_factory=list)

    # User Feedback for Regeneration
    # Stores feedback from the UI to guide the next generation cycle.
    user_feedback: Optional[UserFeedback] = None  # Agent Outputs & Finalization
    # Research findings from the ResearchAgent
    research_findings: Optional[ResearchFindings] = None
    # Quality check results from the QualityAssuranceAgent
    quality_check_results: Optional[QualityAssuranceAgentOutput] = None
    # CV analysis results from the CVAnalysisAgent
    cv_analysis_results: Optional[CVAnalysisResult] = None
    # Path to the final generated PDF file.
    final_output_path: Optional[str] = None
    # Accumulated error messages from the workflow.
    error_messages: List[str] = Field(default_factory=list)

    # CB-02 Fix: Generic field for node-specific metadata
    node_execution_metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True
