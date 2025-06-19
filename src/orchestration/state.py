"""Defines the centralized state model for the LangGraph-based orchestration."""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
import uuid

from ..models.data_models import JobDescriptionData, StructuredCV, UserFeedback


class AgentState(BaseModel):
    """
    Represents the complete, centralized state of the CV generation workflow
    for LangGraph orchestration.
    """
    # Observability
    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    
    # Core Data Models
    structured_cv: StructuredCV
    job_description_data: Optional[JobDescriptionData] = None

    # Input Data for Processing
    # Raw CV text input for parsing
    cv_text: Optional[str] = None
    # Flag to indicate starting from scratch (no existing CV)
    start_from_scratch: Optional[bool] = None

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
    user_feedback: Optional[UserFeedback] = None

    # Agent Outputs & Finalization
    # Research findings from the ResearchAgent
    research_findings: Optional[Dict[str, Any]] = None
    # Quality check results from the QualityAssuranceAgent
    quality_check_results: Optional[Dict[str, Any]] = None
    # Path to the final generated PDF file.
    final_output_path: Optional[str] = None
    # Accumulated error messages from the workflow.
    error_messages: List[str] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True