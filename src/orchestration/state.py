"""State management for the CV generation workflow.

This module defines the centralized state structure used throughout the CV generation
workflow. The state is implemented using TypedDict for immutability and composability
across the workflow graph and its subgraphs.
"""

from datetime import datetime
import uuid
from typing import Any, Dict, List, Optional, Annotated
import operator
try:
    from typing_extensions import TypedDict
except ImportError:
    from typing import TypedDict

from src.models.agent_output_models import (
    CVAnalysisResult,
    QualityAssuranceAgentOutput,
    ResearchFindings,
)
from src.models.cv_models import JobDescriptionData, StructuredCV
from src.models.workflow_models import ContentType, UserFeedback



class GlobalState(TypedDict):
    """
    Global state shared across all workflow graphs and subgraphs.
    
    This TypedDict represents the core state that is accessible to all nodes
    in the main workflow and its subgraphs. It contains essential data models,
    observability information, and workflow control data.
    """

    # Observability
    session_id: str
    trace_id: str

    # Core Data Models
    structured_cv: Optional[StructuredCV]
    job_description_data: Optional[JobDescriptionData]
    cv_text: str  # The raw text of the user's CV

    # Workflow Control & Granular Processing
    # The key of the section currently being processed (e.g., "professional_experience")
    current_section_key: Optional[str]
    # Index to track the current position in the WORKFLOW_SEQUENCE
    current_section_index: Optional[int]
    # A queue of item IDs (subsections) for the current section to be processed one by one.
    items_to_process_queue: List[str]
    # The ID of the specific role, project, or item currently being processed by an agent.
    current_item_id: Optional[str]
    # The type of content currently being processed (for error handling context)
    current_content_type: Optional[ContentType]
    # Flag to indicate if this is the first pass or a user-driven regeneration.
    is_initial_generation: bool

    # Content Generation Queue for Explicit Loop Processing
    # Queue of item IDs that need content generation/optimization, supporting both batch and single-item processing
    content_generation_queue: List[str]

    # User Feedback for Regeneration
    # Stores feedback from the UI to guide the next generation cycle.
    user_feedback: Optional[UserFeedback]  # Agent Outputs & Finalization
    # Research findings from the ResearchAgent
    research_findings: Optional[ResearchFindings]
    # Quality check results from the QualityAssuranceAgent
    quality_check_results: Optional[QualityAssuranceAgentOutput]
    # CV analysis results from the CVAnalysisAgent
    cv_analysis_results: Optional[CVAnalysisResult]
    # Generated key qualifications from KeyQualificationsWriterAgent
    generated_key_qualifications: Optional[List[str]]
    # Path to the final generated PDF file.
    final_output_path: Optional[str]
    # Accumulated error messages from the workflow.
    error_messages: Annotated[List[str], operator.add]

    # CB-02 Fix: Generic field for node-specific metadata
    node_execution_metadata: Dict[str, Any]

    # Workflow status for pausable execution
    # Current workflow status: PROCESSING, AWAITING_FEEDBACK, COMPLETED, ERROR
    workflow_status: str

    # UI display data for feedback interface
    # Data to be displayed in the UI for user feedback
    ui_display_data: Dict[str, Any]
    
    # Automated mode flag for testing (bypasses user feedback requirements)
    automated_mode: bool


class KeyQualificationsState(GlobalState):
    """State for Key Qualifications subgraph.
    
    Extends GlobalState with fields specific to key qualifications generation,
    quality assurance, and feedback handling.
    """
    
    # Key Qualifications specific fields
    qualifications_content: Optional[str]
    qualifications_qa_passed: bool
    qualifications_feedback_count: int
    qualifications_regeneration_count: int


class ProfessionalExperienceState(GlobalState):
    """State for Professional Experience subgraph.
    
    Extends GlobalState with fields specific to professional experience generation,
    quality assurance, and feedback handling.
    """
    
    # Professional Experience specific fields
    experience_content: Optional[str]
    experience_qa_passed: bool
    experience_feedback_count: int
    experience_regeneration_count: int


class ProjectsState(GlobalState):
    """State for Projects subgraph.
    
    Extends GlobalState with fields specific to projects generation,
    quality assurance, and feedback handling.
    """
    
    # Projects specific fields
    projects_content: Optional[str]
    projects_qa_passed: bool
    projects_feedback_count: int
    projects_regeneration_count: int


class ExecutiveSummaryState(GlobalState):
    """State for Executive Summary subgraph.
    
    Extends GlobalState with fields specific to executive summary generation,
    quality assurance, and feedback handling.
    """
    
    # Executive Summary specific fields
    summary_content: Optional[str]
    summary_qa_passed: bool
    summary_feedback_count: int
    summary_regeneration_count: int


# Maintain backward compatibility
AgentState = GlobalState


def create_global_state(
    cv_text: str,
    session_id: Optional[str] = None,
    trace_id: Optional[str] = None,
    automated_mode: bool = False,
    **kwargs: Any
) -> GlobalState:
    """Create a new GlobalState instance with default values.
    
    Args:
        cv_text: The raw text of the user's CV
        session_id: Optional session ID (generates UUID if not provided)
        trace_id: Optional trace ID (generates UUID if not provided)
        automated_mode: Whether to run in automated mode
        **kwargs: Additional fields to override defaults
        
    Returns:
        GlobalState instance with default values
    """
    return GlobalState(
        # Observability
        session_id=session_id or str(uuid.uuid4()),
        trace_id=trace_id or str(uuid.uuid4()),
        
        # Core Data Models
        structured_cv=kwargs.get('structured_cv'),
        job_description_data=kwargs.get('job_description_data'),
        cv_text=cv_text,
        
        # Workflow Control & Granular Processing
        current_section_key=kwargs.get('current_section_key'),
        current_section_index=kwargs.get('current_section_index'),
        items_to_process_queue=kwargs.get('items_to_process_queue', []),
        current_item_id=kwargs.get('current_item_id'),
        current_content_type=kwargs.get('current_content_type'),
        is_initial_generation=kwargs.get('is_initial_generation', True),
        
        # Content Generation Queue
        content_generation_queue=kwargs.get('content_generation_queue', []),
        
        # User Feedback for Regeneration
        user_feedback=kwargs.get('user_feedback'),
        
        # Agent Outputs & Finalization
        research_findings=kwargs.get('research_findings'),
        quality_check_results=kwargs.get('quality_check_results'),
        cv_analysis_results=kwargs.get('cv_analysis_results'),
        generated_key_qualifications=kwargs.get('generated_key_qualifications'),
        final_output_path=kwargs.get('final_output_path'),
        error_messages=kwargs.get('error_messages', []),
        
        # CB-02 Fix: Generic field for node-specific metadata
        node_execution_metadata=kwargs.get('node_execution_metadata', {}),
        
        # Workflow status for pausable execution
        workflow_status=kwargs.get('workflow_status', 'PROCESSING'),
        
        # UI display data for feedback interface
        ui_display_data=kwargs.get('ui_display_data', {}),
        
        # Automated mode flag
        automated_mode=automated_mode,
    )


# Maintain backward compatibility
def create_agent_state(
    cv_text: str,
    session_id: Optional[str] = None,
    trace_id: Optional[str] = None,
    automated_mode: bool = False,
    **kwargs: Any
) -> AgentState:
    """Create a new AgentState instance with default values.
    
    Deprecated: Use create_global_state instead.
    
    Args:
        cv_text: The raw text of the user's CV
        session_id: Optional session ID (generates UUID if not provided)
        trace_id: Optional trace ID (generates UUID if not provided)
        automated_mode: Whether to run in automated mode
        **kwargs: Additional fields to override defaults
        
    Returns:
        AgentState instance with default values
    """
    return create_global_state(
        cv_text=cv_text,
        session_id=session_id,
        trace_id=trace_id,
        automated_mode=automated_mode,
        **kwargs
    )
