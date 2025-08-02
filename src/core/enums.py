"""Core enums for the CV workflow system."""

from enum import Enum


class WorkflowNodes(str, Enum):
    """Enumeration for workflow node identifiers.

    Provides type safety for routing decisions and ensures consistency
    across the workflow graph implementation.
    """

    # Main workflow nodes
    JD_PARSER = "jd_parser"
    CV_PARSER = "cv_parser"
    RESEARCH = "research"
    CV_ANALYZER = "cv_analyzer"
    SUPERVISOR = "supervisor"
    FORMATTER = "formatter"
    ERROR_HANDLER = "error_handler"

    # Subgraph nodes
    KEY_QUALIFICATIONS_SUBGRAPH = "key_qualifications_subgraph"
    PROFESSIONAL_EXPERIENCE_SUBGRAPH = "professional_experience_subgraph"
    PROJECTS_SUBGRAPH = "projects_subgraph"
    EXECUTIVE_SUMMARY_SUBGRAPH = "executive_summary_subgraph"

    # Content generation nodes
    GENERATE = "generate"
    QA = "qa"
    HANDLE_FEEDBACK = "handle_feedback"
    REGENERATE = "regenerate"
    CONTINUE = "continue"
    ERROR = "error"
    PREPARE_REGENERATION = "prepare_regeneration"

    # Section-specific nodes
    KEY_QUALIFICATIONS = "key_qualifications"
    PROFESSIONAL_EXPERIENCE = "professional_experience"
    PROJECT_EXPERIENCE = "project_experience"
    EXECUTIVE_SUMMARY = "executive_summary"
