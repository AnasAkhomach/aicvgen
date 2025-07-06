"""Defines the centralized state model for the LangGraph-based orchestration.

NOTE: AgentState is the single source of truth for workflow execution. It is created from UI input at the start and archived at the end. No UI or persistence logic should modify AgentState during workflow execution.
"""

import uuid
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from src.models.agent_output_models import (
    CVAnalysisResult,
    QualityAssuranceAgentOutput,
    ResearchFindings,
)
from src.models.cv_models import JobDescriptionData, StructuredCV
from src.models.workflow_models import ContentType, UserFeedback


class AgentState(BaseModel):
    """
    Represents the complete, centralized state of the CV generation workflow
    for LangGraph orchestration.
    """

    # Observability
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    # Core Data Models
    structured_cv: StructuredCV
    job_description_data: Optional[JobDescriptionData] = None
    cv_text: str  # The raw text of the user's CV

    # Workflow Control & Granular Processing
    # The key of the section currently being processed (e.g., "professional_experience")
    current_section_key: Optional[str] = None
    # Index to track the current position in the WORKFLOW_SEQUENCE
    current_section_index: int = 0
    # A queue of item IDs (subsections) for the current section to be processed one by one.
    items_to_process_queue: List[str] = Field(default_factory=list)
    # The ID of the specific role, project, or item currently being processed by an agent.
    current_item_id: Optional[str] = None
    # The type of content currently being processed (for error handling context)
    current_content_type: Optional[ContentType] = None
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

    # Workflow status for pausable execution
    workflow_status: str = Field(
        default="PROCESSING",
        description="Current workflow status: PROCESSING, AWAITING_FEEDBACK, COMPLETED, ERROR",
    )

    # UI display data for feedback interface
    ui_display_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Data to be displayed in the UI for user feedback",
    )

    class Config:
        arbitrary_types_allowed = True

    # State mutation methods with validation
    def set_user_feedback(self, feedback: UserFeedback) -> None:
        """Set user feedback with validation.
        Args:
            feedback: UserFeedback instance with validated data
        Raises:
            ValueError: If feedback is invalid
        """
        if not isinstance(feedback, UserFeedback):
            raise ValueError("feedback must be a UserFeedback instance")
        self.user_feedback = feedback

    def set_research_findings(self, findings: ResearchFindings) -> None:
        """Set research findings with validation.
        Args:
            findings: ResearchFindings instance with validated data
        Raises:
            ValueError: If findings is invalid
        """
        if not isinstance(findings, ResearchFindings):
            raise ValueError("findings must be a ResearchFindings instance")
        self.research_findings = findings

    def set_quality_check_results(self, results: QualityAssuranceAgentOutput) -> None:
        """Set quality check results with validation.
        Args:
            results: QualityAssuranceAgentOutput instance with validated data
        Raises:
            ValueError: If results is invalid
        """
        if not isinstance(results, QualityAssuranceAgentOutput):
            raise ValueError("results must be a QualityAssuranceAgentOutput instance")
        self.quality_check_results = results

    def set_cv_analysis_results(self, results: CVAnalysisResult) -> None:
        """Set CV analysis results with validation.
        Args:
            results: CVAnalysisResult instance with validated data
        Raises:
            ValueError: If results is invalid
        """
        if not isinstance(results, CVAnalysisResult):
            raise ValueError("results must be a CVAnalysisResult instance")
        self.cv_analysis_results = results

    def add_error_message(self, message: str) -> None:
        """Add an error message with validation.
        Args:
            message: Error message string
        Raises:
            ValueError: If message is not a valid string
        """
        if not isinstance(message, str) or not message.strip():
            raise ValueError("message must be a non-empty string")
        # pylint: disable=no-member
        self.error_messages.append(message.strip())

    def clear_error_messages(self) -> None:
        """Clear all error messages."""
        # pylint: disable=no-member
        self.error_messages.clear()

    def set_current_section(self, section_key: str, section_index: int) -> None:
        """Set current section with validation.
        Args:
            section_key: The section key being processed
            section_index: The index in the workflow sequence
        Raises:
            ValueError: If parameters are invalid
        """
        if not isinstance(section_key, str) or not section_key.strip():
            raise ValueError("section_key must be a non-empty string")
        if not isinstance(section_index, int) or section_index < 0:
            raise ValueError("section_index must be a non-negative integer")

        self.current_section_key = section_key.strip()
        self.current_section_index = section_index

    def set_current_item(
        self, item_id: str, content_type: Optional[ContentType] = None
    ) -> None:
        """Set current item being processed with validation.
        Args:
            item_id: The ID of the item being processed
            content_type: Optional content type for context
        Raises:
            ValueError: If item_id is invalid
        """
        if not isinstance(item_id, str) or not item_id.strip():
            raise ValueError("item_id must be a non-empty string")

        self.current_item_id = item_id.strip()
        if content_type is not None:
            self.current_content_type = content_type

    def update_processing_queue(self, items: List[str]) -> None:
        """Update the items to process queue with validation.
        Args:
            items: List of item IDs to process
        Raises:
            ValueError: If items list is invalid
        """
        if not isinstance(items, list):
            raise ValueError("items must be a list")

        # Validate all items are non-empty strings
        validated_items = []
        for item in items:
            if not isinstance(item, str) or not item.strip():
                raise ValueError("All items must be non-empty strings")
            validated_items.append(item.strip())

        self.items_to_process_queue = validated_items

    def update_content_generation_queue(self, items: List[str]) -> None:
        """Update the content generation queue with validation.
        Args:
            items: List of item IDs for content generation
        Raises:
            ValueError: If items list is invalid
        """
        if not isinstance(items, list):
            raise ValueError("items must be a list")

        # Validate all items are non-empty strings
        validated_items = []
        for item in items:
            if not isinstance(item, str) or not item.strip():
                raise ValueError("All items must be non-empty strings")
            validated_items.append(item.strip())

        self.content_generation_queue = validated_items

    def set_final_output_path(self, path: str) -> None:
        """Set the final output path with validation.
        Args:
            path: File path to the generated output
        Raises:
            ValueError: If path is invalid
        """
        if not isinstance(path, str) or not path.strip():
            raise ValueError("path must be a non-empty string")
        self.final_output_path = path.strip()

    def update_node_metadata(self, node_name: str, metadata: Dict[str, Any]) -> None:
        """Update node execution metadata with validation.
        Args:
            node_name: Name of the node
            metadata: Metadata dictionary
        Raises:
            ValueError: If parameters are invalid
        """
        if not isinstance(node_name, str) or not node_name.strip():
            raise ValueError("node_name must be a non-empty string")
        if not isinstance(metadata, dict):
            raise ValueError("metadata must be a dictionary")

        self.node_execution_metadata[node_name.strip()] = metadata.copy()

    def set_workflow_status(self, status: str) -> "AgentState":
        """Set workflow status with validation.
        Args:
            status: Workflow status (PROCESSING, AWAITING_FEEDBACK, COMPLETED, ERROR)
        Returns:
            New AgentState instance with updated workflow status
        Raises:
            ValueError: If status is invalid
        """
        valid_statuses = {"PROCESSING", "AWAITING_FEEDBACK", "COMPLETED", "ERROR"}
        if not isinstance(status, str) or status not in valid_statuses:
            raise ValueError(f"status must be one of {valid_statuses}")
        return self.model_copy(update={"workflow_status": status})

    def set_ui_display_data(self, data: Dict[str, Any]) -> "AgentState":
        """Set UI display data with validation.
        Args:
            data: Dictionary containing UI display data
        Returns:
            New AgentState instance with updated UI display data
        Raises:
            ValueError: If data is invalid
        """
        if not isinstance(data, dict):
            raise ValueError("data must be a dictionary")
        return self.model_copy(update={"ui_display_data": data.copy()})

    def update_ui_display_data(self, key: str, value: Any) -> "AgentState":
        """Update specific UI display data field.
        Args:
            key: The key to update
            value: The value to set
        Returns:
            New AgentState instance with updated UI display data
        Raises:
            ValueError: If key is invalid
        """
        if not isinstance(key, str) or not key.strip():
            raise ValueError("key must be a non-empty string")
        # pylint: disable=no-member
        updated_data = self.ui_display_data.copy()
        updated_data[key.strip()] = value
        return self.model_copy(update={"ui_display_data": updated_data})

    # Field validators
    @field_validator("error_messages")
    @classmethod
    def validate_error_messages(cls, v):
        """Validate error messages list."""
        if not isinstance(v, list):
            raise ValueError("error_messages must be a list")
        for msg in v:
            if not isinstance(msg, str):
                raise ValueError("All error messages must be strings")
        return v

    @field_validator("current_section_index")
    @classmethod
    def validate_section_index(cls, v):
        """Validate section index is non-negative."""
        if v < 0:
            raise ValueError("current_section_index must be non-negative")
        return v
