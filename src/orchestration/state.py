"""Defines the centralized state model for the LangGraph-based orchestration."""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from src.models.data_models import JobDescriptionData, StructuredCV


class AgentState(BaseModel):
    """
    Represents the complete, centralized state of the CV generation workflow.
    This model is designed to be compatible with LangGraph's StateGraph.
    """
    structured_cv: StructuredCV
    job_description_data: JobDescriptionData

    # Fields for managing granular processing and user interaction
    current_item_id: Optional[str] = Field(None, description="The ID of the CV item being processed.")
    current_section_key: Optional[str] = Field(None, description="The key of the CV section being processed.")
    user_feedback: Optional[Dict[str, Any]] = Field(None, description="User feedback for regeneration.")

    # Fields for agent outputs
    research_findings: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Findings from the ResearchAgent.")
    error_messages: List[str] = Field(default_factory=list, description="A list of accumulated error messages.")
    final_cv_output_path: Optional[str] = Field(None, description="The file path of the final generated CV.")

    class Config:
        """Pydantic config to allow arbitrary types, needed for some Langchain integrations."""
        arbitrary_types_allowed = True