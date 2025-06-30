"""State Management Utilities.

This module contains standalone, reusable functions for managing application state,
particularly for creating the initial AgentState. It is decoupled from Streamlit's
session state to be usable in any context (e.g., testing, background workers).
"""

from ..orchestration.state import AgentState
from ..models.cv_models import JobDescriptionData, StructuredCV, MetadataModel


def create_initial_agent_state(
    job_description_raw: str,
    cv_text: str,
    start_from_scratch: bool = False,
) -> AgentState:
    """Creates a fully initialized AgentState from raw inputs.

    This function is the canonical factory for AgentState, ensuring that every
    workflow run starts from a consistent, well-defined state. It is deliberately
    decoupled from any UI or session management library.

    Args:
        job_description_raw: The raw text of the job description.
        cv_text: The raw text of the user's CV.
        start_from_scratch: Flag indicating if the user wants to start a new CV.

    Returns:
        AgentState: A fully initialized AgentState object ready for the workflow.
    """
    job_description_data = JobDescriptionData(raw_text=job_description_raw)

    metadata = MetadataModel(
        extra={
            "original_cv_text": cv_text,
            "start_from_scratch": start_from_scratch,
        }
    )
    structured_cv = StructuredCV(metadata=metadata)

    initial_state = AgentState(
        structured_cv=structured_cv,
        job_description_data=job_description_data,
        cv_text=cv_text,
        user_feedback=None,
        error_messages=[],
        current_section_key=None,
        current_item_id=None,
        items_to_process_queue=[],
        is_initial_generation=True,
        final_output_path=None,
        research_findings=None,
    )

    return initial_state
