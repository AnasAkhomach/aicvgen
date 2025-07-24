"""Node helper functions for mapping state to agent inputs and updating CV with agent outputs.

This module provides reusable mapper and updater functions that abstract common
logic used across different writer nodes in the content generation workflow.
"""

from typing import Any, Dict, Optional
from src.models.cv_models import StructuredCV, ItemStatus
from src.models.agent_input_models import (
    KeyQualificationsWriterAgentInput,
    ProfessionalExperienceWriterAgentInput,
    ProjectsWriterAgentInput,
    ExecutiveSummaryWriterAgentInput
)
from src.utils.cv_data_factory import update_item_by_id, get_item_by_id
from src.config.logging_config import get_logger

logger = get_logger(__name__)


def map_state_to_key_qualifications_input(state: Dict[str, Any]) -> KeyQualificationsWriterAgentInput:
    """Map GlobalState to KeyQualificationsWriterAgentInput.
    
    Args:
        state: The current workflow state
        
    Returns:
        KeyQualificationsWriterAgentInput: Mapped input for the agent
    """
    return KeyQualificationsWriterAgentInput(
        structured_cv=state["structured_cv"],
        job_description_data=state["parsed_jd"],
        current_item_id=state.get("current_item_id"),
        research_findings=state.get("research_data"),
        session_id=state.get("session_id", "default")
    )


def map_state_to_professional_experience_input(state: Dict[str, Any]) -> ProfessionalExperienceWriterAgentInput:
    """Map GlobalState to ProfessionalExperienceWriterAgentInput.
    
    Args:
        state: The current workflow state
        
    Returns:
        ProfessionalExperienceWriterAgentInput: Mapped input for the agent
    """
    return ProfessionalExperienceWriterAgentInput(
        structured_cv=state["structured_cv"],
        job_description_data=state["parsed_jd"],
        current_item_id=state.get("current_item_id"),
        research_findings=state.get("research_data"),
        session_id=state.get("session_id", "default")
    )


def map_state_to_projects_input(state: Dict[str, Any]) -> ProjectsWriterAgentInput:
    """Map GlobalState to ProjectsWriterAgentInput.
    
    Args:
        state: The current workflow state
        
    Returns:
        ProjectsWriterAgentInput: Mapped input for the agent
    """
    return ProjectsWriterAgentInput(
        structured_cv=state["structured_cv"],
        job_description_data=state["parsed_jd"],
        current_item_id=state.get("current_item_id"),
        research_findings=state.get("research_data"),
        session_id=state.get("session_id", "default")
    )


def map_state_to_executive_summary_input(state: Dict[str, Any]) -> ExecutiveSummaryWriterAgentInput:
    """Map GlobalState to ExecutiveSummaryWriterAgentInput.
    
    Args:
        state: The current workflow state
        
    Returns:
        ExecutiveSummaryWriterAgentInput: Mapped input for the agent
    """
    # Extract required data from state
    structured_cv = state["structured_cv"]
    parsed_jd = state["parsed_jd"]
    research_data = state.get("research_data", {})
    
    # Extract string fields from structured_cv for agent input
    def extract_section_content(sections, section_name):
        """Extract content from a specific section."""
        for section in sections:
            if section.name.lower() == section_name.lower():
                return "\n".join([item.content for item in section.items if item.content])
        return ""
    
    job_description = getattr(parsed_jd, 'job_description', '') or str(parsed_jd)
    key_qualifications = extract_section_content(structured_cv.sections, "key qualifications")
    professional_experience = extract_section_content(structured_cv.sections, "professional experience")
    projects = extract_section_content(structured_cv.sections, "projects")
    
    return ExecutiveSummaryWriterAgentInput(
        job_description=job_description,
        key_qualifications=key_qualifications,
        professional_experience=professional_experience,
        projects=projects,
        research_findings=research_data
    )


def update_cv_with_key_qualifications_data(state: Dict[str, Any], agent_output: Dict[str, Any], item_id: Optional[str] = None) -> Dict[str, Any]:
    """Update state with key qualifications data from agent output.
    
    Args:
        state: The current workflow state
        agent_output: Output from the KeyQualificationsWriterAgent
        item_id: Optional item ID to update
        
    Returns:
        Dict[str, Any]: Updated state
    """
    try:
        updated_state = {**state, "last_executed_node": "KEY_QUALIFICATIONS_WRITER"}
        
        if "generated_key_qualifications" in agent_output:
            generated_content = agent_output["generated_key_qualifications"]
            
            # Extract bullet points from the result
            if hasattr(generated_content, 'bullet_points') and generated_content.bullet_points:
                content_text = "\n".join([f"• {bullet}" for bullet in generated_content.bullet_points])
            else:
                content_text = str(generated_content)
            
            # Use the item_id from state if not provided
            target_item_id = item_id or state.get("current_item_id")
            
            if target_item_id:
                updated_cv = update_item_by_id(
                    state["structured_cv"],
                    target_item_id,
                    {"content": content_text, "status": ItemStatus.COMPLETED}
                )
                updated_state["structured_cv"] = updated_cv
        
        return updated_state
        
    except Exception as exc:
        logger.error(f"Error updating CV with key qualifications data: {exc}")
        return {**state, "last_executed_node": "KEY_QUALIFICATIONS_WRITER"}


def update_cv_with_professional_experience_data(state: Dict[str, Any], agent_output: Dict[str, Any], item_id: Optional[str] = None) -> Dict[str, Any]:
    """Update state with professional experience data from agent output.
    
    Args:
        state: The current workflow state
        agent_output: Output from the ProfessionalExperienceWriterAgent
        item_id: Optional item ID to update
        
    Returns:
        Dict[str, Any]: Updated state
    """
    try:
        updated_state = {**state, "last_executed_node": "PROFESSIONAL_EXPERIENCE_WRITER"}
        
        if "generated_professional_experience" in agent_output:
            generated_content = agent_output["generated_professional_experience"]
            
            # Extract bullet points or description from the result
            if hasattr(generated_content, 'bullet_points') and generated_content.bullet_points:
                content_text = "\n".join([f"• {bullet}" for bullet in generated_content.bullet_points])
            elif hasattr(generated_content, 'description') and generated_content.description:
                content_text = generated_content.description
            else:
                content_text = str(generated_content)
            
            # Use the item_id from state if not provided
            target_item_id = item_id or state.get("current_item_id")
            
            if target_item_id:
                updated_cv = update_item_by_id(
                    state["structured_cv"],
                    target_item_id,
                    {"content": content_text, "status": ItemStatus.COMPLETED}
                )
                updated_state["structured_cv"] = updated_cv
        
        return updated_state
        
    except Exception as exc:
        logger.error(f"Error updating CV with professional experience data: {exc}")
        return {**state, "last_executed_node": "PROFESSIONAL_EXPERIENCE_WRITER"}


def update_cv_with_project_data(state: Dict[str, Any], agent_output: Dict[str, Any], item_id: Optional[str] = None) -> Dict[str, Any]:
    """Update state with project data from agent output.
    
    Args:
        state: The current workflow state
        agent_output: Output from the ProjectsWriterAgent
        item_id: Optional item ID to update
        
    Returns:
        Dict[str, Any]: Updated state
    """
    try:
        updated_state = {**state, "last_executed_node": "PROJECTS_WRITER"}
        
        if "generated_projects" in agent_output:
            generated_content = agent_output["generated_projects"]
            
            # Extract bullet points or description from the result
            if hasattr(generated_content, 'bullet_points') and generated_content.bullet_points:
                content_text = "\n".join([f"• {bullet}" for bullet in generated_content.bullet_points])
            elif hasattr(generated_content, 'description') and generated_content.description:
                content_text = generated_content.description
            else:
                content_text = str(generated_content)
            
            # Use the item_id from state if not provided
            target_item_id = item_id or state.get("current_item_id")
            
            if target_item_id:
                updated_cv = update_item_by_id(
                     state["structured_cv"],
                     target_item_id,
                     {"content": content_text, "status": ItemStatus.COMPLETED}
                 )
                updated_state["structured_cv"] = updated_cv
        
        return updated_state
        
    except Exception as exc:
        logger.error(f"Error updating CV with project data: {exc}")
        return {**state, "last_executed_node": "PROJECTS_WRITER"}


def update_cv_with_executive_summary_data(state: Dict[str, Any], agent_output: Dict[str, Any], item_id: Optional[str] = None) -> Dict[str, Any]:
    """Update state with executive summary data from agent output.
    
    Args:
        state: The current workflow state
        agent_output: Output from the ExecutiveSummaryWriterAgent
        item_id: Optional item ID to update
        
    Returns:
        Dict[str, Any]: Updated state
    """
    try:
        updated_state = {**state, "last_executed_node": "EXECUTIVE_SUMMARY_WRITER"}
        
        if "generated_executive_summary" in agent_output:
            generated_content = agent_output["generated_executive_summary"]
            
            # Extract summary content
            if hasattr(generated_content, 'summary') and generated_content.summary:
                content_text = generated_content.summary
            else:
                content_text = str(generated_content)
            
            # Use the item_id from state if not provided
            target_item_id = item_id or state.get("current_item_id")
            
            if target_item_id:
                updated_cv = update_item_by_id(
                    state["structured_cv"],
                    target_item_id,
                    {"content": content_text, "status": ItemStatus.COMPLETED}
                )
                updated_state["structured_cv"] = updated_cv
        
        return updated_state
        
    except Exception as exc:
        logger.error(f"Error updating CV with executive summary data: {exc}")
        return {**state, "last_executed_node": "EXECUTIVE_SUMMARY_WRITER"}