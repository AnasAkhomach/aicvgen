"""Workflow control and routing nodes for the CV workflow."""

import logging
from typing import Dict, Any, List

from src.orchestration.state import GlobalState
from src.core.enums import WorkflowNodes

logger = logging.getLogger(__name__)

# Workflow sequence for sections
WORKFLOW_SEQUENCE = [
    "key_qualifications",
    "professional_experience", 
    "project_experience",
    "executive_summary"
]


async def initialize_supervisor_node(state: GlobalState) -> GlobalState:
    """Initialize supervisor state for workflow routing.
    
    Args:
        state: Current workflow state
        
    Returns:
        Updated state with supervisor initialization
    """
    logger.info("Initializing supervisor state")
    
    try:
        # Initialize supervisor state based on structured_cv and workflow sequence
        supervisor_state = _initialize_supervisor_state(
            state["structured_cv"], 
            WORKFLOW_SEQUENCE
        )
        
        # Remove supervisor_metadata from supervisor_state to avoid overwriting node_execution_metadata
        supervisor_state.pop("supervisor_metadata", {})
        
        updated_state = {
            **state,
            **supervisor_state,
            "last_executed_node": "INITIALIZE_SUPERVISOR"
        }
        
        logger.info("Supervisor state initialized successfully")
        return updated_state
        
    except Exception as exc:
        logger.error(f"Error initializing supervisor state: {exc}")
        error_messages = list(state["error_messages"]) if state["error_messages"] else []
        error_messages.append(f"Supervisor initialization failed: {str(exc)}")
        
        return {
            **state,
            "error_messages": error_messages,
            "last_executed_node": "INITIALIZE_SUPERVISOR"
        }


async def supervisor_node(state: GlobalState) -> GlobalState:
    """Supervisor node for workflow routing and control.
    
    Args:
        state: Current workflow state
        
    Returns:
        Updated state with routing decisions
    """
    logger.info("Starting Supervisor node")
    
    try:
        # Handle state access patterns
        current_section_index = state.get("current_section_index", 0)
        workflow_sections = state.get("workflow_sections", WORKFLOW_SEQUENCE)
        error_messages = state.get("error_messages", [])
        user_feedback = state.get("user_feedback", "")
        
        # Check for errors first
        if error_messages:
            logger.warning(f"Errors detected: {error_messages}")
            next_node = WorkflowNodes.ERROR_HANDLER.value
        # Check for user regeneration request
        elif user_feedback and "regenerate" in user_feedback.lower():
            logger.info("User requested regeneration")
            # Stay in current section for regeneration
            current_section = workflow_sections[current_section_index]
            next_node = f"{current_section.upper()}_SUBGRAPH"
        # Check if all sections are completed
        elif current_section_index >= len(workflow_sections):
            logger.info("All content sections completed, proceeding to formatter")
            next_node = WorkflowNodes.FORMATTER.value
        else:
            # Increment section index after subgraph completion
            if state.get("subgraph_completed", False):
                current_section_index += 1
                
                # Find next non-empty section
                while (current_section_index < len(workflow_sections) and 
                       not workflow_sections[current_section_index]):
                    current_section_index += 1
            
            # Determine next node based on current section
            if current_section_index < len(workflow_sections):
                current_section = workflow_sections[current_section_index]
                next_node = f"{current_section.upper()}_SUBGRAPH"
                logger.info(f"Routing to section: {current_section}")
            else:
                next_node = WorkflowNodes.FORMATTER.value
                logger.info("All sections completed, routing to formatter")
        
        # Update node execution metadata
        node_execution_metadata = dict(state.get("node_execution_metadata", {}))
        node_execution_metadata.update({
            "next_node": next_node,
            "last_executed_node": "SUPERVISOR"
        })
        
        updated_state = {
            **state,
            "current_section_index": current_section_index,
            "next_node": next_node,
            "node_execution_metadata": node_execution_metadata,
            "last_executed_node": "SUPERVISOR",
            "subgraph_completed": False  # Reset for next iteration
        }
        
        logger.info(f"Supervisor routing decision: {next_node}")
        return updated_state
        
    except Exception as exc:
        logger.error(f"Error in Supervisor node: {exc}")
        error_messages = list(state["error_messages"]) if state["error_messages"] else []
        error_messages.append(f"Supervisor routing failed: {str(exc)}")
        
        return {
            **state,
            "error_messages": error_messages,
            "next_node": WorkflowNodes.ERROR_HANDLER.value,
            "last_executed_node": "SUPERVISOR"
        }


async def handle_feedback_node(state: GlobalState) -> GlobalState:
    """Handle user feedback for content approval or regeneration.
    
    Args:
        state: Current workflow state
        
    Returns:
        Updated state with feedback processing
    """
    logger.info("Starting Handle Feedback node")
    
    try:
        user_feedback = state.get("user_feedback", "")
        automated_mode = state.get("automated_mode", False)
        current_section_index = state.get("current_section_index", 0)
        workflow_sections = state.get("workflow_sections", WORKFLOW_SEQUENCE)
        
        # Determine current section for UI display
        current_section = None
        if current_section_index < len(workflow_sections):
            current_section = workflow_sections[current_section_index]
        
        # Prepare UI display data
        ui_display_data = {
            "current_section": current_section,
            "section_index": current_section_index,
            "total_sections": len(workflow_sections)
        }
        
        # Add content preview if available
        if current_section and current_section in state["structured_cv"]:
            ui_display_data["content_preview"] = state["structured_cv"][current_section]
        
        # Process feedback
        if automated_mode or (user_feedback and "approve" in user_feedback.lower()):
            # Mark completion and continue
            workflow_status = "PROCESSING"
            next_action = "MARK_COMPLETION"
        elif user_feedback and "regenerate" in user_feedback.lower():
            # Regenerate current section
            workflow_status = "PROCESSING"
            next_action = WorkflowNodes.REGENERATE.value
        else:
            # Wait for user feedback
            workflow_status = "AWAITING_FEEDBACK"
            next_action = "AWAITING_FEEDBACK"
        
        updated_state = {
            **state,
            "workflow_status": workflow_status,
            "next_action": next_action,
            "ui_display_data": ui_display_data,
            "last_executed_node": "HANDLE_FEEDBACK"
        }
        
        logger.info(f"Feedback processed: {next_action}")
        return updated_state
        
    except Exception as exc:
        logger.error(f"Error in Handle Feedback node: {exc}")
        error_messages = list(state["error_messages"]) if state["error_messages"] else []
        error_messages.append(f"Feedback handling failed: {str(exc)}")
        
        return {
            **state,
            "error_messages": error_messages,
            "last_executed_node": "HANDLE_FEEDBACK"
        }


async def mark_subgraph_completion_node(state: GlobalState) -> GlobalState:
    """Mark the completion of a subgraph.
    
    Args:
        state: Current workflow state
        
    Returns:
        Updated state with completion marker
    """
    logger.info("Marking subgraph completion")
    
    try:
        # Update metadata to indicate subgraph completion
        node_execution_metadata = dict(state.get("node_execution_metadata", {}))
        node_execution_metadata["subgraph_completed"] = True
        
        updated_state = {
            **state,
            "subgraph_completed": True,
            "node_execution_metadata": node_execution_metadata,
            "last_executed_node": "MARK_SUBGRAPH_COMPLETION"
        }
        
        logger.info("Subgraph completion marked")
        return updated_state
        
    except Exception as exc:
        logger.error(f"Error marking subgraph completion: {exc}")
        error_messages = list(state["error_messages"]) if state["error_messages"] else []
        error_messages.append(f"Subgraph completion marking failed: {str(exc)}")
        
        return {
            **state,
            "error_messages": error_messages,
            "last_executed_node": "MARK_SUBGRAPH_COMPLETION"
        }


async def entry_router_node(state: GlobalState) -> GlobalState:
    """Entry router to determine workflow starting point.
    
    Args:
        state: Current workflow state
        
    Returns:
        Updated state with entry routing decision
    """
    logger.info("Starting Entry Router node")
    
    try:
        # Check if we have existing job data, research, and analysis
        has_job_data = bool(state.get("parsed_jd"))
        has_research = bool(state.get("research_data"))
        has_analysis = bool(state.get("cv_analysis"))
        
        # Determine starting point
        if has_job_data and has_research and has_analysis:
            # Skip to supervisor if all prerequisite data exists
            next_node = WorkflowNodes.SUPERVISOR.value
            logger.info("Existing data found, routing to supervisor")
        else:
            # Start from JD parsing
            next_node = WorkflowNodes.JD_PARSER.value
            logger.info("No existing data, starting from JD parser")
        
        updated_state = {
            **state,
            "entry_route": next_node,
            "last_executed_node": "ENTRY_ROUTER"
        }
        
        return updated_state
        
    except Exception as exc:
        logger.error(f"Error in Entry Router node: {exc}")
        error_messages = list(state["error_messages"]) if state["error_messages"] else []
        error_messages.append(f"Entry routing failed: {str(exc)}")
        
        return {
            **state,
            "error_messages": error_messages,
            "last_executed_node": "ENTRY_ROUTER"
        }


def _initialize_supervisor_state(structured_cv, workflow_sequence: List[str]) -> Dict[str, Any]:
    """Initialize supervisor state based on structured CV and workflow sequence.
    
    Args:
        structured_cv: The structured CV data (StructuredCV object or None)
        workflow_sequence: List of workflow sections
        
    Returns:
        Dictionary with supervisor state initialization
    """
    # Handle None or empty structured_cv
    if not structured_cv or not hasattr(structured_cv, 'sections') or not structured_cv.sections:
        return {
            "workflow_sections": [],
            "current_section_index": None,
            "current_item_id": None,
            "supervisor_metadata": {
                "supervisor_initialized": True,
                "available_sections": []
            }
        }
    
    # Find first section with items
    current_section_index = None
    current_item_id = None
    
    for index, section in enumerate(structured_cv.sections):
        if section.items:  # Section has items
            current_section_index = index
            current_item_id = str(section.items[0].id)
            break
    
    return {
        "workflow_sections": workflow_sequence,
        "current_section_index": current_section_index,
        "current_item_id": current_item_id,
        "supervisor_metadata": {
            "supervisor_initialized": True,
            "available_sections": [section.name for section in structured_cv.sections if section.items]
        }
    }