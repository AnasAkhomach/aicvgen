"""Routing functions for the CV workflow graph."""

import logging
from typing import Dict, Any, Literal

from src.orchestration.state import GlobalState
from src.core.enums import WorkflowNodes

logger = logging.getLogger(__name__)


def route_after_content_generation(state: GlobalState) -> Literal["REGENERATE", "MARK_COMPLETION", "AWAITING_FEEDBACK", "ERROR"]:
    """Route after content generation based on errors, user feedback, or completion.
    
    Args:
        state: Current workflow state
        
    Returns:
        Next node identifier
    """
    try:
        # Check for errors first
        error_messages = state.get("error_messages", [])
        if error_messages:
            logger.warning(f"Errors detected in content generation: {error_messages}")
            return WorkflowNodes.ERROR.value
        
        # Check user feedback
        user_feedback = state.get("user_feedback", "")
        if user_feedback:
            if "regenerate" in user_feedback.lower():
                logger.info("User requested regeneration")
                return WorkflowNodes.REGENERATE.value
            elif "approve" in user_feedback.lower():
                logger.info("User approved content")
                return "MARK_COMPLETION"
        
        # Check if automated mode is enabled
        automated_mode = state.get("automated_mode", False)
        if automated_mode:
            logger.info("Automated mode enabled, marking completion")
            return "MARK_COMPLETION"
        
        # Default: await user feedback
        logger.info("Awaiting user feedback")
        return "AWAITING_FEEDBACK"
        
    except Exception as exc:
        logger.error(f"Error in content generation routing: {exc}")
        return WorkflowNodes.ERROR.value


def route_from_supervisor(state: GlobalState) -> str:
    """Route from supervisor based on next_node decision.
    
    Args:
        state: Current workflow state
        
    Returns:
        Next node identifier
    """
    try:
        next_node = state.get("next_node")
        
        if not next_node:
            logger.warning("No next_node specified in supervisor state")
            return WorkflowNodes.ERROR_HANDLER.value
        
        # Map section names to subgraph nodes
        section_to_subgraph = {
            "KEY_QUALIFICATIONS_SUBGRAPH": WorkflowNodes.KEY_QUALIFICATIONS_SUBGRAPH.value,
            "PROFESSIONAL_EXPERIENCE_SUBGRAPH": WorkflowNodes.PROFESSIONAL_EXPERIENCE_SUBGRAPH.value,
            "PROJECTS_SUBGRAPH": WorkflowNodes.PROJECTS_SUBGRAPH.value,
            "EXECUTIVE_SUMMARY_SUBGRAPH": WorkflowNodes.EXECUTIVE_SUMMARY_SUBGRAPH.value,
        }
        
        # Return the mapped node or the original next_node
        mapped_node = section_to_subgraph.get(next_node, next_node)
        logger.info(f"Supervisor routing to: {mapped_node}")
        return mapped_node
        
    except Exception as exc:
        logger.error(f"Error in supervisor routing: {exc}")
        return WorkflowNodes.ERROR_HANDLER.value


def route_from_entry(state: GlobalState) -> str:
    """Route from entry point based on available data.
    
    Args:
        state: Current workflow state
        
    Returns:
        Next node identifier
    """
    try:
        entry_route = state.get("entry_route")
        
        if not entry_route:
            logger.warning("No entry_route specified in entry router state")
            return WorkflowNodes.JD_PARSER.value  # Default to JD parser
        
        logger.info(f"Entry routing to: {entry_route}")
        return entry_route
        
    except Exception as exc:
        logger.error(f"Error in entry routing: {exc}")
        return WorkflowNodes.JD_PARSER.value  # Default fallback