"""Utility nodes for formatting and error handling in the CV workflow."""

import logging
from typing import Dict, Any

from src.orchestration.state import GlobalState
from src.agents.formatter_agent import FormatterAgent

logger = logging.getLogger(__name__)


async def formatter_node(state: GlobalState, *, agent: 'FormatterAgent') -> GlobalState:
    """Format the CV into final output (PDF generation).
    
    Args:
        state: Current workflow state
        agent: Formatter agent
        
    Returns:
        Updated state with formatted output
    """
    logger.info("Starting Formatter node")
    
    try:
        # Agent is now explicitly provided via dependency injection
        
        # Generate PDF from structured CV
        pdf_output = await agent.generate_pdf(
            structured_cv=state["structured_cv"],
            template_config=state.get("template_config", {})
        )
        
        # Update state with formatted output
        updated_state = {
            **state,
            "pdf_output": pdf_output,
            "workflow_status": "COMPLETED",
            "last_executed_node": "FORMATTER"
        }
        
        logger.info("Formatter node completed successfully")
        return updated_state
        
    except Exception as exc:
        logger.error(f"Error in Formatter node: {exc}")
        error_messages = list(state["error_messages"]) if state["error_messages"] else []
        error_messages.append(f"PDF formatting failed: {str(exc)}")
        
        return {
            **state,
            "error_messages": error_messages,
            "workflow_status": "ERROR",
            "last_executed_node": "FORMATTER"
        }


async def error_handler_node(state: GlobalState) -> GlobalState:
    """Handle errors and provide recovery actions.
    
    Args:
        state: Current workflow state
        
    Returns:
        Updated state with error handling
    """
    logger.info("Starting Error Handler node")
    
    try:
        error_messages = state.get("error_messages", [])
        last_executed_node = state.get("last_executed_node", "UNKNOWN")
        
        # Log all error messages
        for error_msg in error_messages:
            logger.error(f"Workflow error: {error_msg}")
        
        # Determine recovery action based on error context
        recovery_action = "TERMINATE"  # Default action
        
        # Analyze errors for potential recovery
        if any("timeout" in str(error).lower() for error in error_messages):
            recovery_action = "RETRY"
            logger.info("Timeout detected, suggesting retry")
        elif any("network" in str(error).lower() for error in error_messages):
            recovery_action = "RETRY"
            logger.info("Network error detected, suggesting retry")
        elif any("validation" in str(error).lower() for error in error_messages):
            recovery_action = "REGENERATE"
            logger.info("Validation error detected, suggesting regeneration")
        
        # Prepare error summary for UI
        error_summary = {
            "total_errors": len(error_messages),
            "last_failed_node": last_executed_node,
            "recovery_suggestion": recovery_action,
            "error_details": error_messages[-3:] if error_messages else []  # Last 3 errors
        }
        
        updated_state = {
            **state,
            "workflow_status": "ERROR",
            "error_summary": error_summary,
            "recovery_action": recovery_action,
            "last_executed_node": "ERROR_HANDLER"
        }
        
        logger.info(f"Error handling completed. Recovery action: {recovery_action}")
        return updated_state
        
    except Exception as exc:
        logger.error(f"Error in Error Handler node: {exc}")
        # Even error handler failed - create minimal error state
        return {
            **state,
            "workflow_status": "CRITICAL_ERROR",
            "error_messages": [f"Error handler failed: {str(exc)}"],
            "last_executed_node": "ERROR_HANDLER"
        }