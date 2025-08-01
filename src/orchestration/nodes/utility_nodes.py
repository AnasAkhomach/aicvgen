"""Utility nodes for formatting and error handling in the CV workflow."""

import logging
from typing import Dict, Any

from src.orchestration.state import GlobalState
from src.agents.formatter_agent import FormatterAgent

logger = logging.getLogger(__name__)


async def formatter_node(state: GlobalState, *, agent: "FormatterAgent") -> GlobalState:
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
            template_config=state.get("template_config", {}),
        )

        # Update state with formatted output
        updated_state = {
            **state,
            "pdf_output": pdf_output,
            "workflow_status": "COMPLETED",
            "last_executed_node": "FORMATTER",
        }

        logger.info("Formatter node completed successfully")
        return updated_state

    except Exception as exc:
        logger.error(f"Error in Formatter node: {exc}")
        error_messages = (
            list(state["error_messages"]) if state["error_messages"] else []
        )
        error_messages.append(f"PDF formatting failed: {str(exc)}")

        return {
            **state,
            "error_messages": error_messages,
            "workflow_status": "ERROR",
            "last_executed_node": "FORMATTER",
        }


async def error_handler_node(state: GlobalState) -> GlobalState:
    """Handle errors and provide recovery actions using ErrorRecoveryService.

    CB-004 Fix: Uses current_content_type dynamically, falls back to QUALIFICATION.

    Args:
        state: Current workflow state

    Returns:
        Updated state with error handling
    """
    from src.services.error_recovery import ErrorRecoveryService
    from src.models.workflow_models import ContentType

    logger.info("Starting Error Handler node")

    try:
        error_messages = state.get("error_messages", [])

        # If no error messages, return state unchanged
        if not error_messages:
            return state

        # CB-004 Fix: Use current_content_type, fallback to QUALIFICATION
        current_content_type = state.get("current_content_type")
        if current_content_type is None:
            current_content_type = ContentType.QUALIFICATION

        # Get required parameters for ErrorRecoveryService
        current_item_id = state.get("current_item_id")
        session_id = state.get("session_id")
        trace_id = state.get("trace_id")

        # Initialize ErrorRecoveryService and handle error
        error_service = ErrorRecoveryService(logger=logger)

        # Create an exception object from the error message
        error_message = error_messages[0] if error_messages else "Unknown error"
        error_exception = Exception(error_message)

        recovery_action = await error_service.handle_error(
            error_exception, current_item_id, current_content_type, session_id
        )

        # Clear error messages after handling
        updated_state = {
            **state,
            "error_messages": [],
            "recovery_action": recovery_action.strategy.value
            if recovery_action
            else "terminate",
            "last_executed_node": "ERROR_HANDLER",
        }

        logger.info(
            f"Error handling completed. Recovery action: {recovery_action.strategy.value if recovery_action else 'terminate'}"
        )
        return updated_state

    except Exception as exc:
        logger.error(f"Error in Error Handler node: {exc}")
        # Even error handler failed - create minimal error state
        return {
            **state,
            "workflow_status": "CRITICAL_ERROR",
            "error_messages": [f"Error handler failed: {str(exc)}"],
            "last_executed_node": "ERROR_HANDLER",
        }
