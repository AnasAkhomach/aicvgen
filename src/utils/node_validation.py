"""Node output validation utilities for LangGraph workflow.This module provides decorators and utilities to validate that agent nodes
only return valid AgentState fields, preventing contract breaches.
"""

from functools import wraps
from typing import Dict, Any, Set
from ..orchestration.state import AgentState
from ..config.logging_config import get_structured_logger

logger = get_structured_logger(__name__)


def validate_node_output(node_func):
    """
    Decorator to validate that node functions only return valid AgentState fields.

    This decorator ensures that the dictionary returned by a node function
    contains only keys that are valid fields in the AgentState Pydantic model.
    If invalid keys are found, a CRITICAL error is logged.

    Args:
        node_func: The node function to validate

    Returns:
        The wrapped function with output validation
    """
    @wraps(node_func)
    async def wrapper(state: AgentState) -> Dict[str, Any]:
        # Execute the original node function
        output_dict = await node_func(state)

        # Get valid AgentState field names
        valid_keys: Set[str] = set(AgentState.model_fields.keys())
        returned_keys: Set[str] = set(output_dict.keys())

        # Check for invalid keys
        invalid_keys = returned_keys - valid_keys
        if invalid_keys:
            logger.error(
                f"Node '{node_func.__name__}' returned invalid keys not in AgentState: {list(invalid_keys)}. "
                f"Valid keys are: {list(valid_keys)}"
            )
            # Filter out invalid keys to prevent data loss
            output_dict = {k: v for k, v in output_dict.items() if k in valid_keys}
            logger.warning(
                f"Filtered out invalid keys from node '{node_func.__name__}' output. "
                f"Remaining valid keys: {list(output_dict.keys())}"
            )

        # Log successful validation for debugging
        logger.debug(
            f"Node '{node_func.__name__}' output validation passed. Returned keys: {list(returned_keys)}"
        )

        return output_dict

    return wrapper


def get_valid_agent_state_fields() -> Set[str]:
    """
    Get the set of valid AgentState field names.

    Returns:
        Set of valid field names from the AgentState model
    """
    return set(AgentState.model_fields.keys())


def validate_output_dict(output_dict: Dict[str, Any], context: str = "unknown") -> Dict[str, Any]:
    """
    Standalone function to validate an output dictionary against AgentState fields.

    Args:
        output_dict: Dictionary to validate
        context: Context string for logging (e.g., function name)

    Returns:
        Validated dictionary with invalid keys filtered out
    """
    valid_keys = get_valid_agent_state_fields()
    returned_keys = set(output_dict.keys())

    invalid_keys = returned_keys - valid_keys
    if invalid_keys:
        logger.error(
            f"Invalid keys found in {context}: {list(invalid_keys)}. Valid keys are: {list(valid_keys)}"
        )
        # Filter out invalid keys
        output_dict = {k: v for k, v in output_dict.items() if k in valid_keys}
        logger.warning(
            f"Filtered out invalid keys from {context}. Remaining keys: {list(output_dict.keys())}"
        )

    return output_dict