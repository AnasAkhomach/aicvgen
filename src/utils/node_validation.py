"""Node output validation utilities for LangGraph workflow.This module provides decorators and utilities to validate that agent nodes
only return valid GlobalState fields, preventing contract breaches.
"""

from functools import wraps
from typing import Any, Dict, Set

from pydantic import ValidationError

from ..config.logging_config import get_structured_logger
from ..orchestration.state import GlobalState

logger = get_structured_logger(__name__)


def validate_node_output(node_func):
    """
    Decorator to validate that node functions only return valid GlobalState fields.

    This decorator ensures that the dictionary returned by a node function
    contains only keys that are valid fields in the GlobalState TypedDict.
    If invalid keys are found, a CRITICAL error is logged.

    Args:
        node_func: The node function to validate

    Returns:
        The wrapped function with output validation
    """

    @wraps(node_func)
    async def wrapper(*args, **kwargs) -> Any:
        """The wrapper function that performs validation."""
        # When decorating a method, args[0] is 'self'. The state is in args[1].
        # For regular functions, it might be in args[0] or kwargs.
        state = None
        if len(args) > 1 and isinstance(args[1], dict):
            state = args[1]
        elif len(args) > 0 and isinstance(args[0], dict):
            state = args[0]
        else:
            state = kwargs.get("state")

        if not state or not isinstance(state, dict):
            logger.error(
                "Node '%s' was called without a valid GlobalState.", node_func.__name__
            )
            raise TypeError(
                f"Node '{node_func.__name__}' must be called with GlobalState as an argument."
            )

        # Execute the original node function, passing all arguments through
        output = await node_func(*args, **kwargs)

        # Accept dict outputs (GlobalState is now a TypedDict)
        if isinstance(output, dict):
            output_dict = output
        else:
            logger.error(
                "Node '%s' returned unsupported type: %s.",
                node_func.__name__,
                type(output),
            )
            raise TypeError(
                f"Node '{node_func.__name__}' must return dict, got {type(output)}"
            )

        # Get valid GlobalState field names from TypedDict annotations
        valid_keys: Set[str] = set(GlobalState.__annotations__.keys())
        returned_keys: Set[str] = set(output_dict.keys())

        # Find any invalid keys returned by the node
        invalid_keys = returned_keys - valid_keys
        if invalid_keys:
            logger.error(
                "Node '%s' returned invalid keys: %s. Valid keys are: %s",
                node_func.__name__,
                invalid_keys,
                valid_keys,
            )
            # In a strict production environment, you might want to raise an exception
            # For now, we will filter them out to prevent state corruption
            for key in invalid_keys:
                del output_dict[key]

        # Return the filtered dict (GlobalState is now a TypedDict)
        return output_dict

    return wrapper


def get_valid_agent_state_fields() -> Set[str]:
    """
    Get the set of valid GlobalState field names.

    Returns:
        Set of valid field names from the GlobalState TypedDict
    """
    return set(GlobalState.__annotations__.keys())


def validate_output_dict(
    output_dict: Dict[str, Any], context: str = "unknown"
) -> Dict[str, Any]:
    """
    Standalone function to validate an output dictionary against GlobalState fields.

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
