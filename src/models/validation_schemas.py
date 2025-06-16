"""Pydantic validation schemas for API requests and responses.

This module provides validation functions and error handling for agent inputs and outputs."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, ValidationError
import logging

logger = logging.getLogger(__name__)


def validate_agent_input(input_data: Any, expected_type: type = None) -> bool:
    """Validate agent input data.
    
    Args:
        input_data: The input data to validate
        expected_type: Optional expected type for validation
        
    Returns:
        bool: True if validation passes, False otherwise
    """
    try:
        if expected_type and not isinstance(input_data, expected_type):
            logger.warning(f"Input type mismatch: expected {expected_type}, got {type(input_data)}")
            return False
        return True
    except Exception as e:
        logger.error(f"Validation error: {e}")
        return False


def validate_agent_output(output_data: Any, required_fields: List[str] = None) -> bool:
    """Validate agent output data.
    
    Args:
        output_data: The output data to validate
        required_fields: Optional list of required fields for dict outputs
        
    Returns:
        bool: True if validation passes, False otherwise
    """
    try:
        if required_fields and isinstance(output_data, dict):
            missing_fields = [field for field in required_fields if field not in output_data]
            if missing_fields:
                logger.warning(f"Missing required fields: {missing_fields}")
                return False
        return True
    except Exception as e:
        logger.error(f"Output validation error: {e}")
        return False