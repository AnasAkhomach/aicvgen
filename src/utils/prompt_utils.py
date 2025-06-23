import os
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger("prompt_utils")


def load_prompt_template(path: str, fallback: Optional[str] = None) -> str:
    """Load a prompt template from file, with optional fallback."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logger.warning(f"Failed to load prompt template from {path}: {e}")
        if fallback is not None:
            return fallback
        raise


def format_prompt(template: str, **kwargs) -> str:
    """Format a prompt template with provided keyword arguments."""
    try:
        return template.format(**kwargs)
    except Exception as e:
        logger.error(f"Prompt formatting failed: {e}")
        raise


def ensure_company_name(job_desc_data):
    """
    Ensures job_desc_data has a non-empty company_name. Returns a (possibly updated) copy.
    """
    if not getattr(job_desc_data, "company_name", None):
        # Use model_copy if available (Pydantic v2), else fallback
        if hasattr(job_desc_data, "model_copy"):
            return job_desc_data.model_copy(update={"company_name": "Unknown Company"})
        elif hasattr(job_desc_data, "copy"):
            return job_desc_data.copy(update={"company_name": "Unknown Company"})
        else:
            job_desc_data.company_name = "Unknown Company"
    return job_desc_data
