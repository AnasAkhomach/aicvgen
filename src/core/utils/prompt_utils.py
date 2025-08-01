import logging
from typing import Optional

from src.error_handling.exceptions import TemplateFormattingError

logger = logging.getLogger("prompt_utils")


def load_prompt_template(path: str, fallback: Optional[str] = None) -> str:
    """Load a prompt template from file, with optional fallback."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except (IOError, FileNotFoundError) as e:
        logger.warning("Failed to load prompt template", path=path, error=str(e))
        if fallback is not None:
            return fallback
        raise


def format_prompt(template: str, **kwargs) -> str:
    """Format a prompt template with provided keyword arguments."""
    try:
        return template.format(**kwargs)
    except (KeyError, IndexError) as e:
        logger.error("Prompt formatting failed due to missing key", error=str(e))
        raise TemplateFormattingError(f"Missing key in prompt template: {e}") from e


def ensure_company_name(job_desc_data):
    """
    Ensures job_desc_data has a non-empty company_name. Updates the object directly.
    """
    if not getattr(job_desc_data, "company_name", None):
        job_desc_data.company_name = "Unknown Company"
    return job_desc_data
