"""Facade for managing content templates."""

from typing import Any, Dict, List, Optional

from src.config.logging_config import get_structured_logger
from src.error_handling.exceptions import TemplateError
from src.templates.content_templates import ContentTemplateManager


class CVTemplateManagerFacade:
    """Provides a simplified interface for template management operations."""

    def __init__(self, template_manager: Optional[ContentTemplateManager]):
        self._template_manager = template_manager
        self.logger = get_structured_logger(__name__)

    def get_template(
        self, template_id: str, category: str = None
    ) -> Optional[Dict[str, Any]]:
        """Get a content template."""
        if not self._template_manager:
            self.logger.warning(
                "Template manager not initialized. Cannot get template."
            )
            return None

        try:
            return self._template_manager.get_template(template_id, category)
        except (TemplateError, KeyError, ValueError, IOError) as e:
            self.logger.error(
                "Failed to get template",
                extra={
                    "template_id": template_id,
                    "category": category,
                    "error": str(e),
                },
            )
            return None

    def format_template(
        self, template_id: str, variables: Dict[str, Any], category: str = None
    ) -> Optional[str]:
        """Format a template with variables."""
        if not self._template_manager:
            self.logger.warning(
                "Template manager not initialized. Cannot format template."
            )
            return None

        try:
            template = self._template_manager.get_template(template_id, category)
            if template:
                return self._template_manager.format_template(template, variables)
            return None
        except (TemplateError, KeyError, ValueError, IOError) as e:
            self.logger.error(
                "Failed to format template",
                extra={
                    "template_id": template_id,
                    "category": category,
                    "error": str(e),
                },
            )
            return None

    def list_templates(self, category: str = None) -> List[str]:
        """List available templates."""
        if not self._template_manager:
            self.logger.warning(
                "Template manager not initialized. Cannot list templates."
            )
            return []

        try:
            all_templates = self._template_manager.list_templates()
            if category is None:
                result = []
                for template_list in all_templates.values():
                    result.extend(template_list)
                return result
            else:
                result = []
                for key, template_list in all_templates.items():
                    if category in key:
                        result.extend(template_list)
                return result
        except (TemplateError, IOError) as e:
            self.logger.error(
                "Failed to list templates",
                extra={"category": category, "error": str(e)},
            )
            return []
