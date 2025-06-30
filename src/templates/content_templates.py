"""Content templates for CV generation with Phase 1 infrastructure integration."""

import os
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
from enum import Enum
from dataclasses import dataclass
from datetime import datetime

from src.models.workflow_models import ContentType
from src.config.logging_config import get_structured_logger
from src.error_handling.exceptions import TemplateError

logger = get_structured_logger("content_templates")


class TemplateCategory(Enum):
    """Categories of content templates."""

    PROMPT = "prompt"
    FORMAT = "format"
    VALIDATION = "validation"
    FALLBACK = "fallback"


@dataclass
class ContentTemplate:
    """Structured content template."""

    name: str
    category: TemplateCategory
    content_type: ContentType
    template: str
    variables: List[str]
    description: str
    version: str = "1.0"
    created_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


class ContentTemplateManager:
    """Manager for content templates with caching and validation."""

    def __init__(self, prompt_directory: str = "data/prompts"):
        """Initializes the ContentTemplateManager by loading templates from disk."""
        self.templates: Dict[str, ContentTemplate] = {}
        self._prompt_directory = Path(prompt_directory)
        self._load_templates_from_directory()

        logger.info(
            "ContentTemplateManager initialized",
            template_count=len(self.templates),
            source_directory=str(self._prompt_directory.resolve()),
        )

    def _load_templates_from_directory(self):
        """Scan the prompts directory and load all templates into the cache."""
        if not self._prompt_directory.is_dir():
            logger.warning(
                "Prompt directory not found, skipping template loading.",
                directory=str(self._prompt_directory.resolve()),
            )
            return

        for file_path in self._prompt_directory.glob("**/*.md"):
            if file_path.is_file():
                try:
                    self._load_template_file(file_path)
                except (IOError, OSError) as e:
                    logger.error(
                        "Failed to load template file %s: %s",
                        file_path,
                        e,
                        exc_info=True,
                    )

    def _parse_frontmatter(self, content: str) -> tuple[Dict[str, Any], str]:
        """Parses YAML-like frontmatter from a template file."""
        frontmatter = {}
        body = content
        # Regex to capture frontmatter block and the rest of the content
        match = re.match(r"---\s*\n(.*?)\n---\s*\n(.*)", content, re.DOTALL)
        if match:
            frontmatter_str, body = match.groups()
            # Simple key-value parsing
            for line in frontmatter_str.split("\n"):
                if ":" in line:
                    key, value = line.split(":", 1)
                    frontmatter[key.strip()] = value.strip()
        return frontmatter, body

    def _extract_variables(self, content: str) -> List[str]:
        """Extracts variable names from a template string (e.g., {variable})."""
        return re.findall(r"\{(\w+)\}", content)

    def _load_template_file(self, file_path: Path):
        """Load a single template file and add it to the cache."""
        content = file_path.read_text(encoding="utf-8")
        metadata, template_content = self._parse_frontmatter(content)

        if not metadata:
            logger.warning(
                "No frontmatter found in %s. Skipping.", extra={"file": file_path.name}
            )
            return

        try:
            # Prioritize frontmatter 'name', fall back to file stem
            template_name = metadata.get("name", file_path.stem)

            category_str = metadata.get("category", "prompt").upper()
            content_type_str = metadata.get("content_type", "UNDEFINED").upper()

            template = ContentTemplate(
                name=template_name,
                category=TemplateCategory[category_str],
                content_type=ContentType[content_type_str],
                template=template_content.strip(),
                variables=self._extract_variables(template_content),
                description=metadata.get("description", "No description provided."),
                version=metadata.get("version", "1.0"),
            )
            self.templates[template.name] = template
            logger.info(
                f"Successfully loaded template: '{template.name}' from {file_path.name}"
            )

        except (KeyError, ValueError) as e:
            logger.error(
                "Failed to create template from %s",
                file_path.name,
                exc_info=True,
                extra={"metadata": metadata},
            )

    def register_template(self, template: ContentTemplate):
        """Register a new template."""
        template_key = (
            f"{template.content_type.value}_{template.category.value}_{template.name}"
        )
        self.templates[template_key] = template

        logger.debug(
            "Template registered",
            template_name=template.name,
            content_type=template.content_type.value,
            category=template.category.value,
        )

    def get_template(
        self,
        name: str,
        content_type: ContentType,
        category: TemplateCategory = TemplateCategory.PROMPT,
    ) -> Optional[ContentTemplate]:
        """Get a specific template."""
        template_key = f"{content_type.value}_{category.value}_{name}"
        return self.templates.get(template_key)

    def get_templates_by_type(
        self, content_type: ContentType, category: Optional[TemplateCategory] = None
    ) -> List[ContentTemplate]:
        """Get all templates for a specific content type."""
        templates = []
        for template in self.templates.values():
            if template.content_type == content_type:
                if category is None or template.category == category:
                    templates.append(template)
        return templates

    def format_template(
        self, template: ContentTemplate, variables: Dict[str, Any]
    ) -> str:
        """Format a template with provided variables."""
        try:
            # Ensure all required variables are provided
            missing_vars = [var for var in template.variables if var not in variables]
            if missing_vars:
                logger.warning(
                    "Missing template variables",
                    template_name=template.name,
                    missing_variables=missing_vars,
                )
                # Provide default values for missing variables
                for var in missing_vars:
                    variables[var] = f"[{var}]"

            # Format the template
            formatted_content = template.template.format(**variables)

            logger.debug(
                "Template formatted successfully: %s",
                template.name,
                extra={"content_length": len(formatted_content)},
            )

            return formatted_content

        except (KeyError, IndexError) as e:
            logger.error(
                "Template formatting failed for %s due to missing key: %s",
                template.name,
                e,
                exc_info=True,
            )
            raise TemplateError(
                f"Formatting failed for template '{template.name}' due to missing key: {e}"
            ) from e

    def get_fallback_content(self, content_type: ContentType) -> str:
        """Get fallback content for a content type."""
        template = self.get_template(
            f"fallback_{content_type.value}", content_type, TemplateCategory.FALLBACK
        )

        if template:
            return template.template

        # Ultimate fallback
        return "Professional experience and qualifications relevant to the position."

    def list_templates(self) -> Dict[str, List[str]]:
        """List all available templates by category."""
        template_list = {}

        for template in self.templates.values():
            category_key = f"{template.content_type.value}_{template.category.value}"
            if category_key not in template_list:
                template_list[category_key] = []
            template_list[category_key].append(template.name)

        return template_list


# Convenience functions
def get_prompt_template(
    content_type: ContentType, template_name: str = "basic"
) -> Optional[ContentTemplate]:
    """Get a prompt template for a content type."""
    manager = ContentTemplateManager()
    return manager.get_template(
        f"{content_type.value}_{template_name}", content_type, TemplateCategory.PROMPT
    )


def format_content_prompt(
    content_type: ContentType, template_name: str, variables: Dict[str, Any]
) -> str:
    """Format a content prompt with variables."""
    manager = ContentTemplateManager()
    template = manager.get_template(
        f"{content_type.value}_{template_name}", content_type, TemplateCategory.PROMPT
    )

    if template:
        return manager.format_template(template, variables)

    # Fallback to basic template
    basic_template = manager.get_template(
        f"{content_type.value}_basic", content_type, TemplateCategory.PROMPT
    )

    if basic_template:
        return manager.format_template(basic_template, variables)

    return f"Generate professional {content_type.value} content."


def get_fallback_content(content_type: ContentType) -> str:
    """Get fallback content for a content type."""
    manager = ContentTemplateManager()
    return manager.get_fallback_content(content_type)
