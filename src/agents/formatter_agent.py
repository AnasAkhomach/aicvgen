"""
This module defines the FormatterAgent, responsible for formatting CVs into files.
"""

from pathlib import Path
from typing import Any, Optional

from ..models.agent_models import AgentResult
from src.models.agent_output_models import FormatterAgentOutput
from ..config.logging_config import get_structured_logger
from ..error_handling.exceptions import (
    AgentExecutionError,
    DependencyError,
    TemplateError,
)
from ..models.data_models import StructuredCV
from ..templates.content_templates import ContentTemplateManager
from .agent_base import AgentBase

try:
    from weasyprint import HTML

    WEASYPRINT_AVAILABLE = True
except (ImportError, OSError) as e:
    HTML = None
    WEASYPRINT_AVAILABLE = False
    # Log the error once at import time
    _logger = get_structured_logger(__name__)
    _logger.warning(
        "WeasyPrint not found. PDF generation will be disabled.",
        error=str(e),
        exc_info=False,  # Don't need a full stack trace here
    )


logger = get_structured_logger(__name__)


class FormatterAgent(AgentBase):
    """
    Agent responsible for formatting the tailored CV content into a file (PDF or HTML).
    """

    def __init__(
        self,
        template_manager: ContentTemplateManager,
        settings: dict,
        session_id: str,
    ):
        """Initialize the FormatterAgent with required dependencies."""
        super().__init__(
            name="FormatterAgent",
            description="Formats the tailored CV content into a file.",
            session_id=session_id,
        )
        self.template_manager = template_manager
        self.settings = settings

    def _validate_inputs(self, input_data: dict) -> None:
        """Validates the input for the FormatterAgent."""
        if not isinstance(input_data, dict):
            raise AgentExecutionError("Input validation failed: input_data must be a dict")
        if "structured_cv" not in input_data:
            raise AgentExecutionError("'structured_cv' of type StructuredCV is required.")
        if not isinstance(input_data["structured_cv"], StructuredCV):
            raise AgentExecutionError("'structured_cv' must be of type StructuredCV.")
        format_type = input_data.get("format_type", "pdf").lower()
        if format_type == "pdf" and not WEASYPRINT_AVAILABLE:
            raise DependencyError(
                "PDF generation requires WeasyPrint, but it is not installed or failed to load."
            )

    async def _execute(self, **kwargs: Any) -> AgentResult[FormatterAgentOutput]:
        """Formats a StructuredCV into a file and returns the path."""
        input_data = kwargs.get("input_data", {})
        structured_cv: Optional[StructuredCV] = input_data.get("structured_cv")
        format_type: str = input_data.get("format_type", "pdf")
        template_name: str = input_data.get("template_name", "default_template.html")
        output_path: Optional[str] = input_data.get("output_path")

        self.update_progress(0, "Starting formatting process")
        format_type = format_type.lower()

        try:
            self.update_progress(30, "Rendering CV from template")
            html_content = self._format_html(structured_cv, template_name)

            final_output_path = self._get_output_path(output_path, format_type)
            final_output_path.parent.mkdir(parents=True, exist_ok=True)

            if format_type == "pdf":
                self.update_progress(70, "Generating PDF file")
                self._generate_pdf(html_content, final_output_path)
            else:  # html
                self.update_progress(70, "Generating HTML file")
                self._generate_html(html_content, final_output_path)

            output_data = FormatterAgentOutput(
                output_path=str(final_output_path.resolve())
            )
            self.update_progress(100, "Formatting completed successfully")
            return AgentResult(
                status="success",
                agent_name=self.name,
                output_data=output_data,
                message="CV formatting completed successfully.",
            )
        except AgentExecutionError:
            # Re-raise without modification to preserve original error context
            raise
        except DependencyError:
            # Re-raise without modification
            raise
        except Exception as e:
            logger.error(
                "An unexpected error occurred during formatting.",
                error=str(e),
                exc_info=True,
            )
            raise AgentExecutionError(
                agent_name=self.name,
                message=f"An unexpected error occurred: {e}",
            ) from e

    def _format_html(self, structured_cv: StructuredCV, template_name: str) -> str:
        """Renders the CV data into an HTML string using a template."""
        try:
            return self.template_manager.format_template(
                template_name, structured_cv.model_dump()
            )
        except (TemplateError, KeyError) as e:
            logger.error(
                "Failed to render HTML template.",
                template_name=template_name,
                error=str(e),
                exc_info=True,
            )
            raise AgentExecutionError(
                agent_name=self.name,
                message=f"Failed to render HTML template '{template_name}'. Reason: {e}",
            ) from e

    def _get_output_path(
        self, output_path_str: Optional[str], format_type: str
    ) -> Path:
        """Determines the output file path."""
        if output_path_str:
            return Path(output_path_str)

        # Fallback to a default path if none is provided
        output_dir = Path(self.settings.get("output_dir", "instance/output"))
        # Use the session_id to create a unique filename
        filename = f"generated_cv_{self.session_id}.{format_type}"
        return output_dir / filename

    def _generate_pdf(self, html_content: str, output_path: Path):
        """Generates a PDF file from HTML content."""
        try:
            if HTML is None:
                # This check is redundant if the entry check works, but it's good practice
                raise DependencyError("WeasyPrint is not available for PDF generation.")
            HTML(string=html_content).write_pdf(output_path)
        except Exception as e:
            logger.error(
                "Failed to generate PDF.",
                output_path=str(output_path),
                error=str(e),
                exc_info=True,
            )
            raise AgentExecutionError(
                agent_name=self.name, message=f"Failed to write PDF file: {e}"
            ) from e

    def _generate_html(self, html_content: str, output_path: Path):
        """Generates an HTML file from HTML content."""
        try:
            output_path.write_text(html_content, encoding="utf-8")
        except IOError as e:
            logger.error(
                "Failed to generate HTML file.",
                output_path=str(output_path),
                error=str(e),
                exc_info=True,
            )
            raise AgentExecutionError(
                agent_name=self.name, message=f"Failed to write HTML file: {e}"
            ) from e
