"""
This module defines the FormatterAgent, responsible for formatting CVs.
"""

from pathlib import Path
from typing import Any, Dict, Optional

from jinja2 import Environment, FileSystemLoader, select_autoescape, TemplateError
from pydantic import BaseModel

from ..config.logging_config import get_structured_logger
from ..models.data_models import AgentIO, ContentData, StructuredCV
from ..models.formatter_agent_models import FormatterAgentNodeResult
from ..models.validation_schemas import validate_agent_input
from ..orchestration.state import AgentState

from ..services.llm_service import EnhancedLLMService
from ..services.progress_tracker import ProgressTracker
from ..templates.content_templates import ContentTemplateManager
from ..utils.agent_error_handling import with_node_error_handling
from ..utils.error_utils import handle_errors
from ..utils.prompt_utils import format_prompt
from ..utils.exceptions import AgentExecutionError
from .agent_base import AgentResult, EnhancedAgentBase, AgentExecutionContext

try:
    from weasyprint import HTML

    WEASYPRINT_AVAILABLE = True
except (ImportError, OSError) as e:
    HTML = None
    WEASYPRINT_AVAILABLE = False
    logger = get_structured_logger(__name__)
    logger.warning("WeasyPrint not available: %s. PDF generation will be disabled.", e)

logger = get_structured_logger(__name__)


class FormatterAgentOutput(BaseModel):
    """Output model for the FormatterAgent."""

    formatted_cv_text: str


class FormatterAgent(EnhancedAgentBase):
    """
    Agent responsible for formatting the tailored CV content.
    """

    # pylint: disable=too-many-arguments, too-many-positional-arguments
    def __init__(
        self,
        llm_service: EnhancedLLMService,
        progress_tracker: ProgressTracker,
        template_manager: ContentTemplateManager,
        name: str = "FormatterAgent",
        description: str = "Formats CV content and generates files.",
    ):
        """Initialize the FormatterAgent with required dependencies."""
        input_schema = AgentIO(
            description="Reads structured CV from AgentState for formatting.",
            required_fields=["structured_cv"],
        )
        output_schema = AgentIO(
            description="Populates 'final_output_path' in AgentState.",
            required_fields=["final_output_path"],
        )
        super().__init__(
            name=name,
            description=description,
            input_schema=input_schema,
            output_schema=output_schema,
            progress_tracker=progress_tracker,
        )
        self.llm_service = llm_service
        self.template_manager = template_manager
        self.jinja_env: Optional[Environment] = None
        self._init_template_environment()

    def _init_template_environment(self):
        """Initialize the Jinja2 template environment."""
        try:
            template_dir = Path(__file__).parent.parent / "templates"
            self.jinja_env = Environment(
                loader=FileSystemLoader(template_dir),
                autoescape=select_autoescape(["html", "xml"]),
                trim_blocks=True,
                lstrip_blocks=True,
            )
            logger.info("Jinja2 template environment initialized: %s", template_dir)
        except IOError as e:
            logger.error("Failed to initialize Jinja2 environment: %s", e)
            self.jinja_env = None

    @with_node_error_handling("formatter")
    async def run_as_node(
        self, state: AgentState, _config: Optional[dict] = None
    ) -> Dict[str, Any]:
        """Run formatter agent as a node, returning an update dictionary."""
        logger.info("Starting CV formatting")
        try:
            validate_agent_input("formatter", state)

            result = await self.run(state)
            if result.error_message:
                raise RuntimeError(result.error_message)
            if result.final_output_path:
                logger.info("CV formatting completed")
                return {"final_output_path": result.final_output_path}
            return {}
        except Exception as e:
            logger.error(f"FormatterAgent failed: {e}", exc_info=True)
            from ..utils.exceptions import AgentExecutionError

            raise AgentExecutionError(
                agent_name="FormatterAgent", message=str(e)
            ) from e

    async def run_async(
        self, input_data: Any, context: "AgentExecutionContext"
    ) -> "AgentResult":
        """Asynchronously formats content based on input data."""
        try:
            validated_input = validate_agent_input("formatter", input_data)
            content_data = validated_input.content_data
            if not content_data:
                return AgentResult(
                    success=False,
                    output_data=FormatterAgentOutput(),
                    error_message="Missing content data",
                )

            format_specs = validated_input.get("format_specs", {})
            formatted_text = await self.format_content(content_data, format_specs)
            result = FormatterAgentOutput(formatted_cv_text=formatted_text)
            return AgentResult(success=True, output_data=result, confidence_score=1.0)
        except (ValueError, TypeError) as e:
            logger.error("Error in run_async: %s", e)
            return AgentResult(
                success=False,
                output_data=FormatterAgentOutput(),
                error_message=f"Formatting failed: {e}",
            )

    async def format_content(
        self, content_data: ContentData, specifications: Optional[dict] = None
    ) -> str:
        """Formats content using LLM or a template fallback."""
        specifications = specifications or {}
        try:
            return await self._format_with_llm(content_data, specifications)
        except AgentExecutionError as e:
            logger.warning("LLM formatting failed, falling back to template: %s", e)
            return self._format_with_template(content_data)

    @handle_errors(default_return="")
    async def _format_with_llm(
        self, content_data: ContentData, _specifications: dict
    ) -> str:
        """Use LLM to intelligently format the CV content."""
        content_summary = self._prepare_content_for_llm(content_data)
        prompt = self.template_manager.get_prompt("cv_formatting")
        if not prompt:
            raise ValueError("CV formatting prompt not found.")

        formatted_prompt = format_prompt(prompt, content_summary=content_summary)
        llm_response = await self.llm_service.generate_content(prompt=formatted_prompt)
        response = llm_response.content.strip()
        if not response:
            raise ValueError("Empty response from LLM")
        return response

    def _prepare_content_for_llm(self, content_data: ContentData) -> str:
        """Prepare content data for LLM formatting."""
        content_parts = []

        def add_part(label: str, value: Any):
            if value:
                content_parts.append(f"{label}: {value}")

        add_part("Name", getattr(content_data, "name", None))
        add_part("Email", getattr(content_data, "email", None))
        add_part("Phone", getattr(content_data, "phone", None))
        add_part("LinkedIn", getattr(content_data, "linkedin", None))
        add_part("GitHub", getattr(content_data, "github", None))
        add_part("Professional Summary", getattr(content_data, "summary", None))
        add_part("Skills", getattr(content_data, "skills_section", None))

        # Experience
        if getattr(content_data, "experience_bullets", None):
            exp_text = "Experience:\n"
            for exp in content_data.experience_bullets:
                if isinstance(exp, str):
                    exp_text += f"- {exp}\n"
                elif isinstance(exp, dict):
                    if exp.get("position"):
                        exp_text += f"Position: {exp['position']}\n"
                    for bullet in exp.get("bullets", []):
                        exp_text += f"- {bullet}\n"
            content_parts.append(exp_text)

        return "\n\n".join(content_parts)

    def _format_with_template(self, content_data: ContentData) -> str:
        """Fallback template-based formatting."""
        if not self.jinja_env:
            return "Error: Jinja2 environment not initialized."
        try:
            template = self.jinja_env.get_template("cv_template.md")
            return template.render(cv=content_data)
        except TemplateError as e:
            logger.error("Error rendering template: %s", e)
            return "Error rendering CV from template."

    async def run(self, state_or_content: Any) -> FormatterAgentNodeResult:
        """Main run method for the formatter agent."""
        try:
            params = self._prepare_run_parameters(state_or_content)
            if params.get("error_message"):
                return FormatterAgentNodeResult(
                    final_output_path=None,
                    error_message=params["error_message"],
                )

            structured_cv = params["structured_cv"]
            format_type = params["format_type"]
            template_name = params["template_name"]
            output_path = params["output_path"]

            if format_type == "pdf":
                final_path = self._generate_pdf(
                    structured_cv, template_name, output_path
                )
            else:
                final_path = self._generate_html_file(
                    structured_cv, template_name, output_path
                )
            return FormatterAgentNodeResult(final_output_path=final_path)
        except AgentExecutionError as e:
            logger.error("FormatterAgent.run error: %s", e, exc_info=True)
            return FormatterAgentNodeResult(
                final_output_path=None, error_message=str(e)
            )

    def _prepare_run_parameters(self, state_or_content: Any) -> Dict[str, Any]:
        """Extracts and validates run parameters from input."""
        if isinstance(state_or_content, AgentState):
            structured_cv = state_or_content.structured_cv
            format_type = getattr(state_or_content, "format_type", "pdf")
            template_name = getattr(state_or_content, "template_name", "professional")
            output_path = getattr(state_or_content, "output_path", None)
        elif isinstance(state_or_content, dict):
            structured_cv = state_or_content.get("structured_cv")
            format_type = state_or_content.get("format_type", "pdf")
            template_name = state_or_content.get("template_name", "professional")
            output_path = state_or_content.get("output_path")
        else:
            msg = f"Unsupported input type: {type(state_or_content)}"
            logger.error(msg)
            return {"error_message": msg}

        if not structured_cv:
            return {"error_message": "StructuredCV is missing."}

        return {
            "structured_cv": structured_cv,
            "format_type": format_type,
            "template_name": template_name,
            "output_path": output_path,
        }

    def _generate_pdf(
        self,
        structured_cv: StructuredCV,
        template_name: str,
        output_path: Optional[str],
    ) -> Optional[str]:
        """Generates a PDF from the structured CV."""
        if not WEASYPRINT_AVAILABLE or not self.jinja_env:
            logger.error("PDF generation is not available.")
            return None
        try:
            html_template = self.jinja_env.get_template(f"{template_name}.html")
            html_content = html_template.render(cv=structured_cv)

            final_path = output_path or "cv.pdf"
            HTML(string=html_content).write_pdf(final_path)
            logger.info("Successfully generated PDF: %s", final_path)
            return final_path
        except (TemplateError, IOError) as e:
            logger.error("Failed to generate PDF: %s", e)
            return None

    def _generate_html_file(
        self,
        structured_cv: StructuredCV,
        template_name: str,
        output_path: Optional[str],
    ) -> Optional[str]:
        """Generates an HTML file from the structured CV."""
        if not self.jinja_env:
            logger.error("HTML generation is not available.")
            return None
        try:
            template = self.jinja_env.get_template(f"{template_name}.html")
            html_content = template.render(cv=structured_cv)

            final_path = output_path or "cv.html"
            with open(final_path, "w", encoding="utf-8") as f:
                f.write(html_content)
            logger.info("Successfully generated HTML file: %s", final_path)
            return final_path
        except (TemplateError, IOError) as e:
            logger.error("Failed to generate HTML file: %s", e)
            return None
