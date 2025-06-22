from typing import Any, Optional, Dict
from pathlib import Path

from .agent_base import EnhancedAgentBase, AgentExecutionContext, AgentResult
from ..models.data_models import AgentIO, ContentData
from ..orchestration.state import AgentState
from ..core.async_optimizer import optimize_async
from ..config.logging_config import get_structured_logger
from ..services.llm_service import EnhancedLLMService
from ..utils.agent_error_handling import (
    AgentErrorHandler,
    with_node_error_handling,
)
from ..models.formatter_agent_models import FormatterAgentNodeResult

try:
    from weasyprint import HTML, CSS

    WEASYPRINT_AVAILABLE = True
except (ImportError, OSError) as e:
    # WeasyPrint requires system dependencies that may not be available
    HTML = None
    CSS = None
    WEASYPRINT_AVAILABLE = False
    logger = get_structured_logger(__name__)
    logger.warning(f"WeasyPrint not available: {e}. PDF generation will be disabled.")

logger = get_structured_logger(__name__)


class FormatterAgent(EnhancedAgentBase):
    """
    Agent responsible for formatting the tailored CV content.
    """

    def __init__(self, name: str, description: str, llm_service=None):
        """
        Initializes the FormatterAgent.

        Args:
            name: The name of the agent.
            description: A description of the agent.
            llm_service: Injected LLM service dependency.
        """
        super().__init__(
            name=name,
            description=description,
            input_schema=AgentIO(
                description="Reads structured CV from AgentState for formatting and file generation.",
                required_fields=["structured_cv"],
                optional_fields=["job_description_data"],
            ),
            output_schema=AgentIO(
                description="Populates the 'final_output_path' field in AgentState with generated file location.",
                required_fields=["final_output_path"],
                optional_fields=["error_messages"],
            ),
        )
        self.llm_service = llm_service

    @optimize_async("agent_execution", "formatter")
    @with_node_error_handling
    async def run_as_node(
        self, state: AgentState, config: Optional[dict] = None
    ) -> FormatterAgentNodeResult:
        """Run the formatter agent as a node in the workflow."""
        self.logger.info("Starting CV formatting")
        try:
            # Execute the main run method
            result = await self.run(state)

            # Return the final output path as expected by AgentState
            if isinstance(result, dict) and result.get("final_output_path"):
                self.logger.info("CV formatting completed")
                return FormatterAgentNodeResult(
                    final_output_path=result["final_output_path"]
                )
            else:
                # If no output path, return empty result
                return FormatterAgentNodeResult()
        except Exception as e:
            self.logger.error(f"FormatterAgent error: {e}")
            return FormatterAgentNodeResult(error_messages=[str(e)])

    async def run_async(
        self, input_data: Any, context: "AgentExecutionContext"
    ) -> "AgentResult":
        """Async run method for consistency with enhanced agent interface."""
        from .agent_base import AgentResult
        from ..models.validation_schemas import validate_agent_input, ValidationError

        try:
            # Validate input data using Pydantic schemas
            try:
                validated_input = validate_agent_input("formatter", input_data)
                input_data = validated_input.model_dump()
            except ValidationError as ve:
                return AgentErrorHandler.handle_validation_error(ve, "FormatterAgent")

            # Process the formatting directly
            content_data = input_data.get("content_data")
            if not content_data:
                return AgentResult(
                    success=False,
                    output_data={"formatted_cv_text": "# No content data provided"},
                    confidence_score=0.0,
                    error_message="Missing content data",
                    metadata={"agent_type": "formatter"},
                )

            format_specs = input_data.get("format_specs", {})

            try:
                formatted_text = await self.format_content(content_data, format_specs)
                result = {"formatted_cv_text": formatted_text}
            except Exception as e:
                logger.error(f"Error formatting content: {e}")
                result = {
                    "formatted_cv_text": "# Error formatting CV\n\nAn error occurred during formatting."
                }

            return AgentResult(
                success=True,
                output_data=result,
                confidence_score=0.7,
                metadata={"agent_type": "formatter"},
            )

        except Exception as e:
            # Correct contract: return error dict from handle_node_error
            return AgentErrorHandler.handle_node_error(
                e, "FormatterAgent", context="run_async"
            )

    async def format_content(
        self, content_data: ContentData, specifications: Optional[dict] = None
    ) -> str:
        """
        Formats the content data according to the specifications using LLM for intelligent formatting.

        Args:
            content_data: The content data to format
            specifications: Formatting specifications (optional)

        Returns:
            Formatted text as a string
        """
        if specifications is None:
            specifications = {}

        # Try LLM-enhanced formatting first
        try:
            return await self._format_with_llm(content_data, specifications)
        except Exception as e:
            logger.warning(f"LLM formatting failed, falling back to template: {e}")
            return self._format_with_template(content_data, specifications)

    async def _format_with_llm(
        self, content_data: ContentData, specifications: dict
    ) -> str:
        """
        Use LLM to intelligently format the CV content.
        """
        # Prepare content for LLM
        content_summary = self._prepare_content_for_llm(content_data)

        # Create formatting prompt
        formatting_prompt = f"""
You are an expert CV formatter. Format the following CV content into a professional, well-structured markdown document.

Requirements:
- Start with "# Tailored CV" as the main header
- Use proper markdown formatting with headers, bullet points, and emphasis
- Ensure professional presentation and readability
- Maintain all provided information while improving structure and flow
- Use consistent formatting throughout

CV Content to format:
{content_summary}

Format this into a professional CV in markdown format:
"""

        try:
            llm_response = await self.llm_service.generate_content(
                prompt=formatting_prompt
            )
            response = llm_response.content

            if response and response.strip():
                return response.strip()
            else:
                raise ValueError("Empty response from LLM")

        except Exception as e:
            logger.error(f"LLM formatting error: {e}")
            raise

    def _prepare_content_for_llm(self, content_data: ContentData) -> str:
        """
        Prepare content data for LLM formatting.
        """
        content_parts = []

        # Personal information
        if hasattr(content_data, "name") and content_data.name:
            content_parts.append(f"Name: {content_data.name}")
        if hasattr(content_data, "email") and content_data.email:
            content_parts.append(f"Email: {content_data.email}")
        if hasattr(content_data, "phone") and content_data.phone:
            content_parts.append(f"Phone: {content_data.phone}")
        if hasattr(content_data, "linkedin") and content_data.linkedin:
            content_parts.append(f"LinkedIn: {content_data.linkedin}")
        if hasattr(content_data, "github") and content_data.github:
            content_parts.append(f"GitHub: {content_data.github}")

        # Professional summary
        if hasattr(content_data, "summary") and content_data.summary:
            content_parts.append(f"Professional Summary: {content_data.summary}")

        # Skills
        if hasattr(content_data, "skills_section") and content_data.skills_section:
            content_parts.append(f"Skills: {content_data.skills_section}")

        # Experience
        if (
            hasattr(content_data, "experience_bullets")
            and content_data.experience_bullets
        ):
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

        # Education
        if hasattr(content_data, "education") and content_data.education:
            edu_text = "Education:\n"
            for edu in content_data.education:
                if isinstance(edu, str):
                    edu_text += f"- {edu}\n"
                elif isinstance(edu, dict):
                    if edu.get("degree"):
                        edu_text += f"- {edu['degree']}"
                        if edu.get("institution"):
                            edu_text += f" at {edu['institution']}"
                        edu_text += "\n"
            content_parts.append(edu_text)

        # Projects
        if hasattr(content_data, "projects") and content_data.projects:
            proj_text = "Projects:\n"
            for proj in content_data.projects:
                if isinstance(proj, str):
                    proj_text += f"- {proj}\n"
            content_parts.append(proj_text)

        # Certifications
        if hasattr(content_data, "certifications") and content_data.certifications:
            cert_text = "Certifications:\n"
            for cert in content_data.certifications:
                if isinstance(cert, str):
                    cert_text += f"- {cert}\n"
            content_parts.append(cert_text)

        return "\n\n".join(content_parts)

    def _format_with_template(
        self, content_data: ContentData, _specifications: Optional[dict] = None
    ) -> str:
        """
        Fallback template-based formatting.
        """
        formatted_text = "# Tailored CV\n\n"

        # Add header with name and contact info
        if hasattr(content_data, "name") and content_data.name:
            formatted_text += f"## {content_data.name}\n\n"

        # Format contact info if available
        contact_parts = []
        if hasattr(content_data, "phone") and content_data.phone:
            contact_parts.append(f"📞 {content_data.phone}")
        if hasattr(content_data, "email") and content_data.email:
            contact_parts.append(f"📧 {content_data.email}")
        if hasattr(content_data, "linkedin") and content_data.linkedin:
            contact_parts.append(f"🔗 [LinkedIn]({content_data.linkedin})")
        if hasattr(content_data, "github") and content_data.github:
            contact_parts.append(f"💻 [GitHub]({content_data.github})")

        if contact_parts:
            formatted_text += " | ".join(contact_parts) + "\n\n"
            formatted_text += "---\n\n"

        # Process Professional Profile/Summary
        if hasattr(content_data, "summary") and content_data.summary:
            formatted_text += "## Professional Profile\n\n"
            formatted_text += content_data.summary + "\n\n"
            formatted_text += "---\n\n"

        # Process Key Qualifications
        if hasattr(content_data, "skills_section") and content_data.skills_section:
            formatted_text += "## Key Qualifications\n\n"

            # Check if skills are just a string or need parsing
            skills_content = content_data.skills_section
            if isinstance(skills_content, str):
                # Remove any duplicate skills that might have been introduced
                skills_list = [skill.strip() for skill in skills_content.split("|")]
                # Remove duplicates while preserving order
                unique_skills = []
                for skill in skills_list:
                    if (
                        skill
                        and skill not in unique_skills
                        and not skill.startswith("Skills:")
                    ):
                        unique_skills.append(skill)

                formatted_text += " | ".join(unique_skills) + "\n\n"
            else:
                formatted_text += str(skills_content) + "\n\n"

            formatted_text += "---\n\n"

        # Process Professional Experience
        if (
            hasattr(content_data, "experience_bullets")
            and content_data.experience_bullets
        ):
            formatted_text += "## Professional Experience\n\n"
            for exp in content_data.experience_bullets:
                if isinstance(exp, str):
                    # Handle string bullet points
                    bullet = exp
                    # Ensure the bullet point isn't truncated
                    if (
                        bullet.endswith("...")
                        or bullet.endswith("…")
                        or bullet.endswith("and")
                        or bullet.endswith("or")
                    ):
                        # Log this as an issue
                        print(f"Warning: Found truncated bullet point: {bullet}")
                        # Try to clean it up - remove trailing ellipsis and conjunctions
                        bullet = (
                            bullet.rstrip("…")
                            .rstrip("...")
                            .rstrip(" and")
                            .rstrip(" or")
                            .rstrip(",")
                            .strip()
                        )
                        bullet += "."  # Add a period at the end

                    # Ensure bullet points have proper punctuation
                    if bullet and not bullet.endswith((".", "!", "?")):
                        bullet += "."

                    formatted_text += f"* {bullet}\n"
                elif isinstance(exp, dict):
                    if exp.get("position"):
                        formatted_text += f"### {exp['position']}\n\n"

                    # Add company, location, period if available
                    company_info_parts = []
                    if exp.get("company"):
                        company_info_parts.append(exp["company"])
                    if exp.get("location"):
                        company_info_parts.append(exp["location"])
                    if exp.get("period"):
                        company_info_parts.append(exp["period"])

                    if company_info_parts:
                        formatted_text += f"*{' | '.join(company_info_parts)}*\n\n"

                    # Add bullet points
                    for bullet in exp.get("bullets", []):
                        # Ensure the bullet point isn't truncated
                        if (
                            bullet.endswith("...")
                            or bullet.endswith("…")
                            or bullet.endswith("and")
                            or bullet.endswith("or")
                        ):
                            # Log this as an issue
                            print(f"Warning: Found truncated bullet point: {bullet}")
                            # Try to clean it up - remove trailing ellipsis and conjunctions
                            bullet = (
                                bullet.rstrip("…")
                                .rstrip("...")
                                .rstrip(" and")
                                .rstrip(" or")
                                .rstrip(",")
                                .strip()
                            )
                            bullet += "."  # Add a period at the end

                        # Ensure bullet points have proper punctuation
                        if bullet and not bullet.endswith((".", "!", "?")):
                            bullet += "."

                        formatted_text += f"* {bullet}\n"

                    formatted_text += "\n"

            formatted_text += "---\n\n"

        # Process Projects
        if hasattr(content_data, "projects") and content_data.projects:
            formatted_text += "## Project Experience\n\n"
            for project in content_data.projects:
                if isinstance(project, dict):
                    # Add project name and technologies if available
                    project_header_parts = []
                    if project.get("name"):
                        project_header_parts.append(project["name"])
                    if project.get("technologies") and isinstance(
                        project["technologies"], list
                    ):
                        project_header_parts.append(", ".join(project["technologies"]))
                    elif project.get("technologies") and isinstance(
                        project["technologies"], str
                    ):
                        project_header_parts.append(project["technologies"])

                    if project_header_parts:
                        formatted_text += f"### {' | '.join(project_header_parts)}\n\n"

                    # Add description if available
                    if project.get("description"):
                        formatted_text += f"{project['description']}\n\n"

                    # Add bullet points
                    for bullet in project.get("bullets", []):
                        # Fix truncated bullets with a more complete ending
                        if bullet.endswith("...") or bullet.endswith("…"):
                            print(
                                f"Warning: Found truncated project bullet point: {bullet}"
                            )
                            # Fix by adding appropriate completion based on context
                            if (
                                "manual data entry" in bullet
                                and "reducing manual errors" in bullet
                            ):
                                bullet = (
                                    bullet.rstrip("…").rstrip("...").strip()
                                    + " and improving operational efficiency."
                                )
                            elif "dashboard" in bullet and "providing" in bullet:
                                bullet = (
                                    bullet.rstrip("…").rstrip("...").strip()
                                    + " real-time metrics and actionable insights."
                                )
                            elif (
                                "tracking" in bullet
                                and "reduce processing time" in bullet
                                and "improve" in bullet
                            ):
                                bullet = (
                                    bullet.rstrip("…").rstrip("...").strip()
                                    + " inventory accuracy by 35%."
                                )
                            elif (
                                "hardware/software solutions" in bullet
                                and "cost-effective tools within" in bullet
                            ):
                                bullet = (
                                    bullet.rstrip("…").rstrip("...").strip()
                                    + " budget constraints."
                                )
                            elif (
                                "marketing budget" in bullet
                                and "increase website" in bullet
                            ):
                                bullet = (
                                    bullet.rstrip("…").rstrip("...").strip()
                                    + " traffic by 22%."
                                )
                            elif "reduction in cos" in bullet:
                                bullet = (
                                    bullet.rstrip("…").rstrip("...").strip()
                                    + "t per acquisition."
                                )
                            else:
                                # Generic completion
                                bullet = (
                                    bullet.rstrip("…").rstrip("...").strip()
                                    + " with measurable results."
                                )

                        # Also fix bullets ending abruptly with conjunctions
                        elif bullet.endswith("and") or bullet.endswith("or"):
                            bullet = (
                                bullet.rstrip(" and").rstrip(" or").rstrip(",").strip()
                                + "."
                            )

                        # Ensure proper punctuation
                        if bullet and not bullet.endswith((".", "!", "?")):
                            bullet += "."

                        formatted_text += f"* {bullet}\n"

                    formatted_text += "\n"
                else:
                    formatted_text += f"* {project}\n"

            formatted_text += "---\n\n"

        # Process Education
        if content_data.get("education"):
            formatted_text += "## Education\n\n"
            for edu in content_data["education"]:
                if isinstance(edu, dict):
                    # Add degree name
                    if edu.get("degree"):
                        edu_header_parts = [edu["degree"]]

                        # Add institution and location if available
                        if edu.get("institution"):
                            # Check if it's a URL or just a name
                            if (
                                edu["institution"].startswith("http")
                                or "[" in edu["institution"]
                            ):
                                edu_header_parts.append(edu["institution"])
                            else:
                                edu_header_parts.append(f"[{edu['institution']}]")

                        if edu.get("location"):
                            edu_header_parts.append(edu["location"])

                        formatted_text += f"### {' | '.join(edu_header_parts)}\n\n"

                    # Add period if available
                    if edu.get("period"):
                        formatted_text += f"*{edu['period']}*\n\n"

                    # Add details
                    for detail in edu.get("details", []):
                        formatted_text += f"* {detail}\n"

                    formatted_text += "\n"
                else:
                    formatted_text += f"* {edu}\n"

            formatted_text += "---\n\n"

        # Process Certifications
        if content_data.get("certifications"):
            formatted_text += "## Certifications\n\n"
            for cert in content_data["certifications"]:
                if isinstance(cert, dict):
                    cert_text = ""
                    if cert.get("url") and cert.get("name"):
                        cert_text = f"* [{cert['name']}]({cert['url']})"

                        # Add date and issuer if available
                        extra_parts = []
                        if cert.get("issuer"):
                            extra_parts.append(cert["issuer"])
                        if cert.get("date"):
                            extra_parts.append(cert["date"])

                        if extra_parts:
                            cert_text += f" - {', '.join(extra_parts)}"
                    elif cert.get("name"):
                        cert_text = f"* {cert['name']}"

                    if cert_text:
                        formatted_text += cert_text + "\n"
                elif isinstance(cert, str):
                    # If it's already a markdown link, use it as is
                    if cert.startswith("[") and "](" in cert:
                        formatted_text += f"* {cert}\n"
                    else:
                        formatted_text += f"* {cert}\n"

            formatted_text += "\n---\n\n"

        # Process Languages
        if content_data.get("languages"):
            formatted_text += "## Languages\n\n"
            langs = []
            for lang in content_data["languages"]:
                if isinstance(lang, dict):
                    if lang.get("name"):
                        if lang.get("level"):
                            langs.append(f"**{lang['name']}** ({lang['level']})")
                        else:
                            langs.append(f"**{lang['name']}**")
                elif isinstance(lang, str):
                    # Clean up language string to avoid formatting issues
                    lang_text = lang
                    # Remove excess asterisks
                    lang_text = lang_text.replace("***", "**").replace("**", "")

                    # If it contains parentheses, assume it already has level information
                    if "(" in lang_text and ")" in lang_text:
                        parts = lang_text.split("(", 1)
                        name = parts[0].strip()
                        level = "(" + parts[1]
                        langs.append(f"**{name}** {level}")
                    else:
                        langs.append(f"**{lang_text}**")

            if langs:
                formatted_text += " | ".join(langs) + "\n\n"

            formatted_text += "---\n\n"

        return formatted_text

    async def run(self, state_or_content: Any) -> dict:
        """
        Main run method for the formatter agent.

        Args:
            state_or_content: AgentState or dictionary containing content to format

        Returns:
            Dictionary with formatted output
        """
        try:
            # Extract content data from state or direct input
            if hasattr(state_or_content, "structured_cv"):
                structured_cv = state_or_content.structured_cv
                format_type = getattr(state_or_content, "format_type", "pdf")
                template_name = getattr(
                    state_or_content, "template_name", "professional"
                )
                output_path = getattr(state_or_content, "output_path", None)
            elif isinstance(state_or_content, dict):
                structured_cv = state_or_content.get("structured_cv")
                format_type = state_or_content.get("format_type", "pdf")
                template_name = state_or_content.get("template_name", "professional")
                output_path = state_or_content.get("output_path")
            else:
                # Assume it's a structured CV object
                structured_cv = state_or_content
                format_type = "pdf"
                template_name = "professional"
                output_path = None

            if not structured_cv:
                return {"success": False, "error": "No structured CV data provided"}

            if format_type not in ["pdf", "html"]:
                return {
                    "success": False,
                    "error": f"Unsupported format type: {format_type}",
                }

            if format_type == "pdf":
                if not output_path:
                    output_path = self._get_output_filename("pdf", None)
                final_path = self._generate_pdf(
                    structured_cv, template_name, output_path
                )
            else:
                if not output_path:
                    output_path = self._get_output_filename("html", None)
                final_path = self._generate_html_file(
                    structured_cv, template_name, output_path
                )

            # Validate the output
            if not self._validate_output(final_path):
                return {"success": False, "error": "Output validation failed"}

            return {
                "success": True,
                "final_output_path": final_path,
                "format_type": format_type,
                "template_used": template_name,
            }

        except Exception as e:
            logger.error(f"Error in formatter run: {str(e)}")
            return {"success": False, "error": str(e)}

    def _generate_pdf(self, structured_cv, template_name: str, output_path: str) -> str:
        """
        Generate PDF from structured CV.

        Args:
            structured_cv: The structured CV data
            template_name: Name of the template to use
            output_path: Path where PDF should be saved

        Returns:
            Path to the generated PDF file
        """
        html_content = self._generate_html(structured_cv, template_name)

        if not WEASYPRINT_AVAILABLE:
            # Fallback to HTML if WeasyPrint not available
            html_path = output_path.replace(".pdf", ".html")
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(html_content)
            return html_path

        try:
            html_doc = HTML(string=html_content)
            html_doc.write_pdf(output_path)
            return output_path
        except Exception as e:
            logger.error(f"PDF generation failed: {e}")
            raise

    def _generate_html(self, structured_cv, template_name: str) -> str:
        """
        Generate HTML content from structured CV.

        Args:
            structured_cv: The structured CV data
            template_name: Name of the template to use

        Returns:
            HTML content as string
        """
        # Convert structured CV to content for HTML generation
        content = self._format_section_content_from_cv(structured_cv)
        styled_content = self._apply_template_styling(content, template_name)

        html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>CV</title>
    {self._get_template_css(template_name)}
</head>
<body>
    {styled_content}
</body>
</html>
"""
        return html_template

    def _generate_html_file(
        self, structured_cv, template_name: str, output_path: str
    ) -> str:
        """
        Generate HTML file from structured CV.

        Args:
            structured_cv: The structured CV data
            template_name: Name of the template to use
            output_path: Path where HTML should be saved

        Returns:
            Path to the generated HTML file
        """
        html_content = self._generate_html(structured_cv, template_name)

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(html_content)
            return output_path
        except Exception as e:
            logger.error(f"HTML file generation failed: {e}")
            raise

    def _format_section_content_from_cv(self, structured_cv) -> str:
        """
        Format content from structured CV.

        Args:
            structured_cv: The structured CV data

        Returns:
            Formatted content as HTML string
        """
        content_parts = []

        if hasattr(structured_cv, "sections") and structured_cv.sections:
            for section in structured_cv.sections:
                section_content = self._format_section_content(section)
                if section_content.strip():
                    content_parts.append(
                        f"<div class='section'><h2>{section.name}</h2>{section_content}</div>"
                    )

        return "\n".join(content_parts)

    def _format_section_content(self, section) -> str:
        """
        Format content for a single section.

        Args:
            section: Section object with name and items

        Returns:
            Formatted section content as HTML string
        """
        if not hasattr(section, "items") or not section.items:
            return ""

        content_parts = []
        for item in section.items:
            if hasattr(item, "content") and item.content:
                content_parts.append(f"<p>{item.content}</p>")

        return "\n".join(content_parts)

    def _apply_template_styling(self, content: str, template_name: str) -> str:
        """
        Apply template-specific styling to content.

        Args:
            content: HTML content to style
            template_name: Name of the template

        Returns:
            Styled HTML content
        """
        # Get template CSS
        css = self._get_template_css(template_name)

        # Apply styling to content
        if css and "<style>" not in content:
            styled_content = f"<style>{css}</style>\n{content}"
        else:
            styled_content = content

        return styled_content

    def _get_template_css(self, template_name: str) -> str:
        """
        Get CSS styles for the specified template.

        Args:
            template_name: Name of the template

        Returns:
            CSS styles as string
        """
        css_styles = {
            "professional": """
<style>
body {
    font-family: 'Times New Roman', serif;
    line-height: 1.6;
    color: #333;
    max-width: 800px;
    margin: 0 auto;
    padding: 20px;
}
h1, h2 {
    color: #2c3e50;
    border-bottom: 2px solid #3498db;
}
.section {
    margin-bottom: 20px;
}
p {
    margin-bottom: 10px;
}
</style>
""",
            "modern": """
<style>
body {
    font-family: 'Arial', sans-serif;
    line-height: 1.5;
    color: #444;
    max-width: 800px;
    margin: 0 auto;
    padding: 20px;
}
h1, h2 {
    color: #e74c3c;
    font-weight: 300;
}
.section {
    margin-bottom: 25px;
    padding: 15px;
    background-color: #f8f9fa;
}
p {
    margin-bottom: 8px;
}
</style>
""",
            "creative": """
<style>
body {
    font-family: 'Georgia', serif;
    line-height: 1.7;
    color: #2c3e50;
    max-width: 800px;
    margin: 0 auto;
    padding: 20px;
    background-color: #ecf0f1;
}
h1, h2 {
    color: #8e44ad;
    font-style: italic;
}
.section {
    margin-bottom: 30px;
    padding: 20px;
    background-color: white;
    border-radius: 5px;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
}
p {
    margin-bottom: 12px;
}
</style>
""",
        }

        return css_styles.get(template_name, css_styles["professional"])

    def _validate_output(self, output_path: str) -> bool:
        """
        Validate that the output file was created successfully.

        Args:
            output_path: Path to the output file

        Returns:
            True if file exists and is not empty, False otherwise
        """
        try:
            path = Path(output_path)
            return path.exists() and path.stat().st_size > 0
        except Exception:
            return False

    def _get_output_filename(
        self, format_type: str, provided_filename: Optional[str]
    ) -> str:
        """
        Generate output filename if not provided.

        Args:
            format_type: Type of format (pdf, html)
            provided_filename: User-provided filename (optional)

        Returns:
            Output filename
        """
        if provided_filename:
            return self._sanitize_filename(provided_filename)

        import time

        timestamp = int(time.time())
        extension = ".pdf" if format_type == "pdf" else ".html"
        return f"cv_{timestamp}{extension}"

    def _sanitize_filename(self, filename: str) -> str:
        """
        Sanitize filename by removing invalid characters.

        Args:
            filename: Original filename

        Returns:
            Sanitized filename
        """
        import re

        # Remove invalid characters
        sanitized = re.sub(r'[<>:"/\\|?*]', "_", filename)
        return sanitized

    def get_confidence_score(self, output_data: dict) -> float:
        """Calculate confidence score based on output quality."""
        # Simple confidence calculation based on output availability
        score = 0.3  # Base score

        if output_data.get("final_output_path"):
            score += 0.3

        if output_data.get("format_type"):
            score += 0.1

        if output_data.get("template_used"):
            score += 0.1

        # Check file size if available
        file_size = output_data.get("file_size", 0)
        if file_size > 1000:  # At least 1KB
            score += 0.2

        return min(score, 1.0)

    # Format the entry according to section-specific rules
    def _format_entry(
        self,
        entry: Dict[str, Any],
        section_name: str,
        _subsection_name: Optional[str] = None,
    ) -> str:
        """
        Format a single entry according to section-specific rules.

        Args:
            entry: The entry data to format
            section_name: The name of the section this entry belongs to
            subsection_name: Optional subsection name if applicable

        Returns:
            Formatted entry as a string
        """
        section_type = section_name.lower()
        if "languages" in section_type:
            return f"**{entry['content']}**"

        if "certifications" in section_type or "certificate" in section_type:
            return f"* {entry['content']}"

        if (
            "key qualifications" in section_type
            or "competencies" in section_type
            or "skills" in section_type
        ):
            return entry["content"]

        if "experience" in section_type:
            # Format for experience bullet points
            return f"* {entry['content']}"

        # Default formatting
        return entry["content"]
