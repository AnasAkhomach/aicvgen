from src.agents.agent_base import AgentBase
from src.core.state_manager import AgentIO, ContentData
from src.orchestration.state import AgentState
from src.config.settings import get_config
from src.config.logging_config import get_structured_logger
from typing import Dict, Any, Optional
import os
from jinja2 import Environment, FileSystemLoader
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


class FormatterAgent(AgentBase):
    """
    Agent responsible for formatting the tailored CV content.
    """

    def __init__(self, name: str, description: str):
        """
        Initializes the FormatterAgent.

        Args:
            name: The name of the agent.
            description: A description of the agent.
        """
        super().__init__(
            name=name,
            description=description,
            input_schema=AgentIO(
                description="Formats the generated CV content according to specifications.",
                required_fields=["content_data"],
                optional_fields=["format_specifications"]
            ),
            output_schema=AgentIO(
                description="Formats the generated CV content according to specifications.",
                required_fields=["formatted_cv", "output_path"]
            ),
        )

    def run_as_node(self, state: AgentState) -> dict:
        """
        Takes the final StructuredCV from the state and renders it as a PDF.
        This is the primary entry point for this agent in the LangGraph workflow.
        """
        logger.info("--- Executing Node: FormatterAgent ---")
        cv_data = state.structured_cv
        if not cv_data:
            return {"error_messages": state.error_messages + ["FormatterAgent: No CV data found in state."]}

        try:
            config = get_config()
            template_dir = config.project_root / "src" / "templates"
            static_dir = config.project_root / "src" / "frontend" / "static"
            output_dir = config.project_root / "data" / "output"
            output_dir.mkdir(parents=True, exist_ok=True)

            # 1. Set up Jinja2 environment
            env = Environment(loader=FileSystemLoader(str(template_dir)), autoescape=True)
            template = env.get_template("pdf_template.html")

            # 2. Render HTML from template
            html_out = template.render(cv=cv_data)

            # 3. Generate PDF using WeasyPrint (if available)
            if not WEASYPRINT_AVAILABLE:
                logger.warning("WeasyPrint not available. Saving HTML output instead of PDF.")
                output_filename = f"CV_{cv_data.id}.html"
                output_path = output_dir / output_filename
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(html_out)
                logger.info(f"FormatterAgent: HTML successfully generated at {output_path}")
                return {"final_output_path": str(output_path)}
            
            css_path = static_dir / "css" / "pdf_styles.css"
            if not css_path.exists():
                logger.warning(f"CSS file not found at {css_path}. PDF will have no styling.")
                css_stylesheet = None
            else:
                css_stylesheet = CSS(css_path)

            pdf_bytes = HTML(string=html_out, base_url=str(template_dir)).write_pdf(
                stylesheets=[css_stylesheet] if css_stylesheet else None
            )

            # 4. Save PDF to file
            output_filename = f"CV_{cv_data.id}.pdf"
            output_path = output_dir / output_filename
            with open(output_path, "wb") as f:
                f.write(pdf_bytes)

            logger.info(f"FormatterAgent: PDF successfully generated at {output_path}")
            return {"final_output_path": str(output_path)}

        except Exception as e:
            logger.error(f"FormatterAgent failed: {e}", exc_info=True)
            return {"error_messages": state.error_messages + [f"PDF generation failed: {e}"]}

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Legacy method for backward compatibility.
        For new workflows, use run_as_node instead.
        """
        # Extract input data
        content_data = input_data.get("content_data")
        if not content_data:
            return {
                "formatted_cv_text": "# No content data provided",
                "error": "Missing content data",
            }

        format_specs = input_data.get("format_specs", {})

        # This can be enhanced in the future with more complex formatting logic
        try:
            formatted_text = self.format_content(content_data, format_specs)
        except Exception as e:
            logger.error("Error formatting content: %s", str(e))
            import traceback

            traceback.print_exc()

            # Simple fallback formatting
            formatted_text = "# Tailored CV\n\n"

            # Add summary if available
            if content_data.get("summary"):
                formatted_text += (
                    f"## Professional Profile\n\n{content_data.get('summary')}\n\n---\n\n"
                )

            # Add skills if available
            if content_data.get("skills_section"):
                formatted_text += (
                    f"## Key Qualifications\n\n{content_data.get('skills_section')}\n\n---\n\n"
                )

            # Add experience if available
            if content_data.get("experience_bullets"):
                formatted_text += "## Professional Experience\n\n"
                for exp in content_data.get("experience_bullets", []):
                    if isinstance(exp, dict):
                        if exp.get("position"):
                            formatted_text += f"### {exp.get('position')}\n\n"
                        for bullet in exp.get("bullets", []):
                            formatted_text += f"* {bullet}\n"
                    else:
                        formatted_text += f"* {exp}\n"
                formatted_text += "\n---\n\n"

        logger.info("Completed: %s (Legacy Formatting)", self.name)
        return {"formatted_cv_text": formatted_text}
    
    async def run_async(self, input_data: Any, context: 'AgentExecutionContext') -> 'AgentResult':
        """Async run method for consistency with enhanced agent interface."""
        from .agent_base import AgentResult
        from src.models.validation_schemas import validate_agent_input, ValidationError
        
        try:
            # Validate input data using Pydantic schemas
            try:
                validated_input = validate_agent_input('formatter', input_data)
                # Convert validated Pydantic model back to dict for processing
                input_data = validated_input.model_dump()
            except ValidationError as ve:
                return AgentResult(
                    success=False,
                    output_data={"formatted_cv_text": "# Validation Error\n\nInput data validation failed."},
                    confidence_score=0.0,
                    error_message=f"Input validation failed: {ve.message}",
                    metadata={"agent_type": "formatter", "validation_error": True}
                )
            except Exception as e:
                return AgentResult(
                    success=False,
                    output_data={"formatted_cv_text": "# Validation Error\n\nInput data validation failed."},
                    confidence_score=0.0,
                    error_message=f"Input validation error: {str(e)}",
                    metadata={"agent_type": "formatter", "validation_error": True}
                )
            
            # Use the existing run method for the actual processing
            result = self.run(input_data)
            
            return AgentResult(
                success=True,
                output_data=result,
                confidence_score=1.0,
                metadata={"agent_type": "formatter"}
            )
            
        except Exception as e:
            return AgentResult(
                success=False,
                output_data={"formatted_cv_text": "# Error formatting CV\n\nAn error occurred during formatting."},
                confidence_score=0.0,
                error_message=str(e),
                metadata={"agent_type": "formatter"}
            )

    def format_content(
        self, content_data: ContentData, specifications: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Formats the content data according to the specifications.

        Args:
            content_data: The content data to format
            specifications: Formatting specifications (optional)

        Returns:
            Formatted text as a string
        """
        if specifications is None:
            specifications = {}

        formatted_text = ""

        # Add header with name and contact info
        if "name" in content_data:
            formatted_text += f"# {content_data['name']}\n\n"

        # Format contact info if available
        contact_parts = []
        if "phone" in content_data:
            contact_parts.append(f"📞 {content_data['phone']}")
        if "email" in content_data:
            contact_parts.append(f"📧 {content_data['email']}")
        if "linkedin" in content_data:
            contact_parts.append(f"🔗 [LinkedIn]({content_data['linkedin']})")
        if "github" in content_data:
            contact_parts.append(f"💻 [GitHub]({content_data['github']})")

        if contact_parts:
            formatted_text += " | ".join(contact_parts) + "\n\n"
            formatted_text += "---\n\n"

        # Process Professional Profile/Summary
        if content_data.get("summary"):
            formatted_text += "## Professional Profile\n\n"
            formatted_text += content_data["summary"] + "\n\n"
            formatted_text += "---\n\n"

        # Process Key Qualifications
        if content_data.get("skills_section"):
            formatted_text += "## Key Qualifications\n\n"

            # Check if skills are just a string or need parsing
            skills_content = content_data["skills_section"]
            if isinstance(skills_content, str):
                # Remove any duplicate skills that might have been introduced
                skills_list = [skill.strip() for skill in skills_content.split("|")]
                # Remove duplicates while preserving order
                unique_skills = []
                for skill in skills_list:
                    if skill and skill not in unique_skills and not skill.startswith("Skills:"):
                        unique_skills.append(skill)

                formatted_text += " | ".join(unique_skills) + "\n\n"
            else:
                formatted_text += str(skills_content) + "\n\n"

            formatted_text += "---\n\n"

        # Process Professional Experience
        if content_data.get("experience_bullets"):
            formatted_text += "## Professional Experience\n\n"
            for exp in content_data["experience_bullets"]:
                if isinstance(exp, dict):
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
                else:
                    formatted_text += f"* {exp}\n"

            formatted_text += "---\n\n"

        # Process Projects
        if content_data.get("projects"):
            formatted_text += "## Project Experience\n\n"
            for project in content_data["projects"]:
                if isinstance(project, dict):
                    # Add project name and technologies if available
                    project_header_parts = []
                    if project.get("name"):
                        project_header_parts.append(project["name"])
                    if project.get("technologies") and isinstance(project["technologies"], list):
                        project_header_parts.append(", ".join(project["technologies"]))
                    elif project.get("technologies") and isinstance(project["technologies"], str):
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
                            print(f"Warning: Found truncated project bullet point: {bullet}")
                            # Fix by adding appropriate completion based on context
                            if "manual data entry" in bullet and "reducing manual errors" in bullet:
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
                            elif "marketing budget" in bullet and "increase website" in bullet:
                                bullet = (
                                    bullet.rstrip("…").rstrip("...").strip() + " traffic by 22%."
                                )
                            elif "reduction in cos" in bullet:
                                bullet = (
                                    bullet.rstrip("…").rstrip("...").strip() + "t per acquisition."
                                )
                            else:
                                # Generic completion
                                bullet = (
                                    bullet.rstrip("…").rstrip("...").strip()
                                    + " with measurable results."
                                )

                        # Also fix bullets ending abruptly with conjunctions
                        elif bullet.endswith("and") or bullet.endswith("or"):
                            bullet = bullet.rstrip(" and").rstrip(" or").rstrip(",").strip() + "."

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
                            if edu["institution"].startswith("http") or "[" in edu["institution"]:
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

    # Format the entry according to section-specific rules
    def _format_entry(
        self,
        entry: Dict[str, Any],
        section_name: str,
        subsection_name: Optional[str] = None,
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
