"""Enhanced Content Writer Agent with Phase 1 infrastructure integration."""

from typing import Dict, Any, List, Optional
from datetime import datetime
import json

from .agent_base import EnhancedAgentBase, AgentExecutionContext, AgentResult
from .parser_agent import ParserAgent
from ..services.llm_service import get_llm_service, LLMResponse
from ..models.data_models import (
    ContentType,
    JobDescriptionData,
    StructuredCV,
    Section,
    Subsection,
    Item,
    ItemStatus,
)
from ..config.logging_config import get_structured_logger
from ..config.settings import get_config
from ..core.state_manager import AgentIO
from ..orchestration.state import AgentState
from ..core.async_optimizer import optimize_async
from ..models.validation_schemas import validate_agent_input
from ..utils.exceptions import (
    ValidationError,
)
from ..utils.agent_error_handling import AgentErrorHandler, with_node_error_handling
from ..models.enhanced_content_writer_models import (
    ContentWriterJobData,
    ContentWriterContentItem,
    ContentWriterGenerationContext,
    ContentWriterResult,
)

logger = get_structured_logger("enhanced_content_writer")


class EnhancedContentWriterAgent(EnhancedAgentBase):
    """Enhanced Content Writer Agent with Phase 1 infrastructure integration."""

    def __init__(
        self,
        name: str = "EnhancedContentWriter",
        description: str = "Enhanced agent for generating tailored CV content with advanced error handling and progress tracking",
        content_type: ContentType = ContentType.QUALIFICATION,
    ):
        """Initialize the enhanced content writer agent."""
        super().__init__(
            name=name,
            description=description,
            input_schema=AgentIO(
                description="Reads structured CV, current item ID, and optional research data from AgentState for content generation.",
                required_fields=["structured_cv", "current_item_id"],
                optional_fields=["job_description_data", "research_findings"],
            ),
            output_schema=AgentIO(
                description="Updates the 'structured_cv' field in AgentState with enhanced content.",
                required_fields=["structured_cv"],
                optional_fields=["error_messages"],
            ),
            content_type=content_type,
        )

        # Enhanced services
        self.llm_service = get_llm_service()

        # Initialize settings for prompt loading
        self.settings = get_config()

        # Initialize parser agent for parsing methods
        self.parser_agent = ParserAgent(
            name="ContentWriterParser",
            description="Parser agent for content writer parsing methods",
        )

        # Content generation templates - load from external files
        self.content_templates = {
            ContentType.QUALIFICATION: self._load_prompt_template(
                "key_qualifications_writer"
            ),
            ContentType.EXPERIENCE: self._load_prompt_template("resume_role_writer"),
            ContentType.PROJECT: self._load_prompt_template("project_writer"),
            ContentType.EXECUTIVE_SUMMARY: self._load_prompt_template(
                "executive_summary_writer"
            ),
        }

        logger.info(
            "Enhanced Content Writer Agent initialized",
            agent_name=name,
            supported_content_types=[ct.value for ct in self.content_templates.keys()],
        )

    async def run_async(
        self, input_data: dict, context: AgentExecutionContext
    ) -> AgentResult:
        """Simplified content generation focused on single-item processing.

        This method now exclusively processes one item at a time as specified by current_item_id.
        The complex validation and batch processing logic has been removed for Task 3.3.

        Args:
            input_data: Expected to contain structured_cv, job_description_data, and current_item_id
            context: Agent execution context

        Returns:
            AgentResult with updated structured_cv or error information
        """
        try:
            # Validate input_data using Pydantic model
            job_data = ContentWriterJobData(
                **input_data.get("job_description_data", {})
            )

            # Extract required fields
            structured_cv_data = input_data.get("structured_cv")
            current_item_id = input_data.get("current_item_id")
            research_findings = input_data.get("research_findings")

            # Validate required fields
            if not structured_cv_data:
                return self._create_error_result(
                    input_data,
                    context,
                    ValueError("structured_cv is required"),
                    "validation",
                )

            if not current_item_id:
                return self._create_error_result(
                    input_data,
                    context,
                    ValueError("current_item_id is required"),
                    "validation",
                )

            # At this point, structured_cv_data must be a valid StructuredCV dict or model
            # No parsing of raw CV text is performed here. If the data is not structured, this is a contract error.
            # The parser agent is responsible for all parsing and structuring of raw CV text.

            result = await self._process_single_item(
                structured_cv_data, job_data.dict(), current_item_id, research_findings
            )

            # When returning result:
            result = ContentWriterResult(
                structured_cv=structured_cv_data, error_messages=[]
            )
            return AgentResult(
                success=True,
                output_data=result.dict(),
                confidence_score=1.0,
                metadata={"agent_type": "enhanced_content_writer"},
            )

        except Exception as e:
            return self._create_error_result(input_data, context, e, "run_async")

    async def _generate_content_with_llm(
        self,
        job_data: Dict[str, Any],
        content_item: Dict[str, Any],
        generation_context: Dict[str, Any],
        content_type: ContentType,
        context: AgentExecutionContext,
    ) -> LLMResponse:
        """Generate content using the enhanced LLM service with JSON parsing for role generation."""

        # Build prompt based on content type
        prompt = self._build_prompt(
            job_data, content_item, generation_context, content_type
        )

        # Generate content with LLM service
        response = await self.llm_service.generate_content(
            prompt=prompt,
            content_type=content_type,
            session_id=getattr(context, "session_id", None),
            item_id=getattr(context, "item_id", None),
            max_retries=3,
        )

        # Store raw LLM output in content_item if it's an Item model
        if hasattr(content_item, "raw_llm_output"):
            content_item.raw_llm_output = response.content
        elif isinstance(content_item, dict) and "raw_llm_output" in content_item:
            content_item["raw_llm_output"] = response.content

        # For Experience content type, parse JSON response and validate with Pydantic
        if content_type == ContentType.EXPERIENCE and response.success:
            try:
                # Parse JSON from the response using centralized method
                json_content = await self._generate_and_parse_json(
                    prompt=prompt,
                    session_id=getattr(context, "session_id", None),
                    trace_id=getattr(context, "trace_id", None),
                )

                # Parse and validate with Pydantic model
                from ..models.validation_schemas import LLMRoleGenerationOutput

                validated_data = LLMRoleGenerationOutput.model_validate(json_content)

                # Convert validated data to formatted content
                formatted_content = self._format_role_generation_output(validated_data)

                # Update response content with formatted output
                response.content = formatted_content
                response.metadata["json_validated"] = True
                response.metadata["validation_model"] = "LLMRoleGenerationOutput"

                logger.info(
                    f"Successfully parsed and validated role generation JSON for {content_type}"
                )

            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing failed for role generation: {e}")
                response.metadata["json_parse_error"] = str(e)
                response.metadata["json_validated"] = False
                # Keep original content as fallback

            except ValidationError as e:
                logger.error(f"Pydantic validation failed for role generation: {e}")
                response.metadata["validation_error"] = str(e)
                response.metadata["json_validated"] = False
                # Keep original content as fallback

            except Exception as e:
                logger.error(
                    f"Unexpected error during role generation JSON processing: {e}"
                )
                response.metadata["processing_error"] = str(e)
                response.metadata["json_validated"] = False
                # Keep original content as fallback

        return response

    def _build_prompt(
        self,
        job_data: Dict[str, Any],
        content_item: Dict[str, Any],
        generation_context: Dict[str, Any],
        content_type: ContentType,
    ) -> str:
        """Build a specialized prompt based on content type."""

        # Defensive programming: Ensure job_data is a dictionary
        if not isinstance(job_data, dict):
            logger.error(
                f"Expected job_data to be a dict, but got {type(job_data)}. Input (first 200 chars): {str(job_data)[:200]}"
            )
            # Fallback to an empty dictionary to prevent AttributeError and allow graceful degradation
            job_data = {}

        template = self.content_templates.get(
            content_type, self.content_templates[ContentType.QUALIFICATION]
        )

        # Handle EXPERIENCE content type with resume_role_prompt template
        if content_type == ContentType.EXPERIENCE:
            return self._build_experience_prompt(
                template, job_data, content_item, generation_context
            )

        # Extract relevant information for other content types
        job_title = job_data.get("title", "the position")
        job_description = job_data.get("raw_text", job_data.get("description", ""))
        company_name = job_data.get("company", "the company")

        # Ensure content_item is a dictionary
        if not isinstance(content_item, dict):
            content_item = {}

        # Content-specific information and prompt formatting
        if content_type == ContentType.QUALIFICATION:
            # Handle "Big 10" skills generation with updated template variables
            my_talents = generation_context.get(
                "my_talents",
                job_data.get(
                    "my_talents",
                    "Professional with diverse technical and analytical skills",
                ),
            )

            # Format the prompt with the updated key_qualifications_prompt variables
            prompt = template.format(
                main_job_description_raw=job_description[
                    :2000
                ],  # Increased limit for better analysis
                my_talents=my_talents,
            )
        elif content_type == ContentType.PROJECT:
            item_title = content_item.get("name", "project")
            item_description = content_item.get("description", "")

            # Format the prompt
            prompt = template.format(
                job_title=job_title,
                job_description=job_description[:1000],  # Limit length
                company_name=company_name,
                item_title=item_title,
                item_description=item_description,
                additional_context=generation_context.get("additional_context", ""),
            )
        else:  # EXECUTIVE_SUMMARY
            item_title = "Executive Summary"
            item_description = generation_context.get("cv_summary", "")

            # Format the prompt
            prompt = template.format(
                job_title=job_title,
                job_description=job_description[:1000],  # Limit length
                company_name=company_name,
                item_title=item_title,
                item_description=item_description,
                additional_context=generation_context.get("additional_context", ""),
            )

        return prompt

    def _build_experience_prompt(
        self,
        template: str,
        job_data: Dict[str, Any],
        content_item: Dict[str, Any],
        generation_context: Dict[str, Any],
    ) -> str:
        """Build prompt specifically for experience content using resume_role_prompt template."""

        # Defensive programming: Ensure job_data is a dictionary
        if not isinstance(job_data, dict):
            logger.error(
                f"Expected job_data to be a dict, but got {type(job_data)}. Input (first 200 chars): {str(job_data)[:200]}"
            )
            # Fallback to an empty dictionary to prevent AttributeError and allow graceful degradation
            job_data = {}

        # No parsing of content_item here. It must be structured already.

        # Extract target skills from job data
        target_skills = job_data.get("skills", [])
        if isinstance(target_skills, list):
            target_skills_text = "\n".join([f"- {skill}" for skill in target_skills])
        else:
            target_skills_text = "- No specific skills identified"

        # Build batched structured output for the role
        role_info = self._format_role_info(content_item, generation_context)

        # Format the template with the correct variables
        prompt = template.replace("{{Target Skills}}", target_skills_text)
        prompt = prompt.replace("{{batched_structured_output}}", role_info)

        return prompt

    def _format_role_info(
        self, content_item: Dict[str, Any], generation_context: Dict[str, Any]
    ) -> str:
        """Format role information for the resume template. Assumes content_item is structured."""

        # No parsing of raw CV text here. Only formatting of structured data.
        if not isinstance(content_item, dict):
            logger.error(
                f"Expected dict for content_item, got {type(content_item)}: {content_item}"
            )
            content_item = {"name": "Unknown Role", "items": []}

        # Extract role information
        role_name = content_item.get("name", "Unknown Role")
        company_name = content_item.get("company", "Unknown Company")

        # Extract accomplishments from items
        accomplishments = []
        if "items" in content_item:
            for item in content_item["items"]:
                if isinstance(item, dict) and item.get("item_type") == "bullet_point":
                    accomplishments.append(item.get("content", ""))

        # Format the role info block
        role_info = f"""<role_info_start>
<info>
Role: {role_name}
Organization: {company_name}
Description: {role_name} position with focus on data analysis and technical implementation
</info>
<accomplishments>
"""

        for accomplishment in accomplishments:
            role_info += f"- {accomplishment}\n"

        role_info += "</accomplishments>\n</role_info_end>"

        return role_info

    def _format_single_role(
        self, role_data: Dict[str, Any], generation_context: Dict[str, Any]
    ) -> str:
        """Format a single role for the resume template. Assumes role_data is structured."""

        # No parsing of raw CV text here. Only formatting of structured data.
        if not isinstance(role_data, dict):
            logger.error(
                f"Expected dict for role_data, got {type(role_data)}: {role_data}"
            )
            role_data = {"name": "Unknown Role", "items": []}

        # Extract role information
        role_name = role_data.get("name", "Unknown Role")
        company_name = role_data.get("company", "Unknown Company")

        # Extract accomplishments from items
        accomplishments = []
        if "items" in role_data:
            for item in role_data["items"]:
                if isinstance(item, dict) and item.get("item_type") == "bullet_point":
                    accomplishments.append(item.get("content", ""))

        # Format the role info block
        role_info = f"""<role_info_start>
<info>
Role: {role_name}
Organization: {company_name}
Description: {role_name} position with focus on data analysis and technical implementation
</info>
<accomplishments>
"""

        for accomplishment in accomplishments:
            role_info += f"- {accomplishment}\n"

        role_info += "</accomplishments>\n</role_info_end>"

        return role_info

    async def _post_process_content(
        self,
        llm_response: LLMResponse,
        content_type: ContentType,
        context: AgentExecutionContext,
    ) -> LLMResponse:
        """Post-process generated content for quality and formatting."""

        if not llm_response.success:
            return llm_response

        content = llm_response.content

        # Basic cleaning
        content = content.strip()

        # Content-specific post-processing
        if content_type == ContentType.QUALIFICATION:
            content = self._format_qualification_content(content)
        elif content_type == ContentType.EXPERIENCE:
            content = self._format_experience_content(content)
        elif content_type == ContentType.PROJECT:
            content = self._format_project_content(content)
        elif content_type == ContentType.EXECUTIVE_SUMMARY:
            content = self._format_executive_summary_content(content)

        # Update the response with processed content
        llm_response.content = content
        llm_response.metadata["post_processed"] = True

        return llm_response

    def _calculate_confidence_score(
        self,
        llm_response: LLMResponse,
        job_data: Dict[str, Any],
        content_item: Dict[str, Any],
    ) -> float:
        """Calculate confidence score based on content quality metrics."""

        if not llm_response.success:
            return 0.0

        content = llm_response.content
        base_score = 0.7  # Base confidence

        # Length check
        if 50 <= len(content) <= 500:
            base_score += 0.1

        # Job relevance (simple keyword matching)
        job_keywords = self._extract_keywords(job_data.get("raw_text", ""))
        content_keywords = self._extract_keywords(content.lower())

        if job_keywords and content_keywords:
            overlap = len(job_keywords.intersection(content_keywords))
            relevance_score = min(overlap / len(job_keywords), 0.2)
            base_score += relevance_score

        # Processing time penalty (if too fast, might be low quality)
        if llm_response.processing_time > 2.0:
            base_score += 0.05

        return min(base_score, 1.0)

    def _extract_keywords(self, text: str) -> set:
        """Extract relevant keywords from text."""
        import re

        # Simple keyword extraction
        words = re.findall(r"\b\w{3,}\b", text.lower())

        # Filter out common words
        stop_words = {
            "the",
            "and",
            "for",
            "are",
            "but",
            "not",
            "you",
            "all",
            "can",
            "had",
            "her",
            "was",
            "one",
            "our",
            "out",
            "day",
            "get",
            "has",
            "him",
            "his",
            "how",
            "its",
            "may",
            "new",
            "now",
            "old",
            "see",
            "two",
            "who",
            "boy",
            "did",
            "man",
            "men",
            "put",
            "say",
            "she",
            "too",
            "use",
            "will",
            "with",
        }

        return {word for word in words if word not in stop_words and len(word) > 3}

    def _determine_content_type(self, content_item: Dict[str, Any]) -> ContentType:
        """Determine content type from item data."""

        # Ensure content_item is a dictionary before checking keys
        if not isinstance(content_item, dict):
            logger.warning(
                f"Expected dict for content_item, got {type(content_item)}: {content_item}"
            )
            return ContentType.QUALIFICATION

        if "position" in content_item or "company" in content_item:
            return ContentType.EXPERIENCE
        elif "name" in content_item and "technologies" in content_item:
            return ContentType.PROJECT
        elif "summary" in content_item or "executive" in str(content_item).lower():
            return ContentType.EXECUTIVE_SUMMARY
        else:
            return ContentType.QUALIFICATION

    def _generate_fallback_content(
        self, content_item: Dict[str, Any], content_type: ContentType
    ) -> str:
        """Generate fallback content when LLM fails."""

        # Ensure content_item is a dictionary
        if not isinstance(content_item, dict):
            content_item = {}

        # Safely get values from content_item (now guaranteed to be a dictionary)
        position = content_item.get("position", "previous role")
        project_name = content_item.get("name", "project")

        fallbacks = {
            ContentType.QUALIFICATION: "Strong technical skills and experience relevant to the position.",
            ContentType.EXPERIENCE: f"Valuable experience in {position} contributing to professional growth.",
            ContentType.PROJECT: f"Successful completion of {project_name} demonstrating technical capabilities.",
            ContentType.EXECUTIVE_SUMMARY: "Experienced professional with a strong background in delivering results and contributing to organizational success.",
        }

        return fallbacks.get(
            content_type,
            "Professional experience and qualifications relevant to the position.",
        )

    # Content formatting methods
    def _format_qualification_content(self, content: str) -> str:
        """Format qualification content."""
        # Ensure it starts with a strong statement
        if not content.startswith(("Strong", "Experienced", "Skilled", "Proficient")):
            content = f"Strong {content.lower()}"
        return content

    def _format_experience_content(self, content: str) -> str:
        """Format experience content."""
        # Ensure bullet points or clear structure
        if "•" not in content and "*" not in content and "-" not in content:
            # Add basic structure if missing
            sentences = content.split(". ")
            if len(sentences) > 1:
                content = "\n".join(
                    [
                        f"• {sentence.strip()}"
                        for sentence in sentences
                        if sentence.strip()
                    ]
                )
        return content

    def _build_single_item_prompt(
        self,
        subsection: Subsection,
        section: Section,
        job_description: Optional[JobDescriptionData],
        research_findings: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Build a focused prompt for a single subsection/item."""
        try:
            # Get the appropriate template for the content type
            template = self.content_templates.get(
                section.content_type, self._get_fallback_template()
            )

            # Prepare context variables
            context_vars = {
                "job_title": (
                    job_description.job_title if job_description else "Target Position"
                ),
                "company_name": (
                    job_description.company_name
                    if job_description and job_description.company_name
                    else "Target Company"
                ),
                "job_description": (
                    job_description.main_job_description_raw
                    if job_description
                    else "No job description provided"
                ),
                "item_title": subsection.name if subsection else section.name,
                "item_description": (
                    subsection.items[0].content
                    if subsection and subsection.items
                    else "No existing content"
                ),
                "additional_context": f"This is for the {section.name} section of a CV",
            }

            # Add research findings to context if available
            if research_findings:
                research_context = self._format_research_findings(research_findings)
                context_vars["research_insights"] = research_context
                # Update template to include research insights
                template = template.replace(
                    "{additional_context}",
                    "{additional_context}\n\nResearch Insights:\n{research_insights}",
                )

            # Format the template with context variables
            formatted_prompt = template.format(**context_vars)

            logger.info(
                f"Built single item prompt for {subsection.name if subsection else section.name} with research insights: {bool(research_findings)}"
            )
            return formatted_prompt

        except Exception as e:
            logger.error(f"Error building single item prompt: {str(e)}")
            return self._get_fallback_template().format(
                job_title="Target Position",
                company_name="Target Company",
                job_description="No job description provided",
                item_title=subsection.name if subsection else "Unknown Section",
                item_description="No existing content",
                additional_context="CV content generation",
            )

    def _format_project_content(self, content: str) -> str:
        """Format project content."""
        # Ensure clear project description
        return content

    def _format_research_findings(self, research_findings: Dict[str, Any]) -> str:
        """Format research findings for inclusion in prompts."""
        if not research_findings:
            return "No research insights available."

        formatted_insights = []

        # Format job requirements analysis
        if "job_requirements" in research_findings:
            job_reqs = research_findings["job_requirements"]
            if job_reqs.get("key_skills"):
                formatted_insights.append(
                    f"Key Skills Required: {', '.join(job_reqs['key_skills'])}"
                )
            if job_reqs.get("experience_level"):
                formatted_insights.append(
                    f"Experience Level: {job_reqs['experience_level']}"
                )

        # Format relevant CV content
        if "relevant_cv_content" in research_findings:
            cv_content = research_findings["relevant_cv_content"]
            if cv_content.get("matching_skills"):
                formatted_insights.append(
                    f"Matching Skills from CV: {', '.join(cv_content['matching_skills'])}"
                )
            if cv_content.get("relevant_experiences"):
                exp_list = [
                    exp.get("title", "Unknown")
                    for exp in cv_content["relevant_experiences"][:3]
                ]
                formatted_insights.append(
                    f"Relevant Experiences: {', '.join(exp_list)}"
                )

        # Format section relevance scores
        if "section_relevance" in research_findings:
            relevance = research_findings["section_relevance"]
            high_relevance = [
                section for section, score in relevance.items() if score > 0.7
            ]
            if high_relevance:
                formatted_insights.append(
                    f"High Relevance Sections: {', '.join(high_relevance)}"
                )

        # Format company information
        if "company_info" in research_findings:
            company = research_findings["company_info"]
            if company.get("industry"):
                formatted_insights.append(f"Company Industry: {company['industry']}")
            if company.get("size"):
                formatted_insights.append(f"Company Size: {company['size']}")

        return (
            "\n".join(formatted_insights)
            if formatted_insights
            else "Research insights available but not formatted."
        )

    def _format_executive_summary_content(self, content: str) -> str:
        """Format executive summary content."""
        # Ensure professional tone and appropriate length
        if len(content) > 300:
            # Truncate if too long
            content = content[:297] + "..."
        return content

    def _format_role_generation_output(self, validated_data) -> str:
        """Format the validated role generation data into the expected content format."""
        from ..models.validation_schemas import LLMRoleGenerationOutput

        if not isinstance(validated_data, LLMRoleGenerationOutput):
            logger.error(
                f"Expected LLMRoleGenerationOutput, got {type(validated_data)}"
            )
            return "Error: Invalid data format for role generation"

        formatted_content = ""

        for role in validated_data.roles:
            formatted_content += f"## {role.organization_description}\n\n"
            formatted_content += f"**Role:** {role.role_description}\n\n"
            formatted_content += f"**Suggested Resume Bullet Points:**\n\n"

            for skill_bullets in role.suggested_resume_bullet_points:
                formatted_content += f"### {skill_bullets.skill}\n\n"
                for bullet in skill_bullets.bullet_points:
                    formatted_content += f"• {bullet}\n"
                formatted_content += "\n"

            formatted_content += "\n---\n\n"

        return formatted_content.strip()

    def _load_prompt_template(self, prompt_key: str) -> str:
        """Load a prompt template from external file using centralized configuration."""
        try:
            prompt_path = self.settings.get_prompt_path_by_key(prompt_key)
            with open(prompt_path, "r", encoding="utf-8") as f:
                template = f.read()
            logger.info(f"Successfully loaded prompt template: {prompt_key}")
            return template
        except Exception as e:
            logger.error(f"Error loading prompt template {prompt_key}: {e}")
            # Fallback to basic template
            return self._get_fallback_template()

    def _get_fallback_template(self) -> str:
        """Fallback template if external prompt loading fails."""
        return """
Generate professional content for a {job_title} position at {company_name}.

Job Requirements:
{job_description}

Content to Enhance:
{item_title}: {item_description}

Additional Context:
{additional_context}

Generate enhanced content that aligns with the job requirements and demonstrates value to the employer.
"""

    @with_node_error_handling
    @optimize_async("agent_execution", "enhanced_content_writer")
    async def run_as_node(self, state: AgentState) -> dict:
        """
        Executes the content generation logic as a LangGraph node.
        Processes a single item specified by `current_item_id` in the state.

        Args:
            state: The current state of the workflow.

        Returns:
            A dictionary containing the updated 'structured_cv'.
        """
        # Validate input using proper validation function
        validated_input = validate_agent_input("enhanced_content_writer", state)
        logger.info(
            "Input validation passed for EnhancedContentWriterAgent",
        )

        logger.info(
            f"EnhancedContentWriterAgent processing item: {state.current_item_id}"
        )

        if not state.current_item_id:
            logger.error("Content writer called without current_item_id")
            error_list = state.error_messages or []
            error_list.append("ContentWriter failed: No current_item_id provided.")
            return {"error_messages": error_list}

        # Process the single item using the async method with research findings
        result = await self._process_single_item(
            state.structured_cv.model_dump(),
            (
                state.job_description_data.model_dump()
                if state.job_description_data
                else {}
            ),
            state.current_item_id,
            state.research_findings,
        )

        if result.success:
            # Extract the updated CV from the result
            updated_cv_data = result.output_data.get("structured_cv")
            if updated_cv_data:
                # Convert back to StructuredCV model
                updated_cv = StructuredCV.model_validate(updated_cv_data)
                logger.info(f"Successfully processed item {state.current_item_id}")
                return {"structured_cv": updated_cv}

        # If not successful, add error to state
        logger.error(
            f"Failed to process item {state.current_item_id}: {result.error_message}"
        )
        error_list = state.error_messages or []
        error_list.append(f"ContentWriter Error: {result.error_message}")
        return {"error_messages": error_list}

    async def generate_big_10_skills(
        self, job_description: str, my_talents: str = ""
    ) -> Dict[str, Any]:
        """
        Generate the "Big 10" skills specifically for Key Qualifications section.
        Returns a structured response with the skills list and raw LLM output.
        """
        try:
            # Load the key qualifications prompt template
            template = self._load_prompt_template("key_qualifications_writer")

            # Format the prompt with job description and talents
            prompt = template.format(
                main_job_description_raw=job_description[:2000],
                my_talents=my_talents
                or "Professional with diverse technical and analytical skills",
            )

            logger.info("Generating Big 10 skills with enhanced content writer")

            # Generate content using LLM
            response = await self.llm_service.generate_content(
                prompt=prompt, content_type=ContentType.QUALIFICATION
            )

            if not response or not response.content or not response.content.strip():
                logger.warning("Empty response from LLM for Big 10 skills generation")
                return {
                    "skills": [],
                    "raw_llm_output": "",
                    "success": False,
                    "error": "Empty response from LLM",
                }

            # Parse the response into individual skills
            skills_list = self.parser_agent._parse_big_10_skills(response.content)

            logger.info(f"Successfully generated {len(skills_list)} skills")

            # Format the skills for display
            formatted_content = self._format_big_10_skills_display(skills_list)

            return {
                "skills": skills_list,
                "raw_llm_output": response.content,
                "success": True,
                "formatted_content": formatted_content,
            }

        except Exception as e:
            logger.error(f"Error generating Big 10 skills: {str(e)}")
            return {
                "skills": [],
                "raw_llm_output": "",
                "success": False,
                "error": str(e),
            }

    def _format_big_10_skills_display(self, skills: List[str]) -> str:
        """
        Format the Big 10 skills for display in the CV.
        """
        if not skills:
            return "No skills generated"

        # Format as bullet points
        formatted_skills = "\n".join([f"• {skill}" for skill in skills])
        return formatted_skills

    def _ensure_job_data_structure(self, job_data: dict) -> dict:
        """
        Ensure job_data has all required fields with default values to prevent AttributeError
        """
        default_structure = {
            "title": "",
            "company": "",
            "location": "",
            "experience_level": "",
            "employment_type": "",
            "description": "",
            "requirements": [],
            "responsibilities": [],
            "benefits": [],
            "skills": [],
            "salary_range": "",
            "remote_work": False,
            "industry": "",
            "department": "",
        }

        # Merge provided data with defaults, ensuring all required fields exist
        for key, default_value in default_structure.items():
            if key not in job_data or job_data[key] is None:
                job_data[key] = default_value

        return job_data

    def _convert_string_to_job_data(self, job_string: str) -> dict:
        """
        Convert a string representation to a structured job_data dictionary
        """
        # Basic structure with the string as description
        return self._ensure_job_data_structure(
            {
                "description": job_string,
                "title": "Unknown Position",
                "company": "Unknown Company",
            }
        )

    async def _process_single_item(
        self,
        structured_cv_data: Dict[str, Any],
        job_data: Dict[str, Any],
        item_id: str,
        research_findings: Optional[Dict[str, Any]] = None,
    ) -> AgentResult:
        """
        Process a single subsection item for granular workflow (Task 3.1).

        Args:
            structured_cv_data: The current CV data structure
            job_data: Job description data for context
            item_id: The ID of the specific item to process
            research_findings: Research insights from ResearchAgent

        Returns:
            AgentResult with the updated CV structure
        """

        try:
            # Validate and parse the structured CV data
            if isinstance(structured_cv_data, dict):
                structured_cv = StructuredCV.model_validate(structured_cv_data)
            else:
                structured_cv = structured_cv_data

            # Parse job description data
            job_description = (
                JobDescriptionData.model_validate(job_data) if job_data else None
            )

            # Find the target item by ID using the built-in method
            target_item, target_section, target_subsection = (
                structured_cv.find_item_by_id(str(item_id))
            )

            if not target_item:
                logger.error(f"Could not find item with ID: {item_id}")
                return AgentResult(
                    success=False,
                    error_message=f"Item with ID {item_id} not found",
                    output_data={"structured_cv": structured_cv.model_dump()},
                )

            # Build focused prompt for this specific item with research insights
            prompt = self._build_single_item_prompt(
                target_subsection, target_section, job_description, research_findings
            )

            # Generate content using LLM with fallback handling
            try:
                response: LLMResponse = await self.llm_service.generate_content(prompt)

                # Parse the response and update the specific item
                if response.success and response.content:
                    # Store raw LLM output for transparency (REQ-FUNC-UI-6)
                    raw_output = response.content

                    # Update the target item with new content
                    target_item.content = response.content.strip()
                    target_item.status = ItemStatus.GENERATED
                    target_item.raw_llm_output = raw_output

                    logger.info(f"Successfully processed item: {item_id}")

                    return AgentResult(
                        success=True,
                        output_data={"structured_cv": structured_cv.model_dump()},
                        confidence_score=0.8,
                        metadata={
                            "processed_item_id": item_id,
                            "raw_llm_output": raw_output,
                        },
                    )
                else:
                    # LLM failed, use fallback content generation
                    logger.warning(
                        f"LLM generation failed for item {item_id}, using fallback: {response.error_message}"
                    )
                    fallback_content = self._generate_item_fallback_content(
                        target_item, target_section, target_subsection, job_description
                    )

                    # Update the target item with fallback content
                    target_item.content = fallback_content
                    target_item.status = ItemStatus.GENERATED_FALLBACK
                    target_item.raw_llm_output = f"LLM_FAILED: {response.error_message}"
                    if not target_item.metadata:
                        target_item.metadata = {}
                    target_item.metadata.update(
                        {"fallback_used": True, "error": response.error_message}
                    )

                    logger.info(f"Applied fallback content for item: {item_id}")

                    return AgentResult(
                        success=True,  # Still successful, just with fallback
                        output_data={"structured_cv": structured_cv.model_dump()},
                        confidence_score=0.3,  # Lower confidence for fallback
                        metadata={
                            "processed_item_id": item_id,
                            "fallback_used": True,
                            "llm_error": response.error_message,
                        },
                    )

            except Exception as llm_error:
                # Handle any LLM service exceptions
                logger.error(f"LLM service exception for item {item_id}: {llm_error}")
                fallback_content = self._generate_item_fallback_content(
                    target_item, target_section, target_subsection, job_description
                )

                # Update the target item with fallback content
                target_item.content = fallback_content
                target_item.status = ItemStatus.GENERATED_FALLBACK
                target_item.raw_llm_output = f"LLM_EXCEPTION: {str(llm_error)}"
                if not target_item.metadata:
                    target_item.metadata = {}
                target_item.metadata.update(
                    {"fallback_used": True, "error": str(llm_error)}
                )

                logger.info(
                    f"Applied fallback content after LLM exception for item: {item_id}"
                )

                return AgentResult(
                    success=False,  # Return failure for LLM exceptions
                    error_message=str(llm_error),
                    output_data={"structured_cv": structured_cv.model_dump()},
                    confidence_score=0.3,  # Lower confidence for fallback
                    metadata={
                        "processed_item_id": item_id,
                        "fallback_used": True,
                        "llm_exception": str(llm_error),
                    },
                )

        except Exception as e:
            logger.error(f"Error processing single item {item_id}: {e}")
            return AgentResult(
                success=False,
                error_message=str(e),
                output_data={"structured_cv": structured_cv_data},
            )

    def _build_experience_prompt_for_subsection(
        self, subsection: Subsection, job_data: Dict[str, Any]
    ) -> str:
        """
        Build a focused prompt for a single subsection/role.

        Args:
            subsection: The subsection to generate content for
            job_data: Job description data for context

        Returns:
            Formatted prompt string
        """
        # Get the experience prompt template
        template = self.content_templates.get(ContentType.EXPERIENCE, "")

        # Extract job information
        job_title = job_data.get("title", "the position")
        job_description = job_data.get("raw_text", job_data.get("description", ""))
        target_skills = job_data.get("skills", [])

        # Format target skills
        if isinstance(target_skills, list):
            target_skills_text = "\n".join([f"- {skill}" for skill in target_skills])
        else:
            target_skills_text = "- No specific skills identified"

        # Format role information from subsection
        role_name = subsection.name
        company = (
            subsection.metadata.get("company", "Unknown Company")
            if subsection.metadata
            else "Unknown Company"
        )
        dates = (
            subsection.metadata.get("dates", "Unknown Dates")
            if subsection.metadata
            else "Unknown Dates"
        )

        # Build role info string
        role_info = f"Role: {role_name}\nCompany: {company}\nDates: {dates}\n"

        # Add existing bullet points if any
        if subsection.items:
            role_info += "\nCurrent bullet points:\n"
            for item in subsection.items:
                role_info += f"- {item.content}\n"

        # Format the template
        prompt = template.replace("{{Target Skills}}", target_skills_text)
        prompt = prompt.replace("{{batched_structured_output}}", role_info)
        prompt = prompt.replace("{{job_title}}", job_title)
        prompt = prompt.replace(
            "{{job_description}}", job_description[:1000]
        )  # Limit length

        return prompt

    def _create_error_result(
        self,
        input_data: Dict[str, Any],
        context: AgentExecutionContext,
        error: Exception,
        error_type: str,
    ) -> AgentResult:
        """Create a standardized error result with fallback content."""
        # Return error result with fallback content
        content_item = input_data.get("content_item", {})
        # Ensure content_item is a dictionary for fallback generation
        if not isinstance(content_item, dict):
            content_item = {}
        content_type_fallback = (
            getattr(context, "content_type", None) or ContentType.QUALIFICATION
        )

        # Simple fallback content for error cases
        fallback_content = "⚠️ Content generation failed. Please try again."

        return AgentResult(
            success=False,
            output_data={
                "content": fallback_content,
                "content_type": content_type_fallback.value,
                "confidence_score": 0.1,
                "error": str(error),
                "fallback_used": True,
            },
            confidence_score=0.1,
            error_message=str(error),
            metadata={"fallback_used": True, "error_type": error_type},
        )

    def _generate_item_fallback_content(
        self,
        *args,
        **kwargs,
    ) -> str:
        """
        Generate fallback content when LLM fails for specific items.
        Returns:
            Fallback content string with user-friendly message
        """
        return "⚠️ The LLM did not respond or the content was not correctly generated. Please wait 10 seconds and try to regenerate!"


# Factory function for creating specialized content writers
def create_content_writer(content_type: ContentType) -> EnhancedContentWriterAgent:
    """Create a specialized content writer for a specific content type."""

    type_names = {
        ContentType.QUALIFICATION: "QualificationWriter",
        ContentType.EXPERIENCE: "ExperienceWriter",
        ContentType.EXPERIENCE_ITEM: "ExperienceItemWriter",
        ContentType.PROJECT: "ProjectWriter",
        ContentType.PROJECT_ITEM: "ProjectItemWriter",
        ContentType.EXECUTIVE_SUMMARY: "ExecutiveSummaryWriter",
        ContentType.SKILL: "SkillWriter",
        ContentType.ACHIEVEMENT: "AchievementWriter",
        ContentType.EDUCATION: "EducationWriter",
        ContentType.SKILLS: "SkillsWriter",
        ContentType.PROJECTS: "ProjectsWriter",
        ContentType.ANALYSIS: "AnalysisWriter",
        ContentType.QUALITY_CHECK: "QualityCheckWriter",
        ContentType.OPTIMIZATION: "OptimizationWriter",
        ContentType.CV_ANALYSIS: "CVAnalysisWriter",
        ContentType.CV_PARSING: "CVParsingWriter",
        ContentType.ACHIEVEMENTS: "AchievementsWriter",
    }

    type_descriptions = {
        ContentType.QUALIFICATION: "Specialized agent for generating qualification statements",
        ContentType.EXPERIENCE: "Specialized agent for enhancing work experience descriptions",
        ContentType.EXPERIENCE_ITEM: "Specialized agent for enhancing individual experience items",
        ContentType.PROJECT: "Specialized agent for creating project descriptions",
        ContentType.PROJECT_ITEM: "Specialized agent for creating individual project items",
        ContentType.EXECUTIVE_SUMMARY: "Specialized agent for crafting executive summaries",
        ContentType.SKILL: "Specialized agent for enhancing skill descriptions",
        ContentType.ACHIEVEMENT: "Specialized agent for highlighting achievements",
        ContentType.EDUCATION: "Specialized agent for education content",
        ContentType.SKILLS: "Specialized agent for skills section content",
        ContentType.PROJECTS: "Specialized agent for projects section content",
        ContentType.ANALYSIS: "Specialized agent for content analysis",
        ContentType.QUALITY_CHECK: "Specialized agent for quality assurance",
        ContentType.OPTIMIZATION: "Specialized agent for content optimization",
        ContentType.CV_ANALYSIS: "Specialized agent for CV analysis and assessment",
        ContentType.CV_PARSING: "Specialized agent for CV parsing and extraction",
        ContentType.ACHIEVEMENTS: "Specialized agent for achievements section content",
    }

    return EnhancedContentWriterAgent(
        name=type_names[content_type],
        description=type_descriptions[content_type],
        content_type=content_type,
    )
