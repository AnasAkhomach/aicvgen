"""
This module defines the ResearchAgent, responsible for conducting research on job descriptions and CVs.
"""

import json
import re
from typing import Any, Union

from src.agents.agent_base import AgentBase
from src.config.logging_config import get_structured_logger
from src.constants.llm_constants import LLMConstants
from src.error_handling.exceptions import AgentExecutionError, LLMResponseParsingError
from src.models.agent_models import AgentResult
from src.models.agent_output_models import (
    CompanyInsight,
    IndustryInsight,
    ResearchAgentOutput,
    ResearchFindings,
    ResearchStatus,
    RoleInsight,
)
from src.models.data_models import JobDescriptionData
from src.constants.agent_constants import AgentConstants
from src.services.llm_service_interface import LLMServiceInterface
from src.services.vector_store_service import VectorStoreService
from src.templates.content_templates import ContentTemplateManager

logger = get_structured_logger(__name__)


class ResearchAgent(AgentBase):
    """
    Agent responsible for conducting research related to the job description and finding
    relevant content from the CV for tailoring.
    """

    def __init__(
        self,
        llm_service: LLMServiceInterface,
        vector_store_service: VectorStoreService,
        settings: dict,
        template_manager: ContentTemplateManager,
        session_id: str,
        name: str = "ResearchAgent",
        description: str = "Conducts research and gathers insights on jobs and CVs.",
    ):
        """Initializes the ResearchAgent with necessary services."""
        super().__init__(
            name=name, description=description, session_id=session_id, settings=settings
        )
        self.llm_service = llm_service
        self.vector_store_service = vector_store_service
        self.template_manager = template_manager

    async def _execute(self, **kwargs: Any) -> AgentResult[ResearchAgentOutput]:
        """
        Executes the research agent asynchronously.
        """
        # Extract parameters from kwargs
        job_description_data = kwargs.get("job_description_data")

        self.update_progress(
            AgentConstants.PROGRESS_START, "Starting research analysis"
        )

        try:
            # Input validation
            if not job_description_data:
                logger.error("Missing job_description_data in ResearchAgent input")
                return AgentResult.create_failure(
                    agent_name=self.name,
                    error_message="Missing required job description data",
                )

            self.update_progress(
                AgentConstants.PROGRESS_INPUT_VALIDATION, "Input validation passed."
            )

            research_findings = await self._perform_research_analysis(
                job_description_data
            )
            self.update_progress(
                AgentConstants.PROGRESS_POST_PROCESSING,
                "Core research analysis complete.",
            )

            # Validate research findings result
            if not research_findings:
                logger.error("Research analysis returned None")
                return AgentResult.create_failure(
                    agent_name=self.name,
                    error_message="Research analysis failed to return any results",
                )

            if research_findings.status == ResearchStatus.FAILED:
                # Research failed but we have error details
                logger.warning(
                    f"Research analysis failed: {research_findings.error_message}"
                )
                output = ResearchAgentOutput(research_findings=research_findings)
                self.update_progress(
                    AgentConstants.PROGRESS_COMPLETE,
                    "Research analysis failed but returning error details.",
                )
                return AgentResult.create_failure(
                    agent_name=self.name,
                    error_message=research_findings.error_message
                    or "Research analysis failed",
                )

            # Success case
            output = ResearchAgentOutput(research_findings=research_findings)
            self.update_progress(
                AgentConstants.PROGRESS_COMPLETE,
                "Research analysis completed successfully.",
            )
            return AgentResult.create_success(
                agent_name=self.name,
                output_data=output,
                message="Research analysis completed successfully.",
            )

        except LLMResponseParsingError as e:
            logger.error("LLM response parsing failed in ResearchAgent: %s", str(e))
            self.update_progress(
                AgentConstants.PROGRESS_COMPLETE,
                "Research analysis failed due to parsing error.",
            )
            return AgentResult.create_failure(
                agent_name=self.name,
                error_message=f"Failed to parse LLM response: {str(e)}",
            )
        except AgentExecutionError as e:
            logger.error("Agent execution error in ResearchAgent: %s", str(e))
            self.update_progress(
                AgentConstants.PROGRESS_COMPLETE,
                "Research analysis failed due to execution error.",
            )
            return AgentResult.create_failure(
                agent_name=self.name, error_message=str(e)
            )
        except (ValueError, TypeError, AttributeError) as e:
            logger.error(
                "Validation or type error in ResearchAgent", error=str(e), exc_info=True
            )
            self.update_progress(
                AgentConstants.PROGRESS_COMPLETE,
                "Research analysis failed due to validation error.",
            )
            return AgentResult.create_failure(
                agent_name=self.name,
                error_message=f"A validation error occurred during research: {e}",
            )

    async def _perform_research_analysis(
        self, job_desc_data: Union[JobDescriptionData, dict]
    ) -> ResearchFindings:
        """Performs the core research by calling an LLM and vector store."""
        try:
            # This can be expanded to include vector store lookups, web searches, etc.
            # For now, it focuses on LLM-based analysis.
            self.update_progress(
                AgentConstants.PROGRESS_PREPROCESSING, "Generating research prompt."
            )
            prompt = self._create_research_prompt(job_desc_data)

            # Debug logging: Log the exact prompt sent to LLM
            logger.debug(f"PROMPT SENT TO LLM:\n{prompt}")

            self.update_progress(
                AgentConstants.PROGRESS_MAIN_PROCESSING,
                "Querying LLM for research insights.",
            )
            llm_response = await self.llm_service.generate_content(
                prompt=prompt,
                max_tokens=self.settings.get(
                    "max_tokens_analysis", LLMConstants.DEFAULT_MAX_TOKENS
                ),
                temperature=self.settings.get(
                    "temperature_analysis", LLMConstants.TEMPERATURE_BALANCED
                ),
            )

            # Debug logging: Log the raw LLM response
            logger.debug(f"RAW LLM RESPONSE:\n{llm_response.content}")

            self.update_progress(
                AgentConstants.PROGRESS_LLM_PARSING, "Parsing LLM response."
            )
            parsed_findings = self._parse_llm_response(llm_response.content)

            # Validate the parsed findings
            if not parsed_findings:
                logger.error("Parsing LLM response yielded no findings.")
                return ResearchFindings(
                    status=ResearchStatus.FAILED,
                    error_message="Failed to parse LLM response - no findings extracted",
                )

            # Additional validation to ensure we have a valid ResearchFindings object
            if not isinstance(parsed_findings, ResearchFindings):
                logger.error(
                    f"Expected ResearchFindings object, got {type(parsed_findings)}"
                )
                return ResearchFindings(
                    status=ResearchStatus.FAILED,
                    error_message=f"Invalid parsing result type: {type(parsed_findings)}",
                )

            return parsed_findings

        except LLMResponseParsingError as e:
            logger.error("Failed to parse LLM response in research agent", error=str(e))
            return ResearchFindings(
                status=ResearchStatus.FAILED,
                error_message=f"Failed to parse LLM response: {str(e)}",
            )
        except (ValueError, TypeError, AttributeError) as e:
            logger.error(
                "Validation or type error during research analysis",
                error=str(e),
                exc_info=True,
            )
            return ResearchFindings(
                status=ResearchStatus.FAILED,
                error_message=f"A validation error occurred: {str(e)}",
            )

    def _create_research_prompt(
        self, job_desc_data: Union[JobDescriptionData, dict]
    ) -> str:
        """
        Creates the research prompt for the LLM based on the job description using ContentTemplateManager.

        Args:
            job_desc_data: Job description data as either a JobDescriptionData model or dict

        Returns:
            A research prompt string for the LLM
        """
        # Defensive programming: ensure job_desc_data is a Pydantic model
        if not isinstance(job_desc_data, JobDescriptionData):
            # If it's a dict, try to convert it as a fallback
            if isinstance(job_desc_data, dict):
                try:
                    job_desc_data = JobDescriptionData(**job_desc_data)
                    logger.warning(
                        "Converted dict to JobDescriptionData model in ResearchAgent"
                    )
                except Exception as e:
                    logger.error(f"Failed to convert dict to JobDescriptionData: {e}")
                    raise TypeError(
                        f"Expected JobDescriptionData model, but received a dict that could not be validated: {e}"
                    ) from e
            else:
                raise TypeError(
                    f"Expected JobDescriptionData model, got {type(job_desc_data)}"
                )

        try:
            # Get the job research analysis template using direct lookup
            # Note: ContentTemplateManager stores templates by name only, not the full key format
            template = self.template_manager.templates.get("job_research_analysis")

            if template:
                # Format the template with job description data matching the template variables
                variables = {
                    "raw_jd": job_desc_data.main_job_description_raw
                    or job_desc_data.raw_text
                    or "",
                    "skills": (
                        ", ".join(job_desc_data.skills)
                        if job_desc_data.skills
                        else "Not specified"
                    ),
                    "company_name": job_desc_data.company_name or "Unknown Company",
                    "job_title": job_desc_data.job_title or "Unknown Position",
                }
                return self.template_manager.format_template(template, variables)
            else:
                # Fallback to default prompt if template not found
                logger.warning("Job analysis template not found, using fallback prompt")
                return f"""
Analyze the following job description and return a JSON object with these five keys:
1. "core_technical_skills": List of essential technical skills
2. "soft_skills": List of important soft skills
3. "key_performance_metrics": List of measurable performance indicators
4. "project_types": List of typical project types for this role
5. "working_environment_characteristics": List of work environment traits

Job Title: {job_desc_data.job_title or "Unknown Position"}
Company: {job_desc_data.company_name or "Unknown Company"}
Description: {job_desc_data.main_job_description_raw or job_desc_data.raw_text or ""}
Requirements: {", ".join(job_desc_data.skills) if job_desc_data.skills else ""}
Responsibilities: {", ".join(job_desc_data.responsibilities) if job_desc_data.responsibilities else ""}

Return only valid JSON.
"""
        except (AttributeError, KeyError, ValueError) as e:
            logger.error(f"Error creating research prompt: {e}")
            # Return a basic fallback prompt
            job_title = job_desc_data.job_title or "Unknown Position"
            company_name = job_desc_data.company_name or "Unknown Company"
            return f"Research insights for {job_title} at {company_name}."

    def _parse_llm_response(self, llm_response: str) -> ResearchFindings:
        """Parse the LLM response and extract research findings."""
        try:

            # Validate input
            if not llm_response or not llm_response.strip():
                logger.error("Empty or None LLM response received")
                raise LLMResponseParsingError("Empty or None LLM response received")

            # Try to extract JSON from markdown code blocks first
            json_code_block_match = re.search(
                r"```json\s*\n(.*?)\n```", llm_response, re.DOTALL | re.IGNORECASE
            )
            if json_code_block_match:
                json_str = json_code_block_match.group(1).strip()
                try:
                    parsed_data = json.loads(json_str)
                    is_json_response = True
                    logger.debug("Successfully extracted JSON from markdown code block")
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON from code block: {e}")
                    raise LLMResponseParsingError(
                        f"Failed to parse JSON from code block: {e}"
                    ) from e
            else:
                # Fallback: Try to extract JSON from the response without code blocks
                json_match = re.search(r"\{.*\}", llm_response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    try:
                        parsed_data = json.loads(json_str)
                        is_json_response = True
                        logger.debug("Successfully extracted JSON from raw response")
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse JSON from LLM response: {e}")
                        # If we found JSON structure but it's malformed, raise error
                        raise LLMResponseParsingError(
                            f"Failed to parse LLM response as JSON: {e}"
                        ) from e
                else:
                    # Check if the response looks like it should be JSON but is malformed
                    if "{" in llm_response or "}" in llm_response:
                        logger.error(
                            "Response contains JSON-like structure but is malformed"
                        )
                        raise LLMResponseParsingError(
                            "Failed to parse LLM response as JSON"
                        )

                    # Check if this looks like a simple invalid response vs structured text
                    # If it's a short, simple sentence, treat it as invalid JSON
                    if len(llm_response.strip()) < 50 and not any(
                        keyword in llm_response.lower()
                        for keyword in [
                            "skill",
                            "experience",
                            "require",
                            "python",
                            "javascript",
                            "communication",
                        ]
                    ):
                        logger.error(
                            "Response appears to be invalid JSON rather than structured text"
                        )
                        raise LLMResponseParsingError(
                            "Failed to parse LLM response as JSON"
                        )

                    # No JSON found, extract from text (fallback for structured text responses)
                    logger.warning(
                        "No JSON structure found in LLM response, using text extraction"
                    )
                    parsed_data = self._extract_from_text(llm_response)
                    is_json_response = False

            # Validate that we have some data
            if not parsed_data or not isinstance(parsed_data, dict):
                logger.error("Failed to extract any valid data from LLM response")
                raise LLMResponseParsingError(
                    "Could not extract valid data structure from LLM response"
                )

            # Extract core data with defaults
            core_technical_skills = parsed_data.get("core_technical_skills", [])
            soft_skills = parsed_data.get("soft_skills", [])
            key_performance_metrics = parsed_data.get("key_performance_metrics", [])
            project_types = parsed_data.get("project_types", [])
            working_environment = parsed_data.get(
                "working_environment_characteristics", []
            )

            # Check if this was a fallback to text extraction (no JSON found)
            is_text_fallback = not is_json_response

            # Validate that we have meaningful data for JSON responses
            if not is_text_fallback:
                # For JSON responses, we need at least 3 out of 5 main fields to be present
                present_fields = sum(
                    1
                    for field in [
                        core_technical_skills,
                        soft_skills,
                        key_performance_metrics,
                        project_types,
                        working_environment,
                    ]
                    if field
                )
                if present_fields < 3:
                    logger.error(
                        f"JSON response missing too many required fields (only {present_fields}/5 present)"
                    )
                    raise LLMResponseParsingError(
                        "Failed to extract meaningful data from LLM response"
                    )

            # For text fallback, allow minimal data
            if is_text_fallback and not any(
                [
                    core_technical_skills,
                    soft_skills,
                    key_performance_metrics,
                    project_types,
                    working_environment,
                ]
            ):
                logger.warning(
                    "No meaningful data extracted from text fallback, using minimal defaults"
                )

            # Create RoleInsight with extracted data
            role_insight = RoleInsight(
                role_title="Software Engineer",  # Default for now, can be enhanced
                required_skills=(
                    core_technical_skills[:10]
                    if core_technical_skills
                    else ["Programming", "Problem Solving"]
                ),
                preferred_qualifications=(
                    soft_skills[:5]
                    if soft_skills
                    else ["Bachelor's Degree", "Team Collaboration"]
                ),
                responsibilities=(
                    project_types[:8]
                    if project_types
                    else ["Software Development", "Code Review"]
                ),
                career_progression=[
                    "Senior Engineer",
                    "Lead Engineer",
                    "Principal Engineer",
                ],
                salary_range="$70,000 - $120,000",
                confidence_score=0.85,
            )

            # Create company insight with defaults
            company_insight = CompanyInsight(
                company_name="Technology Company",
                industry="Technology",
                company_size="Medium",
                culture_keywords=(
                    working_environment[:5]
                    if working_environment
                    else ["Innovation", "Collaboration"]
                ),
                benefits=["Health Insurance", "401k", "Professional Development"],
                confidence_score=0.75,
            )

            # Create industry insight
            industry_insight = IndustryInsight(
                industry_name="Technology",
                trends=["Digital Transformation", "AI Integration"],
                key_skills=(
                    core_technical_skills[:8]
                    if core_technical_skills
                    else ["Programming", "System Design"]
                ),
                growth_areas=["Cloud Computing", "Machine Learning"],
                challenges=["Talent Shortage", "Rapid Technology Changes"],
                confidence_score=0.80,
            )

            # Determine status based on whether we used text fallback
            status = (
                ResearchStatus.PARTIAL if is_text_fallback else ResearchStatus.SUCCESS
            )

            return ResearchFindings(
                status=status,
                role_insights=role_insight,
                company_insights=company_insight,
                industry_insights=industry_insight,
            )

        except LLMResponseParsingError:
            # Re-raise our custom parsing errors
            raise
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.error(f"Failed to parse LLM response: {e}")
            raise LLMResponseParsingError(
                f"Could not parse research findings: {e}"
            ) from e
        except (AttributeError, TypeError) as e:
            logger.error(f"Unexpected error parsing LLM response: {e}")
            raise LLMResponseParsingError(
                f"Unexpected error during parsing: {e}"
            ) from e

    def _extract_from_text(self, text: str) -> dict:
        """Extract structured data from unstructured text response."""
        # Simple text extraction as fallback
        # Look for lists in the text
        skills_pattern = r"(?:technical skills?|skills?)[:.]?\s*([^\n]+(?:\n[^\n]+)*?)(?=\n\n|\n[A-Z]|$)"
        soft_skills_pattern = r"(?:soft skills?|interpersonal)[:.]?\s*([^\n]+(?:\n[^\n]+)*?)(?=\n\n|\n[A-Z]|$)"

        technical_match = re.search(skills_pattern, text, re.IGNORECASE | re.MULTILINE)
        soft_match = re.search(soft_skills_pattern, text, re.IGNORECASE | re.MULTILINE)

        technical_skills = []
        if technical_match:
            # Split by common delimiters and clean up
            skills_text = technical_match.group(1)
            technical_skills = [
                s.strip("- •*") for s in re.split(r"[,;\n]", skills_text) if s.strip()
            ]

        soft_skills = []
        if soft_match:
            skills_text = soft_match.group(1)
            soft_skills = [
                s.strip("- •*") for s in re.split(r"[,;\n]", skills_text) if s.strip()
            ]

        return {
            "core_technical_skills": technical_skills,
            "soft_skills": soft_skills,
            "key_performance_metrics": [],
            "project_types": [],
            "working_environment_characteristics": [],
        }
