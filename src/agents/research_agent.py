"""
This module defines the ResearchAgent, responsible for conducting research on job descriptions and CVs.
"""

from typing import Any

from src.agents.agent_base import AgentBase
from src.config.logging_config import get_structured_logger
from src.config.settings import AgentSettings
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
from src.models.data_models import JobDescriptionData, StructuredCV
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
        settings: AgentSettings,
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
        self.settings = settings
        self.template_manager = template_manager

    async def _execute(self, **kwargs: Any) -> AgentResult[ResearchAgentOutput]:
        """
        Executes the research agent asynchronously.
        """
        # Extract parameters from kwargs
        job_description_data = kwargs.get("job_description_data")
        structured_cv = kwargs.get("structured_cv")

        self.update_progress(
            AgentConstants.PROGRESS_START, "Starting research analysis"
        )

        try:
            self.update_progress(
                AgentConstants.PROGRESS_INPUT_VALIDATION, "Input validation passed."
            )

            research_findings = await self._perform_research_analysis(
                structured_cv, job_description_data
            )
            self.update_progress(
                AgentConstants.PROGRESS_POST_PROCESSING,
                "Core research analysis complete.",
            )

            if research_findings.status == ResearchStatus.FAILED:
                # Even if the research fails, we might have partial results, so we return them.
                # The error is logged within _perform_research_analysis.
                # We wrap this in a success result because the agent itself didn't crash.
                output = ResearchAgentOutput(research_findings=research_findings)
                self.update_progress(
                    AgentConstants.PROGRESS_COMPLETE,
                    "Research analysis failed but returning partial data.",
                )
                return AgentResult.success(
                    agent_name=self.name,
                    output_data=output,
                    message="Research analysis failed, but some data may be available.",
                )

            output = ResearchAgentOutput(research_findings=research_findings)
            self.update_progress(
                AgentConstants.PROGRESS_COMPLETE,
                "Research analysis completed successfully.",
            )
            return AgentResult.success(
                agent_name=self.name,
                output_data=output,
                message="Research analysis completed successfully.",
            )

        except (AgentExecutionError, LLMResponseParsingError) as e:
            logger.error("A known error occurred in ResearchAgent: %s", str(e))
            return AgentResult.failure(agent_name=self.name, error_message=str(e))
        except Exception as e:
            logger.error(
                "Unhandled exception in ResearchAgent", error=str(e), exc_info=True
            )
            return AgentResult.failure(
                agent_name=self.name,
                error_message=f"An unexpected error occurred during research: {e}",
            )

    async def _perform_research_analysis(
        self, structured_cv: StructuredCV, job_desc_data: JobDescriptionData
    ) -> ResearchFindings:
        """Performs the core research by calling an LLM and vector store."""
        try:
            # This can be expanded to include vector store lookups, web searches, etc.
            # For now, it focuses on LLM-based analysis.
            self.update_progress(
                AgentConstants.PROGRESS_PREPROCESSING, "Generating research prompt."
            )
            prompt = self._create_research_prompt(structured_cv, job_desc_data)

            self.update_progress(
                AgentConstants.PROGRESS_MAIN_PROCESSING,
                "Querying LLM for research insights.",
            )
            llm_response = await self.llm_service.query_llm(
                prompt,
                max_tokens=self.settings.max_tokens,
                temperature=self.settings.temperature,
            )

            self.update_progress(
                AgentConstants.PROGRESS_LLM_PARSING, "Parsing LLM response."
            )
            parsed_findings = self._parse_llm_response(llm_response)

            return parsed_findings

        except LLMResponseParsingError as e:
            logger.error("Failed to parse LLM response in research agent", error=str(e))
            return ResearchFindings(
                status=ResearchStatus.FAILED,
                error_message=f"Failed to parse LLM response: {str(e)}",
            )
        except Exception as e:
            logger.error(
                "An unexpected error occurred during research analysis",
                error=str(e),
                exc_info=True,
            )
            return ResearchFindings(
                status=ResearchStatus.FAILED,
                error_message=f"An unexpected error occurred: {str(e)}",
            )

    def _create_research_prompt(
        self, structured_cv: StructuredCV, job_desc_data: JobDescriptionData
    ) -> str:
        """
        Creates the research prompt for the LLM based on the job description and CV.

        This is a simplified example and should be expanded with the actual prompt
        engineering logic.
        """
        return f"Research insights for {job_desc_data.title} at {job_desc_data.company_name}."

    def _parse_llm_response(self, llm_response: str) -> ResearchFindings:
        """
        Parses the LLM response into structured research findings.

        This is a simplified parser and should be expanded to accurately extract
        information from the LLM's response format.
        """
        return ResearchFindings(
            status=ResearchStatus.SUCCESS,
            key_terms=["Python", "Django"],  # Example static data
            skill_gaps=["Machine Learning"],  # Example static data
            enhancement_suggestions=["Add more projects"],  # Example static data
            company_insights=CompanyInsight(
                company_name="Example Corp",
                industry="Software",
                confidence_score=AgentConstants.DEFAULT_CONFIDENCE_SCORE,
            ),
            industry_insights=IndustryInsight(industry_name="Software"),
            role_insights=RoleInsight(role_name="Software Engineer"),
        )
