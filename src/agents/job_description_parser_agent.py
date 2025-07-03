"""
This module defines the JobDescriptionParserAgent, responsible for parsing job descriptions.
"""

from typing import Any
from ..config.logging_config import get_structured_logger
from ..error_handling.exceptions import (
    AgentExecutionError,
    LLMResponseParsingError,
)
from ..models.agent_models import AgentResult
from ..models.agent_output_models import ParserAgentOutput
from ..models.data_models import (
    JobDescriptionData,
)
from ..services.llm_cv_parser_service import LLMCVParserService
from ..services.llm_service import EnhancedLLMService
from ..templates.content_templates import ContentTemplateManager
from .agent_base import AgentBase


logger = get_structured_logger(__name__)


class JobDescriptionParserAgent(AgentBase):
    """Agent responsible for parsing job descriptions."""

    def __init__(
        self,
        llm_service: EnhancedLLMService,
        template_manager: ContentTemplateManager,
        settings: dict,
        session_id: str,
    ):
        """Initialize the JobDescriptionParserAgent with required dependencies."""
        super().__init__(
            name="JobDescriptionParserAgent",
            description="Parses raw text of job descriptions into structured data.",
            session_id=session_id,
        )
        self.llm_cv_parser_service = LLMCVParserService(
            llm_service, settings, template_manager
        )

    def _validate_inputs(self, input_data: dict) -> None:
        """Validate inputs for parser agent."""
        if not input_data.get("raw_text"):
            raise AgentExecutionError(
                agent_name=self.name,
                message="'raw_text' is a required field.",
            )

    async def _execute(self, **kwargs: Any) -> AgentResult:
        """Execute the core parsing logic."""
        input_data = kwargs.get("input_data", {})
        raw_text = input_data.get("raw_text")

        output = ParserAgentOutput()
        self.update_progress(40, "Parsing job description")
        parsed_data = await self.parse_job_description(raw_text)
        output.job_description_data = parsed_data

        self.update_progress(100, "Parsing completed")
        return AgentResult(success=True, output_data=output)

    async def parse_job_description(self, raw_text: str) -> JobDescriptionData:
        """Parses a raw job description using an LLM."""
        if not raw_text.strip():
            return JobDescriptionData(raw_text=raw_text)
        try:
            return await self.llm_cv_parser_service.parse_job_description_with_llm(
                raw_text
            )
        except LLMResponseParsingError as e:
            raise AgentExecutionError(
                agent_name=self.name,
                message=f"Failed to parse job description from LLM response: {e}",
            ) from e

    async def run_as_node(self, state: dict) -> dict:
        """Runs the agent as a node in the workflow."""
        self.update_progress(0, "Starting JD parser node.")
        try:
            # Extract relevant data from the state
            job_description_text = state.get("job_description_data", {}).get("raw_text")

            # Parse Job Description
            if job_description_text:
                job_description_data = await self.parse_job_description(
                    job_description_text
                )
                state["job_description_data"] = job_description_data

            self.update_progress(100, "JD parser node completed.")
            return state
        except Exception as e:
            logger.error(f"Error in JD parser node: {e}", exc_info=True)
            state["error_messages"] = state.get("error_messages", []) + [str(e)]
            return state
