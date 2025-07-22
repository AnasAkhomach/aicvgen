"""
This module defines the JobDescriptionParserAgent, responsible for parsing job descriptions.
"""

from typing import Any
from src.agents.agent_base import AgentBase
from src.config.logging_config import get_structured_logger
from src.constants.agent_constants import AgentConstants
from src.error_handling.exceptions import (AgentExecutionError, LLMResponseParsingError)

from src.models.agent_output_models import ParserAgentOutput
from src.models.data_models import JobDescriptionData
from src.services.llm_cv_parser_service import LLMCVParserService
from src.services.llm_service_interface import LLMServiceInterface
from src.templates.content_templates import ContentTemplateManager

logger = get_structured_logger(__name__)


class JobDescriptionParserAgent(AgentBase):
    """Agent responsible for parsing job descriptions."""

    def __init__(
        self,
        llm_service: LLMServiceInterface,
        template_manager: ContentTemplateManager,
        settings: dict,
        session_id: str,
    ):
        """Initialize the JobDescriptionParserAgent with required dependencies."""
        super().__init__(
            name="JobDescriptionParserAgent",
            description="Parses raw text of job descriptions into structured data.",
            session_id=session_id,
            settings=settings,
        )
        self.llm_cv_parser_service = LLMCVParserService(
            llm_service, settings, template_manager
        )

    async def _execute(self, **kwargs: Any) -> dict[str, Any]:
        """Execute the core parsing logic."""
        input_data = kwargs.get("input_data", {})
        raw_text = input_data.get("raw_text")

        self.update_progress(AgentConstants.PROGRESS_MAIN_PROCESSING, "Parsing job description")
        parsed_data = await self.parse_job_description(raw_text)

        self.update_progress(AgentConstants.PROGRESS_COMPLETE, "Parsing completed")
        return {"job_description_data": parsed_data}

    async def parse_job_description(self, raw_text: str) -> JobDescriptionData:
        """Parses a raw job description using an LLM."""
        if not raw_text.strip():
            return JobDescriptionData(raw_text=raw_text)
        try:
            # Extract system instruction from settings if available
            system_instruction = self.settings.get("system_instruction")
            
            return await self.llm_cv_parser_service.parse_job_description_with_llm(
                raw_text,
                session_id=self.session_id,
                system_instruction=system_instruction
            )
        except LLMResponseParsingError as e:
            raise AgentExecutionError(
                agent_name=self.name,
                message=f"Failed to parse job description from LLM response: {e}",
            ) from e
