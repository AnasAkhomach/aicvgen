"""
This module defines the ParserAgent, responsible for parsing CVs and job descriptions.
"""

from typing import Any

from ..config.logging_config import get_structured_logger
from ..error_handling.exceptions import (
    AgentExecutionError,
    DataConversionError,
    LLMResponseParsingError,
    VectorStoreError,
)
from ..models.agent_models import AgentResult
from ..models.agent_output_models import ParserAgentOutput
from ..models.data_models import (
    JobDescriptionData,
    StructuredCV,
)
from ..services.llm_cv_parser_service import LLMCVParserService
from ..services.llm_service import EnhancedLLMService
from ..services.vector_store_service import VectorStoreService
from ..templates.content_templates import ContentTemplateManager
from ..utils.cv_data_factory import (
    convert_parser_output_to_structured_cv,
)
from .agent_base import AgentBase

logger = get_structured_logger(__name__)


class ParserAgent(AgentBase):
    """Agent responsible for parsing job descriptions and CVs."""

    def __init__(
        self,
        llm_service: EnhancedLLMService,
        vector_store_service: VectorStoreService,
        template_manager: ContentTemplateManager,
        settings: dict,
        session_id: str,
    ):
        """Initialize the ParserAgent with required dependencies."""
        super().__init__(
            name="ParserAgent",
            description="Parses raw text of CVs and job descriptions into structured data.",
            session_id=session_id,
        )
        self.vector_store_service = vector_store_service
        self.llm_cv_parser_service = LLMCVParserService(
            llm_service, settings, template_manager
        )

    async def run(self, **kwargs: Any) -> AgentResult:
        """Parses a raw document (CV or job description) into a structured format."""
        self.update_progress(0, "Starting parsing process")
        input_data = kwargs.get("input_data", {})

        try:
            raw_text = input_data.get("raw_text")
            doc_type = input_data.get("type")

            if raw_text is None or doc_type is None:
                raise AgentExecutionError(
                    agent_name=self.name,
                    message="'raw_text' and 'type' are required fields.",
                )

            output = ParserAgentOutput()
            if doc_type == "cv":
                self.update_progress(20, "Parsing CV")
                parsed_data = await self.parse_cv(raw_text)
                output.structured_cv = parsed_data
            elif doc_type == "job":
                self.update_progress(20, "Parsing job description")
                parsed_data = await self.parse_job_description(raw_text)
                output.job_description_data = parsed_data
            else:
                raise AgentExecutionError(
                    agent_name=self.name,
                    message=f"Unsupported document type: {doc_type}",
                )

            self.update_progress(100, "Parsing completed")
            return AgentResult(success=True, output_data=output)

        except AgentExecutionError as e:
            logger.error(
                "Agent execution error in ParserAgent: %s", str(e), exc_info=True
            )
            return AgentResult(
                success=False,
                error_message=str(e),
                output_data=ParserAgentOutput(),
            )
        except (AttributeError, TypeError, ValueError) as e:
            logger.error(
                "Unhandled exception in ParserAgent: %s", str(e), exc_info=True
            )
            return AgentResult(
                success=False,
                error_message=f"An unexpected error occurred in ParserAgent: {e}",
                output_data=ParserAgentOutput(),
            )

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

    async def parse_cv(self, raw_text: str) -> StructuredCV:
        """Parses a raw CV using an LLM and converts it to a structured format."""
        if not raw_text.strip():
            raise AgentExecutionError(
                agent_name=self.name, message="Cannot parse an empty CV."
            )
        try:
            self.update_progress(40, "Calling LLM for CV parsing")
            llm_output = await self.llm_cv_parser_service.parse_cv_with_llm(raw_text)

            self.update_progress(70, "Converting LLM output to structured format")
            structured_cv = convert_parser_output_to_structured_cv(llm_output)

            self.update_progress(90, "Storing CV vectors")
            await self._store_cv_vectors(structured_cv)

            return structured_cv
        except (LLMResponseParsingError, DataConversionError) as e:
            raise AgentExecutionError(
                agent_name=self.name, message=f"Failed during CV parsing process: {e}"
            ) from e

    async def _store_cv_vectors(self, structured_cv: StructuredCV):
        """Stores the vectors for the parsed CV content."""
        try:
            await self.vector_store_service.add_structured_cv(structured_cv)
            logger.info("Successfully stored vectors for CV.")
        except VectorStoreError as e:
            logger.warning("Failed to store CV vectors", error=str(e))
            # Non-critical error, so we don't re-raise as AgentExecutionError
