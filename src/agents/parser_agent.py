"""
This module defines the ParserAgent, responsible for parsing CVs and job descriptions.
"""

from typing import Any
from uuid import uuid4

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
    determine_section_content_type,
    determine_item_type,
)
from .agent_base import AgentBase

from ..models.data_models import Section, Subsection, Item, ItemStatus


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

    def _validate_inputs(self, input_data: dict) -> None:
        """Validate inputs for parser agent."""
        if not input_data.get("raw_text") or not input_data.get("type"):
            raise AgentExecutionError(
                agent_name=self.name,
                message="'raw_text' and 'type' are required fields.",
            )

    async def _execute(self, **kwargs: Any) -> AgentResult:
        """Execute the core parsing logic."""
        input_data = kwargs.get("input_data", {})
        raw_text = input_data.get("raw_text")
        doc_type = input_data.get("type")

        output = ParserAgentOutput()
        if doc_type == "cv":
            self.update_progress(40, "Parsing CV")
            parsed_data = await self.parse_cv(raw_text)
            output.structured_cv = parsed_data
        elif doc_type == "job":
            self.update_progress(40, "Parsing job description")
            parsed_data = await self.parse_job_description(raw_text)
            output.job_description_data = parsed_data
        else:
            raise AgentExecutionError(
                agent_name=self.name,
                message=f"Unsupported document type: {doc_type}",
            )

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
            # Create a StructuredCV directly from CVParsingResult
            structured_cv = self._convert_cv_parsing_result_to_structured_cv(
                llm_output, raw_text
            )

            self.update_progress(90, "Storing CV vectors")
            await self._store_cv_vectors(structured_cv)

            return structured_cv
        except (LLMResponseParsingError, DataConversionError) as e:
            raise AgentExecutionError(
                agent_name=self.name, message=f"Failed during CV parsing process: {e}"
            ) from e

    def _convert_cv_parsing_result_to_structured_cv(
        self, parsing_result, cv_text: str
    ) -> StructuredCV:
        """Convert CVParsingResult to StructuredCV."""

        # Create base structure
        structured_cv = StructuredCV.create_empty(cv_text=cv_text)

        # Convert sections
        sections = []
        for section_data in parsing_result.sections:
            section = Section(
                id=uuid4(),
                title=section_data.name,
                content_type=determine_section_content_type(section_data.name),
                status=ItemStatus.GENERATED,
                subsections=[],
            )

            # Add direct items to section if any
            if section_data.items:
                subsection = Subsection(
                    id=uuid4(),
                    title="Main",
                    status=ItemStatus.GENERATED,
                    items=[],
                )

                for item_content in section_data.items:
                    item = Item(
                        id=uuid4(),
                        content=item_content,
                        item_type=determine_item_type(section_data.name),
                        status=ItemStatus.GENERATED,
                    )
                    subsection.items.append(item)
                section.subsections.append(subsection)

            # Add subsections
            for subsection_data in section_data.subsections:
                subsection = Subsection(
                    id=uuid4(),
                    title=subsection_data.name,
                    status=ItemStatus.GENERATED,
                    items=[],
                )

                for item_content in subsection_data.items:
                    item = Item(
                        id=uuid4(),
                        content=item_content,
                        item_type=determine_item_type(section_data.name),
                        status=ItemStatus.GENERATED,
                    )
                    subsection.items.append(item)
                section.subsections.append(subsection)

            sections.append(section)

        structured_cv.sections = sections
        return structured_cv

    async def _store_cv_vectors(self, structured_cv: StructuredCV):
        """Stores the vectors for the parsed CV content."""
        try:
            await self.vector_store_service.add_structured_cv(structured_cv)
            logger.info("Successfully stored vectors for CV.")
        except VectorStoreError as e:
            logger.warning("Failed to store CV vectors", error=str(e))
            # Non-critical error, so we don't re-raise as AgentExecutionError

    async def run_as_node(self, state: dict) -> dict:
        """Runs the agent as a node in the workflow."""
        self.update_progress(0, "Starting parser node.")
        try:
            # Extract relevant data from the state
            raw_cv_text = state.get("cv_text")
            job_description_text = state.get("job_description_data", {}).get("raw_text")

            # Parse CV
            if raw_cv_text:
                structured_cv = await self.parse_cv(raw_cv_text)
                state["structured_cv"] = structured_cv

            # Parse Job Description
            if job_description_text:
                job_description_data = await self.parse_job_description(
                    job_description_text
                )
                state["job_description_data"] = job_description_data

            self.update_progress(100, "Parser node completed.")
            return state
        except Exception as e:
            logger.error(f"Error in parser node: {e}", exc_info=True)
            state["error_messages"] = state.get("error_messages", []) + [str(e)]
            return state
