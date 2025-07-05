"""
This module defines the UserCVParserAgent, responsible for parsing user CVs.
"""

from typing import Any
from uuid import uuid4

from src.agents.agent_base import AgentBase
from src.config.logging_config import get_structured_logger
from src.error_handling.exceptions import AgentExecutionError, DataConversionError, LLMResponseParsingError, VectorStoreError
from src.models.agent_models import AgentResult
from src.models.agent_output_models import ParserAgentOutput
from src.models.data_models import Item, ItemStatus, Section, StructuredCV, Subsection
from src.services.llm_cv_parser_service import LLMCVParserService
from src.services.llm_service_interface import LLMServiceInterface
from src.constants.agent_constants import AgentConstants
from src.services.vector_store_service import VectorStoreService
from src.templates.content_templates import ContentTemplateManager
from src.utils.cv_data_factory import determine_item_type, determine_section_content_type

logger = get_structured_logger(__name__)


class UserCVParserAgent(AgentBase):
    """Agent responsible for parsing user CVs."""

    def __init__(
        self,
        llm_service: LLMServiceInterface,
        vector_store_service: VectorStoreService,
        template_manager: ContentTemplateManager,
        settings: dict,
        session_id: str,
    ):
        """Initialize the UserCVParserAgent with required dependencies."""
        super().__init__(
            name="UserCVParserAgent",
            description="Parses raw text of user CVs into structured data.",
            session_id=session_id,
            settings=settings,
        )
        self.vector_store_service = vector_store_service
        self.llm_cv_parser_service = LLMCVParserService(
            llm_service, settings, template_manager
        )

    async def _execute(self, **kwargs: Any) -> AgentResult:
        """Execute the core parsing logic."""
        input_data = kwargs.get("input_data", {})
        raw_text = input_data.get("raw_text")

        output = ParserAgentOutput()
        self.update_progress(AgentConstants.PROGRESS_MAIN_PROCESSING, "Parsing CV")
        parsed_data = await self.parse_cv(raw_text)
        output.structured_cv = parsed_data

        self.update_progress(AgentConstants.PROGRESS_COMPLETE, "Parsing completed")
        return AgentResult(success=True, output_data=output)

    async def parse_cv(self, raw_text: str) -> StructuredCV:
        """Parses a raw CV using an LLM and converts it to a structured format."""
        if not raw_text.strip():
            raise AgentExecutionError(
                agent_name=self.name, message="Cannot parse an empty CV."
            )
        try:
            self.update_progress(AgentConstants.PROGRESS_MAIN_PROCESSING, "Calling LLM for CV parsing")
            llm_output = await self.llm_cv_parser_service.parse_cv_with_llm(raw_text)

            self.update_progress(AgentConstants.PROGRESS_PARSING_COMPLETE, "Converting LLM output to structured format")
            # Create a StructuredCV directly from CVParsingResult
            structured_cv = self._convert_cv_parsing_result_to_structured_cv(
                llm_output, raw_text
            )

            self.update_progress(AgentConstants.PROGRESS_VECTOR_STORAGE, "Storing CV vectors")
            await self._store_cv_vectors(structured_cv)

            self.update_progress(AgentConstants.PROGRESS_COMPLETE, "Parsing completed")
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
