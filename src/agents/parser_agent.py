"""
This module defines the ParserAgent, responsible for parsing CVs and job descriptions.
"""

from typing import Any, Dict, List, Optional

from ..config.logging_config import get_structured_logger
from ..config.settings import AgentSettings, get_config
from ..models.data_models import (
    Item,
    ItemStatus,
    JobDescriptionData,
    StructuredCV,
    Section,
    AgentIO,
)
from ..models.agent_output_models import ParserAgentOutput
from ..models.validation_schemas import validate_agent_input
from ..orchestration.state import AgentState
from ..services.llm_cv_parser_service import LLMCVParserService
from ..services.llm_service import EnhancedLLMService
from ..services.progress_tracker import ProgressTracker
from ..services.vector_store_service import VectorStoreService
from ..templates.content_templates import ContentTemplateManager
from ..utils.exceptions import AgentExecutionError
from .agent_base import EnhancedAgentBase, AgentExecutionContext, AgentResult
from .cv_conversion_utils import convert_parsing_result_to_structured_cv
from .cv_structure_utils import create_empty_cv_structure

# Set up structured logging
logger = get_structured_logger(__name__)


# pylint: disable=too-many-arguments, too-many-positional-arguments
class ParserAgent(EnhancedAgentBase):
    """Agent responsible for parsing job descriptions and CVs."""

    def __init__(
        self,
        llm_service: EnhancedLLMService,
        vector_store_service: VectorStoreService,
        progress_tracker: ProgressTracker,
        settings: AgentSettings,
        template_manager: ContentTemplateManager,
        name: str = "ParserAgent",
        description: str = ("Parses job descriptions and CVs into structured data."),
    ):
        """Initialize the ParserAgent with required dependencies."""
        input_schema = AgentIO(
            description="Job description or CV text to parse",
            required_fields=["raw_text"],
        )
        output_schema = AgentIO(
            description="Parsed job or CV data",
            required_fields=["parsed_data"],
        )
        super().__init__(
            name=name,
            description=description,
            input_schema=input_schema,
            output_schema=output_schema,
            progress_tracker=progress_tracker,
        )
        self.llm_service = llm_service
        self.vector_store_service = vector_store_service
        self.template_manager = template_manager
        self.settings = settings
        self.llm_cv_parser_service = LLMCVParserService(
            llm_service, settings, template_manager
        )
        self.current_session_id: Optional[str] = None
        self.current_trace_id: Optional[str] = None

    async def parse_job_description(
        self, raw_text: str, trace_id: Optional[str] = None
    ) -> JobDescriptionData:
        """
        Parses a raw job description using an LLM and extracts key information.
        """
        if not raw_text:
            self.log_decision(
                "Empty job description provided, returning default structure.",
                None,
                "validation",
            )
            return JobDescriptionData(raw_text=raw_text)

        try:
            return await self.llm_cv_parser_service.parse_job_description_with_llm(
                raw_text,
                session_id=self.current_session_id,
                trace_id=trace_id,
            )
        except AgentExecutionError as e:
            logger.error("Failed to parse job description: %s", e)
            return JobDescriptionData(
                raw_text=raw_text,
                error=str(e),
                status=ItemStatus.GENERATION_FAILED,
            )

    async def parse_cv_with_llm(
        self, cv_text: str, job_data: JobDescriptionData
    ) -> StructuredCV:
        """
        Parses CV text into a StructuredCV object using an LLM-first approach.
        """
        structured_cv = self._initialize_structured_cv(cv_text, job_data)
        if not cv_text:
            logger.warning("Empty CV text provided to ParserAgent.")
            return structured_cv

        try:
            parsing_result = await self.llm_cv_parser_service.parse_cv_with_llm(
                cv_text,
                session_id=self.current_session_id,
                trace_id=self.current_trace_id,
            )
            structured_cv = convert_parsing_result_to_structured_cv(
                parsing_result, cv_text, job_data
            )
            logger.info(
                "Successfully parsed CV with %d sections using LLM",
                len(structured_cv.sections),
            )
            return structured_cv
        except AgentExecutionError as e:
            logger.error("Failed to parse CV with LLM: %s", e)
            structured_cv.metadata["parsing_error"] = str(e)
            return structured_cv

    def _initialize_structured_cv(
        self, cv_text: str, job_data: JobDescriptionData
    ) -> StructuredCV:
        """
        Initialize a StructuredCV object with metadata.
        """
        structured_cv = StructuredCV()
        if job_data:
            structured_cv.metadata["job_description"] = job_data.model_dump()
        else:
            structured_cv.metadata["job_description"] = {}
        structured_cv.metadata["original_cv_text"] = cv_text
        return structured_cv

    def _parse_bullet_points(self, content: str) -> List[str]:
        """
        Parse generated content into individual bullet points.
        """
        config = get_config()
        lines = content.strip().split("\n")
        bullet_points = []
        for line in lines:
            clean_line = line.strip().lstrip("-•* ").strip()
            if clean_line:
                bullet_points.append(clean_line)
        if not bullet_points and content.strip():
            return [content.strip()]
        return bullet_points[: config.output.max_bullet_points_per_role]

    def create_empty_cv_structure(self, job_data: JobDescriptionData) -> StructuredCV:
        """
        Creates an empty CV structure for the "Start from Scratch" option.
        """
        return create_empty_cv_structure(job_data)

    async def run_async(
        self, input_data: Any, context: AgentExecutionContext
    ) -> AgentResult:
        """
        Execute the parser agent asynchronously.

        Args:
            input_data: Input data containing raw text to parse
            context: Agent execution context

        Returns:
            AgentResult with parsed data or error information
        """
        try:
            # Validate input
            if not isinstance(input_data, dict):
                error_msg = f"Expected dict input, got {type(input_data)}"
                return AgentResult(
                    success=False,
                    result=None,
                    error=error_msg,
                    trace_id=context.trace_id,
                )

            raw_text = input_data.get("raw_text")
            if not raw_text:
                error_msg = "raw_text is required for parsing"
                return AgentResult(
                    success=False,
                    result=None,
                    error=error_msg,
                    trace_id=context.trace_id,
                )  # Parse job description            # Parse job description
            job_data = await self.parse_job_description(raw_text, context.trace_id)

            # Create proper Pydantic output model
            output_data = ParserAgentOutput(job_description_data=job_data)

            # Return successful result
            return AgentResult(
                success=True,
                output_data=output_data,
                error_message=None,
            )

        except Exception as e:
            logger.error(
                "ParserAgent run_async failed: %s",
                e,
                extra={"trace_id": context.trace_id},
            )
            # Return error result with empty output data (using Pydantic model)
            return AgentResult(
                success=False,
                output_data=ParserAgentOutput(),
                error_message=str(e),
            )

    async def run_as_node(self, state: AgentState) -> Dict[str, Any]:
        """Main execution method for the agent node. Fails fast on errors."""
        try:
            logger.info(
                "ParserAgent node running.",
                extra={"trace_id": state.trace_id},
            )
            self._initialize_run(state)

            job_data = await self._process_job_description(state)
            final_cv = await self._process_cv(state, job_data)
            final_cv = self._ensure_current_item_exists(final_cv, state)

            return {"structured_cv": final_cv, "job_description_data": job_data}
        except Exception as e:
            logger.error(
                "ParserAgent failed: %s", e, extra={"trace_id": state.trace_id}
            )
            # Fail fast - let the orchestration layer handle recovery
            raise AgentExecutionError(agent_name="ParserAgent", message=str(e)) from e

    def _initialize_run(self, state: AgentState):
        """Initializes the agent run by setting IDs and validating input."""
        if getattr(self, "_dependency_error", None):
            raise AgentExecutionError(
                "ParserAgent", f"dependency error: {self._dependency_error}"
            )
        self.current_session_id = getattr(state, "session_id", None)
        self.current_trace_id = state.trace_id
        validate_agent_input("parser", state)

    async def _process_job_description(self, state: AgentState) -> JobDescriptionData:
        """Parses the job description from the state."""
        logger.info("Processing job description.", extra={"trace_id": state.trace_id})
        raw_text = getattr(state.job_description_data, "raw_text", "")
        if not raw_text:
            logger.warning(
                "No job description text found in state.",
                extra={"trace_id": state.trace_id},
            )
            return JobDescriptionData(raw_text="")

        job_data = await self.parse_job_description(raw_text, trace_id=state.trace_id)

        update_fields = {}
        if not getattr(job_data, "company_name", None):
            update_fields["company_name"] = "Unknown Company"
        if not getattr(job_data, "title", None):
            update_fields["title"] = "Unknown Title"
        if update_fields:
            job_data = job_data.model_copy(update=update_fields)
        return job_data

    async def _process_cv(
        self, state: AgentState, job_data: JobDescriptionData
    ) -> StructuredCV:
        """Determines the CV processing path and executes it."""
        logger.info("Processing CV.", extra={"trace_id": state.trace_id})
        cv_meta = state.structured_cv.metadata if state.structured_cv else {}
        start_from_scratch = cv_meta.get("extra", {}).get("start_from_scratch", False)
        original_cv_text = cv_meta.get("extra", {}).get("original_cv_text", "")

        if start_from_scratch:
            logger.info(
                "Creating empty CV for 'Start from Scratch'.",
                extra={"trace_id": state.trace_id},
            )
            return self.create_empty_cv_structure(job_data)
        if original_cv_text:
            logger.info("Parsing provided CV text.", extra={"trace_id": state.trace_id})
            return await self.parse_cv_with_llm(original_cv_text, job_data)

        logger.warning(
            "No CV text and not starting from scratch. Passing CV through.",
            extra={"trace_id": state.trace_id},
        )
        return state.structured_cv or self.create_empty_cv_structure(job_data)

    def _ensure_current_item_exists(
        self, final_cv: StructuredCV, state: AgentState
    ) -> StructuredCV:
        """Ensures current_item_id from state exists in the CV."""
        current_item_id = getattr(state, "current_item_id", None)
        if not current_item_id:
            return final_cv

        logger.info(
            "Ensuring item with ID %s exists in CV.",
            current_item_id,
            extra={"trace_id": state.trace_id},
        )

        if final_cv and any(
            str(item.id) == str(current_item_id)
            for section in final_cv.sections
            for item in section.items
        ):
            return final_cv

        logger.warning(
            "Item with ID %s not found. Adding a placeholder.",
            current_item_id,
            extra={"trace_id": state.trace_id},
        )

        if not final_cv:
            final_cv = self.create_empty_cv_structure(JobDescriptionData(raw_text=""))
        if not final_cv.sections:
            final_cv.sections.append(Section(name="Auto-Generated Section"))

        # Add a placeholder item
        # (Assuming a default section and item structure)
        # This part might need more robust logic based on application needs
        placeholder_item = Item(id=current_item_id, content="Placeholder")
        final_cv.sections[0].items.append(placeholder_item)

        return final_cv
