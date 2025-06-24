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
    Subsection,
    ItemType,
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
            logger.error(f"Failed to parse job description: {e}")
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
            structured_cv = self._convert_parsing_result_to_structured_cv(
                parsing_result, cv_text, job_data
            )
            logger.info(
                f"Successfully parsed CV with {len(structured_cv.sections)} sections using LLM"
            )
            return structured_cv
        except AgentExecutionError as e:
            logger.error(f"Failed to parse CV with LLM: {e}")
            structured_cv.metadata.extra["parsing_error"] = str(e)
            return structured_cv

    def _initialize_structured_cv(
        self, cv_text: str, job_data: JobDescriptionData
    ) -> StructuredCV:
        """
        Initialize a StructuredCV object with metadata.
        """
        structured_cv = StructuredCV()
        if job_data:
            structured_cv.metadata.extra["job_description"] = job_data.model_dump()
        else:
            structured_cv.metadata.extra["job_description"] = {}
        structured_cv.metadata.extra["original_cv_text"] = cv_text
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

    def _create_structured_cv_with_metadata(
        self, cv_text: str, job_data: JobDescriptionData
    ) -> StructuredCV:
        structured_cv = StructuredCV(sections=[])
        if hasattr(job_data, "model_dump"):
            structured_cv.metadata.extra["job_description"] = job_data.model_dump()
        elif isinstance(job_data, dict):
            structured_cv.metadata.extra["job_description"] = job_data
        else:
            structured_cv.metadata.extra["job_description"] = {}
        structured_cv.metadata.extra["original_cv_text"] = cv_text
        return structured_cv

    def _add_contact_info_to_metadata(
        self, structured_cv: StructuredCV, parsing_result: Any
    ) -> None:
        try:
            personal_info = parsing_result.personal_info
            structured_cv.metadata.extra.update(
                {
                    "name": personal_info.name,
                    "email": personal_info.email,
                    "phone": personal_info.phone,
                    "linkedin": personal_info.linkedin,
                    "github": personal_info.github,
                    "location": personal_info.location,
                }
            )
        except Exception as e:
            structured_cv.metadata.extra["parsing_error"] = str(e)
            logger.error(f"Error accessing personal info: {e}")
            logger.error(f"Parsing result type: {type(parsing_result)}")
            logger.error(f"Parsing result: {parsing_result}")

    def _determine_section_content_type(self, section_name: str) -> str:
        dynamic_sections = [
            "Executive Summary",
            "Key Qualifications",
            "Professional Experience",
            "Project Experience",
        ]
        return (
            "DYNAMIC"
            if any(section_name.lower() == s.lower() for s in dynamic_sections)
            else "STATIC"
        )

    def _determine_item_type(self, section_name: str):
        section_lower = section_name.lower()
        if "qualification" in section_lower or "skill" in section_lower:
            return ItemType.KEY_QUALIFICATION
        elif "executive" in section_lower or "summary" in section_lower:
            return ItemType.EXECUTIVE_SUMMARY_PARA
        elif "education" in section_lower:
            return ItemType.EDUCATION_ENTRY
        elif "certification" in section_lower:
            return ItemType.CERTIFICATION_ENTRY
        elif "language" in section_lower:
            return ItemType.LANGUAGE_ENTRY
        else:
            return ItemType.BULLET_POINT

    def _add_section_items(
        self, section: Section, parsed_section: Any, content_type: str
    ) -> None:
        # Ensure section.items is a list (handle FieldInfo edge case)
        if not isinstance(section.items, list):
            section.items = list(section.items) if section.items else []
        for item_content in parsed_section.items:
            if item_content.strip():
                item_type = self._determine_item_type(parsed_section.name)
                status = (
                    ItemStatus.INITIAL
                    if content_type == "DYNAMIC"
                    else ItemStatus.STATIC
                )
                item = Item(
                    content=item_content.strip(), status=status, item_type=item_type
                )
                section.items.append(item)

    def _add_section_subsections(
        self, section: Section, parsed_section: Any, content_type: str
    ) -> None:
        # Ensure section.subsections is a list (handle FieldInfo edge case)
        if not isinstance(section.subsections, list):
            section.subsections = (
                list(section.subsections) if section.subsections else []
            )
        for parsed_subsection in getattr(parsed_section, "subsections", []):
            subsection = Subsection(name=parsed_subsection.name, items=[])
            # Ensure subsection.items is a list
            if not isinstance(subsection.items, list):
                subsection.items = list(subsection.items) if subsection.items else []
            for item_content in parsed_subsection.items:
                if item_content.strip():
                    item_type = self._determine_item_type(parsed_section.name)
                    status = (
                        ItemStatus.INITIAL
                        if content_type == "DYNAMIC"
                        else ItemStatus.STATIC
                    )
                    item = Item(
                        content=item_content.strip(), status=status, item_type=item_type
                    )
                    subsection.items.append(item)
            section.subsections.append(subsection)

    def _convert_sections_to_structured_cv(
        self, structured_cv: StructuredCV, parsing_result: Any
    ) -> None:
        section_order = 0
        for parsed_section in parsing_result.sections:
            content_type = self._determine_section_content_type(parsed_section.name)
            section = Section(
                name=parsed_section.name, content_type=content_type, order=section_order
            )
            section_order += 1
            self._add_section_items(section, parsed_section, content_type)
            self._add_section_subsections(section, parsed_section, content_type)
            structured_cv.sections.append(section)

    def _convert_parsing_result_to_structured_cv(
        self, parsing_result: Any, cv_text: str, job_data: JobDescriptionData
    ) -> StructuredCV:
        try:
            structured_cv = self._create_structured_cv_with_metadata(cv_text, job_data)
            self._add_contact_info_to_metadata(structured_cv, parsing_result)
            self._convert_sections_to_structured_cv(structured_cv, parsing_result)
            return structured_cv
        except Exception as e:
            logger.error(f"Error converting parsing result to StructuredCV: {e}")
            structured_cv = self._create_structured_cv_with_metadata(cv_text, job_data)
            structured_cv.metadata.extra["error"] = str(e)
            return structured_cv

    def create_empty_cv_structure(
        self, job_data: Optional[JobDescriptionData]
    ) -> StructuredCV:
        """
        Creates an empty CV structure for the "Start from Scratch" option.
        """
        structured_cv = StructuredCV()
        # Add metadata - handle both dict and JobDescriptionData object types
        if job_data:
            if hasattr(job_data, "to_dict"):
                structured_cv.metadata.extra["job_description"] = job_data.to_dict()
            elif hasattr(job_data, "model_dump"):
                structured_cv.metadata.extra["job_description"] = job_data.model_dump()
            elif isinstance(job_data, dict):
                structured_cv.metadata.extra["job_description"] = job_data
            else:
                structured_cv.metadata.extra["job_description"] = {}
        else:
            structured_cv.metadata.extra["job_description"] = {}
        structured_cv.metadata.extra["start_from_scratch"] = True

        # Create standard CV sections with proper order
        sections = [
            {"name": "Executive Summary", "type": "DYNAMIC", "order": 0},
            {"name": "Key Qualifications", "type": "DYNAMIC", "order": 1},
            {"name": "Professional Experience", "type": "DYNAMIC", "order": 2},
            {"name": "Project Experience", "type": "DYNAMIC", "order": 3},
            {"name": "Education", "type": "STATIC", "order": 4},
            {"name": "Certifications", "type": "STATIC", "order": 5},
            {"name": "Languages", "type": "STATIC", "order": 6},
        ]

        for section_info in sections:
            section = Section(
                name=section_info["name"],
                content_type=section_info["type"],
                order=section_info["order"],
                items=[],
                subsections=[],
            )
            # Ensure section.items and section.subsections are lists
            if not isinstance(section.items, list):
                section.items = list(section.items) if section.items else []
            if not isinstance(section.subsections, list):
                section.subsections = (
                    list(section.subsections) if section.subsections else []
                )
            if section.name == "Executive Summary":
                section.items.append(
                    Item(
                        content="",
                        status=ItemStatus.TO_REGENERATE,
                        item_type=Item.ItemType.EXECUTIVE_SUMMARY_PARA,
                    )
                )
            if section.name == "Key Qualifications":
                skills = None
                if job_data:
                    if hasattr(job_data, "skills"):
                        skills = job_data.skills
                    elif isinstance(job_data, dict) and "skills" in job_data:
                        skills = job_data["skills"]
                if skills:
                    for skill in skills[:8]:
                        section.items.append(
                            Item(
                                content=skill,
                                status=ItemStatus.TO_REGENERATE,
                                item_type=Item.ItemType.KEY_QUALIFICATION,
                            )
                        )
                else:
                    for i in range(6):
                        section.items.append(
                            Item(
                                content=f"Key qualification {i+1}",
                                status=ItemStatus.TO_REGENERATE,
                                item_type=Item.ItemType.KEY_QUALIFICATION,
                            )
                        )
            if section.name == "Professional Experience":
                subsection = Subsection(name="Position Title at Company Name", items=[])
                if not isinstance(subsection.items, list):
                    subsection.items = (
                        list(subsection.items) if subsection.items else []
                    )
                for _ in range(3):
                    subsection.items.append(
                        Item(
                            content="",
                            status=ItemStatus.TO_REGENERATE,
                            item_type=Item.ItemType.BULLET_POINT,
                        )
                    )
                section.subsections.append(subsection)
            if section.name == "Project Experience":
                subsection = Subsection(name="Project Name", items=[])
                if not isinstance(subsection.items, list):
                    subsection.items = (
                        list(subsection.items) if subsection.items else []
                    )
                for _ in range(2):
                    subsection.items.append(
                        Item(
                            content="",
                            status=ItemStatus.TO_REGENERATE,
                            item_type=Item.ItemType.BULLET_POINT,
                        )
                    )
                # Final check before append
                if not isinstance(section.subsections, list):
                    section.subsections = (
                        list(section.subsections) if section.subsections else []
                    )
                section.subsections.append(subsection)
            # Ensure structured_cv.sections is a list before appending
            if not isinstance(structured_cv.sections, list):
                structured_cv.sections = (
                    list(structured_cv.sections) if structured_cv.sections else []
                )
            structured_cv.sections.append(section)
        return structured_cv

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
                f"ParserAgent run_async failed: {e}",
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
            logger.error(f"ParserAgent failed: {e}", extra={"trace_id": state.trace_id})
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
            f"Ensuring item with ID {current_item_id} exists in CV.",
            extra={"trace_id": state.trace_id},
        )

        if final_cv and any(
            str(item.id) == str(current_item_id)
            for section in final_cv.sections
            for item in section.items
        ):
            return final_cv

        logger.warning(
            f"Item with ID {current_item_id} not found. Adding a placeholder.",
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
