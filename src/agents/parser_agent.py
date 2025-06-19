from .agent_base import EnhancedAgentBase, AgentExecutionContext, AgentResult
from ..services.llm_service import get_llm_service
from ..services.vector_db import get_enhanced_vector_db
from .agent_base import EnhancedAgentBase
from ..models.data_models import (
    JobDescriptionData,
    StructuredCV,
    Section,
    Subsection,
    Item,
    ItemStatus,
    ItemType,
    AgentIO,
    PersonalInfo,
    Experience,
    Education,
    Skill,
    Project,
    Certification,
    Language,
    CVParsingResult,
    CVParsingPersonalInfo,
    CVParsingSection,
    CVParsingSubsection,
)
from ..config.logging_config import get_structured_logger
from ..models.data_models import AgentDecisionLog, AgentExecutionLog
from ..config.settings import get_config
from ..services.llm_service import LLMResponse
from ..orchestration.state import AgentState
from ..core.async_optimizer import optimize_async
from ..utils.exceptions import (
    LLMResponseParsingError,
    ValidationError,
    WorkflowPreconditionError,
    AgentExecutionError,
    ConfigurationError,
    StateManagerError,
)
from ..utils.agent_error_handling import (
    AgentErrorHandler,
    LLMErrorHandler,
    with_error_handling,
    with_node_error_handling,
)
import json  # Import json for parsing LLM output
from typing import List, Optional, Dict, Any, Union
import re  # For regex parsing of Markdown
import asyncio

# Set up structured logging
logger = get_structured_logger(__name__)


class ParserAgent(EnhancedAgentBase):
    """Agent responsible for parsing job descriptions and extracting key information, and parsing CVs into StructuredCV objects."""

    def __init__(self, name: str, description: str, llm_service=None, llm=None):
        self._job_data = {}  # Initialize job data storage
        """
        Initialize the ParserAgent.

        Args:
            name: The name of the agent.
            description: A description of the agent's purpose.
            llm_service: Optional LLM service instance. If None, will use get_llm_service().
            llm: Alternative parameter name for LLM service (for backward compatibility).
        """
        # Define input and output schemas
        input_schema = AgentIO(
            description="Reads raw text for job description and CV from the AgentState.",
            required_fields=[
                "job_description_data.raw_text",
                "structured_cv.metadata.original_cv_text",
            ],
            optional_fields=["start_from_scratch"],
        )
        output_schema = AgentIO(
            description="Populates the 'structured_cv' and 'job_description_data' fields in AgentState.",
            required_fields=["structured_cv", "job_description_data"],
            optional_fields=["error_messages"],
        )

        # Call parent constructor
        super().__init__(name, description, input_schema, output_schema)

        # Accept either llm_service or llm parameter
        self.llm = llm_service or llm or get_llm_service()

        # Initialize settings for prompt loading
        self.settings = get_config()

    # Removed _load_prompt method - now using centralized settings-based prompt loading

    async def parse_job_description(
        self, raw_text: str, trace_id: Optional[str] = None
    ) -> JobDescriptionData:
        """
        Parses a raw job description using an LLM and extracts key information.
        Uses robust JSON validation with Pydantic models.

        Args:
            raw_text: The raw job description as a string.

        Returns:
            A JobDescriptionData object with the parsed content.
        """
        if not raw_text:
            # Log validation decision with structured logging
            self.log_decision(
                "Empty job description provided, returning default structure",
                None,
                "validation",
            )

            return JobDescriptionData(
                raw_text=raw_text,
                skills=[],
                responsibilities=[],
                company_values=[],
                industry_terms=[],
                experience_level="N/A",
            )

        return await self._parse_job_description_with_llm(raw_text, trace_id=trace_id)

    async def _parse_job_description_with_llm(
        self, raw_text: str, trace_id: Optional[str] = None
    ) -> JobDescriptionData:
        """
        Internal method to parse job description using LLM.

        Args:
            raw_text: The raw job description as a string.

        Returns:
            A JobDescriptionData object with the parsed content.
        """
        # Load the updated prompt
        prompt_path = self.settings.get_prompt_path_by_key("job_description_parser")
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt_template = f.read()
        prompt = prompt_template.format(raw_text=raw_text)

        try:
            # Use centralized JSON generation and parsing
            parsed_data = await self._generate_and_parse_json(
                prompt=prompt,
                session_id=getattr(self, "current_session_id", None),
                trace_id=trace_id,
            )

            # Validate with Pydantic
            from ..models.validation_schemas import LLMJobDescriptionOutput

            validated_output = LLMJobDescriptionOutput.model_validate(parsed_data)

            # 5. Map validated data to the application's main data model
            job_data = JobDescriptionData(
                raw_text=raw_text,
                skills=validated_output.skills,
                experience_level=validated_output.experience_level,
                responsibilities=validated_output.responsibilities,
                industry_terms=validated_output.industry_terms,
                company_values=validated_output.company_values,
                status=ItemStatus.GENERATED,
            )
            logger.info(
                "Job description successfully parsed and validated using LLM-generated JSON."
            )
            return job_data

        except (json.JSONDecodeError, ValidationError, LLMResponseParsingError) as e:
            error_message = f"Failed to parse or validate LLM response for job description: {str(e)}"
            logger.error(error_message, exc_info=True)
            # Create a failed state object
            return JobDescriptionData(
                raw_text=raw_text,
                skills=[],
                responsibilities=[],
                company_values=[],
                industry_terms=[],
                experience_level="N/A",
                error=error_message,
                status=ItemStatus.GENERATION_FAILED,
            )

    # Removed _parse_job_description_with_regex method - replaced with LLM-first approach

    async def parse_cv_with_llm(
        self, cv_text: str, job_data: JobDescriptionData
    ) -> StructuredCV:
        """
        Parses CV text into a StructuredCV object using LLM-first approach.

        Args:
            cv_text: The raw CV text as a string.
            job_data: The parsed job description data.

        Returns:
            A StructuredCV object representing the parsed CV.
        """
        # Create a new StructuredCV with initial metadata
        structured_cv = self._initialize_structured_cv(cv_text, job_data)

        # If no CV text, return empty structure
        if not cv_text:
            logger.warning("Empty CV text provided to ParserAgent.")
            return structured_cv

        try:
            # Parse CV content using LLM
            parsing_result = await self._parse_cv_content_with_llm(cv_text)

            # Convert LLM result to StructuredCV format
            structured_cv = self._convert_parsing_result_to_structured_cv(
                parsing_result, cv_text, job_data
            )

            logger.info(
                f"Successfully parsed CV with {len(structured_cv.sections)} sections using LLM"
            )
            return structured_cv

        except Exception as e:
            logger.error(f"Failed to parse CV with LLM: {str(e)}")
            # Fallback to empty structure with error metadata
            structured_cv.metadata["parsing_error"] = str(e)
            return structured_cv

    def _initialize_structured_cv(
        self, cv_text: str, job_data: JobDescriptionData
    ) -> StructuredCV:
        """
        Initialize a StructuredCV object with metadata.

        Args:
            cv_text: The raw CV text as a string.
            job_data: The parsed job description data.

        Returns:
            A StructuredCV object with initialized metadata.
        """
        structured_cv = StructuredCV()

        # Add metadata - handle both dict and JobDescriptionData object types
        if job_data:
            if hasattr(job_data, "model_dump"):
                # Pydantic BaseModel object
                structured_cv.metadata["job_description"] = job_data.model_dump()
            elif isinstance(job_data, dict):
                # Already a dictionary
                structured_cv.metadata["job_description"] = job_data
            else:
                # Fallback for other types
                structured_cv.metadata["job_description"] = {}
        else:
            structured_cv.metadata["job_description"] = {}
        structured_cv.metadata["original_cv_text"] = cv_text

        return structured_cv

    async def _parse_cv_content_with_llm(self, cv_text: str) -> CVParsingResult:
        """
        Parse CV content using LLM and return structured result.

        Args:
            cv_text: The raw CV text as a string.

        Returns:
            A CVParsingResult object containing parsed CV data.

        Raises:
            Exception: If LLM parsing fails or validation errors occur.
        """
        # Load the CV parsing prompt using settings
        prompt_path = self.settings.get_prompt_path_by_key("cv_parser")
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt_template = f.read()
        prompt = prompt_template.replace("{{raw_cv_text}}", cv_text)

        # Use centralized LLM generation and JSON parsing
        parsing_data = await self._generate_and_parse_json(
            prompt=prompt,
            session_id=getattr(self, "current_session_id", None),
            trace_id=getattr(self, "current_trace_id", None),
        )

        # Convert dictionary to CVParsingResult object
        try:
            parsing_result = CVParsingResult(**parsing_data)
            logger.info(
                f"Successfully created CVParsingResult with personal_info: {parsing_result.personal_info}"
            )
            return parsing_result
        except Exception as parse_error:
            logger.error(f"Failed to create CVParsingResult: {parse_error}")
            logger.error(f"Parsing data: {parsing_data}")
            raise parse_error

    def _convert_parsing_result_to_structured_cv(
        self,
        parsing_result: CVParsingResult,
        cv_text: str,
        job_data: JobDescriptionData,
    ) -> StructuredCV:
        """
        Convert CVParsingResult to StructuredCV format.

        Args:
            parsing_result: The parsed result from LLM
            cv_text: The original CV text
            job_data: The job description data

        Returns:
            A StructuredCV object
        """
        try:
            # Create new StructuredCV with metadata
            structured_cv = self._create_structured_cv_with_metadata(cv_text, job_data)

            # Add personal info to metadata
            self._add_contact_info_to_metadata(structured_cv, parsing_result)

            # Convert sections
            self._convert_sections_to_structured_cv(structured_cv, parsing_result)

            return structured_cv
        except Exception as e:
            # Handle validation errors by returning empty CV with error metadata
            logger.error(f"Error converting parsing result to StructuredCV: {str(e)}")
            structured_cv = self._create_structured_cv_with_metadata(cv_text, job_data)
            structured_cv.metadata["error"] = str(e)
            return structured_cv

    def _create_structured_cv_with_metadata(
        self, cv_text: str, job_data: JobDescriptionData
    ) -> StructuredCV:
        """
        Create a StructuredCV object with basic metadata.

        Args:
            cv_text: The original CV text
            job_data: The job description data

        Returns:
            A StructuredCV object with initialized metadata
        """
        structured_cv = StructuredCV(sections=[])

        # Set metadata
        if isinstance(job_data, JobDescriptionData):
            structured_cv.metadata["job_description"] = job_data.model_dump()
        else:
            structured_cv.metadata["job_description"] = {}
        structured_cv.metadata["original_cv_text"] = cv_text

        return structured_cv

    def _add_contact_info_to_metadata(
        self, structured_cv: StructuredCV, parsing_result: CVParsingResult
    ) -> None:
        """
        Add personal contact information to StructuredCV metadata.

        Args:
            structured_cv: The StructuredCV object to update
            parsing_result: The parsed result containing personal info
        """
        try:
            personal_info = parsing_result.personal_info
            structured_cv.metadata.update(
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
            structured_cv.metadata["parsing_error"] = str(e)
            logger.error(f"Error accessing personal info: {e}")
            logger.error(f"Parsing result type: {type(parsing_result)}")
            logger.error(f"Parsing result: {parsing_result}")

    def _convert_sections_to_structured_cv(
        self, structured_cv: StructuredCV, parsing_result: CVParsingResult
    ) -> None:
        """
        Convert parsed sections to StructuredCV format.

        Args:
            structured_cv: The StructuredCV object to update
            parsing_result: The parsed result containing sections
        """
        section_order = 0
        for parsed_section in parsing_result.sections:
            # Determine if this is a dynamic section (to be tailored) or static section
            content_type = self._determine_section_content_type(parsed_section.name)

            section = Section(
                name=parsed_section.name, content_type=content_type, order=section_order
            )
            section_order += 1

            # Add direct items to section
            self._add_section_items(section, parsed_section, content_type)

            # Add subsections
            self._add_section_subsections(section, parsed_section, content_type)

            structured_cv.sections.append(section)

    def _determine_section_content_type(self, section_name: str) -> str:
        """
        Determine if a section is dynamic (tailorable) or static.

        Args:
            section_name: The name of the section

        Returns:
            "DYNAMIC" or "STATIC" content type
        """
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

    def _add_section_items(
        self, section: Section, parsed_section, content_type: str
    ) -> None:
        """
        Add items directly to a section.

        Args:
            section: The Section object to update
            parsed_section: The parsed section data
            content_type: "DYNAMIC" or "STATIC"
        """
        for item_content in parsed_section.items:
            if item_content.strip():
                # Determine item type based on section
                item_type = self._determine_item_type(parsed_section.name)

                # Determine status (dynamic sections start as INITIAL, static as STATIC)
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
        self, section: Section, parsed_section, content_type: str
    ) -> None:
        """
        Add subsections to a section.

        Args:
            section: The Section object to update
            parsed_section: The parsed section data
            content_type: "DYNAMIC" or "STATIC"
        """
        for parsed_subsection in parsed_section.subsections:
            subsection = Subsection(name=parsed_subsection.name)

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

    def _determine_item_type(self, section_name: str) -> ItemType:
        """
        Determine the appropriate ItemType based on section name.

        Args:
            section_name: The name of the section

        Returns:
            The appropriate ItemType
        """
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

    def parse_cv_text(self, cv_text: str, job_data: JobDescriptionData) -> StructuredCV:
        """
        Synchronous wrapper for parse_cv_with_llm for backward compatibility.

        Args:
            cv_text: The raw CV text as a string.
            job_data: The parsed job description data.

        Returns:
            A StructuredCV object representing the parsed CV.
        """
        try:
            # Check if we're already in an async context
            loop = asyncio.get_running_loop()
            # If we're in an async context, we need to handle this differently
            logger.warning(
                "parse_cv_text called in async context, consider using parse_cv_with_llm directly"
            )
            # Create a task for the async method
            task = asyncio.create_task(self.parse_cv_with_llm(cv_text, job_data))
            return asyncio.run_coroutine_threadsafe(task, loop).result()
        except RuntimeError:
            # No event loop running, we can create one
            return asyncio.run(self.parse_cv_with_llm(cv_text, job_data))

        # The old regex-based parsing logic has been replaced with LLM-first approach
        # This method now serves as a backward compatibility wrapper
        pass

    # Old LLM enhancement method removed - replaced with LLM-first parsing

    # Old section-specific LLM parsing methods removed - replaced with unified LLM-first parsing

    def create_empty_cv_structure(self, job_data: JobDescriptionData) -> StructuredCV:
        """
        Creates an empty CV structure for the "Start from Scratch" option.

        Args:
            job_data: The parsed job description data.

        Returns:
            A StructuredCV object with empty sections.
        """
        # Create a new StructuredCV
        structured_cv = StructuredCV()

        # Add metadata - handle both dict and JobDescriptionData object types
        if job_data:
            if hasattr(job_data, "to_dict"):
                # JobDescriptionData object
                structured_cv.metadata["job_description"] = job_data.to_dict()
            elif isinstance(job_data, dict):
                # Already a dictionary
                structured_cv.metadata["job_description"] = job_data
            else:
                # Fallback for other types
                structured_cv.metadata["job_description"] = {}
        else:
            structured_cv.metadata["job_description"] = {}
        structured_cv.metadata["start_from_scratch"] = True

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

        # Create and add the sections
        for section_info in sections:
            section = Section(
                name=section_info["name"],
                content_type=section_info["type"],
                order=section_info["order"],
            )

            # For Executive Summary, add an empty item
            if section.name == "Executive Summary":
                section.items.append(  # pylint: disable=no-member
                    Item(
                        content="",
                        status=ItemStatus.TO_REGENERATE,
                        item_type=ItemType.EXECUTIVE_SUMMARY_PARA,
                    )
                )

            # For Key Qualifications, add empty items based on skills from job description
            if section.name == "Key Qualifications":
                skills = None
                if job_data:
                    if hasattr(job_data, "skills"):
                        skills = job_data.skills
                    elif isinstance(job_data, dict) and "skills" in job_data:
                        skills = job_data["skills"]

                if skills:
                    # Limit to 8 skills maximum
                    for skill in skills[:8]:
                        section.items.append(  # pylint: disable=no-member
                            Item(
                                content=skill,
                                status=ItemStatus.TO_REGENERATE,
                                item_type=ItemType.KEY_QUALIFICATION,
                            )
                        )
                else:
                    # Add default empty items if no skills available
                    for i in range(6):  # Default 6 qualifications
                        section.items.append(  # pylint: disable=no-member
                            Item(
                                content=f"Key qualification {i+1}",
                                status=ItemStatus.TO_REGENERATE,
                                item_type=ItemType.KEY_QUALIFICATION,
                            )
                        )

            # For Professional Experience, add an empty subsection
            if section.name == "Professional Experience":
                subsection = Subsection(name="Position Title at Company Name")
                # Add some default bullet points
                for _ in range(3):
                    subsection.items.append(  # pylint: disable=no-member
                        Item(
                            content="",
                            status=ItemStatus.TO_REGENERATE,
                            item_type=ItemType.BULLET_POINT,
                        )
                    )
                section.subsections.append(subsection)  # pylint: disable=no-member

            # For Project Experience, add an empty subsection
            if section.name == "Project Experience":
                subsection = Subsection(name="Project Name")
                # Add some default bullet points
                for _ in range(2):
                    subsection.items.append(  # pylint: disable=no-member
                        Item(
                            content="",
                            status=ItemStatus.TO_REGENERATE,
                            item_type=ItemType.BULLET_POINT,
                        )
                    )
                section.subsections.append(subsection)  # pylint: disable=no-member

            structured_cv.sections.append(section)  # pylint: disable=no-member

        return structured_cv

    def get_job_data(self) -> Dict[str, Any]:
        """
        Returns the previously parsed job description data.

        Returns:
            The parsed job description data as a dictionary, or an empty dict if no job has been parsed
        """
        if hasattr(self, "_job_data") and self._job_data:
            return self._job_data
        return {}

    def get_confidence_score(self, output_data: Any) -> float:
        """
        Returns a confidence score for the agent's output based on data completeness.

        Args:
            output_data: The output data generated by the agent.

        Returns:
            A float representing the confidence score (0.0 to 1.0).
        """
        if not output_data:
            return 0.0

        # Base confidence score
        confidence = 0.3

        # Check for required fields in job description data
        required_fields = [
            "job_title",
            "required_skills",
            "experience_level",
            "education_requirements",
        ]
        present_fields = 0

        for field in required_fields:
            if field in output_data and output_data[field]:
                present_fields += 1

        # Calculate confidence based on completeness
        field_completeness = present_fields / len(required_fields)
        confidence += field_completeness * 0.7

        return min(confidence, 1.0)

    # Removed _extract_skills_from_text method - replaced with LLM-first approach

    # Removed _extract_sections, _is_section_header, and parse_cv methods - replaced with LLM-first approach

    async def run_async(
        self, input_data: Any, context: "AgentExecutionContext"
    ) -> "AgentResult":
        """
        Legacy async method required by abstract base class.

        This method provides compatibility with the abstract base class
        but is not the primary execution path for LangGraph workflows.
        """
        from .agent_base import AgentResult
        from ..orchestration.state import AgentState
        from ..models.data_models import JobDescriptionData, StructuredCV

        try:
            # Simple implementation that returns a basic result
            # The main execution path is through run_as_node
            return AgentResult(
                success=True,
                output_data={
                    "message": "ParserAgent executed via run_async - use run_as_node for full functionality"
                },
                confidence_score=1.0,
                metadata={"agent_type": "parser", "execution_method": "run_async"},
            )

        except Exception as e:
            # Use standardized error handling
            fallback_data = AgentErrorHandler.create_fallback_data("parser")
            return AgentErrorHandler.handle_general_error(
                e, "parser", fallback_data, "run_async"
            )

    @optimize_async("agent_execution", "parser")
    async def run_as_node(self, state: AgentState) -> dict:
        """
        Executes the complete parsing logic as a LangGraph node.

        This method now correctly handles parsing the job description,
        parsing an existing CV text from metadata, or creating an empty CV
        structure based on the initial state, ensuring the agent's full
        capability is exposed to the workflow.
        """
        logger.info(
            "ParserAgent node running with consolidated logic.",
            extra={"trace_id": state.trace_id},
        )

        try:
            # Set current session and trace IDs for centralized JSON parsing
            self.current_session_id = getattr(state, "session_id", None)
            self.current_trace_id = state.trace_id

            # Initialize job_data to None
            job_data = None

            # 1. Always parse the job description first, if it exists.
            if state.job_description_data and state.job_description_data.raw_text:
                logger.info(
                    "Starting job description parsing.",
                    extra={"trace_id": state.trace_id},
                )
                job_data = await self.parse_job_description(
                    state.job_description_data.raw_text, trace_id=state.trace_id
                )
            else:
                logger.warning(
                    "No job description text found in the state.",
                    extra={"trace_id": state.trace_id},
                )
                # Create empty JobDescriptionData if none exists
                job_data = JobDescriptionData(raw_text="")

            # 2. Determine the CV processing path from the structured_cv metadata.
            # This metadata is set by the create_initial_agent_state function.
            cv_metadata = state.structured_cv.metadata if state.structured_cv else {}
            start_from_scratch = cv_metadata.get("start_from_scratch", False)
            original_cv_text = cv_metadata.get("original_cv_text", "")

            final_cv = None
            if start_from_scratch:
                logger.info(
                    "Creating empty CV structure for 'Start from Scratch' option.",
                    extra={"trace_id": state.trace_id},
                )
                final_cv = self.create_empty_cv_structure(job_data)
            elif original_cv_text:
                logger.info(
                    "Parsing provided CV text with LLM-first approach.",
                    extra={"trace_id": state.trace_id},
                )
                final_cv = await self.parse_cv_with_llm(original_cv_text, job_data)
            else:
                logger.warning(
                    "No CV text provided and not starting from scratch. Passing CV state through."
                )
                final_cv = state.structured_cv

            # 3. Return the complete, updated state.
            return {"structured_cv": final_cv, "job_description_data": job_data}

        except Exception as e:
            # Use standardized error handling for node execution
            return AgentErrorHandler.handle_node_error(
                e, "parser", state, "run_as_node"
            )
