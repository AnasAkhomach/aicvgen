"""
This module defines the KeyQualificationsWriterAgent, responsible for generating the Key Qualifications section of the CV.
"""

from typing import Any, Dict, Union

from ..models.agent_models import AgentResult
from ..models.agent_output_models import EnhancedContentWriterOutput
from ..models.data_models import (
    ContentType,
    JobDescriptionData,
    StructuredCV,
)
from ..orchestration.state import AgentState
from ..services.llm_service import EnhancedLLMService
from ..templates.content_templates import ContentTemplateManager
from ..utils.cv_data_factory import get_item_by_id, update_item_by_id, add_item_to_section

from ..config.logging_config import get_structured_logger
from ..error_handling.exceptions import AgentExecutionError
from .agent_base import AgentBase
from ..models.cv_models import Item, ItemStatus, ItemType

logger = get_structured_logger(__name__)


class KeyQualificationsWriterAgent(AgentBase):
    """Agent for generating tailored Key Qualifications content."""

    def __init__(
        self,
        llm_service: EnhancedLLMService,
        template_manager: ContentTemplateManager,
        settings: dict,
        session_id: str,
    ):
        """Initialize the Key Qualifications writer agent."""
        super().__init__(
            name="KeyQualificationsWriter",
            description="Generates tailored Key Qualifications for the CV.",
            session_id=session_id,
        )
        self.llm_service = llm_service
        self.template_manager = template_manager
        self.settings = settings

    async def run_as_node(self, state: Union[AgentState, dict]) -> dict:
        """Execute the agent as a LangGraph node, returning a dictionary."""
        if hasattr(state, 'model_dump'):
            state_dict = state.model_dump()
        else:
            state_dict = state.copy() if isinstance(state, dict) else {}

        try:
            result = await self.run(
                structured_cv=state_dict.get('structured_cv'),
                job_description_data=state_dict.get('job_description_data'),
                research_findings=state_dict.get('research_findings')
            )

            if result.success:
                state_dict["structured_cv"] = result.output_data.updated_structured_cv
                if "error_messages" not in state_dict:
                    state_dict["error_messages"] = []
            else:
                if "error_messages" not in state_dict:
                    state_dict["error_messages"] = []
                state_dict["error_messages"].append(result.error_message)

        except Exception as e:
            logger.error(f"KeyQualificationsWriterAgent execution failed: {str(e)}")
            if "error_messages" not in state_dict:
                state_dict["error_messages"] = []
            state_dict["error_messages"].append(str(e))

        return state_dict

    def _validate_inputs(self, input_data: dict) -> None:
        """Validate inputs for Key Qualifications writer agent."""

        if "structured_cv" not in input_data or input_data["structured_cv"] is None:
            raise AgentExecutionError(
                agent_name=self.name,
                message="Missing or invalid 'structured_cv' in input_data.",
            )
        if "job_description_data" not in input_data or input_data["job_description_data"] is None:
            raise AgentExecutionError(
                agent_name=self.name,
                message="Missing or invalid 'job_description_data' in input_data.",
            )
        
        # Convert dict back to Pydantic objects if needed
        if isinstance(input_data["structured_cv"], dict):
            input_data["structured_cv"] = StructuredCV(**input_data["structured_cv"])
        if isinstance(input_data["job_description_data"], dict):
            input_data["job_description_data"] = JobDescriptionData(**input_data["job_description_data"])

    async def _execute(self, **kwargs: Any) -> AgentResult:
        """Execute the core content generation logic for Key Qualifications."""
        structured_cv: StructuredCV = kwargs.get("structured_cv")
        job_description_data: JobDescriptionData = kwargs.get("job_description_data")
        research_findings = kwargs.get("research_findings")

        self._validate_inputs({"structured_cv": structured_cv, "job_description_data": job_description_data})

        self.update_progress(40, "Generating Key Qualifications content.")
        generated_qualifications = await self._generate_key_qualifications(
            structured_cv, job_description_data, research_findings
        )

        self.update_progress(
            80, "Updating CV with generated Key Qualifications."
        )

        # Find the Key Qualifications section or create it if it doesn't exist
        qual_section = None
        for section in structured_cv.sections:
            if section.name == "Key Qualifications":
                qual_section = section
                break

        if not qual_section:
            # If section doesn't exist, create a placeholder. This might need refinement
            # based on how structured_cv is initialized.
            # For now, we'll assume it exists or is handled upstream.
            raise AgentExecutionError(
                agent_name=self.name,
                message="Key Qualifications section not found in structured_cv. It should be pre-initialized.",
            )

        # Clear existing items and add new ones
        qual_section.items = [
            Item(
                content=qual,
                status=ItemStatus.GENERATED,
                item_type=ItemType.KEY_QUALIFICATION,
            )
            for qual in generated_qualifications
        ]

        output_data = EnhancedContentWriterOutput(
            updated_structured_cv=structured_cv,
            item_id="key_qualifications_section",  # Placeholder ID for the section
            generated_content="; ".join(generated_qualifications),
        )

        self.update_progress(100, "Key Qualifications generation completed successfully.")
        return AgentResult(
            success=True,
            output_data=output_data,
            metadata={
                "agent_name": self.name,
                "message": "Successfully generated Key Qualifications.",
            },
        )

    async def _generate_key_qualifications(
        self,
        structured_cv: StructuredCV,
        job_data: JobDescriptionData,
        research_findings: Dict[str, Any] | None,
    ) -> list[str]:
        """Generates key qualifications using an LLM."""
        prompt_template = self.template_manager.get_template_by_type(ContentType.QUALIFICATION)
        if not prompt_template:
            raise AgentExecutionError(
                agent_name=self.name,
                message=f"No prompt template found for type {ContentType.QUALIFICATION}",
            )

        # Prepare context for the prompt
        cv_summary = ""
        # Find executive summary from sections
        for section in structured_cv.sections:
            if section.name == "Executive Summary" and section.items:
                cv_summary = section.items[0].content
                break
        # You might want to extract more relevant CV data here for the prompt

        prompt = prompt_template.format(
            job_description=job_data.model_dump_json(indent=2),
            cv_summary=cv_summary,
            research_findings=research_findings,
        )

        response = await self.llm_service.generate_content(
            prompt=prompt,
            max_tokens=self.settings.get("max_tokens_content_generation", 1024),
            temperature=self.settings.get("temperature_content_generation", 0.7),
        )

        if not response or not response.content:
            raise AgentExecutionError(
                agent_name=self.name, message="LLM failed to generate valid Key Qualifications content."
            )

        # Assuming the LLM returns a list of qualifications, one per line or comma-separated
        # This parsing logic might need to be more robust based on actual LLM output format
        qualifications = []
        for line in response.content.split('\n'):
            line = line.strip()
            if line:
                # Remove bullet points and other common prefixes
                line = line.lstrip('- •*').strip()
                if line:
                    qualifications.append(line)
        return qualifications