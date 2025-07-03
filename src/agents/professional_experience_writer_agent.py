"""
This module defines the ProfessionalExperienceWriterAgent, responsible for generating the Professional Experience section of the CV.
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


class ProfessionalExperienceWriterAgent(AgentBase):
    """Agent for generating tailored Professional Experience content."""

    def __init__(
        self,
        llm_service: EnhancedLLMService,
        template_manager: ContentTemplateManager,
        settings: dict,
        session_id: str,
    ):
        """Initialize the Professional Experience writer agent."""
        super().__init__(
            name="ProfessionalExperienceWriter",
            description="Generates tailored Professional Experience for the CV.",
            session_id=session_id,
        )
        self.llm_service = llm_service
        self.template_manager = template_manager
        self.settings = settings

    async def run_as_node(self, state: Union[AgentState, dict]) -> dict:
        """Execute the agent as a node in the workflow, returning a dictionary."""
        if hasattr(state, 'model_dump'):
            state_dict = state.model_dump()
        else:
            state_dict = state.copy() if isinstance(state, dict) else {}

        try:
            result = await self._execute(
                structured_cv=state_dict.get('structured_cv'),
                job_description_data=state_dict.get('job_description_data'),
                current_item_id=state_dict.get('current_item_id'),
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
            logger.error(f"ProfessionalExperienceWriterAgent execution failed: {str(e)}")
            if "error_messages" not in state_dict:
                state_dict["error_messages"] = []
            state_dict["error_messages"].append(str(e))

        return state_dict

    def _validate_inputs(self, input_data: dict) -> None:
        """Validate inputs for Professional Experience writer agent."""
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
        if "current_item_id" not in input_data or not isinstance(input_data["current_item_id"], str):
            raise AgentExecutionError(
                agent_name=self.name,
                message="Missing or invalid 'current_item_id' in input_data.",
            )

        if isinstance(input_data["structured_cv"], dict):
            input_data["structured_cv"] = StructuredCV(**input_data["structured_cv"])
        if isinstance(input_data["job_description_data"], dict):
            input_data["job_description_data"] = JobDescriptionData(**input_data["job_description_data"])

    async def _execute(self, **kwargs: Any) -> AgentResult:
        """Execute the core content generation logic for Professional Experience."""
        structured_cv_data = kwargs.get("structured_cv")
        if isinstance(structured_cv_data, dict):
            structured_cv = StructuredCV(**structured_cv_data)
        else:
            structured_cv = structured_cv_data

        job_description_data = kwargs.get("job_description_data")
        if isinstance(job_description_data, dict):
            job_description_data = JobDescriptionData(**job_description_data)
        else:
            job_description_data = job_description_data
            
        current_item_id: str = kwargs.get("current_item_id")
        research_findings = kwargs.get("research_findings")

        self._validate_inputs({
            "structured_cv": structured_cv,
            "job_description_data": job_description_data,
            "current_item_id": current_item_id
        })

        self.update_progress(40, f"Generating Professional Experience content for item {current_item_id}.")

        # Get the specific professional experience item to be enhanced
        experience_item = get_item_by_id(structured_cv, current_item_id)
        if not experience_item or experience_item.item_type != ItemType.EXPERIENCE_ROLE_TITLE:
            raise AgentExecutionError(
                agent_name=self.name,
                message=f"Item with ID '{current_item_id}' not found or is not a professional experience item.",
            )

        generated_content = await self._generate_professional_experience_content(
            structured_cv, job_description_data, experience_item, research_findings
        )

        self.update_progress(
            80, f"Updating CV with generated Professional Experience for item {current_item_id}."
        )

        # Update the specific item with the generated content
        updated_cv = update_item_by_id(
            structured_cv,
            current_item_id,
            {"content": generated_content, "status": ItemStatus.GENERATED}
        )

        output_data = EnhancedContentWriterOutput(
            updated_structured_cv=updated_cv,
            item_id=current_item_id,
            generated_content=generated_content,
        )

        self.update_progress(100, "Professional Experience generation completed successfully.")
        return AgentResult(
            success=True,
            output_data=output_data,
            metadata={
                "agent_name": self.name,
                "message": f"Successfully generated Professional Experience for item '{current_item_id}'.",
            },
        )

    async def _generate_professional_experience_content(
        self,
        structured_cv: StructuredCV,
        job_data: JobDescriptionData,
        experience_item: Dict[str, Any],
        research_findings: Dict[str, Any] | None,
    ) -> str:
        """Generates professional experience content for a specific item using an LLM."""
        prompt_template = self.template_manager.get_template_by_type(ContentType.EXPERIENCE)
        if not prompt_template:
            raise AgentExecutionError(
                agent_name=self.name,
                message=f"No prompt template found for type {ContentType.EXPERIENCE}",
            )

        # Prepare context for the prompt
        # You might want to include other sections like Key Qualifications or Executive Summary
        # to provide more context to the LLM.
        key_qualifications_content = "; ".join([item.content for section in structured_cv.sections if section.name.lower().replace(" ", "_") == "key_qualifications" for item in section.items])

        prompt = prompt_template.format(
            job_description=job_data.model_dump_json(indent=2),
            experience_item=experience_item, # Pass the specific experience item
            key_qualifications=key_qualifications_content,
            research_findings=research_findings,
        )

        response = await self.llm_service.generate_content(
            prompt=prompt,
            max_tokens=self.settings.get("max_tokens_content_generation", 1024),
            temperature=self.settings.get("temperature_content_generation", 0.7),
        )

        if not response or not response.content:
            raise AgentExecutionError(
                agent_name=self.name, message="LLM failed to generate valid Professional Experience content."
            )

        return response.content