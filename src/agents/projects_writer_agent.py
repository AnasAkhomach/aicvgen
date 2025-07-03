"""
This module defines the ProjectsWriterAgent, responsible for generating the Projects section of the CV.
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


class ProjectsWriterAgent(AgentBase):
    """Agent for generating tailored Projects content."""

    def __init__(
        self,
        llm_service: EnhancedLLMService,
        template_manager: ContentTemplateManager,
        settings: dict,
        session_id: str,
    ):
        """Initialize the Projects writer agent."""
        super().__init__(
            name="ProjectsWriter",
            description="Generates tailored Projects for the CV.",
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
            logger.error(f"ProjectsWriterAgent execution failed: {str(e)}")
            if "error_messages" not in state_dict:
                state_dict["error_messages"] = []
            state_dict["error_messages"].append(str(e))

        return state_dict

    def _validate_inputs(self, input_data: dict) -> None:
        """Validate inputs for Projects writer agent."""
        if "structured_cv" not in input_data or not isinstance(input_data["structured_cv"], StructuredCV):
            raise AgentExecutionError(
                agent_name=self.name,
                message="Missing or invalid 'structured_cv' in input_data.",
            )
        if "job_description_data" not in input_data or not isinstance(input_data["job_description_data"], JobDescriptionData):
            raise AgentExecutionError(
                agent_name=self.name,
                message="Missing or invalid 'job_description_data' in input_data.",
            )
        if "current_item_id" not in input_data or not isinstance(input_data["current_item_id"], str):
            raise AgentExecutionError(
                agent_name=self.name,
                message="Missing or invalid 'current_item_id' in input_data.",
            )

    async def _execute(self, **kwargs: Any) -> AgentResult:
        """Execute the core content generation logic for Projects."""
        structured_cv: StructuredCV = kwargs.get("structured_cv")
        job_description_data: JobDescriptionData = kwargs.get("job_description_data")
        current_item_id: str = kwargs.get("current_item_id")
        research_findings = kwargs.get("research_findings")

        self._validate_inputs({
            "structured_cv": structured_cv,
            "job_description_data": job_description_data,
            "current_item_id": current_item_id
        })

        self.update_progress(40, f"Generating Projects content for item {current_item_id}.")

        # Get the specific project item to be enhanced
        project_item = get_item_by_id(structured_cv, current_item_id)
        if not project_item or project_item.item_type != ItemType.PROJECT_DESCRIPTION_BULLET:
            raise AgentExecutionError(
                agent_name=self.name,
                message=f"Item with ID '{current_item_id}' not found or is not a project experience item.",
            )

        generated_content = await self._generate_project_content(
            structured_cv, job_description_data, project_item, research_findings
        )

        self.update_progress(
            80, f"Updating CV with generated Projects for item {current_item_id}."
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

        self.update_progress(100, "Projects generation completed successfully.")
        return AgentResult(
            success=True,
            output_data=output_data,
            metadata={
                "agent_name": self.name,
                "message": f"Successfully generated Projects for item '{current_item_id}'.",
            },
        )

    async def _generate_project_content(
        self,
        structured_cv: StructuredCV,
        job_data: JobDescriptionData,
        project_item: Dict[str, Any],
        research_findings: Dict[str, Any] | None,
    ) -> str:
        """Generates project content for a specific item using an LLM."""
        prompt_template = self.template_manager.get_template_by_type(ContentType.PROJECT)
        if not prompt_template:
            raise AgentExecutionError(
                agent_name=self.name,
                message=f"No prompt template found for type {ContentType.PROJECT}",
            )

        # Prepare context for the prompt
        key_qualifications_content = "; ".join([item.content for section in structured_cv.sections if section.name.lower().replace(" ", "_") == "key_qualifications" for item in section.items])
        professional_experience_content = "\n".join([item.content for section in structured_cv.sections if section.name.lower().replace(" ", "_") == "professional_experience" for item in section.items])

        prompt = prompt_template.format(
            job_description=job_data.model_dump_json(indent=2),
            project_item=project_item, # Pass the specific project item
            key_qualifications=key_qualifications_content,
            professional_experience=professional_experience_content,
            research_findings=research_findings,
        )

        response = await self.llm_service.generate_content(
            prompt=prompt,
            max_tokens=self.settings.get("max_tokens_content_generation", 1024),
            temperature=self.settings.get("temperature_content_generation", 0.7),
        )

        if not response or not response.content:
            raise AgentExecutionError(
                agent_name=self.name, message="LLM failed to generate valid Projects content."
            )

        return response.content