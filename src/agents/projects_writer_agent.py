"""
This module defines the ProjectsWriterAgent, responsible for generating the Projects section of the CV.
"""

from typing import Any, Dict

from src.agents.agent_base import AgentBase
from src.config.logging_config import get_structured_logger
from src.constants.agent_constants import AgentConstants
from src.constants.llm_constants import LLMConstants
from src.error_handling.exceptions import AgentExecutionError
from src.models.agent_input_models import ProjectsWriterAgentInput

from src.models.agent_output_models import EnhancedContentWriterOutput
from src.models.cv_models import ItemStatus, ItemType
from src.models.data_models import (ContentType, JobDescriptionData, StructuredCV)
from src.services.llm_service_interface import LLMServiceInterface
from src.templates.content_templates import ContentTemplateManager
from src.utils.cv_data_factory import get_item_by_id, update_item_by_id

logger = get_structured_logger(__name__)


class ProjectsWriterAgent(AgentBase):
    """Agent for generating tailored Projects content."""

    def __init__(
        self,
        llm_service: LLMServiceInterface,
        template_manager: ContentTemplateManager,
        settings: dict,
        session_id: str,
    ):
        """Initialize the Projects writer agent."""
        super().__init__(
            name="ProjectsWriter",
            description="Generates tailored Projects for the CV.",
            session_id=session_id,
            settings=settings,
        )
        self.llm_service = llm_service
        self.template_manager = template_manager
        self.settings = settings

    def _validate_inputs(self, **kwargs: Any) -> ProjectsWriterAgentInput:
        """Validate inputs for Projects writer agent using Pydantic model.
        
        Args:
            **kwargs: Input arguments to validate
            
        Returns:
            ProjectsWriterAgentInput: Validated input model
            
        Raises:
            AgentExecutionError: If validation fails
        """
        try:
            return ProjectsWriterAgentInput(**kwargs)
        except Exception as e:
            raise AgentExecutionError(
                agent_name=self.name,
                message=f"Input validation failed: {e}",
            )

    async def _execute(self, **kwargs: Any) -> dict[str, Any]:
        """Execute the core content generation logic for Projects."""
        try:
            # Validate inputs using Pydantic model
            validated_inputs = self._validate_inputs(**kwargs)
            
            structured_cv = validated_inputs.structured_cv
            job_description_data = validated_inputs.job_description_data
            current_item_id = validated_inputs.current_item_id
            research_findings = validated_inputs.research_findings

            self.update_progress(
                AgentConstants.PROGRESS_MAIN_PROCESSING, f"Generating Projects content for item {current_item_id}."
            )

            # Get the specific project item to be enhanced
            project_item = get_item_by_id(structured_cv, current_item_id)
            if (
                not project_item
                or project_item.item_type != ItemType.PROJECT_DESCRIPTION_BULLET
            ):
                return {"error_messages": [f"Item with ID '{current_item_id}' not found or is not a project experience item."]}

            generated_content = await self._generate_project_content(
                structured_cv, job_description_data, project_item, research_findings
            )

            self.update_progress(
                AgentConstants.PROGRESS_POST_PROCESSING, f"Updating CV with generated Projects for item {current_item_id}."
            )

            # Update the specific item with the generated content
            updated_cv = update_item_by_id(
                structured_cv,
                current_item_id,
                {"content": generated_content, "status": ItemStatus.GENERATED},
            )

            self.update_progress(AgentConstants.PROGRESS_COMPLETE, "Projects generation completed successfully.")
            return {
                "structured_cv": updated_cv,
                "current_item_id": current_item_id
            }
        except AgentExecutionError as e:
            logger.error(f"Agent execution error in {self.name}: {str(e)}")
            return {"error_messages": [str(e)]}
        except Exception as e:
            logger.error(f"Unexpected error in {self.name}: {str(e)}", exc_info=True)
            return {"error_messages": [f"Unexpected error during Projects generation: {str(e)}"]}

    async def _generate_project_content(
        self,
        structured_cv: StructuredCV,
        job_data: JobDescriptionData,
        project_item: Dict[str, Any],
        research_findings: Dict[str, Any] | None,
    ) -> str:
        """Generates project content for a specific item using an LLM."""
        prompt_template = self.template_manager.get_template_by_type(
            ContentType.PROJECT
        )
        if not prompt_template:
            raise AgentExecutionError(
                agent_name=self.name,
                message=f"No prompt template found for type {ContentType.PROJECT}",
            )

        # Prepare context for the prompt
        key_qualifications_content = "; ".join(
            [
                item.content
                for section in structured_cv.sections
                if section.name.lower().replace(" ", "_") == "key_qualifications"
                for item in section.items
            ]
        )
        professional_experience_content = "\n".join(
            [
                item.content
                for section in structured_cv.sections
                if section.name.lower().replace(" ", "_") == "professional_experience"
                for item in section.items
            ]
        )

        prompt = self.template_manager.format_template(
            prompt_template,
            {
                "job_description": job_data.model_dump_json(indent=2),
                "project_item": project_item,  # Pass the specific project item
                "key_qualifications": key_qualifications_content,
                "professional_experience": professional_experience_content,
                "research_findings": research_findings,
            },
        )

        response = await self.llm_service.generate_content(
            prompt=prompt,
            content_type=ContentType.PROJECT,
            max_tokens=self.settings.get("max_tokens_content_generation", LLMConstants.DEFAULT_MAX_TOKENS),
            temperature=self.settings.get("temperature_content_generation", LLMConstants.TEMPERATURE_BALANCED),
        )

        if not response or not response.content:
            raise AgentExecutionError(
                agent_name=self.name,
                message="LLM failed to generate valid Projects content.",
            )

        return response.content
