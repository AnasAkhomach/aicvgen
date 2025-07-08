"""
This module defines the ProfessionalExperienceWriterAgent, responsible for generating the Professional Experience section of the CV.
"""

from typing import Any, Dict

from src.agents.agent_base import AgentBase
from src.config.logging_config import get_structured_logger
from src.constants.agent_constants import AgentConstants
from src.constants.llm_constants import LLMConstants
from src.error_handling.exceptions import AgentExecutionError

from src.models.agent_output_models import EnhancedContentWriterOutput
from src.models.cv_models import ItemStatus, ItemType
from src.models.data_models import (ContentType, JobDescriptionData, StructuredCV)
from src.services.llm_service_interface import LLMServiceInterface
from src.templates.content_templates import ContentTemplateManager
from src.utils.cv_data_factory import get_item_by_id, update_item_by_id

logger = get_structured_logger(__name__)


class ProfessionalExperienceWriterAgent(AgentBase):
    """Agent for generating tailored Professional Experience content."""

    def __init__(
        self,
        llm_service: LLMServiceInterface,
        template_manager: ContentTemplateManager,
        settings: dict,
        session_id: str,
    ):
        """Initialize the Professional Experience writer agent."""
        super().__init__(
            name="ProfessionalExperienceWriter",
            description="Generates tailored Professional Experience for the CV.",
            session_id=session_id,
            settings=settings,
        )
        self.llm_service = llm_service
        self.template_manager = template_manager

    def _validate_inputs(self, input_data: dict) -> None:
        """Validate inputs for Professional Experience writer agent."""
        if "structured_cv" not in input_data or input_data["structured_cv"] is None:
            raise AgentExecutionError(
                agent_name=self.name,
                message="Missing or invalid 'structured_cv' in input_data.",
            )
        if (
            "job_description_data" not in input_data
            or input_data["job_description_data"] is None
        ):
            raise AgentExecutionError(
                agent_name=self.name,
                message="Missing or invalid 'job_description_data' in input_data.",
            )
        if "current_item_id" not in input_data or not isinstance(
            input_data["current_item_id"], str
        ):
            raise AgentExecutionError(
                agent_name=self.name,
                message="Missing or invalid 'current_item_id' in input_data.",
            )

        if isinstance(input_data["structured_cv"], dict):
            input_data["structured_cv"] = StructuredCV(**input_data["structured_cv"])
        if isinstance(input_data["job_description_data"], dict):
            input_data["job_description_data"] = JobDescriptionData(
                **input_data["job_description_data"]
            )

    async def _execute(self, **kwargs: Any) -> dict[str, Any]:
        """Execute the core content generation logic for Professional Experience."""
        try:
            # Validate and convert inputs in place
            self._validate_inputs(kwargs)

            # Extract validated inputs (already converted by _validate_inputs)
            structured_cv: StructuredCV = kwargs.get("structured_cv")
            job_description_data: JobDescriptionData = kwargs.get("job_description_data")
            current_item_id = kwargs.get("current_item_id")
            research_findings = kwargs.get("research_findings")

            self.update_progress(
                AgentConstants.PROGRESS_MAIN_PROCESSING,
                f"Generating Professional Experience content for item {current_item_id}.",
            )

            # Get the specific professional experience item to be enhanced
            experience_item = get_item_by_id(structured_cv, current_item_id)
            if (
                not experience_item
                or experience_item.item_type != ItemType.EXPERIENCE_ROLE_TITLE
            ):
                return {"error_messages": [f"Item with ID '{current_item_id}' not found or is not a professional experience item."]}

            generated_content = await self._generate_professional_experience_content(
                structured_cv, job_description_data, experience_item, research_findings
            )

            self.update_progress(
                AgentConstants.PROGRESS_POST_PROCESSING,
                f"Updating CV with generated Professional Experience for item {current_item_id}.",
            )

            # Update the specific item with the generated content
            updated_cv = update_item_by_id(
                structured_cv,
                current_item_id,
                {"content": generated_content, "status": ItemStatus.GENERATED},
            )

            self.update_progress(
                AgentConstants.PROGRESS_COMPLETE, "Professional Experience generation completed successfully."
            )
            return {
                "structured_cv": updated_cv,
                "current_item_id": current_item_id
            }
        except AgentExecutionError as e:
            logger.error(f"Agent execution error in {self.name}: {str(e)}")
            return {"error_messages": [str(e)]}
        except (AttributeError, TypeError, ValueError, KeyError) as e:
            logger.error(f"Error processing Professional Experience data: {str(e)}")
            return {"error_messages": [f"Error processing Professional Experience data: {str(e)}"]}
        except Exception as e:
            logger.error(f"Unexpected error in {self.name}: {str(e)}", exc_info=True)
            return {"error_messages": [f"Unexpected error during Professional Experience generation: {str(e)}"]}

    async def _generate_professional_experience_content(
        self,
        structured_cv: StructuredCV,
        job_data: JobDescriptionData,
        experience_item: Dict[str, Any],
        research_findings: Dict[str, Any] | None,
    ) -> str:
        """Generates professional experience content for a specific item using an LLM."""
        prompt_template = self.template_manager.get_template_by_type(
            ContentType.EXPERIENCE
        )
        if not prompt_template:
            raise AgentExecutionError(
                agent_name=self.name,
                message=f"No prompt template found for type {ContentType.EXPERIENCE}",
            )

        # Prepare context for the prompt
        # You might want to include other sections like Key Qualifications or Executive Summary
        # to provide more context to the LLM.
        key_qualifications_content = "; ".join(
            [
                item.content
                for section in structured_cv.sections
                if section.name.lower().replace(" ", "_") == "key_qualifications"
                for item in section.items
            ]
        )

        prompt = self.template_manager.format_template(
            prompt_template,
            {
                "job_description": job_data.model_dump_json(indent=2),
                "experience_item": experience_item,  # Pass the specific experience item
                "key_qualifications": key_qualifications_content,
                "research_findings": research_findings,
            },
        )

        response = await self.llm_service.generate_content(
            prompt=prompt,
            max_tokens=self.settings.get("max_tokens_content_generation", LLMConstants.MAX_TOKENS_GENERATION),
            temperature=self.settings.get("temperature_content_generation", LLMConstants.TEMPERATURE_BALANCED),
        )

        if not response or not response.content:
            raise AgentExecutionError(
                agent_name=self.name,
                message="LLM failed to generate valid Professional Experience content.",
            )

        return response.content
