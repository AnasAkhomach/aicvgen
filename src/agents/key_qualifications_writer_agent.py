"""
This module defines the KeyQualificationsWriterAgent, responsible for generating the Key Qualifications section of the CV.
"""

from typing import Any, Dict

from src.agents.agent_base import AgentBase
from src.config.logging_config import get_structured_logger
from src.constants.agent_constants import AgentConstants
from src.constants.llm_constants import LLMConstants
from src.error_handling.exceptions import AgentExecutionError
from src.models.agent_models import AgentResult
from src.models.agent_output_models import EnhancedContentWriterOutput
from src.models.cv_models import Item, ItemStatus, ItemType
from src.models.data_models import (ContentType, JobDescriptionData, StructuredCV)
from src.services.llm_service_interface import LLMServiceInterface
from src.templates.content_templates import ContentTemplateManager

logger = get_structured_logger(__name__)


class KeyQualificationsWriterAgent(AgentBase):
    """Agent for generating tailored Key Qualifications content."""

    def __init__(
        self,
        llm_service: LLMServiceInterface,
        template_manager: ContentTemplateManager,
        settings: dict,
        session_id: str,
    ):
        """Initialize the Key Qualifications writer agent."""
        super().__init__(
            name="KeyQualificationsWriter",
            description="Generates tailored Key Qualifications for the CV.",
            session_id=session_id,
            settings=settings,
        )
        self.llm_service = llm_service
        self.template_manager = template_manager

    def _validate_inputs(self, input_data: dict) -> None:
        """Validate inputs for Key Qualifications writer agent."""

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

        # Convert dict back to Pydantic objects if needed
        if isinstance(input_data["structured_cv"], dict):
            input_data["structured_cv"] = StructuredCV(**input_data["structured_cv"])
        if isinstance(input_data["job_description_data"], dict):
            input_data["job_description_data"] = JobDescriptionData(
                **input_data["job_description_data"]
            )

    async def _execute(self, **kwargs: Any) -> AgentResult:
        """Execute the core content generation logic for Key Qualifications."""
        try:
            # Validate and convert inputs in place
            self._validate_inputs(kwargs)
            
            structured_cv: StructuredCV = kwargs.get("structured_cv")
            job_description_data: JobDescriptionData = kwargs.get("job_description_data")
            research_findings = kwargs.get("research_findings")

            self.update_progress(AgentConstants.PROGRESS_MAIN_PROCESSING, "Generating Key Qualifications content.")
            generated_qualifications = await self._generate_key_qualifications(
                structured_cv, job_description_data, research_findings
            )

            self.update_progress(AgentConstants.PROGRESS_POST_PROCESSING, "Updating CV with generated Key Qualifications.")

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

            self.update_progress(
                AgentConstants.PROGRESS_COMPLETE, "Key Qualifications generation completed successfully."
            )
            return AgentResult(
                success=True,
                output_data=output_data,
                metadata={
                    "agent_name": self.name,
                    "message": "Successfully generated Key Qualifications.",
                },
            )
        except AgentExecutionError:
            # Re-raise AgentExecutionError to maintain error context
            raise
        except (AttributeError, TypeError, ValueError, KeyError) as e:
            # Handle common exceptions with proper context
            raise AgentExecutionError(
                agent_name=self.name,
                message=f"Error processing Key Qualifications data: {str(e)}",
            ) from e
        except Exception as e:
            # Handle any other unexpected exceptions with proper context
            logger.error(f"Unexpected error in {self.name}: {str(e)}", exc_info=True)
            raise AgentExecutionError(
                agent_name=self.name,
                message=f"Unexpected error during Key Qualifications generation: {str(e)}",
            ) from e

    async def _generate_key_qualifications(
        self,
        structured_cv: StructuredCV,
        job_data: JobDescriptionData,
        research_findings: Dict[str, Any] | None,
    ) -> list[str]:
        """Generates key qualifications using an LLM."""
        prompt_template = self.template_manager.get_template_by_type(
            ContentType.QUALIFICATION
        )
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

        prompt = self.template_manager.format_template(
            prompt_template,
            {
                "job_description": job_data.model_dump_json(indent=2),
                "cv_summary": cv_summary,
                "research_findings": research_findings,
            },
        )

        response = await self.llm_service.generate_content(
            prompt=prompt,
            content_type=ContentType.QUALIFICATION,
            max_tokens=self.settings.get("max_tokens_content_generation", LLMConstants.DEFAULT_MAX_TOKENS),
            temperature=self.settings.get("temperature_content_generation", LLMConstants.TEMPERATURE_BALANCED),
        )

        if not response or not response.content:
            raise AgentExecutionError(
                agent_name=self.name,
                message="LLM failed to generate valid Key Qualifications content.",
            )

        # Assuming the LLM returns a list of qualifications, one per line or comma-separated
        # This parsing logic might need to be more robust based on actual LLM output format
        qualifications = []
        for line in response.content.split("\n"):
            line = line.strip()
            if line:
                # Remove bullet points and other common prefixes
                line = line.lstrip("- •*").strip()
                if line:
                    qualifications.append(line)
        return qualifications
