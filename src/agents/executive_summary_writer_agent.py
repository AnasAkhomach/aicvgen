"""
This module defines the ExecutiveSummaryWriterAgent, responsible for generating the Executive Summary section of the CV.
"""

from typing import Any, Dict

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


class ExecutiveSummaryWriterAgent(AgentBase):
    """Agent for generating tailored Executive Summary content."""

    def __init__(
        self,
        llm_service: EnhancedLLMService,
        template_manager: ContentTemplateManager,
        settings: dict,
        session_id: str,
    ):
        """Initialize the Executive Summary writer agent."""
        super().__init__(
            name="ExecutiveSummaryWriter",
            description="Generates tailored Executive Summary for the CV.",
            session_id=session_id,
        )
        self.llm_service = llm_service
        self.template_manager = template_manager
        self.settings = settings

    def _validate_inputs(self, input_data: dict) -> None:
        """Validate inputs for Executive Summary writer agent."""

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
        """Execute the core content generation logic for Executive Summary."""
        structured_cv: StructuredCV = kwargs.get("structured_cv")
        job_description_data: JobDescriptionData = kwargs.get("job_description_data")
        research_findings = kwargs.get("research_findings")

        self._validate_inputs({"structured_cv": structured_cv, "job_description_data": job_description_data})

        self.update_progress(40, "Generating Executive Summary content.")
        generated_summary = await self._generate_executive_summary(
            structured_cv, job_description_data, research_findings
        )

        self.update_progress(
            80, "Updating CV with generated Executive Summary."
        )

        # Find the Executive Summary section or create it if it doesn't exist
        summary_section = None
        for section in structured_cv.sections:
            if section.name.lower() == "executive summary":
                summary_section = section
                break

        if not summary_section:
            raise AgentExecutionError(
                agent_name=self.name,
                message="Executive Summary section not found in structured_cv. It should be pre-initialized.",
            )

        # Update the content of the executive summary section
        # Assuming executive summary is a single item within its section
        if summary_section.items:
            summary_item = summary_section.items[0]
            summary_item.content = generated_summary
            summary_item.status = ItemStatus.GENERATED
            item_id = str(summary_item.id)
        else:
            # If for some reason there are no items, add one
            new_item = Item(
                content=generated_summary,
                status=ItemStatus.GENERATED,
                item_type=ItemType.EXECUTIVE_SUMMARY_PARA,
            )
            summary_section.items.append(new_item)
            item_id = str(new_item.id)

        output_data = EnhancedContentWriterOutput(
            updated_structured_cv=structured_cv,
            item_id=item_id,  # Use the actual item ID
            generated_content=generated_summary,
        )

        self.update_progress(100, "Executive Summary generation completed successfully.")
        return AgentResult(
            success=True,
            output_data=output_data,
            metadata={
                "agent_name": self.name,
                "message": "Successfully generated Executive Summary.",
            },
        )

    async def _generate_executive_summary(
        self,
        structured_cv: StructuredCV,
        job_data: JobDescriptionData,
        research_findings: Dict[str, Any] | None,
    ) -> str:
        """Generates executive summary content using an LLM."""
        prompt_template = self.template_manager.get_template_by_type(ContentType.EXECUTIVE_SUMMARY)
        if not prompt_template:
            raise AgentExecutionError(
                agent_name=self.name,
                message=f"No prompt template found for type {ContentType.EXECUTIVE_SUMMARY}",
            )

        # Prepare context for the prompt
        key_qualifications_content = "; ".join([item.content for section in structured_cv.sections if section.name.lower().replace(" ", "_") == "key_qualifications" for item in section.items])
        professional_experience_content = "\n".join([item.content for section in structured_cv.sections if section.name.lower().replace(" ", "_") == "professional_experience" for item in section.items])
        projects_content = "\n".join([item.content for section in structured_cv.sections if section.name.lower().replace(" ", "_") == "project_experience" for item in section.items])

        prompt = prompt_template.format(
            job_description=job_data.model_dump_json(indent=2),
            key_qualifications=key_qualifications_content,
            professional_experience=professional_experience_content,
            projects=projects_content,
            research_findings=research_findings,
        )

        response = await self.llm_service.generate_content(
            prompt=prompt,
            max_tokens=self.settings.get("max_tokens_content_generation", 1024),
            temperature=self.settings.get("temperature_content_generation", 0.7),
        )

        if not response or not response.content:
            raise AgentExecutionError(
                agent_name=self.name, message="LLM failed to generate valid Executive Summary content."
            )

        return response.content

    async def run_as_node(self, state) -> dict:
        """Run the agent as a node in the workflow graph.
        
        Args:
            state: Either an AgentState object or a dictionary containing the state
            
        Returns:
            dict: Updated state dictionary
        """
        # Handle both AgentState objects and dictionaries
        if hasattr(state, 'model_dump'):
            # It's an AgentState object, convert to dict
            state_dict = state.model_dump()
        else:
            # It's already a dictionary
            state_dict = state
            
        try:
            result = await self._execute(**state_dict)
            
            if result.success:
                # Update the state with the new structured_cv
                state_dict["structured_cv"] = result.output_data.updated_structured_cv
                state_dict["error_messages"] = []
            else:
                # Add error messages to state
                if "error_messages" not in state_dict:
                    state_dict["error_messages"] = []
                state_dict["error_messages"].extend(result.error_messages or [])
                
        except Exception as e:
            logger.error(f"ExecutiveSummaryWriterAgent execution failed: {str(e)}")
            if "error_messages" not in state_dict:
                state_dict["error_messages"] = []
            state_dict["error_messages"].append(str(e))
            
        return state_dict