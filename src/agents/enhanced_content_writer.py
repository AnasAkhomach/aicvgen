"""
This module defines the EnhancedContentWriterAgent, responsible for generating tailored CV content.
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
from ..utils.cv_data_factory import get_item_by_id, update_item_by_id

from ..config.logging_config import get_structured_logger
from ..error_handling.exceptions import AgentExecutionError
from .agent_base import AgentBase

logger = get_structured_logger(__name__)


class EnhancedContentWriterAgent(AgentBase):
    """Agent for generating tailored CV content with advanced error handling."""

    def __init__(
        self,
        llm_service: EnhancedLLMService,
        template_manager: ContentTemplateManager,
        settings: dict,
        session_id: str,
    ):
        """Initialize the enhanced content writer agent."""
        super().__init__(
            name="EnhancedContentWriter",
            description="Generates tailored CV content.",
            session_id=session_id,
        )
        self.llm_service = llm_service
        self.template_manager = template_manager
        self.settings = settings

    def _validate_inputs(self, input_data: dict) -> None:
        """Validate inputs for enhanced content writer agent.

        Note: This agent accepts parameters directly via kwargs rather than through input_data.
        This method is implemented to satisfy the base class contract but actual validation
        is performed in _execute method.
        """
        # The enhanced content writer agent receives its inputs via direct kwargs
        # rather than through input_data, so this method is intentionally minimal
        return

    async def _execute(self, **kwargs: Any) -> AgentResult:
        """Execute the core content generation logic."""
        # Extract parameters from kwargs
        structured_cv = kwargs.get("structured_cv")
        job_description_data = kwargs.get("job_description_data")
        item_id = kwargs.get("item_id")
        research_findings = kwargs.get("research_findings")

        if not structured_cv or not job_description_data or not item_id:
            raise AgentExecutionError(
                agent_name=self.name,
                message="Missing required parameters: structured_cv, job_description_data, or item_id",
            )

        content_item = get_item_by_id(structured_cv, item_id)
        if not content_item:
            raise AgentExecutionError(
                agent_name=self.name,
                message=f"Item with ID '{item_id}' not found in structured_cv.",
            )

        self.update_progress(40, f"Generating content for item {item_id}")
        enhanced_content = await self._generate_enhanced_content(
            structured_cv, job_description_data, content_item, research_findings
        )

        self.update_progress(
            80, f"Updating CV with enhanced content for item {item_id}"
        )
        updated_cv = update_item_by_id(
            structured_cv, item_id, {"enhanced_content": enhanced_content}
        )

        output_data = EnhancedContentWriterOutput(
            updated_structured_cv=updated_cv,
            item_id=item_id,
            generated_content=enhanced_content,
        )

        self.update_progress(100, "Content generation completed successfully")
        return AgentResult(
            success=True,
            output_data=output_data,
            metadata={
                "agent_name": self.name,
                "message": f"Successfully generated content for item '{item_id}'.",
            },
        )

    async def _generate_enhanced_content(
        self,
        _cv_data: StructuredCV,  # Currently unused but kept for interface consistency
        job_data: JobDescriptionData,
        content_item: Dict[str, Any],
        research_findings: Dict[str, Any] | None,
    ) -> str:
        """Generates enhanced content for a specific CV item using an LLM."""
        content_type_str = content_item.get("type", "generic").upper()
        content_type = ContentType[content_type_str]

        prompt_template = self.template_manager.get_template_by_type(content_type)
        if not prompt_template:
            raise AgentExecutionError(
                agent_name=self.name,
                message=f"No prompt template found for type {content_type}",
            )

        prompt = prompt_template.format(
            job_description=job_data.model_dump_json(indent=2),
            cv_section=content_item,
            research_findings=research_findings,
        )

        response = await self.llm_service.generate_content(
            prompt=prompt,
            max_tokens=self.settings.get("max_tokens_content_generation", 1024),
            temperature=self.settings.get("temperature_content_generation", 0.7),
        )

        if not response or not response.content:
            raise AgentExecutionError(
                agent_name=self.name, message="LLM failed to generate valid content."
            )

        return response.content

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
                item_id=state_dict.get('current_item_id'),
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
            logger.error(f"EnhancedContentWriterAgent execution failed: {str(e)}")
            if "error_messages" not in state_dict:
                state_dict["error_messages"] = []
            state_dict["error_messages"].append(str(e))

        return state_dict
