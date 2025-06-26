"""
This module defines the EnhancedContentWriterAgent, responsible for generating tailored CV content.
"""

from typing import Any, Dict, Optional

from src.models.agent_models import AgentResult
from src.models.agent_output_models import EnhancedContentWriterOutput
from src.models.data_models import (
    ContentType,
    JobDescriptionData,
    StructuredCV,
)
from src.services.llm_service import EnhancedLLMService
from src.templates.content_templates import ContentTemplateManager
from src.utils.cv_data_factory import get_item_by_id, update_item_by_id

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

    async def run(
        self,
        structured_cv: StructuredCV,
        job_description_data: JobDescriptionData,
        item_id: str,
        research_findings: Optional[Dict[str, Any]] = None,
    ) -> AgentResult[EnhancedContentWriterOutput]:
        """Generate enhanced content for a specific item in the CV."""
        self.update_progress(0, "Starting content generation")

        try:
            content_item = get_item_by_id(structured_cv, item_id)
            if not content_item:
                raise AgentExecutionError(
                    agent_name=self.name,
                    message=f"Item with ID '{item_id}' not found in structured_cv.",
                )

            self.update_progress(20, f"Generating content for item {item_id}")
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

        except AgentExecutionError as e:
            logger.error(
                f"A known error occurred in EnhancedContentWriterAgent: {str(e)}"
            )
            return AgentResult(
                success=False,
                error_message=str(e),
                metadata={"agent_name": self.name},
            )
        except Exception as e:
            logger.error(
                "Unhandled exception in EnhancedContentWriterAgent",
                error=str(e),
                exc_info=True,
            )
            err = AgentExecutionError(
                agent_name=self.name,
                message=f"An unexpected error occurred while generating content for item '{item_id}'",
                original_exception=e,
            )
            return AgentResult(
                success=False,
                error_message=str(err),
                metadata={"agent_name": self.name},
            )

    async def _generate_enhanced_content(
        self,
        cv_data: StructuredCV,
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
