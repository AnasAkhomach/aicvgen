"""This module defines the CleaningAgent, responsible for processing and cleaning raw text or LLM outputs."""

from typing import Any, List, Tuple

from ..config.logging_config import get_structured_logger
from ..error_handling.exceptions import AgentExecutionError
from ..models.agent_models import AgentResult
from ..models.agent_output_models import CleaningAgentOutput
from ..services.llm_service import EnhancedLLMService
from ..templates.content_templates import ContentTemplateManager
from .agent_base import AgentBase


logger = get_structured_logger(__name__)


class CleaningAgent(AgentBase):
    """Agent responsible for cleaning and structuring raw LLM outputs or other text."""

    def __init__(
        self,
        llm_service: EnhancedLLMService,
        template_manager: ContentTemplateManager,
        settings: dict,
        session_id: str = "default",
    ):
        """Initialize the CleaningAgent with required dependencies."""
        super().__init__(
            name="CleaningAgent",
            description="Cleans and structures raw text or LLM outputs.",
            session_id=session_id,
        )
        self.llm_service = llm_service
        self.template_manager = template_manager
        self.settings = settings

    async def run(self, **kwargs: Any) -> AgentResult:
        """Process and clean raw LLM output, a list of strings, or a dictionary."""
        self.update_progress(0, "Starting cleaning process")
        input_data = kwargs.get("input_data")

        try:
            if input_data is None:
                raise AgentExecutionError(message="Input data cannot be None.")

            raw_output_for_model = None
            output_type_for_model = None
            modifications = []

            if isinstance(input_data, dict):
                raw_content = input_data.get("raw_output")
                output_type = input_data.get("output_type")
                if raw_content is None:
                    raise AgentExecutionError(
                        message="Dictionary input must contain a 'raw_output' key."
                    )
                cleaned_data, mods = self._clean_generic_output(raw_content)
                raw_output_for_model = raw_content
                output_type_for_model = output_type
                modifications.extend(mods)
            elif isinstance(input_data, list):
                cleaned_data, mods = self._clean_skills_list(input_data)
                raw_output_for_model = str(input_data)
                output_type_for_model = "skills_list"
                modifications.extend(mods)
            elif isinstance(input_data, str):
                cleaned_data, mods = self._clean_generic_output(input_data)
                raw_output_for_model = input_data
                output_type_for_model = "string"
                modifications.extend(mods)
            else:
                raise AgentExecutionError(
                    message=f"Unsupported input type for CleaningAgent: {type(input_data)}"
                )

            output = CleaningAgentOutput(
                cleaned_data=cleaned_data,
                raw_output=raw_output_for_model,
                output_type=output_type_for_model,
                modifications_made=modifications,
            )

            self.update_progress(100, "Cleaning process completed")
            return AgentResult(success=True, output_data=output)

        except AgentExecutionError as e:
            logger.error("Agent execution error in CleaningAgent: %s", e, exc_info=True)
            return AgentResult(
                success=False,
                error_message=str(e),
                output_data=CleaningAgentOutput(cleaned_data=input_data or None),
            )
        except Exception as e:
            logger.error("Unhandled exception in CleaningAgent: %s", e, exc_info=True)
            return AgentResult(
                success=False,
                error_message=f"An unexpected error occurred in CleaningAgent: {e}",
                output_data=CleaningAgentOutput(cleaned_data=input_data or None),
            )

    def _clean_generic_output(self, raw_output: str) -> Tuple[str, List[str]]:
        """Cleans generic text output by removing unnecessary formatting."""
        self.update_progress(30, "Cleaning generic output")
        cleaned_text = raw_output.strip()
        modifications = ["Stripped leading/trailing whitespace."]
        self.update_progress(70, "Generic output cleaned")
        return cleaned_text, modifications

    def _clean_skills_list(self, skills_list: List[str]) -> Tuple[List[str], List[str]]:
        """Cleans a list of skills by standardizing format."""
        self.update_progress(30, "Cleaning skills list")
        cleaned_skills = [skill.strip().lower() for skill in skills_list]
        modifications = [
            "Standardized all skills to lowercase and stripped whitespace."
        ]
        self.update_progress(70, "Skills list cleaned")
        return cleaned_skills, modifications
