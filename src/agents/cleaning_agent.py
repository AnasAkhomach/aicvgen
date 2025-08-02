"""This module defines the CleaningAgent, responsible for processing and cleaning raw text or LLM outputs."""

from typing import Any, List, Tuple

from src.agents.agent_base import AgentBase
from src.config.logging_config import get_structured_logger
from src.constants.agent_constants import AgentConstants

from src.models.agent_output_models import CleaningAgentOutput
from src.services.llm_service_interface import LLMServiceInterface
from src.templates.content_templates import ContentTemplateManager

logger = get_structured_logger(__name__)


class CleaningAgent(AgentBase):
    """Agent responsible for cleaning and structuring raw LLM outputs or other text."""

    def __init__(
        self,
        llm_service: LLMServiceInterface,
        template_manager: ContentTemplateManager,
        settings: dict,
        session_id: str = "default",
    ):
        """Initialize the CleaningAgent with required dependencies."""
        super().__init__(
            name="CleaningAgent",
            description="Cleans and structures raw text or LLM outputs.",
            session_id=session_id,
            settings=settings,
        )
        self.llm_service = llm_service
        self.template_manager = template_manager

    async def _execute(self, **kwargs: Any) -> dict[str, Any]:
        """Execute the core cleaning logic."""
        input_data = kwargs.get("input_data")

        raw_output_for_model = None
        output_type_for_model = None
        modifications = []
        cleaned_data = None

        if isinstance(input_data, dict):
            raw_content = input_data.get("raw_output")
            output_type = input_data.get("output_type")
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

        output = CleaningAgentOutput(
            cleaned_data=cleaned_data,
            raw_output=raw_output_for_model,
            output_type=output_type_for_model,
            modifications_made=modifications,
        )

        self.update_progress(100, "Cleaning process completed")
        return {
            "cleaned_data": cleaned_data,
            "raw_output": raw_output_for_model,
            "output_type": output_type_for_model,
            "modifications_made": modifications,
        }

    def _clean_generic_output(self, raw_output: str) -> Tuple[str, List[str]]:
        """Cleans generic text output by removing unnecessary formatting."""
        self.update_progress(
            AgentConstants.PROGRESS_PREPROCESSING, "Cleaning generic output"
        )
        cleaned_text = raw_output.strip()
        modifications = ["Stripped leading/trailing whitespace."]
        self.update_progress(
            AgentConstants.PROGRESS_CLEANING_COMPLETE, "Generic output cleaned"
        )
        return cleaned_text, modifications

    def _clean_skills_list(self, skills_list: List[str]) -> Tuple[List[str], List[str]]:
        """Cleans a list of skills by standardizing format."""
        self.update_progress(
            AgentConstants.PROGRESS_PREPROCESSING, "Cleaning skills list"
        )
        cleaned_skills = [skill.strip().lower() for skill in skills_list]
        modifications = [
            "Standardized all skills to lowercase and stripped whitespace."
        ]
        self.update_progress(
            AgentConstants.PROGRESS_CLEANING_COMPLETE, "Skills list cleaned"
        )
        return cleaned_skills, modifications
