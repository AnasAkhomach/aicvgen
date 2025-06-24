"""CleaningAgent for processing and cleaning raw LLM outputs.

This agent implements the "Generate -> Clean" pattern described in Task 3.2 & 3.4.
It takes raw LLM output and processes it into structured, clean content.
"""

from typing import List, Any, Dict
import re
import json
from datetime import datetime

from ..agents.agent_base import EnhancedAgentBase, AgentExecutionContext, AgentResult
from ..config.settings import get_config
from ..utils.agent_error_handling import AgentErrorHandler, with_node_error_handling
from ..models.data_models import AgentIO, ContentType
from ..orchestration.state import AgentState
from ..models.agent_output_models import CleaningAgentOutput
from ..models.cleaning_agent_models import CleanedDataModel
from ..services.llm_service import EnhancedLLMService

from ..services.progress_tracker import ProgressTracker
from ..templates.content_templates import ContentTemplateManager
from src.utils.json_utils import extract_json_from_response


class CleaningAgent(EnhancedAgentBase):
    """Agent responsible for cleaning and structuring raw LLM outputs.

    This agent processes raw LLM responses to extract structured data,
    particularly for the "Big 10" skills generation feature.
    """

    def __init__(
        self,
        llm_service: EnhancedLLMService,
        progress_tracker: ProgressTracker,
        template_manager: ContentTemplateManager,
    ):
        """Initialize the CleaningAgent with required dependencies.

        Args:
            llm_service: LLM service instance for content processing.
            progress_tracker: Progress tracker service dependency.
            template_manager: The template manager for loading prompts.
        """
        input_schema = AgentIO(
            description="Raw LLM output, or a list of strings, and processing instructions",
            required_fields=[],
            optional_fields=[
                "raw_output",
                "output_type",
                "context",
                "validation_rules",
            ],
        )

        output_schema = AgentIO(
            description="Cleaned and structured output",
            required_fields=["cleaned_data", "confidence_score"],
            optional_fields=["validation_errors", "metadata"],
        )

        super().__init__(
            name="CleaningAgent",
            description="Processes and cleans raw LLM outputs or lists into structured data",
            input_schema=input_schema,
            output_schema=output_schema,
            progress_tracker=progress_tracker,
            content_type=ContentType.SKILLS,
        )

        # Required service dependencies (constructor injection)
        self.llm_service = llm_service
        self.template_manager = template_manager

    async def run_async(
        self, input_data: Any, context: AgentExecutionContext
    ) -> AgentResult:
        """Process and clean raw LLM output or a list of strings."""
        start_time = datetime.now()

        try:
            self.logger.info(
                "Starting cleaning process",
                session_id=context.session_id,
                input_type=type(input_data).__name__,
            )

            cleaned_data: Any = None
            raw_output_for_model = ""
            output_type_for_model = "generic"

            # Route to appropriate cleaning method based on input type
            if isinstance(input_data, list):
                cleaned_data = await self._clean_skills_list(input_data, context)
                raw_output_for_model = json.dumps(input_data)
                output_type_for_model = "skills_list"

            elif isinstance(input_data, dict):
                raw_output = input_data.get("raw_output", "")
                output_type = input_data.get("output_type", "generic")
                raw_output_for_model = raw_output
                output_type_for_model = output_type

                if output_type == "content_item":
                    cleaned_data = await self._clean_content_item(raw_output, context)
                else:
                    cleaned_data = await self._clean_generic_output(raw_output, context)

            elif isinstance(input_data, str):
                raw_output_for_model = input_data
                cleaned_data = await self._clean_generic_output(input_data, context)

            else:
                raise TypeError(
                    f"Unsupported input type for CleaningAgent: {type(input_data)}"
                )

            # Calculate confidence score based on cleaning success
            confidence_score = self._calculate_confidence_score(
                raw_output_for_model, cleaned_data
            )

            processing_time = (datetime.now() - start_time).total_seconds()

            self.logger.info(
                "Cleaning completed successfully",
                session_id=context.session_id,
                confidence_score=confidence_score,
                processing_time=processing_time,
            )

            return AgentResult(
                success=True,
                output_data=CleanedDataModel(
                    cleaned_data=cleaned_data,
                    confidence_score=confidence_score,
                    raw_output=raw_output_for_model,
                    output_type=output_type_for_model,
                ),
                confidence_score=confidence_score,
                processing_time=processing_time,
            )

        except (TypeError, ValueError) as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            error_result = AgentErrorHandler.handle_general_error(
                e, "CleaningAgent", context="run_async"
            )
            return error_result

    async def _clean_skills_list(
        self, skills: List[str], context: AgentExecutionContext
    ) -> List[str]:
        """Refine a list of skill strings by cleaning and standardizing them.

        This method operates on already-structured data and does not perform parsing.

        Args:
            skills: A list of skill strings.
            context: Execution context.

        Returns:
            A list of cleaned skill strings.
        """
        self.logger.info(f"Cleaning skill list with {len(skills)} skills.")
        cleaned_skills = []
        for skill in skills:
            # Example cleaning: standardize capitalization, remove trailing punctuation.
            cleaned_skill = skill.strip().title().rstrip(".,;")
            if cleaned_skill:
                cleaned_skills.append(cleaned_skill)
        self.logger.info(f"Returning {len(cleaned_skills)} cleaned skills.")
        return cleaned_skills

    async def _clean_content_item(
        self, raw_output: str, context: AgentExecutionContext
    ) -> Dict[str, Any]:
        """Clean raw LLM output for content items.

        Args:
            raw_output: Raw LLM response
            context: Execution context

        Returns:
            Cleaned content string
        """
        # Remove common LLM artifacts
        cleaned = raw_output.strip()

        # Remove markdown formatting if present
        cleaned = re.sub(r"\*\*(.*?)\*\*", r"\1", cleaned)  # Bold
        cleaned = re.sub(r"\*(.*?)\*", r"\1", cleaned)  # Italic
        cleaned = re.sub(r"`(.*?)`", r"\1", cleaned)  # Code

        # Remove excessive whitespace
        cleaned = re.sub(r"\n\s*\n", "\n\n", cleaned)
        cleaned = re.sub(r" +", " ", cleaned)

        return cleaned.strip()

    async def _clean_generic_output(
        self, raw_output: str, _context: AgentExecutionContext
    ) -> str:
        """Clean generic raw LLM output.

        Args:
            raw_output: Raw LLM response
            context: Execution context

        Returns:
            Cleaned output string
        """
        # Basic cleaning for generic output
        cleaned = raw_output.strip()

        # Remove excessive whitespace
        cleaned = re.sub(r"\n\s*\n\s*\n", "\n\n", cleaned)
        cleaned = re.sub(r" +", " ", cleaned)

        return cleaned

    def _calculate_confidence_score(self, raw_output: str, cleaned_data: Any) -> float:
        """Calculate confidence score for the cleaning operation.

        Args:
            raw_output: Original raw output
            cleaned_data: Cleaned data

        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not raw_output or not cleaned_data:
            return 0.0

        # Base score
        score = 0.5

        # Increase score if we successfully extracted structured data
        if isinstance(cleaned_data, list) and len(cleaned_data) > 0:
            score += 0.3
        elif isinstance(cleaned_data, str) and len(cleaned_data) > 10:
            score += 0.2

            # Increase score if the output seems well-formatted        if isinstance(cleaned_data, list):
            # Check if skills look reasonable (not too short/long)
            config = get_config()
            reasonable_skills = sum(
                1
                for skill in cleaned_data
                if isinstance(skill, str)
                and config.output.min_skill_length <= len(skill) <= 50
            )
            if reasonable_skills >= len(cleaned_data) * 0.8:
                score += 0.2

        return min(1.0, score)

    @with_node_error_handling("CleaningAgent", "run_as_node")
    async def run_as_node(self, state: AgentState) -> Dict[str, Any]:
        """Execute cleaning agent as a LangGraph node."""
        try:
            from ..models.data_models import ProcessingStatus
            from ..models.validation_schemas import validate_agent_input

            # Validate input
            validated_input = validate_agent_input("cleaning", state)
            self.logger.info("Input validation passed for CleaningAgent")

            # Extract raw output from state - could be from various sources
            raw_output = ""
            output_type = "generic"

            # Check for raw LLM output in state
            if hasattr(state, "raw_llm_output") and state.raw_llm_output:
                raw_output = state.raw_llm_output
                output_type = "skills"  # Default for cleaning agent
            elif hasattr(state, "generated_content") and state.generated_content:
                raw_output = str(state.generated_content)
                output_type = "content_item"

            # Create execution context
            context = AgentExecutionContext(
                session_id=getattr(state, "session_id", "default")
            )

            # Run cleaning
            result = await self.run_async(
                {
                    "raw_output": raw_output,
                    "output_type": output_type,
                    "context": {},
                    "validation_rules": {},
                },
                context,
            )

            if result.success:
                cleaned_data = result.output_data.get("cleaned_data", {})
                return {"cleaned_data": cleaned_data}
            else:
                error_msg = result.error_message or "CleaningAgent failed"
                raise RuntimeError(error_msg)
        except Exception as e:
            self.logger.error(f"CleaningAgent failed: {e}", exc_info=True)
            from ..utils.exceptions import AgentExecutionError

            raise AgentExecutionError(agent_name="CleaningAgent", message=str(e)) from e


def get_cleaning_agent() -> CleaningAgent:
    """Factory function to get a CleaningAgent instance."""
    from ..core.dependency_injection import get_container

    container = get_container()
    container.register_agents()

    return container.get(CleaningAgent, "CleaningAgent")
