"""CleaningAgent for processing and cleaning raw LLM outputs.

This agent implements the "Generate -> Clean" pattern described in Task 3.2 & 3.4.
It takes raw LLM output and processes it into structured, clean content.
"""

from typing import List, Any
import re
import json
from datetime import datetime

from ..agents.agent_base import EnhancedAgentBase, AgentExecutionContext, AgentResult
from ..utils.agent_error_handling import AgentErrorHandler, with_node_error_handling
from ..models.data_models import AgentIO, ContentType
from ..orchestration.state import AgentState
from ..models.cleaning_agent_models import CleaningAgentNodeResult
from ..services.llm_service import EnhancedLLMService


class CleaningAgent(EnhancedAgentBase):
    """Agent responsible for cleaning and structuring raw LLM outputs.

    This agent processes raw LLM responses to extract structured data,
    particularly for the "Big 10" skills generation feature.
    """

    def __init__(self, llm_service):
        input_schema = AgentIO(
            description="Raw LLM output and processing instructions",
            required_fields=["raw_output", "output_type"],
            optional_fields=["context", "validation_rules"],
        )

        output_schema = AgentIO(
            description="Cleaned and structured output",
            required_fields=["cleaned_data", "confidence_score"],
            optional_fields=["validation_errors", "metadata"],
        )

        super().__init__(
            name="CleaningAgent",
            description="Processes and cleans raw LLM outputs into structured data",
            input_schema=input_schema,
            output_schema=output_schema,
            content_type=ContentType.SKILLS,
        )

        self.llm_service = llm_service

    async def run_async(
        self, input_data: Any, context: AgentExecutionContext
    ) -> AgentResult:
        """Process and clean raw LLM output.

        Args:
            input_data: Dictionary containing:
                - raw_output: The raw LLM response text
                - output_type: Type of output to clean (e.g., 'big_10_skills')
                - context: Optional context for cleaning
                - validation_rules: Optional validation rules
            context: Execution context

        Returns:
            AgentResult with cleaned data
        """
        start_time = datetime.now()

        try:
            self.logger.info(
                "Starting cleaning process",
                session_id=context.session_id,
                output_type=input_data.get("output_type"),
                raw_output_length=len(input_data.get("raw_output", "")),
            )

            raw_output = input_data.get("raw_output", "")
            output_type = input_data.get("output_type", "generic")

            # Route to appropriate cleaning method
            if output_type == "big_10_skills":
                cleaned_data = await self._clean_big_10_skills(raw_output, context)
            elif output_type == "content_item":
                cleaned_data = await self._clean_content_item(raw_output, context)
            else:
                cleaned_data = await self._clean_generic_output(raw_output, context)

            # Calculate confidence score based on cleaning success
            confidence_score = self._calculate_confidence_score(
                raw_output, cleaned_data
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
                output_data={
                    "cleaned_data": cleaned_data,
                    "confidence_score": confidence_score,
                    "raw_output": raw_output,
                    "output_type": output_type,
                },
                confidence_score=confidence_score,
                processing_time=processing_time,
            )

        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            error_result = AgentErrorHandler.handle_general_error(
                e, "CleaningAgent", context="run_async"
            )

            return AgentResult(
                success=False,
                output_data=None,
                confidence_score=0.0,
                processing_time=processing_time,
                error_message=error_result.error_message,
            )

    async def _clean_big_10_skills(
        self, raw_output: str, context: AgentExecutionContext
    ) -> List[str]:
        """Clean raw LLM output to extract Big 10 skills list.

        Args:
            raw_output: Raw LLM response containing skills
            context: Execution context

        Returns:
            List of cleaned skill strings
        """
        try:
            # Try to parse as JSON first using centralized method
            if raw_output.strip().startswith("{") or raw_output.strip().startswith("["):
                try:
                    # Use centralized JSON extraction from parent class
                    json_str = super()._extract_json_from_response(raw_output)
                    parsed = json.loads(json_str)
                    if isinstance(parsed, list):
                        return [str(skill).strip() for skill in parsed[:10]]
                    if isinstance(parsed, dict) and "skills" in parsed:
                        return [str(skill).strip() for skill in parsed["skills"][:10]]
                except (json.JSONDecodeError, Exception):
                    # Log the JSON parsing failure and continue with regex fallback
                    self.logger.warning(
                        "JSON parsing failed for skills extraction, using regex fallback",
                        session_id=getattr(context, "session_id", None),
                    )

            # Extract skills using regex patterns
            skills = []

            # Pattern 1: Numbered list (1. Skill, 2. Skill, etc.)
            numbered_pattern = r"\d+\.\s*([^\n]+)"
            numbered_matches = re.findall(numbered_pattern, raw_output)
            if numbered_matches:
                skills.extend([skill.strip() for skill in numbered_matches])

            # Pattern 2: Bullet points (- Skill, * Skill, etc.)
            bullet_pattern = r"[\-\*•]\s*([^\n]+)"
            bullet_matches = re.findall(bullet_pattern, raw_output)
            if bullet_matches and not skills:
                skills.extend([skill.strip() for skill in bullet_matches])

            # Pattern 3: Comma-separated list
            if not skills:
                # Look for comma-separated skills
                lines = raw_output.split("\n")
                for line in lines:
                    if "," in line and len(line.split(",")) > 2:
                        skills.extend([skill.strip() for skill in line.split(",")])
                        break

            # Clean and validate skills
            cleaned_skills = []
            for skill in skills:
                # Remove common prefixes/suffixes
                skill = re.sub(
                    r"^(skill:?|ability:?|competency:?)\s*",
                    "",
                    skill,
                    flags=re.IGNORECASE,
                )
                skill = skill.strip(".,;:\"'")

                # Skip empty or very short skills
                if len(skill.strip()) > 2:
                    cleaned_skills.append(skill.strip())

            # Return top 10 skills
            return cleaned_skills[:10]

        except (ValueError, TypeError, AttributeError, json.JSONDecodeError) as e:
            self.logger.warning(
                "Failed to clean Big 10 skills, returning fallback",
                error=str(e),
                session_id=context.session_id,
            )
            # Fallback: split by lines and take first 10 non-empty lines
            lines = [line.strip() for line in raw_output.split("\n") if line.strip()]
            return lines[:10]

    async def _clean_content_item(
        self, raw_output: str, _context: AgentExecutionContext
    ) -> str:
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

        # Increase score if the output seems well-formatted
        if isinstance(cleaned_data, list):
            # Check if skills look reasonable (not too short/long)
            reasonable_skills = sum(
                1
                for skill in cleaned_data
                if isinstance(skill, str) and 3 <= len(skill) <= 50
            )
            if reasonable_skills >= len(cleaned_data) * 0.8:
                score += 0.2

        return min(1.0, score)

    @with_node_error_handling("CleaningAgent", "run_as_node")
    async def run_as_node(self, state: AgentState) -> CleaningAgentNodeResult:
        """Execute cleaning agent as a LangGraph node."""
        from ..models.data_models import ProcessingStatus

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

        # Return state updates as a Pydantic model
        return CleaningAgentNodeResult(
            cleaned_data=(
                result.output_data.get("cleaned_data", {}) if result.success else {}
            ),
            processing_status=(
                ProcessingStatus.COMPLETED
                if result.success
                else ProcessingStatus.FAILED
            ),
        )


def get_cleaning_agent() -> CleaningAgent:
    """Factory function to get a CleaningAgent instance."""
    return CleaningAgent()
