"""
This module defines specialized agents that build upon the base agent functionalities.

It includes:
- The `EnhancedParserAgent`, which acts as an adapter for the `ParserAgent`.
  It wraps the core parsing logic within a standardized `run` interface,
  making it compatible with the enhanced agent execution framework. This allows the
  `ParserAgent`, originally designed for a different workflow (like LangGraph),
  to be seamlessly integrated into the broader multi-agent system.

- Helper functions, such as `_check_item_quality_basic`, which provide common,
  reusable logic for tasks like content quality assessment. This avoids code duplication
  and ensures consistent quality checks across different agents.

The primary goal of this module is to orchestrate and enhance the functionality of core agents,
making them easier to use, test, and integrate into complex workflows.
"""

from typing import Any, Dict, Optional

from pydantic import ValidationError

from src.agents.agent_base import AgentBase
from src.agents.parser_agent import ParserAgent
from src.models.agent_models import AgentResult
from src.models.agent_output_models import ParserAgentOutput
from src.models.data_models import JobDescriptionData, StructuredCV
from src.config.logging_config import get_structured_logger
from src.error_handling.exceptions import (
    AgentExecutionError,
    DataConversionError,
    LLMResponseParsingError,
)
from src.core.dependency_injection import get_container

logger = get_structured_logger("specialized_agents")


# Helper function for quality checks (used by other agents)
def _check_item_quality_basic(item: dict, quality_criteria: dict) -> dict:
    """Basic quality check for content items."""
    issues = []
    warnings = []
    score = 1.0

    content = item.get("content", "")

    # Length checks
    min_length = quality_criteria.get("min_length", 20)
    max_length = quality_criteria.get("max_length", 1000)

    if len(content) < min_length:
        issues.append(f"Content too short (minimum {min_length} characters)")
        score -= 0.3
    elif len(content) > max_length:
        warnings.append(
            f"Content might be too long (maximum {max_length} characters recommended)"
        )
        score -= 0.1

    # Grammar and formatting checks
    if not content.strip():
        issues.append("Empty content")
        score = 0.0
    elif content != content.strip():
        warnings.append("Content has leading/trailing whitespace")
        score -= 0.05

    # Professional language check (basic)
    unprofessional_words = ["awesome", "cool", "stuff", "things"]
    for word in unprofessional_words:
        if word.lower() in content.lower():
            warnings.append(f"Consider replacing informal word: '{word}'")
            score -= 0.05

    # Confidence score check
    confidence = item.get("confidence_score", 1.0)
    if confidence < 0.5:
        warnings.append("Low confidence score from content generation")
        score -= 0.1

    return {
        "passed": len(issues) == 0,
        "score": max(score, 0.0),
        "issues": issues,
        "warnings": warnings,
    }


class EnhancedParserAgent(AgentBase):
    """Enhanced wrapper for the original ParserAgent.

    This agent acts as an adapter, providing a standard `run` interface
    for the `ParserAgent`.
    """

    def __init__(self, parser_agent: ParserAgent, session_id: str):
        """Initializes the EnhancedParserAgent."""
        super().__init__(
            name="EnhancedParserAgent",
            description="Enhanced parser agent for CV and job description parsing",
            session_id=session_id,
        )
        self.parser_agent = parser_agent

    async def run(
        self, cv_text: str, job_description_text: Optional[str] = None
    ) -> AgentResult[ParserAgentOutput]:
        """
        Executes the underlying ParserAgent's logic and returns the result.
        """
        self.update_progress(0, "Starting enhanced parsing process.")

        try:
            # The underlying parser agent now has a standard run method
            parser_result = await self.parser_agent.run(
                cv_text=cv_text, job_description_text=job_description_text
            )

            if parser_result.is_error():
                logger.warning(
                    "Underlying ParserAgent execution failed.",
                    error=parser_result.error.message,
                )
            else:
                self.update_progress(100, "Enhanced parsing completed successfully.")

            # Propagate the result (success or error) from the inner agent
            return parser_result

        except Exception as e:
            logger.error(
                "An unexpected error occurred in EnhancedParserAgent wrapper",
                error=str(e),
                exc_info=True,
            )
            err = AgentExecutionError(
                agent_name=self.name,
                message=f"An unexpected wrapper error occurred in EnhancedParserAgent: {e}",
                details={"original_exception": str(e)},
            )
            self.update_progress(
                100, f"Enhanced parser failed with unexpected wrapper error: {e}"
            )
            return AgentResult.error(agent_name=self.name, error=err)

    async def run_as_node(self, state: Any) -> Dict[str, Any]:
        """The EnhancedParserAgent is an adapter and should not be used as a graph node directly."""
        raise NotImplementedError(
            "EnhancedParserAgent is an adapter and does not implement run_as_node itself."
        )


def create_cv_analysis_agent(session_id: str) -> Any:
    """Factory for CV Analysis Agent via DI container."""
    container = get_container()
    return container.get_by_name("cv_analysis_agent", session_id=session_id)


def create_quality_assurance_agent(session_id: str) -> Any:
    """Factory for Quality Assurance Agent via DI container."""
    container = get_container()
    return container.get_by_name("qa_agent", session_id=session_id)


def create_enhanced_parser_agent(session_id: str) -> Any:
    """Factory for Enhanced Parser Agent via DI container."""
    container = get_container()
    return container.get_by_name("enhanced_parser_agent", session_id=session_id)


def create_formatter_agent(session_id: str) -> Any:
    """Factory for Formatter Agent via DI container."""
    container = get_container()
    return container.get_by_name("formatter_agent", session_id=session_id)


def create_cleaning_agent(session_id: str) -> Any:
    """Factory for Cleaning Agent via DI container."""
    container = get_container()
    return container.get_by_name("cleaning_agent", session_id=session_id)


def create_enhanced_content_writer_agent(session_id: str) -> Any:
    """Factory for Enhanced Content Writer Agent via DI container."""
    container = get_container()
    return container.get_by_name("enhanced_content_writer_agent", session_id=session_id)


def create_research_agent(session_id: str) -> Any:
    """Factory for Research Agent via DI container."""
    container = get_container()
    return container.get_by_name("research_agent", session_id=session_id)
