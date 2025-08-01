"""
This module defines the UserCVParserAgent, responsible for parsing user CVs.
"""

from typing import Any

from src.agents.agent_base import AgentBase
from src.config.logging_config import get_structured_logger
from src.error_handling.exceptions import AgentExecutionError
from src.models.cv_models import StructuredCV
from src.services.llm_cv_parser_service import LLMCVParserService
from src.constants.agent_constants import AgentConstants

logger = get_structured_logger(__name__)


class UserCVParserAgent(AgentBase):
    """Agent responsible for parsing user CVs."""

    def __init__(self, parser_service: LLMCVParserService, session_id: str, **kwargs):
        """Initialize the UserCVParserAgent with required dependencies."""
        super().__init__(
            name="UserCVParserAgent",
            description="Agent responsible for parsing user CVs into structured format",
            session_id=session_id,
        )
        self._parser_service = parser_service
        self.logger = logger

    async def run(self, cv_text: str) -> StructuredCV:
        """Parse the raw CV text into a StructuredCV object using the dedicated service.

        Args:
            cv_text: The raw CV text to parse

        Returns:
            StructuredCV: The parsed and structured CV data
        """
        self.logger.info("Starting CV parsing with LLMCVParserService.")

        if not cv_text or not cv_text.strip():
            raise AgentExecutionError(
                agent_name=self.name, message="Cannot parse empty CV text"
            )

        self.update_progress(AgentConstants.PROGRESS_MAIN_PROCESSING, "Parsing CV")

        # Parse CV directly to StructuredCV
        structured_cv = await self._parser_service.parse_cv_to_structured_cv(
            cv_text=cv_text, session_id=self.session_id
        )

        self.update_progress(AgentConstants.PROGRESS_COMPLETE, "Parsing completed")
        self.logger.info("Successfully parsed CV into StructuredCV object.")
        return structured_cv

    async def _execute(self, **kwargs: Any) -> dict[str, Any]:
        """Execute the core parsing logic (legacy interface for workflow compatibility)."""
        cv_text = kwargs.get("cv_text")
        if not cv_text:
            raise AgentExecutionError(
                agent_name=self.name, message="cv_text is required for parsing"
            )

        structured_cv = await self.run(cv_text)
        return {"structured_cv": structured_cv}
