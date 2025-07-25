"""
Service for LLM-based CV and job description parsing.
"""

import logging
from typing import Any, Optional

from src.config.settings import Settings
from src.error_handling.exceptions import (LLMResponseParsingError, TemplateError)
from src.models.cv_models import JobDescriptionData
from src.models.llm_data_models import CVParsingResult
from src.models.workflow_models import ContentType
from src.services.llm_service_interface import LLMServiceInterface
from src.templates.content_templates import ContentTemplateManager
from src.utils.json_utils import parse_llm_json_response

logger = logging.getLogger(__name__)


class LLMCVParserService:
    """Service for parsing CVs and job descriptions using an LLM."""

    def __init__(
        self,
        llm_service: LLMServiceInterface,
        settings: Settings,
        template_manager: ContentTemplateManager,
    ):
        """Initialize the LLMCVParserService.

        Args:
            llm_service: The LLMServiceInterface instance.
            settings: The application settings.
            template_manager: The ContentTemplateManager for prompt templates.
        """
        self.llm_service = llm_service
        self.settings = settings
        self.template_manager = template_manager

    async def parse_cv_with_llm(
        self,
        cv_text: str,
        session_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        system_instruction: Optional[str] = None,
    ) -> CVParsingResult:
        """Parse a CV using the LLM.

        Args:
            cv_text: The raw text of the CV.
            session_id: The session ID for the request.
            trace_id: The trace ID for the request.

        Returns:
            The parsed CV data.
        """
        content_template = self.template_manager.get_template(
            name="cv_parsing_prompt", content_type=ContentType.CV_PARSING
        )
        if not content_template:
            raise TemplateError(
                "CV parsing prompt template 'cv_parsing_prompt' not found."
            )

        prompt = self.template_manager.format_template(
            content_template, {"raw_cv_text": cv_text}
        )

        if self.llm_service is None:
            raise RuntimeError("LLM service is not available.")
        parsing_data = await self._generate_and_parse_json(
            prompt=prompt,
            session_id=session_id,
            trace_id=trace_id,
            system_instruction=system_instruction,
        )
        return CVParsingResult(**parsing_data)

    async def parse_job_description_with_llm(
        self,
        raw_text: str,
        session_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        system_instruction: Optional[str] = None,
    ) -> JobDescriptionData:
        """Parse a job description using the LLM.

        Args:
            raw_text: The raw text of the job description.
            session_id: The session ID for the request.
            trace_id: The trace ID for the request.

        Returns:
            The parsed job description data.
        """
        content_template = self.template_manager.get_template(
            name="job_description_parser", content_type=ContentType.JOB_ANALYSIS
        )
        if not content_template:
            raise TemplateError(
                "Job description parsing prompt template 'job_description_parser' not found."
            )

        prompt = self.template_manager.format_template(
            content_template, {"raw_job_description": raw_text}
        )
        if self.llm_service is None:
            raise RuntimeError("LLM service is not available.")
        parsing_data = await self._generate_and_parse_json(
            prompt=prompt,
            session_id=session_id,
            trace_id=trace_id,
            system_instruction=system_instruction,
        )
        parsing_data["raw_text"] = raw_text  # Preserve original raw text
        return JobDescriptionData(**parsing_data)

    async def _generate_and_parse_json(
        self, prompt: str, session_id: Optional[str], trace_id: Optional[str], system_instruction: Optional[str] = None
    ) -> Any:
        """Generates content and robustly parses the JSON output."""
        llm_response = await self.llm_service.generate_content(
            prompt=prompt,
            content_type=ContentType.JSON,  # Assuming JSON output
            session_id=session_id,
            trace_id=trace_id,
            system_instruction=system_instruction,
        )

        raw_text = llm_response.content
        if not raw_text or not isinstance(raw_text, str):
            raise LLMResponseParsingError(
                "Received empty or non-string response from LLM.",
                raw_response=str(raw_text),
                trace_id=trace_id,
            )

        # Use the centralized JSON parsing utility
        try:
            return parse_llm_json_response(raw_text)
        except LLMResponseParsingError as e:
            # Re-raise with additional context (trace_id)
            logger.error(
                "Failed to parse JSON from LLM response: %s",
                e,
                extra={"trace_id": trace_id},
            )
            # Create a new exception with trace_id context
            raise LLMResponseParsingError(
                e.args[0],  # Use the original error message
                raw_response=raw_text,
                trace_id=trace_id,
            ) from e
