"""
Service for LLM-based CV and job description parsing.
"""

from typing import Optional, Any
import json
from ..models.data_models import (
    JobDescriptionData,
    CVParsingResult,
    ContentType,
)
from ..templates.content_templates import ContentTemplateManager
from ..services.llm_service import EnhancedLLMService
from ..config.settings import Settings


class LLMCVParserService:
    """Service for parsing CVs and job descriptions using an LLM."""

    def __init__(
        self,
        llm_service: EnhancedLLMService,
        settings: Settings,
        template_manager: ContentTemplateManager,
    ):
        """Initialize the LLMCVParserService.

        Args:
            llm_service: The EnhancedLLMService instance.
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
            name="cv_parser", content_type=ContentType.CV_PARSING
        )
        if not content_template:
            raise ValueError("CV parsing prompt template 'cv_parser' not found.")

        prompt = self.template_manager.format_template(
            content_template, {"raw_cv_text": cv_text}
        )

        if self.llm_service is None:
            raise RuntimeError("LLM service is not available.")
        parsing_data = await self._generate_and_parse_json(
            prompt=prompt,
            session_id=session_id,
            trace_id=trace_id,
        )
        return CVParsingResult(**parsing_data)

    async def parse_job_description_with_llm(
        self,
        raw_text: str,
        session_id: Optional[str] = None,
        trace_id: Optional[str] = None,
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
            raise ValueError(
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
        )
        return JobDescriptionData(**parsing_data)

    async def _generate_and_parse_json(
        self, prompt: str, session_id: Optional[str], trace_id: Optional[str]
    ) -> Any:
        """Generate content with the LLM and parse the JSON response.

        Args:
            prompt: The prompt to send to the LLM.
            session_id: The session ID for the request.
            trace_id: The trace ID for the request.

        Returns:
            The parsed JSON data from the LLM response.
        """
        # Centralized LLM call and JSON parsing logic
        llm_response = await self.llm_service.generate(
            prompt=prompt,
            session_id=session_id,
            trace_id=trace_id,
        )
        # Assume llm_response is JSON or can be parsed as such
        return json.loads(llm_response)
