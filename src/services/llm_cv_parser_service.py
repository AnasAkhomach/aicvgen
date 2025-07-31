"""
Service for LLM-based CV and job description parsing.
"""

import logging
from typing import Optional

from src.config.settings import Settings
from src.error_handling.exceptions import LLMResponseParsingError, TemplateError
from src.models.cv_models import JobDescriptionData
from src.models.llm_data_models import (
    CVParsingResult,
    CVParsingStructuredOutput,
    JobDescriptionStructuredOutput,
)
from src.models.workflow_models import ContentType
from src.services.llm_service_interface import LLMServiceInterface
from src.templates.content_templates import ContentTemplateManager

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
        """Parse a CV using the LLM with structured output.

        Args:
            cv_text: The raw text of the CV.
            session_id: The session ID for the request.
            trace_id: The trace ID for the request.
            system_instruction: Optional system instruction for the LLM.

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

        # Use structured output instead of manual JSON parsing
        structured_output = await self.llm_service.generate_structured_content(
            prompt=prompt,
            response_model=CVParsingStructuredOutput,
            session_id=session_id,
            trace_id=trace_id,
            system_instruction=system_instruction,
        )

        # Convert structured output to CVParsingResult for backward compatibility
        return CVParsingResult(
            personal_info=structured_output.personal_info,
            sections=structured_output.sections,
        )

    async def parse_job_description_with_llm(
        self,
        raw_text: str,
        session_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        system_instruction: Optional[str] = None,
    ) -> JobDescriptionData:
        """Parse a job description using the LLM with structured output.

        Args:
            raw_text: The raw text of the job description.
            session_id: The session ID for the request.
            trace_id: The trace ID for the request.
            system_instruction: Optional system instruction for the LLM.

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

        # Use structured output instead of manual JSON parsing
        structured_output = await self.llm_service.generate_structured_content(
            prompt=prompt,
            response_model=JobDescriptionStructuredOutput,
            session_id=session_id,
            trace_id=trace_id,
            system_instruction=system_instruction,
        )

        # Convert structured output to JobDescriptionData for backward compatibility
        return JobDescriptionData(
            raw_text=raw_text,  # Preserve original raw text
            job_title=structured_output.job_title,
            company_name=structured_output.company_name,
            main_job_description_raw=structured_output.main_job_description_raw,
            skills=structured_output.skills,
            experience_level=structured_output.experience_level,
            responsibilities=structured_output.responsibilities,
            industry_terms=structured_output.industry_terms,
            company_values=structured_output.company_values,
            error=structured_output.error,
        )
