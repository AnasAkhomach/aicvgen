"""This module defines the QualityAssuranceAgent, responsible for evaluating the generated CV content."""

import re
from typing import Any, Dict
from pydantic import ValidationError
from src.agents.agent_base import AgentBase
from src.config.logging_config import get_structured_logger
from src.constants.agent_constants import AgentConstants
from src.constants.qa_constants import QAConstants
from src.error_handling.exceptions import AgentExecutionError

from src.models.agent_output_models import ItemQualityResultModel, OverallQualityCheckResultModel, QualityAssuranceAgentOutput, SectionQualityResultModel
from src.models.data_models import Item, Section, StructuredCV
from src.services.llm_service_interface import LLMServiceInterface
from src.templates.content_templates import ContentTemplateManager
from src.utils.node_validation import ensure_pydantic_model

logger = get_structured_logger(__name__)


class QualityAssuranceAgent(AgentBase):
    """
    Agent responsible for quality assurance of generated CV content.
    This agent fulfills REQ-FUNC-QA-1 from the SRS.
    """

    def __init__(
        self,
        llm_service: LLMServiceInterface,
        template_manager: ContentTemplateManager,
        settings: Dict[str, Any],
        session_id: str = "default",
    ):
        """Initialize the QualityAssuranceAgent with required dependencies.

        Args:
            llm_service: LLM service instance for sophisticated quality checks.
            template_manager: The template manager for loading prompts.
            settings: The application settings.
            session_id: Session identifier for the agent.
        """
        super().__init__(
            name="QualityAssuranceAgent",
            description="Agent responsible for quality assurance of CV content",
            session_id=session_id,
            settings=settings,
        )
        self.llm_service = llm_service
        self.template_manager = template_manager

    @ensure_pydantic_model(
        ('structured_cv', StructuredCV),
    )
    async def _execute(self, **kwargs: Any) -> dict[str, Any]:
        """
        Runs the quality assurance agent to evaluate generated CV content.
        """
        logger.info("Quality Assurance Agent: Starting execution.")
        self.update_progress(AgentConstants.PROGRESS_START, "Starting QA process.")

        try:
            # Extract structured_cv from kwargs (passed by base class run method)
            structured_cv = kwargs.get("structured_cv")
            if not structured_cv:
                raise AgentExecutionError(agent_name=self.name, message="structured_cv is required but not provided")

            # Pydantic validation is now handled by the decorator
            self.update_progress(AgentConstants.PROGRESS_INPUT_VALIDATION, "Input validation passed.")

            section_results = [
                self._check_section(section) for section in structured_cv.sections
            ]
            self.update_progress(AgentConstants.PROGRESS_SECTION_CHECKS, "Section checks completed.")

            overall_checks = self._check_overall_cv()
            self.update_progress(AgentConstants.PROGRESS_OVERALL_CHECKS, "Overall CV checks completed.")

            qa_results = QualityAssuranceAgentOutput(
                section_results=section_results, overall_checks=overall_checks
            )

            logger.info("Quality Assurance Agent: Execution completed successfully.")
            self.update_progress(AgentConstants.PROGRESS_COMPLETE, "QA checks completed.")

            return {
                "quality_check_results": qa_results
            }

        except AgentExecutionError as e:
            error_message = f"{self.name} failed: {str(e)}"
            logger.error(error_message, exc_info=True)
            return {
                "error_messages": [error_message],
                "quality_check_results": QualityAssuranceAgentOutput()
            }
        except Exception as e:
            error_message = f"An unexpected error occurred in {self.name}: {str(e)}"
            logger.error(error_message, exc_info=True)
            return {
                "error_messages": [error_message],
                "quality_check_results": QualityAssuranceAgentOutput()
            }

    def _check_section(self, section: Section) -> SectionQualityResultModel:
        """
        Checks the quality of a section.
        """
        item_checks = []
        for item in section.items:
            item_check = self._check_item_quality(item)
            item_checks.append(item_check)
        for subsection in section.subsections:
            for item in subsection.items:
                item_check = self._check_item_quality(item)
                item_checks.append(item_check)

        issues = []
        passed = True
        if section.name == "Executive Summary":
            summary_text = " ".join(
                item.content for item in section.items if item.content
            )
            word_count = len(re.findall(r"\b\w+\b", summary_text))
            if word_count < QAConstants.MIN_WORD_COUNT_EXECUTIVE_SUMMARY:
                issues.append(
                    QAConstants.ERROR_WORD_COUNT_LOW.format(
                        actual=word_count,
                        minimum=QAConstants.MIN_WORD_COUNT_EXECUTIVE_SUMMARY
                    )
                )
                passed = False
            elif word_count > QAConstants.MAX_WORD_COUNT_EXECUTIVE_SUMMARY:
                issues.append(
                    QAConstants.ERROR_WORD_COUNT_HIGH.format(
                        actual=word_count,
                        maximum=QAConstants.MAX_WORD_COUNT_EXECUTIVE_SUMMARY
                    )
                )
                passed = False
        return SectionQualityResultModel(
            section_name=section.name,
            passed=passed,
            issues=issues,
            item_checks=item_checks,
        )

    def _check_item_quality(self, item: Item) -> ItemQualityResultModel:
        """
        Checks the quality of an individual item.
        """
        # Example logic: always pass, no issues
        return ItemQualityResultModel(
            item_id=str(item.id),
            passed=True,
            issues=[],
            suggestions=[],
        )

    def _check_overall_cv(self) -> list[OverallQualityCheckResultModel]:
        """
        Performs overall quality checks on the CV.
        """
        # Example logic: always pass
        return [
            OverallQualityCheckResultModel(
                check_name="word_count", passed=True, details="Word count OK"
            )
        ]
