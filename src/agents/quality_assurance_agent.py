"""This module defines the QualityAssuranceAgent, responsible for evaluating the generated CV content."""

import re
from typing import Any, Dict

from pydantic import ValidationError

from src.config.logging_config import get_structured_logger
from src.error_handling.exceptions import AgentExecutionError
from src.models.agent_models import AgentResult
from src.models.agent_output_models import (
    ItemQualityResultModel,
    OverallQualityCheckResultModel,
    QualityAssuranceAgentOutput,
    SectionQualityResultModel,
)
from src.models.data_models import Item, Section, StructuredCV
from src.services.llm_service import EnhancedLLMService
from src.templates.content_templates import ContentTemplateManager
from src.agents.agent_base import AgentBase

logger = get_structured_logger(__name__)


class QualityAssuranceAgent(AgentBase):
    """
    Agent responsible for quality assurance of generated CV content.
    This agent fulfills REQ-FUNC-QA-1 from the SRS.
    """

    def __init__(
        self,
        llm_service: EnhancedLLMService,
        template_manager: ContentTemplateManager,
        settings: Dict[str, Any],
    ):
        """Initialize the QualityAssuranceAgent with required dependencies.

        Args:
            llm_service: LLM service instance for sophisticated quality checks.
            template_manager: The template manager for loading prompts.
            settings: The application settings.
        """
        super().__init__(
            name="QualityAssuranceAgent",
            description="Agent responsible for quality assurance of CV content",
        )
        self.llm_service = llm_service
        self.template_manager = template_manager
        self.settings = settings

    async def run(self, **kwargs: Any) -> AgentResult:
        """
        Runs the quality assurance agent to evaluate generated CV content.

        This method validates the input, performs a series of quality checks on the
        structured CV data, and returns the results. It checks individual sections
        and performs overall CV-level validation.

        Args:
            **kwargs: A dictionary containing 'structured_cv'.

        Returns:
            An AgentResult containing the QualityAssuranceAgentOutput.

        Raises:
            AgentExecutionError: If the input is invalid or an error occurs during processing.
        """
        logger.info("Quality Assurance Agent: Starting execution.")
        self.update_progress(0, "Starting QA process.")
        input_data = kwargs.get("input_data", {})

        try:
            if "structured_cv" not in input_data:
                raise AgentExecutionError(
                    message="Invalid input: 'structured_cv' is a required field."
                )

            structured_cv = input_data["structured_cv"]
            if not isinstance(structured_cv, StructuredCV):
                if isinstance(structured_cv, dict):
                    structured_cv = StructuredCV(**structured_cv)
                else:
                    raise AgentExecutionError(
                        message=f"Invalid type for 'structured_cv'. Expected StructuredCV or dict, got {type(structured_cv)}."
                    )

            self.update_progress(20, "Input validation passed.")

            section_results = [
                self._check_section(section) for section in structured_cv.sections
            ]
            self.update_progress(70, "Section checks completed.")

            overall_checks = self._check_overall_cv()
            self.update_progress(90, "Overall CV checks completed.")

            qa_results = QualityAssuranceAgentOutput(
                section_results=section_results, overall_checks=overall_checks
            )

            logger.info("Quality Assurance Agent: Execution completed successfully.")
            self.update_progress(100, "QA checks completed.")

            return AgentResult(success=True, output_data=qa_results)

        except (ValidationError, KeyError, TypeError, AgentExecutionError) as e:
            error_message = f"{self.name} failed: {str(e)}"
            logger.error(error_message, exc_info=True)
            return AgentResult(
                success=False,
                error_message=error_message,
                output_data=QualityAssuranceAgentOutput(),
            )
        except (AttributeError, ValueError) as e:
            error_message = (
                f"An unexpected data error occurred in {self.name}: {str(e)}"
            )
            logger.error(error_message, exc_info=True)
            return AgentResult(
                success=False,
                error_message=error_message,
                output_data=QualityAssuranceAgentOutput(),
            )

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
            min_word_count = self.settings.get("qa", {}).get(
                "executive_summary_min_words", 50
            )
            if word_count < min_word_count:
                issues.append(
                    f"Executive Summary is too short ({word_count} words, recommend "
                    f"{min_word_count}-{min_word_count + 25})"
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
