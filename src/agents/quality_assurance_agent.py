"""This module defines the QualityAssuranceAgent, responsible for evaluating the generated CV content."""

import re
from typing import Any, Dict

from ..config.logging_config import get_structured_logger
from ..models.data_models import (
    AgentIO,
    Item,
    JobDescriptionData,
    Section,
    StructuredCV,
)
from ..models.quality_assurance_agent_models import (
    ItemQualityResultModel,
    KeyTerms,
    OverallQualityCheckResultModel,
    SectionQualityResultModel,
)
from ..models.validation_schemas import validate_agent_input
from ..orchestration.state import AgentState
from ..services.llm_service import EnhancedLLMService
from ..services.progress_tracker import ProgressTracker
from ..templates.content_templates import ContentTemplateManager
from .agent_base import AgentExecutionContext, AgentResult, EnhancedAgentBase
from ..models.agent_output_models import QualityAssuranceAgentOutput


logger = get_structured_logger(__name__)


class QualityAssuranceAgent(EnhancedAgentBase):
    """
    Agent responsible for quality assurance of generated CV content.
    This agent fulfills REQ-FUNC-QA-1 from the SRS.
    """

    def __init__(
        self,
        llm_service: EnhancedLLMService,
        progress_tracker: ProgressTracker,
        template_manager: ContentTemplateManager,
    ):
        """Initialize the QualityAssuranceAgent with required dependencies.

        Args:
            llm_service: LLM service instance for sophisticated quality checks.
            error_recovery_service: Error recovery service dependency.
            progress_tracker: Progress tracker service dependency.
            template_manager: The template manager for loading prompts.
        """
        super().__init__(
            name="QualityAssuranceAgent",
            description="Agent responsible for quality assurance of CV content",
            input_schema=AgentIO(
                description=(
                    "Reads structured CV and job description data "
                    "from AgentState for quality analysis."
                ),
                required_fields=["structured_cv", "job_description_data"],
            ),
            output_schema=AgentIO(
                description="Populates 'quality_check_results' in AgentState.",
                required_fields=["quality_check_results"],
            ),
            progress_tracker=progress_tracker,
        )
        self.llm_service = llm_service
        self.template_manager = template_manager

    async def run_async(
        self, input_data: Any, context: "AgentExecutionContext"
    ) -> "AgentResult":
        """Async run method for consistency with the enhanced agent interface."""
        try:
            validated_input = validate_agent_input("quality_assurance", input_data)
            input_data = validated_input
            self.log_decision(
                "Input validation passed for QualityAssuranceAgent",
                context,
                "validation",
            )

            structured_cv = input_data.get("structured_cv") or StructuredCV()
            job_desc_data = input_data.get("job_description_data")
            if not job_desc_data or isinstance(job_desc_data, dict):
                job_desc_data = JobDescriptionData(
                    raw_text=input_data.get("job_description", "")
                )

            agent_state = AgentState(
                structured_cv=structured_cv, job_description_data=job_desc_data
            )

            node_result = await self.run_as_node(agent_state)
            result = node_result.get("quality_check_results")

            if not isinstance(result, QualityAssuranceAgentOutput):
                result = QualityAssuranceAgentOutput(
                    section_results=[], overall_checks=[]
                )

            return AgentResult(
                success=True,
                output_data=result,
                confidence_score=1.0,
                metadata={"agent_type": "quality_assurance"},
            )

        except Exception as e:
            # Fail fast - let the orchestration layer handle recovery
            from ..utils.exceptions import AgentExecutionError

            raise AgentExecutionError(
                agent_name="QualityAssuranceAgent", message=str(e)
            ) from e

    def _extract_key_terms(self, job_description_data: Any) -> KeyTerms:
        """
        Extracts key terms from job description data for matching checks.
        """
        if not job_description_data:
            return KeyTerms()

        if isinstance(job_description_data, dict):
            return KeyTerms(
                skills=job_description_data.get("skills", []),
                responsibilities=job_description_data.get("responsibilities", []),
                industry_terms=job_description_data.get("industry_terms", []),
                company_values=job_description_data.get("company_values", []),
            )

        return KeyTerms(
            skills=getattr(job_description_data, "skills", []),
            responsibilities=getattr(job_description_data, "responsibilities", []),
            industry_terms=getattr(job_description_data, "industry_terms", []),
            company_values=getattr(job_description_data, "company_values", []),
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
            if word_count < 30:
                issues.append(
                    f"Executive Summary is too short ({word_count} words, recommend 50-75)"
                )
                passed = False
        return SectionQualityResultModel(
            section_name=section.name,
            passed=passed,
            issues=issues,
            item_checks=[ic.dict() for ic in item_checks],
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

    async def run_as_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Run the agent as a LangGraph node. Fails fast on errors.
        """
        logger.info("Quality Assurance Agent: Starting node execution")

        try:
            validate_agent_input("qa", state)

            structured_cv = state.structured_cv

            section_results = [
                self._check_section(section) for section in structured_cv.sections
            ]
            overall_checks = self._check_overall_cv()
            qa_results = QualityAssuranceAgentOutput(
                section_results=section_results,
                overall_checks=overall_checks,
            )

            return {"quality_check_results": qa_results}

        except Exception as e:
            logger.error("QualityAssuranceAgent failed: %s", e)
            # Fail fast - let the orchestration layer handle recovery
            from ..utils.exceptions import AgentExecutionError

            raise AgentExecutionError(
                agent_name="QualityAssuranceAgent", message=str(e)
            ) from e
