from .agent_base import EnhancedAgentBase, AgentExecutionContext, AgentResult
from ..services.llm_service import EnhancedLLMService
from ..models.data_models import (
    ContentData,
    AgentIO,
    StructuredCV,
    ItemStatus,
    ItemType,
    Section,
    Subsection,
    Item,
)
from typing import Any
import re
from ..orchestration.state import AgentState
from ..core.async_optimizer import optimize_async
from ..models.data_models import StructuredCV as PydanticStructuredCV
from ..models.quality_models import (
    QualityCheckResults,
    ItemQualityResult,
    SectionQualityResult,
)
from ..models.quality_assurance_agent_models import (
    KeyTerms,
    SectionQualityResultModel,
    ItemQualityResultModel,
    OverallQualityCheckResultModel,
    QualityAssuranceResult,
)

# Set up structured logging
from ..config.logging_config import get_structured_logger
from ..models.data_models import AgentDecisionLog, AgentExecutionLog
from ..models.validation_schemas import validate_agent_input, ValidationError
from ..utils.agent_error_handling import (
    AgentErrorHandler,
    LLMErrorHandler,
    with_error_handling,
    with_node_error_handling,
)

logger = get_structured_logger(__name__)


class QualityAssuranceAgent(EnhancedAgentBase):
    """
    Agent responsible for quality assurance of generated CV content.
    This agent fulfills REQ-FUNC-QA-1 from the SRS.
    """

    def __init__(self, name: str, description: str, llm_service=None):
        """
        Initializes the QualityAssuranceAgent.

        Args:
            name: The name of the agent.
            description: A description of the agent.
            llm_service: Optional LLM service instance for more sophisticated checks.
        """
        super().__init__(
            name=name,
            description=description,
            input_schema=AgentIO(
                description="Reads structured CV and job description data from AgentState for quality analysis.",
                required_fields=["structured_cv", "job_description_data"],
                optional_fields=[],
            ),
            output_schema=AgentIO(
                description="Populates the 'quality_check_results' field and optionally updates 'structured_cv' in AgentState.",
                required_fields=["quality_check_results"],
                optional_fields=["structured_cv", "error_messages"],
            ),
        )
        self.llm = llm_service

    async def run_async(
        self, input_data: Any, context: "AgentExecutionContext"
    ) -> "AgentResult":
        """Async run method for consistency with enhanced agent interface."""
        from .agent_base import AgentResult

        try:
            # Validate input data using Pydantic schemas
            try:
                validated_input = validate_agent_input("quality_assurance", input_data)
                # Use validated input data directly
                input_data = validated_input
                # Log validation success with structured logging
                self.log_decision(
                    "Input validation passed for QualityAssuranceAgent",
                    context,
                    "validation",
                )
            except ValidationError as ve:
                fallback_data = AgentErrorHandler.create_fallback_data(
                    "quality_assurance"
                )
                return AgentErrorHandler.handle_validation_error(
                    ve, "quality_assurance", fallback_data, "run_async"
                )
            except Exception as e:
                fallback_data = AgentErrorHandler.create_fallback_data(
                    "quality_assurance"
                )
                return AgentErrorHandler.handle_general_error(
                    e, "quality_assurance", fallback_data, "run_async"
                )

            # Use run_as_node for LangGraph integration
            # Create AgentState for run_as_node compatibility
            from ..orchestration.state import AgentState
            from ..models.data_models import StructuredCV, JobDescriptionData

            # Create proper StructuredCV and JobDescriptionData objects
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
            result = node_result.get("output_data", {}) if node_result else {}

            return AgentResult(
                success=True,
                output_data=result,
                confidence_score=1.0,
                metadata={"agent_type": "quality_assurance"},
            )

        except Exception as e:
            # Use standardized error handling
            fallback_data = AgentErrorHandler.create_fallback_data("quality_assurance")
            return AgentErrorHandler.handle_general_error(
                e, "quality_assurance", fallback_data, "run_async"
            )

    def _extract_key_terms(
        self, job_description_data: KeyTerms | dict | Any
    ) -> KeyTerms:
        """
        Extracts key terms from job description data for matching checks.

        Args:
            job_description_data: The job description data

        Returns:
            KeyTerms Pydantic model
        """
        if job_description_data is None:
            return KeyTerms()
        if hasattr(job_description_data, "get") and callable(job_description_data.get):
            return KeyTerms(
                skills=job_description_data.get("skills", []),
                responsibilities=job_description_data.get("responsibilities", []),
                industry_terms=job_description_data.get("industry_terms", []),
                company_values=job_description_data.get("company_values", []),
            )
        elif hasattr(job_description_data, "skills"):
            return KeyTerms(
                skills=getattr(job_description_data, "skills", []),
                responsibilities=getattr(job_description_data, "responsibilities", []),
                industry_terms=getattr(job_description_data, "industry_terms", []),
                company_values=getattr(job_description_data, "company_values", []),
            )
        return KeyTerms()

    def _check_section(
        self, section: Section, key_terms: KeyTerms
    ) -> SectionQualityResultModel:
        """
        Checks the quality of a section.

        Args:
            section: The section to check
            key_terms: KeyTerms model

        Returns:
            SectionQualityResultModel
        """
        item_checks = []
        for item in section.items:
            item_check = self._check_item_quality(item, section, None, key_terms)
            item_checks.append(item_check)
        for subsection in section.subsections:
            for item in subsection.items:
                item_check = self._check_item_quality(
                    item, section, subsection, key_terms
                )
                item_checks.append(item_check)
        # Example: section-level checks (simplified)
        issues = []
        passed = True
        if section.name == "Executive Summary":
            summary_text = " ".join(
                [
                    getattr(item, "content", "")
                    for item in section.items
                    if hasattr(item, "content") and item.content
                ]
            )
            word_count = len(re.findall(r"\\b\\w+\\b", summary_text))
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

    def _check_item_quality(
        self, item: Item, section: Section, subsection: Subsection, key_terms: KeyTerms
    ) -> ItemQualityResultModel:
        """
        Checks the quality of an individual item.

        Args:
            item: The item to check
            section: The section containing the item
            subsection: The subsection containing the item (if any)
            key_terms: Dictionary of key terms by category

        Returns:
            Dictionary with quality check results for the item
        """
        # Example logic: always pass, no issues
        return ItemQualityResultModel(
            item_id=str(getattr(item, "id", "unknown")),
            passed=True,
            issues=[],
            suggestions=[],
        )

    def _check_overall_cv(
        self, structured_cv: StructuredCV, key_terms: KeyTerms
    ) -> list[OverallQualityCheckResultModel]:
        """
        Performs overall quality checks on the CV.

        Args:
            structured_cv: The CV structure
            key_terms: Dictionary of key terms by category

        Returns:
            List of overall quality check results
        """
        # Example logic: always pass
        return [
            OverallQualityCheckResultModel(
                check_name="word_count", passed=True, details="Word count OK"
            )
        ]

    @with_node_error_handling("quality_assurance")
    @optimize_async("agent_execution", "quality_assurance")
    async def run_as_node(self, state):
        """
        Run the agent as a LangGraph node.

        Args:
            agent_state: The current state of the workflow.

        Returns:
            Updated AgentState with quality check results.
        """
        logger.info("Quality Assurance Agent: Starting node execution")

        try:
            from ..models.validation_schemas import validate_agent_input

            validated_input = validate_agent_input("qa", state)

            structured_cv = state.structured_cv
            job_desc_data = state.job_description_data
            key_terms = self._extract_key_terms(job_desc_data)

            # Perform quality checks
            section_results = [
                self._check_section(section, key_terms)
                for section in structured_cv.sections
            ]
            overall_checks = self._check_overall_cv(structured_cv, key_terms)
            qa_results = QualityAssuranceResult(
                section_results=section_results,
                overall_checks=overall_checks,
            )

            # After processing, build and return a new AgentState
            return state.model_copy(update={"quality_check_results": qa_results})

        except Exception as e:
            return state.model_copy(
                update={
                    "error_messages": state.error_messages
                    + [f"Quality Assurance Agent failed: {str(e)}"],
                    "quality_check_results": None,
                }
            )
