from .agent_base import EnhancedAgentBase, AgentExecutionContext, AgentResult
from ..services.llm_service import get_llm_service
from ..core.state_manager import (
    ContentData,
    AgentIO,
    StructuredCV,
    ItemStatus,
    ItemType,
    Section,
    Subsection,
    Item,
)
from typing import Dict, Any, List, Tuple
from datetime import datetime
import re
import asyncio
from ..orchestration.state import AgentState
from ..core.async_optimizer import optimize_async
from ..models.data_models import StructuredCV as PydanticStructuredCV
from ..models.quality_models import QualityCheckResults, QualityStatus, QualityCheck, QualityCheckType, ItemQualityResult, SectionQualityResult

# Set up structured logging
from ..config.logging_config import get_structured_logger
from ..models.data_models import AgentDecisionLog, AgentExecutionLog
from ..models.validation_schemas import validate_agent_input, ValidationError
from ..utils.agent_error_handling import (
    AgentErrorHandler,
    LLMErrorHandler,
    with_error_handling,
    with_node_error_handling
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
        self.llm = llm_service or get_llm_service()

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
                fallback_data = AgentErrorHandler.create_fallback_data("quality_assurance")
                return AgentErrorHandler.handle_validation_error(
                    ve, "quality_assurance", fallback_data, "run_async"
                )
            except Exception as e:
                fallback_data = AgentErrorHandler.create_fallback_data("quality_assurance")
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
        self, job_description_data: Dict[str, Any]
    ) -> Dict[str, List[str]]:
        """
        Extracts key terms from job description data for matching checks.

        Args:
            job_description_data: The job description data

        Returns:
            Dictionary of key terms by category
        """
        key_terms = {
            "skills": [],
            "responsibilities": [],
            "industry_terms": [],
            "company_values": [],
        }

        # Handle None case
        if job_description_data is None:
            return key_terms

        # Extract from dict or object
        if hasattr(job_description_data, "get") and callable(job_description_data.get):
            # It's a dictionary-like object
            key_terms["skills"] = job_description_data.get("skills", [])
            key_terms["responsibilities"] = job_description_data.get(
                "responsibilities", []
            )
            key_terms["industry_terms"] = job_description_data.get("industry_terms", [])
            key_terms["company_values"] = job_description_data.get("company_values", [])
        elif hasattr(job_description_data, "skills"):
            # It's a JobDescriptionData object
            key_terms["skills"] = getattr(job_description_data, "skills", [])
            key_terms["responsibilities"] = getattr(
                job_description_data, "responsibilities", []
            )
            key_terms["industry_terms"] = getattr(
                job_description_data, "industry_terms", []
            )
            key_terms["company_values"] = getattr(
                job_description_data, "company_values", []
            )

        # Normalize and clean terms
        for category, terms in key_terms.items():
            key_terms[category] = [term.lower().strip() for term in terms]

        return key_terms

    def _check_section(
        self, section: Section, key_terms: Dict[str, List[str]]
    ) -> Dict[str, Any]:
        """
        Checks the quality of a section.

        Args:
            section: The section to check
            key_terms: Dictionary of key terms by category

        Returns:
            Dictionary with quality check results for the section
        """
        section_result = {
            "section_name": section.name,
            "item_checks": [],
            "total_items": 0,
            "passed_items": 0,
            "warning_items": 0,
            "failed_items": 0,
            "checks": [],
        }

        # Check items directly in the section
        for item in section.items:
            item_check = self._check_item_quality(item, section, None, key_terms)
            section_result["item_checks"].append(item_check.to_dict())

            # Update section counts
            section_result["total_items"] += 1
            if item_check.status == QualityStatus.SUCCESS:
                section_result["passed_items"] += 1
            elif item_check.status == QualityStatus.WARNING:
                section_result["warning_items"] += 1
            elif item_check.status == QualityStatus.FAILED:
                section_result["failed_items"] += 1

        # Check items in subsections
        for subsection in section.subsections:
            for item in subsection.items:
                item_check = self._check_item_quality(item, section, subsection, key_terms)
                section_result["item_checks"].append(item_check.to_dict())

                # Update section counts
                section_result["total_items"] += 1
                if item_check.status == QualityStatus.SUCCESS:
                    section_result["passed_items"] += 1
                elif item_check.status == QualityStatus.WARNING:
                    section_result["warning_items"] += 1
                elif item_check.status == QualityStatus.FAILED:
                    section_result["failed_items"] += 1

        # Add section-level checks
        if section.name == "Executive Summary":
            # Check if summary is too long or too short
            summary_text = " ".join(
                [item.content for item in section.items if item.content]
            )
            word_count = len(re.findall(r"\b\w+\b", summary_text))

            if word_count < 30:
                section_result["checks"].append(
                    {
                        "check": "summary_length",
                        "status": "warning",
                        "message": f"Executive Summary is too short ({word_count} words, recommend 50-75)",
                    }
                )
            elif word_count > 100:
                section_result["checks"].append(
                    {
                        "check": "summary_length",
                        "status": "warning",
                        "message": f"Executive Summary is too long ({word_count} words, recommend 50-75)",
                    }
                )
            else:
                section_result["checks"].append(
                    {
                        "check": "summary_length",
                        "status": "pass",
                        "message": f"Executive Summary length is good ({word_count} words)",
                    }
                )

        elif section.name == "Key Qualifications":
            # Check if we have enough key qualifications
            if len(section.items) < 4:
                section_result["checks"].append(
                    {
                        "check": "key_quals_count",
                        "status": "warning",
                        "message": f"Only {len(section.items)} Key Qualifications found, recommend at least 6",
                    }
                )
            else:
                section_result["checks"].append(
                    {
                        "check": "key_quals_count",
                        "status": "pass",
                        "message": f"Found {len(section.items)} Key Qualifications",
                    }
                )

        elif section.name == "Professional Experience":
            # Check if experience roles have enough bullet points
            if len(section.subsections) == 0:
                section_result["checks"].append(
                    {
                        "check": "exp_roles_count",
                        "status": "warning",
                        "message": "No roles found in Professional Experience",
                    }
                )
            else:
                role_count = len(section.subsections)
                avg_bullets = (
                    sum(len(ss.items) for ss in section.subsections) / role_count
                )

                if avg_bullets < 2:
                    section_result["checks"].append(
                        {
                            "check": "exp_bullets_avg",
                            "status": "warning",
                            "message": f"Roles have only {avg_bullets:.1f} bullet points on average, recommend 3+",
                        }
                    )
                else:
                    section_result["checks"].append(
                        {
                            "check": "exp_bullets_avg",
                            "status": "pass",
                            "message": f"Roles have {avg_bullets:.1f} bullet points on average",
                        }
                    )

        return section_result

    def _check_item_quality(
        self,
        item: Item,
        section: Section,
        subsection: Subsection = None,
        key_terms: Dict[str, List[str]] = None,
    ) -> ItemQualityResult:
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
        item_result = {
            "item_id": item.id,
            "section": section.name,
            "subsection": subsection.name if subsection else None,
            "content": (
                item.content[:50] + "..." if len(item.content) > 50 else item.content
            ),
            "status": "pass",  # Default to pass
            "checks": [],
        }

        # Skip checks for empty content
        if not item.content.strip():
            item_result["status"] = "fail"
            item_result["checks"].append(
                {
                    "check": "empty_content",
                    "status": "fail",
                    "message": "Content is empty",
                }
            )
            # Convert checks to QualityCheck objects
        quality_checks = []
        for check in item_result["checks"]:
            quality_checks.append(QualityCheck(
                check_type=QualityCheckType.CONTENT,
                description=check["message"],
                passed=check["status"] == "pass",
                severity=check["status"] if check["status"] in ["warning", "fail"] else "info",
                suggestion=check.get("message") if check["status"] != "pass" else None
            ))
        
        # Calculate quality score based on checks
        total_checks = len(quality_checks)
        passed_checks = sum(1 for check in quality_checks if check.passed)
        quality_score = passed_checks / total_checks if total_checks > 0 else 1.0
        
        # Determine status
        if item_result["status"] == "pass":
            status = QualityStatus.SUCCESS
        elif item_result["status"] == "warning":
            status = QualityStatus.WARNING
        else:
            status = QualityStatus.FAILED
        
        return ItemQualityResult(
            status=status,
            item_id=item_result["item_id"],
            quality_score=quality_score,
            checks=quality_checks,
            suggestions=[check.suggestion for check in quality_checks if check.suggestion],
            processing_time_seconds=0.0
        )

        # Check content length
        word_count = len(re.findall(r"\b\w+\b", item.content))

        if item.item_type == ItemType.BULLET_POINT:
            if word_count < 7:
                item_result["checks"].append(
                    {
                        "check": "bullet_length",
                        "status": "warning",
                        "message": f"Bullet point is too short ({word_count} words, recommend 10-20)",
                    }
                )
                item_result["status"] = "warning"
            elif word_count > 25:
                item_result["checks"].append(
                    {
                        "check": "bullet_length",
                        "status": "warning",
                        "message": f"Bullet point is too long ({word_count} words, recommend 10-20)",
                    }
                )
                item_result["status"] = "warning"
            else:
                item_result["checks"].append(
                    {
                        "check": "bullet_length",
                        "status": "pass",
                        "message": f"Bullet point length is good ({word_count} words)",
                    }
                )

        elif item.item_type == ItemType.EXECUTIVE_SUMMARY_PARA:
            if word_count < 30:
                item_result["checks"].append(
                    {
                        "check": "summary_length",
                        "status": "warning",
                        "message": f"Summary paragraph is too short ({word_count} words, recommend 50-75)",
                    }
                )
                item_result["status"] = "warning"
            elif word_count > 100:
                item_result["checks"].append(
                    {
                        "check": "summary_length",
                        "status": "warning",
                        "message": f"Summary paragraph is too long ({word_count} words, recommend 50-75)",
                    }
                )
                item_result["status"] = "warning"
            else:
                item_result["checks"].append(
                    {
                        "check": "summary_length",
                        "status": "pass",
                        "message": f"Summary paragraph length is good ({word_count} words)",
                    }
                )

        elif item.item_type == ItemType.KEY_QUALIFICATION:
            if word_count > 5:
                item_result["checks"].append(
                    {
                        "check": "key_qual_length",
                        "status": "warning",
                        "message": f"Key qualification is too long ({word_count} words, recommend 2-4)",
                    }
                )
                item_result["status"] = "warning"
            else:
                item_result["checks"].append(
                    {
                        "check": "key_qual_length",
                        "status": "pass",
                        "message": f"Key qualification length is good ({word_count} words)",
                    }
                )

        # Check for key terms from job description
        if key_terms:
            found_terms = []
            content_lower = item.content.lower()

            # Check for skills
            for skill in key_terms.get("skills", []):
                if skill.lower() in content_lower:
                    found_terms.append(skill)

            # Check for responsibilities
            for resp in key_terms.get("responsibilities", []):
                if any(term.lower() in content_lower for term in resp.split()):
                    found_terms.append(resp)

            # Check for industry terms
            for term in key_terms.get("industry_terms", []):
                if term.lower() in content_lower:
                    found_terms.append(term)

            # Add check result
            if found_terms:
                item_result["checks"].append(
                    {
                        "check": "key_terms_match",
                        "status": "pass",
                        "message": f"Found {len(found_terms)} key terms",
                        "terms": found_terms,
                    }
                )
            else:
                # Only add a warning for dynamic content that should match job
                if (
                    section.content_type == "DYNAMIC"
                    and item.status != ItemStatus.STATIC
                ):
                    item_result["checks"].append(
                        {
                            "check": "key_terms_match",
                            "status": "warning",
                            "message": "No key terms from job description found",
                        }
                    )
                    if item_result["status"] == "pass":
                        item_result["status"] = "warning"

        # Check for action verbs in bullet points
        if item.item_type == ItemType.BULLET_POINT:
            action_verbs = [
                "achieved",
                "improved",
                "trained",
                "managed",
                "created",
                "developed",
                "increased",
                "decreased",
                "designed",
                "led",
                "implemented",
                "built",
                "coordinated",
                "negotiated",
                "launched",
            ]

            # Check if bullet point starts with an action verb
            first_word = item.content.split()[0].lower().rstrip(",.:;")
            if first_word in action_verbs:
                item_result["checks"].append(
                    {
                        "check": "action_verb",
                        "status": "pass",
                        "message": f"Starts with action verb: {first_word}",
                    }
                )
            else:
                item_result["checks"].append(
                    {
                        "check": "action_verb",
                        "status": "warning",
                        "message": f"Does not start with a common action verb",
                    }
                )
                if item_result["status"] == "pass":
                    item_result["status"] = "warning"

        return item_result

    def _check_overall_cv(
        self, structured_cv: StructuredCV, key_terms: Dict[str, List[str]]
    ) -> List[Dict[str, Any]]:
        """
        Performs overall quality checks on the CV.

        Args:
            structured_cv: The CV structure
            key_terms: Dictionary of key terms by category

        Returns:
            List of overall quality check results
        """
        overall_checks = []

        # Check total word count
        total_words = 0
        for section in structured_cv.sections:
            # Count words in direct items
            for item in section.items:
                total_words += len(re.findall(r"\b\w+\b", item.content))

            # Count words in subsection items
            for subsection in section.subsections:
                for item in subsection.items:
                    total_words += len(re.findall(r"\b\w+\b", item.content))

        if total_words < 200:
            overall_checks.append(
                {
                    "check": "total_word_count",
                    "status": "warning",
                    "message": f"CV content is too short ({total_words} words, recommend 400+)",
                }
            )
        elif total_words > 800:
            overall_checks.append(
                {
                    "check": "total_word_count",
                    "status": "warning",
                    "message": f"CV content is very long ({total_words} words, recommend under 700)",
                }
            )
        else:
            overall_checks.append(
                {
                    "check": "total_word_count",
                    "status": "pass",
                    "message": f"CV content length is good ({total_words} words)",
                }
            )

        # Check if all required sections exist
        required_sections = [
            "Executive Summary",
            "Key Qualifications",
            "Professional Experience",
        ]
        missing_sections = [
            section
            for section in required_sections
            if not any(s.name == section for s in structured_cv.sections)
        ]

        if missing_sections:
            overall_checks.append(
                {
                    "check": "required_sections",
                    "status": "warning",
                    "message": f"Missing required sections: {', '.join(missing_sections)}",
                }
            )
        else:
            overall_checks.append(
                {
                    "check": "required_sections",
                    "status": "pass",
                    "message": "All required sections present",
                }
            )

        # Check relevance to job description
        if key_terms:
            all_key_terms = []
            for terms in key_terms.values():
                all_key_terms.extend(terms)

            # Count total mentions of key terms
            term_mentions = 0
            for section in structured_cv.sections:
                for item in section.items:
                    term_mentions += sum(
                        1
                        for term in all_key_terms
                        if term.lower() in item.content.lower()
                    )

                for subsection in section.subsections:
                    for item in subsection.items:
                        term_mentions += sum(
                            1
                            for term in all_key_terms
                            if term.lower() in item.content.lower()
                        )

            if term_mentions < 5:
                overall_checks.append(
                    {
                        "check": "job_relevance",
                        "status": "warning",
                        "message": f"Low relevance to job description ({term_mentions} key term mentions)",
                    }
                )
            else:
                overall_checks.append(
                    {
                        "check": "job_relevance",
                        "status": "pass",
                        "message": f"Good relevance to job description ({term_mentions} key term mentions)",
                    }
                )

        return overall_checks

    @with_node_error_handling
    @optimize_async("agent_execution", "quality_assurance")
    async def run_as_node(self, state: AgentState) -> AgentState:
        """
        Run the agent as a LangGraph node.
        
        Args:
            state: The current state of the workflow.
            
        Returns:
            Updated AgentState with quality check results.
        """
        logger.info("Quality Assurance Agent: Starting node execution")
        
        try:
            # Extract key terms from job description
            key_terms = self._extract_key_terms(state.job_description_data)
            
            # Perform quality checks
            section_results = []
            for section in state.structured_cv.sections:
                section_result = self._check_section(section, key_terms)
                section_results.append(section_result)
            
            # Perform overall CV checks
            overall_checks = self._check_overall_cv(state.structured_cv, key_terms)
            
            # Create quality check results
            quality_results = {
                "overall_status": "pass",
                "section_results": section_results,
                "overall_checks": overall_checks,
                "timestamp": datetime.now().isoformat(),
                "agent_name": self.name
            }
            
            # Determine overall status
            total_failed = sum(sr.get("failed_items", 0) for sr in section_results)
            total_warnings = sum(sr.get("warning_items", 0) for sr in section_results)
            
            if total_failed > 0:
                quality_results["overall_status"] = "fail"
            elif total_warnings > 0:
                quality_results["overall_status"] = "warning"
            
            # Update state
            state.quality_check_results = quality_results
            
            logger.info(f"Quality Assurance Agent: Completed with status {quality_results['overall_status']}")
            
        except Exception as e:
            error_msg = f"Quality Assurance Agent failed: {str(e)}"
            logger.error(error_msg)
            state.error_messages.append(error_msg)
            
        return state
