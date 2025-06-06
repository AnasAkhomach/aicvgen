from src.agents.agent_base import AgentBase
from src.services.llm import LLM
from src.core.state_manager import (
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
import logging
import re

# Set up logging
logger = logging.getLogger(__name__)


class QualityAssuranceAgent(AgentBase):
    """
    Agent responsible for quality assurance of generated CV content.
    This agent fulfills REQ-FUNC-QA-1 from the SRS.
    """

    def __init__(self, name: str, description: str, llm: LLM = None):
        """
        Initializes the QualityAssuranceAgent.

        Args:
            name: The name of the agent.
            description: A description of the agent.
            llm: Optional LLM instance for more sophisticated checks.
        """
        super().__init__(
            name=name,
            description=description,
            input_schema=AgentIO(
                input={
                    "structured_cv": StructuredCV,
                    "job_description_data": Dict[str, Any],
                },
                output={
                    "quality_check_results": Dict[str, Any],
                    "updated_structured_cv": StructuredCV,
                },
                description="Checks quality of generated CV content.",
            ),
            output_schema=AgentIO(
                input={
                    "structured_cv": StructuredCV,
                    "job_description_data": Dict[str, Any],
                },
                output={
                    "quality_check_results": Dict[str, Any],
                    "updated_structured_cv": StructuredCV,
                },
                description="Results of quality checks and updated CV structure.",
            ),
        )
        self.llm = llm

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs quality checks on the generated CV content.

        Args:
            input_data: A dictionary containing:
                - structured_cv (StructuredCV): The CV structure to check
                - job_description_data (Dict): Job description data for context

        Returns:
            A dictionary containing:
                - quality_check_results: Results of the quality checks
                - updated_structured_cv: The updated CV structure with any fixes
        """
        structured_cv = input_data.get("structured_cv")
        job_description_data = input_data.get("job_description_data", {})

        if not structured_cv:
            logger.warning("No StructuredCV provided to QualityAssuranceAgent")
            return {
                "quality_check_results": {"error": "No CV data provided"},
                "updated_structured_cv": StructuredCV(),
            }

        # Extract job requirements for matching checks
        key_terms = self._extract_key_terms(job_description_data)

        # Perform quality checks
        check_results = {
            "item_checks": [],
            "section_checks": [],
            "overall_checks": [],
            "summary": {
                "total_items": 0,
                "passed_items": 0,
                "warning_items": 0,
                "failed_items": 0,
            },
        }

        # Check each section and item
        for section in structured_cv.sections:
            section_check = self._check_section(section, key_terms)
            check_results["section_checks"].append(section_check)

            # Update summary counts
            check_results["summary"]["total_items"] += section_check["total_items"]
            check_results["summary"]["passed_items"] += section_check["passed_items"]
            check_results["summary"]["warning_items"] += section_check["warning_items"]
            check_results["summary"]["failed_items"] += section_check["failed_items"]

            # Add individual item checks to the results
            check_results["item_checks"].extend(section_check["item_checks"])

        # Perform overall CV checks
        overall_check = self._check_overall_cv(structured_cv, key_terms)
        check_results["overall_checks"] = overall_check

        # Calculate overall quality score
        if check_results["summary"]["total_items"] > 0:
            quality_score = (
                check_results["summary"]["passed_items"] / check_results["summary"]["total_items"]
            ) * 100
            check_results["summary"]["quality_score"] = quality_score
        else:
            check_results["summary"]["quality_score"] = 0

        # Return results and updated CV
        return {
            "quality_check_results": check_results,
            "updated_structured_cv": structured_cv,
        }

    def _extract_key_terms(self, job_description_data: Dict[str, Any]) -> Dict[str, List[str]]:
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

        # Extract from dict or object
        if hasattr(job_description_data, "get") and callable(job_description_data.get):
            # It's a dictionary-like object
            key_terms["skills"] = job_description_data.get("skills", [])
            key_terms["responsibilities"] = job_description_data.get("responsibilities", [])
            key_terms["industry_terms"] = job_description_data.get("industry_terms", [])
            key_terms["company_values"] = job_description_data.get("company_values", [])
        elif hasattr(job_description_data, "skills"):
            # It's a JobDescriptionData object
            key_terms["skills"] = getattr(job_description_data, "skills", [])
            key_terms["responsibilities"] = getattr(job_description_data, "responsibilities", [])
            key_terms["industry_terms"] = getattr(job_description_data, "industry_terms", [])
            key_terms["company_values"] = getattr(job_description_data, "company_values", [])

        # Normalize and clean terms
        for category, terms in key_terms.items():
            key_terms[category] = [term.lower().strip() for term in terms]

        return key_terms

    def _check_section(self, section: Section, key_terms: Dict[str, List[str]]) -> Dict[str, Any]:
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
            item_check = self._check_item(item, section, None, key_terms)
            section_result["item_checks"].append(item_check)

            # Update section counts
            section_result["total_items"] += 1
            if item_check["status"] == "pass":
                section_result["passed_items"] += 1
            elif item_check["status"] == "warning":
                section_result["warning_items"] += 1
            elif item_check["status"] == "fail":
                section_result["failed_items"] += 1

        # Check items in subsections
        for subsection in section.subsections:
            for item in subsection.items:
                item_check = self._check_item(item, section, subsection, key_terms)
                section_result["item_checks"].append(item_check)

                # Update section counts
                section_result["total_items"] += 1
                if item_check["status"] == "pass":
                    section_result["passed_items"] += 1
                elif item_check["status"] == "warning":
                    section_result["warning_items"] += 1
                elif item_check["status"] == "fail":
                    section_result["failed_items"] += 1

        # Add section-level checks
        if section.name == "Executive Summary":
            # Check if summary is too long or too short
            summary_text = " ".join([item.content for item in section.items if item.content])
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
                avg_bullets = sum(len(ss.items) for ss in section.subsections) / role_count

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

    def _check_item(
        self,
        item: Item,
        section: Section,
        subsection: Subsection = None,
        key_terms: Dict[str, List[str]] = None,
    ) -> Dict[str, Any]:
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
            "content": (item.content[:50] + "..." if len(item.content) > 50 else item.content),
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
            return item_result

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

        elif item.item_type == ItemType.SUMMARY_PARAGRAPH:
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

        elif item.item_type == ItemType.KEY_QUAL:
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
                if section.content_type == "DYNAMIC" and item.status != ItemStatus.STATIC:
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
                        1 for term in all_key_terms if term.lower() in item.content.lower()
                    )

                for subsection in section.subsections:
                    for item in subsection.items:
                        term_mentions += sum(
                            1 for term in all_key_terms if term.lower() in item.content.lower()
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
