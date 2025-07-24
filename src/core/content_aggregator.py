"""Content Aggregator for CV Generation System

This module provides the ContentAggregator class that collects individual content pieces
from different agents and assembles them into a complete StructuredCV structure.
"""

from typing import Any, Dict, List
from uuid import uuid4

from src.config.logging_config import get_structured_logger
from src.error_handling.exceptions import CATCHABLE_EXCEPTIONS
from src.models.data_models import (Item, ItemStatus, MetadataModel, Section, StructuredCV)

logger = get_structured_logger(__name__)


class ContentAggregator:
    """Aggregates individual content pieces from agents into a complete StructuredCV structure."""

    def __init__(self):
        """Initialize the content aggregator with content type mappings."""
        # Map content types to StructuredCV section names
        self.section_map = {
            "qualification": "Key Qualifications",
            "executive_summary": "Executive Summary",
            "experience": "Professional Experience",
            "skills": "Technical Skills",
            "project": "Projects",
            "education": "Education",
            "certification": "Certifications",
            "language": "Languages",
        }

    def aggregate_results(
        self, task_results: List[Dict[str, Any]], state_manager=None
    ) -> StructuredCV:
        """Aggregate task results into a complete StructuredCV structure.

        Args:
            task_results: List of task results from various agents
            state_manager: Optional state manager to get existing StructuredCV data

        Returns:
            StructuredCV object containing aggregated CV content
        """
        logger.info(
            f"Starting content aggregation for {len(task_results)} task results"
        )

        # Start with existing StructuredCV from state manager if available
        if state_manager:
            structured_cv = state_manager.get_structured_cv()
            if structured_cv:
                logger.info("Using existing StructuredCV from state manager")
            else:
                structured_cv = StructuredCV()
                logger.info("Created new StructuredCV structure")
        else:
            structured_cv = StructuredCV()
            logger.info("Created new StructuredCV structure")

        content_found = False

        for i, result in enumerate(task_results):
            try:
                agent_type = result.get("agent_type", "unknown")
                logger.info(
                    "Processing result %s from agent: %s",
                    extra={"result_index": i, "agent_type": agent_type},
                )

                # Handle content_writer agent results
                if agent_type == "content_writer":
                    content_found = (
                        self._process_content_writer_result(result, structured_cv)
                        or content_found
                    )

                # Handle generic content structure
                else:
                    content_found = (
                        self._process_generic_result(result, structured_cv)
                        or content_found
                    )

            except CATCHABLE_EXCEPTIONS as e:
                logger.error(
                    "Error processing result %s: %s",
                    extra={"result_index": i, "error": str(e)},
                )
                continue

        # Populate Big 10 skills from state manager if available
        if state_manager:
            self._populate_big_10_skills(structured_cv, state_manager)

        if content_found:
            section_names = [section.name for section in structured_cv.sections]
            logger.info(
                f"Content aggregation successful. Sections populated: {section_names}"
            )
        else:
            logger.warning("No valid content found during aggregation")

        return structured_cv

    def _process_content_writer_result(
        self, result: Dict[str, Any], structured_cv: StructuredCV
    ) -> bool:
        """Process content writer agent results.

        Args:
            result: Task result from content writer agent
            structured_cv: StructuredCV structure to populate

        Returns:
            True if content was successfully processed, False otherwise
        """
        try:
            # Extract content from agent result dictionary
            agent_content = result.get("content", {})

            if isinstance(agent_content, dict) and "content" in agent_content:
                # Extract the actual content from nested structure
                actual_content = agent_content["content"]
                content_type = agent_content.get("content_type", "unknown")

                logger.info(
                    f"Found content writer content: type={content_type}, content_length={len(str(actual_content))}"
                )

                # Map content type to appropriate section
                if content_type in self.section_map:
                    section_name = self.section_map[content_type]
                    self._add_content_to_section(
                        structured_cv, section_name, actual_content
                    )
                    logger.info(
                        "Mapped %s content to %s section",
                        extra={
                            "content_type": content_type,
                            "section_name": section_name,
                        },
                    )
                    return True
                else:
                    # Try to infer content type from content
                    return self._infer_and_assign_content(actual_content, structured_cv)

            # Handle direct string content (fallback)
            elif isinstance(agent_content, str):
                return self._infer_and_assign_content(agent_content, structured_cv)

        except CATCHABLE_EXCEPTIONS as e:
            logger.error("Error processing content writer result", error=str(e))

        return False

    # _process_optimization_result method removed - ContentOptimizationAgent was never implemented

    def _process_generic_result(
        self, result: Dict[str, Any], structured_cv: StructuredCV
    ) -> bool:
        """Process generic agent results.

        Args:
            result: Generic task result
            structured_cv: StructuredCV structure to populate

        Returns:
            True if content was successfully processed, False otherwise
        """
        try:
            # Try to extract content from various possible structures
            content = None

            if "content" in result:
                content = result["content"]
            elif "result" in result and isinstance(result["result"], dict):
                if "content" in result["result"]:
                    content = result["result"]["content"]

            if content:
                if isinstance(content, dict):
                    # Try to merge structured content into sections
                    for content_type, section_name in self.section_map.items():
                        if content_type in content and content[content_type]:
                            self._add_content_to_section(
                                structured_cv, section_name, content[content_type]
                            )
                    return True
                elif isinstance(content, str):
                    # Try to infer content type
                    return self._infer_and_assign_content(content, structured_cv)

        except CATCHABLE_EXCEPTIONS as e:
            logger.error("Error processing generic result", error=str(e))

        return False

    def _add_content_to_section(
        self, structured_cv: StructuredCV, section_name: str, content: Any
    ) -> bool:
        """Add content to a specific section in the StructuredCV.

        Args:
            structured_cv: The StructuredCV to modify
            section_name: Name of the section to add content to
            content: Content to add (string, list, or dict)

        Returns:
            True if content was successfully added, False otherwise
        """
        try:
            # Find existing section or create new one
            section = self._find_or_create_section(structured_cv, section_name)

            # Handle different content types
            if isinstance(content, str):
                # Add as a single item
                item = Item(id=uuid4(), content=content, metadata=MetadataModel())
                section.items.append(item)
            elif isinstance(content, list):
                # Add each item in the list
                for item_content in content:
                    if isinstance(item_content, str):
                        item = Item(
                            id=uuid4(), content=item_content, metadata=MetadataModel()
                        )
                        section.items.append(item)
            elif isinstance(content, dict):
                # Handle structured content (could be subsections)
                for key, value in content.items():
                    if isinstance(value, str):
                        item = Item(
                            id=uuid4(),
                            content=value,
                            metadata=MetadataModel(extra={"key": key}),
                        )
                        section.items.append(item)

            return True
        except CATCHABLE_EXCEPTIONS as e:
            logger.error(
                "Error adding content to section %s: %s",
                extra={"section_name": section_name, "error": str(e)},
            )
            return False

    def _find_or_create_section(
        self, structured_cv: StructuredCV, section_name: str
    ) -> Section:
        """Find an existing section or create a new one.

        Args:
            structured_cv: The StructuredCV to search/modify
            section_name: Name of the section to find or create

        Returns:
            The found or newly created Section object
        """
        # Look for existing section
        for section in structured_cv.sections:
            if section.name.lower() == section_name.lower():
                return section

        # Create new section if not found
        new_section = Section(
            id=uuid4(),
            name=section_name,
            content_type="DYNAMIC",
            order=len(structured_cv.sections),
            status=ItemStatus.GENERATED,
        )
        structured_cv.sections.append(new_section)
        logger.info("Created new section", section_name=section_name)
        return new_section

    def _infer_and_assign_content(
        self, content: str, structured_cv: StructuredCV
    ) -> bool:
        """Infer content type and assign to appropriate section.

        Args:
            content: String content to analyze and assign
            structured_cv: StructuredCV structure to populate

        Returns:
            True if content was successfully assigned, False otherwise
        """
        try:
            content_lower = content.lower()

            # Check for skills first (more specific keywords)
            if any(
                keyword in content_lower
                for keyword in [
                    "proficient in",
                    "skilled in",
                    "technologies",
                    "programming languages",
                    "technical skills",
                ]
            ) or content_lower.startswith(
                ("python", "java", "javascript", "react", "node", "aws", "docker")
            ):
                self._add_content_to_section(structured_cv, "Technical Skills", content)
                logger.info("Inferred content as skills")
                return True

            # Check for work experience (specific employment terms)
            elif any(
                keyword in content_lower
                for keyword in [
                    "worked as",
                    "employed at",
                    "position at",
                    "role at",
                    "engineer at",
                    "developer at",
                ]
            ):
                self._add_content_to_section(
                    structured_cv, "Professional Experience", content
                )
                logger.info("Inferred content as experience")
                return True

            # Check for projects
            elif any(
                keyword in content_lower
                for keyword in [
                    "developed project",
                    "built project",
                    "project portfolio",
                    "github project",
                ]
            ):
                self._add_content_to_section(structured_cv, "Projects", content)
                logger.info("Inferred content as project")
                return True

            # Check for summary/profile content (broader terms)
            elif any(
                keyword in content_lower
                for keyword in [
                    "professional with",
                    "experienced professional",
                    "summary",
                    "profile",
                    "objective",
                    "about me",
                ]
            ):
                self._add_content_to_section(
                    structured_cv, "Executive Summary", content
                )
                logger.info("Inferred content as executive summary")
                return True

            else:
                # Default to executive summary if no specific type detected
                self._add_content_to_section(
                    structured_cv, "Executive Summary", content
                )
                logger.info("Assigned content to executive summary as default")
                return True

        except CATCHABLE_EXCEPTIONS as e:
            logger.error("Error inferring content type", error=str(e))

        return False

    def _populate_big_10_skills(
        self, structured_cv: StructuredCV, state_manager
    ) -> None:
        """Populate Big 10 skills data from the state manager.

        Args:
            structured_cv: StructuredCV structure to populate
            state_manager: State manager containing Big 10 skills data
        """
        try:
            # Get the structured CV from state manager using public method
            state_cv = state_manager.get_structured_cv()

            if state_cv and hasattr(state_cv, "big_10_skills"):
                # Copy Big 10 skills data
                structured_cv.big_10_skills = state_cv.big_10_skills or []

                # Create or update Key Qualifications section with Big 10 skills
                if structured_cv.big_10_skills:
                    qualifications_section = self._find_or_create_section(
                        structured_cv, "Key Qualifications"
                    )

                    # Clear existing skills items to avoid duplicates
                    qualifications_section.items = [
                        item
                        for item in qualifications_section.items
                        if not item.metadata.extra.get("is_big_10_skill", False)
                    ]

                    # Add each Big 10 skill as an item
                    for skill in structured_cv.big_10_skills:
                        item = Item(
                            id=uuid4(),
                            content=f"• {skill}",
                            metadata=MetadataModel(extra={"is_big_10_skill": True}),
                        )
                        qualifications_section.items.append(item)

                    logger.info(
                        f"Populated {len(structured_cv.big_10_skills)} Big 10 skills"
                    )
                else:
                    logger.info("No Big 10 skills found in state manager")
            else:
                logger.info(
                    "No structured CV or Big 10 skills data found in state manager"
                )

        except CATCHABLE_EXCEPTIONS as e:
            logger.error("Error populating Big 10 skills", error=str(e))

    def validate_structured_cv(self, structured_cv: StructuredCV) -> bool:
        """Validate that the StructuredCV has meaningful content.

        Args:
            structured_cv: StructuredCV to validate

        Returns:
            True if StructuredCV is valid, False otherwise
        """
        try:
            # Check if at least one section has content
            if not structured_cv.sections:
                logger.warning("StructuredCV validation failed: no sections found")
                return False

            # Check if at least one section has items
            for section in structured_cv.sections:
                if section.items:
                    # Check if at least one item has meaningful content
                    for item in section.items:
                        if item.content and item.content.strip():
                            return True

            logger.warning(
                "StructuredCV validation failed: no sections with content items"
            )
            return False

        except CATCHABLE_EXCEPTIONS as e:
            logger.error("Error validating StructuredCV", error=str(e))
            return False
