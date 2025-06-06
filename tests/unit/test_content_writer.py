import logging
import os
import sys
import json
import time
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path

from src.agents.content_writer_agent import ContentWriterAgent
from src.agents.tools_agent import ToolsAgent
from src.services.llm import LLM
from src.core.state_manager import (
    JobDescriptionData,
    StructuredCV,
    Section,
    Subsection,
    Item,
    ItemStatus,
    ItemType,
)
from src.config.logging_config import setup_test_logging

# Configure test logging
test_log_path = Path("logs/debug/test_content_writer.log")
logger = setup_test_logging("test_content_writer", test_log_path)


def mock_llm_for_testing():
    """Create a mock LLM that logs prompt requests but doesn't actually call the API"""
    original_llm = LLM()
    original_generate = original_llm.generate_content

    # Function to wrap the original generate_content
    def logging_generate_content(prompt):
        logger.info(f"LLM PROMPT RECEIVED:\n{prompt[:300]}...")
        logger.info(f"FULL PROMPT LENGTH: {len(prompt)} characters")

        # Log the prompt to a file for inspection
        with open("last_prompt.txt", "w", encoding="utf-8") as f:
            f.write(prompt)

        # Return a simplified response based on prompt content
        if "executive summary" in prompt.lower():
            logger.info("Executive Summary generation prompt detected")
            return "Experienced data analyst specializing in transforming complex datasets into actionable insights using Python, SQL, and data visualization tools."
        elif "key qualifications" in prompt.lower():
            logger.info("Key Qualifications generation prompt detected")
            return "Data Analysis | SQL | Python | Statistical Analysis | Data Visualization | Problem Solving | Communication"
        elif "professional experience" in prompt.lower():
            logger.info("Professional Experience generation prompt detected")
            return "• Analyzed large datasets to identify trends and opportunities, resulting in 15% increase in operational efficiency\n• Developed automated reporting solutions using Python, saving 10 hours per week\n• Created interactive dashboards to visualize KPIs for executive stakeholders"
        else:
            logger.info("Other prompt type detected")
            return "Mock response for testing: This would be generated content in the real system."

    # Replace the original method with our logging version
    original_llm.generate_content = logging_generate_content
    return original_llm


def create_test_structured_cv():
    """Create a test structured CV with sections marked for regeneration"""
    structured_cv = StructuredCV()

    # Add sections
    sections = [
        {"name": "Executive Summary", "type": "DYNAMIC", "order": 0},
        {"name": "Key Qualifications", "type": "DYNAMIC", "order": 1},
        {"name": "Professional Experience", "type": "DYNAMIC", "order": 2},
        {"name": "Project Experience", "type": "DYNAMIC", "order": 3},
        {"name": "Education", "type": "STATIC", "order": 4},
        {"name": "Certifications", "type": "STATIC", "order": 5},
        {"name": "Languages", "type": "STATIC", "order": 6},
    ]

    for section_info in sections:
        section = Section(
            name=section_info["name"],
            content_type=section_info["type"],
            order=section_info["order"],
        )

        # Add items based on section type
        if section.name == "Executive Summary":
            section.items.append(
                Item(
                    content="Data analyst with experience in SQL and Python.",
                    status=ItemStatus.TO_REGENERATE,
                    item_type=ItemType.SUMMARY_PARAGRAPH,
                )
            )

        elif section.name == "Key Qualifications":
            skills = ["SQL", "Python", "Data Analysis", "Power BI", "Excel"]
            for skill in skills:
                section.items.append(
                    Item(
                        content=skill,
                        status=ItemStatus.TO_REGENERATE,
                        item_type=ItemType.KEY_QUAL,
                    )
                )

        elif section.name == "Professional Experience":
            # Add a subsection for work experience
            subsection = Subsection(name="Data Analyst at XYZ Corp")
            # Add bullet points
            for _ in range(3):
                subsection.items.append(
                    Item(
                        content="Analyzed data and created reports.",
                        status=ItemStatus.TO_REGENERATE,
                        item_type=ItemType.BULLET_POINT,
                    )
                )
            section.subsections.append(subsection)

        elif section.name == "Project Experience":
            # Add a subsection for a project
            subsection = Subsection(name="Data Analysis Project")
            # Add bullet points
            for _ in range(2):
                subsection.items.append(
                    Item(
                        content="Implemented data analysis solutions.",
                        status=ItemStatus.TO_REGENERATE,
                        item_type=ItemType.BULLET_POINT,
                    )
                )
            section.subsections.append(subsection)

        # Add section to structured CV
        structured_cv.sections.append(section)

    return structured_cv


def create_test_job_data():
    """Create test job description data"""
    return JobDescriptionData(
        raw_text="Data Analyst position requiring SQL, Python, and data visualization skills.",
        skills=["SQL", "Python", "Data Analysis", "Data Visualization", "Power BI"],
        experience_level="Mid-level",
        responsibilities=["Analyze data", "Create reports", "Visualize insights"],
        industry_terms=["Data-driven", "Analytics"],
        company_values=["Innovation", "Quality"],
    )


def main():
    """Test the content writer agent directly"""
    logger.info("=========== STARTING CONTENT WRITER TEST ===========")

    # Create a mock LLM
    mock_llm = mock_llm_for_testing()
    logger.info("Mock LLM created for testing")

    # Create tools agent
    tools_agent = ToolsAgent(
        name="ToolsAgent", description="Agent for providing content processing tools"
    )

    # Create content writer agent
    content_writer_agent = ContentWriterAgent(
        name="ContentWriterAgent",
        description="Agent for generating tailored CV content",
        llm=mock_llm,
        tools_agent=tools_agent,
    )

    # Create test data
    structured_cv = create_test_structured_cv()
    job_data = create_test_job_data()

    # Log what we're testing with
    logger.info(
        f"Testing with structured CV containing {len(structured_cv.sections)} sections"
    )

    # Count items marked for regeneration
    items_to_regenerate = []
    for section in structured_cv.sections:
        # Check direct items
        for item in section.items:
            if item.status == ItemStatus.TO_REGENERATE:
                items_to_regenerate.append(item.id)

        # Check items in subsections
        for subsection in section.subsections:
            for item in subsection.items:
                if item.status == ItemStatus.TO_REGENERATE:
                    items_to_regenerate.append(item.id)

    logger.info(f"Found {len(items_to_regenerate)} items marked for regeneration")

    # Create research results (mock data)
    research_results = {
        "key_matches": {
            "skills": ["SQL", "Python", "Data Analysis"],
            "experience": ["Data analysis experience", "Report creation"],
            "summary_points": ["Strong analytical skills", "Technical expertise"],
        },
        "similiar_job_matches": ["Data Analyst", "Business Intelligence Analyst"],
    }

    # Create input data for the content writer agent
    input_data = {
        "structured_cv": structured_cv,
        "job_description_data": job_data,
        "research_results": research_results,
        "regenerate_item_ids": items_to_regenerate,
    }

    # Run the content writer agent
    logger.info("Running content writer agent")
    try:
        result = content_writer_agent.run(input_data)

        # Check the result
        if result:
            logger.info(
                f"Content writer returned a result with {len(result.sections)} sections"
            )

            # Check generated content
            for section in result.sections:
                if section.content_type == "DYNAMIC":
                    logger.info(f"Section: {section.name}")

                    # Check items directly in the section
                    for item in section.items:
                        if item.status == ItemStatus.GENERATED:
                            logger.info(f"  Generated item: {item.content[:50]}...")

                    # Check items in subsections
                    for subsection in section.subsections:
                        logger.info(f"  Subsection: {subsection.name}")
                        for item in subsection.items:
                            if item.status == ItemStatus.GENERATED:
                                logger.info(
                                    f"    Generated item: {item.content[:50]}..."
                                )

            # Save the result to a file
            with open("content_writer_result.json", "w", encoding="utf-8") as f:
                json.dump(result.to_dict(), f, indent=2)
            logger.info("Results saved to content_writer_result.json")
        else:
            logger.error("Content writer did not return a result")

    except Exception as e:
        logger.error(f"Error testing content writer: {str(e)}", exc_info=True)

    logger.info("=========== CONTENT WRITER TEST COMPLETED ===========")


if __name__ == "__main__":
    main()
