import logging
import os
import sys
import json
import time
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path

from src.core.orchestrator import Orchestrator
from src.agents.parser_agent import ParserAgent
from src.utils.template_renderer import TemplateRenderer
from src.agents.vector_store_agent import VectorStoreAgent
from src.services.vector_db import VectorDB, VectorStoreConfig
from src.agents.content_writer_agent import ContentWriterAgent
from src.agents.research_agent import ResearchAgent
from src.agents.cv_analyzer_agent import CVAnalyzerAgent
from src.agents.tools_agent import ToolsAgent
from src.agents.formatter_agent import FormatterAgent
from src.agents.quality_assurance_agent import QualityAssuranceAgent
from src.services.llm import LLM
from src.core.state_manager import AgentIO
from src.config.logging_config import setup_test_logging

# Configure test logging
test_log_path = Path("logs/debug/test_workflow.log")
logger = setup_test_logging("test_workflow", test_log_path)


def mock_llm_for_testing():
    """Create a mock LLM that logs prompt requests but doesn't actually call the API"""
    original_llm = LLM()
    original_generate = original_llm.generate_content

    # Function to wrap the original generate_content
    def logging_generate_content(prompt):
        logger.info(f"LLM PROMPT RECEIVED:\n{prompt[:300]}...")
        logger.info(f"FULL PROMPT LENGTH: {len(prompt)} characters")

        # For testing purposes, return a simplified mock response
        if "job description" in prompt.lower():
            logger.info("Job description parsing prompt detected")
            return json.dumps(
                {
                    "skills": ["Python", "SQL", "Data Analysis"],
                    "experience_level": "Mid-level",
                    "responsibilities": ["Data analysis", "Reporting"],
                    "industry_terms": ["Data-driven", "Analytics"],
                    "company_values": ["Innovation", "Quality"],
                }
            )
        elif "executive summary" in prompt.lower():
            logger.info("Executive Summary generation prompt detected")
            return "Experienced data analyst specializing in transforming complex datasets into actionable insights using Python, SQL, and data visualization tools."
        elif "key qualifications" in prompt.lower():
            logger.info("Key Qualifications generation prompt detected")
            return "Data Analysis | SQL | Python | Statistical Analysis | Data Visualization | Problem Solving | Communication"
        elif "professional experience" in prompt.lower():
            logger.info("Professional Experience generation prompt detected")
            return "• Analyzed large datasets to identify trends and opportunities, resulting in 15% increase in operational efficiency\n• Developed automated reporting solutions using Python, saving 10 hours per week\n• Created interactive dashboards to visualize KPIs for executive stakeholders"
        else:
            # Call the original function for other cases, but with a timeout warning
            logger.info("Other prompt type detected, using original LLM")
            return "Mock response for testing: This would be generated content in the real system."

    # Replace the original method with our logging version
    original_llm.generate_content = logging_generate_content
    return original_llm


def load_cv_template():
    """Load the CV template from file"""
    try:
        with open("src/templates/cv_template.md", "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error loading CV template: {e}")
        return None


def load_job_description():
    """Load a sample job description"""
    return """
    Data Analyst

    Company Overview:
    We are a growing technology company seeking a skilled Data Analyst to join our team. The ideal candidate will have experience working with large datasets and the ability to communicate insights effectively.

    Responsibilities:
    - Analyze large datasets to identify trends and opportunities
    - Create dashboards and reports to visualize key metrics
    - Collaborate with cross-functional teams to support data-driven decision making
    - Develop and maintain SQL queries for data extraction
    - Automate reporting processes using Python

    Requirements:
    - Bachelor's degree in a relevant field
    - 2+ years of experience in data analysis
    - Proficiency in SQL and Python
    - Experience with data visualization tools (Power BI, Tableau)
    - Strong problem-solving and communication skills

    We offer:
    - Competitive salary
    - Flexible work arrangements
    - Opportunities for professional development
    - Collaborative and innovative work environment
    """


def main():
    """Run the full workflow test"""
    logger.info("=========== STARTING WORKFLOW TEST ===========")

    # Create a mock LLM for testing
    mock_llm = mock_llm_for_testing()
    logger.info("Mock LLM created for testing")

    # Initialize agents
    parser_agent = ParserAgent(
        name="ParserAgent",
        description="Agent for parsing job descriptions",
        llm=mock_llm,
    )

    vector_db_config = VectorStoreConfig(dimension=768, index_type="IndexFlatL2")
    vector_db = VectorDB(config=vector_db_config)

    vector_store_agent = VectorStoreAgent(
        name="VectorStoreAgent",
        description="Agent for managing vector store",
        model=mock_llm,
        input_schema=AgentIO(input={}, output={}, description="vector store agent"),
        output_schema=AgentIO(input={}, output={}, description="vector store agent"),
        vector_db=vector_db,
    )

    tools_agent = ToolsAgent(
        name="ToolsAgent", description="Agent for providing content processing tools"
    )

    content_writer_agent = ContentWriterAgent(
        name="ContentWriterAgent",
        description="Agent for generating tailored CV content",
        llm=mock_llm,
        tools_agent=tools_agent,
    )

    research_agent = ResearchAgent(
        name="ResearchAgent",
        description="Agent for researching job-related information",
        llm=mock_llm,
        vector_db=vector_db,
    )

    cv_analyzer_agent = CVAnalyzerAgent(
        name="CVAnalyzerAgent", description="Agent for analyzing user CVs", llm=mock_llm
    )

    formatter_agent = FormatterAgent(
        name="FormatterAgent", description="Agent for formatting CV content"
    )

    quality_assurance_agent = QualityAssuranceAgent(
        name="QualityAssuranceAgent",
        description="Agent for performing quality checks on CV content",
        llm=mock_llm,
    )

    template_renderer = TemplateRenderer(
        name="TemplateRenderer",
        description="Agent for rendering CV templates",
        model=mock_llm,
        input_schema=AgentIO(input={}, output={}, description="template renderer"),
        output_schema=AgentIO(input={}, output={}, description="template renderer"),
    )

    # Initialize orchestrator
    logger.info("Initializing orchestrator")
    orchestrator = Orchestrator(
        parser_agent=parser_agent,
        template_renderer=template_renderer,
        vector_store_agent=vector_store_agent,
        content_writer_agent=content_writer_agent,
        research_agent=research_agent,
        cv_analyzer_agent=cv_analyzer_agent,
        tools_agent=tools_agent,
        formatter_agent=formatter_agent,
        quality_assurance_agent=quality_assurance_agent,
        llm=mock_llm,
    )

    # Load job description and CV
    job_description = load_job_description()
    user_cv = load_cv_template()

    # Log what we're working with
    logger.info(f"Job description length: {len(job_description)} characters")
    logger.info(f"CV template length: {len(user_cv)} characters")

    # Test parser agent directly first
    logger.info("\n== TESTING PARSER AGENT DIRECTLY ==")
    parse_result = parser_agent.run(
        {"job_description": job_description, "cv_text": user_cv}
    )

    if "job_description_data" in parse_result:
        logger.info("Job description parsed successfully")
        job_data_dict = parse_result["job_description_data"].to_dict()
        logger.info(f"Extracted skills: {job_data_dict.get('skills', [])}")
    else:
        logger.error("Failed to parse job description")

    if "structured_cv" in parse_result and parse_result["structured_cv"]:
        logger.info("CV parsed successfully")
        structured_cv = parse_result["structured_cv"]
        logger.info(f"Number of sections: {len(structured_cv.sections)}")

        # Log section types
        for section in structured_cv.sections:
            logger.info(f"Section: {section.name} - Type: {section.content_type}")
    else:
        logger.error("Failed to parse CV")

    # Test content writer directly
    logger.info("\n== TESTING CONTENT WRITER DIRECTLY ==")
    job_data = parse_result["job_description_data"]
    structured_cv = parse_result["structured_cv"]

    # Log section count and types
    dynamic_sections = [
        section
        for section in structured_cv.sections
        if section.content_type == "DYNAMIC"
    ]
    static_sections = [
        section
        for section in structured_cv.sections
        if section.content_type == "STATIC"
    ]
    logger.info(f"Dynamic sections: {len(dynamic_sections)}")
    logger.info(f"Static sections: {len(static_sections)}")

    # Run full workflow
    logger.info("\n== RUNNING FULL WORKFLOW ==")
    try:
        result = orchestrator.run_workflow(
            job_description=job_description,
            user_cv_data=user_cv,
            workflow_id="test-workflow-" + str(int(time.time())),
        )

        logger.info(
            f"Workflow completed with status: {result.get('status', 'unknown')}"
        )
        logger.info(f"Final workflow stage: {result.get('stage', 'unknown')}")

        # Check if we have structured CV data
        if "structured_cv" in result and result["structured_cv"]:
            structured_cv = result["structured_cv"]
            logger.info(
                f"Final structured CV has {len(structured_cv.sections)} sections"
            )

            # Check dynamic sections
            for section in structured_cv.sections:
                if section.content_type == "DYNAMIC":
                    logger.info(f"Dynamic section: {section.name}")
                    # For the Executive Summary, check if content was generated
                    if section.name == "Executive Summary" and section.items:
                        has_content = any(
                            item.content.strip() for item in section.items
                        )
                        logger.info(f"Executive Summary has content: {has_content}")

        # Save the result
        with open("workflow_test_result.json", "w", encoding="utf-8") as f:
            # Convert result to JSON-compatible format
            json_result = {}
            for key, value in result.items():
                if hasattr(value, "to_dict"):
                    json_result[key] = value.to_dict()
                elif isinstance(value, (dict, list, str, int, float, bool, type(None))):
                    json_result[key] = value
                else:
                    json_result[key] = str(value)
            json.dump(json_result, f, indent=2)

        logger.info("Test results saved to workflow_test_result.json")

    except Exception as e:
        logger.error(f"Error in workflow test: {str(e)}", exc_info=True)

    logger.info("=========== WORKFLOW TEST COMPLETED ===========")


if __name__ == "__main__":
    main()
