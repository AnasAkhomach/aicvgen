"""Isolated test script for PDF generation pipeline debugging.

This script allows quick testing and debugging of the PDF generation process
by directly invoking the FormatterAgent with sample data.
"""

import pytest
import asyncio
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.agents.formatter_agent import FormatterAgent
from src.models.data_models import (
    StructuredCV,
    JobDescriptionData,
    Section,
    Item,
    ItemStatus,
    ItemType,
    PersonalInfo,
    Experience,
    Education,
    Skill,
)
from src.config.settings import get_config, AppConfig
from src.config.logging_config import get_structured_logger

pytestmark = pytest.mark.asyncio

logger = get_structured_logger(__name__)


def create_sample_job_data() -> JobDescriptionData:
    """Create sample job description data for testing."""
    return JobDescriptionData(
        raw_text="Senior Software Engineer position at TechCorp requiring Python, React, and AWS experience.",
        skills=["Python", "React", "AWS", "Docker", "PostgreSQL"],
        experience_level="Senior (5+ years)",
        responsibilities=[
            "Design and implement scalable web applications",
            "Lead technical architecture decisions",
            "Mentor junior developers",
            "Collaborate with cross-functional teams",
        ],
        industry_terms=["Agile", "CI/CD", "Microservices", "REST APIs"],
        company_values=["Innovation", "Collaboration", "Quality", "Customer Focus"],
    )


def create_sample_structured_cv() -> StructuredCV:
    """Create sample structured CV data for testing."""
    cv = StructuredCV(
        personal_info={
            "name": "John Doe",
            "email": "john.doe@email.com",
            "phone": "+1-555-0123",
            "location": "San Francisco, CA",
            "linkedin": "linkedin.com/in/johndoe",
            "github": "github.com/johndoe",
        }
    )

    # Executive Summary Section
    exec_section = Section(
        name="Executive Summary", content_type="executive_summary", order=1
    )
    exec_section.items.append(
        Item(
            content="Experienced software engineer with 6+ years developing scalable web applications using Python and React. Proven track record of leading technical initiatives and mentoring development teams.",
            status=ItemStatus.GENERATED,
            item_type=ItemType.EXECUTIVE_SUMMARY_PARA,
        )
    )
    cv.sections.append(exec_section)

    # Key Qualifications Section
    qual_section = Section(
        name="Key Qualifications", content_type="qualifications", order=2
    )
    qualifications = [
        "Expert in Python development with Django and Flask frameworks",
        "Proficient in React.js and modern JavaScript (ES6+)",
        "Extensive experience with AWS cloud services (EC2, S3, RDS, Lambda)",
        "Strong background in containerization with Docker and Kubernetes",
        "Database design and optimization with PostgreSQL and MongoDB",
    ]
    for qual in qualifications:
        qual_section.items.append(
            Item(
                content=qual,
                status=ItemStatus.GENERATED,
                item_type=ItemType.KEY_QUALIFICATION,
            )
        )
    cv.sections.append(qual_section)

    # Professional Experience Section
    exp_section = Section(
        name="Professional Experience", content_type="experience", order=3
    )

    # Add a subsection for a job
    from src.models.data_models import Subsection

    job_subsection = Subsection(
        name="Senior Software Engineer | TechStart Inc. | 2020 - Present"
    )

    job_achievements = [
        "Led development of microservices architecture serving 1M+ daily users",
        "Implemented CI/CD pipeline reducing deployment time by 60%",
        "Mentored team of 4 junior developers, improving code quality metrics by 40%",
        "Designed and built real-time analytics dashboard using React and WebSocket APIs",
    ]

    for achievement in job_achievements:
        job_subsection.items.append(
            Item(
                content=achievement,
                status=ItemStatus.GENERATED,
                item_type=ItemType.BULLET_POINT,
            )
        )

    exp_section.subsections.append(job_subsection)
    cv.sections.append(exp_section)

    # Project Experience Section
    project_section = Section(
        name="Project Experience", content_type="projects", order=4
    )

    project_subsection = Subsection(name="E-commerce Platform | Personal Project")

    project_details = [
        "Built full-stack e-commerce platform using Django REST API and React frontend",
        "Integrated Stripe payment processing and AWS S3 for media storage",
        "Implemented Redis caching and Elasticsearch for product search functionality",
    ]

    for detail in project_details:
        project_subsection.items.append(
            Item(
                content=detail,
                status=ItemStatus.GENERATED,
                item_type=ItemType.BULLET_POINT,
            )
        )

    project_section.subsections.append(project_subsection)
    cv.sections.append(project_section)

    return cv


async def test_pdf_generation():
    """Test the PDF generation pipeline with sample data."""
    logger.info("Starting PDF generation pipeline test")

    try:
        # Initialize the formatter agent
        formatter = FormatterAgent()
        logger.info("FormatterAgent initialized successfully")

        # Create sample data
        job_data = create_sample_job_data()
        structured_cv = create_sample_structured_cv()

        logger.info("Sample data created")
        logger.info(f"Job data: {job_data.skills}")
        logger.info(
            f"CV sections: {[section.name for section in structured_cv.sections]}"
        )

        # Prepare input for formatter
        formatter_input = {
            "structured_cv": structured_cv,
            "job_description_data": job_data,
            "output_format": "pdf",
        }

        # Run the formatter
        logger.info("Running FormatterAgent...")
        result = await formatter.run(formatter_input)

        if "pdf_path" in result:
            pdf_path = result["pdf_path"]
            logger.info(f"PDF generated successfully: {pdf_path}")

            # Check if file exists and get size
            if os.path.exists(pdf_path):
                file_size = os.path.getsize(pdf_path)
                logger.info(f"PDF file size: {file_size} bytes")

                # Create a copy with timestamp for debugging
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                debug_path = pdf_path.replace(".pdf", f"_debug_{timestamp}.pdf")

                import shutil

                shutil.copy2(pdf_path, debug_path)
                logger.info(f"Debug copy created: {debug_path}")

                return {
                    "success": True,
                    "pdf_path": pdf_path,
                    "debug_path": debug_path,
                    "file_size": file_size,
                }
            else:
                logger.error(f"PDF file not found at expected path: {pdf_path}")
                return {"success": False, "error": "PDF file not created"}
        else:
            logger.error("FormatterAgent did not return a PDF path")
            logger.error(f"Formatter result: {result}")
            return {
                "success": False,
                "error": "No PDF path in result",
                "result": result,
            }

    except Exception as e:
        logger.error(f"Error in PDF generation test: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


async def test_pdf_with_custom_data(
    cv_data: Dict[str, Any] = None, job_data: Dict[str, Any] = None
):
    """Test PDF generation with custom data.

    Args:
        cv_data: Custom CV data dictionary
        job_data: Custom job description data dictionary
    """
    logger.info("Starting PDF generation test with custom data")

    try:
        formatter = FormatterAgent()

        # Use provided data or fallback to samples
        if cv_data:
            # Convert dict to StructuredCV if needed
            if isinstance(cv_data, dict):
                structured_cv = StructuredCV.model_validate(cv_data)
            else:
                structured_cv = cv_data
        else:
            structured_cv = create_sample_structured_cv()

        if job_data:
            # Convert dict to JobDescriptionData if needed
            if isinstance(job_data, dict):
                job_desc_data = JobDescriptionData.model_validate(job_data)
            else:
                job_desc_data = job_data
        else:
            job_desc_data = create_sample_job_data()

        formatter_input = {
            "structured_cv": structured_cv,
            "job_description_data": job_desc_data,
            "output_format": "pdf",
        }

        result = await formatter.run(formatter_input)
        return result

    except Exception as e:
        logger.error(f"Error in custom PDF generation test: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


def run_pdf_test():
    """Convenience function to run the PDF test."""
    return asyncio.run(test_pdf_generation())


def run_custom_pdf_test(cv_data=None, job_data=None):
    """Convenience function to run the PDF test with custom data."""
    return asyncio.run(test_pdf_with_custom_data(cv_data, job_data))


if __name__ == "__main__":
    print("PDF Pipeline Test Script")
    print("=" * 50)

    # Run the basic test
    result = run_pdf_test()

    print("\nTest Results:")
    print(json.dumps(result, indent=2, default=str))

    if result.get("success"):
        print(f"\n✅ PDF generation successful!")
        print(f"📄 PDF location: {result.get('pdf_path')}")
        print(f"🐛 Debug copy: {result.get('debug_path')}")
        print(f"📊 File size: {result.get('file_size')} bytes")
    else:
        print(f"\n❌ PDF generation failed: {result.get('error')}")
        if "result" in result:
            print(f"Formatter output: {result['result']}")
