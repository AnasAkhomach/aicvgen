import logging
import os
from src.agents.parser_agent import ParserAgent
from src.services.llm import LLM
from src.core.state_manager import JobDescriptionData

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("parser_test.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def load_cv_template():
    """Load the CV template from file"""
    try:
        with open("src/templates/cv_template.md", "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error loading CV template: {e}")
        return None


def main():
    """Test the parser agent section detection"""
    logger.info("Starting parser agent test")

    # Initialize LLM
    llm = LLM()

    # Initialize ParserAgent
    parser_agent = ParserAgent(
        name="ParserAgent", description="Testing section detection", llm=llm
    )

    # Load CV template
    cv_text = load_cv_template()
    if not cv_text:
        logger.error("Failed to load CV template")
        return

    # Create a simple job description
    job_data = JobDescriptionData(
        raw_text="Test job description",
        skills=["Python", "SQL"],
        experience_level="Mid-level",
        responsibilities=["Testing"],
        industry_terms=["AI"],
        company_values=["Quality"],
    )

    # Parse CV
    logger.info("Parsing CV text")
    structured_cv = parser_agent.parse_cv_text(cv_text, job_data)

    # Print sections and their content types
    print("\n=== SECTION DETECTION RESULTS ===")
    print(f"Found {len(structured_cv.sections)} sections:")

    for section in structured_cv.sections:
        section_type = "DYNAMIC" if section.content_type == "DYNAMIC" else "STATIC"
        print(f"Section: {section.name} - Type: {section_type}")

        # List items in section
        item_count = len(section.items)
        subsection_count = len(section.subsections)
        print(f"  - {item_count} direct items, {subsection_count} subsections")

        # Print subsection info
        for subsection in section.subsections:
            subsection_items = len(subsection.items)
            print(f"    * Subsection: {subsection.name} - {subsection_items} items")

    print("\n=== TEST COMPLETED ===")


if __name__ == "__main__":
    main()
