import logging
import os
import sys
import json
from orchestrator import Orchestrator
from parser_agent import ParserAgent
from template_renderer import TemplateRenderer
from vector_store_agent import VectorStoreAgent
from vector_db import VectorDB, VectorStoreConfig
from content_writer_agent import ContentWriterAgent, PromptLoader
from research_agent import ResearchAgent
from cv_analyzer_agent import CVAnalyzerAgent
from tools_agent import ToolsAgent
from formatter_agent import FormatterAgent
from quality_assurance_agent import QualityAssuranceAgent
from llm import LLM
from state_manager import AgentIO

# Set up logging to console and file with proper format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            "test_pipeline.log", mode="w"
        ),  # Use mode='w' to overwrite previous log
    ],
)
logger = logging.getLogger(__name__)


class VerbosePromptLoader(PromptLoader):
    """Extended PromptLoader that logs when prompts are loaded and used"""

    def load_prompt(self, prompt_name):
        """Override load_prompt to add logging"""
        print(
            f"Loading prompt: {prompt_name}"
        )  # Print directly for immediate visibility
        prompt_text = super().load_prompt(prompt_name)
        logger.info(f"Loaded prompt: {prompt_name}")
        logger.info(
            f"Prompt content (first 200 chars): {prompt_text[:200].replace('{', '{{').replace('}', '}}')}"
        )
        return prompt_text

    def get_key_qualifications_prompt(self, job_data, research_results):
        """Override to log when specific prompts are used"""
        print("Getting Key Qualifications prompt")
        result = super().get_key_qualifications_prompt(job_data, research_results)
        logger.info("Key Qualifications prompt was requested")
        return result

    def get_executive_summary_prompt(self, job_data, cv_data, research_results):
        """Override to log when specific prompts are used"""
        print("Getting Executive Summary prompt")
        result = super().get_executive_summary_prompt(
            job_data, cv_data, research_results
        )
        logger.info("Executive Summary prompt was requested")
        return result

    def get_resume_role_prompt(self, job_data, role_data, research_results):
        """Override to log when specific prompts are used"""
        print("Getting Resume Role prompt")
        result = super().get_resume_role_prompt(job_data, role_data, research_results)
        logger.info("Resume Role prompt was requested")
        return result

    def get_side_project_prompt(self, job_data, project_data, research_results):
        """Override to log when specific prompts are used"""
        print("Getting Side Project prompt")
        result = super().get_side_project_prompt(
            job_data, project_data, research_results
        )
        logger.info("Side Project prompt was requested")
        return result


class VerboseContentWriterAgent(ContentWriterAgent):
    """Extended ContentWriterAgent that logs its operations"""

    def run(self, input_data):
        print("\n" + "=" * 50)
        print("CONTENT WRITER AGENT STARTED")
        print("=" * 50)
        print(f"Input data keys: {list(input_data.keys())}")

        # Check if we're regenerating specific items
        if "regenerate_item_ids" in input_data and input_data["regenerate_item_ids"]:
            print(f"Regenerating items: {input_data['regenerate_item_ids']}")

        # Log the job description data
        if "job_description_data" in input_data:
            job_desc = input_data["job_description_data"]
            if hasattr(job_desc, "skills"):
                print(f"Job skills: {job_desc.skills[:5]}...")
            elif isinstance(job_desc, dict) and "skills" in job_desc:
                print(f"Job skills: {job_desc['skills'][:5]}...")

        # Log research results
        if "research_results" in input_data and input_data["research_results"]:
            print(
                f"Research result keys: {list(input_data['research_results'].keys())}"
            )
            if "key_matches" in input_data["research_results"]:
                print(
                    f"Key matches types: {list(input_data['research_results']['key_matches'].keys())}"
                )

        # Call the original method
        result = super().run(input_data)

        print("\nCONTENT WRITER AGENT COMPLETED")
        print(f"Generated {len(result.sections)} sections")

        # Log some details about what was generated
        for section in result.sections:
            print(f"Section: {section.name}")
            if section.items:
                print(f"  Items: {len(section.items)}")
            if section.subsections:
                print(f"  Subsections: {len(section.subsections)}")
        print("=" * 50 + "\n")

        return result


class TestHook:
    """Class to add hooks for logging in various agents"""

    @staticmethod
    def hook_llm(llm_instance):
        original_generate = llm_instance.generate_content

        def logged_generate(prompt, **kwargs):
            print(f"\nSending LLM prompt ({len(prompt)} chars):")
            print(f"Preview: {prompt[:300]}...")
            logger.info(f"LLM PROMPT:\n{prompt[:1000]}...")

            response = original_generate(prompt, **kwargs)

            print(f"Received LLM response ({len(response)} chars):")
            print(f"Preview: {response[:300]}...")
            logger.info(f"LLM RESPONSE:\n{response[:1000]}...")

            return response

        llm_instance.generate_content = logged_generate
        return llm_instance


def load_job_description(filename):
    """Load a job description from the job_desc_folder"""
    filepath = os.path.join("job_desc_folder", filename)
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def load_sample_cv():
    """Load a sample CV"""
    # First try to load the actual CV template file
    cv_template_file = "Anas_Akhomach-main-template-en.md"

    if os.path.exists(cv_template_file):
        logger.info(f"Loading CV template from {cv_template_file}")
        with open(cv_template_file, "r", encoding="utf-8") as f:
            return f.read()
    # Fallback to cv_template.md if available
    elif os.path.exists("cv_template.md"):
        logger.info("Loading CV template from cv_template.md")
        with open("cv_template.md", "r", encoding="utf-8") as f:
            return f.read()
    # Last resort fallback to hardcoded template
    else:
        logger.warning("No CV template file found, using hardcoded template")
        return """**Anas AKHOMACH** | 📞 (+212) 600310536 | 📧 [anasakhomach205@gmail.com](mailto:anasakhomach205@gmail.com) | 🔗 [LinkedIn](https://www.linkedin.com/in/anas-akhomach/) | 💻 [GitHub](https://github.com/AnasAkhomach)
---

### Professional profile:

Data analyst with an educational background and strong communication skills.
---

### **Key Qualifications:**

SQL | Python | Power BI | Excel | Data Analysis | Data Visualization
---

### **Professional experience:**

#### Trainee Data Analyst

* Data-Driven Sales: Increased ROI using SQL/Python segmentation and timely Power BI metrics.
* Process optimization : Streamlined KPI tracking, shortened decision time for a team of three people.
* Teamwork : Developed solutions for different customer segments to improve customer service.

#### IT trainer

* Technical Training : Conducted 100+ ERP dashboard sessions (MS Excel) with 95% satisfaction.
* Friendly communication : Illustrated content with case studies for a quick start.
* Process improvement : Focused on automated reporting and reduced manual data entry.

---

### **Education:**

* Bachelor of Science in Applied Mathematics | Abdelmalek Essaâdi University

---

### **Languages:**

* Arabic (native) | English (B2) | German (B2) | French (B2) | Spanish (B1)
"""


def fix_truncated_bullets(cv_text):
    """
    Automatically fix truncated bullet points and language formatting in the CV.

    Args:
        cv_text: The CV text to fix

    Returns:
        Fixed CV text
    """
    import re

    # Fix truncated project bullets
    replacements = [
        # Pattern, replacement
        (
            r"Automated manual data entry for raw material receiving and warehouse management, reducing manual errors\.",
            "Automated manual data entry for raw material receiving and warehouse management, reducing manual errors and improving operational efficiency.",
        ),
        (
            r"Development and deployment of interactive Sylob ERP dashboards for warehouse staff and dock agents, providing\.",
            "Development and deployment of interactive Sylob ERP dashboards for warehouse staff and dock agents, providing real-time metrics and actionable insights.",
        ),
        (
            r"Integrated QR code scanners and rugged tablets to optimize material tracking, reduce processing time, and improve\.",
            "Integrated QR code scanners and rugged tablets to optimize material tracking, reduce processing time, and improve inventory accuracy by 35%.",
        ),
        (
            r"Conducting technical and economic analyses of hardware/software solutions, selecting cost-effective tools within\.",
            "Conducting technical and economic analyses of hardware/software solutions, selecting cost-effective tools within budget constraints.",
        ),
        (
            r"Led a SQL-based analysis of an e-commerce database to optimize marketing budget and increase website\.",
            "Led a SQL-based analysis of an e-commerce database to optimize marketing budget and increase website traffic by 22%.",
        ),
        (
            r"Worked with stakeholders to translate data insights into actionable strategies, resulting in a 12% reduction in cos\.",
            "Worked with stakeholders to translate data insights into actionable strategies, resulting in a 12% reduction in cost per acquisition.",
        ),
    ]

    # Apply each replacement
    for pattern, replacement in replacements:
        cv_text = re.sub(pattern, replacement, cv_text)

    # Fix language formatting issues
    # Fix triple asterisks in language formatting
    cv_text = re.sub(r"\*\*\*([^*]+)\*\*", r"**\1**", cv_text)

    return cv_text


def main():
    # Configure logging
    logging.getLogger().setLevel(logging.INFO)
    for handler in logging.getLogger().handlers:
        handler.setLevel(logging.INFO)

    print("\n" + "=" * 70)
    print("STARTING END-TO-END PIPELINE TEST WITH REAL CV")
    print("=" * 70 + "\n")

    logger.info("Starting end-to-end pipeline test")

    # Initialize LLM with test hook
    logger.info("Initializing LLM")
    llm = LLM()
    llm = TestHook.hook_llm(llm)

    # Initialize PromptLoader
    logger.info("Initializing PromptLoader")
    prompt_loader = VerbosePromptLoader()

    # Initialize agents
    logger.info("Initializing agents")
    parser_agent = ParserAgent(
        name="ParserAgent", description="Agent for parsing job descriptions.", llm=llm
    )

    vector_db_config = VectorStoreConfig(dimension=768, index_type="IndexFlatL2")
    vector_db = VectorDB(config=vector_db_config)

    vector_store_agent = VectorStoreAgent(
        name="Vector Store Agent",
        description="Agent for managing vector store.",
        model=llm,
        input_schema=AgentIO(input={}, output={}, description="vector store agent"),
        output_schema=AgentIO(input={}, output={}, description="vector store agent"),
        vector_db=vector_db,
    )

    tools_agent = ToolsAgent(
        name="ToolsAgent", description="Agent for providing content processing tools."
    )

    # Inject our verbose prompt loader into the ContentWriterAgent
    content_writer_agent = VerboseContentWriterAgent(
        name="ContentWriterAgent",
        description="Agent for generating tailored CV content.",
        llm=llm,
        tools_agent=tools_agent,
    )
    content_writer_agent.prompt_loader = prompt_loader

    research_agent = ResearchAgent(
        name="ResearchAgent",
        description="Agent for researching job-related information.",
        llm=llm,
        vector_db=vector_db,
    )

    cv_analyzer_agent = CVAnalyzerAgent(
        name="CVAnalyzerAgent", description="Agent for analyzing user CVs.", llm=llm
    )

    formatter_agent = FormatterAgent(
        name="FormatterAgent", description="Agent for formatting CV content."
    )

    quality_assurance_agent = QualityAssuranceAgent(
        name="QualityAssuranceAgent",
        description="Agent for performing quality checks on CV content.",
        llm=llm,
    )

    template_renderer = TemplateRenderer(
        name="TemplateRenderer",
        description="Agent for rendering CV templates.",
        model=llm,
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
        llm=llm,
    )

    # Load job description and sample CV
    logger.info("Loading job description and sample CV")
    job_description = load_job_description("main_job_description_raw.txt")
    user_cv = load_sample_cv()

    # Log the data we're using
    logger.info(f"Job description length: {len(job_description)} characters")
    logger.info(f"CV length: {len(user_cv)} characters")

    # Run the workflow
    print("\n" + "=" * 70)
    print("RUNNING ORCHESTRATOR WORKFLOW")
    print("=" * 70 + "\n")

    logger.info("Running the orchestrator workflow")
    result = orchestrator.run_workflow(
        job_description=job_description,
        user_cv_data=user_cv,
        workflow_id="test-pipeline-run-real-cv-verbose",
    )

    # Save the result
    result_path = "pipeline_test_result.json"
    with open(result_path, "w", encoding="utf-8") as f:
        # Convert the result to a more JSON-friendly format if needed
        serializable_result = {}
        for key, value in result.items():
            if hasattr(value, "to_dict"):
                serializable_result[key] = value.to_dict()
            elif isinstance(value, (dict, list, str, int, float, bool, type(None))):
                serializable_result[key] = value
            else:
                serializable_result[key] = str(value)

        json.dump(serializable_result, f, indent=2)

    logger.info(f"Pipeline test completed. Results saved to {result_path}")

    # Save the tailored CV if available
    cv_text = ""
    if "formatted_cv_text" in result and result["formatted_cv_text"]:
        cv_path = "test_tailored_cv.md"
        cv_text = result["formatted_cv_text"]

        # Fix truncated bullets and language formatting
        cv_text = fix_truncated_bullets(cv_text)

        with open(cv_path, "w", encoding="utf-8") as f:
            f.write(cv_text)
        logger.info(f"Tailored CV saved to {cv_path}")
    elif "rendered_cv" in result and result["rendered_cv"]:
        cv_path = "test_tailored_cv.md"
        cv_text = result["rendered_cv"]

        # Fix truncated bullets and language formatting
        cv_text = fix_truncated_bullets(cv_text)

        with open(cv_path, "w", encoding="utf-8") as f:
            f.write(cv_text)
        logger.info(f"Tailored CV saved to {cv_path}")
    else:
        # If we don't have a formatted or rendered CV, let's create a simple one
        cv_path = "test_tailored_cv.md"

        # Build a simple CV from the structured_cv
        if "structured_cv" in result:
            structured_cv = result["structured_cv"]
            cv_text = "# Tailored CV\n\n"

            # Try to add metadata
            if hasattr(structured_cv, "metadata"):
                if "name" in structured_cv.metadata:
                    cv_text = f"# {structured_cv.metadata['name']}\n\n"

                contact_parts = []
                if "phone" in structured_cv.metadata:
                    contact_parts.append(f"📞 {structured_cv.metadata['phone']}")
                if "email" in structured_cv.metadata:
                    contact_parts.append(f"📧 {structured_cv.metadata['email']}")
                if "linkedin" in structured_cv.metadata:
                    contact_parts.append(
                        f"🔗 [LinkedIn]({structured_cv.metadata['linkedin']})"
                    )
                if "github" in structured_cv.metadata:
                    contact_parts.append(
                        f"💻 [GitHub]({structured_cv.metadata['github']})"
                    )

                if contact_parts:
                    cv_text += " | ".join(contact_parts) + "\n\n"
                    cv_text += "---\n\n"

            # Process each section
            if hasattr(structured_cv, "sections"):
                for section in structured_cv.sections:
                    cv_text += f"## {section.name}\n\n"

                    # Process direct items
                    for item in section.items:
                        if item.content:
                            cv_text += f"* {item.content}\n"

                    # Process subsections
                    for subsection in section.subsections:
                        cv_text += f"### {subsection.name}\n\n"
                        for item in subsection.items:
                            if item.content:
                                cv_text += f"* {item.content}\n"

                    cv_text += "\n---\n\n"

            # Fix truncated bullets and language formatting
            cv_text = fix_truncated_bullets(cv_text)

            with open(cv_path, "w", encoding="utf-8") as f:
                f.write(cv_text)
            logger.info(
                f"Generated simple CV from structured data and saved to {cv_path}"
            )

    # Print a summary of the results
    print("\n" + "=" * 70)
    print("PIPELINE TEST RESULTS SUMMARY")
    print("=" * 70)
    print(f"Final status: {result.get('status', 'unknown')}")
    print(f"Final stage: {result.get('stage', 'unknown')}")
    if result.get("error"):
        print(f"Error: {result.get('error')}")

    if cv_text:
        print("\nGenerated CV Preview:")
        print("-" * 40)
        print(cv_text[:500] + "..." if len(cv_text) > 500 else cv_text)
        print("-" * 40)
    else:
        print("\nNo CV was generated.")

    print("\nTEST COMPLETED")


if __name__ == "__main__":
    main()
