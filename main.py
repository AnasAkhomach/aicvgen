# main.py
from orchestrator import Orchestrator
from parser_agent import ParserAgent
from template_renderer import TemplateRenderer
from vector_store_agent import VectorStoreAgent
from vector_db import VectorDB
from llm import LLM
from state_manager import VectorStoreConfig, CVData, AgentIO # Import AgentIO
import os # Import os for file operations
from content_writer_agent import ContentWriterAgent # Import ContentWriterAgent
from research_agent import ResearchAgent # Import ResearchAgent

if __name__ == "__main__":
    print("Starting workflow")

    # Initialize components
    model = LLM()
    # Initialize specialized agents
    parser_agent = ParserAgent(name="ParserAgent", description="Agent for parsing job descriptions.", llm=model)
    template_renderer = TemplateRenderer(name="TemplateRenderer", description="Agent for rendering CV templates.", model=model, input_schema=AgentIO(input={}, output={}, description="template renderer"), output_schema=AgentIO(input={}, output={}, description="template renderer"))
    vector_db_config = VectorStoreConfig(dimension=768, index_type="IndexFlatL2")
    vector_db = VectorDB(config=vector_db_config)
    vector_store_agent = VectorStoreAgent(name="Vector Store Agent", description="Agent for managing vector store.", model=model, input_schema=AgentIO(input={}, output={}, description="vector store agent"), output_schema=AgentIO(input={}, output={}, description="vector store agent"), vector_db=vector_db)
    content_writer_agent = ContentWriterAgent(name="ContentWriterAgent", description="Agent for generating tailored CV content.", llm=model)
    research_agent = ResearchAgent(name="ResearchAgent", description="Agent for researching job-related information.", llm=model) # Initialize ResearchAgent

    # Initialize Orchestrator with all required agents
    # The Orchestrator.__init__ method will need to be updated to accept research_agent
    orchestrator = Orchestrator(parser_agent, template_renderer, vector_store_agent, content_writer_agent, research_agent)

    # Prepare input data
    job_description = """
    We are looking for a Data Scientist with experience in Python and Machine Learning.
    You will be responsible for developing and implementing machine learning models.
    """

    # Prepare user_cv_data as a CVData dictionary
    user_cv_data: CVData = {
        "raw_text": """
        My CV

        Summary: Experienced professional looking for new opportunities.

        Experience:
        - Developed machine learning models using Python.
        - Implemented data analysis pipelines.
        - Worked on various projects involving data science.

        Skills: Python, Machine Learning, Data Analysis, SQL.
        """,
        "experiences": [
            "Developed machine learning models using Python at Company A.",
            "Implemented data analysis pipelines for Project X.",
            "Worked on various data science projects in the finance domain."
        ]
    }

    # Run the workflow
    rendered_cv = orchestrator.run_workflow(job_description, user_cv_data)

    # Define the output file path
    output_file_path = "tailored_cv.md"

    # Write the rendered CV to a file
    try:
        with open(output_file_path, "w") as f:
            f.write(rendered_cv)
        print(f"--- Rendered CV saved to {output_file_path} ---")
    except IOError as e:
        print(f"Error writing rendered CV to file {output_file_path}: {e}")


    print("Ending workflow")
