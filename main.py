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
import streamlit as st # Import Streamlit for UI

def main():
    st.title("AI CV Generator")

    # Initialize components
    model = LLM()
    parser_agent = ParserAgent(name="ParserAgent", description="Agent for parsing job descriptions.", llm=model)
    template_renderer = TemplateRenderer(name="TemplateRenderer", description="Agent for rendering CV templates.", model=model, input_schema=AgentIO(input={}, output={}, description="template renderer"), output_schema=AgentIO(input={}, output={}, description="template renderer"))
    vector_db_config = VectorStoreConfig(dimension=768, index_type="IndexFlatL2")
    vector_db = VectorDB(config=vector_db_config)
    vector_store_agent = VectorStoreAgent(name="Vector Store Agent", description="Agent for managing vector store.", model=model, input_schema=AgentIO(input={}, output={}, description="vector store agent"), output_schema=AgentIO(input={}, output={}, description="vector store agent"), vector_db=vector_db)
    content_writer_agent = ContentWriterAgent(name="ContentWriterAgent", description="Agent for generating tailored CV content.", llm=model)
    research_agent = ResearchAgent(name="ResearchAgent", description="Agent for researching job-related information.", llm=model)
    orchestrator = Orchestrator(parser_agent, template_renderer, vector_store_agent, content_writer_agent, research_agent)

    # Input fields for job description and CV
    job_description = st.text_area("Job Description", "Enter the job description here...")
    user_cv = st.text_area("User CV", "Paste the user's CV here...")

    if st.button("Generate CV"):
        if not job_description or not user_cv:
            st.error("Please provide both a job description and a CV.")
        else:
            user_cv_data = CVData(raw_text=user_cv, experiences=[], summary="", skills=[], education=[], projects=[])
            rendered_cv = orchestrator.run_workflow(job_description, user_cv_data)

            # Display the rendered CV
            st.subheader("Generated CV")
            st.markdown(rendered_cv)

            # Feedback section
            st.subheader("Feedback")
            feedback = st.text_area("Provide your feedback here...")
            if st.button("Submit Feedback"):
                st.success("Thank you for your feedback!")

if __name__ == "__main__":
    main()
