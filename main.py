# main.py
from orchestrator import Orchestrator
from parser_agent import ParserAgent
from template_renderer import TemplateRenderer
from vector_store_agent import VectorStoreAgent
from vector_db import VectorDB
from llm import LLM
from state_manager import VectorStoreConfig, CVData, AgentIO
from content_writer_agent import ContentWriterAgent
from research_agent import ResearchAgent
from tools_agent import ToolsAgent
from cv_analyzer_agent import CVAnalyzerAgent
from formatter_agent import FormatterAgent
from quality_assurance_agent import QualityAssuranceAgent
import streamlit as st

def main():
    st.title("AI CV Generator - MVP")
    st.write("Enter a job description and your CV to generate a tailored CV")

    # Initialize components
    with st.spinner("Initializing AI components..."):
        try:
            model = LLM()
            parser_agent = ParserAgent(name="ParserAgent", description="Agent for parsing job descriptions.", llm=model)
            template_renderer = TemplateRenderer(name="TemplateRenderer", description="Agent for rendering CV templates.", model=model, input_schema=AgentIO(input={}, output={}, description="template renderer"), output_schema=AgentIO(input={}, output={}, description="template renderer"))
            vector_db_config = VectorStoreConfig(dimension=768, index_type="IndexFlatL2")
            vector_db = VectorDB(config=vector_db_config)
            vector_store_agent = VectorStoreAgent(name="Vector Store Agent", description="Agent for managing vector store.", model=model, input_schema=AgentIO(input={}, output={}, description="vector store agent"), output_schema=AgentIO(input={}, output={}, description="vector store agent"), vector_db=vector_db)
            tools_agent = ToolsAgent(name="ToolsAgent", description="Agent for content processing.")
            content_writer_agent = ContentWriterAgent(name="ContentWriterAgent", description="Agent for generating tailored CV content.", llm=model, tools_agent=tools_agent)
            research_agent = ResearchAgent(name="ResearchAgent", description="Agent for researching job-related information.", llm=model)
            cv_analyzer_agent = CVAnalyzerAgent(name="CVAnalyzerAgent", description="Agent for analyzing CVs.", llm=model)
            formatter_agent = FormatterAgent(name="FormatterAgent", description="Agent for formatting CV content.")
            quality_assurance_agent = QualityAssuranceAgent(name="QualityAssuranceAgent", description="Agent for quality assurance checks.")
            orchestrator = Orchestrator(parser_agent, template_renderer, vector_store_agent, content_writer_agent, research_agent, cv_analyzer_agent, tools_agent, formatter_agent, quality_assurance_agent, model)
        except Exception as e:
            st.error(f"Error initializing components: {str(e)}")
            return

    # Input fields for job description and CV
    job_description = st.text_area("Job Description", "Software Engineer position at Google", height=250)
    user_cv = st.text_area("Your CV", "John Doe\nSoftware Engineer with 5+ years of experience.\nExperience:\n- Worked on several projects.\nSkills:\n- Python\n- Java", height=250)

    # State management for feedback
    if 'awaiting_feedback' not in st.session_state:
        st.session_state.awaiting_feedback = False
    
    if 'workflow_result' not in st.session_state:
        st.session_state.workflow_result = None
        
    if 'job_description' not in st.session_state:
        st.session_state.job_description = None
        
    if 'user_cv_data' not in st.session_state:
        st.session_state.user_cv_data = None
    
    # New state management variables for handling regeneration and workflow stages
    if 'workflow_stage' not in st.session_state:
        st.session_state.workflow_stage = None
        
    if 'regenerate_section' not in st.session_state:
        st.session_state.regenerate_section = None
        
    if 'regenerate_full' not in st.session_state:
        st.session_state.regenerate_full = False
        
    if 'should_run_workflow' not in st.session_state:
        st.session_state.should_run_workflow = False
    
    # Process section regeneration if needed
    if st.session_state.should_run_workflow:
        with st.spinner("Processing your feedback..."):
            try:
                job_description = st.session_state.job_description
                user_cv_data = st.session_state.user_cv_data
                
                # Determine what type of regeneration is needed
                if st.session_state.regenerate_section:
                    section = st.session_state.regenerate_section
                    regenerate_only = []
                    section_feedback = []
                    
                    if section == "summary":
                        section_feedback = ["summary"]
                        regenerate_only = ["summary"]
                        feedback_message = "Please improve the professional summary."
                    elif section == "experience":
                        section_feedback = ["experience section"]
                        regenerate_only = ["experience_bullets"]
                        feedback_message = "Please improve the experience section."
                    elif section.startswith("experience_item_"):
                        # Extract the index of the experience item to regenerate
                        item_index = int(section.split("_")[-1])
                        section_feedback = [f"experience item {item_index}"]
                        regenerate_only = ["experience_bullets"]
                        feedback_message = f"Please improve experience item {item_index}."
                        # Add the specific item index
                        user_feedback = {
                            "approved": False,
                            "comments": feedback_message,
                            "sections_feedback": section_feedback,
                            "regenerate_only": regenerate_only,
                            "experience_item_index": item_index
                        }
                    elif section == "add_experience":
                        section_feedback = ["add new experience"]
                        regenerate_only = ["experience_bullets"]
                        feedback_message = "Please add a new relevant experience entry based on the job description."
                        user_feedback = {
                            "approved": False,
                            "comments": feedback_message,
                            "sections_feedback": section_feedback,
                            "regenerate_only": regenerate_only,
                            "add_experience": True
                        }
                    elif section == "skills":
                        section_feedback = ["skills section"]
                        regenerate_only = ["skills_section"]
                        feedback_message = "Please improve the skills section."
                    elif section == "projects":
                        section_feedback = ["project section"]
                        regenerate_only = ["projects"]
                        feedback_message = "Please improve the projects section."
                    elif section == "add_project":
                        section_feedback = ["add new project"]
                        regenerate_only = ["projects"]
                        feedback_message = "Please add a new relevant project entry based on the job description."
                        user_feedback = {
                            "approved": False,
                            "comments": feedback_message,
                            "sections_feedback": section_feedback,
                            "regenerate_only": regenerate_only,
                            "add_project": True
                        }
                    elif section.startswith("project_item_"):
                        # Extract the index of the project to regenerate
                        item_index = int(section.split("_")[-1])
                        section_feedback = [f"project item {item_index}"]
                        regenerate_only = ["projects"]
                        feedback_message = f"Please improve project item {item_index}."
                        # Add the specific item index
                        user_feedback = {
                            "approved": False,
                            "comments": feedback_message,
                            "sections_feedback": section_feedback,
                            "regenerate_only": regenerate_only,
                            "project_item_index": item_index
                        }
                    elif section == "education":
                        section_feedback = ["education section"]
                        regenerate_only = ["education"]
                        feedback_message = "Please improve the education section."
                    elif section == "certifications":
                        section_feedback = ["certifications section"]
                        regenerate_only = ["certifications"]
                        feedback_message = "Please improve the certifications section."
                    elif section == "languages":
                        section_feedback = ["languages section"]
                        regenerate_only = ["languages"]
                        feedback_message = "Please improve the languages section."
                    
                    # Only set user_feedback if it wasn't already set for specific items
                    if not 'user_feedback' in locals():
                        user_feedback = {
                            "approved": False,
                            "comments": feedback_message,
                            "sections_feedback": section_feedback,
                            "regenerate_only": regenerate_only
                        }
                elif st.session_state.regenerate_full:
                    user_feedback = {
                        "approved": False,
                        "comments": "Please make improvements to the CV.",
                        "sections_feedback": ["content", "skills section", "experience section", "formatting"],
                        "regenerate_all": True
                    }
                else:
                    user_feedback = None
                
                # Run the workflow with the appropriate feedback
                result = orchestrator.run_workflow(
                    job_description, 
                    user_cv_data, 
                    user_feedback=user_feedback
                )
                
                # Reset flags
                st.session_state.should_run_workflow = False
                st.session_state.regenerate_section = None
                st.session_state.regenerate_full = False
                
                # Update state based on result
                st.session_state.workflow_result = result
                
                if isinstance(result, dict) and result.get("stage") == "awaiting_feedback":
                    st.session_state.awaiting_feedback = True
                    st.session_state.workflow_stage = "feedback"
                elif isinstance(result, dict) and result.get("stage") == "complete":
                    st.session_state.awaiting_feedback = False
                    st.session_state.workflow_stage = "complete"
                else:
                    st.error("Unexpected error while processing your feedback.")
                    
            except Exception as e:
                import traceback
                st.error(f"Error regenerating content: {str(e)}")
                st.write("Detailed error:")
                st.code(traceback.format_exc())
                
                # Reset flags on error
                st.session_state.should_run_workflow = False
                st.session_state.regenerate_section = None
                st.session_state.regenerate_full = False
    
    # Generate CV button
    if st.button("Generate Tailored CV") and not st.session_state.awaiting_feedback:
        if not job_description or not user_cv:
            st.error("Please provide both a job description and a CV.")
            return
        
        with st.spinner("Analyzing job description and generating CV..."):
            try:
                # Create a CVData object from the raw text
                user_cv_data = CVData(raw_text=user_cv, experiences=[], summary="", skills=[], education=[], projects=[])
                
                # Store the job description and CV data in session state
                st.session_state.job_description = job_description
                st.session_state.user_cv_data = user_cv_data
                
                st.write("Running workflow...")
                
                # Run the workflow
                result = orchestrator.run_workflow(job_description, user_cv_data)
                
                # Store result for later use
                st.session_state.workflow_result = result
                
                st.write(f"Workflow completed with stage: {result.get('stage') if isinstance(result, dict) else 'unknown'}")
                
                # Check if we need user feedback
                if isinstance(result, dict) and result.get("stage") == "awaiting_feedback":
                    st.session_state.awaiting_feedback = True
                    st.session_state.workflow_stage = "feedback"
                elif isinstance(result, dict) and result.get("stage") == "complete":
                    # Display the rendered CV
                    st.subheader("Generated CV")
                    st.markdown(result.get("rendered_cv", "No CV was generated."))
                    
                    # Download button for the CV
                    st.download_button(
                        label="Download CV as Markdown",
                        data=result.get("rendered_cv", ""),
                        file_name="tailored_cv.md",
                        mime="text/markdown"
                    )
                elif isinstance(result, dict) and result.get("status") == "error":
                    st.error(f"Error: {result.get('error', 'Unknown error occurred.')}")
                    st.write("Full error details:")
                    st.json(result)
                else:
                    # Legacy behavior (direct string return)
                    st.subheader("Generated CV")
                    st.markdown(str(result))
            except Exception as e:
                import traceback
                st.error(f"Error generating CV: {str(e)}")
                st.write("Detailed error:")
                st.code(traceback.format_exc())

    # Display formatted CV with feedback mechanisms if workflow is awaiting feedback
    if st.session_state.awaiting_feedback and st.session_state.workflow_stage == "feedback":
        st.header("Generated CV - For Review")
        
        result = st.session_state.workflow_result
        formatted_cv = result.get("formatted_cv", {})
        
        # Check if formatted_cv is None or empty
        if formatted_cv is None:
            st.write("No formatted CV available. Please complete all approvals and try again.")
            return
        
        # Check if formatted_cv is a string or dictionary
        if isinstance(formatted_cv, str):
            # If it's a string, display it directly
            st.markdown(formatted_cv)
            
            # Add regeneration button
            if st.button("Regenerate Full CV"):
                st.session_state.regenerate_full = True
                st.session_state.should_run_workflow = True
                st.rerun()
                
        else:
            # If it's a dictionary, proceed with structured display
            # Professional Profile / Summary section
            summary_col, summary_buttons = st.columns([0.8, 0.2])
            
            with summary_col:
                st.subheader("Professional Profile")
                st.write(formatted_cv.get("summary", "No summary available"))
            
            with summary_buttons:
                st.write("")
                st.write("")
                if st.button("Regenerate Summary"):
                    st.session_state.regenerate_section = "summary"
                    st.session_state.should_run_workflow = True
                    st.rerun()
            
            # Skills section
            skills_col, skills_buttons = st.columns([0.8, 0.2])
            
            with skills_col:
                st.subheader("Key Qualifications")
                skills_data = formatted_cv.get("skills_section", {})
                
                # Handle both direct string skills and structured skills data
                if isinstance(skills_data, dict) and "skills" in skills_data:
                    skills = skills_data.get("skills", [])
                    if skills:
                        st.write(" | ".join(skills))
                    else:
                        st.write("No skills available")
                else:
                    st.write(skills_data or "No skills available")
                    
            with skills_buttons:
                st.write("")
                st.write("")
                if st.button("Regenerate Skills"):
                    st.session_state.regenerate_section = "skills"
                    st.session_state.should_run_workflow = True
                    st.rerun()
            
            # Projects section
            st.subheader("Projects")
            
            projects = formatted_cv.get("projects", [])
            if not projects:
                st.write("No projects available")
                
                # Add general regeneration button if no projects 
                if st.button("Generate Projects"):
                    st.session_state.regenerate_section = "projects"
                    st.session_state.should_run_workflow = True
                    st.rerun()
            else:
                # For each project, add a regenerate button next to it
                for idx, project in enumerate(projects):
                    project_col, btn_col = st.columns([0.85, 0.15])
                    
                    with project_col:
                        if isinstance(project, dict):
                            # Handle structured project data
                            name = project.get("name", "")
                            description = project.get("description", "")
                            technologies = project.get("technologies", [])
                            
                            st.markdown(f"**{name}**")
                            st.write(description)
                            if technologies:
                                st.write(f"*Technologies: {', '.join(technologies)}*")
                        else:
                            # Simple string format
                            st.write(project)
                    
                    with btn_col:
                        # Button to regenerate this specific project
                        if st.button(f"Regenerate", key=f"proj_item_{idx}"):
                            st.session_state.regenerate_section = f"project_item_{idx}"
                            st.session_state.should_run_workflow = True
                            st.rerun()
                    
                    st.write("---")  # Separator between items
                
                # Add button to generate a new project
                if st.button("Add New Project"):
                    st.session_state.regenerate_section = "add_project"
                    st.session_state.should_run_workflow = True
                    st.rerun()
            
            # Experience section
            st.subheader("Professional Experience")
            
            experience_items = formatted_cv.get("experience_bullets", [])
            if not experience_items:
                st.write("No experience items available")
                
                # Add general regeneration button if no experience
                if st.button("Generate Experience"):
                    st.session_state.regenerate_section = "experience"
                    st.session_state.should_run_workflow = True
                    st.rerun()
            else:
                # For each experience item, add a regenerate button next to it
                for idx, item in enumerate(experience_items):
                    item_col, btn_col = st.columns([0.85, 0.15])
                    
                    with item_col:
                        if isinstance(item, dict):
                            # Handle structured experience data
                            company = item.get("company", "")
                            position = item.get("position", "")
                            period = item.get("period", "")
                            location = item.get("location", "")
                            bullets = item.get("bullets", [])
                            
                            header_text = f"**{position}**"
                            
                            if company:
                                header_text += f" at *{company}*"
                            
                            details = []
                            if period:
                                details.append(period)
                            if location:
                                details.append(location)
                                
                            if details:
                                header_text += f" ({', '.join(details)})"
                                
                            st.markdown(header_text)
                            
                            for bullet in bullets:
                                st.markdown(f"* {bullet}")
                        else:
                            # Simple string format
                            st.write(item)
                    
                    with btn_col:
                        # Button to regenerate this specific experience item
                        if st.button(f"Regenerate", key=f"exp_item_{idx}"):
                            st.session_state.regenerate_section = f"experience_item_{idx}"
                            st.session_state.should_run_workflow = True
                            st.rerun()
                    
                    st.write("---")  # Separator between items
                
                # Add button to generate a new experience item
                if st.button("Add New Experience"):
                    st.session_state.regenerate_section = "add_experience"
                    st.session_state.should_run_workflow = True
                    st.rerun()
            
            # Education section
            education_items = formatted_cv.get("education", [])
            if education_items:
                st.subheader("Education")
                
                for edu in education_items:
                    if isinstance(edu, dict):
                        degree = edu.get("degree", "")
                        institution = edu.get("institution", "")
                        location = edu.get("location", "")
                        period = edu.get("period", "")
                        details = edu.get("details", [])
                        
                        header_text = f"**{degree}**"
                        
                        if institution:
                            header_text += f" - *{institution}*"
                        
                        info = []
                        if location:
                            info.append(location)
                        if period:
                            info.append(period)
                            
                        if info:
                            header_text += f" ({', '.join(info)})"
                            
                        st.markdown(header_text)
                        
                        for detail in details:
                            st.markdown(f"* {detail}")
                            
                        st.write("---")
                    else:
                        st.write(edu)
                        st.write("---")
            
            # Certifications section
            certifications = formatted_cv.get("certifications", [])
            if certifications:
                st.subheader("Certifications")
                
                for cert in certifications:
                    if isinstance(cert, dict):
                        name = cert.get("name", "")
                        issuer = cert.get("issuer", "")
                        date = cert.get("date", "")
                        url = cert.get("url", "")
                        
                        if url:
                            cert_text = f"* [{name}]({url})"
                        else:
                            cert_text = f"* {name}"
                        
                        if issuer or date:
                            cert_text += " ("
                            if issuer:
                                cert_text += issuer
                            if issuer and date:
                                cert_text += ", "
                            if date:
                                cert_text += date
                            cert_text += ")"
                        
                        st.markdown(cert_text)
                    else:
                        st.markdown(f"* {cert}")
                
                st.write("---")
            
            # Languages section
            languages = formatted_cv.get("languages", [])
            if languages:
                st.subheader("Languages")
                
                if isinstance(languages, list):
                    for lang in languages:
                        if isinstance(lang, dict):
                            name = lang.get("name", "")
                            level = lang.get("level", "")
                            
                            if name and level:
                                st.markdown(f"**{name}** ({level})")
                            elif name:
                                st.markdown(f"**{name}**")
                        else:
                            st.write(lang)
                else:
                    st.write(languages)
            
            # Overall feedback
            st.write("---")
            with st.form("feedback_form"):
                feedback = st.text_area("Provide overall feedback (optional)", height=100)
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.form_submit_button("Approve CV"):
                        # Handle approval
                        with st.spinner("Finalizing your CV..."):
                            try:
                                result = orchestrator.run_workflow(
                                    st.session_state.job_description,
                                    st.session_state.user_cv_data,
                                    user_feedback={
                                        "approved": True,
                                        "comments": feedback
                                    }
                                )
                                st.session_state.workflow_result = result
                                st.session_state.awaiting_feedback = False
                                st.session_state.workflow_stage = "complete"
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error finalizing CV: {str(e)}")
                
                with col2:
                    if st.form_submit_button("Regenerate Full CV"):
                        st.session_state.regenerate_full = True
                        st.session_state.should_run_workflow = True
                        st.rerun()
    
    # Display final CV if workflow is complete
    elif st.session_state.workflow_stage == "complete":
        st.header("Your Finalized CV")
        
        if st.session_state.workflow_result:
            rendered_cv = st.session_state.workflow_result.get("rendered_cv", "")
            if rendered_cv:
                st.write(rendered_cv)
                
                # Option to download the CV
                st.download_button(
                    label="Download CV as Text",
                    data=rendered_cv,
                    file_name="tailored_cv.txt",
                    mime="text/plain"
                )
            else:
                st.error("No rendered CV available. Please try again.")
        else:
            st.error("No workflow result available. Please try again.")

    # Simple feedback section
    with st.expander("Provide Feedback"):
        st.text_area("Your feedback", "")
        if st.button("Send Feedback"):
            st.success("Thank you for your feedback!")

if __name__ == "__main__":
    main()
