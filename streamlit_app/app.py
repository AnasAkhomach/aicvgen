# main.py
from orchestrator import Orchestrator
from parser_agent import ParserAgent
from template_renderer import TemplateRenderer
from vector_store_agent import VectorStoreAgent
from vector_db import VectorDB
from llm import LLM
from state_manager import (
    VectorStoreConfig, 
    CVData, 
    AgentIO, 
    StateManager, 
    StructuredCV,
    ItemStatus,
    Item,
    Section,
    Subsection
)
from content_writer_agent import ContentWriterAgent
from research_agent import ResearchAgent
from tools_agent import ToolsAgent
from cv_analyzer_agent import CVAnalyzerAgent
from formatter_agent import FormatterAgent
from quality_assurance_agent import QualityAssuranceAgent
from template_manager import TemplateManager
import streamlit as st
import os
import uuid
import json
import time
import logging
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s - [%(filename)s:%(lineno)d]',
    handlers=[
        logging.FileHandler('debug.log', mode='a'),
        logging.FileHandler('error.log', mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add performance logging
def log_performance(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"Function {func.__name__} took {end_time - start_time:.2f} seconds to execute")
        return result
    return wrapper

# Create a constant for the template path
TEMPLATE_FILE_PATH = "cv_template.md"

# Helper functions for the UI
def display_section(section, state_manager):
    """Display a section with its items or subsections in a card-based UI."""
    if not section:
        return
    
    # Create expander for the section
    with st.expander(f"{section.name}", expanded=True):
        st.markdown(f"#### {section.name}")
        st.write(f"Type: {section.content_type}")
        
        # Display items directly in the section
        if section.items:
            for item in section.items:
                display_item(item, section, None, state_manager)
        
        # Display subsections
        for subsection in section.subsections:
            st.markdown(f"##### {subsection.name}")
            
            # Display items in the subsection
            for item in subsection.items:
                display_item(item, section, subsection, state_manager)

def display_item(item, section, subsection, state_manager):
    """Display an individual item with editing and feedback controls."""
    # Create a card-like container for the item
    with st.container():
        cols = st.columns([3, 1, 1, 1])
        
        # Status indicator and content
        with cols[0]:
            # Display status with color coding
            if item.status == ItemStatus.INITIAL:
                st.markdown(f"<span style='color:blue'>INITIAL</span>", unsafe_allow_html=True)
            elif item.status == ItemStatus.GENERATED:
                st.markdown(f"<span style='color:orange'>GENERATED</span>", unsafe_allow_html=True)
            elif item.status == ItemStatus.USER_EDITED:
                st.markdown(f"<span style='color:purple'>USER EDITED</span>", unsafe_allow_html=True)
            elif item.status == ItemStatus.TO_REGENERATE:
                st.markdown(f"<span style='color:red'>TO REGENERATE</span>", unsafe_allow_html=True)
            elif item.status == ItemStatus.ACCEPTED:
                st.markdown(f"<span style='color:green'>ACCEPTED</span>", unsafe_allow_html=True)
            elif item.status == ItemStatus.STATIC:
                st.markdown(f"<span style='color:gray'>STATIC</span>", unsafe_allow_html=True)
            
            # Editable content
            edited_content = st.text_area(
                f"Edit {item.item_type.value}",
                value=item.content,
                key=f"item_{item.id}",
                height=100
            )
            
            # Update content if edited
            if edited_content != item.content:
                if state_manager.update_item_content(item.id, edited_content):
                    if item.status != ItemStatus.USER_EDITED and item.status != ItemStatus.ACCEPTED:
                        state_manager.update_item_status(item.id, ItemStatus.USER_EDITED)
                    st.success("Content updated")
                    # Save state after update
                    state_manager.save_state()
        
        # Accept button
        with cols[1]:
            if st.button("Accept", key=f"accept_{item.id}"):
                if state_manager.update_item_status(item.id, ItemStatus.ACCEPTED):
                    st.success("Item accepted")
                    # Save state after update
                    state_manager.save_state()
        
        # Regenerate button (not for STATIC items)
        with cols[2]:
            if item.status != ItemStatus.STATIC:
                if st.button("Regenerate", key=f"regen_{item.id}"):
                    if state_manager.update_item_status(item.id, ItemStatus.TO_REGENERATE):
                        st.success("Item marked for regeneration")
                        # Save state after update
                        state_manager.save_state()
        
        # Feedback field
        with cols[3]:
            feedback = st.text_input(
                "Feedback",
                value=item.user_feedback or "",
                key=f"feedback_{item.id}"
            )
            if feedback != (item.user_feedback or ""):
                item_obj = state_manager.get_item(item.id)
                if item_obj:
                    item_obj.user_feedback = feedback
                    st.success("Feedback saved")
                    # Save state after update
                    state_manager.save_state()
        
        st.markdown("---")

def main():
    st.title("AI CV Generator - MVP")
    
    # Initialize session state for StateManager
    if 'state_manager' not in st.session_state:
        st.session_state.state_manager = StateManager()
    
    # Initialize session state for tracking regeneration
    if 'regenerate_items' not in st.session_state:
        st.session_state.regenerate_items = []

    # Check if the template file exists, if not, create it from the default template
    if not os.path.exists(TEMPLATE_FILE_PATH):
        default_template = """**Anas AKHOMACH** | 📞 (+212) 600310536 | 📧 [anasakhomach205@gmail.com](mailto:anasakhomach205@gmail.com) | 🔗 [LinkedIn](https://www.linkedin.com/in/anas-akhomach/) | 💻 [GitHub](https://github.com/AnasAkhomach)  
---

### Executive Summary

Data analyst with an educational background and strong communication skills. I combine in-depth knowledge of SQL, Python and Power BI with the ability to communicate complex topics in an easily understandable way.  
---

### Key Qualifications

Process optimization | Multilingual Service | Friendly communication | Data-Driven Sales | SQL | Python | Power BI | Excel
---

### Professional Experience

#### Trainee Data Analyst 

[*STE Smart-Send*](https://annoncelegale.flasheconomie.com/smart-send/) *| Tetouan, Morocco (Jan. 2024 – Mar. 2024)*

* Data-Driven Sales: Increased ROI using SQL/Python segmentation and timely Power BI metrics.  
* Process optimization: Streamlined KPI tracking, shortened decision time for a team of three people.  
* Teamwork: Developed solutions for different customer segments to improve customer service.

#### IT trainer

[*Supply Chain Management Center*](https://www.scmc.ma/) *| Tetouan, Morocco (June – Sep 2022, Jun – Sep 2024)*

* Technical Training: Conducted 100+ ERP dashboard sessions (MS Excel) with 95% satisfaction.  
* Friendly communication: Illustrated content with case studies for a quick start.  
* Process improvement: Focused on automated reporting and reduced manual data entry.

#### Mathematics teacher

[*Martile Secondary School*](https://www.facebook.com/ETChamsMartil/?locale=fr_FR) *| Tetouan, Morocco (Sep. 2017 – Jun. 2024)* 

* User Retention: Increased the class average from 3.7 to 3.3 through personalized learning plans and GeoGebra.  
* Friendly communication: Successfully guided 5+ students to top 10 placements in math competitions.  
* Multilingual Service: Supported non-native speakers in diverse classroom environments.

---

### Project Experience

#### ERP process automation and dashboard development | Sylob ERP, SQL, Excel VBA

* Automated manual data entry for raw material receiving and warehouse management, reducing manual errors and improving operational efficiency.
* Development and deployment of interactive Sylob ERP dashboards for warehouse staff and dock agents, providing real-time metrics and actionable insights.
* Integrated QR code scanners and rugged tablets to optimize material tracking, reduce processing time, and improve inventory accuracy by 35%.

#### SQL Analytics Project | SQL, Data Visualization

* Led a SQL-based analysis of an e-commerce database to optimize marketing budget and increase website traffic by 22%.
* Conducted A/B tests that improved checkout page conversion rates by 15% and reduced bounce rates by 22%.
* Worked with stakeholders to translate data insights into actionable strategies, resulting in a 12% reduction in cost per acquisition.

---

### Education

* Bachelor of Science in Applied Mathematics | Abdelmalek Essaâdi University | Tetouan, Morocco
* Professional Qualification in Pedagogy | CRMEF | Tangier, Morocco

---

### Certifications

* Business Intelligence Analyst | Maven Analytics (2024)
* Microsoft & LinkedIn Learning: Become a Data Analyst, SQL for Data Science, Advanced Excel

---

### Languages

* Arabic (native) | English (B2) | German (B2) | French (B2) | Spanish (B1)
"""
        # Create the template file
        with open(TEMPLATE_FILE_PATH, 'w', encoding='utf-8') as file:
            file.write(default_template)
        st.success(f"Created default CV template file at {TEMPLATE_FILE_PATH}")
    
    # Load the template
    template_manager = TemplateManager(template_path=TEMPLATE_FILE_PATH)
    
    # Add template info in sidebar
    with st.sidebar:
        st.subheader("CV Template Settings")
        st.info("""Your CV template is being used as the base. 
        
        Dynamic sections (will be tailored to match the job):
        - Executive Summary
        - Key Qualifications
        - Professional Experience
        - Project Experience
        
        Static sections (will remain unchanged):
        - Contact information
        - Education
        - Certifications
        - Languages
        """)
        
        # Add debug mode option
        debug_mode = st.checkbox("Enable Debug Mode", value=False)
        if debug_mode:
            st.info("Debug mode enabled. Check the debug.log file for detailed logs.")
            # Configure logging
            logging.basicConfig(level=logging.INFO, 
                         format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                         handlers=[
                             logging.FileHandler("debug.log"),
                             logging.StreamHandler()
                         ])
            # Show section detection info
            if os.path.exists("debug.log"):
                with open("debug.log", "r") as f:
                    log_content = f.read()
                    section_logs = [line for line in log_content.split('\n') if "Parsed section" in line]
                    if section_logs:
                        st.subheader("Section Detection Logs")
                        for log in section_logs[-10:]:  # Show last 10 section detection logs
                            st.code(log)
        
        if st.button("Edit Template"):
            # This could open the template in an editor or show an editing UI
            st.write("Template editing is not implemented yet. You can manually edit the cv_template.md file.")
            
        # Load saved session button
        st.subheader("Session Management")
        session_id = st.text_input("Session ID (leave empty for new session)", key="session_id_input")
        if st.button("Load Session"):
            if session_id:
                st.session_state.state_manager = StateManager(session_id=session_id)
                if st.session_state.state_manager.load_state():
                    st.success(f"Loaded session {session_id}")
                else:
                    st.error(f"Failed to load session {session_id}")
            else:
                st.warning("Please enter a session ID to load")

    # Initialize orchestrator variable
    orchestrator = None

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

    # Read template content to use as a placeholder
    template_content = ""
    try:
        with open(TEMPLATE_FILE_PATH, 'r', encoding='utf-8') as file:
            template_content = file.read()
    except Exception as e:
        st.warning(f"Could not read template file: {str(e)}")
        template_content = "John Doe\nSoftware Engineer with 5+ years of experience.\nExperience:\n- Worked on several projects.\nSkills:\n- Python\n- Java"

    # Create tabs for different stages of the process
    tab1, tab2, tab3 = st.tabs(["Input", "Review & Edit", "Export"])
    
    with tab1:
        st.subheader("Enter Job Description and CV")
        # Input fields for job description and CV
        job_description = st.text_area("Job Description", "Software Engineer position at Google", height=250)
        
        # Option to start from scratch
        start_from_scratch = st.checkbox("Start from scratch (no base CV)", key="start_from_scratch")
        
        if start_from_scratch:
            st.info("You'll create a new CV from scratch based on the job description.")
            user_cv = ""
        else:
            user_cv = st.text_area("Your CV", template_content, height=250, key="user_cv_input")
        
        if st.button("Start Tailoring Process", key="start_process_button"):
            if job_description:
                # Create session ID if not existing
                if not st.session_state.state_manager.session_id:
                    st.session_state.state_manager.session_id = str(uuid.uuid4())
                
                # Parse job description and CV
                with st.spinner("Parsing job description and CV..."):
                    parse_result = parser_agent.run({
                        "job_description": job_description,
                        "cv_text": user_cv,
                        "start_from_scratch": start_from_scratch
                    })
                
                # Store results in state manager
                job_data = parse_result["job_description_data"]
                structured_cv = parse_result["structured_cv"]
                
                # Store in state manager
                st.session_state.state_manager._structured_cv = structured_cv
                
                # Save state
                state_file = st.session_state.state_manager.save_state()
                if state_file:
                    st.success(f"Parsed data saved to {state_file}")
                    st.session_state.session_id = structured_cv.id
                    st.info(f"Your session ID is: {structured_cv.id}")
                
                # Display instructions to go to Review tab
                st.success("Parsing complete! Go to the Review & Edit tab to continue.")
                
                # If starting from scratch or there are items marked TO_REGENERATE, 
                # flag that we need to trigger content generation
                items_to_generate = structured_cv.get_items_by_status(ItemStatus.TO_REGENERATE)
                if items_to_generate:
                    st.session_state.regenerate_items = [item.id for item in items_to_generate]
            else:
                st.error("Please provide a job description.")
    
    with tab2:
        st.subheader("Review and Edit CV Content")
        
        # Check if we have a structured CV to display
        structured_cv = st.session_state.state_manager.get_structured_cv()
        
        if structured_cv:
            # Display session ID
            st.info(f"Session ID: {structured_cv.id}")
            
            # Check if we have items to regenerate
            if st.session_state.regenerate_items:
                with st.spinner("Generating content for marked items..."):
                    # Call the ContentWriterAgent to regenerate the marked items
                    try:
                        # Get the current structured CV state
                        current_cv = st.session_state.state_manager.get_structured_cv()
                        
                        # Call the content writer agent to regenerate specific items
                        result = orchestrator.content_writer_agent.run({
                            "structured_cv": current_cv,
                            "regenerate_item_ids": st.session_state.regenerate_items,
                            "job_description_data": orchestrator.parser_agent.get_job_data(),
                            "research_results": orchestrator.research_agent.get_research_results() if hasattr(orchestrator.research_agent, "get_research_results") else {}
                        })
                        
                        # Update the state manager with the modified structured CV
                        st.session_state.state_manager._structured_cv = result
                        
                        # Clear the regenerate list
                        st.session_state.regenerate_items = []
                        
                        # Save state
                        st.session_state.state_manager.save_state()
                        
                        logger.info(f"Successfully regenerated {len(st.session_state.regenerate_items)} items")
                        st.success("Content regeneration complete!")
                    except Exception as e:
                        logger.error(f"Error during content regeneration: {str(e)}\n{traceback.format_exc()}")
                        st.error(f"Error generating content: {str(e)}")
                    
                    # Force a rerun to update the UI
                st.rerun()
                
            # Display the structured CV content for review and editing
            if structured_cv.sections:
                # Sort sections by order
                sorted_sections = sorted(structured_cv.sections, key=lambda s: s.order)
                
                # Display each section
                for section in sorted_sections:
                    display_section(section, st.session_state.state_manager)
            else:
                st.warning("No sections found in the CV.")
            
            # Button to regenerate all TO_REGENERATE items
            items_to_regenerate = structured_cv.get_items_by_status(ItemStatus.TO_REGENERATE)
            if items_to_regenerate:
                if st.button("Regenerate Marked Items", key="regenerate_button"):
                    st.session_state.regenerate_items = [item.id for item in items_to_regenerate]
                    st.rerun()
            
            # Button to finalize the CV
            if st.button("Finalize CV", key="finalize_button"):
                # Check if all dynamic items have been accepted or are static
                all_items_ready = True
                problem_items = []
                
                for section in structured_cv.sections:
                    if section.content_type == "DYNAMIC":
                        # Check items directly in the section
                        for item in section.items:
                            if item.status not in [ItemStatus.ACCEPTED, ItemStatus.STATIC, ItemStatus.USER_EDITED]:
                                all_items_ready = False
                                problem_items.append(f"{section.name}: {item.content[:30]}...")
                        
                        # Check items in subsections
                        for subsection in section.subsections:
                            for item in subsection.items:
                                if item.status not in [ItemStatus.ACCEPTED, ItemStatus.STATIC, ItemStatus.USER_EDITED]:
                                    all_items_ready = False
                                    problem_items.append(f"{section.name} > {subsection.name}: {item.content[:30]}...")
                
                if all_items_ready:
                    # Move to the Export tab
                    st.success("CV is ready for export! Go to the Export tab.")
                    # Select the Export tab
                    # (Note: Streamlit doesn't support programmatically selecting tabs yet,
                    # so this is just a message to the user)
                else:
                    st.error("Some items still need review before finalizing.")
                    st.warning("Please accept or edit the following items:")
                    for item in problem_items[:5]:  # Show first 5 problem items
                        st.write(f"- {item}")
                    if len(problem_items) > 5:
                        st.write(f"...and {len(problem_items) - 5} more.")
        else:
            st.info("No CV data found. Please go to the Input tab to start the process.")
    
    with tab3:
        st.subheader("Export Tailored CV")
        
        # Check if we have a structured CV to export
        structured_cv = st.session_state.state_manager.get_structured_cv()
        
        if structured_cv:
            # Display session ID
            st.info(f"Session ID: {structured_cv.id}")
            
            # Convert StructuredCV to ContentData for compatibility with existing code
            content_data = structured_cv.to_content_data()
            
            # Format output options
            output_format = st.selectbox(
                "Select output format",
                ["Markdown", "PDF"],
                key="output_format"
            )
            
            if st.button("Generate Final CV", key="generate_final_button"):
                with st.spinner("Generating final CV..."):
                    try:
                        # For the MVP, we'll just show the Markdown content
                        # In a real implementation, this would call the formatter and template_renderer
                        
                        # Generate rendered Markdown (placeholder for real rendering)
                        rendered_cv = f"""
# {content_data.get('name', 'Your Name')}

## Contact Information
- Email: {content_data.get('email', '')}
- Phone: {content_data.get('phone', '')}
- LinkedIn: {content_data.get('linkedin', '')}
- GitHub: {content_data.get('github', '')}

## Executive Summary
{content_data.get('summary', '')}

## Key Qualifications
{content_data.get('skills_section', '')}

## Professional Experience
"""
                        # Add experience bullets
                        for bullet in content_data.get('experience_bullets', []):
                            rendered_cv += f"- {bullet}\n"
                        
                        # Add projects
                        rendered_cv += "\n## Project Experience\n"
                        for project in content_data.get('projects', []):
                            rendered_cv += f"### {project.get('name', 'Project')}\n"
                            for bullet in project.get('bullets', []):
                                rendered_cv += f"- {bullet}\n"
                        
                        # Add education
                        rendered_cv += "\n## Education\n"
                        for edu in content_data.get('education', []):
                            rendered_cv += f"- {edu}\n"
                        
                        # Add certifications
                        rendered_cv += "\n## Certifications\n"
                        for cert in content_data.get('certifications', []):
                            rendered_cv += f"- {cert}\n"
                        
                        # Add languages
                        rendered_cv += "\n## Languages\n"
                        for lang in content_data.get('languages', []):
                            rendered_cv += f"- {lang}\n"
                        
                        # Save the rendered CV
                        with open("tailored_cv.md", "w", encoding="utf-8") as f:
                            f.write(rendered_cv)
                        
                        # Display the rendered CV
                        st.markdown("### Final CV Preview")
                        st.markdown(rendered_cv)
                        
                        # Download button
                        st.download_button(
                            label="Download Tailored CV",
                            data=rendered_cv,
                            file_name="tailored_cv.md",
                            mime="text/markdown"
                        )
                        
                        # If PDF was selected, show a message
                        if output_format == "PDF":
                            st.info("PDF generation is not implemented in the MVP. Please use the Markdown output.")
                    
                    except Exception as e:
                        st.error(f"Error generating final CV: {str(e)}")
            
            # Display preview of current content
            st.subheader("Current CV Content Preview")
            
            # Create a simple preview of the current content
            preview = f"""
# {content_data.get('name', 'Your Name')}

## Summary
{content_data.get('summary', '')}

## Skills
{content_data.get('skills_section', '')}

## Experience
"""
            for bullet in content_data.get('experience_bullets', [])[:3]:  # Show first 3 bullets
                preview += f"- {bullet}\n"
            
            if len(content_data.get('experience_bullets', [])) > 3:
                preview += f"... and {len(content_data.get('experience_bullets', [])) - 3} more\n"
            
            st.markdown(preview)
        else:
            st.info("No CV data found. Please go to the Input tab to start the process.")

    # Existing state management code below
    # ...

if __name__ == "__main__":
    main()
