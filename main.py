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
from template_manager import TemplateManager
import streamlit as st
import os

# Create a constant for the template path
TEMPLATE_FILE_PATH = "cv_template.md"

def main():
    st.title("AI CV Generator - MVP")
    st.write("Enter a job description and we'll tailor your CV to match it")

    # Check if the template file exists, if not, create it from the default template
    if not os.path.exists(TEMPLATE_FILE_PATH):
        default_template = """**Anas AKHOMACH** | 📞 (+212) 600310536 | 📧 [anasakhomach205@gmail.com](mailto:anasakhomach205@gmail.com) | 🔗 [LinkedIn](https://www.linkedin.com/in/anas-akhomach/) | 💻 [GitHub](https://github.com/AnasAkhomach)  
---

### Executive Summary:

Data analyst with an educational background and strong communication skills . I combine in-depth knowledge of SQL , Python and Power BI with the ability to communicate complex topics in an easily understandable way.  
---

### **Key Qualifications:**

Process optimization  |  Multilingual Service  |  Friendly communication  |  Data-Driven Sales  |  Process optimization  |  Multilingual Service  |  Friendly communication  |  Data-Driven Sales   
---

### **Professional Experience:**

#### Trainee Data Analyst 

[*STE Smart-Send*](https://annoncelegale.flasheconomie.com/smart-send/) *| Tetouan, Morocco (Jan. 2024 – Mar. 2024\)*

* Data-Driven Sales: Increased ROI using SQL/Python segmentation and timely Power BI metrics.  
* Process optimization : Streamlined KPI tracking, shortened decision time for a team of three people.  
* Teamwork : Developed solutions for different customer segments to improve customer service.

#### IT trainer

[*Supply Chain Management Center*](https://www.scmc.ma/) *| Tetouan, Morocco (June – Sep 2022, Jun – Sep 2024\)*

* Technical Training : Conducted 100+ ERP dashboard sessions (MS Excel) with 95% satisfaction.  
* Friendly communication : Illustrated content with case studies for a quick start.  
* Process improvement : Focused on automated reporting and reduced manual data entry.

#### Mathematics teacher

[*Martile Secondary School*](https://www.facebook.com/ETChamsMartil/?locale=fr_FR) *| Tetouan, Morocco (Sep. 2017 – Jun. 2024\)* 

* User Retention : Increased the class average from 3.7 to 3.3 through personalized learning plans and GeoGebra.  
* Friendly communication : Successfully guided 5+ students to top 10 placements in math competitions.  
* Multilingual Service : Supported non-native speakers in diverse classroom environments.

#### Indie Mobile Game Developer

Unity

* Bullet 1: (AI-adjusted description)  
* Bullet 2: (AI-adjusted description)  
* Bullet 3: (AI-adjusted description)

---

### **Project Experience:**

#### ERP process automation and dynamic dashboard development | Sylob ERP, SQL, Excel VBA, QR scanner

* Automated manual data entry for raw material receiving and warehouse management, reducing manual errors and  
* Development and deployment of interactive Sylob ERP dashboards for warehouse staff and dock agents, providing  
* Integrated QR code scanners and rugged tablets to optimize material tracking, reduce processing time, and improve   
* Conducting technical and economic analyses of hardware/software solutions, selecting cost-effective tools within

#### SQL Analytics Consultant | Personal Project

* Led a SQL-based analysis of an e-commerce database to optimize marketing budget and increase website   
* Conducted A/B tests that improved checkout page conversion rates by 15% and reduced bounce rates by 22%.  
* Worked with stakeholders to translate data insights into actionable strategies, resulting in a 12% reduction in cos

---

### **Education:**

#### Professional Qualification in Pedagogy | [CRMEF](https://crmeftth.ma/) | Tangier, Morocco.

* Specialization in digital teaching methods and curriculum development.

#### Bachelor of Science in Applied Mathematics | [Abdelmalek Essaâdi University](https://www.uae.ac.ma/) | Tetouan, Morocco.

* Focus: Statistics, probability, data analysis, optimization.

#### Abitur (Mathematics & Engineering) | Lycée fqih Daoued | Mdiq, Morocco.

* Award: Best grade in mathematics (3.2).

---

### **Certifications:**

* [Certificate Assessment for Foreign University Degrees](https://drive.google.com/file/d/1IAqe9mDrTQEqXh-SZxq0KXVCE-oQDfuA/view?usp=sharing) ( ZAB ) | Germany (June 2024).  
* [Business Intelligence Analyst](https://certificates.mavenanalytics.io/12a3154f-87eb-44a9-8410-fdecffa8975f) (Maven Analytics, 2024).  
* [Microsoft & LinkedIn Learning](https://www.linkedin.com/in/anas-akhomach/details/certifications/) : Become a Data Analyst, Docker Foundations, SQL for Data Science, Advanced Excel

---

### **Languages:**

* **Arabic** (native speaker)  |  **English** \[ [TOEIC](https://www.ets.org/toeic/about.html) \]( [B2](https://drive.google.com/file/d/1CNV3dUMpyp6LKNVX0WVu-WdXYydtEzW-/view?usp=sharing) )  |   **German** \[ [telc](https://drive.google.com/file/d/17mMpFnVtimdROAH0vTbdqi_VHN_4HYo5/view?usp=sharing) \](B2)  |  **French** (B2)  |  **Spanish** (B1)  
* German : B2 (GER) – Active improvement through.
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
        The following sections will remain unchanged:
        - Contact information
        - Education
        - Certifications
        - Languages
        
        The following sections will be tailored to match the job description:
        - Executive Summary
        - Key Qualifications
        - Professional Experience
        - Project Experience
        """)
        
        if st.button("Edit Template"):
            # This could open the template in an editor or show an editing UI
            st.write("Template editing is not implemented yet. You can manually edit the cv_template.md file.")

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

    # Input fields for job description and CV
    job_description = st.text_area("Job Description", "Software Engineer position at Google", height=250)
    user_cv = st.text_area("Your CV", template_content, height=250)

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
        
    # New state variables to track section approvals
    if 'section_approvals' not in st.session_state:
        st.session_state.section_approvals = {
            'summary': False,
            'skills': False,
            'experience_items': {},  # Dictionary to track approval of individual experience items by index
            'projects': {},          # Dictionary to track approval of individual project items by index
            'education': False,
            'certifications': False,
            'languages': False
        }
    
    # Helper function to check if all sections are approved
    def all_sections_approved():
        # Get the content to check
        content = None
        if "content_data" in st.session_state.workflow_result:
            content = st.session_state.workflow_result["content_data"]
        elif "original_content_data" in st.session_state.workflow_result:
            content = st.session_state.workflow_result["original_content_data"]
        else:
            content = st.session_state.workflow_result.get("formatted_cv", {})
            
        # Check if summary and skills are approved
        if not st.session_state.section_approvals['summary'] or not st.session_state.section_approvals['skills']:
            return False
            
        # Check if all experience items are approved
        experience_items = content.get("experience_bullets", [])
        for idx in range(len(experience_items)):
            if idx not in st.session_state.section_approvals['experience_items'] or not st.session_state.section_approvals['experience_items'][idx]:
                return False
                
        # Check if all project items are approved (either from AI-generated or template)
        project_items = content.get("projects", [])
        if not project_items and template_manager:
            # If no AI-generated projects but template has projects, use those
            project_items = template_manager.extract_project_items()
            
        for idx in range(len(project_items)):
            if idx not in st.session_state.section_approvals['projects'] or not st.session_state.section_approvals['projects'][idx]:
                return False
                
        # Check if education, certifications, and languages are approved (if they exist)
        if content.get("education") and not st.session_state.section_approvals['education']:
            return False
        if content.get("certifications") and not st.session_state.section_approvals['certifications']:
            return False
        if content.get("languages") and not st.session_state.section_approvals['languages']:
            return False
            
        return True
    
    # Process section regeneration if needed
    if st.session_state.should_run_workflow:
        with st.spinner("Processing your feedback..."):
            try:
                job_description = st.session_state.job_description
                user_cv_data = st.session_state.user_cv_data
                
                # Initialize user_feedback variable
                user_feedback = None
                
                # Determine what type of regeneration is needed
                if st.session_state.regenerate_section:
                    section = st.session_state.regenerate_section
                    regenerate_only = []
                    section_feedback = []
                    
                    if section == "summary":
                        section_feedback = ["executive summary"]
                        regenerate_only = ["summary"]
                        feedback_message = "Please improve the executive summary."
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
                    elif section.startswith("bullet_"):
                        # Extract the experience index and bullet index to regenerate
                        # Format: bullet_exp-idx_bullet-idx
                        parts = section.split("_")
                        exp_index = int(parts[1])
                        bullet_index = int(parts[2])
                        section_feedback = [f"bullet point {bullet_index} in experience item {exp_index}"]
                        regenerate_only = ["experience_bullets"]
                        feedback_message = f"Please improve bullet point {bullet_index} in experience item {exp_index}."
                        
                        # Add the specific indices
                        user_feedback = {
                            "approved": False,
                            "comments": feedback_message,
                            "sections_feedback": section_feedback,
                            "regenerate_only": regenerate_only,
                            "experience_item_index": exp_index,
                            "bullet_index": bullet_index
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
                    elif section.startswith("proj_bullet_"):
                        # Extract the project index and bullet index to regenerate
                        # Format: proj_bullet_proj-idx_bullet-idx
                        parts = section.split("_")
                        proj_index = int(parts[2])
                        bullet_index = int(parts[3])
                        section_feedback = [f"bullet point {bullet_index} in project {proj_index}"]
                        regenerate_only = ["projects"]
                        feedback_message = f"Please improve bullet point {bullet_index} in project {proj_index}."
                        
                        # Add the specific indices
                        user_feedback = {
                            "approved": False,
                            "comments": feedback_message,
                            "sections_feedback": section_feedback,
                            "regenerate_only": regenerate_only,
                            "project_item_index": proj_index,
                            "bullet_index": bullet_index
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
                    if not user_feedback:
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
                        "sections_feedback": ["executive summary", "skills section", "experience section", "formatting"],
                        "regenerate_all": True
                    }
                else:
                    user_feedback = None
                
                # Run the workflow with the appropriate feedback
                # The user_feedback parameter is defined in orchestrator.py's run_workflow method
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
                
                # Reset section approvals for the new CV
                st.session_state.section_approvals = {
                    'summary': False,
                    'skills': False,
                    'experience_items': {},
                    'projects': {},
                    'education': False,
                    'certifications': False,
                    'languages': False
                }
                
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
        # Try to get the content data from the result
        if "content_data" in result:
            content_data = result["content_data"]
        elif "original_content_data" in result:
            content_data = result["original_content_data"]
        else:
            # Fallback to the formatted_cv field which might be a string
            content_data = result.get("formatted_cv", {})
        
        # Add debugging information
        with st.expander("Debug Information (click to expand)"):
            st.write("Result Structure:")
            st.json(result)
            st.write("Content Data Structure:")
            st.json(content_data)
            st.write("Section Approvals Structure:")
            st.json(st.session_state.section_approvals)
        
        # Check if content_data is None or empty
        if content_data is None:
            st.write("No content data available. Please complete all approvals and try again.")
            return
        
        # Check if content_data is a string or dictionary
        if isinstance(content_data, str):
            # If it's a string, display it directly
            st.markdown(content_data)
            
            # Add regeneration button
            if st.button("Regenerate Full CV"):
                st.session_state.regenerate_full = True
                st.session_state.should_run_workflow = True
                st.rerun()
                
        else:
            # If it's a dictionary, proceed with structured display
            # Get the template data
            template_projects = []
            template_experiences = []
            template_education = []
            template_certifications = []
            template_languages = []
            
            if template_manager:
                template_projects = template_manager.extract_project_items()
                template_experiences = template_manager.extract_experience_items()
                template_education = template_manager.get_section("education")
                template_certifications = template_manager.get_section("certifications")
                template_languages = template_manager.get_section("languages")
            
            # For static sections, if they're empty in the generated content, use template content
            if not content_data.get("education") and template_education:
                content_data["education"] = template_education
                
            if not content_data.get("certifications") and template_certifications:
                content_data["certifications"] = template_certifications
                
            if not content_data.get("languages") and template_languages:
                content_data["languages"] = template_languages

            # For dynamic sections, if they're empty, create placeholders with structure from template
            if not content_data.get("experience_bullets") and template_experiences:
                content_data["experience_bullets"] = template_experiences
            elif not content_data.get("experience_bullets"):
                # Create placeholder structure for professional experience if no template exists
                content_data["experience_bullets"] = [
                    {
                        "position": "Data Analyst Apprentice",
                        "company": "STE SMART-SEND",
                        "period": "Jan 2024 - Mar 2024",
                        "location": "Tetouan, Morocco",
                        "bullets": [
                            "Data Analysis & Reporting: Improved ROI of customer acquisition campaigns by 10% through segmented data analysis.",
                            "Power BI & Excel Expertise: Developed 2 Power BI dashboards for real-time KPI tracking, reducing team decision-making.",
                            "Analytical & Problem-Solving Skills: Utilized SQL and Python for data analysis to optimize targeting.",
                            "Process Optimization & Automation: Reduced team decision-making time through real-time KPI tracking dashboards.",
                            "Controlling & Financial Analysis: Contributed to financial analysis through data-driven insights on customer acquisition campaigns."
                        ]
                    },
                    {
                        "position": "IT Trainer",
                        "company": "Supply Chain Management Center",
                        "period": "June - Sep 2022, Jun - Sep 2024",
                        "location": "Tetouan, Morocco",
                        "bullets": [
                            "Data Analysis & Reporting: Delivered 100+ ERP dashboard sessions with 95% satisfaction rate, utilizing data analysis to optimize reporting processes.",
                            "Power BI & Excel Expertise: Created 100+ ERP dashboards using MS Excel, demonstrating expertise in data visualization and reporting.",
                            "ERP System Knowledge (Navision/Business Central): Delivered ERP system training, optimizing reporting processes through technical training sessions and process automation initiatives.",
                            "Process Optimization & Automation: Focused on automated reporting, reducing manual data entry by 30% through process optimization and automation initiatives.",
                            "Analytical & Problem-Solving Skills: Simplified onboarding using case studies, demonstrating analytical and problem-solving skills to improve training effectiveness."
                        ]
                    },
                    {
                        "position": "Mathematics Teacher",
                        "company": "Martile Secondary School",
                        "period": "Sep 2017 - Jun 2024",
                        "location": "Tetouan, Morocco",
                        "bullets": [
                            "Analytical & Problem-Solving Skills: Utilized personalized learning plans and GeoGebra tools to improve class average.",
                            "Data Analysis & Reporting: Improved class average from 3.7 to 3.3 through data-driven teaching methods.",
                            "Process Optimization & Automation: Streamlined student support through multilingual assistance."
                        ]
                    },
                    {
                        "position": "Indie Mobile Game Developer",
                        "company": "Independent Studio",
                        "period": "",
                        "location": "",
                        "bullets": [
                            "Process Optimization & Automation: Implemented automated build pipelines, reducing production cycles by 30% per title.",
                            "Analytical & Problem-Solving Skills: Conducted market analysis of top 100 casual games to identify genre trends and mechanics.",
                            "Data Analysis & Reporting: Monitored and analyzed game performance across 10+ published titles."
                        ]
                    },
                    {
                        "position": "Volunteer Hiking Expedition Leader",
                        "company": "Community Mountain Guides Network",
                        "period": "",
                        "location": "",
                        "bullets": [
                            "Analytical & Problem-Solving Skills: Created 15+ adaptive trail plans balancing weather patterns, group fitness levels, and daylight hours.",
                            "Process Optimization & Automation: Optimized time management system reducing late returns by 95% via strategic pause scheduling.",
                            "Data Analysis & Reporting: Analyzed post-hike feedback to increase participant retention by 40% through route customization."
                        ]
                    }
                ]
                
            if not content_data.get("projects") and template_projects:
                content_data["projects"] = template_projects
            elif not content_data.get("projects"):
                # Create placeholder structure for projects if no template exists
                content_data["projects"] = [
                    {
                        "name": "ERP Process Automation & Dashboarding",
                        "technologies": "Sylob ERP, SQL, Excel VBA, QR Scanners",
                        "bullets": [
                            "Data Analysis & Reporting: Built interactive ERP dashboards for production priorities and inventory access using Excel VBA, enhancing real-time inventory accuracy.",
                            "Power BI & Excel Expertise: Utilized Excel VBA to automate raw material entry and warehouse management, reducing errors and improving inventory accuracy.",
                            "Controlling & Financial Analysis: Conducted techno-economic analysis and deployed cost-effective hardware/software under a €10,000 budget.",
                            "Process Optimization & Automation: Automated raw material entry and warehouse management, reducing errors and improved real-time inventory accuracy using Sylob ERP and Excel VBA.",
                            "Analytical & Problem-Solving Skills: Integrated QR code workflows, cutting dock processing time and improving traceability, enhancing overall operational efficiency."
                        ]
                    },
                    {
                        "name": "E-Commerce Performance Optimization",
                        "technologies": "SQL, Google Analytics, A/B Testing Tools, Excel",
                        "bullets": [
                            "Data Analysis & Reporting: Led full-cycle analysis of 100K+ transaction e-commerce database, optimizing marketing spend and user experience.",
                            "Power BI & Excel Expertise: Created automated SQL reports in Excel tracking 15+ KPIs for real-time campaign performance monitoring.",
                            "Controlling & Financial Analysis: Translated data insights into strategies lowering customer acquisition costs by 12% (CPA).",
                            "Process Optimization & Automation: Designed and executed A/B test framework improving payment page conversion by 15% and reducing bounce rates by 22%.",
                            "Analytical & Problem-Solving Skills: Optimized e-commerce performance through data-driven strategies, reducing costs and enhancing user experience."
                        ]
                    }
                ]
                
            # Executive Summary / Summary section
            with st.container():
                st.markdown("## Executive Summary")
                with st.container(border=True):
                    summary_col, summary_buttons = st.columns([0.8, 0.2])
                    
                    with summary_col:
                        st.write(content_data.get("summary", "No summary available"))
                        
                        # Show approval status
                        if st.session_state.section_approvals['summary']:
                            st.success("✅ Approved")
                    
                    with summary_buttons:
                        # Only show regenerate button (remove approve button)
                        if st.button("Regenerate", key="regenerate_summary"):
                            st.session_state.regenerate_section = "summary"
                            st.session_state.should_run_workflow = True
                            st.session_state.section_approvals['summary'] = False
                            st.rerun()
            
            # Skills section
            with st.container():
                st.markdown("## Key Qualifications")
                with st.container(border=True):
                    skills_col, skills_buttons = st.columns([0.8, 0.2])
                    
                    with skills_col:
                        skills_data = content_data.get("skills_section", {})
                        
                        # Handle both direct string skills and structured skills data
                        if isinstance(skills_data, dict) and "skills" in skills_data:
                            skills = skills_data.get("skills", [])
                            if skills:
                                st.write(" | ".join(skills))
                            else:
                                st.write("No skills available")
                        else:
                            st.write(skills_data or "No skills available")
                        
                        # Show approval status
                        if st.session_state.section_approvals['skills']:
                            st.success("✅ Approved")
                            
                    with skills_buttons:
                        # Only show regenerate button
                        if st.button("Regenerate", key="regenerate_skills"):
                            st.session_state.regenerate_section = "skills"
                            st.session_state.should_run_workflow = True
                            st.session_state.section_approvals['skills'] = False
                            st.rerun()
            
            # Professional Experience section
            with st.container():
                st.markdown("## Professional Experience")
                experience_items = content_data.get("experience_bullets", [])
                if not experience_items:
                    with st.container(border=True):
                        st.write("No experience items available")
                        
                        # Add general regeneration button if no experience
                        if st.button("Generate Experience"):
                            st.session_state.regenerate_section = "experience"
                            st.session_state.should_run_workflow = True
                            st.rerun()
                else:
                    # For each experience item, add regenerate buttons
                    for idx, item in enumerate(experience_items):
                        with st.container(border=True):
                            if isinstance(item, dict):
                                # Handle structured experience data
                                company = item.get("company", "")
                                position = item.get("position", "")
                                period = item.get("period", "")
                                location = item.get("location", "")
                                bullets = item.get("bullets", [])
                                
                                item_col, btn_col = st.columns([0.8, 0.2])
                                
                                with item_col:
                                    # Create header with position and company
                                    header_text = f"**{position}**"
                                    if company:
                                        header_text += f" @ {company}"
                                    
                                    details = []
                                    if period:
                                        details.append(period)
                                    if location:
                                        details.append(location)
                                        
                                    if details:
                                        header_text += f" ({', '.join(details)})"
                                        
                                    st.markdown(header_text)
                                    
                                    # Display bullets with individual regenerate buttons
                                    for bullet_idx, bullet in enumerate(bullets):
                                        bullet_col, bullet_btn_col = st.columns([0.9, 0.1])
                                        with bullet_col:
                                            st.markdown(f"* {bullet}")
                                        with bullet_btn_col:
                                            if st.button("🔄", key=f"bullet_{idx}_{bullet_idx}"):
                                                st.session_state.regenerate_section = f"bullet_{idx}_{bullet_idx}"
                                                st.session_state.should_run_workflow = True
                                                st.rerun()
                                    
                                    # Show approval status (implied by not regenerating)
                                    if idx in st.session_state.section_approvals['experience_items'] and st.session_state.section_approvals['experience_items'][idx]:
                                        st.success("✅ Approved")
                                
                                with btn_col:
                                    # Button to regenerate this specific experience item
                                    if st.button("Regenerate", key=f"exp_item_{idx}"):
                                        st.session_state.regenerate_section = f"experience_item_{idx}"
                                        st.session_state.should_run_workflow = True
                                        if idx in st.session_state.section_approvals['experience_items']:
                                            st.session_state.section_approvals['experience_items'][idx] = False
                                        st.rerun()
                            else:
                                # Simple string format
                                item_col, btn_col = st.columns([0.8, 0.2])
                                
                                with item_col:
                                    st.write(item)
                                    
                                    # Show approval status
                                    if idx in st.session_state.section_approvals['experience_items'] and st.session_state.section_approvals['experience_items'][idx]:
                                        st.success("✅ Approved")
                                
                                with btn_col:
                                    # Button to regenerate this specific experience item
                                    if st.button("Regenerate", key=f"exp_item_{idx}"):
                                        st.session_state.regenerate_section = f"experience_item_{idx}"
                                        st.session_state.should_run_workflow = True
                                        if idx in st.session_state.section_approvals['experience_items']:
                                            st.session_state.section_approvals['experience_items'][idx] = False
                                        st.rerun()
                    
                    # Add button to generate a new experience item
                    if st.button("Add New Experience"):
                        st.session_state.regenerate_section = "add_experience"
                        st.session_state.should_run_workflow = True
                        st.rerun()
            
            # Projects section
            with st.container():
                st.markdown("## Project Experience")
                
                projects = content_data.get("projects", [])
                
                if not projects:
                    with st.container(border=True):
                        st.write("No projects available")
                        
                        # Add general regeneration button if no projects 
                        if st.button("Generate Projects"):
                            st.session_state.regenerate_section = "projects"
                            st.session_state.should_run_workflow = True
                            st.rerun()
                else:
                    # For each project, add regenerate buttons
                    for idx, project in enumerate(projects):
                        with st.container(border=True):
                            if isinstance(project, dict):
                                # Handle structured project data
                                name = project.get("name", "")
                                technologies = project.get("technologies", "")
                                bullets = project.get("bullets", [])
                                
                                project_col, btn_col = st.columns([0.8, 0.2])
                                
                                with project_col:
                                    # Create header with project name and technologies
                                    header_text = f"## {name}"
                                    if technologies:
                                        header_text += f" | **Tools**: {technologies}"
                                    
                                    st.markdown(header_text)
                                    
                                    # Display bullets with individual regenerate buttons
                                    for bullet_idx, bullet in enumerate(bullets):
                                        bullet_col, bullet_btn_col = st.columns([0.9, 0.1])
                                        with bullet_col:
                                            st.markdown(f"- {bullet}")
                                        with bullet_btn_col:
                                            if st.button("🔄", key=f"proj_bullet_{idx}_{bullet_idx}"):
                                                st.session_state.regenerate_section = f"proj_bullet_{idx}_{bullet_idx}"
                                                st.session_state.should_run_workflow = True
                                                st.rerun()
                                    
                                    # Show approval status (implied by not regenerating)
                                    if idx in st.session_state.section_approvals['projects'] and st.session_state.section_approvals['projects'][idx]:
                                        st.success("✅ Approved")
                                
                                with btn_col:
                                    # Button to regenerate this specific project
                                    if st.button("Regenerate", key=f"proj_item_{idx}"):
                                        st.session_state.regenerate_section = f"project_item_{idx}"
                                        st.session_state.should_run_workflow = True
                                        if idx in st.session_state.section_approvals['projects']:
                                            st.session_state.section_approvals['projects'][idx] = False
                                        st.rerun()
                            else:
                                # Simple string format
                                project_col, btn_col = st.columns([0.8, 0.2])
                                
                                with project_col:
                                    st.write(project)
                                    
                                    # Show approval status
                                    if idx in st.session_state.section_approvals['projects'] and st.session_state.section_approvals['projects'][idx]:
                                        st.success("✅ Approved")
                                
                                with btn_col:
                                    # Button to regenerate this specific project
                                    if st.button("Regenerate", key=f"proj_item_{idx}"):
                                        st.session_state.regenerate_section = f"project_item_{idx}"
                                        st.session_state.should_run_workflow = True
                                        if idx in st.session_state.section_approvals['projects']:
                                            st.session_state.section_approvals['projects'][idx] = False
                                        st.rerun()
                    
                    # Add button to generate a new project
                    if st.button("Add New Project"):
                        st.session_state.regenerate_section = "add_project"
                        st.session_state.should_run_workflow = True
                        st.rerun()
                        
            # Education section
            education_items = content_data.get("education", [])
            if education_items:
                with st.container():
                    st.markdown("## Education")
                    with st.container(border=True):
                        edu_col, edu_btn_col = st.columns([0.8, 0.2])
                        
                        with edu_col:
                            for edu in education_items:
                                if isinstance(edu, dict):
                                    degree = edu.get("degree", "")
                                    institution = edu.get("institution", "")
                                    location = edu.get("location", "")
                                    period = edu.get("period", "")
                                    details = edu.get("details", [])
                                    
                                    with st.container(border=True):
                                        # Format education entry according to preferred structure
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
                                else:
                                    with st.container(border=True):
                                        st.write(edu)
                            
                            # Show approval status (implied by static content)
                            if st.session_state.section_approvals['education']:
                                st.success("✅ Static Section - From Template")
                        
                        with edu_btn_col:
                            # No regeneration buttons for static sections
                            pass
            
            # Certifications section
            certifications = content_data.get("certifications", [])
            if certifications:
                with st.container():
                    st.markdown("## Certifications")
                    with st.container(border=True):
                        cert_col, cert_btn_col = st.columns([0.8, 0.2])
                        
                        with cert_col:
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
                            
                            # Show approval status (implied by static content)
                            if st.session_state.section_approvals['certifications']:
                                st.success("✅ Static Section - From Template")
                        
                        with cert_btn_col:
                            # No regeneration buttons for static sections
                            pass
            
            # Languages section
            languages = content_data.get("languages", [])
            if languages:
                with st.container():
                    st.markdown("## Language Proficiency")
                    with st.container(border=True):
                        lang_col, lang_btn_col = st.columns([0.8, 0.2])
                        
                        with lang_col:
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
                            
                            # Show approval status (implied by static content)
                            if st.session_state.section_approvals['languages']:
                                st.success("✅ Static Section - From Template")
                        
                        with lang_btn_col:
                            # No regeneration buttons for static sections
                            pass
            
            # Overall feedback
            with st.container(border=True):
                # Auto-approve all static sections
                st.session_state.section_approvals['education'] = True
                st.session_state.section_approvals['certifications'] = True
                st.session_state.section_approvals['languages'] = True
                
                # Auto-approve projects and experience items that weren't regenerated
                for idx in range(len(content_data.get("projects", []))):
                    if idx not in st.session_state.section_approvals['projects']:
                        st.session_state.section_approvals['projects'][idx] = True
                
                for idx in range(len(content_data.get("experience_bullets", []))):
                    if idx not in st.session_state.section_approvals['experience_items']:
                        st.session_state.section_approvals['experience_items'][idx] = True
                
                # Auto-approve summary and skills if not regenerated
                if not st.session_state.section_approvals['summary']:
                    st.session_state.section_approvals['summary'] = True
                
                if not st.session_state.section_approvals['skills']:
                    st.session_state.section_approvals['skills'] = True
                
                # Show finalize button
                if st.button("Finalize CV", key="finalize"):
                    with st.spinner("Finalizing your CV..."):
                        try:
                            # Create a final version with current content
                            result = orchestrator.run_workflow(
                                st.session_state.job_description,
                                st.session_state.user_cv_data,
                                user_feedback={
                                    "approved": True,
                                    "comments": "Ready to finalize",
                                    "content_to_keep": content_data
                                }
                            )
                            st.session_state.workflow_result = result
                            st.session_state.awaiting_feedback = False
                            st.session_state.workflow_stage = "complete"
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error finalizing CV: {str(e)}")
                
                # Add regenerate full CV button
                if st.button("Regenerate Full CV"):
                    st.session_state.regenerate_full = True
                    st.session_state.should_run_workflow = True
                    
                    # Reset all approval statuses
                    st.session_state.section_approvals = {
                        'summary': False,
                        'skills': False,
                        'experience_items': {},
                        'projects': {},
                        'education': False,
                        'certifications': False,
                        'languages': False
                    }
                    
                    st.rerun()
    
    # Display final CV if workflow is complete
    elif st.session_state.workflow_stage == "complete":
        st.header("Your Finalized CV")
        
        if st.session_state.workflow_result:
            rendered_cv = st.session_state.workflow_result.get("rendered_cv", "")
            if rendered_cv:
                # Display in a nicely formatted card
                with st.container(border=True):
                    st.markdown(rendered_cv)
                
                # Option to download the CV
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        label="Download CV as Markdown",
                        data=rendered_cv,
                        file_name="tailored_cv.md",
                        mime="text/markdown"
                    )
                
                with col2:
                    st.download_button(
                        label="Download CV as Text",
                        data=rendered_cv,
                        file_name="tailored_cv.txt",
                        mime="text/plain"
                    )
                
                # Add option to regenerate
                if st.button("Start Over"):
                    # Reset all state variables
                    st.session_state.workflow_result = None
                    st.session_state.awaiting_feedback = False
                    st.session_state.workflow_stage = None
                    st.session_state.regenerate_section = None
                    st.session_state.regenerate_full = False
                    st.session_state.should_run_workflow = False
                    st.session_state.section_approvals = {
                        'summary': False,
                        'skills': False,
                        'experience_items': {},
                        'projects': {},
                        'education': False,
                        'certifications': False,
                        'languages': False
                    }
                    st.rerun()
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
