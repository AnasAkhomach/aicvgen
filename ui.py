import streamlit as st
from orchestrator import Orchestrator, WorkflowState, CVData # Import necessary types
from uuid import uuid4

# --- Agent and LLM Initialization (Placeholder) ---
# In a real application, you would initialize your agents and LLM here
# and pass them to the Orchestrator. For this UI implementation,
# we'll assume simple initialization or that the Orchestrator handles it
# based on configuration. Replace with your actual agent/LLM initialization.
from llm import LLM
from parser_agent import ParserAgent
from template_renderer import TemplateRenderer
from vector_store_agent import VectorStoreAgent
from vector_db import VectorDB, VectorStoreConfig
from content_writer_agent import ContentWriterAgent
from research_agent import ResearchAgent
from cv_analyzer_agent import CVAnalyzerAgent
from tools_agent import ToolsAgent
from formatter_agent import FormatterAgent
from quality_assurance_agent import QualityAssuranceAgent
from state_manager import AgentIO

try:
    # Assuming LLM can be initialized without args for now, adjust as needed
    llm_instance = LLM()
    # Initialize other agents, passing llm_instance and other dependencies
    parser_agent_instance = ParserAgent(name="ParserAgent", description="Agent for parsing job descriptions.", llm=llm_instance)
    template_renderer_instance = TemplateRenderer(name="TemplateRenderer", description="Agent for rendering CV templates.", model=llm_instance, input_schema=AgentIO(input={}, output={}, description="template renderer"), output_schema=AgentIO(input={}, output={}, description="template renderer"))
    vector_db_config = VectorStoreConfig(dimension=768, index_type="IndexFlatL2")
    vector_db_instance = VectorDB(config=vector_db_config)
    vector_store_agent_instance = VectorStoreAgent(name="Vector Store Agent", description="Agent for managing vector store.", model=llm_instance, input_schema=AgentIO(input={}, output={}, description="vector store agent"), output_schema=AgentIO(input={}, output={}, description="vector store agent"), vector_db=vector_db_instance)
    tools_agent_instance = ToolsAgent(name="ToolsAgent", description="Agent for providing content processing tools.")
    content_writer_agent_instance = ContentWriterAgent(name="ContentWriterAgent", description="Agent for generating tailored CV content.", llm=llm_instance, tools_agent=tools_agent_instance)
    research_agent_instance = ResearchAgent(name="ResearchAgent", description="Agent for researching job-related information.", llm=llm_instance)
    cv_analyzer_agent_instance = CVAnalyzerAgent(name="CVAnalyzerAgent", description="Agent for analyzing user CVs.", llm=llm_instance)
    formatter_agent_instance = FormatterAgent(name="FormatterAgent", description="Agent for formatting CV content.")
    quality_assurance_agent_instance = QualityAssuranceAgent(name="QualityAssuranceAgent", description="Agent for performing quality checks on CV content.")

except Exception as e:
    st.error(f"Error initializing agents or LLM: {e}")
    st.stop() # Stop execution if agents fail to initialize

# --- Streamlit Session State Initialization ---
# Initialize orchestrator and workflow state in session state
if 'orchestrator' not in st.session_state:
    st.session_state.orchestrator = Orchestrator(
        parser_agent=parser_agent_instance,
        template_renderer=template_renderer_instance,
        vector_store_agent=vector_store_agent_instance,
        content_writer_agent=content_writer_agent_instance,
        research_agent=research_agent_instance,
        cv_analyzer_agent=cv_analyzer_agent_instance,
        tools_agent=tools_agent_instance,
        formatter_agent=formatter_agent_instance,
        quality_assurance_agent=quality_assurance_agent_instance,
        llm=llm_instance
    )
if 'thread_id' not in st.session_state:
    st.session_state.thread_id = str(uuid4())
if 'workflow_state' not in st.session_state:
    st.session_state.workflow_state = None
if 'workflow_running' not in st.session_state:
    st.session_state.workflow_running = False
if 'error_message' not in st.session_state:
    st.session_state.error_message = None

# --- UI Layout ---
st.title("AI CV Generator - Human-in-the-Loop")

# Display errors if any occurred during previous runs
if st.session_state.error_message:
    st.error(st.session_state.error_message)
    st.session_state.error_message = None # Clear error after displaying

# Step 1: Upload Job Description and CV
st.header("Upload Job Description and CV")
job_description_input = st.text_area("Paste the Job Description here:", key="job_description_input", height=200)
user_cv_input = st.text_area("Paste your CV here:", key="user_cv_input", height=200)

if st.button("Start Workflow", key="start_workflow_button", disabled=st.session_state.workflow_running):
    if job_description_input and user_cv_input:
        # Reset thread_id for a new workflow run
        st.session_state.thread_id = str(uuid4())
        st.session_state.workflow_running = True
        st.session_state.error_message = None # Clear any previous errors

        # Initialize the state for a new workflow run
        initial_state = WorkflowState(
             job_description={"raw_text": job_description_input},
             user_cv=CVData(raw_text=user_cv_input, experiences=[], summary="", skills=[], education=[], projects=[]), # Provide a basic CVData structure
             extracted_skills={},
             generated_content={},
             formatted_cv_text="",
             rendered_cv="",
             feedback=[],
             revision_history=[],
             current_stage={"stage_name": "Initialized", "description": "Workflow initialized", "is_completed": True},
             workflow_id=st.session_state.thread_id,
             relevant_experiences=[],
             research_results={},
             quality_assurance_results={},
             review_status="pending", # Initialize review_status
             review_feedback="" # Initialize review_feedback
        )

        try:
            with st.spinner("Starting workflow..."):
                # The invoke method with a checkpointer will automatically save state and pause at human_review
                st.session_state.workflow_state = st.session_state.orchestrator.graph.invoke(
                    initial_state,
                    config={"configurable": {"thread_id": st.session_state.thread_id}}
                )
            st.success("Workflow started. Check the next section for results or review.")
        except Exception as e:
            st.session_state.error_message = f"An error occurred during workflow execution: {e}"
            st.session_state.workflow_running = False
            st.experimental_rerun() # Rerun to show the error

    else:
        st.session_state.error_message = "Please provide both the job description and your CV."
        st.experimental_rerun() # Rerun to show the error

# Step 2: Display Workflow Status and Human Review Section
st.header("Workflow Progress and Review")

if st.session_state.workflow_state:
    current_stage = st.session_state.workflow_state.get("current_stage", {}).get("stage_name", "Unknown")
    st.info(f"Current Workflow Stage: {current_stage}")

    if current_stage == "Human Review":
        st.subheader("Human Review Required")
        formatted_cv_text = st.session_state.workflow_state.get("formatted_cv_text", "No formatted content available for review.")

        # Use the key to preserve edits across reruns
        edited_cv_content = st.text_area("Review and Edit CV Content:", value=formatted_cv_text, height=500, key=f"review_cv_content_{st.session_state.thread_id}")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Approve CV", key="approve_button"):
                st.session_state.workflow_running = True # Indicate workflow is running again
                st.session_state.error_message = None # Clear errors

                # Update the state with approval and edited content before invoking
                updated_state = st.session_state.workflow_state
                updated_state["review_status"] = "approve"
                updated_state["formatted_cv_text"] = edited_cv_content # Save edited content

                try:
                    with st.spinner("Approving and continuing workflow..."):
                         # Invoke the graph with the updated state and thread_id
                        st.session_state.workflow_state = st.session_state.orchestrator.graph.invoke(
                            updated_state, # Pass the updated state
                            config={"configurable": {"thread_id": st.session_state.thread_id}}
                             # No need to pass input={"review_status": "approve"} here
                             # as we are updating the state directly
                        )
                    st.success("CV Approved. Workflow Continuing...")
                    # After approval, the workflow should proceed to rendering and finish
                    st.session_state.workflow_running = False # Assuming workflow finishes after rendering
                    st.experimental_rerun() # Rerun to update the UI stage

                except Exception as e:
                    st.session_state.error_message = f"An error occurred during approval: {e}"
                    st.session_state.workflow_running = False
                    st.experimental_rerun() # Rerun to show the error


        with col2:
            if st.button("Reject and Refine", key="reject_button"):
                 st.session_state.workflow_running = True # Indicate workflow is running again
                 st.session_state.error_message = None # Clear errors

                 # Update the state with rejection and feedback (edited content) before invoking
                 updated_state = st.session_state.workflow_state
                 updated_state["review_status"] = "reject"
                 updated_state["formatted_cv_text"] = edited_cv_content # Save edited content
                 updated_state["review_feedback"] = "Rejected by user with edits." # Add more specific feedback if needed

                 try:
                     with st.spinner("Rejecting and requesting refinement..."):
                         # Invoke the graph with the updated state and thread_id
                         st.session_state.workflow_state = st.session_state.orchestrator.graph.invoke(
                             updated_state, # Pass the updated state
                             config={"configurable": {"thread_id": st.session_state.thread_id}}
                              # No need to pass input={"review_status": "reject"} here
                              # as we are updating the state directly
                         )
                     st.warning("CV Rejected. Workflow Looping for Refinement...")
                     # The workflow should loop back to content generation, so it's still running
                     # st.session_state.workflow_running = True # Keep running is already set above
                     st.experimental_rerun() # Rerun to update the UI stage

                 except Exception as e:
                    st.session_state.error_message = f"An error occurred during rejection: {e}"
                    st.session_state.workflow_running = False
                    st.experimental_rerun() # Rerun to show the error


    elif current_stage == "END":
         st.subheader("Workflow Completed")
         final_cv = st.session_state.workflow_state.get("rendered_cv", "Could not retrieve final rendered CV.")
         st.text_area("Final Tailored CV:", value=final_cv, height=600, key=f"final_cv_display_completed_{st.session_state.thread_id}", disabled=True)
         st.session_state.workflow_running = False # Workflow is finished


    # Display other relevant information based on state (Optional)
    # You might want to display these only if the workflow is NOT at Human Review or END stage
    if current_stage not in ["Human Review", "END"]:
        if "extracted_skills" in st.session_state.workflow_state and st.session_state.workflow_state["extracted_skills"]:
             st.subheader("Extracted Skills")
             # Ensure extracted_skills is a displayable format, e.g., list of strings
             skills_display = st.session_state.workflow_state["extracted_skills"]
             if isinstance(skills_display, dict):
                 st.json(skills_display) # Or format it nicely
             else:
                 st.write(skills_display)

        if "generated_content" in st.session_state.workflow_state and st.session_state.workflow_state["generated_content"]:
            st.subheader("Generated Content Sections (Raw)")
            # Display raw generated content for debugging or review
            st.json(st.session_state.workflow_state["generated_content"])

    # Display current stage description and completion status
    current_stage_info = st.session_state.workflow_state.get("current_stage", {})
    if current_stage_info:
        st.markdown(f"**Stage Description:** {current_stage_info.get('description', 'N/A')}")
        st.markdown(f"**Completed:** {current_stage_info.get('is_completed', 'N/A')}")


elif st.session_state.workflow_state and st.session_state.workflow_state.get("current_stage", {}).get("stage_name") == "END":
     st.subheader("Workflow Completed")
     final_cv = st.session_state.workflow_state.get("rendered_cv", "Could not retrieve final rendered CV.")
     st.text_area("Final Tailored CV:", value=final_cv, height=600, key=f"final_cv_display_completed_{st.session_state.thread_id}", disabled=True)
     st.session_state.workflow_running = False # Workflow is finished
else:
    st.info("Upload a job description and CV and click 'Start Workflow' to begin.")

# Optional: Display full state for debugging
# st.subheader("Full Workflow State (for debugging)")
# st.write(st.session_state.workflow_state)
