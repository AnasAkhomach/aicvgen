import streamlit as st
from orchestrator import Orchestrator

# Initialize orchestrator
orchestrator = Orchestrator()

# Streamlit UI setup
st.title("AI CV Generator - Human-in-the-Loop")

# Step 1: Upload Job Description and CV
st.header("Upload Job Description and CV")
job_description = st.text_area("Paste the Job Description here:")
user_cv = st.text_area("Paste your CV here:")

if st.button("Submit Job Description and CV"):
    if job_description and user_cv:
        orchestrator.start_workflow(job_description, user_cv)
        st.success("Job description and CV submitted successfully!")
    else:
        st.error("Please provide both the job description and your CV.")

# Step 2: Display Extracted Skills and Generated CV Sections
st.header("Extracted Skills and Generated CV")
if orchestrator.current_state():
    extracted_skills = orchestrator.get_extracted_skills()
    generated_cv_sections = orchestrator.get_generated_cv_sections()

    st.subheader("Extracted Skills")
    st.write(extracted_skills)

    st.subheader("Generated CV Sections")
    for section in generated_cv_sections:
        st.text_area(f"Edit Section: {section['title']}", value=section['content'], key=section['title'])

    if st.button("Approve and Submit Feedback"):
        feedback = {section['title']: st.session_state[section['title']] for section in generated_cv_sections}
        orchestrator.submit_feedback(feedback)
        st.success("Feedback submitted successfully!")
else:
    st.info("Please upload a job description and CV to start.")