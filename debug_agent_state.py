#!/usr/bin/env python3

import sys
sys.path.append('src')

from src.orchestration.state import AgentState
from src.models.cv_models import StructuredCV, Section, Item, ItemStatus, JobDescriptionData
from src.models.workflow_models import ContentType
from src.models.agent_output_models import ResearchFindings, CVAnalysisResult, ResearchStatus
from datetime import datetime

# Create sample data
sample_cv_item = Item(
    content="Python programming expertise",
    status=ItemStatus.PENDING
)

sample_cv_section = Section(
    name="key_qualifications",
    items=[sample_cv_item]
)

sample_structured_cv = StructuredCV(
    sections=[sample_cv_section]
)

sample_job_description_data = JobDescriptionData(
    raw_text="Senior Software Engineer at TechCorp. We are looking for a senior software engineer with Python, Machine Learning, and 5+ years experience.",
    job_title="Senior Software Engineer",
    company_name="TechCorp",
    main_job_description_raw="We are looking for a senior software engineer...",
    responsibilities=["Develop software", "Lead projects"],
    skills=["Python", "Machine Learning", "TensorFlow", "AWS"]
)

try:
    # Test AgentState creation
    agent_state = AgentState(
        session_id="test_session_123",
        trace_id="test_trace_456",
        cv_text="Sample CV text content",
        structured_cv=sample_structured_cv,
        job_description_data=sample_job_description_data,
        research_findings=ResearchFindings(
            status=ResearchStatus.SUCCESS,
            research_timestamp=datetime.now(),
            key_terms=["Python", "Software Engineering"],
            enhancement_suggestions=["Add more technical details"]
        ),
        cv_analysis_results=CVAnalysisResult(
            summary="Sample analysis",
            key_skills=["Python", "AWS"],
            match_score=0.8
        ),
        current_section_key="key_qualifications",
        current_section_index=0,
        items_to_process_queue=["qual_1", "qual_2"],
        current_item_id="qual_1",
        current_content_type=ContentType.QUALIFICATION,
        is_initial_generation=True,
        error_messages=[],
        node_execution_metadata={}
    )
    print("AgentState created successfully!")
    print(f"Session ID: {agent_state.session_id}")
    print(f"Current section: {agent_state.current_section_key}")
except Exception as e:
    print(f"Error creating AgentState: {e}")
    import traceback
    traceback.print_exc()