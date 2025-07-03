#!/usr/bin/env python3

import asyncio
import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.agents.executive_summary_writer_agent import ExecutiveSummaryWriterAgent
from src.orchestration.state import AgentState
from src.models.cv_models import StructuredCV, Section, Item, ItemType, ItemStatus
from src.models.data_models import JobDescriptionData
from uuid import uuid4
from unittest.mock import AsyncMock, MagicMock
from src.models.llm_service_models import LLMServiceResponse
from src.models.data_models import ContentType

async def debug_agent():
    # Create mock services
    mock_llm_service = AsyncMock()
    mock_llm_service.generate_content.return_value = LLMServiceResponse(content="Generated executive summary.")
    
    mock_template_manager = MagicMock()
    mock_template_manager.get_template_by_type.return_value = "Job Description: {job_description}\nKey Qualifications: {key_qualifications}\nProfessional Experience: {professional_experience}\nProjects: {projects}\nResearch Findings: {research_findings}"
    
    mock_settings = {"max_tokens_content_generation": 1024, "temperature_content_generation": 0.7}
    
    # Create sample data
    sample_structured_cv = StructuredCV(
        sections=[
            Section(name="Executive Summary", items=[Item(id=uuid4(), content="Original summary", item_type=ItemType.EXECUTIVE_SUMMARY_PARA)]),
            Section(name="Key Qualifications", items=[
                Item(id=uuid4(), content="Strategic thinker", item_type=ItemType.KEY_QUALIFICATION)
            ]),
            Section(name="Professional Experience", items=[
                Item(id=uuid4(), content="Led cross-functional teams.", item_type=ItemType.EXPERIENCE_ROLE_TITLE)
            ]),
            Section(name="Project Experience", items=[
                Item(id=uuid4(), content="Delivered complex projects.", item_type=ItemType.PROJECT_DESCRIPTION_BULLET)
            ])
        ]
    )
    
    sample_job_description_data = JobDescriptionData(
        raw_text="Software Engineer position at Tech Corp. Remote work. Responsibilities include developing software and writing tests. Required skills: Python, JavaScript, React.",
        job_title="Software Engineer",
        company_name="Tech Corp",
        skills=["Python", "JavaScript", "React"],
        responsibilities=["Develop software", "Write tests"]
    )
    
    # Create agent
    agent = ExecutiveSummaryWriterAgent(
        llm_service=mock_llm_service,
        template_manager=mock_template_manager,
        settings=mock_settings,
        session_id="test_session",
    )
    
    # Create initial state
    initial_state = AgentState(
        session_id="test_session",
        structured_cv=sample_structured_cv,
        job_description_data=sample_job_description_data,
        research_findings={"company_vision": "future-proof"},
        cv_text="mock cv text"
    )
    
    print("Running agent...")
    print(f"Initial state structured_cv: {initial_state.structured_cv is not None}")
    print(f"Initial state job_description_data: {initial_state.job_description_data is not None}")
    
    # Debug: Check what model_dump() produces
    state_dict = initial_state.model_dump()
    print(f"State dict keys: {list(state_dict.keys())}")
    print(f"structured_cv in dict: {'structured_cv' in state_dict}")
    print(f"job_description_data in dict: {'job_description_data' in state_dict}")
    
    result = await agent.run_as_node(initial_state)
    
    print(f"Result type: {type(result)}")
    print(f"Error messages: {result.error_messages}")
    print(f"Session ID: {result.session_id}")
    
    if result.error_messages:
        print("Errors found:")
        for error in result.error_messages:
            print(f"  - {error}")
    else:
        print("No errors!")
        # Check the executive summary
        summary_section = next((s for s in result.structured_cv.sections if s.name == "Executive Summary"), None)
        if summary_section:
            print(f"Summary content: {summary_section.items[0].content}")
        else:
            print("No Executive Summary section found")

if __name__ == "__main__":
    asyncio.run(debug_agent())