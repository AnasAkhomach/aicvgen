# mini_test.py
from orchestrator import Orchestrator
from parser_agent import ParserAgent
from template_renderer import TemplateRenderer
from vector_store_agent import VectorStoreAgent
from vector_db import VectorDB
from llm import LLM
from state_manager import VectorStoreConfig, CVData, AgentIO
from unittest.mock import MagicMock

# Set up the LLM
llm = LLM(timeout=30)

# Set up the parser agent
parser_agent = ParserAgent(
    name="Job Description Parser",
    description="Parses job descriptions to extract key information.",
    llm=llm
)

# Create mocks for other agents
template_renderer = MagicMock()
vector_db = MagicMock()
vector_store_agent = MagicMock()
content_writer_agent = MagicMock()
research_agent = MagicMock()
cv_analyzer_agent = MagicMock()
tools_agent = MagicMock()
formatter_agent = MagicMock()
quality_assurance_agent = MagicMock()

# Initialize the orchestrator with real parser agent but mocked others
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
    llm=llm
)

# Set up test state
test_state = {
    "job_description": "Software Engineer position at Google. Required skills: Python, Java, JavaScript, and cloud technologies. Experience level: 3-5 years.",
    "user_cv": {"raw_text": "John Doe\nSoftware Engineer with 5+ years experience\nSkills: Python, Java", "experiences": [], "summary": "", "skills": [], "education": [], "projects": []},
    "workflow_id": "test-workflow-id",
    "status": "in_progress",
    "stage": "started",
    "error": None,
    "parsed_job_description": None,
    "analyzed_cv": None,
    "content_data": {},
    "formatted_cv": None,
    "quality_analysis": None,
    "rendered_cv": None,
}

# Run just the parse job description node
print("Starting parse job description test...")
updated_state = orchestrator.run_parse_job_description_node(test_state)

# Print results
print("\nUpdated state:")
print(f"Stage: {updated_state.get('stage')}")
print(f"Error: {updated_state.get('error')}")
print(f"Status: {updated_state.get('status')}")

if updated_state.get('parsed_job_description'):
    print("\nParsed job description:")
    parsed_job = updated_state.get('parsed_job_description')
    print(f"Skills: {parsed_job.skills}")
    print(f"Experience Level: {parsed_job.experience_level}")
    print(f"Responsibilities: {parsed_job.responsibilities}")
    print(f"Industry Terms: {parsed_job.industry_terms}")
    print(f"Company Values: {parsed_job.company_values}")
else:
    print("\nNo parsed job description found in state.") 