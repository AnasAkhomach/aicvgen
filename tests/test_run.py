# test_run.py
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

# Set up the LLM
llm = LLM(timeout=30)

# Set up the vector database config and vector database
vector_config = VectorStoreConfig(dimension=1536, index_type="IndexFlatL2")
vector_db = VectorDB(config=vector_config)

# Set up all agents
parser_agent = ParserAgent(
    name="Job Description Parser",
    description="Parses job descriptions to extract key information.",
    llm=llm,
)

template_renderer = TemplateRenderer(
    name="CV Template Renderer",
    description="Renders the CV using templates.",
    model=llm,
    input_schema=AgentIO(
        input={"cv_content": dict},
        output=str,
        description="Renders a CV based on structured content.",
    ),
    output_schema=AgentIO(
        input={"cv_content": dict},
        output=str,
        description="The rendered CV in markdown format.",
    ),
)

vector_store_agent = VectorStoreAgent(
    name="Vector Store Agent",
    description="Manages vector store operations.",
    model=llm,
    input_schema=AgentIO(input={}, output={}, description="Vector store operations input."),
    output_schema=AgentIO(input={}, output={}, description="Vector store operations output."),
    vector_db=vector_db,
)

research_agent = ResearchAgent(
    name="Research Agent", description="Conducts research for CV enhancement.", llm=llm
)

cv_analyzer_agent = CVAnalyzerAgent(
    name="CV Analyzer", description="Analyzes CVs to extract key information.", llm=llm
)

tools_agent = ToolsAgent(
    name="Tools Agent",
    description="Provides utility functions.",
)

content_writer_agent = ContentWriterAgent(
    name="Content Writer",
    description="Generates content for the CV.",
    llm=llm,
    tools_agent=tools_agent,
)

formatter_agent = FormatterAgent(
    name="Formatter Agent", description="Formats CV content into presentable text."
)

quality_assurance_agent = QualityAssuranceAgent(
    name="Quality Assurance Agent",
    description="Ensures CV quality and alignment with job requirements.",
)

# Initialize the orchestrator
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
    llm=llm,
)

# Test data
job_description = "Software Engineer position at Google"
user_cv_data = CVData(
    raw_text="John Doe\nSoftware Engineer with 5+ years of experience.\nExperience:\n- Worked on several projects.\nSkills:\n- Python\n- Java",
    experiences=[],
    summary="",
    skills=[],
    education=[],
    projects=[],
)

# Run the workflow
print("Starting workflow test...")
result = orchestrator.run_workflow(job_description, user_cv_data)
print("\nWorkflow result:")
print(result)
