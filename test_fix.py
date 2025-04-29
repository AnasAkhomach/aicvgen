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

def run_test():
    # Initialize components
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
    
    # Create a basic CV
    user_cv_data = CVData(raw_text="John Doe\nSoftware Engineer", experiences=[], summary="", skills=[], education=[], projects=[])
    
    # Run the workflow
    result = orchestrator.run_workflow("Software Engineer job description", user_cv_data)
    
    print(f"Result: {result}")

if __name__ == "__main__":
    run_test() 