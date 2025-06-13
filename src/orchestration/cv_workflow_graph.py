"""LangGraph-based workflow orchestration for CV generation.

This module defines the state machine workflow using LangGraph's StateGraph.
It replaces the procedural orchestration logic with a stateful graph approach.
"""

import logging
from typing import Dict, Any
from langgraph.graph import StateGraph, END

from src.orchestration.state import AgentState
from src.agents.parser_agent import ParserAgent
from src.agents.research_agent import ResearchAgent
from src.agents.enhanced_content_writer import EnhancedContentWriterAgent
from src.agents.quality_assurance_agent import QualityAssuranceAgent
from src.services.llm_service import LLMService
from src.services.vector_service import VectorService
from src.core.config import Config

logger = logging.getLogger(__name__)

# Initialize services and agents
config = Config()
llm_service = LLMService(config)
vector_service = VectorService(config)

# Agent instances
parser_agent = ParserAgent(llm_service)
research_agent = ResearchAgent(llm_service, vector_service)
content_writer_agent = EnhancedContentWriterAgent(llm_service, vector_service)
qa_agent = QualityAssuranceAgent(llm_service)

# Node wrapper functions to adapt agent methods for LangGraph
def parse_inputs_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Parse job description and prepare initial state."""
    logger.info("Executing parse_inputs_node")
    agent_state = AgentState.model_validate(state)
    result = parser_agent.run_as_node(agent_state)
    return result

def research_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Conduct research based on job description."""
    logger.info("Executing research_node")
    agent_state = AgentState.model_validate(state)
    result = research_agent.run_as_node(agent_state)
    return result

def content_writer_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Generate or enhance CV content."""
    logger.info("Executing content_writer_node")
    agent_state = AgentState.model_validate(state)
    result = content_writer_agent.run_as_node(agent_state)
    return result

def qa_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Perform quality assurance on generated content."""
    logger.info("Executing qa_node")
    agent_state = AgentState.model_validate(state)
    result = qa_agent.run_as_node(agent_state)
    return result

def build_cv_workflow_graph() -> StateGraph:
    """Build and compile the CV generation workflow graph.
    
    Returns:
        Compiled StateGraph application ready for execution.
    """
    logger.info("Building CV workflow graph")
    
    # Create the StateGraph with AgentState as the state schema
    workflow = StateGraph(AgentState)
    
    # Add nodes for each agent/step
    workflow.add_node("parse_inputs", parse_inputs_node)
    workflow.add_node("research", research_node)
    workflow.add_node("generate_key_qualifications", content_writer_node)
    workflow.add_node("qa_key_qualifications", qa_node)
    workflow.add_node("generate_experience_item", content_writer_node)
    workflow.add_node("qa_experience_item", qa_node)
    workflow.add_node("generate_summary", content_writer_node)
    
    # Define the workflow edges
    workflow.set_entry_point("parse_inputs")
    workflow.add_edge("parse_inputs", "research")
    workflow.add_edge("research", "generate_key_qualifications")
    workflow.add_edge("generate_key_qualifications", "qa_key_qualifications")
    
    # Conditional logic will be expanded here to handle loops and user feedback
    # For now, a simplified linear flow is demonstrated.
    workflow.add_edge("qa_key_qualifications", "generate_experience_item")
    workflow.add_edge("generate_experience_item", "qa_experience_item")
    workflow.add_edge("qa_experience_item", "generate_summary")
    workflow.add_edge("generate_summary", END)
    
    # Compile the graph into a runnable application
    logger.info("Compiling CV workflow graph")
    return workflow.compile()

# Singleton instance of the compiled graph
cv_graph_app = build_cv_workflow_graph()