"""LangGraph-based CV Generation Workflow.

This module defines the state machine workflow using LangGraph's StateGraph.
It implements granular, item-by-item processing with user feedback loops.
"""

import logging
from typing import Dict, Any
from langgraph.graph import StateGraph, END

from src.orchestration.state import AgentState
from src.agents.parser_agent import ParserAgent
from src.agents.enhanced_content_writer import EnhancedContentWriterAgent
from src.agents.quality_assurance_agent import QualityAssuranceAgent
from src.agents.research_agent import ResearchAgent
from src.agents.formatter_agent import FormatterAgent
from src.services.llm_service import get_llm_service
from src.models.data_models import UserAction

logger = logging.getLogger(__name__)

# Define the workflow sequence for sections
WORKFLOW_SEQUENCE = ["key_qualifications", "professional_experience", "project_experience", "executive_summary"]

# Initialize services and agents
llm_service = get_llm_service()
parser_agent = ParserAgent(name="ParserAgent", description="Parses CV and JD.", llm_service=llm_service)
content_writer_agent = EnhancedContentWriterAgent()
qa_agent = QualityAssuranceAgent(name="QAAgent", description="Performs quality checks.", llm_service=llm_service)
from src.services.vector_db import get_enhanced_vector_db
vector_db = get_enhanced_vector_db()
research_agent = ResearchAgent(name="ResearchAgent", description="Conducts research and finds relevant CV content.", llm_service=llm_service, vector_db=vector_db)
formatter_agent = FormatterAgent(name="FormatterAgent", description="Generates PDF output from structured CV data.")

# Node wrapper functions for granular workflow
async def parser_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Parse job description and CV. Queue setup is now handled by generate_skills_node."""
    logger.info("Executing parser_node")
    logger.info(f"Parser input state keys: {list(state.keys())}")
    
    agent_state = AgentState.model_validate(state)
    logger.info(f"AgentState validation successful. Has structured_cv: {agent_state.structured_cv is not None}")
    logger.info(f"AgentState validation successful. Has job_description_data: {agent_state.job_description_data is not None}")
    
    # Parse inputs using the parser agent
    result = await parser_agent.run_as_node(agent_state)
    logger.info(f"Parser result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
    
    return result

async def content_writer_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Generate content for the current item specified in state.current_item_id."""
    agent_state = AgentState.model_validate(state)
    logger.info(f"Executing content_writer_node for item: {agent_state.current_item_id}")
    
    if not agent_state.current_item_id:
        logger.error("ContentWriter called without current_item_id")
        return {"error_messages": agent_state.error_messages + ["ContentWriter failed: No item ID."]}
    
    result = await content_writer_agent.run_as_node(agent_state)
    return result

async def qa_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Perform quality assurance on the generated content."""
    agent_state = AgentState.model_validate(state)
    logger.info(f"Executing qa_node for item: {agent_state.current_item_id}")
    result = await qa_agent.run_as_node(agent_state)
    return result

async def research_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Conduct research on job description and find relevant CV content."""
    logger.info("Executing research_node")
    agent_state = AgentState.model_validate(state)
    result = await research_agent.run_as_node(agent_state)
    return result

async def process_next_item_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Pop the next item from the queue and set it as current."""
    logger.info("Executing process_next_item_node")
    agent_state = AgentState.model_validate(state)
    
    if not agent_state.items_to_process_queue:
        logger.warning("No items in queue to process")
        return {}
    
    # Pop the next item from the queue
    queue_copy = agent_state.items_to_process_queue.copy()
    next_item_id = queue_copy.pop(0)
    
    logger.info(f"Processing next item: {next_item_id}")
    return {
        "current_item_id": next_item_id,
        "items_to_process_queue": queue_copy
    }

async def prepare_next_section_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Move to the next section in the workflow sequence."""
    logger.info("Executing prepare_next_section_node")
    agent_state = AgentState.model_validate(state)
    
    try:
        current_index = WORKFLOW_SEQUENCE.index(agent_state.current_section_key)
        if current_index + 1 >= len(WORKFLOW_SEQUENCE):
            logger.info("All sections completed")
            return {}
        
        next_section_key = WORKFLOW_SEQUENCE[current_index + 1]
        logger.info(f"Moving to next section: {next_section_key}")
        
        # Find the next section and populate the queue
        next_section = None
        for section in agent_state.structured_cv.sections:
            if section.name.lower().replace(' ', '_') == next_section_key:
                next_section = section
                break
        
        if next_section:
            item_queue = [str(item.id) for item in next_section.items]
            if next_section.subsections:
                for subsection in next_section.subsections:
                    item_queue.extend([str(item.id) for item in subsection.items])
            
            return {
                "current_section_key": next_section_key,
                "items_to_process_queue": item_queue,
                "current_item_id": None  # Reset current item
            }
    
    except (ValueError, IndexError) as e:
        logger.error(f"Error preparing next section: {e}")
        return {}
    
    return {}

async def formatter_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Generate the final PDF output using FormatterAgent."""
    logger.info("Executing formatter_node")
    agent_state = AgentState.model_validate(state)
    
    # Use FormatterAgent to generate PDF (now async)
    result = await formatter_agent.run_as_node(agent_state)
    
    # Update state with the result
    updated_state = agent_state.model_copy()
    if "final_output_path" in result:
        updated_state.final_output_path = result["final_output_path"]
    if "error_messages" in result:
        updated_state.error_messages.extend(result["error_messages"])
    
    return updated_state.model_dump()

async def generate_skills_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Generates the 'Big 10' skills and updates the CV state."""
    logger.info("--- Executing Node: generate_skills_node ---")
    agent_state = AgentState.model_validate(state)
    
    my_talents = ""  # Placeholder for now, could be extracted from original CV
    
    # Call the async method directly
    result = await content_writer_agent.generate_big_10_skills(
        agent_state.job_description_data.raw_text,
        my_talents
    )
    
    if result["success"]:
        updated_cv = agent_state.structured_cv.model_copy(deep=True)
        updated_cv.big_10_skills = result["skills"]
        updated_cv.big_10_skills_raw_output = result["raw_llm_output"]
        
        # Find the Key Qualifications section to populate it
        qual_section = None
        for section in updated_cv.sections:
            # Normalize section name for matching
            if section.name.lower().replace(":", "").strip() == "key qualifications":
                qual_section = section
                break
        
        if not qual_section:
            error_msg = "Could not find 'Key Qualifications' section to populate skills."
            logger.error(error_msg)
            return {"error_messages": agent_state.error_messages + [error_msg]}
        
        # Overwrite items with new skills and set up the processing queue for this section
        from src.models.data_models import Item, ItemStatus, ItemType
        qual_section.items = [Item(content=skill, status=ItemStatus.GENERATED, item_type=ItemType.KEY_QUALIFICATION) for skill in result["skills"]]
        item_queue = [str(item.id) for item in qual_section.items]
        
        logger.info(f"Populated 'Key Qualifications' with {len(item_queue)} skills and set up queue.")
        
        return {
            "structured_cv": updated_cv, 
            "items_to_process_queue": item_queue,
            "current_section_key": "key_qualifications",
            "is_initial_generation": True
        }
    else:
        return {"error_messages": agent_state.error_messages + [f"Skills generation failed: {result['error']}"]}

async def route_after_review(state: Dict[str, Any]) -> str:
    """Route based on user feedback and queue status."""
    agent_state = AgentState.model_validate(state)
    
    # Check if there's user feedback requiring regeneration
    if (agent_state.user_feedback and 
        agent_state.user_feedback.action == UserAction.REGENERATE):
        logger.info("User requested regeneration, routing to content_writer")
        return "regenerate"
    
    # Check if there are more items in the current section queue
    if agent_state.items_to_process_queue:
        logger.info("More items in queue, processing next item")
        return "next_item"
    
    # Check if there are more sections to process
    try:
        current_index = WORKFLOW_SEQUENCE.index(agent_state.current_section_key)
        if current_index + 1 < len(WORKFLOW_SEQUENCE):
            logger.info("Moving to next section")
            return "next_section"
    except (ValueError, IndexError):
        pass
    
    # All sections and items completed
    logger.info("All processing completed, routing to formatter")
    return "complete"

def build_cv_workflow_graph() -> StateGraph:
    """Build and return the granular CV workflow graph."""
    
    # Create the state graph
    workflow = StateGraph(AgentState)
    
    # Add nodes for granular processing
    workflow.add_node("parser", parser_node)
    workflow.add_node("research", research_node)
    workflow.add_node("generate_skills", generate_skills_node)
    workflow.add_node("process_next_item", process_next_item_node)
    workflow.add_node("content_writer", content_writer_node)
    workflow.add_node("qa", qa_node)
    workflow.add_node("prepare_next_section", prepare_next_section_node)
    workflow.add_node("formatter", formatter_node)
    
    # Set entry point
    workflow.set_entry_point("parser")
    
    # Add edges
    workflow.add_edge("parser", "research")
    workflow.add_edge("research", "generate_skills")
    workflow.add_edge("generate_skills", "process_next_item")
    workflow.add_edge("process_next_item", "content_writer")
    workflow.add_edge("content_writer", "qa")
    
    # Add conditional routing after QA/review
    workflow.add_conditional_edges(
        "qa",
        route_after_review,
        {
            "regenerate": "content_writer",  # User wants to regenerate current item
            "next_item": "process_next_item",  # More items in current section
            "next_section": "prepare_next_section",  # Move to next section
            "complete": "formatter"  # All processing done
        }
    )
    
    # After preparing next section, process its first item
    workflow.add_edge("prepare_next_section", "process_next_item")
    
    # Formatter ends the workflow
    workflow.add_edge("formatter", END)
    
    return workflow

# Singleton instance of the compiled graph
_workflow_graph = None

def get_cv_workflow_graph():
    """Get the compiled CV workflow graph (singleton pattern)."""
    global _workflow_graph
    if _workflow_graph is None:
        logger.info("Compiling CV workflow graph")
        _workflow_graph = build_cv_workflow_graph().compile()
    return _workflow_graph

# Export the compiled graph app for use in enhanced_orchestrator
cv_graph_app = get_cv_workflow_graph()