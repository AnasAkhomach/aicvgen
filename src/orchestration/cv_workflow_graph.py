"""LangGraph-based CV Generation Workflow.

This module defines the state machine workflow using LangGraph's StateGraph.
It implements granular, item-by-item processing with user feedback loops.
"""

from typing import Dict, Any
from langgraph.graph import StateGraph, END

from ..orchestration.state import AgentState
from ..agents.parser_agent import ParserAgent
from ..agents.enhanced_content_writer import EnhancedContentWriterAgent
from ..agents.quality_assurance_agent import QualityAssuranceAgent
from ..agents.research_agent import ResearchAgent
from ..agents.formatter_agent import FormatterAgent
from ..agents.cv_analyzer_agent import CVAnalyzerAgent
from ..config.settings import get_config
from ..services.llm_service import EnhancedLLMService
from ..models.data_models import UserAction
from ..config.logging_config import get_structured_logger
from ..utils.node_validation import validate_node_output
from ..models.research_models import ResearchFindings
from ..models.quality_models import QualityCheckResults
from ..services.vector_store_service import VectorStoreService
from ..core.performance_optimizer import get_performance_optimizer
from ..services.rate_limiter import get_rate_limiter

logger = get_structured_logger(__name__)

# Define the workflow sequence for sections
WORKFLOW_SEQUENCE = [
    "key_qualifications",
    "professional_experience",
    "project_experience",
    "executive_summary",
]

# Initialize services and agents
settings = get_config()

performance_optimizer = get_performance_optimizer()
rate_limiter = get_rate_limiter()
llm_service = EnhancedLLMService(
    settings=settings,
    performance_optimizer=performance_optimizer,
    rate_limiter=rate_limiter,
)
parser_agent = ParserAgent(
    name="ParserAgent", description="Parses CV and JD.", llm_service=llm_service
)
content_writer_agent = EnhancedContentWriterAgent(llm_service=llm_service)
qa_agent = QualityAssuranceAgent(
    name="QAAgent", description="Performs quality checks.", llm_service=llm_service
)
vector_store_service = VectorStoreService(settings=settings)
research_agent = ResearchAgent(
    name="ResearchAgent",
    description="Conducts research and finds relevant CV content.",
    llm_service=llm_service,
    vector_db=vector_store_service,
)
formatter_agent = FormatterAgent(
    name="FormatterAgent",
    description="Generates PDF output from structured CV data.",
    llm_service=llm_service,
)
cv_analyzer_agent = CVAnalyzerAgent(
    name="CVAnalyzerAgent",
    description="Analyzes CV and extracts relevant information.",
    llm_service=llm_service,
    settings=settings,
)


# Node wrapper functions for granular workflow
@validate_node_output
async def parser_node(state: AgentState) -> AgentState:
    logger.info("Executing parser_node")
    logger.info(f"Parser input state - trace_id: {state.trace_id}")
    logger.info(
        f"AgentState validation successful. Has structured_cv: {state.structured_cv is not None}"
    )
    logger.info(
        f"AgentState validation successful. Has job_description_data: {state.job_description_data is not None}"
    )
    result = await parser_agent.run_as_node(state)
    if isinstance(result, dict):
        return state.model_copy(update=result)
    return result


@validate_node_output
async def content_writer_node(state: AgentState) -> AgentState:
    logger.info(f"Executing content_writer_node for item: {state.current_item_id}")

    if not state.current_item_id:
        # Try to get the next item from queue if available
        if state.items_to_process_queue:
            queue_copy = state.items_to_process_queue.copy()
            next_item_id = queue_copy.pop(0)
            logger.info(f"Auto-setting current_item_id to: {next_item_id}")
            # Update state and continue
            state = state.model_copy(
                update={
                    "current_item_id": next_item_id,
                    "items_to_process_queue": queue_copy,
                }
            )
        else:
            logger.error(
                "ContentWriter called without current_item_id and no items in queue"
            )
            return state.model_copy(
                update={
                    "error_messages": state.error_messages
                    + ["ContentWriter failed: No item ID."]
                }
            )

    result = await content_writer_agent.run_as_node(state)
    if isinstance(result, dict):
        return state.model_copy(update=result)
    return result


@validate_node_output
async def qa_node(state: AgentState) -> AgentState:
    logger.info(f"Executing qa_node for item: {state.current_item_id}")
    result = await qa_agent.run_as_node(state)
    if isinstance(result, dict):
        return state.model_copy(update=result)
    return result


@validate_node_output
async def research_node(state: AgentState) -> AgentState:
    logger.info("Executing research_node")
    result = await research_agent.run_as_node(state)
    if isinstance(result, dict):
        return state.model_copy(update=result)
    return result


@validate_node_output
async def process_next_item_node(state: AgentState) -> AgentState:
    logger.info("Executing process_next_item_node")

    if not state.items_to_process_queue:
        logger.warning("No items in queue to process")
        return state

    # Pop the next item from the queue
    queue_copy = state.items_to_process_queue.copy()
    next_item_id = queue_copy.pop(0)

    logger.info(f"Processing next item: {next_item_id}")
    return state.model_copy(
        update={
            "current_item_id": next_item_id,
            "items_to_process_queue": queue_copy,
        }
    )


@validate_node_output
async def prepare_next_section_node(state: AgentState) -> AgentState:
    logger.info("Executing prepare_next_section_node")

    try:
        current_index = WORKFLOW_SEQUENCE.index(state.current_section_key)
        if current_index + 1 >= len(WORKFLOW_SEQUENCE):
            logger.info("All sections completed")
            return state

        next_section_key = WORKFLOW_SEQUENCE[current_index + 1]
        logger.info(f"Moving to next section: {next_section_key}")

        # Find the next section and populate the queue
        next_section = None
        for section in state.structured_cv.sections:
            if section.name.lower().replace(" ", "_") == next_section_key:
                next_section = section
                break

        if next_section:
            item_queue = [str(item.id) for item in next_section.items]
            if next_section.subsections:
                for subsection in next_section.subsections:
                    item_queue.extend([str(item.id) for item in subsection.items])

            return state.model_copy(
                update={
                    "current_section_key": next_section_key,
                    "items_to_process_queue": item_queue,
                    "current_item_id": None,  # Reset current item
                }
            )

    except (ValueError, IndexError) as e:
        logger.error(f"Error preparing next section: {e}")
        return state

    return state


@validate_node_output
async def formatter_node(state: AgentState) -> AgentState:
    logger.info("Executing formatter_node")

    # Use FormatterAgent to generate PDF (now async)
    result = await formatter_agent.run_as_node(state)

    # Update state with the result
    updated_state = state.model_copy()
    if isinstance(result, dict):
        if "final_output_path" in result:
            updated_state.final_output_path = result["final_output_path"]
        if "error_messages" in result:
            updated_state.error_messages.extend(result["error_messages"])

    return updated_state


@validate_node_output
async def setup_generation_queue_node(state: AgentState) -> AgentState:
    logger.info("--- Executing Node: setup_generation_queue_node ---")
    content_queue = []

    # Collect all items from all sections that need content generation
    for section in state.structured_cv.sections:
        for item in section.items:
            content_queue.append(str(item.id))
        # Include subsection items if any
        if section.subsections:
            for subsection in section.subsections:
                for item in subsection.items:
                    content_queue.append(str(item.id))

    logger.info(
        f"Setup content generation queue with {len(content_queue)} items: {content_queue}"
    )

    return state.model_copy(update={"content_generation_queue": content_queue})


@validate_node_output
async def pop_next_item_node(state: AgentState) -> AgentState:
    logger.info("--- Executing Node: pop_next_item_node ---")

    if not state.content_generation_queue:
        logger.warning("Content generation queue is empty")
        return state

    # Pop the next item from the queue
    queue_copy = state.content_generation_queue.copy()
    next_item_id = queue_copy.pop(0)

    logger.info(
        f"Popped item {next_item_id} from content generation queue. Remaining: {len(queue_copy)}"
    )

    return state.model_copy(
        update={
            "current_item_id": next_item_id,
            "content_generation_queue": queue_copy,
        }
    )


@validate_node_output
async def prepare_regeneration_node(state: AgentState) -> AgentState:
    logger.info("--- Executing Node: prepare_regeneration_node ---")

    if not state.user_feedback or not state.user_feedback.item_id:
        logger.error("No user feedback or item_id for regeneration")
        return state.model_copy(
            update={
                "error_messages": state.error_messages
                + ["No item specified for regeneration"]
            }
        )

    item_id = str(state.user_feedback.item_id)
    logger.info(f"Preparing regeneration for item: {item_id}")

    return state.model_copy(
        update={
            "content_generation_queue": [item_id],
            "current_item_id": None,  # Will be set by pop_next_item_node
            "is_initial_generation": False,
        }
    )


@validate_node_output
async def generate_skills_node(state: AgentState) -> AgentState:
    logger.info("--- Executing Node: generate_skills_node ---")

    my_talents = ""  # Placeholder for now, could be extracted from original CV

    # Call the async method directly
    result = await content_writer_agent.generate_big_10_skills(
        state.job_description_data.raw_text, my_talents
    )

    if result["success"]:
        updated_cv = state.structured_cv.model_copy(deep=True)
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
            error_msg = (
                "Could not find 'Key Qualifications' section to populate skills."
            )
            logger.error(error_msg)
            return state.model_copy(
                update={"error_messages": state.error_messages + [error_msg]}
            )

        # Overwrite items with new skills and set up the processing queue for this section
        from ..models.data_models import Item, ItemStatus, ItemType

        qual_section.items = [
            Item(
                content=skill,
                status=ItemStatus.GENERATED,
                item_type=ItemType.KEY_QUALIFICATION,
            )
            for skill in result["skills"]
        ]
        item_queue = [str(item.id) for item in qual_section.items]

        logger.info(
            f"Populated 'Key Qualifications' with {len(item_queue)} skills and set up queue."
        )

        return state.model_copy(
            update={
                "structured_cv": updated_cv,
                "items_to_process_queue": item_queue,
                "current_section_key": "key_qualifications",
                "is_initial_generation": True,
            }
        )
    else:
        return state.model_copy(
            update={
                "error_messages": state.error_messages
                + [f"Skills generation failed: {result['error']}"]
            }
        )


@validate_node_output
async def error_handler_node(state: AgentState) -> AgentState:
    """
    Handle workflow errors by logging them and preparing for termination.

    This node is entered if the 'error_messages' field in the state is
    populated, indicating that a previous node encountered an error.
    """
    error_list = state.error_messages

    logger.error(f"Workflow terminated due to errors: {error_list}")

    # Return empty dict as we're terminating the workflow
    return state


@validate_node_output
async def cv_analyzer_node(state: AgentState) -> AgentState:
    """Analyze the user's CV and store results in state.cv_analysis_results."""
    logger.info("Executing cv_analyzer_node")
    result = await cv_analyzer_agent.run_as_node(state)
    # result is a CVAnalyzerNodeResult with cv_analysis_results as CVAnalysisResult
    return state.model_copy(update={"cv_analysis_results": result.cv_analysis_results})


def should_continue_generation(state: Dict[str, Any]) -> str:
    """Router function to determine if content generation loop should continue. Validates state."""
    agent_state = AgentState.model_validate(state)

    # Check for errors first
    if agent_state.error_messages:
        logger.warning("Errors detected in state, routing to error handler")
        return "error"

    # Check if there are more items in the content generation queue
    if agent_state.content_generation_queue:
        logger.info(
            f"Content generation queue has {len(agent_state.content_generation_queue)} items remaining, continuing loop"
        )
        return "continue"

    # No more items to process
    logger.info("Content generation queue is empty, completing workflow")
    return "complete"


async def route_after_qa(state: Dict[str, Any]) -> str:
    """Route after QA based on user feedback and workflow state. Validates state."""
    agent_state = AgentState.model_validate(state)

    # Priority 1: Check for errors first
    if agent_state.error_messages:
        logger.warning("Errors detected in state, routing to error handler")
        return "error"

    # Priority 2: Check if there's user feedback requiring regeneration
    if (
        agent_state.user_feedback
        and agent_state.user_feedback.action == UserAction.REGENERATE
    ):
        logger.info("User requested regeneration, routing to prepare_regeneration")
        return "regenerate"

    # Priority 3: Continue with content generation loop
    return should_continue_generation(state)


def build_cv_workflow_graph() -> StateGraph:
    """Build and return the refactored CV workflow graph with explicit content generation loop."""

    # Create the state graph
    workflow = StateGraph(AgentState)

    # Add nodes for the new architecture
    workflow.add_node("parser", parser_node)
    workflow.add_node("research", research_node)
    workflow.add_node("generate_skills", generate_skills_node)
    workflow.add_node("setup_generation_queue", setup_generation_queue_node)
    workflow.add_node("pop_next_item", pop_next_item_node)
    workflow.add_node("content_writer", content_writer_node)
    workflow.add_node("qa", qa_node)
    workflow.add_node("prepare_regeneration", prepare_regeneration_node)
    workflow.add_node("formatter", formatter_node)
    workflow.add_node("error_handler", error_handler_node)
    workflow.add_node("cv_analyzer", cv_analyzer_node)

    # Set entry point
    workflow.set_entry_point("parser")

    # Add linear edges for initial setup
    workflow.add_edge("parser", "research")
    workflow.add_edge("research", "generate_skills")
    workflow.add_edge("generate_skills", "setup_generation_queue")
    workflow.add_edge("setup_generation_queue", "pop_next_item")
    workflow.add_edge("pop_next_item", "content_writer")
    workflow.add_edge("content_writer", "qa")

    # Add conditional routing after QA
    workflow.add_conditional_edges(
        "qa",
        route_after_qa,
        {
            "error": "error_handler",  # Route to error handler if errors detected
            "regenerate": "prepare_regeneration",  # User wants to regenerate current item
            "continue": "pop_next_item",  # More items in content generation queue
            "complete": "formatter",  # All processing done
        },
    )

    # After preparing regeneration, pop the item and continue
    workflow.add_edge("prepare_regeneration", "pop_next_item")

    # Formatter ends the workflow
    workflow.add_edge("formatter", END)

    # Error handler terminates the workflow
    workflow.add_edge("error_handler", END)

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
