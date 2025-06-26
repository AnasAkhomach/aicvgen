"""LangGraph-based CV Generation Workflow.

This module defines the state machine workflow using LangGraph's StateGraph.
It implements granular, item-by-item processing with user feedback loops.
"""

from typing import Dict, Any, Optional
from langgraph.graph import StateGraph, END
import uuid

from ..orchestration.state import AgentState
from ..models.data_models import UserAction
from ..config.logging_config import get_structured_logger
from ..utils.node_validation import validate_node_output
from ..core.dependency_injection import get_container, DependencyContainer
from ..services.session_manager import SessionManager

logger = get_structured_logger(__name__)

# Define the workflow sequence for sections
WORKFLOW_SEQUENCE = [
    "key_qualifications",
    "professional_experience",
    "project_experience",
    "executive_summary",
]


class CVWorkflowGraph:
    """Manages the CV generation workflow using a state graph."""

    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id or str(uuid.uuid4())
        self.container: DependencyContainer = get_container()
        self.session_manager: SessionManager = self.container.get_by_name(
            "session_manager"
        )
        self.workflow = self._build_graph()
        self.app = self.workflow.compile()

        logger.info("CVWorkflowGraph initialized for session %s", self.session_id)

    def _get_agent(self, agent_name: str) -> Any:
        """Retrieve an agent from the container for the current session."""
        return self.container.get_by_name(
            agent_name, session_id=self.session_id
        )  # Node wrapper functions for granular workflow

    @validate_node_output
    async def parser_node(
        self, state: AgentState, config: Optional[Dict] = None
    ) -> AgentState:
        """Execute parser node to process CV and job description."""
        logger.info("Executing parser_node")
        logger.info("Parser input state - trace_id: %s", state.trace_id)
        logger.info(
            "AgentState validation successful. Has structured_cv: %s",
            state.structured_cv is not None,
        )
        logger.info(
            "AgentState validation successful. Has job_description_data: %s",
            state.job_description_data is not None,
        )

        try:
            parser_agent = self._get_agent("parser_agent")
            result = await parser_agent.run_as_node(state)
            if isinstance(result, dict):
                return state.model_copy(update=result)
            return result
        except (KeyError, AttributeError, RuntimeError) as exc:
            logger.error("Parser node failed: %s", exc)
            # Add error to state for centralized handling
            error_msg = f"ParserAgent failed: {str(exc)}"
            return state.model_copy(
                update={"error_messages": state.error_messages + [error_msg]}
            )

    @validate_node_output
    async def content_writer_node(
        self, state: AgentState, config: Optional[Dict] = None
    ) -> AgentState:
        """Execute content writer node for current item."""
        logger.info("Executing content_writer_node for item: %s", state.current_item_id)

        if not state.current_item_id:
            # Try to get the next item from queue if available
            if state.items_to_process_queue:
                queue_copy = state.items_to_process_queue.copy()
                next_item_id = queue_copy.pop(0)
                logger.info("Auto-setting current_item_id to: %s", next_item_id)
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

        content_writer_agent = self._get_agent("writer_agent")
        result = await content_writer_agent.run_as_node(state)
        if isinstance(result, dict):
            return state.model_copy(update=result)
        return result

    @validate_node_output
    async def qa_node(
        self, state: AgentState, config: Optional[Dict] = None
    ) -> AgentState:
        """Execute QA node for current item."""
        logger.info("Executing qa_node for item: %s", state.current_item_id)
        qa_agent = self._get_agent("qa_agent")
        result = await qa_agent.run_as_node(state)
        if isinstance(result, dict):
            return state.model_copy(update=result)
        return result

    async def process_next_item_node(
        self, state: AgentState, config: Optional[Dict] = None
    ) -> AgentState:
        """Process the next item from the queue."""
        logger.info("Executing process_next_item_node")

        if not state.items_to_process_queue:
            logger.warning("No items in queue to process")
            return state

        # Pop the next item from the queue
        queue_copy = state.items_to_process_queue.copy()
        next_item_id = queue_copy.pop(0)

        logger.info("Processing next item: %s", next_item_id)
        return state.model_copy(
            update={
                "current_item_id": next_item_id,
                "items_to_process_queue": queue_copy,
            }
        )

    @validate_node_output
    async def setup_generation_queue_node(
        self, state: AgentState, config: Optional[Dict] = None
    ) -> AgentState:
        """Set up the content generation queue with all items."""
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
            "Setup content generation queue with %s items: %s",
            len(content_queue),
            content_queue,
        )

        return state.model_copy(update={"content_generation_queue": content_queue})

    @validate_node_output
    async def pop_next_item_node(
        self, state: AgentState, config: Optional[Dict] = None
    ) -> AgentState:
        """Pop the next item from the content generation queue."""
        logger.info("--- Executing Node: pop_next_item_node ---")

        if not state.content_generation_queue:
            logger.warning("Content generation queue is empty")
            return state

        # Pop the next item from the queue
        queue_copy = state.content_generation_queue.copy()
        next_item_id = queue_copy.pop(0)

        logger.info(
            f"Popped item {next_item_id} from content generation queue. "
            f"Remaining: {len(queue_copy)}"
        )

        return state.model_copy(
            update={
                "current_item_id": next_item_id,
                "content_generation_queue": queue_copy,
            }
        )

    @validate_node_output
    async def prepare_regeneration_node(
        self, state: AgentState, config: Optional[Dict] = None
    ) -> AgentState:
        """Prepare for item regeneration based on user feedback."""
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
        logger.info("Preparing regeneration for item: %s", item_id)

        return state.model_copy(
            update={
                "content_generation_queue": [item_id],
                "current_item_id": None,  # Will be set by pop_next_item_node
                "is_initial_generation": False,
            }
        )

    @validate_node_output
    async def generate_skills_node(
        self, state: AgentState, config: Optional[Dict] = None
    ) -> AgentState:
        """Generate skills using the content writer agent."""
        logger.info("--- Executing Node: generate_skills_node ---")

        my_talents = ""  # Placeholder for now, could be extracted from original CV

        # Get the content writer agent and call the async method directly
        content_writer_agent = self._get_agent("writer_agent")
        result = await content_writer_agent.generate_big_10_skills(
            job_data={
                "job_description": state.job_description_data.raw_text,
                "my_talents": my_talents,
            }
        )

        if result["success"]:
            updated_cv = state.structured_cv.model_copy(deep=True)
            updated_cv.big_10_skills = result["skills"]
            updated_cv.big_10_skills_raw_output = result[
                "raw_llm_output"
            ]  # Find the Key Qualifications section to populate it
            qual_section = None
            for section in updated_cv.sections:
                # Normalize section name for matching (handle both formats)
                normalized_name = (
                    section.name.lower().replace(":", "").replace("_", " ").strip()
                )
                if normalized_name in ["key qualifications", "qualifications"]:
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

            # Import late to avoid circular imports
            from ..models.data_models import (
                Item,
                ItemStatus,
                ItemType,
            )  # pylint: disable=import-outside-toplevel

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
                f"Populated 'Key Qualifications' with {len(item_queue)} skills "
                f"and set up queue."
            )

            return state.model_copy(
                update={
                    "structured_cv": updated_cv,
                    "items_to_process_queue": item_queue,
                    "current_section_key": "key_qualifications",
                    "is_initial_generation": True,
                }
            )

        return state.model_copy(
            update={
                "error_messages": state.error_messages
                + [f"Skills generation failed: {result['error']}"]
            }
        )

    @validate_node_output
    async def formatter_node(self, state: AgentState, **kwargs) -> AgentState:
        """Execute formatter node to generate PDF output."""
        logger.info("Executing formatter_node")

        # Use FormatterAgent to generate PDF (now async)
        formatter_agent = self._get_agent("formatter_agent")
        result = await formatter_agent.run_as_node(state)

        # Update state with the result
        updated_state = state.model_copy()
        if isinstance(result, dict):
            if "final_output_path" in result:
                updated_state.final_output_path = result["final_output_path"]
            if "error_messages" in result:
                updated_state.error_messages.extend(result["error_messages"])

        return updated_state

    async def error_handler_node(self, state: AgentState, **kwargs) -> AgentState:
        """
        Centralized error handling node that processes agent failures and determines
        recovery actions. This implements Task A-04's centralized error recovery.
        """
        logger.info("Executing error_handler_node")

        # Check if there are any recent errors to handle
        if not state.error_messages:
            logger.warning("Error handler called but no error messages found")
            return state

        last_error = state.error_messages[-1]
        logger.error("Handling error: %s", last_error)

        try:
            # Get error recovery service from container
            container = self.container

            # Import ErrorRecoveryService locally to avoid import-time issues
            from ..services.error_recovery import (
                ErrorRecoveryService,
            )  # pylint: disable=import-outside-toplevel

            logger.info("Getting error recovery service from container")
            error_recovery_service = container.get_by_name("error_recovery_service")
            logger.info(
                f"Got error recovery service: {error_recovery_service}"
            )  # Use the error recovery service to determine the next action
            from ..models.data_models import ContentType

            recovery_action = await error_recovery_service.handle_error(
                Exception(last_error),
                state.current_item_id or "unknown",
                ContentType.QUALIFICATION,  # Default content type
                self.session_id,
                0,  # retry_count managed at orchestration level
                {
                    "workflow_step": "agent_execution",
                    "trace_id": state.trace_id,
                },
            )  # Apply recovery action
            updates = {}
            if recovery_action.strategy.value == "skip_item":
                logger.info("Skipping failed item: %s", state.current_item_id)
                updates["current_item_id"] = None

            elif recovery_action.strategy.value == "immediate_retry":
                logger.info("Marking item for retry: %s", state.current_item_id)
                # For now, we'll just clear the error and let the workflow retry

            elif recovery_action.strategy.value == "fallback_content":
                logger.info("Using fallback content")
                updates["fallback_content"] = (
                    recovery_action.fallback_content
                )  # Clear all errors (since the workflow ends after error handling)
            updates["error_messages"] = []

            return state.model_copy(update=updates)

        except (ImportError, KeyError, AttributeError, RuntimeError) as exc:
            logger.error("Error handler itself failed: %s", exc)
            # Return state with additional error
            return state.model_copy(
                update={
                    "error_messages": state.error_messages
                    + [f"Error handler failed: {exc}"]
                }
            )

    @validate_node_output
    async def cv_analyzer_node(self, state: AgentState, **kwargs) -> AgentState:
        """Analyze the user's CV and store results in state.cv_analysis_results."""
        logger.info("Executing cv_analyzer_node")
        cv_analyzer_agent = self._get_agent("cv_analyzer_agent")
        result = await cv_analyzer_agent.run_as_node(
            state
        )  # result is a CVAnalyzerNodeResult with cv_analysis_results as CVAnalysisResult
        return (
            state.model_copy(update={"cv_analysis_results": result.cv_analysis_results})
            @ validate_node_output
        )

    async def research_node(self, state: AgentState, **kwargs) -> AgentState:
        """Execute research node."""
        logger.info("Executing research_node")
        research_agent = self._get_agent("research_agent")
        result = await research_agent.run_as_node(state)
        if isinstance(result, dict):
            return state.model_copy(update=result)
        return result

    @validate_node_output
    async def writer_node(self, state: AgentState, **kwargs) -> AgentState:
        """Execute writer node for current item."""
        logger.info(
            f"Executing writer_node for item: {getattr(state, 'current_item_id', 'unknown')}"
        )
        writer_agent = self._get_agent("writer_agent")
        result = await writer_agent.run_as_node(state)
        if isinstance(result, dict):
            return state.model_copy(update=result)
        return result

    def should_continue_generation(self, state: Dict[str, Any]) -> str:
        """Router function to determine if content generation loop should continue."""
        agent_state = AgentState.model_validate(state)

        # Check for errors first
        if agent_state.error_messages:
            logger.warning("Errors detected in state, routing to error handler")
            return "error"

        # Check if there are more items in the content generation queue
        if agent_state.content_generation_queue:
            logger.info(
                f"Content generation queue has {len(agent_state.content_generation_queue)} "
                f"items remaining, continuing loop"
            )
            return "continue"

        # No more items to process
        logger.info("Content generation queue is empty, completing workflow")
        return "complete"

    def route_after_qa(self, state: Dict[str, Any]) -> str:
        """Route after QA based on user feedback and workflow state."""
        agent_state = AgentState.model_validate(state)

        # Priority 1: Check for user feedback first to ensure user intent is honored
        if (
            agent_state.user_feedback
            and agent_state.user_feedback.action == UserAction.REGENERATE
        ):
            logger.info("User requested regeneration, routing to prepare_regeneration")
            return "regenerate"

        # Priority 2: Check for errors if no explicit user action
        if agent_state.error_messages:
            logger.warning("Errors detected in state, routing to error handler")
            return "error"

        # Priority 3: Continue with content generation loop
        return self.should_continue_generation(state)

    async def handle_feedback_node(self, state: AgentState, **kwargs) -> AgentState:
        """Handles user feedback to refine content."""
        logger.info(
            f"Executing handle_feedback_node for item: {getattr(state, 'current_item_id', 'unknown')}"
        )  # This node would incorporate user feedback and re-run the writer
        # For now, it's a placeholder
        return state

    def _build_graph(self) -> StateGraph:
        """Build and return the refactored CV workflow graph with explicit content generation loop."""
        # Create the state graph
        workflow = StateGraph(AgentState)

        # Add nodes for the new architecture
        workflow.add_node("parser", self.parser_node)
        workflow.add_node("research", self.research_node)
        workflow.add_node("generate_skills", self.generate_skills_node)
        workflow.add_node("setup_generation_queue", self.setup_generation_queue_node)
        workflow.add_node("pop_next_item", self.pop_next_item_node)
        workflow.add_node("content_writer", self.content_writer_node)
        workflow.add_node("qa", self.qa_node)
        workflow.add_node("prepare_regeneration", self.prepare_regeneration_node)
        workflow.add_node("formatter", self.formatter_node)
        workflow.add_node("error_handler", self.error_handler_node)
        workflow.add_node("cv_analyzer", self.cv_analyzer_node)

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
            self.route_after_qa,
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

    async def invoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke the workflow with the given inputs."""
        return await self.app.ainvoke(
            inputs, config={"configurable": {"thread_id": self.session_id}}
        )
