"""LangGraph-based CV Generation Workflow.

This module defines the state machine workflow using LangGraph's StateGraph.
It implements granular, item-by-item processing with user feedback loops.
"""

from typing import Dict, Any, Optional
from langgraph.graph import StateGraph, END, MessageGraph
import uuid

from ..orchestration.state import AgentState
from ..models.workflow_models import UserAction
from ..config.logging_config import get_structured_logger
from ..utils.node_validation import validate_node_output
from ..core.container import get_container
from ..services.error_recovery import ErrorRecoveryService
from ..models.cv_models import Item, ItemStatus, ItemType
from ..models.workflow_models import ContentType

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
        self.container = get_container()
        self.workflow = self._build_graph()
        self.app = self.workflow.compile()

        logger.info(f"CVWorkflowGraph initialized for session {self.session_id}")

    def _get_agent(self, agent_name: str) -> Any:
        """Retrieve an agent from the container for the current session."""
        if hasattr(self.container, agent_name):
            return getattr(self.container, agent_name)()
        else:
            raise AttributeError(f"Agent '{agent_name}' not found in container")

    @validate_node_output
    async def jd_parser_node(
        self, state: AgentState, config: Optional[Dict] = None
    ) -> AgentState:
        """Execute parser node to process job description."""
        logger.info(f"Executing jd_parser_node")
        try:
            jd_parser_agent = self._get_agent("job_description_parser_agent")
            result = await jd_parser_agent.run_as_node(state)
            if isinstance(result, dict):
                return state.model_copy(update=result)
            return result
        except (KeyError, AttributeError, RuntimeError) as exc:
            logger.error("JD Parser node failed: %s", exc)
            error_msg = f"JobDescriptionParserAgent failed: {str(exc)}"
            return state.model_copy(
                update={"error_messages": state.error_messages + [error_msg]}
            )

    @validate_node_output
    async def cv_parser_node(
        self, state: AgentState, config: Optional[Dict] = None
    ) -> AgentState:
        """Execute parser node to process CV."""
        logger.info(f"Executing cv_parser_node")
        try:
            cv_parser_agent = self._get_agent("user_cv_parser_agent")
            result = await cv_parser_agent.run_as_node(state)
            if isinstance(result, dict):
                return state.model_copy(update=result)
            return result
        except (KeyError, AttributeError, RuntimeError) as exc:
            logger.error("CV Parser node failed: %s", exc)
            error_msg = f"UserCVParserAgent failed: {str(exc)}"
            return state.model_copy(
                update={"error_messages": state.error_messages + [error_msg]}
            )

    @validate_node_output
    async def research_node(self, state: AgentState, **kwargs) -> AgentState:
        """Execute research node."""
        logger.info("Executing research_node")
        research_agent = self._get_agent("research_agent")
        result = await research_agent.run_as_node(state)
        if isinstance(result, dict):
            return state.model_copy(update=result)
        return result

    @validate_node_output
    async def cv_analyzer_node(self, state: AgentState, **kwargs) -> AgentState:
        """Analyze the user's CV and store results in state.cv_analysis_results."""
        logger.info("Executing cv_analyzer_node")
        cv_analyzer_agent = self._get_agent("cv_analyzer_agent")
        result = await cv_analyzer_agent.run_as_node(state)
        return state.model_copy(
            update={"cv_analysis_results": result.cv_analysis_results}
        )

    @validate_node_output
    async def key_qualifications_writer_node(
        self, state: AgentState, config: Optional[Dict] = None
    ) -> AgentState:
        """Execute key qualifications writer node."""
        logger.info(f"Executing key_qualifications_writer_node")
        agent = self._get_agent("key_qualifications_writer_agent")
        result = await agent.run_as_node(state)
        if isinstance(result, dict):
            return state.model_copy(update=result)
        return result

    @validate_node_output
    async def professional_experience_writer_node(
        self, state: AgentState, config: Optional[Dict] = None
    ) -> AgentState:
        """Execute professional experience writer node."""
        logger.info(f"Executing professional_experience_writer_node")
        agent = self._get_agent("professional_experience_writer_agent")
        result = await agent.run_as_node(state)
        if isinstance(result, dict):
            return state.model_copy(update=result)
        return result

    @validate_node_output
    async def projects_writer_node(
        self, state: AgentState, config: Optional[Dict] = None
    ) -> AgentState:
        """Execute projects writer node."""
        logger.info(f"Executing projects_writer_node")
        agent = self._get_agent("projects_writer_agent")
        result = await agent.run_as_node(state)
        if isinstance(result, dict):
            return state.model_copy(update=result)
        return result

    @validate_node_output
    async def executive_summary_writer_node(
        self, state: AgentState, config: Optional[Dict] = None
    ) -> AgentState:
        """Execute executive summary writer node."""
        logger.info(f"Executing executive_summary_writer_node")
        agent = self._get_agent("executive_summary_writer_agent")
        result = await agent.run_as_node(state)
        if isinstance(result, dict):
            return state.model_copy(update=result)
        return result

    @validate_node_output
    async def qa_node(
        self, state: AgentState, config: Optional[Dict] = None
    ) -> AgentState:
        """Execute QA node for current item."""
        logger.info(f"Executing qa_node for item: {state.current_item_id}")
        qa_agent = self._get_agent("qa_agent")
        result = await qa_agent.run_as_node(state)
        if isinstance(result, dict):
            return state.model_copy(update=result)
        return result

    @validate_node_output
    async def formatter_node(self, state: AgentState, **kwargs) -> AgentState:
        """Execute formatter node to generate PDF output."""
        logger.info("Executing formatter_node")
        formatter_agent = self._get_agent("formatter_agent")
        result = await formatter_agent.run_as_node(state)
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
        recovery actions.
        """
        logger.info("Executing error_handler_node")
        if not state.error_messages:
            logger.warning("Error handler called but no error messages found")
            return state

        last_error = state.error_messages[-1]
        logger.error("Handling error: %s", last_error)

        try:
            error_recovery_service = ErrorRecoveryService()
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
            )
            updates = {}
            if recovery_action.strategy.value == "skip_item":
                logger.info(f"Skipping failed item: {state.current_item_id}")
                updates["current_item_id"] = None
            elif recovery_action.strategy.value == "immediate_retry":
                logger.info(f"Marking item for retry: {state.current_item_id}")
            elif recovery_action.strategy.value == "fallback_content":
                logger.info("Using fallback content")
                updates["fallback_content"] = recovery_action.fallback_content
            updates["error_messages"] = []
            return state.model_copy(update=updates)
        except Exception as exc:
            logger.error("Error handler itself failed: %s", exc)
            return state.model_copy(
                update={
                    "error_messages": state.error_messages
                    + [f"Error handler failed: {exc}"]
                }
            )

    async def supervisor_node(self, state: AgentState, config: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Supervisor node to route to the correct content generation subgraph
        or to the formatter node if all content is generated.
        """
        logger.info(f"Executing supervisor_node. Current section index: {state.current_section_index}")

        if state.error_messages:
            logger.warning("Errors detected in state, routing to error handler from supervisor.")
            return {"next_node": "error_handler"}

        if state.user_feedback and state.user_feedback.action == UserAction.REGENERATE:
            logger.info("User requested regeneration, routing to prepare_regeneration from supervisor.")
            return {"next_node": "prepare_regeneration"}

        if state.current_section_index >= len(WORKFLOW_SEQUENCE):
            logger.info("All content sections processed, routing to formatter.")
            return {"next_node": "formatter"}

        next_section_key = WORKFLOW_SEQUENCE[state.current_section_index]
        logger.info(f"Routing to {next_section_key}_subgraph.")
        return {"next_node": f"{next_section_key}_subgraph"}

    async def handle_feedback_node(self, state: AgentState, **kwargs) -> AgentState:
        """Handles user feedback to refine content."""
        logger.info(
            f"Executing handle_feedback_node for item: {getattr(state, 'current_item_id', 'unknown')}"
        )
        if state.user_feedback and state.user_feedback.action == UserAction.REGENERATE:
            logger.info(f"User feedback: Regenerate for item {state.current_item_id}")
            return state.model_copy(update={"user_feedback": None})
        elif state.user_feedback and state.user_feedback.action == UserAction.APPROVE:
            logger.info(f"User feedback: Approved for item {state.current_item_id}")
            return state.model_copy(update={"user_feedback": None})
        return state

    def _route_after_content_generation(self, state: Dict[str, Any]) -> str:
        """
        Router function for subgraphs after content generation and QA.
        Determines if content needs regeneration or if the subgraph should end.
        """
        agent_state = AgentState.model_validate(state)

        if agent_state.error_messages:
            logger.warning("Errors detected in state, routing to error handler from content generation subgraph.")
            return "error"

        if agent_state.user_feedback and agent_state.user_feedback.action == UserAction.REGENERATE:
            logger.info("User requested regeneration, looping back within subgraph.")
            return "regenerate"

        # If no regeneration requested and no errors, assume content is approved or acceptable
        logger.info("Content approved or no regeneration requested, continuing to next item/section.")
        return "continue"

    def _route_from_supervisor(self, state: Dict[str, Any]) -> str:
        """
        Router function for the main graph, called by the supervisor node.
        Determines the next step based on the 'next_node' field set by the supervisor.
        """
        agent_state = AgentState.model_validate(state)
        return agent_state.node_execution_metadata.get("next_node", "error_handler")


    def _build_key_qualifications_subgraph(self) -> StateGraph:
        """Builds the subgraph for Key Qualifications generation."""
        workflow = StateGraph(AgentState)
        workflow.add_node("generate", self.key_qualifications_writer_node)
        workflow.add_node("qa", self.qa_node)
        workflow.add_node("handle_feedback", self.handle_feedback_node)

        workflow.set_entry_point("generate")
        workflow.add_edge("generate", "qa")
        workflow.add_edge("qa", "handle_feedback")

        workflow.add_conditional_edges(
            "handle_feedback",
            self._route_after_content_generation,
            {
                "regenerate": "generate",
                "continue": END,
                "error": "error_handler",
            },
        )
        return workflow

    def _build_professional_experience_subgraph(self) -> StateGraph:
        """Builds the subgraph for Professional Experience generation."""
        workflow = StateGraph(AgentState)
        workflow.add_node("generate", self.professional_experience_writer_node)
        workflow.add_node("qa", self.qa_node)
        workflow.add_node("handle_feedback", self.handle_feedback_node)

        workflow.set_entry_point("generate")
        workflow.add_edge("generate", "qa")
        workflow.add_edge("qa", "handle_feedback")

        workflow.add_conditional_edges(
            "handle_feedback",
            self._route_after_content_generation,
            {
                "regenerate": "generate",
                "continue": END,
                "error": "error_handler",
            },
        )
        return workflow

    def _build_projects_subgraph(self) -> StateGraph:
        """Builds the subgraph for Projects generation."""
        workflow = StateGraph(AgentState)
        workflow.add_node("generate", self.projects_writer_node)
        workflow.add_node("qa", self.qa_node)
        workflow.add_node("handle_feedback", self.handle_feedback_node)

        workflow.set_entry_point("generate")
        workflow.add_edge("generate", "qa")
        workflow.add_edge("qa", "handle_feedback")

        workflow.add_conditional_edges(
            "handle_feedback",
            self._route_after_content_generation,
            {
                "regenerate": "generate",
                "continue": END,
                "error": "error_handler",
            },
        )
        return workflow

    def _build_executive_summary_subgraph(self) -> StateGraph:
        """Builds the subgraph for Executive Summary generation."""
        workflow = StateGraph(AgentState)
        workflow.add_node("generate", self.executive_summary_writer_node)
        workflow.add_node("qa", self.qa_node)
        workflow.add_node("handle_feedback", self.handle_feedback_node)

        workflow.set_entry_point("generate")
        workflow.add_edge("generate", "qa")
        workflow.add_edge("qa", "handle_feedback")

        workflow.add_conditional_edges(
            "handle_feedback",
            self._route_after_content_generation,
            {
                "regenerate": "generate",
                "continue": END,
                "error": "error_handler",
            },
        )
        return workflow

    def _build_graph(self) -> StateGraph:
        """Build and return the refactored CV workflow graph with supervisor and subgraphs."""
        workflow = StateGraph(AgentState)

        # Add main graph nodes
        workflow.add_node("jd_parser", self.jd_parser_node)
        workflow.add_node("cv_parser", self.cv_parser_node)
        workflow.add_node("research", self.research_node)
        workflow.add_node("cv_analyzer", self.cv_analyzer_node)
        workflow.add_node("supervisor", self.supervisor_node)
        workflow.add_node("formatter", self.formatter_node)
        workflow.add_node("error_handler", self.error_handler_node)

        # Add subgraphs as nodes
        workflow.add_node("key_qualifications_subgraph", self._build_key_qualifications_subgraph())
        workflow.add_node("professional_experience_subgraph", self._build_professional_experience_subgraph())
        workflow.add_node("projects_subgraph", self._build_projects_subgraph())
        workflow.add_node("executive_summary_subgraph", self._build_executive_summary_subgraph())

        # Set entry point
        workflow.set_entry_point("jd_parser")

        # Define main graph edges
        workflow.add_edge("jd_parser", "cv_parser")
        workflow.add_edge("cv_parser", "research")
        workflow.add_edge("research", "cv_analyzer")
        workflow.add_edge("cv_analyzer", "supervisor")

        # Conditional routing from supervisor
        workflow.add_conditional_edges(
            "supervisor",
            self._route_from_supervisor,
            {
                "key_qualifications_subgraph": "key_qualifications_subgraph",
                "professional_experience_subgraph": "professional_experience_subgraph",
                "projects_subgraph": "projects_subgraph",
                "executive_summary_subgraph": "executive_summary_subgraph",
                "formatter": "formatter",
                "error_handler": "error_handler",
            },
        )

        # After each subgraph, return to the supervisor to determine the next section
        workflow.add_edge("key_qualifications_subgraph", "supervisor")
        workflow.add_edge("professional_experience_subgraph", "supervisor")
        workflow.add_edge("projects_subgraph", "supervisor")
        workflow.add_edge("executive_summary_subgraph", "supervisor")

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