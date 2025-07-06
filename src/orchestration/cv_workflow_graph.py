"""LangGraph-based CV Generation Workflow.

This module defines the state machine workflow using LangGraph's StateGraph.
It implements granular, item-by-item processing with user feedback loops.
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional
from langgraph.graph import END, StateGraph

from src.agents.agent_base import AgentBase
from src.config.logging_config import get_structured_logger
from src.config.settings import get_config
from src.models.workflow_models import ContentType, UserAction
from src.orchestration.state import AgentState
from src.services.error_recovery import ErrorRecoveryService
from src.utils.node_validation import validate_node_output

logger = get_structured_logger(__name__)


class WorkflowNodes(Enum):
    """Enum for all workflow node names and routing outcomes to ensure type safety."""

    # Main graph nodes
    JD_PARSER = "jd_parser"
    CV_PARSER = "cv_parser"
    RESEARCH = "research"
    CV_ANALYZER = "cv_analyzer"
    SUPERVISOR = "supervisor"
    FORMATTER = "formatter"
    ERROR_HANDLER = "error_handler"

    # Subgraph nodes
    KEY_QUALIFICATIONS_SUBGRAPH = "key_qualifications_subgraph"
    PROFESSIONAL_EXPERIENCE_SUBGRAPH = "professional_experience_subgraph"
    PROJECTS_SUBGRAPH = "projects_subgraph"
    EXECUTIVE_SUMMARY_SUBGRAPH = "executive_summary_subgraph"

    # Workflow sequence section identifiers
    KEY_QUALIFICATIONS = "key_qualifications"
    PROFESSIONAL_EXPERIENCE = "professional_experience"
    PROJECT_EXPERIENCE = "project_experience"
    EXECUTIVE_SUMMARY = "executive_summary"

    # Subgraph internal nodes
    GENERATE = "generate"
    QA = "qa"
    HANDLE_FEEDBACK = "handle_feedback"

    # Routing outcomes
    REGENERATE = "regenerate"
    CONTINUE = "continue"
    ERROR = "error"
    PREPARE_REGENERATION = "prepare_regeneration"


# Define the workflow sequence for sections
WORKFLOW_SEQUENCE = [
    WorkflowNodes.KEY_QUALIFICATIONS.value,
    WorkflowNodes.PROFESSIONAL_EXPERIENCE.value,
    WorkflowNodes.PROJECT_EXPERIENCE.value,
    WorkflowNodes.EXECUTIVE_SUMMARY.value,
]


class CVWorkflowGraph:
    """Manages the CV generation workflow using a state graph."""

    def __init__(
        self,
        session_id: Optional[str] = None,
        job_description_parser_agent: Optional[AgentBase] = None,
        user_cv_parser_agent: Optional[AgentBase] = None,
        research_agent: Optional[AgentBase] = None,
        cv_analyzer_agent: Optional[AgentBase] = None,
        key_qualifications_writer_agent: Optional[AgentBase] = None,
        professional_experience_writer_agent: Optional[AgentBase] = None,
        projects_writer_agent: Optional[AgentBase] = None,
        executive_summary_writer_agent: Optional[AgentBase] = None,
        qa_agent: Optional[AgentBase] = None,
        formatter_agent: Optional[AgentBase] = None,
    ):
        self.session_id = session_id or str(uuid.uuid4())

        # Inject agent instances directly
        self.job_description_parser_agent = job_description_parser_agent
        self.user_cv_parser_agent = user_cv_parser_agent
        self.research_agent = research_agent
        self.cv_analyzer_agent = cv_analyzer_agent
        self.key_qualifications_writer_agent = key_qualifications_writer_agent
        self.professional_experience_writer_agent = professional_experience_writer_agent
        self.projects_writer_agent = projects_writer_agent
        self.executive_summary_writer_agent = executive_summary_writer_agent
        self.qa_agent = qa_agent
        self.formatter_agent = formatter_agent

        self.workflow = self._build_graph()
        self.app = self.workflow.compile()

        logger.info(f"CVWorkflowGraph initialized for session {self.session_id}")

    @validate_node_output
    async def jd_parser_node(
        self, state: AgentState, config: Optional[Dict] = None
    ) -> AgentState:
        """Execute parser node to process job description."""
        logger.info(f"Executing jd_parser_node")

        # Skip if job description data is already available
        if state.job_description_data:
            logger.info("Job description data already available, skipping parsing")
            return state

        try:
            if not self.job_description_parser_agent:
                raise RuntimeError("JobDescriptionParserAgent not injected")
            result = await self.job_description_parser_agent.run_as_node(state)
            if isinstance(result, dict):
                return state.model_copy(update=result)
            return result
        except (KeyError, AttributeError, RuntimeError) as exc:
            logger.error("JD Parser node failed: %s", exc)
            error_msg = f"JobDescriptionParserAgent failed: {str(exc)}"
            return state.model_copy(
                update={"error_messages": [*state.error_messages, error_msg]}
            )

    @validate_node_output
    async def cv_parser_node(
        self, state: AgentState, config: Optional[Dict] = None
    ) -> AgentState:
        """Execute parser node to process CV."""
        logger.info(f"Executing cv_parser_node")

        # Skip if structured CV data is already available
        if state.structured_cv:
            logger.info("Structured CV data already available, skipping parsing")
            return state

        try:
            if not self.user_cv_parser_agent:
                raise RuntimeError("UserCVParserAgent not injected")
            result = await self.user_cv_parser_agent.run_as_node(state)
            if isinstance(result, dict):
                return state.model_copy(update=result)
            return result
        except (KeyError, AttributeError, RuntimeError) as exc:
            logger.error("CV Parser node failed: %s", exc)
            error_msg = f"UserCVParserAgent failed: {str(exc)}"
            return state.model_copy(
                update={"error_messages": [*state.error_messages, error_msg]}
            )

    @validate_node_output
    async def research_node(self, state: AgentState, **kwargs) -> AgentState:
        """Execute research node."""
        logger.info("Executing research_node")

        # Skip if research findings are already available
        if state.research_findings:
            logger.info("Research findings already available, skipping research")
            return state

        if not self.research_agent:
            raise RuntimeError("ResearchAgent not injected")
        result = await self.research_agent.run_as_node(state)
        if isinstance(result, dict):
            return state.model_copy(update=result)
        return result

    @validate_node_output
    async def cv_analyzer_node(self, state: AgentState, **kwargs) -> AgentState:
        """Analyze the user's CV and store results in state.cv_analysis_results."""
        logger.info("Executing cv_analyzer_node")

        # Skip if CV analysis results are already available
        if state.cv_analysis_results:
            logger.info("CV analysis results already available, skipping analysis")
            return state

        if not self.cv_analyzer_agent:
            raise RuntimeError("CVAnalyzerAgent not injected")
        result = await self.cv_analyzer_agent.run_as_node(state)
        return result

    @validate_node_output
    async def key_qualifications_writer_node(
        self, state: AgentState, config: Optional[Dict] = None
    ) -> AgentState:
        """Execute key qualifications writer node."""
        logger.info(f"Executing key_qualifications_writer_node")
        if not self.key_qualifications_writer_agent:
            raise RuntimeError("KeyQualificationsWriterAgent not injected")
        result = await self.key_qualifications_writer_agent.run_as_node(state)
        if isinstance(result, dict):
            return state.model_copy(update=result)
        return result

    @validate_node_output
    async def professional_experience_writer_node(
        self, state: AgentState, config: Optional[Dict] = None
    ) -> AgentState:
        """Execute professional experience writer node."""
        logger.info(f"Executing professional_experience_writer_node")
        if not self.professional_experience_writer_agent:
            raise RuntimeError("ProfessionalExperienceWriterAgent not injected")
        result = await self.professional_experience_writer_agent.run_as_node(state)
        if isinstance(result, dict):
            return state.model_copy(update=result)
        return result

    @validate_node_output
    async def projects_writer_node(
        self, state: AgentState, config: Optional[Dict] = None
    ) -> AgentState:
        """Execute projects writer node."""
        logger.info(f"Executing projects_writer_node")
        if not self.projects_writer_agent:
            raise RuntimeError("ProjectsWriterAgent not injected")
        result = await self.projects_writer_agent.run_as_node(state)
        if isinstance(result, dict):
            return state.model_copy(update=result)
        return result

    @validate_node_output
    async def executive_summary_writer_node(
        self, state: AgentState, config: Optional[Dict] = None
    ) -> AgentState:
        """Execute executive summary writer node."""
        logger.info(f"Executing executive_summary_writer_node")
        if not self.executive_summary_writer_agent:
            raise RuntimeError("ExecutiveSummaryWriterAgent not injected")
        result = await self.executive_summary_writer_agent.run_as_node(state)
        if isinstance(result, dict):
            return state.model_copy(update=result)
        return result

    @validate_node_output
    async def qa_node(
        self, state: AgentState, config: Optional[Dict] = None
    ) -> AgentState:
        """Execute QA node for current item."""
        logger.info(f"Executing qa_node for item: {state.current_item_id}")
        if not self.qa_agent:
            raise RuntimeError("QualityAssuranceAgent not injected")
        result = await self.qa_agent.run_as_node(state)
        if isinstance(result, dict):
            return state.model_copy(update=result)
        return result

    @validate_node_output
    async def formatter_node(self, state: AgentState, **kwargs) -> AgentState:
        """Execute formatter node to generate PDF output."""
        logger.info("Executing formatter_node")
        if not self.formatter_agent:
            raise RuntimeError("FormatterAgent not injected")
        result = await self.formatter_agent.run_as_node(state)

        # CB-003 Fix: Handle AgentState result properly
        if isinstance(result, AgentState):
            # Return the result state directly to preserve all fields
            return result
        elif isinstance(result, dict):
            # Fallback for dictionary results
            updates = {}
            if "final_output_path" in result:
                updates["final_output_path"] = result["final_output_path"]
            if "error_messages" in result:
                updates["error_messages"] = [
                    *state.error_messages,
                    *result["error_messages"],
                ]
            return state.model_copy(update=updates)

        # Default fallback
        return state

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
            # Use current_content_type from state, fallback to QUALIFICATION if not set
            content_type = state.current_content_type or ContentType.QUALIFICATION
            recovery_action = await error_recovery_service.handle_error(
                Exception(last_error),
                state.current_item_id or "unknown",
                content_type,
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
                    "error_messages": [
                        *state.error_messages,
                        f"Error handler failed: {exc}",
                    ]
                }
            )

    async def supervisor_node(
        self, state: AgentState, config: Optional[Dict] = None
    ) -> AgentState:
        """
        Supervisor node to route to the correct content generation subgraph
        or to the formatter node if all content is generated.
        """
        logger.info(
            f"Executing supervisor_node. Current section index: {state.current_section_index}"
        )

        # Check if we're returning from a completed subgraph
        # If the last executed node was a subgraph, increment the section index
        last_node = state.node_execution_metadata.get("last_executed_node")
        current_index = state.current_section_index

        if last_node and last_node.endswith("_subgraph") and not state.user_feedback:
            # A subgraph completed successfully, move to next section
            current_index += 1
            logger.info(
                f"Subgraph {last_node} completed, incrementing section index to {current_index}"
            )

        if state.error_messages:
            logger.warning("Errors detected in state, routing to error handler.")
            next_node = WorkflowNodes.ERROR_HANDLER.value
        elif (
            state.user_feedback and state.user_feedback.action == UserAction.REGENERATE
        ):
            logger.info("User requested regeneration, staying in current section.")
            next_section_key = WORKFLOW_SEQUENCE[current_index]
            # Map section keys to subgraph node names
            section_to_subgraph = {
                WorkflowNodes.KEY_QUALIFICATIONS.value: WorkflowNodes.KEY_QUALIFICATIONS_SUBGRAPH.value,
                WorkflowNodes.PROFESSIONAL_EXPERIENCE.value: WorkflowNodes.PROFESSIONAL_EXPERIENCE_SUBGRAPH.value,
                WorkflowNodes.PROJECT_EXPERIENCE.value: WorkflowNodes.PROJECTS_SUBGRAPH.value,
                WorkflowNodes.EXECUTIVE_SUMMARY.value: WorkflowNodes.EXECUTIVE_SUMMARY_SUBGRAPH.value,
            }
            next_node = section_to_subgraph.get(
                next_section_key, WorkflowNodes.ERROR_HANDLER.value
            )
        elif current_index >= len(WORKFLOW_SEQUENCE):
            logger.info("All content sections processed, routing to formatter.")
            next_node = WorkflowNodes.FORMATTER.value
        else:
            next_section_key = WORKFLOW_SEQUENCE[current_index]
            logger.info(f"Routing to {next_section_key} subgraph.")
            # Map section keys to subgraph node names
            section_to_subgraph = {
                WorkflowNodes.KEY_QUALIFICATIONS.value: WorkflowNodes.KEY_QUALIFICATIONS_SUBGRAPH.value,
                WorkflowNodes.PROFESSIONAL_EXPERIENCE.value: WorkflowNodes.PROFESSIONAL_EXPERIENCE_SUBGRAPH.value,
                WorkflowNodes.PROJECT_EXPERIENCE.value: WorkflowNodes.PROJECTS_SUBGRAPH.value,
                WorkflowNodes.EXECUTIVE_SUMMARY.value: WorkflowNodes.EXECUTIVE_SUMMARY_SUBGRAPH.value,
            }
            next_node = section_to_subgraph.get(
                next_section_key, WorkflowNodes.ERROR_HANDLER.value
            )

        # Update node_execution_metadata with the next_node decision and mark current node as last executed
        updated_metadata = {
            **state.node_execution_metadata,
            "next_node": next_node,
            "last_executed_node": WorkflowNodes.SUPERVISOR.value,
        }

        return state.model_copy(
            update={
                "node_execution_metadata": updated_metadata,
                "current_section_index": current_index,
            }
        )

    async def handle_feedback_node(self, state: AgentState, **kwargs) -> AgentState:
        """Handles user feedback to refine content."""
        logger.info(
            f"Executing handle_feedback_node for item: {getattr(state, 'current_item_id', 'unknown')}"
        )

        # If no user feedback yet, set status to awaiting feedback and prepare UI data
        if not state.user_feedback:
            logger.info(
                "No user feedback found, setting workflow status to AWAITING_FEEDBACK"
            )

            # Prepare UI display data based on current section and content
            current_section = (
                WORKFLOW_SEQUENCE[state.current_section_index]
                if state.current_section_index < len(WORKFLOW_SEQUENCE)
                else "unknown"
            )

            ui_data = {
                "section": current_section,
                "item_id": state.current_item_id,
                "content_type": (
                    state.current_content_type.value
                    if state.current_content_type
                    else "unknown"
                ),
                "timestamp": datetime.now().isoformat(),
                "requires_feedback": True,
                "feedback_options": ["approve", "regenerate"],
            }

            # Add section-specific content to UI data
            if hasattr(state, "structured_cv") and state.structured_cv:
                # Find the relevant section content for display
                for section in state.structured_cv.sections:
                    if section.name == current_section:
                        ui_data["content_preview"] = {
                            "section_name": section.name,
                            "items_count": len(section.items),
                            "items": [
                                {
                                    "id": item.id,
                                    "content": (
                                        item.content[:200] + "..."
                                        if len(item.content) > 200
                                        else item.content
                                    ),
                                }
                                for item in section.items[:3]
                            ],  # Show first 3 items as preview
                        }
                        break

            return state.model_copy(
                update={
                    "workflow_status": "AWAITING_FEEDBACK",
                    "ui_display_data": ui_data,
                }
            )

        # Process existing user feedback
        if state.user_feedback.action == UserAction.REGENERATE:
            logger.info(f"User feedback: Regenerate for item {state.current_item_id}")
            return state.model_copy(
                update={
                    "user_feedback": None,
                    "workflow_status": "PROCESSING",
                    "ui_display_data": {},
                }
            )
        elif state.user_feedback.action == UserAction.APPROVE:
            logger.info(f"User feedback: Approved for item {state.current_item_id}")
            return state.model_copy(
                update={
                    "user_feedback": None,
                    "workflow_status": "PROCESSING",
                    "ui_display_data": {},
                }
            )

        # Fallback: return state copy
        return state.model_copy()

    async def mark_subgraph_completion_node(
        self, state: AgentState, **kwargs
    ) -> AgentState:
        """
        Node to mark that a subgraph has completed successfully.
        This updates the metadata to indicate which subgraph just finished.
        """
        current_section_key = WORKFLOW_SEQUENCE[state.current_section_index]
        subgraph_name = f"{current_section_key}_subgraph"

        logger.info(f"Marking completion of {subgraph_name}")

        # Update metadata to indicate this subgraph completed
        updated_metadata = {
            **state.node_execution_metadata,
            "last_executed_node": subgraph_name,
        }

        return state.model_copy(update={"node_execution_metadata": updated_metadata})

    def _route_after_content_generation(self, state: Dict[str, Any]) -> str:
        """
        Router function for subgraphs after content generation and QA.
        Determines if content needs regeneration or if the subgraph should end.
        """
        agent_state = AgentState.model_validate(state)

        if agent_state.error_messages:
            logger.warning(
                "Errors detected in state, routing to error handler from content generation subgraph."
            )
            return WorkflowNodes.ERROR.value

        if (
            agent_state.user_feedback
            and agent_state.user_feedback.action == UserAction.REGENERATE
        ):
            logger.info("User requested regeneration, looping back within subgraph.")
            return WorkflowNodes.REGENERATE.value

        # If no regeneration requested and no errors, route to completion marker
        logger.info(
            "Content approved or no regeneration requested, marking completion."
        )
        return "MARK_COMPLETION"

    def _route_from_supervisor(self, state: Dict[str, Any]) -> str:
        """
        Router function for the main graph, called by the supervisor node.
        Determines the next step based on the 'next_node' field set by the supervisor.
        """
        agent_state = AgentState.model_validate(state)
        return agent_state.node_execution_metadata.get(
            "next_node", WorkflowNodes.ERROR_HANDLER.value
        )

    async def _entry_router_node(
        self, state: AgentState, config: Optional[Dict] = None
    ) -> AgentState:
        """
        Entry router node that determines whether to start from initial parsing
        or skip directly to content generation based on existing data.
        """
        logger.info("Executing entry router to determine workflow starting point")

        # Check if initial parsing steps are already completed
        has_job_data = state.job_description_data is not None
        has_cv_data = state.structured_cv is not None
        has_research = state.research_findings is not None
        has_analysis = state.cv_analysis_results is not None

        if has_job_data and has_cv_data and has_research and has_analysis:
            logger.info("Initial parsing already completed, routing to supervisor")
            next_node = WorkflowNodes.SUPERVISOR.value
        else:
            logger.info("Starting from initial parsing steps")
            next_node = WorkflowNodes.JD_PARSER.value

        # Update metadata with routing decision
        updated_metadata = {**state.node_execution_metadata, "entry_route": next_node}

        return state.model_copy(update={"node_execution_metadata": updated_metadata})

    def _route_from_entry(self, state: Dict[str, Any]) -> str:
        """
        Router function for the entry point.
        Determines whether to start from JD_PARSER or SUPERVISOR.
        """
        agent_state = AgentState.model_validate(state)
        return agent_state.node_execution_metadata.get(
            "entry_route", WorkflowNodes.JD_PARSER.value
        )

    def _build_key_qualifications_subgraph(self) -> StateGraph:
        """Builds the subgraph for Key Qualifications generation."""
        workflow = StateGraph(AgentState)
        workflow.add_node(
            WorkflowNodes.GENERATE.value, self.key_qualifications_writer_node
        )
        workflow.add_node(WorkflowNodes.QA.value, self.qa_node)
        workflow.add_node(
            WorkflowNodes.HANDLE_FEEDBACK.value, self.handle_feedback_node
        )
        workflow.add_node("MARK_COMPLETION", self.mark_subgraph_completion_node)

        workflow.set_entry_point(WorkflowNodes.GENERATE.value)
        workflow.add_edge(WorkflowNodes.GENERATE.value, WorkflowNodes.QA.value)
        workflow.add_edge(WorkflowNodes.QA.value, WorkflowNodes.HANDLE_FEEDBACK.value)

        workflow.add_conditional_edges(
            WorkflowNodes.HANDLE_FEEDBACK.value,
            self._route_after_content_generation,
            {
                WorkflowNodes.REGENERATE.value: WorkflowNodes.GENERATE.value,
                "MARK_COMPLETION": "MARK_COMPLETION",
                WorkflowNodes.ERROR.value: END,
            },
        )
        workflow.add_edge("MARK_COMPLETION", END)
        return workflow

    def _build_professional_experience_subgraph(self) -> StateGraph:
        """Builds the subgraph for Professional Experience generation."""
        workflow = StateGraph(AgentState)
        workflow.add_node(
            WorkflowNodes.GENERATE.value, self.professional_experience_writer_node
        )
        workflow.add_node(WorkflowNodes.QA.value, self.qa_node)
        workflow.add_node(
            WorkflowNodes.HANDLE_FEEDBACK.value, self.handle_feedback_node
        )
        workflow.add_node("MARK_COMPLETION", self.mark_subgraph_completion_node)

        workflow.set_entry_point(WorkflowNodes.GENERATE.value)
        workflow.add_edge(WorkflowNodes.GENERATE.value, WorkflowNodes.QA.value)
        workflow.add_edge(WorkflowNodes.QA.value, WorkflowNodes.HANDLE_FEEDBACK.value)

        workflow.add_conditional_edges(
            WorkflowNodes.HANDLE_FEEDBACK.value,
            self._route_after_content_generation,
            {
                WorkflowNodes.REGENERATE.value: WorkflowNodes.GENERATE.value,
                "MARK_COMPLETION": "MARK_COMPLETION",
                WorkflowNodes.ERROR.value: END,
            },
        )
        workflow.add_edge("MARK_COMPLETION", END)
        return workflow

    def _build_projects_subgraph(self) -> StateGraph:
        """Builds the subgraph for Projects generation."""
        workflow = StateGraph(AgentState)
        workflow.add_node(WorkflowNodes.GENERATE.value, self.projects_writer_node)
        workflow.add_node(WorkflowNodes.QA.value, self.qa_node)
        workflow.add_node(
            WorkflowNodes.HANDLE_FEEDBACK.value, self.handle_feedback_node
        )
        workflow.add_node("MARK_COMPLETION", self.mark_subgraph_completion_node)

        workflow.set_entry_point(WorkflowNodes.GENERATE.value)
        workflow.add_edge(WorkflowNodes.GENERATE.value, WorkflowNodes.QA.value)
        workflow.add_edge(WorkflowNodes.QA.value, WorkflowNodes.HANDLE_FEEDBACK.value)

        workflow.add_conditional_edges(
            WorkflowNodes.HANDLE_FEEDBACK.value,
            self._route_after_content_generation,
            {
                WorkflowNodes.REGENERATE.value: WorkflowNodes.GENERATE.value,
                "MARK_COMPLETION": "MARK_COMPLETION",
                WorkflowNodes.ERROR.value: END,
            },
        )
        workflow.add_edge("MARK_COMPLETION", END)
        return workflow

    def _build_executive_summary_subgraph(self) -> StateGraph:
        """Builds the subgraph for Executive Summary generation."""
        workflow = StateGraph(AgentState)
        workflow.add_node(
            WorkflowNodes.GENERATE.value, self.executive_summary_writer_node
        )
        workflow.add_node(WorkflowNodes.QA.value, self.qa_node)
        workflow.add_node(
            WorkflowNodes.HANDLE_FEEDBACK.value, self.handle_feedback_node
        )
        workflow.add_node("MARK_COMPLETION", self.mark_subgraph_completion_node)

        workflow.set_entry_point(WorkflowNodes.GENERATE.value)
        workflow.add_edge(WorkflowNodes.GENERATE.value, WorkflowNodes.QA.value)
        workflow.add_edge(WorkflowNodes.QA.value, WorkflowNodes.HANDLE_FEEDBACK.value)

        workflow.add_conditional_edges(
            WorkflowNodes.HANDLE_FEEDBACK.value,
            self._route_after_content_generation,
            {
                WorkflowNodes.REGENERATE.value: WorkflowNodes.GENERATE.value,
                "MARK_COMPLETION": "MARK_COMPLETION",
                WorkflowNodes.ERROR.value: END,
            },
        )
        workflow.add_edge("MARK_COMPLETION", END)
        return workflow

    def _build_graph(self) -> StateGraph:
        """Build and return the refactored CV workflow graph with supervisor and subgraphs."""
        workflow = StateGraph(AgentState)

        # Add main graph nodes
        workflow.add_node(WorkflowNodes.JD_PARSER.value, self.jd_parser_node)
        workflow.add_node(WorkflowNodes.CV_PARSER.value, self.cv_parser_node)
        workflow.add_node(WorkflowNodes.RESEARCH.value, self.research_node)
        workflow.add_node(WorkflowNodes.CV_ANALYZER.value, self.cv_analyzer_node)
        workflow.add_node(WorkflowNodes.SUPERVISOR.value, self.supervisor_node)
        workflow.add_node(WorkflowNodes.FORMATTER.value, self.formatter_node)
        workflow.add_node(WorkflowNodes.ERROR_HANDLER.value, self.error_handler_node)

        # Add subgraphs as nodes (compile them first)
        workflow.add_node(
            WorkflowNodes.KEY_QUALIFICATIONS_SUBGRAPH.value,
            self._build_key_qualifications_subgraph().compile(),
        )
        workflow.add_node(
            WorkflowNodes.PROFESSIONAL_EXPERIENCE_SUBGRAPH.value,
            self._build_professional_experience_subgraph().compile(),
        )
        workflow.add_node(
            WorkflowNodes.PROJECTS_SUBGRAPH.value,
            self._build_projects_subgraph().compile(),
        )
        workflow.add_node(
            WorkflowNodes.EXECUTIVE_SUMMARY_SUBGRAPH.value,
            self._build_executive_summary_subgraph().compile(),
        )

        # Add a conditional entry point router
        workflow.add_node("ENTRY_ROUTER", self._entry_router_node)
        workflow.set_entry_point("ENTRY_ROUTER")

        # Define main graph edges
        workflow.add_conditional_edges(
            "ENTRY_ROUTER",
            self._route_from_entry,
            {
                WorkflowNodes.JD_PARSER.value: WorkflowNodes.JD_PARSER.value,
                WorkflowNodes.SUPERVISOR.value: WorkflowNodes.SUPERVISOR.value,
            },
        )
        workflow.add_edge(WorkflowNodes.JD_PARSER.value, WorkflowNodes.CV_PARSER.value)
        workflow.add_edge(WorkflowNodes.CV_PARSER.value, WorkflowNodes.RESEARCH.value)
        workflow.add_edge(WorkflowNodes.RESEARCH.value, WorkflowNodes.CV_ANALYZER.value)
        workflow.add_edge(
            WorkflowNodes.CV_ANALYZER.value, WorkflowNodes.SUPERVISOR.value
        )

        # Conditional routing from supervisor
        workflow.add_conditional_edges(
            WorkflowNodes.SUPERVISOR.value,
            self._route_from_supervisor,
            {
                WorkflowNodes.KEY_QUALIFICATIONS_SUBGRAPH.value: WorkflowNodes.KEY_QUALIFICATIONS_SUBGRAPH.value,
                WorkflowNodes.PROFESSIONAL_EXPERIENCE_SUBGRAPH.value: WorkflowNodes.PROFESSIONAL_EXPERIENCE_SUBGRAPH.value,
                WorkflowNodes.PROJECTS_SUBGRAPH.value: WorkflowNodes.PROJECTS_SUBGRAPH.value,
                WorkflowNodes.EXECUTIVE_SUMMARY_SUBGRAPH.value: WorkflowNodes.EXECUTIVE_SUMMARY_SUBGRAPH.value,
                WorkflowNodes.FORMATTER.value: WorkflowNodes.FORMATTER.value,
                WorkflowNodes.ERROR_HANDLER.value: WorkflowNodes.ERROR_HANDLER.value,
            },
        )

        # After each subgraph, return to the supervisor to determine the next section
        workflow.add_edge(
            WorkflowNodes.KEY_QUALIFICATIONS_SUBGRAPH.value,
            WorkflowNodes.SUPERVISOR.value,
        )
        workflow.add_edge(
            WorkflowNodes.PROFESSIONAL_EXPERIENCE_SUBGRAPH.value,
            WorkflowNodes.SUPERVISOR.value,
        )
        workflow.add_edge(
            WorkflowNodes.PROJECTS_SUBGRAPH.value, WorkflowNodes.SUPERVISOR.value
        )
        workflow.add_edge(
            WorkflowNodes.EXECUTIVE_SUMMARY_SUBGRAPH.value,
            WorkflowNodes.SUPERVISOR.value,
        )

        # Formatter ends the workflow
        workflow.add_edge(WorkflowNodes.FORMATTER.value, END)

        # Error handler terminates the workflow
        workflow.add_edge(WorkflowNodes.ERROR_HANDLER.value, END)

        return workflow

    async def invoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke the workflow with the given inputs."""
        return await self.app.ainvoke(
            inputs, config={"configurable": {"thread_id": self.session_id}}
        )

    async def trigger_workflow_step(self, state: AgentState) -> AgentState:
        """Trigger the next workflow step based on current state.

        This method allows resuming workflow execution from a paused state
        by streaming through workflow steps and saving state after each step.
        The method stops streaming when workflow_status becomes "AWAITING_FEEDBACK".

        Args:
            state: Current workflow state

        Returns:
            Updated state after executing workflow steps
        """
        logger.info(f"Triggering workflow step for session {self.session_id}")

        try:
            # Set workflow status to processing
            state = state.set_workflow_status("PROCESSING")
            state = state.set_ui_display_data({})

            # Convert state to dict for workflow invocation
            state_dict = state.model_dump()

            # Stream through workflow steps
            async for step in self.app.astream(
                state_dict, config={"configurable": {"thread_id": self.session_id}}
            ):
                # Update state from step result
                if isinstance(step, dict):
                    # Extract the actual state from the step result
                    # LangGraph astream yields {node_name: result} format
                    for node_name, node_result in step.items():
                        if isinstance(node_result, dict):
                            state = AgentState(**node_result)
                        elif hasattr(node_result, "model_dump"):
                            state = node_result
                        else:
                            # If it's already an AgentState, use it directly
                            state = node_result
                        break  # Take the first (and typically only) result

                # Save state to JSON file after each step
                self._save_state_to_file(state)

                # Check if we should pause for user feedback
                if state.workflow_status == "AWAITING_FEEDBACK":
                    logger.info(
                        f"Workflow paused for feedback in session {self.session_id}"
                    )
                    break

                # Check if workflow completed or errored
                if state.workflow_status in ["COMPLETED", "ERROR"]:
                    logger.info(
                        f"Workflow finished with status {state.workflow_status} in session {self.session_id}"
                    )
                    break

            return state

        except Exception as exc:
            logger.error(f"Error triggering workflow step: {exc}")
            error_msg = f"Workflow step execution failed: {str(exc)}"

            # Update state with error information
            state = state.model_copy(
                update={
                    "error_messages": [*state.error_messages, error_msg],
                    "workflow_status": "ERROR",
                }
            )

            # Save error state to file
            try:
                self._save_state_to_file(state)
            except Exception as save_error:
                logger.error(f"Failed to save error state: {save_error}")

            return state

    def _save_state_to_file(self, state: AgentState) -> None:
        """Save the current state to a JSON file.

        Args:
            state: The state to save
        """
        try:
            config = get_config()
            sessions_dir = config.paths.project_root / "instance" / "sessions"
            sessions_dir.mkdir(parents=True, exist_ok=True)

            session_file = sessions_dir / f"{self.session_id}.json"

            # Write state to file
            with open(session_file, "w", encoding="utf-8") as f:
                f.write(state.model_dump_json(indent=2))

            logger.debug(f"State saved to {session_file}")

        except Exception as exc:
            logger.error(f"Failed to save state to file: {exc}")
            raise
