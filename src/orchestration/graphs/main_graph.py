"""Main graph assembly for the CV workflow."""

import logging
from functools import partial
from typing import Dict, Any

from langgraph.graph import StateGraph, END

from src.orchestration.state import GlobalState
from src.core.enums import WorkflowNodes

# Import node functions
from src.orchestration.nodes.parsing_nodes import (
    jd_parser_node,
    research_node,
    cv_analyzer_node
)
from src.orchestration.nodes.content_nodes import (
    key_qualifications_writer_node,
    key_qualifications_updater_node,
    professional_experience_writer_node,
    projects_writer_node,
    executive_summary_writer_node,
    qa_node
)
from src.orchestration.nodes.workflow_nodes import (
    initialize_supervisor_node,
    supervisor_node,
    handle_feedback_node,
    mark_subgraph_completion_node,
    entry_router_node
)
from src.orchestration.nodes.utility_nodes import (
    formatter_node,
    error_handler_node
)
from src.orchestration.nodes.routing_nodes import (
    route_after_content_generation,
    route_from_supervisor,
    route_from_entry
)

logger = logging.getLogger(__name__)


def build_key_qualifications_subgraph() -> StateGraph:
    """Build the key qualifications content generation subgraph.
    
    Returns:
        Compiled subgraph for key qualifications
    """
    workflow = StateGraph(GlobalState)
    
    # Add nodes
    workflow.add_node(WorkflowNodes.GENERATE.value, key_qualifications_writer_node)
    workflow.add_node(WorkflowNodes.REGENERATE.value, key_qualifications_updater_node)
    workflow.add_node(WorkflowNodes.QA.value, qa_node)
    workflow.add_node(WorkflowNodes.HANDLE_FEEDBACK.value, handle_feedback_node)
    workflow.add_node("MARK_COMPLETION", mark_subgraph_completion_node)
    
    # Set entry point
    workflow.set_entry_point(WorkflowNodes.GENERATE.value)
    
    # Add edges
    workflow.add_edge(WorkflowNodes.GENERATE.value, WorkflowNodes.QA.value)
    workflow.add_edge(WorkflowNodes.REGENERATE.value, WorkflowNodes.QA.value)
    workflow.add_edge(WorkflowNodes.QA.value, WorkflowNodes.HANDLE_FEEDBACK.value)
    
    # Conditional routing after feedback
    workflow.add_conditional_edges(
        WorkflowNodes.HANDLE_FEEDBACK.value,
        route_after_content_generation,
        {
            WorkflowNodes.REGENERATE.value: WorkflowNodes.GENERATE.value,
            "MARK_COMPLETION": "MARK_COMPLETION",
            "AWAITING_FEEDBACK": END,
            WorkflowNodes.ERROR.value: END,
        },
    )
    workflow.add_edge("MARK_COMPLETION", END)
    
    return workflow


def build_professional_experience_subgraph() -> StateGraph:
    """Build the professional experience content generation subgraph.
    
    Returns:
        Compiled subgraph for professional experience
    """
    workflow = StateGraph(GlobalState)
    
    # Add nodes
    workflow.add_node(WorkflowNodes.GENERATE.value, professional_experience_writer_node)
    workflow.add_node(WorkflowNodes.REGENERATE.value, professional_experience_writer_node)  # Same function for regeneration
    workflow.add_node(WorkflowNodes.QA.value, qa_node)
    workflow.add_node(WorkflowNodes.HANDLE_FEEDBACK.value, handle_feedback_node)
    workflow.add_node("MARK_COMPLETION", mark_subgraph_completion_node)
    
    # Set entry point
    workflow.set_entry_point(WorkflowNodes.GENERATE.value)
    
    # Add edges
    workflow.add_edge(WorkflowNodes.GENERATE.value, WorkflowNodes.QA.value)
    workflow.add_edge(WorkflowNodes.REGENERATE.value, WorkflowNodes.QA.value)
    workflow.add_edge(WorkflowNodes.QA.value, WorkflowNodes.HANDLE_FEEDBACK.value)
    
    # Conditional routing after feedback
    workflow.add_conditional_edges(
        WorkflowNodes.HANDLE_FEEDBACK.value,
        route_after_content_generation,
        {
            WorkflowNodes.REGENERATE.value: WorkflowNodes.GENERATE.value,
            "MARK_COMPLETION": "MARK_COMPLETION",
            "AWAITING_FEEDBACK": END,
            WorkflowNodes.ERROR.value: END,
        },
    )
    workflow.add_edge("MARK_COMPLETION", END)
    
    return workflow


def build_projects_subgraph() -> StateGraph:
    """Build the projects content generation subgraph.
    
    Returns:
        Compiled subgraph for projects
    """
    workflow = StateGraph(GlobalState)
    
    # Add nodes
    workflow.add_node(WorkflowNodes.GENERATE.value, projects_writer_node)
    workflow.add_node(WorkflowNodes.REGENERATE.value, projects_writer_node)  # Same function for regeneration
    workflow.add_node(WorkflowNodes.QA.value, qa_node)
    workflow.add_node(WorkflowNodes.HANDLE_FEEDBACK.value, handle_feedback_node)
    workflow.add_node("MARK_COMPLETION", mark_subgraph_completion_node)
    
    # Set entry point
    workflow.set_entry_point(WorkflowNodes.GENERATE.value)
    
    # Add edges
    workflow.add_edge(WorkflowNodes.GENERATE.value, WorkflowNodes.QA.value)
    workflow.add_edge(WorkflowNodes.REGENERATE.value, WorkflowNodes.QA.value)
    workflow.add_edge(WorkflowNodes.QA.value, WorkflowNodes.HANDLE_FEEDBACK.value)
    
    # Conditional routing after feedback
    workflow.add_conditional_edges(
        WorkflowNodes.HANDLE_FEEDBACK.value,
        route_after_content_generation,
        {
            WorkflowNodes.REGENERATE.value: WorkflowNodes.GENERATE.value,
            "MARK_COMPLETION": "MARK_COMPLETION",
            "AWAITING_FEEDBACK": END,
            WorkflowNodes.ERROR.value: END,
        },
    )
    workflow.add_edge("MARK_COMPLETION", END)
    
    return workflow


def build_executive_summary_subgraph() -> StateGraph:
    """Build the executive summary content generation subgraph.
    
    Returns:
        Compiled subgraph for executive summary
    """
    workflow = StateGraph(GlobalState)
    
    # Add nodes
    workflow.add_node(WorkflowNodes.GENERATE.value, executive_summary_writer_node)
    workflow.add_node(WorkflowNodes.REGENERATE.value, executive_summary_writer_node)  # Same function for regeneration
    workflow.add_node(WorkflowNodes.QA.value, qa_node)
    workflow.add_node(WorkflowNodes.HANDLE_FEEDBACK.value, handle_feedback_node)
    workflow.add_node("MARK_COMPLETION", mark_subgraph_completion_node)
    
    # Set entry point
    workflow.set_entry_point(WorkflowNodes.GENERATE.value)
    
    # Add edges
    workflow.add_edge(WorkflowNodes.GENERATE.value, WorkflowNodes.QA.value)
    workflow.add_edge(WorkflowNodes.REGENERATE.value, WorkflowNodes.QA.value)
    workflow.add_edge(WorkflowNodes.QA.value, WorkflowNodes.HANDLE_FEEDBACK.value)
    
    # Conditional routing after feedback
    workflow.add_conditional_edges(
        WorkflowNodes.HANDLE_FEEDBACK.value,
        route_after_content_generation,
        {
            WorkflowNodes.REGENERATE.value: WorkflowNodes.GENERATE.value,
            "MARK_COMPLETION": "MARK_COMPLETION",
            "AWAITING_FEEDBACK": END,
            WorkflowNodes.ERROR.value: END,
        },
    )
    workflow.add_edge("MARK_COMPLETION", END)
    
    return workflow


def build_main_workflow_graph():
    """Build and return the main CV workflow graph.
    
    Returns:
        Compiled main workflow graph
    """
    logger.info("Building main CV workflow graph")
    
    workflow = StateGraph(GlobalState)
    
    # Add main graph nodes
    workflow.add_node(WorkflowNodes.JD_PARSER.value, jd_parser_node)
    workflow.add_node(WorkflowNodes.RESEARCH.value, research_node)
    workflow.add_node(WorkflowNodes.CV_ANALYZER.value, cv_analyzer_node)
    workflow.add_node("INITIALIZE_SUPERVISOR", initialize_supervisor_node)
    workflow.add_node(WorkflowNodes.SUPERVISOR.value, supervisor_node)
    workflow.add_node(WorkflowNodes.FORMATTER.value, formatter_node)
    workflow.add_node(WorkflowNodes.ERROR_HANDLER.value, error_handler_node)
    
    # Add subgraphs as compiled nodes
    workflow.add_node(
        WorkflowNodes.KEY_QUALIFICATIONS_SUBGRAPH.value,
        build_key_qualifications_subgraph().compile(),
    )
    workflow.add_node(
        WorkflowNodes.PROFESSIONAL_EXPERIENCE_SUBGRAPH.value,
        build_professional_experience_subgraph().compile(),
    )
    workflow.add_node(
        WorkflowNodes.PROJECTS_SUBGRAPH.value,
        build_projects_subgraph().compile(),
    )
    workflow.add_node(
        WorkflowNodes.EXECUTIVE_SUMMARY_SUBGRAPH.value,
        build_executive_summary_subgraph().compile(),
    )
    
    # Add entry router
    workflow.add_node("ENTRY_ROUTER", entry_router_node)
    workflow.set_entry_point("ENTRY_ROUTER")
    
    # Define main graph edges
    workflow.add_conditional_edges(
        "ENTRY_ROUTER",
        route_from_entry,
        {
            WorkflowNodes.JD_PARSER.value: WorkflowNodes.JD_PARSER.value,
            WorkflowNodes.SUPERVISOR.value: "INITIALIZE_SUPERVISOR",
        },
    )
    
    # Sequential processing chain
    workflow.add_edge(WorkflowNodes.JD_PARSER.value, WorkflowNodes.RESEARCH.value)
    workflow.add_edge(WorkflowNodes.RESEARCH.value, WorkflowNodes.CV_ANALYZER.value)
    workflow.add_edge(WorkflowNodes.CV_ANALYZER.value, "INITIALIZE_SUPERVISOR")
    workflow.add_edge("INITIALIZE_SUPERVISOR", WorkflowNodes.SUPERVISOR.value)
    
    # Conditional routing from supervisor
    workflow.add_conditional_edges(
        WorkflowNodes.SUPERVISOR.value,
        route_from_supervisor,
        {
            WorkflowNodes.KEY_QUALIFICATIONS_SUBGRAPH.value: WorkflowNodes.KEY_QUALIFICATIONS_SUBGRAPH.value,
            WorkflowNodes.PROFESSIONAL_EXPERIENCE_SUBGRAPH.value: WorkflowNodes.PROFESSIONAL_EXPERIENCE_SUBGRAPH.value,
            WorkflowNodes.PROJECTS_SUBGRAPH.value: WorkflowNodes.PROJECTS_SUBGRAPH.value,
            WorkflowNodes.EXECUTIVE_SUMMARY_SUBGRAPH.value: WorkflowNodes.EXECUTIVE_SUMMARY_SUBGRAPH.value,
            WorkflowNodes.FORMATTER.value: WorkflowNodes.FORMATTER.value,
            WorkflowNodes.ERROR_HANDLER.value: WorkflowNodes.ERROR_HANDLER.value,
        },
    )
    
    # Return to supervisor after each subgraph
    workflow.add_edge(
        WorkflowNodes.KEY_QUALIFICATIONS_SUBGRAPH.value,
        WorkflowNodes.SUPERVISOR.value,
    )
    workflow.add_edge(
        WorkflowNodes.PROFESSIONAL_EXPERIENCE_SUBGRAPH.value,
        WorkflowNodes.SUPERVISOR.value,
    )
    workflow.add_edge(
        WorkflowNodes.PROJECTS_SUBGRAPH.value, 
        WorkflowNodes.SUPERVISOR.value
    )
    workflow.add_edge(
        WorkflowNodes.EXECUTIVE_SUMMARY_SUBGRAPH.value,
        WorkflowNodes.SUPERVISOR.value,
    )
    
    # Terminal nodes
    workflow.add_edge(WorkflowNodes.FORMATTER.value, END)
    workflow.add_edge(WorkflowNodes.ERROR_HANDLER.value, END)
    
    logger.info("Main CV workflow graph built successfully")
    return workflow.compile()


def create_cv_workflow_graph_with_di(container, session_id: str):
    """Create and return a compiled CV workflow graph using DI container.
    
    Args:
        container: Dependency injection container
        session_id: Session identifier
        
    Returns:
        Compiled LangGraph workflow
    """
    # Get agents from DI container and bind them to node functions using partial
    jd_parser_node_with_agent = partial(
        jd_parser_node, 
        agent=container.job_description_parser_agent()
    )
    research_node_with_agent = partial(
        research_node, 
        agent=container.research_agent()
    )
    cv_analyzer_node_with_agent = partial(
        cv_analyzer_node, 
        agent=container.cv_analyzer_agent()
    )
    key_qualifications_writer_node_with_agent = partial(
        key_qualifications_writer_node, 
        agent=container.key_qualifications_writer_agent()
    )
    key_qualifications_updater_node_with_agent = partial(
        key_qualifications_updater_node, 
        agent=container.key_qualifications_writer_agent()
    )
    professional_experience_writer_node_with_agent = partial(
        professional_experience_writer_node, 
        agent=container.professional_experience_writer_agent()
    )
    projects_writer_node_with_agent = partial(
        projects_writer_node, 
        agent=container.projects_writer_agent()
    )
    executive_summary_writer_node_with_agent = partial(
        executive_summary_writer_node, 
        agent=container.executive_summary_writer_agent()
    )
    qa_node_with_agent = partial(
        qa_node, 
        agent=container.qa_agent()
    )
    formatter_node_with_agent = partial(
        formatter_node, 
        agent=container.formatter_agent()
    )
    
    # Build subgraphs with agent-enabled nodes
    key_qualifications_subgraph = _build_key_qualifications_subgraph(
        key_qualifications_writer_node_with_agent,
        key_qualifications_updater_node_with_agent,
        qa_node_with_agent
    )
    
    professional_experience_subgraph = _build_professional_experience_subgraph(
        professional_experience_writer_node_with_agent,
        qa_node_with_agent
    )
    
    projects_subgraph = _build_projects_subgraph(
        projects_writer_node_with_agent,
        qa_node_with_agent
    )
    
    executive_summary_subgraph = _build_executive_summary_subgraph(
        executive_summary_writer_node_with_agent,
        qa_node_with_agent
    )
    
    # Create and compile the main workflow graph
    main_workflow_graph = StateGraph(GlobalState)
    
    # Add all nodes
    main_workflow_graph.add_node("JD_PARSER", jd_parser_node_with_agent)
    main_workflow_graph.add_node("RESEARCH", research_node_with_agent)
    main_workflow_graph.add_node("CV_ANALYZER", cv_analyzer_node_with_agent)
    main_workflow_graph.add_node("INITIALIZE_SUPERVISOR", initialize_supervisor_node)
    main_workflow_graph.add_node("SUPERVISOR", supervisor_node)
    main_workflow_graph.add_node("FORMATTER", formatter_node_with_agent)
    main_workflow_graph.add_node("ERROR_HANDLER", error_handler_node)
    
    # Add subgraphs
    main_workflow_graph.add_node("KEY_QUALIFICATIONS", key_qualifications_subgraph)
    main_workflow_graph.add_node("PROFESSIONAL_EXPERIENCE", professional_experience_subgraph)
    main_workflow_graph.add_node("PROJECTS", projects_subgraph)
    main_workflow_graph.add_node("EXECUTIVE_SUMMARY", executive_summary_subgraph)
    
    # Set entry point
    main_workflow_graph.add_node("ENTRY_ROUTER", entry_router_node)
    main_workflow_graph.set_entry_point("ENTRY_ROUTER")
    
    # Add conditional edges
    main_workflow_graph.add_conditional_edges(
        "ENTRY_ROUTER",
        route_from_entry,
        {
            "JD_PARSER": "JD_PARSER",
            "SUPERVISOR": "SUPERVISOR"
        }
    )
    
    # Add edges between main nodes
    main_workflow_graph.add_edge("JD_PARSER", "RESEARCH")
    main_workflow_graph.add_edge("RESEARCH", "CV_ANALYZER")
    main_workflow_graph.add_edge("CV_ANALYZER", "INITIALIZE_SUPERVISOR")
    main_workflow_graph.add_edge("INITIALIZE_SUPERVISOR", "SUPERVISOR")
    
    # Add conditional routing from supervisor
    main_workflow_graph.add_conditional_edges(
        "SUPERVISOR",
        route_from_supervisor,
        {
            "KEY_QUALIFICATIONS": "KEY_QUALIFICATIONS",
            "PROFESSIONAL_EXPERIENCE": "PROFESSIONAL_EXPERIENCE",
            "PROJECTS": "PROJECTS",
            "EXECUTIVE_SUMMARY": "EXECUTIVE_SUMMARY",
            "FORMATTER": "FORMATTER",
            "ERROR_HANDLER": "ERROR_HANDLER",
            "END": END
        }
    )
    
    # Add edges from subgraphs back to supervisor
    main_workflow_graph.add_edge("KEY_QUALIFICATIONS", "SUPERVISOR")
    main_workflow_graph.add_edge("PROFESSIONAL_EXPERIENCE", "SUPERVISOR")
    main_workflow_graph.add_edge("PROJECTS", "SUPERVISOR")
    main_workflow_graph.add_edge("EXECUTIVE_SUMMARY", "SUPERVISOR")
    
    # Add final edges
    main_workflow_graph.add_edge("FORMATTER", END)
    main_workflow_graph.add_edge("ERROR_HANDLER", END)
    
    # Compile and return the graph
    compiled_graph = main_workflow_graph.compile()
    
    # Add methods that the workflow manager expects
    class WorkflowGraphWrapper:
        def __init__(self, graph):
            self.graph = graph
            self.session_id = session_id
        
        async def invoke(self, state):
            """Invoke the workflow graph with LangSmith tracing."""
            # Use session_id as thread_id for LangSmith tracing correlation
            config = {"configurable": {"thread_id": self.session_id}}
            return await self.graph.ainvoke(state, config=config)
        
        async def trigger_workflow_step(self, state):
            """Trigger a workflow step with pause mechanism and LangSmith tracing."""
            # For now, just invoke the graph with tracing configuration
            # This can be enhanced later with proper pause/resume logic
            # Use session_id as thread_id for LangSmith tracing correlation
            config = {"configurable": {"thread_id": self.session_id}}
            return await self.graph.ainvoke(state, config=config)
    
    return WorkflowGraphWrapper(compiled_graph)


def _build_key_qualifications_subgraph(writer_node, updater_node, qa_node):
    """Helper function to build key qualifications subgraph with custom nodes."""
    workflow = StateGraph(GlobalState)
    
    workflow.add_node(WorkflowNodes.GENERATE.value, writer_node)
    workflow.add_node(WorkflowNodes.REGENERATE.value, updater_node)
    workflow.add_node(WorkflowNodes.QA.value, qa_node)
    workflow.add_node(WorkflowNodes.HANDLE_FEEDBACK.value, handle_feedback_node)
    workflow.add_node("MARK_COMPLETION", mark_subgraph_completion_node)
    
    workflow.set_entry_point(WorkflowNodes.GENERATE.value)
    
    workflow.add_edge(WorkflowNodes.GENERATE.value, WorkflowNodes.QA.value)
    workflow.add_edge(WorkflowNodes.REGENERATE.value, WorkflowNodes.QA.value)
    workflow.add_edge(WorkflowNodes.QA.value, WorkflowNodes.HANDLE_FEEDBACK.value)
    
    workflow.add_conditional_edges(
        WorkflowNodes.HANDLE_FEEDBACK.value,
        route_after_content_generation,
        {
            WorkflowNodes.REGENERATE.value: WorkflowNodes.GENERATE.value,
            "MARK_COMPLETION": "MARK_COMPLETION",
            "AWAITING_FEEDBACK": END,
            WorkflowNodes.ERROR.value: END,
        },
    )
    workflow.add_edge("MARK_COMPLETION", END)
    
    return workflow.compile()


def _build_professional_experience_subgraph(writer_node, qa_node):
    """Helper function to build professional experience subgraph with custom nodes."""
    workflow = StateGraph(GlobalState)
    
    workflow.add_node(WorkflowNodes.GENERATE.value, writer_node)
    workflow.add_node(WorkflowNodes.REGENERATE.value, writer_node)
    workflow.add_node(WorkflowNodes.QA.value, qa_node)
    workflow.add_node(WorkflowNodes.HANDLE_FEEDBACK.value, handle_feedback_node)
    workflow.add_node("MARK_COMPLETION", mark_subgraph_completion_node)
    
    workflow.set_entry_point(WorkflowNodes.GENERATE.value)
    
    workflow.add_edge(WorkflowNodes.GENERATE.value, WorkflowNodes.QA.value)
    workflow.add_edge(WorkflowNodes.REGENERATE.value, WorkflowNodes.QA.value)
    workflow.add_edge(WorkflowNodes.QA.value, WorkflowNodes.HANDLE_FEEDBACK.value)
    
    workflow.add_conditional_edges(
        WorkflowNodes.HANDLE_FEEDBACK.value,
        route_after_content_generation,
        {
            WorkflowNodes.REGENERATE.value: WorkflowNodes.GENERATE.value,
            "MARK_COMPLETION": "MARK_COMPLETION",
            "AWAITING_FEEDBACK": END,
            WorkflowNodes.ERROR.value: END,
        },
    )
    workflow.add_edge("MARK_COMPLETION", END)
    
    return workflow.compile()


def _build_projects_subgraph(writer_node, qa_node):
    """Helper function to build projects subgraph with custom nodes."""
    workflow = StateGraph(GlobalState)
    
    workflow.add_node(WorkflowNodes.GENERATE.value, writer_node)
    workflow.add_node(WorkflowNodes.REGENERATE.value, writer_node)
    workflow.add_node(WorkflowNodes.QA.value, qa_node)
    workflow.add_node(WorkflowNodes.HANDLE_FEEDBACK.value, handle_feedback_node)
    workflow.add_node("MARK_COMPLETION", mark_subgraph_completion_node)
    
    workflow.set_entry_point(WorkflowNodes.GENERATE.value)
    
    workflow.add_edge(WorkflowNodes.GENERATE.value, WorkflowNodes.QA.value)
    workflow.add_edge(WorkflowNodes.REGENERATE.value, WorkflowNodes.QA.value)
    workflow.add_edge(WorkflowNodes.QA.value, WorkflowNodes.HANDLE_FEEDBACK.value)
    
    workflow.add_conditional_edges(
        WorkflowNodes.HANDLE_FEEDBACK.value,
        route_after_content_generation,
        {
            WorkflowNodes.REGENERATE.value: WorkflowNodes.GENERATE.value,
            "MARK_COMPLETION": "MARK_COMPLETION",
            "AWAITING_FEEDBACK": END,
            WorkflowNodes.ERROR.value: END,
        },
    )
    workflow.add_edge("MARK_COMPLETION", END)
    
    return workflow.compile()


def _build_executive_summary_subgraph(writer_node, qa_node):
    """Helper function to build executive summary subgraph with custom nodes."""
    workflow = StateGraph(GlobalState)
    
    workflow.add_node(WorkflowNodes.GENERATE.value, writer_node)
    workflow.add_node(WorkflowNodes.REGENERATE.value, writer_node)
    workflow.add_node(WorkflowNodes.QA.value, qa_node)
    workflow.add_node(WorkflowNodes.HANDLE_FEEDBACK.value, handle_feedback_node)
    workflow.add_node("MARK_COMPLETION", mark_subgraph_completion_node)
    
    workflow.set_entry_point(WorkflowNodes.GENERATE.value)
    
    workflow.add_edge(WorkflowNodes.GENERATE.value, WorkflowNodes.QA.value)
    workflow.add_edge(WorkflowNodes.REGENERATE.value, WorkflowNodes.QA.value)
    workflow.add_edge(WorkflowNodes.QA.value, WorkflowNodes.HANDLE_FEEDBACK.value)
    
    workflow.add_conditional_edges(
        WorkflowNodes.HANDLE_FEEDBACK.value,
        route_after_content_generation,
        {
            WorkflowNodes.REGENERATE.value: WorkflowNodes.GENERATE.value,
            "MARK_COMPLETION": "MARK_COMPLETION",
            "AWAITING_FEEDBACK": END,
            WorkflowNodes.ERROR.value: END,
        },
    )
    workflow.add_edge("MARK_COMPLETION", END)
    
    return workflow.compile()