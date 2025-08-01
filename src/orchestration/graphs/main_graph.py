"""Main workflow graph for CV generation orchestration.

This module defines the main workflow graph that orchestrates the entire CV generation process.
It integrates multiple specialized subgraphs and coordinates the flow between different stages
of CV analysis, content generation, and formatting."""

import logging
from typing import Dict, Callable
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph

from src.orchestration.state import GlobalState
from src.core.enums import WorkflowNodes

# Import node functions
from src.orchestration.nodes.parsing_nodes import (
    user_cv_parser_node,
    jd_parser_node,
    # research_node,  # Deferred to post-MVP
    # cv_analyzer_node,  # Deferred to post-MVP
)
from src.orchestration.nodes.content_nodes import (
    key_qualifications_writer_node,
    key_qualifications_updater_node,
    professional_experience_writer_node,
    professional_experience_updater_node,
    projects_writer_node,
    projects_updater_node,
    executive_summary_writer_node,
    executive_summary_updater_node,
    # qa_node,  # Deferred to post-MVP
)
from src.orchestration.nodes.workflow_nodes import (
    initialize_supervisor_node,
    supervisor_node,
    handle_feedback_node,
    mark_subgraph_completion_node,
    entry_router_node,
)
from src.orchestration.nodes.utility_nodes import (
    error_handler_node,
)  # formatter_node deferred to post-MVP
from src.orchestration.nodes.routing_nodes import (
    route_after_content_generation,
    route_from_supervisor,
    route_from_entry,
)


logger = logging.getLogger(__name__)


def create_node_functions(container) -> Dict[str, Callable]:
    """Create node functions with injected agents from DI container.

    Args:
        container: Dependency injection container

    Returns:
        Dict[str, Callable]: Dictionary of node functions with injected agents
    """

    # Create agent-bound node functions using runtime session_id extraction
    async def user_cv_parser_node_func(state: GlobalState, config=None) -> GlobalState:
        session_id = state.get("session_id")
        agent_factory = container.agent_factory()
        agent = agent_factory.create_user_cv_parser_agent(session_id=session_id)
        return await user_cv_parser_node(state, agent=agent)

    async def jd_parser_node_func(state: GlobalState, config=None) -> GlobalState:
        session_id = state.get("session_id")
        agent_factory = container.agent_factory()
        agent = agent_factory.create_job_description_parser_agent(session_id=session_id)
        return await jd_parser_node(state, agent=agent)

    # async def research_node_func(state: GlobalState, config=None) -> GlobalState:
    #     session_id = state.get('session_id')
    #     agent = container.research_agent(session_id=session_id)
    #     return await research_node(state, agent=agent)

    # async def cv_analyzer_node_func(state: GlobalState, config=None) -> GlobalState:
    #     session_id = state.get('session_id')
    #     agent = container.cv_analyzer_agent(session_id=session_id)
    #     return await cv_analyzer_node(state, agent=agent)

    async def key_qualifications_writer_node_func(
        state: GlobalState, config=None
    ) -> GlobalState:
        session_id = state.get("session_id")
        agent_factory = container.agent_factory()
        agent = agent_factory.create_key_qualifications_writer_agent(
            session_id=session_id
        )
        return await key_qualifications_writer_node(state, agent=agent)

    async def key_qualifications_updater_node_func(
        state: GlobalState, config=None
    ) -> GlobalState:
        session_id = state.get("session_id")
        agent_factory = container.agent_factory()
        agent = agent_factory.create_key_qualifications_updater_agent(
            session_id=session_id
        )
        return await key_qualifications_updater_node(state, agent=agent)

    async def professional_experience_writer_node_func(
        state: GlobalState, config=None
    ) -> GlobalState:
        session_id = state.get("session_id")
        agent_factory = container.agent_factory()
        agent = agent_factory.create_professional_experience_writer_agent(
            session_id=session_id
        )
        return await professional_experience_writer_node(state, agent=agent)

    async def professional_experience_updater_node_func(
        state: GlobalState, config=None
    ) -> GlobalState:
        session_id = state.get("session_id")
        agent_factory = container.agent_factory()
        agent = agent_factory.create_professional_experience_updater_agent(
            session_id=session_id
        )
        return await professional_experience_updater_node(state, agent=agent)

    async def projects_writer_node_func(state: GlobalState, config=None) -> GlobalState:
        session_id = state.get("session_id")
        agent_factory = container.agent_factory()
        agent = agent_factory.create_projects_writer_agent(session_id=session_id)
        return await projects_writer_node(state, agent=agent)

    async def projects_updater_node_func(
        state: GlobalState, config=None
    ) -> GlobalState:
        session_id = state.get("session_id")
        agent_factory = container.agent_factory()
        agent = agent_factory.create_projects_updater_agent(session_id=session_id)
        return await projects_updater_node(state, agent=agent)

    async def executive_summary_writer_node_func(
        state: GlobalState, config=None
    ) -> GlobalState:
        session_id = state.get("session_id")
        agent_factory = container.agent_factory()
        agent = agent_factory.create_executive_summary_writer_agent(
            session_id=session_id
        )
        return await executive_summary_writer_node(state, agent=agent)

    async def executive_summary_updater_node_func(
        state: GlobalState, config=None
    ) -> GlobalState:
        session_id = state.get("session_id")
        agent_factory = container.agent_factory()
        agent = agent_factory.create_executive_summary_updater_agent(
            session_id=session_id
        )
        return await executive_summary_updater_node(state, agent=agent)

    # async def qa_node_func(state: GlobalState, config=None) -> GlobalState:
    #     session_id = state.get('session_id')
    #     agent = container.quality_assurance_agent(session_id=session_id)
    #     return await qa_node(state, agent=agent)

    # async def formatter_node_func(state: GlobalState, config=None) -> GlobalState:
    #     session_id = state.get('session_id')
    #     agent = container.formatter_agent(session_id=session_id)
    #     return await formatter_node(state, agent=agent)

    return {
        "user_cv_parser_node": user_cv_parser_node_func,
        "jd_parser_node": jd_parser_node_func,
        # "research_node": research_node_func,  # Deferred to post-MVP
        # "cv_analyzer_node": cv_analyzer_node_func,  # Deferred to post-MVP
        "key_qualifications_writer_node": key_qualifications_writer_node_func,
        "key_qualifications_updater_node": key_qualifications_updater_node_func,
        "professional_experience_writer_node": professional_experience_writer_node_func,
        "professional_experience_updater_node": professional_experience_updater_node_func,
        "projects_writer_node": projects_writer_node_func,
        "projects_updater_node": projects_updater_node_func,
        "executive_summary_writer_node": executive_summary_writer_node_func,
        "executive_summary_updater_node": executive_summary_updater_node_func,
        # "qa_node": qa_node_func,  # Deferred to post-MVP
        # "formatter_node": formatter_node_func,  # Deferred to post-MVP
    }


def build_key_qualifications_subgraph(
    writer_node_func, updater_node_func
) -> StateGraph:
    """Build the key qualifications content generation subgraph.

    Args:
        writer_node_func: Pre-configured writer node function
        updater_node_func: Pre-configured updater node function

    Returns:
        Compiled subgraph for key qualifications
    """
    workflow = StateGraph(GlobalState)

    # Add nodes (simplified MVP workflow without QA)
    workflow.add_node(WorkflowNodes.GENERATE.value, writer_node_func)
    workflow.add_node(WorkflowNodes.REGENERATE.value, updater_node_func)
    workflow.add_node(WorkflowNodes.HANDLE_FEEDBACK.value, handle_feedback_node)
    workflow.add_node("MARK_COMPLETION", mark_subgraph_completion_node)

    # Set entry point
    workflow.set_entry_point(WorkflowNodes.GENERATE.value)

    # Add edges (simplified workflow: generate -> feedback -> completion)
    workflow.add_edge(WorkflowNodes.GENERATE.value, WorkflowNodes.HANDLE_FEEDBACK.value)
    workflow.add_edge(
        WorkflowNodes.REGENERATE.value, WorkflowNodes.HANDLE_FEEDBACK.value
    )

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

    return workflow.compile()


def build_professional_experience_subgraph(
    writer_node_func, updater_node_func
) -> StateGraph:
    """Build the professional experience content generation subgraph.

    Args:
        writer_node_func: Pre-configured writer node function
        updater_node_func: Pre-configured updater node function

    Returns:
        Compiled subgraph for professional experience
    """
    workflow = StateGraph(GlobalState)

    # Add nodes (simplified MVP workflow without QA)
    workflow.add_node(WorkflowNodes.GENERATE.value, writer_node_func)
    workflow.add_node(WorkflowNodes.REGENERATE.value, updater_node_func)
    workflow.add_node(WorkflowNodes.HANDLE_FEEDBACK.value, handle_feedback_node)
    workflow.add_node("MARK_COMPLETION", mark_subgraph_completion_node)

    # Set entry point
    workflow.set_entry_point(WorkflowNodes.GENERATE.value)

    # Add edges (simplified workflow: generate -> feedback -> completion)
    workflow.add_edge(WorkflowNodes.GENERATE.value, WorkflowNodes.HANDLE_FEEDBACK.value)
    workflow.add_edge(
        WorkflowNodes.REGENERATE.value, WorkflowNodes.HANDLE_FEEDBACK.value
    )

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

    return workflow.compile()


def build_projects_subgraph(writer_node_func, updater_node_func) -> StateGraph:
    """Build the projects content generation subgraph.

    Args:
        writer_node_func: Pre-configured writer node function
        updater_node_func: Pre-configured updater node function

    Returns:
        Compiled subgraph for projects
    """
    workflow = StateGraph(GlobalState)

    # Add nodes (simplified MVP workflow without QA)
    workflow.add_node(WorkflowNodes.GENERATE.value, writer_node_func)
    workflow.add_node(WorkflowNodes.REGENERATE.value, updater_node_func)
    workflow.add_node(WorkflowNodes.HANDLE_FEEDBACK.value, handle_feedback_node)
    workflow.add_node("MARK_COMPLETION", mark_subgraph_completion_node)

    # Set entry point
    workflow.set_entry_point(WorkflowNodes.GENERATE.value)

    # Add edges (simplified workflow: generate -> feedback -> completion)
    workflow.add_edge(WorkflowNodes.GENERATE.value, WorkflowNodes.HANDLE_FEEDBACK.value)
    workflow.add_edge(
        WorkflowNodes.REGENERATE.value, WorkflowNodes.HANDLE_FEEDBACK.value
    )

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

    return workflow.compile()


def build_executive_summary_subgraph(writer_node_func, updater_node_func) -> StateGraph:
    """Build the executive summary content generation subgraph.

    Args:
        writer_node_func: Pre-configured writer node function
        updater_node_func: Pre-configured updater node function

    Returns:
        Compiled subgraph for executive summary
    """
    workflow = StateGraph(GlobalState)

    # Add nodes (simplified MVP workflow without QA)
    workflow.add_node(WorkflowNodes.GENERATE.value, writer_node_func)
    workflow.add_node(WorkflowNodes.REGENERATE.value, updater_node_func)
    workflow.add_node(WorkflowNodes.HANDLE_FEEDBACK.value, handle_feedback_node)
    workflow.add_node("MARK_COMPLETION", mark_subgraph_completion_node)

    # Set entry point
    workflow.set_entry_point(WorkflowNodes.GENERATE.value)

    # Add edges (simplified workflow: generate -> feedback -> completion)
    workflow.add_edge(WorkflowNodes.GENERATE.value, WorkflowNodes.HANDLE_FEEDBACK.value)
    workflow.add_edge(
        WorkflowNodes.REGENERATE.value, WorkflowNodes.HANDLE_FEEDBACK.value
    )

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

    return workflow.compile()


# Removed create_node_functions_with_agents - functionality moved to NodeConfiguration class


def build_main_workflow_graph(node_functions: Dict[str, Callable]) -> StateGraph:
    """Build the main workflow graph from pre-configured node functions.

    Pure declarative assembly of pre-configured nodes into the complete workflow.
    This function handles only graph structure and routing logic.

    Args:
        node_functions: Dictionary of pre-configured node functions with injected agents

    Returns:
        StateGraph: The compiled workflow graph ready for execution
    """
    logger.info("Building main CV workflow graph")

    workflow = StateGraph(GlobalState)

    # Add main graph nodes (MVP workflow - deferred agents removed)
    workflow.add_node("USER_CV_PARSER", node_functions["user_cv_parser_node"])
    workflow.add_node(WorkflowNodes.JD_PARSER.value, node_functions["jd_parser_node"])
    # workflow.add_node(WorkflowNodes.RESEARCH.value, node_functions["research_node"])  # Deferred to post-MVP
    # workflow.add_node(WorkflowNodes.CV_ANALYZER.value, node_functions["cv_analyzer_node"])  # Deferred to post-MVP
    workflow.add_node("INITIALIZE_SUPERVISOR", initialize_supervisor_node)
    workflow.add_node(WorkflowNodes.SUPERVISOR.value, supervisor_node)
    # workflow.add_node(WorkflowNodes.FORMATTER.value, node_functions["formatter_node"])  # Deferred to post-MVP
    workflow.add_node(WorkflowNodes.ERROR_HANDLER.value, error_handler_node)

    # Build subgraphs with pre-configured node functions (MVP workflow without QA)
    key_qualifications_subgraph = build_key_qualifications_subgraph(
        node_functions["key_qualifications_writer_node"],
        node_functions["key_qualifications_updater_node"],
    )
    professional_experience_subgraph = build_professional_experience_subgraph(
        node_functions["professional_experience_writer_node"],
        node_functions["professional_experience_updater_node"],
    )
    projects_subgraph = build_projects_subgraph(
        node_functions["projects_writer_node"],
        node_functions["projects_updater_node"],
    )
    executive_summary_subgraph = build_executive_summary_subgraph(
        node_functions["executive_summary_writer_node"],
        node_functions["executive_summary_updater_node"],
    )

    # Add subgraphs as compiled nodes
    workflow.add_node(
        WorkflowNodes.KEY_QUALIFICATIONS_SUBGRAPH.value,
        key_qualifications_subgraph,
    )
    workflow.add_node(
        WorkflowNodes.PROFESSIONAL_EXPERIENCE_SUBGRAPH.value,
        professional_experience_subgraph,
    )
    workflow.add_node(
        WorkflowNodes.PROJECTS_SUBGRAPH.value,
        projects_subgraph,
    )
    workflow.add_node(
        WorkflowNodes.EXECUTIVE_SUMMARY_SUBGRAPH.value,
        executive_summary_subgraph,
    )

    # Add entry router
    workflow.add_node("ENTRY_ROUTER", entry_router_node)
    workflow.set_entry_point("ENTRY_ROUTER")

    # Define main graph edges
    workflow.add_conditional_edges(
        "ENTRY_ROUTER",
        route_from_entry,
        {
            "USER_CV_PARSER": "USER_CV_PARSER",
            WorkflowNodes.JD_PARSER.value: WorkflowNodes.JD_PARSER.value,
            WorkflowNodes.SUPERVISOR.value: "INITIALIZE_SUPERVISOR",
        },
    )

    # Simplified MVP processing chain (deferred agents removed)
    workflow.add_edge("USER_CV_PARSER", WorkflowNodes.JD_PARSER.value)
    workflow.add_edge(WorkflowNodes.JD_PARSER.value, "INITIALIZE_SUPERVISOR")
    # workflow.add_edge(WorkflowNodes.RESEARCH.value, WorkflowNodes.CV_ANALYZER.value)  # Deferred to post-MVP
    # workflow.add_edge(WorkflowNodes.CV_ANALYZER.value, "INITIALIZE_SUPERVISOR")  # Deferred to post-MVP
    workflow.add_edge("INITIALIZE_SUPERVISOR", WorkflowNodes.SUPERVISOR.value)

    # Conditional routing from supervisor (MVP workflow - FORMATTER deferred)
    workflow.add_conditional_edges(
        WorkflowNodes.SUPERVISOR.value,
        route_from_supervisor,
        {
            WorkflowNodes.KEY_QUALIFICATIONS_SUBGRAPH.value: WorkflowNodes.KEY_QUALIFICATIONS_SUBGRAPH.value,
            WorkflowNodes.PROFESSIONAL_EXPERIENCE_SUBGRAPH.value: WorkflowNodes.PROFESSIONAL_EXPERIENCE_SUBGRAPH.value,
            WorkflowNodes.PROJECTS_SUBGRAPH.value: WorkflowNodes.PROJECTS_SUBGRAPH.value,
            WorkflowNodes.EXECUTIVE_SUMMARY_SUBGRAPH.value: WorkflowNodes.EXECUTIVE_SUMMARY_SUBGRAPH.value,
            # WorkflowNodes.FORMATTER.value: WorkflowNodes.FORMATTER.value,  # Deferred to post-MVP
            WorkflowNodes.ERROR_HANDLER.value: WorkflowNodes.ERROR_HANDLER.value,
            END: END,  # Direct completion for MVP
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
        WorkflowNodes.PROJECTS_SUBGRAPH.value, WorkflowNodes.SUPERVISOR.value
    )
    workflow.add_edge(
        WorkflowNodes.EXECUTIVE_SUMMARY_SUBGRAPH.value,
        WorkflowNodes.SUPERVISOR.value,
    )

    # Terminal nodes (MVP workflow)
    # workflow.add_edge(WorkflowNodes.FORMATTER.value, END)  # Deferred to post-MVP
    workflow.add_edge(WorkflowNodes.ERROR_HANDLER.value, END)

    logger.info("Main CV workflow graph built successfully")
    return workflow.compile()


def create_cv_workflow_graph_with_di(container) -> CompiledStateGraph:
    """Create CV workflow graph with dependency injection.

    Orchestrates the creation of the compiled CV workflow graph using the new
    simplified node configuration approach. This function handles dependency
    injection at a higher level and returns a directly executable graph.
    Session ID is now extracted at runtime from the state.

    Args:
        container: Dependency injection container

    Returns:
        CompiledStateGraph: Compiled workflow graph ready for execution
    """
    # Create node functions directly from container
    node_functions = create_node_functions(container)

    # Build and compile the main workflow graph
    compiled_graph = build_main_workflow_graph(node_functions)

    return compiled_graph
