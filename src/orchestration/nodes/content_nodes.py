"""Content generation nodes for the CV workflow.

This module contains nodes responsible for generating different sections
of the CV content using specialized agents and the AgentNodeFactory pattern.
"""

from typing import TYPE_CHECKING

from src.config.logging_config import get_logger
from src.orchestration.factories import AgentNodeFactory, WriterNodeFactory
from src.orchestration.node_helpers import (
    map_state_to_key_qualifications_input,
    map_state_to_professional_experience_input,
    map_state_to_projects_input,
    map_state_to_executive_summary_input,
    update_cv_with_key_qualifications_data,
    update_cv_with_professional_experience_data,
    update_cv_with_project_data,
    update_cv_with_executive_summary_data
)

if TYPE_CHECKING:
    from src.agents.executive_summary_writer_agent import ExecutiveSummaryWriterAgent
    from src.agents.key_qualifications_writer_agent import KeyQualificationsWriterAgent
    from src.agents.professional_experience_writer_agent import (
        ProfessionalExperienceWriterAgent,
    )
    from src.agents.projects_writer_agent import ProjectsWriterAgent
    from src.agents.quality_assurance_agent import QualityAssuranceAgent
    from src.orchestration.state import GlobalState

logger = get_logger(__name__)


async def key_qualifications_writer_node(state: 'GlobalState', *, agent: 'KeyQualificationsWriterAgent') -> 'GlobalState':
    """Generate key qualifications content using AgentNodeFactory.
    
    Args:
        state: Current workflow state
        agent: Key qualifications writer agent
        
    Returns:
        Updated state with key qualifications content
    """
    factory = WriterNodeFactory(
        agent=agent,
        input_mapper=map_state_to_key_qualifications_input,
        output_updater=update_cv_with_key_qualifications_data,
        node_name="Key Qualifications Writer",
        required_sections=["key qualifications"]
    )
    
    return await factory.execute_node(state)


# Note: key_qualifications_updater_node removed as it's now handled by the factory pattern
# The WriterNodeFactory handles both creation and updates through the same interface


async def professional_experience_writer_node(state: 'GlobalState', *, agent: 'ProfessionalExperienceWriterAgent') -> 'GlobalState':
    """Generate professional experience content using AgentNodeFactory.
    
    Args:
        state: Current workflow state
        agent: Professional experience writer agent
        
    Returns:
        Updated state with professional experience content
    """
    factory = WriterNodeFactory(
        agent=agent,
        input_mapper=map_state_to_professional_experience_input,
        output_updater=update_cv_with_professional_experience_data,
        node_name="Professional Experience Writer",
        required_sections=["professional experience"]
    )
    
    return await factory.execute_node(state)


async def projects_writer_node(state: 'GlobalState', *, agent: 'ProjectsWriterAgent') -> 'GlobalState':
    """Generate projects content using AgentNodeFactory.
    
    Args:
        state: Current workflow state
        agent: Projects writer agent
        
    Returns:
        Updated state with projects content
    """
    factory = WriterNodeFactory(
        agent=agent,
        input_mapper=map_state_to_projects_input,
        output_updater=update_cv_with_project_data,
        node_name="Projects Writer",
        required_sections=["projects"]
    )
    
    return await factory.execute_node(state)


async def executive_summary_writer_node(state: 'GlobalState', *, agent: 'ExecutiveSummaryWriterAgent') -> 'GlobalState':
    """Generate executive summary content using AgentNodeFactory.
    
    Args:
        state: Current workflow state
        agent: Executive summary writer agent
        
    Returns:
        Updated state with executive summary content
    """
    factory = WriterNodeFactory(
        agent=agent,
        input_mapper=map_state_to_executive_summary_input,
        output_updater=update_cv_with_executive_summary_data,
        node_name="Executive Summary Writer",
        required_sections=["executive summary"]
    )
    
    return await factory.execute_node(state)


async def qa_node(state: 'GlobalState', *, agent: 'QualityAssuranceAgent') -> 'GlobalState':
    """Quality assurance node for generated content using AgentNodeFactory.
    
    Args:
        state: Current workflow state
        agent: Quality assurance agent
        
    Returns:
        Updated state with QA results
    """
    # QA node uses a simpler AgentNodeFactory since it doesn't update CV content
    # but rather adds QA results to the state
    def map_state_to_qa_input(state: 'GlobalState') -> dict:
        """Map state to QA agent input."""
        from src.models.agent_input_models import extract_agent_inputs
        return extract_agent_inputs(state, "quality_assurance_agent")
    
    def update_state_with_qa_results(state: 'GlobalState', agent_output: dict) -> 'GlobalState':
        """Update state with QA results."""
        return {
            **state,
            "qa_results": agent_output.get("qa_results", {}),
            "last_executed_node": "QA"
        }
    
    factory = AgentNodeFactory(
        agent=agent,
        input_mapper=map_state_to_qa_input,
        output_updater=update_state_with_qa_results,
        node_name="Quality Assurance"
    )
    
    return await factory.execute_node(state)