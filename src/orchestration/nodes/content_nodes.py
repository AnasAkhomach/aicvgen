"""Content generation nodes for the CV generation workflow.

This module contains nodes responsible for generating and updating different
sections of the CV using specialized agents.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.models.workflow_state import GlobalState
    from src.agents.key_qualifications_writer_agent import KeyQualificationsWriterAgent
    from src.agents.key_qualifications_updater_agent import (
        KeyQualificationsUpdaterAgent,
    )
    from src.agents.professional_experience_writer_agent import (
        ProfessionalExperienceWriterAgent,
    )
    from src.agents.professional_experience_updater_agent import (
        ProfessionalExperienceUpdaterAgent,
    )
    from src.agents.projects_writer_agent import ProjectsWriterAgent
    from src.agents.projects_updater_agent import ProjectsUpdaterAgent
    from src.agents.executive_summary_writer_agent import ExecutiveSummaryWriterAgent
    from src.agents.executive_summary_updater_agent import ExecutiveSummaryUpdaterAgent
    from src.agents.quality_assurance_agent import QualityAssuranceAgent


async def key_qualifications_writer_node(
    state: "GlobalState", *, agent: "KeyQualificationsWriterAgent"
) -> "GlobalState":
    """Generate key qualifications content.

    Args:
        state: Current workflow state
        agent: Key qualifications writer agent

    Returns:
        Updated state with key qualifications content
    """
    return await agent.run_as_node(state)


async def key_qualifications_updater_node(
    state: "GlobalState", *, agent: "KeyQualificationsUpdaterAgent"
) -> "GlobalState":
    """Update structured CV with generated key qualifications.

    Args:
        state: Current workflow state
        agent: Key qualifications updater agent

    Returns:
        Updated state with key qualifications integrated into structured CV
    """
    return await agent.run_as_node(state)


async def professional_experience_writer_node(
    state: "GlobalState", *, agent: "ProfessionalExperienceWriterAgent"
) -> "GlobalState":
    """Generate professional experience content.

    Args:
        state: Current workflow state
        agent: Professional experience writer agent

    Returns:
        Updated state with professional experience content
    """
    return await agent.run_as_node(state)


async def professional_experience_updater_node(
    state: "GlobalState", *, agent: "ProfessionalExperienceUpdaterAgent"
) -> "GlobalState":
    """Update structured CV with generated professional experience.

    Args:
        state: Current workflow state
        agent: Professional experience updater agent

    Returns:
        Updated state with professional experience integrated into structured CV
    """
    return await agent.run_as_node(state)


async def projects_writer_node(
    state: "GlobalState", *, agent: "ProjectsWriterAgent"
) -> "GlobalState":
    """Generate projects content.

    Args:
        state: Current workflow state
        agent: Projects writer agent

    Returns:
        Updated state with projects content
    """
    return await agent.run_as_node(state)


async def projects_updater_node(
    state: "GlobalState", *, agent: "ProjectsUpdaterAgent"
) -> "GlobalState":
    """Update structured CV with generated projects.

    Args:
        state: Current workflow state
        agent: Projects updater agent

    Returns:
        Updated state with projects integrated into structured CV
    """
    return await agent.run_as_node(state)


async def executive_summary_writer_node(
    state: "GlobalState", *, agent: "ExecutiveSummaryWriterAgent"
) -> "GlobalState":
    """Generate executive summary content.

    Args:
        state: Current workflow state
        agent: Executive summary writer agent

    Returns:
        Updated state with executive summary content
    """
    return await agent.run_as_node(state)


async def executive_summary_updater_node(
    state: "GlobalState", *, agent: "ExecutiveSummaryUpdaterAgent"
) -> "GlobalState":
    """Update structured CV with generated executive summary.

    Args:
        state: Current workflow state
        agent: Executive summary updater agent

    Returns:
        Updated state with executive summary integrated into structured CV
    """
    return await agent.run_as_node(state)


async def qa_node(
    state: "GlobalState", *, agent: "QualityAssuranceAgent"
) -> "GlobalState":
    """Quality assurance node.

    Args:
        state: Current workflow state
        agent: Quality assurance agent

    Returns:
        Updated state with QA results
    """
    return await agent.run_as_node(state)
