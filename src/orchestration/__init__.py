"""Orchestration module for CV generation workflow."""

from .state import (
    AgentState,  # Backward compatibility alias
    GlobalState,
    KeyQualificationsState,
    ProfessionalExperienceState,
    ProjectsState,
    ExecutiveSummaryState,
    create_agent_state,
    create_global_state,
)
from .factories import (
    AgentNodeFactory,
    WriterNodeFactory,
)
from .node_helpers import (
    map_state_to_key_qualifications_input,
    map_state_to_professional_experience_input,
    map_state_to_projects_input,
    map_state_to_executive_summary_input,
    update_cv_with_key_qualifications_data,
    update_cv_with_professional_experience_data,
    update_cv_with_project_data,
    update_cv_with_executive_summary_data,
)

__all__ = [
    "AgentState",  # Backward compatibility
    "GlobalState",
    "KeyQualificationsState",
    "ProfessionalExperienceState",
    "ProjectsState",
    "ExecutiveSummaryState",
    "create_agent_state",
    "create_global_state",
    "AgentNodeFactory",
    "WriterNodeFactory",
    "map_state_to_key_qualifications_input",
    "map_state_to_professional_experience_input",
    "map_state_to_projects_input",
    "map_state_to_executive_summary_input",
    "update_cv_with_key_qualifications_data",
    "update_cv_with_professional_experience_data",
    "update_cv_with_project_data",
    "update_cv_with_executive_summary_data",
]
