"""Factory for creating agent input validators.

This module provides a factory pattern for creating agent input validators,
replacing complex validation logic with a clean, extensible approach.
"""

import logging
from typing import TYPE_CHECKING, Any, Dict, Type

from pydantic import BaseModel, ValidationError

from src.models.agent_input_models import CleaningAgentInput
from src.models.agent_input_models import (
    CVAnalyzerAgentInput,
    FormatterAgentInput,
    QualityAssuranceAgentInput,
    ResearchAgentInput,
    UserCVParserAgentInput,
    JobDescriptionParserAgentInput,
    ExecutiveSummaryWriterAgentInput,
    ProfessionalExperienceWriterAgentInput,
    KeyQualificationsWriterAgentInput,
    KeyQualificationsUpdaterAgentInput,
    ProjectsWriterAgentInput,
)
from src.models.validation_schemas import (
    ContentWriterAgentInput,
    ParserAgentInput,
)

if TYPE_CHECKING:
    from src.orchestration.state import AgentState

logger = logging.getLogger(__name__)


class ValidatorFactory:
    """Factory for creating agent input validators."""

    # Registry of agent types to their corresponding input models
    _VALIDATOR_REGISTRY: Dict[str, Type[BaseModel]] = {
        "parser": ParserAgentInput,
        "content_writer": ContentWriterAgentInput,
        "research": ResearchAgentInput,
        "qa": QualityAssuranceAgentInput,
        "formatter": FormatterAgentInput,
        "cv_analyzer": CVAnalyzerAgentInput,
        "cleaning": CleaningAgentInput,
        "ResearchAgent": ResearchAgentInput,
        "CVAnalyzerAgent": CVAnalyzerAgentInput,
        "QualityAssuranceAgent": QualityAssuranceAgentInput,
        "FormatterAgent": FormatterAgentInput,
        "UserCVParserAgent": UserCVParserAgentInput,
        "JobDescriptionParserAgent": JobDescriptionParserAgentInput,
        "ExecutiveSummaryWriter": ExecutiveSummaryWriterAgentInput,
        "ProfessionalExperienceWriter": ProfessionalExperienceWriterAgentInput,
        "KeyQualificationsWriter": KeyQualificationsWriterAgentInput,
        "KeyQualificationsUpdaterAgent": KeyQualificationsUpdaterAgentInput,
        "ProjectsWriter": ProjectsWriterAgentInput,
        "CleaningAgent": CleaningAgentInput,
    }

    @classmethod
    def validate_agent_input(cls, agent_type: str, state: "AgentState") -> Any:
        """Validate agent input data against the appropriate Pydantic model.
        
        Args:
            agent_type: The type of agent to validate input for
            state: The agent state containing input data
            
        Returns:
            Validated input model instance
            
        Raises:
            ValueError: If validation fails or agent type is unknown
        """
        validator_class = cls._VALIDATOR_REGISTRY.get(agent_type)
        if not validator_class:
            logger.warning(f"No validator found for agent type: {agent_type}")
            return state
            
        try:
            return cls._create_validator_instance(validator_class, agent_type, state)
        except ValidationError as e:
            logger.error("Validation error", agent_type=agent_type, error=str(e))
            raise ValueError(f"Input validation failed for {agent_type}: {e}") from e
    
    @classmethod
    def _create_validator_instance(cls, validator_class: Type[BaseModel], agent_type: str, state: "AgentState") -> BaseModel:
        """Create a validator instance based on the agent type.
        
        Args:
            validator_class: The Pydantic model class to instantiate
            agent_type: The type of agent
            state: The agent state containing input data
            
        Returns:
            Instantiated validator model
        """
        if agent_type == "parser":
            return validator_class(
                cv_text=state.cv_text,
                job_description_data=state.job_description_data,
            )
        elif agent_type == "content_writer":
            return validator_class(
                structured_cv=state.structured_cv,
                research_findings=getattr(state, "research_findings", None),
                current_item_id=state.current_item_id,
            )
        elif agent_type == "research":
            return validator_class(
                job_description_data=state.job_description_data,
                structured_cv=state.structured_cv,
            )
        elif agent_type == "qa":
            return validator_class(
                structured_cv=state.structured_cv,
                current_item_id=state.current_item_id,
            )
        elif agent_type == "formatter":
            return validator_class(
                structured_cv=state.structured_cv,
                job_description_data=getattr(state, "job_description_data", None),
            )
        elif agent_type == "cv_analyzer":
            return validator_class(
                cv_text=state.cv_text,
                job_description_data=state.job_description_data,
            )
        elif agent_type == "cleaning":
            return validator_class(
                structured_cv=state.structured_cv,
            )
        else:
            # This should not happen due to registry check, but included for safety
            raise ValueError(f"Unknown agent type: {agent_type}")
    
    @classmethod
    def register_validator(cls, agent_type: str, validator_class: Type[BaseModel]) -> None:
        """Register a new validator for an agent type.
        
        Args:
            agent_type: The agent type identifier
            validator_class: The Pydantic model class for validation
        """
        cls._VALIDATOR_REGISTRY[agent_type] = validator_class
        logger.info(f"Registered validator for agent type: {agent_type}")
    
    @classmethod
    def get_supported_agent_types(cls) -> list[str]:
        """Get list of supported agent types.
        
        Returns:
            List of supported agent type identifiers
        """
        return list(cls._VALIDATOR_REGISTRY.keys())