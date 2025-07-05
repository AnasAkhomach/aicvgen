"""Factory module for creating agent instances."""

from typing import Any, Dict

from src.agents.cleaning_agent import CleaningAgent
from src.agents.cv_analyzer_agent import CVAnalyzerAgent
from src.agents.executive_summary_writer_agent import ExecutiveSummaryWriterAgent
from src.agents.formatter_agent import FormatterAgent
from src.agents.job_description_parser_agent import JobDescriptionParserAgent
from src.agents.key_qualifications_writer_agent import KeyQualificationsWriterAgent
from src.agents.professional_experience_writer_agent import (
    ProfessionalExperienceWriterAgent,
)
from src.agents.projects_writer_agent import ProjectsWriterAgent
from src.agents.quality_assurance_agent import QualityAssuranceAgent
from src.agents.research_agent import ResearchAgent
from src.agents.user_cv_parser_agent import UserCVParserAgent


class AgentFactory:
    """Factory for creating agent instances with dependency injection.

    This factory accepts specific service dependencies instead of the entire container
    to avoid circular dependencies.
    """

    def __init__(self, llm_service, template_manager, vector_store_service):
        """Initialize the factory with specific service dependencies.

        Args:
            llm_service: The LLM service for agent communication.
            template_manager: The template manager for content generation.
            vector_store_service: The vector store service for embeddings.
        """
        self._llm_service = llm_service
        self._template_manager = template_manager
        self._vector_store_service = vector_store_service

    def create_cv_analyzer_agent(self, session_id: str = "default") -> CVAnalyzerAgent:
        """Create a CV analyzer agent instance."""
        return CVAnalyzerAgent(llm_service=self._llm_service, session_id=session_id)

    def create_key_qualifications_writer_agent(
        self, settings: Dict[str, Any] = None, session_id: str = "default"
    ) -> KeyQualificationsWriterAgent:
        """Create a key qualifications writer agent instance."""
        return KeyQualificationsWriterAgent(
            llm_service=self._llm_service,
            template_manager=self._template_manager,
            settings=settings or {},
            session_id=session_id,
        )

    def create_professional_experience_writer_agent(
        self, settings: Dict[str, Any] = None, session_id: str = "default"
    ) -> ProfessionalExperienceWriterAgent:
        """Create a professional experience writer agent instance."""
        return ProfessionalExperienceWriterAgent(
            llm_service=self._llm_service,
            template_manager=self._template_manager,
            settings=settings or {},
            session_id=session_id,
        )

    def create_projects_writer_agent(
        self, settings: Dict[str, Any] = None, session_id: str = "default"
    ) -> ProjectsWriterAgent:
        """Create a projects writer agent instance."""
        return ProjectsWriterAgent(
            llm_service=self._llm_service,
            template_manager=self._template_manager,
            settings=settings or {},
            session_id=session_id,
        )

    def create_executive_summary_writer_agent(
        self, settings: Dict[str, Any] = None, session_id: str = "default"
    ) -> ExecutiveSummaryWriterAgent:
        """Create an executive summary writer agent instance."""
        return ExecutiveSummaryWriterAgent(
            llm_service=self._llm_service,
            template_manager=self._template_manager,
            settings=settings or {},
            session_id=session_id,
        )

    def create_cleaning_agent(
        self, settings: Dict[str, Any] = None, session_id: str = "default"
    ) -> CleaningAgent:
        """Create a cleaning agent instance."""
        return CleaningAgent(
            llm_service=self._llm_service,
            template_manager=self._template_manager,
            settings=settings or {},
            session_id=session_id,
        )

    def create_quality_assurance_agent(
        self, settings: Dict[str, Any] = None, session_id: str = "default"
    ) -> QualityAssuranceAgent:
        """Create a quality assurance agent instance."""
        return QualityAssuranceAgent(
            llm_service=self._llm_service,
            template_manager=self._template_manager,
            settings=settings or {},
            session_id=session_id,
        )

    def create_formatter_agent(
        self, settings: Dict[str, Any] = None, session_id: str = "default"
    ) -> FormatterAgent:
        """Create a formatter agent instance."""
        return FormatterAgent(
            template_manager=self._template_manager,
            settings=settings or {},
            session_id=session_id,
        )

    def create_research_agent(
        self, settings: Dict[str, Any] = None, session_id: str = "default"
    ) -> ResearchAgent:
        """Create a research agent instance."""
        return ResearchAgent(
            llm_service=self._llm_service,
            vector_store_service=self._vector_store_service,
            settings=settings or {},
            template_manager=self._template_manager,
            session_id=session_id,
        )

    def create_job_description_parser_agent(
        self, settings: Dict[str, Any] = None, session_id: str = "default"
    ) -> JobDescriptionParserAgent:
        """Create a job description parser agent instance."""
        return JobDescriptionParserAgent(
            llm_service=self._llm_service,
            template_manager=self._template_manager,
            settings=settings or {},
            session_id=session_id,
        )

    def create_user_cv_parser_agent(
        self, settings: Dict[str, Any] = None, session_id: str = "default"
    ) -> UserCVParserAgent:
        """Create a user CV parser agent instance."""
        return UserCVParserAgent(
            llm_service=self._llm_service,
            vector_store_service=self._vector_store_service,
            template_manager=self._template_manager,
            settings=settings or {},
            session_id=session_id,
        )
