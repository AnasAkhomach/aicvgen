"""Factory module for creating agent instances."""

from typing import Any, Callable, Dict, Optional

from src.agents.cleaning_agent import CleaningAgent
from src.agents.cv_analyzer_agent import CVAnalyzerAgent
from src.agents.executive_summary_writer_agent import ExecutiveSummaryWriterAgent
from src.agents.formatter_agent import FormatterAgent
from src.agents.job_description_parser_agent import JobDescriptionParserAgent
from src.agents.key_qualifications_writer_agent import KeyQualificationsWriterAgent
from src.agents.key_qualifications_updater_agent import KeyQualificationsUpdaterAgent
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

    def __init__(self, llm_service, template_manager, vector_store_service, session_id_provider: Callable[[], str] = None):
        """Initialize the factory with specific service dependencies.

        Args:
            llm_service: The LLM service for agent communication.
            template_manager: The template manager for content generation.
            vector_store_service: The vector store service for embeddings.
            session_id_provider: Function that returns the current session ID.
        """
        self._llm_service = llm_service
        self._template_manager = template_manager
        self._vector_store_service = vector_store_service
        self._session_id_provider = session_id_provider or (lambda: "default")

    def create_cv_analyzer_agent(self, session_id: Optional[str] = None) -> CVAnalyzerAgent:
        """Create a CV analyzer agent instance."""
        if session_id is None:
            session_id = self._session_id_provider()
        return CVAnalyzerAgent(llm_service=self._llm_service, session_id=session_id)

    def create_key_qualifications_writer_agent(
        self, settings: Dict[str, Any] = None, session_id: Optional[str] = None
    ) -> KeyQualificationsWriterAgent:
        """Create a key qualifications writer agent instance."""
        if session_id is None:
            session_id = self._session_id_provider()
        return KeyQualificationsWriterAgent(
            llm_service=self._llm_service,
            template_manager=self._template_manager,
            settings=settings or {},
            session_id=session_id,
        )

    def create_key_qualifications_updater_agent(
        self, session_id: Optional[str] = None
    ) -> KeyQualificationsUpdaterAgent:
        """Create a key qualifications updater agent instance."""
        if session_id is None:
            session_id = self._session_id_provider()
        return KeyQualificationsUpdaterAgent(
            name="KeyQualificationsUpdaterAgent"
        )

    def create_professional_experience_writer_agent(
        self, settings: Dict[str, Any] = None, session_id: Optional[str] = None
    ) -> ProfessionalExperienceWriterAgent:
        """Create a professional experience writer agent instance."""
        if session_id is None:
            session_id = self._session_id_provider()
        return ProfessionalExperienceWriterAgent(
            llm_service=self._llm_service,
            template_manager=self._template_manager,
            settings=settings or {},
            session_id=session_id,
        )

    def create_projects_writer_agent(
        self, settings: Dict[str, Any] = None, session_id: Optional[str] = None
    ) -> ProjectsWriterAgent:
        """Create a projects writer agent instance."""
        if session_id is None:
            session_id = self._session_id_provider()
        return ProjectsWriterAgent(
            llm_service=self._llm_service,
            template_manager=self._template_manager,
            settings=settings or {},
            session_id=session_id,
        )

    def create_executive_summary_writer_agent(
        self, settings: Dict[str, Any] = None, session_id: Optional[str] = None
    ) -> ExecutiveSummaryWriterAgent:
        """Create an executive summary writer agent instance."""
        if session_id is None:
            session_id = self._session_id_provider()
        return ExecutiveSummaryWriterAgent(
            llm_service=self._llm_service,
            template_manager=self._template_manager,
            settings=settings or {},
            session_id=session_id,
        )

    def create_cleaning_agent(
        self, settings: Dict[str, Any] = None, session_id: Optional[str] = None
    ) -> CleaningAgent:
        """Create a cleaning agent instance."""
        if session_id is None:
            session_id = self._session_id_provider()
        return CleaningAgent(
            llm_service=self._llm_service,
            template_manager=self._template_manager,
            settings=settings or {},
            session_id=session_id,
        )

    def create_quality_assurance_agent(
        self, settings: Dict[str, Any] = None, session_id: Optional[str] = None
    ) -> QualityAssuranceAgent:
        """Create a quality assurance agent instance."""
        if session_id is None:
            session_id = self._session_id_provider()
        return QualityAssuranceAgent(
            llm_service=self._llm_service,
            template_manager=self._template_manager,
            settings=settings or {},
            session_id=session_id,
        )

    def create_formatter_agent(
        self, settings: Dict[str, Any] = None, session_id: Optional[str] = None
    ) -> FormatterAgent:
        """Create a formatter agent instance."""
        if session_id is None:
            session_id = self._session_id_provider()
        return FormatterAgent(
            template_manager=self._template_manager,
            settings=settings or {},
            session_id=session_id,
        )

    def create_research_agent(
        self, settings: Dict[str, Any] = None, session_id: Optional[str] = None
    ) -> ResearchAgent:
        """Create a research agent instance."""
        if session_id is None:
            session_id = self._session_id_provider()
        return ResearchAgent(
            llm_service=self._llm_service,
            vector_store_service=self._vector_store_service,
            settings=settings or {},
            template_manager=self._template_manager,
            session_id=session_id,
        )

    def create_job_description_parser_agent(
        self, settings: Dict[str, Any] = None, session_id: Optional[str] = None
    ) -> JobDescriptionParserAgent:
        """Create a job description parser agent instance."""
        if session_id is None:
            session_id = self._session_id_provider()
        return JobDescriptionParserAgent(
            llm_service=self._llm_service,
            template_manager=self._template_manager,
            settings=settings or {},
            session_id=session_id,
        )

    def create_user_cv_parser_agent(
        self, settings: Dict[str, Any] = None, session_id: Optional[str] = None
    ) -> UserCVParserAgent:
        """Create a user CV parser agent instance."""
        if session_id is None:
            session_id = self._session_id_provider()
        return UserCVParserAgent(
            llm_service=self._llm_service,
            vector_store_service=self._vector_store_service,
            template_manager=self._template_manager,
            settings=settings or {},
            session_id=session_id,
        )
