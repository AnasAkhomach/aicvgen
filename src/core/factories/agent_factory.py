"""Factory module for creating agent instances."""

from typing import Any, Callable, Dict, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import BaseOutputParser, PydanticOutputParser
from langchain_core.language_models import BaseLanguageModel


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
from src.agents.professional_experience_updater_agent import ProfessionalExperienceUpdaterAgent
from src.agents.projects_writer_agent import ProjectsWriterAgent
from src.agents.projects_updater_agent import ProjectsUpdaterAgent
from src.agents.executive_summary_updater_agent import ExecutiveSummaryUpdaterAgent
from src.agents.quality_assurance_agent import QualityAssuranceAgent
from src.agents.research_agent import ResearchAgent
from src.agents.user_cv_parser_agent import UserCVParserAgent
from src.models.agent_output_models import KeyQualificationsLLMOutput, ProfessionalExperienceLLMOutput, ProjectLLMOutput, ExecutiveSummaryLLMOutput
from src.config.logging_config import get_logger


class AgentFactory:
    """Factory for creating agent instances with dependency injection.

    This factory accepts specific service dependencies instead of the entire container
    to avoid circular dependencies.
    """

    def __init__(self, llm_service, template_manager, vector_store_service, llm_cv_parser_service=None, session_id_provider: Callable[[], str] = None):
        """Initialize the factory with specific service dependencies.

        Args:
            llm_service: The LLM service for agent communication.
            template_manager: The template manager for content generation.
            vector_store_service: The vector store service for embeddings.
            llm_cv_parser_service: The LLM CV parser service for parsing operations.
            session_id_provider: Function that returns the current session ID.
        """
        self._llm_service = llm_service
        self._template_manager = template_manager
        self._vector_store_service = vector_store_service
        self._llm_cv_parser_service = llm_cv_parser_service
        self._session_id_provider = session_id_provider or (lambda: "default")

    def create_cv_analyzer_agent(self, settings: Dict[str, Any] = None, session_id: Optional[str] = None) -> CVAnalyzerAgent:
        """Create a CV analyzer agent instance."""
        if session_id is None:
            session_id = self._session_id_provider()
        return CVAnalyzerAgent(
            llm_service=self._llm_service,
            settings=settings or {},
            session_id=session_id
        )

    def create_key_qualifications_writer_agent(
        self, settings: Dict[str, Any] = None, session_id: Optional[str] = None
    ) -> KeyQualificationsWriterAgent:
        """Create a key qualifications writer agent instance."""
        if session_id is None:
            session_id = self._session_id_provider()
        
        # Get the LLM model from the service
        llm = self._get_llm_model()
        
        # Get the prompt template
        prompt = self._get_prompt_template("key_qualifications_prompt")
        
        # Get the parser
        parser = self._get_output_parser("KeyQualificationsLLMOutput")
        
        return KeyQualificationsWriterAgent(
            llm=llm,
            prompt=prompt,
            parser=parser,
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
            session_id=session_id,
            name="KeyQualificationsUpdaterAgent"
        )

    def create_professional_experience_updater_agent(
        self, session_id: Optional[str] = None
    ) -> ProfessionalExperienceUpdaterAgent:
        """Create a professional experience updater agent instance."""
        if session_id is None:
            session_id = self._session_id_provider()
        return ProfessionalExperienceUpdaterAgent(
            session_id=session_id,
            name="ProfessionalExperienceUpdaterAgent"
        )

    def create_projects_updater_agent(
        self, session_id: Optional[str] = None
    ) -> ProjectsUpdaterAgent:
        """Create a projects updater agent instance."""
        if session_id is None:
            session_id = self._session_id_provider()
        return ProjectsUpdaterAgent(
            name="ProjectsUpdaterAgent",
            session_id=session_id
        )

    def create_executive_summary_updater_agent(
        self, session_id: Optional[str] = None
    ) -> ExecutiveSummaryUpdaterAgent:
        """Create an executive summary updater agent instance."""
        if session_id is None:
            session_id = self._session_id_provider()
        return ExecutiveSummaryUpdaterAgent(
            session_id=session_id,
            name="ExecutiveSummaryUpdaterAgent"
        )

    def create_professional_experience_writer_agent(
        self, settings: Dict[str, Any] = None, session_id: Optional[str] = None
    ) -> ProfessionalExperienceWriterAgent:
        """Create a professional experience writer agent instance."""
        if session_id is None:
            session_id = self._session_id_provider()
        
        # Get the LLM model from the service
        llm = self._get_llm_model()
        
        # Get the prompt template
        prompt = self._get_prompt_template("professional_experience_prompt")
        
        # Get the parser
        parser = self._get_output_parser("ProfessionalExperienceLLMOutput")
        
        return ProfessionalExperienceWriterAgent(
            llm=llm,
            prompt=prompt,
            parser=parser,
            settings=settings or {},
            session_id=session_id,
        )

    def create_projects_writer_agent(
        self, settings: Dict[str, Any] = None, session_id: Optional[str] = None
    ) -> ProjectsWriterAgent:
        """Create a projects writer agent instance."""
        if session_id is None:
            session_id = self._session_id_provider()
        
        # Get the LLM model from the service
        llm = self._get_llm_model()
        
        # Get the prompt template
        prompt = self._get_prompt_template("projects_prompt")
        
        # Get the parser
        parser = self._get_output_parser("ProjectsLLMOutput")
        
        return ProjectsWriterAgent(
            llm=llm,
            prompt=prompt,
            parser=parser,
            settings=settings or {},
            session_id=session_id,
        )

    def create_executive_summary_writer_agent(
        self, settings: Dict[str, Any] = None, session_id: Optional[str] = None
    ) -> ExecutiveSummaryWriterAgent:
        """Create an executive summary writer agent instance."""
        if session_id is None:
            session_id = self._session_id_provider()
        
        # Get the LLM model from the service
        llm = self._get_llm_model()
        
        # Get the prompt template
        prompt = self._get_prompt_template("executive_summary_prompt")
        
        # Get the parser
        parser = self._get_output_parser("ExecutiveSummaryLLMOutput")
        
        return ExecutiveSummaryWriterAgent(
            llm=llm,
            prompt=prompt,
            parser=parser,
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
            llm_cv_parser_service=self._llm_cv_parser_service,
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

    def _get_llm_model(self) -> BaseLanguageModel:
        """Get the LLM model from the service."""
        return self._llm_service.get_llm()

    def _get_prompt_template(self, template_name: str) -> ChatPromptTemplate:
         """Get a prompt template by name and convert to ChatPromptTemplate."""
         # Map template names to content types
         content_type_map = {
             "key_qualifications_prompt": "KEY_QUALIFICATIONS",
             "professional_experience_prompt": "PROFESSIONAL_EXPERIENCE",
             "projects_prompt": "PROJECTS",
             "executive_summary_prompt": "EXECUTIVE_SUMMARY"
         }
         
         from src.models.workflow_models import ContentType
         content_type_str = content_type_map.get(template_name, "CV_ANALYSIS")
         content_type = ContentType[content_type_str]
         
         # Get the template from the manager
         template = self._template_manager.get_template_by_type(content_type)
         
         if template is None:
             raise ValueError(f"Template not found for {template_name}")
         
         # Convert to ChatPromptTemplate
         return ChatPromptTemplate.from_template(template.template)

    def _get_output_parser(self, output_model_name: str) -> BaseOutputParser:
        """Get an output parser for the specified model."""
        if output_model_name == "KeyQualificationsLLMOutput":
            return PydanticOutputParser(pydantic_object=KeyQualificationsLLMOutput)
        elif output_model_name == "ProfessionalExperienceLLMOutput":
            return PydanticOutputParser(pydantic_object=ProfessionalExperienceLLMOutput)
        elif output_model_name == "ProjectsLLMOutput":
             return PydanticOutputParser(pydantic_object=ProjectLLMOutput)
        elif output_model_name == "ExecutiveSummaryLLMOutput":
            return PydanticOutputParser(pydantic_object=ExecutiveSummaryLLMOutput)
        else:
            raise ValueError(f"Unknown output model: {output_model_name}")
