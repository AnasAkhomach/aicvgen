"""Module for initializing core CV system components."""

from dataclasses import asdict
from typing import Any, Dict, Optional

from src.config.logging_config import get_structured_logger
from src.config.settings import get_config
from src.core.async_optimizer import AsyncOptimizer
from src.core.caching_strategy import get_intelligent_cache_manager
from src.core.container import get_container
from src.core.performance_optimizer import PerformanceOptimizer
from src.integration.config import EnhancedCVConfig
from src.orchestration.graphs.main_graph import create_cv_workflow_graph_with_di

from src.templates.content_templates import ContentTemplateManager
from src.utils.security_utils import redact_sensitive_data


class CVComponentInitializer:
    """Initializes and manages core components of the CV generation system."""

    def __init__(self, config: EnhancedCVConfig, session_id: str):
        self.config = config
        self.logger = get_structured_logger(__name__)
        self.settings = get_config()
        self._session_id = session_id

        self.template_manager: Optional[ContentTemplateManager] = None
        self.vector_db: Optional[Any] = None  # Use Any for now, refine later
        self.orchestrator: Optional[Any] = None  # Workflow graph wrapper
        self.agents: Dict[str, Any] = {}
        self.performance_optimizer: Optional[PerformanceOptimizer] = None
        self.async_optimizer: Optional[AsyncOptimizer] = None
        self.intelligent_cache: Optional[Any] = None  # Use Any for now, refine later

    def initialize_components(self):
        """Initialize all enhanced CV system components."""
        try:
            redacted_config = redact_sensitive_data(asdict(self.config))
            self.logger.info(
                "Initializing CV system components",
                extra={"mode": self.config.mode.value, "config": redacted_config},
            )

            if self.config.enable_templates and not self.template_manager:
                self.template_manager = ContentTemplateManager()
                self.logger.info("Template manager initialized")

            if self.config.enable_vector_db:
                container = get_container()
                self.vector_db = container.vector_store_service()
                self.logger.info("Vector database initialized")

            if self.config.enable_specialized_agents:
                self._initialize_agents()
                self.logger.info("Specialized agents initialized")

            if self.config.enable_orchestration:
                container = get_container()
                self.orchestrator = create_cv_workflow_graph_with_di(
                    container, self._session_id
                )
                self.logger.info("Enhanced orchestration components initialized")

            if self.config.enable_performance_monitoring:
                self.performance_optimizer = PerformanceOptimizer()
                self.async_optimizer = AsyncOptimizer()
                self.intelligent_cache = get_intelligent_cache_manager()
                self.logger.info("Performance optimization components initialized")

            self.logger.info("CV system integration initialized successfully")

        except Exception as e:  # Catching broad exception for now, will refine later
            self.logger.error(
                "Failed to initialize CV system components",
                extra={
                    "error": str(e),
                    "config": redact_sensitive_data(asdict(self.config)),
                },
            )
            raise

    def _initialize_agents(self):
        """Initialize all specialized agents using DI container and session_id."""
        try:
            container = get_container()

            agent_map = {
                "cv_analysis": container.cv_analyzer_agent,
                "quality_assurance": container.quality_assurance_agent,
                "formatter": container.formatter_agent,
                "cleaning": container.cleaning_agent,
                "enhanced_content_writer": container.enhanced_content_writer_agent,
                "job_description_parser": container.job_description_parser_agent,
                "key_qualifications_writer": container.key_qualifications_writer_agent,
                "professional_experience_writer": container.professional_experience_writer_agent,
                "projects_writer": container.projects_writer_agent,
                "executive_summary_writer": container.executive_summary_writer_agent,
                "research": container.research_agent,
            }

            for agent_type, agent_provider in agent_map.items():
                try:
                    # Override session_id for each agent if they accept it
                    # This assumes agents are providers.Factory and accept session_id
                    # If not, this override will fail or be ignored.
                    agent_provider.override(session_id=self._session_id)
                    self.agents[agent_type] = agent_provider()
                except Exception as e:
                    self.logger.warning(
                        f"{agent_type} not available in container or failed to initialize: {e}"
                    )

            self.logger.info(
                "Agents initialized",
                extra={
                    "agent_count": len(self.agents),
                    "agent_types": list(self.agents.keys()),
                },
            )
        except Exception as e:  # Catching broad exception for now, will refine later
            self.logger.error("Failed to initialize agents", extra={"error": str(e)})
            raise
