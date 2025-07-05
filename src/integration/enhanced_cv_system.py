"""Integration layer for enhanced CV system components.

This module provides a unified interface for coordinating and managing
all enhanced CV generation features including agents, orchestration, templates,
vector database, and workflows.
"""

from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Union

from src.config.logging_config import get_structured_logger
from src.config.settings import get_config
from src.error_handling.exceptions import VectorStoreError
from src.integration.config import EnhancedCVConfig, IntegrationMode
from src.integration.cv_agent_manager import CVAgentManager
from src.integration.cv_component_initializer import CVComponentInitializer
from src.integration.cv_system_monitor import CVSystemMonitor
from src.integration.cv_template_manager_facade import CVTemplateManagerFacade
from src.integration.cv_vector_store_facade import CVVectorStoreFacade
from src.integration.cv_workflow_executor import CVWorkflowExecutor, WorkflowDependencies
from src.models.cv_models import JobDescriptionData, StructuredCV
from src.models.workflow_models import ContentType, WorkflowType
from src.orchestration.state import AgentState
from src.services.error_recovery import ErrorRecoveryService


class EnhancedCVIntegration:
    """Main integration class for enhanced CV system components."""

    def __init__(
        self,
        config: Optional[EnhancedCVConfig] = None,
        session_id: Optional[str] = None,
    ):
        self.config = config or EnhancedCVConfig(mode=IntegrationMode.PRODUCTION)
        self.logger = get_structured_logger(__name__)
        self.settings = get_config()
        self.error_recovery = ErrorRecoveryService()
        self._session_id = session_id or "default"

        # Component instances
        self._initializer = CVComponentInitializer(self.config, self._session_id)
        self._initializer.initialize_components()

        self._template_facade = CVTemplateManagerFacade(
            self._initializer.template_manager
        )
        self._vector_store_facade = CVVectorStoreFacade(self._initializer.vector_db)
        self._orchestrator = self._initializer.orchestrator
        self._agent_manager = CVAgentManager(self._initializer.agents)
        self._performance_optimizer = self._initializer.performance_optimizer
        self._async_optimizer = self._initializer.async_optimizer
        self._intelligent_cache = self._initializer.intelligent_cache

        self._system_monitor: Optional[CVSystemMonitor] = None

        dependencies = WorkflowDependencies(
            orchestrator=self._orchestrator,
            error_recovery_service=self.error_recovery,
            performance_optimizer=self._performance_optimizer,
            async_optimizer=self._async_optimizer,
            intelligent_cache=self._intelligent_cache,
        )
        
        self._workflow_executor = CVWorkflowExecutor(
            dependencies=dependencies,
            session_id=self._session_id,
            enable_error_recovery=self.config.enable_error_recovery,
        )

        self._system_monitor = CVSystemMonitor(
            template_manager=self._initializer.template_manager,
            vector_db=self._initializer.vector_db,
            orchestrator=self._orchestrator,
            agents=self._initializer.agents,
        )

    # Template Management
    def get_template(
        self, template_id: str, category: str = None
    ) -> Optional[Dict[str, Any]]:
        """Get a content template."""
        return self._template_facade.get_template(template_id, category)

    def format_template(
        self, template_id: str, variables: Dict[str, Any], category: str = None
    ) -> Optional[str]:
        """Format a template with variables."""
        return self._template_facade.format_template(template_id, variables, category)

    def list_templates(self, category: str = None) -> List[str]:
        """List available templates."""
        return self._template_facade.list_templates(category)

    # Vector Database Operations
    async def store_content(
        self,
        content: str,
        content_type: ContentType,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Store content in vector database."""
        if not self._vector_store_facade:
            return None

        try:
            return await self._vector_store_facade.store_content(
                content, content_type, metadata
            )
        except (VectorStoreError, TypeError, ValueError) as e:
            self.logger.error(
                "Failed to store content",
                extra={"content_type": content_type.value, "error": str(e)},
            )
            return None

    async def search_content(
        self, query: str, content_type: Optional[ContentType] = None, limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Search for similar content."""
        if not self._vector_store_facade:
            return []
        return await self._vector_store_facade.search_content(
            query, content_type, limit
        )

    async def find_similar_content(
        self, content: str, content_type: Optional[ContentType] = None, limit: int = 3
    ) -> List[Dict[str, Any]]:
        """Find content similar to the provided content."""
        if not self._vector_store_facade:
            return []
        return await self._vector_store_facade.find_similar_content(
            content, content_type, limit
        )

    # Agent Operations
    def get_agent(self, agent_type: str):
        """Get an agent instance."""
        if not self._agent_manager:
            return None
        return self._agent_manager.get_agent(agent_type)

    def list_agents(self) -> List[str]:
        """List available agent types."""
        if not self._agent_manager:
            return []
        return self._agent_manager.list_agents()

    def get_orchestrator(self):
        """Get the orchestrator instance."""
        return self._orchestrator

    @property
    def orchestrator(self):
        """Get the orchestrator instance as a property."""
        return self._orchestrator

    def _get_performance_context(self):
        """Get performance optimization context."""
        if self._performance_optimizer:
            return self._performance_optimizer.optimized_execution(
                operation_name="workflow_execution",
                operation_type="workflow_execution",
                expected_duration=30.0,
            )
        else:
            # Return a no-op context manager if performance optimizer is not available

            return nullcontext()

    def _get_async_context(self):
        """Get async optimization context."""
        if self._async_optimizer:
            return self._async_optimizer.optimized_execution(
                operation_type="workflow_execution", operation_name="cv_workflow"
            )
        else:
            # Return a no-op context manager if async optimizer is not available

            return nullcontext()

    # Convenience workflow methods
    async def generate_basic_cv(
        self,
        personal_info: Dict[str, Any],
        experience: List[Dict[str, Any]],
        education: List[Dict[str, Any]],
        session_id: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate a basic CV."""
        # Create AgentState from the provided data

        structured_cv = StructuredCV.create_empty()
        # You might want to populate structured_cv with personal_info, experience, education here

        agent_state = AgentState(
            structured_cv=structured_cv,
            session_metadata={
                "personal_info": personal_info,
                "experience": experience,
                "education": education,
                **kwargs,
            },
        )

        return await self._workflow_executor.execute_workflow(
            WorkflowType.BASIC_CV_GENERATION,
            agent_state,
            session_id,
        )

    async def generate_job_tailored_cv(
        self,
        personal_info: Dict[str, Any],
        experience: List[Dict[str, Any]],
        job_description: Union[str, Dict[str, Any]],
        session_id: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate a job-tailored CV."""
        # Handle both string and dict job descriptions
        if isinstance(job_description, str):
            # Convert string to dict format expected by the workflow
            job_desc_dict = {
                "description": job_description.strip(),
                "raw_text": job_description.strip(),
            }
        else:
            job_desc_dict = job_description

        # Create AgentState from the provided data

        structured_cv = StructuredCV.create_empty()

        # Create job description data
        job_description_data = JobDescriptionData(
            raw_text=job_desc_dict.get("raw_text", ""),
            description=job_desc_dict.get("description", ""),
            requirements=job_desc_dict.get("requirements", []),
            skills=job_desc_dict.get("skills", []),
            company_info=job_desc_dict.get("company_info", {}),
        )

        agent_state = AgentState(
            structured_cv=structured_cv,
            job_description_data=job_description_data,
            session_metadata={
                "personal_info": personal_info,
                "experience": experience,
                **kwargs,
            },
        )

        return await self._workflow_executor.execute_workflow(
            WorkflowType.JOB_TAILORED_CV,
            agent_state,
            session_id,
        )

    async def optimize_cv(
        self, existing_cv: Dict[str, Any], session_id: Optional[str] = None, **kwargs
    ) -> Dict[str, Any]:
        """Optimize an existing CV."""
        # Create AgentState from existing CV data

        # Try to convert existing_cv to StructuredCV if it's not already
        if isinstance(existing_cv, dict):
            structured_cv = StructuredCV.model_validate(existing_cv)
        else:
            structured_cv = existing_cv

        agent_state = AgentState(
            structured_cv=structured_cv,
            session_metadata={"optimization_request": True, **kwargs},
        )

        return await self.execute_workflow(
            WorkflowType.CV_OPTIMIZATION,
            agent_state,
            session_id,
        )

    async def check_cv_quality(
        self, cv_content: Dict[str, Any], session_id: Optional[str] = None, **kwargs
    ) -> Dict[str, Any]:
        """Perform quality assurance on CV content."""
        # Create AgentState from CV content

        # Try to convert cv_content to StructuredCV if it's not already
        if isinstance(cv_content, dict):
            structured_cv = StructuredCV.model_validate(cv_content)
        else:
            structured_cv = cv_content

        agent_state = AgentState(
            structured_cv=structured_cv,
            session_metadata={"quality_check_request": True, **kwargs},
        )

        return await self.execute_workflow(
            WorkflowType.QUALITY_ASSURANCE,
            agent_state,
            session_id,
        )

    # Statistics and Monitoring
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        if not self._system_monitor:
            return {}
        return self._system_monitor.get_performance_stats()

    def reset_performance_stats(self):
        """Reset performance statistics."""
        if self._system_monitor:
            self._system_monitor.reset_performance_stats()

    # Health Check
    def health_check(self) -> Dict[str, Any]:
        """Perform a health check on all components."""
        if not self._system_monitor:
            return {"status": "unhealthy", "error": "System monitor not initialized"}
        return self._system_monitor.health_check()


class EnhancedCVIntegrationSingleton:
    """Singleton for managing EnhancedCVIntegration instance."""

    _instance = None

    @classmethod
    def get_instance(
        cls, config: Optional[EnhancedCVConfig] = None
    ) -> EnhancedCVIntegration:
        """Get enhanced CV system integration instance."""
        if cls._instance is None:
            cls._instance = EnhancedCVIntegration(config)
        return cls._instance

    @classmethod
    def reset_instance(cls):
        """Reset the enhanced CV system integration instance."""
        cls._instance = None


def get_enhanced_cv_integration(
    config: Optional[EnhancedCVConfig] = None,
) -> EnhancedCVIntegration:
    """Get enhanced CV system integration instance."""
    return EnhancedCVIntegrationSingleton.get_instance(config)


def reset_enhanced_cv_integration():
    """Reset the global enhanced CV system integration instance."""
    EnhancedCVIntegrationSingleton.reset_instance()


# Convenience functions
async def generate_cv(
    workflow_type: Union[WorkflowType, str],
    input_data: Union[AgentState, Dict[str, Any]],
    session_id: Optional[str] = None,
    config: Optional[EnhancedCVConfig] = None,
) -> Dict[str, Any]:
    """Generate a CV using the specified workflow."""
    integration = get_enhanced_cv_integration(config)

    # Convert dict input to AgentState if needed
    if isinstance(input_data, dict):
        # Create AgentState from dictionary
        agent_state = AgentState(
            structured_cv=input_data.get("structured_cv"),
            job_description_data=input_data.get("job_description_data"),
            error_messages=input_data.get("error_messages", []),
            user_action=input_data.get("user_action"),
            session_metadata=input_data.get("session_metadata", {}),
        )
    else:
        agent_state = input_data

    return await integration._workflow_executor.execute_workflow(
        workflow_type, agent_state, session_id
    )


def get_cv_templates(category: str = None) -> List[str]:
    """Get available CV templates."""
    integration = get_enhanced_cv_integration()
    if not integration._template_facade:
        return []
    return integration._template_facade.list_templates(category)


async def search_cv_examples(
    query: str, content_type: Optional[ContentType] = None, limit: int = 5
) -> List[Dict[str, Any]]:
    """Search for CV examples."""
    integration = get_enhanced_cv_integration()
    if not integration._vector_store_facade:
        return []
    return await integration._vector_store_facade.search_content(
        query, content_type, limit
    )


def get_system_health() -> Dict[str, Any]:
    """Get system health status."""
    integration = get_enhanced_cv_integration()
    if not integration._system_monitor:
        return {"status": "unhealthy", "error": "System monitor not initialized"}
    return integration._system_monitor.health_check()


def get_system_stats() -> Dict[str, Any]:
    """Get system performance statistics."""
    integration = get_enhanced_cv_integration()
    if not integration._system_monitor:
        return {}
    return integration._system_monitor.get_performance_stats()
