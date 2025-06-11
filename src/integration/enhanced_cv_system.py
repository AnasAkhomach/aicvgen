"""Integration layer for enhanced CV system components.

This module provides a unified interface for coordinating and managing
all enhanced CV generation features including agents, orchestration, templates,
vector database, and workflows.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum

from ..models.data_models import ContentType, ProcessingStatus
from ..config.logging_config import get_structured_logger
from ..config.settings import get_config
from ..services.error_recovery import get_error_recovery_service, RecoveryStrategy
from ..utils.security_utils import redact_sensitive_data

# Enhanced CV system imports
from ..agents.enhanced_content_writer import EnhancedContentWriterAgent
from ..agents.specialized_agents import (
    CVAnalysisAgent, ContentOptimizationAgent, QualityAssuranceAgent,
    get_agent, list_available_agents
)
from ..templates.content_templates import (
    get_template_manager, ContentTemplateManager
)
from ..services.vector_db import get_enhanced_vector_db
from ..orchestration.agent_orchestrator import (
    get_agent_orchestrator, AgentOrchestrator
)
from ..orchestration.workflow_definitions import (
    get_workflow_builder, WorkflowBuilder, WorkflowType,
    execute_basic_cv_generation, execute_job_tailored_cv,
    execute_cv_optimization, execute_quality_assurance,
    execute_comprehensive_cv, execute_quick_update
)


class IntegrationMode(Enum):
    """Integration modes for different use cases."""
    DEVELOPMENT = "development"
    PRODUCTION = "production"
    TESTING = "testing"
    DEMO = "demo"


@dataclass
class EnhancedCVConfig:
    """Configuration for enhanced CV system integration."""
    mode: IntegrationMode
    enable_vector_db: bool = True
    enable_orchestration: bool = True
    enable_templates: bool = True
    enable_specialized_agents: bool = True
    vector_db_path: Optional[str] = None
    template_cache_size: int = 100
    orchestration_timeout: timedelta = timedelta(minutes=30)
    max_concurrent_agents: int = 5
    enable_performance_monitoring: bool = True
    enable_error_recovery: bool = True
    debug_mode: bool = False
    api_key: Optional[str] = None
    enable_caching: bool = True
    enable_monitoring: bool = True
    
    def to_dict(self):
        """Convert config to dictionary with JSON-serializable values."""
        result = {}
        for field_name, field_value in self.__dict__.items():
            if isinstance(field_value, IntegrationMode):
                result[field_name] = field_value.value
            elif isinstance(field_value, timedelta):
                result[field_name] = field_value.total_seconds()
            else:
                result[field_name] = field_value
        return result
    
    @classmethod
    def from_dict(cls, data: dict):
        """Create config from dictionary."""
        # Convert mode back to enum
        if 'mode' in data and isinstance(data['mode'], str):
            data['mode'] = IntegrationMode(data['mode'])
        # Convert timeout back to timedelta
        if 'orchestration_timeout' in data and isinstance(data['orchestration_timeout'], (int, float)):
            data['orchestration_timeout'] = timedelta(seconds=data['orchestration_timeout'])
        return cls(**data)


class EnhancedCVIntegration:
    """Main integration class for enhanced CV system components."""

    def __init__(self, config: Optional[EnhancedCVConfig] = None):
        self.config = config or EnhancedCVConfig(mode=IntegrationMode.PRODUCTION)
        self.logger = get_structured_logger(__name__)
        self.settings = get_config()
        self.error_recovery = get_error_recovery_service()

        # Component instances
        self._template_manager: Optional[ContentTemplateManager] = None
        self._vector_db = None
        self._orchestrator: Optional[AgentOrchestrator] = None
        self._workflow_builder: Optional[WorkflowBuilder] = None
        self._agents: Dict[str, Any] = {}

        # Performance tracking
        self._performance_stats = {
            "requests_processed": 0,
            "total_processing_time": 0.0,
            "errors_encountered": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }

        # Initialize components
        self._initialize_components()

    def _initialize_components(self):
        """Initialize all enhanced CV system components."""
        try:
            # Redact sensitive data from config before logging
            redacted_config = redact_sensitive_data(self.config.to_dict())
            self.logger.info("Initializing enhanced CV system components", extra={
                "mode": self.config.mode.value,
                "config": redacted_config
            })

            # Initialize template manager
            if self.config.enable_templates:
                self._template_manager = get_template_manager()
                self.logger.info("Template manager initialized")

            # Initialize vector database
            if self.config.enable_vector_db:
                self._vector_db = get_enhanced_vector_db()
                self.logger.info("Vector database initialized")

            # Initialize orchestrator
            if self.config.enable_orchestration:
                self._orchestrator = get_agent_orchestrator()
                self._workflow_builder = get_workflow_builder(self._orchestrator)
                self.logger.info("Orchestration components initialized")

            # Initialize specialized agents
            if self.config.enable_specialized_agents:
                self._initialize_agents()
                self.logger.info("Specialized agents initialized")

            self.logger.info("Enhanced CV system integration initialized successfully")

        except Exception as e:
            self.logger.error("Failed to initialize enhanced CV system components", extra={
                "error": str(e),
                "config": self.config.to_dict()
            })
            if self.config.enable_error_recovery:
                # Note: handle_error is async, but we're in a sync context
                # For now, we'll skip the error recovery call during initialization
                self.logger.warning("Error recovery skipped during initialization due to async/sync mismatch")
            raise

    def _initialize_agents(self):
        """Initialize all specialized agents."""
        try:
            # Enhanced content writer
            self._agents["enhanced_content_writer"] = EnhancedContentWriterAgent()

            # Specialized agents
            self._agents["cv_analysis"] = get_agent("cv_analysis")
            self._agents["content_optimization"] = get_agent("content_optimization")
            self._agents["quality_assurance"] = get_agent("quality_assurance")

            self.logger.info("Agents initialized", extra={
                "agent_count": len(self._agents),
                "agent_types": list(self._agents.keys())
            })

        except Exception as e:
            self.logger.error("Failed to initialize agents", extra={"error": str(e)})
            raise

    # Template Management
    def get_template(self, template_id: str, category: str = None) -> Optional[Dict[str, Any]]:
        """Get a content template."""
        if not self._template_manager:
            return None

        try:
            return self._template_manager.get_template(template_id, category)
        except Exception as e:
            self.logger.error("Failed to get template", extra={
                "template_id": template_id,
                "category": category,
                "error": str(e)
            })
            return None

    def format_template(
        self,
        template_id: str,
        variables: Dict[str, Any],
        category: str = None
    ) -> Optional[str]:
        """Format a template with variables."""
        if not self._template_manager:
            return None

        try:
            return self._template_manager.format_template(template_id, variables, category)
        except Exception as e:
            self.logger.error("Failed to format template", extra={
                "template_id": template_id,
                "category": category,
                "error": str(e)
            })
            return None

    def list_templates(self, category: str = None) -> List[str]:
        """List available templates."""
        if not self._template_manager:
            return []

        try:
            return self._template_manager.list_templates(category)
        except Exception as e:
            self.logger.error("Failed to list templates", extra={
                "category": category,
                "error": str(e)
            })
            return []

    # Vector Database Operations
    async def store_content(
        self,
        content: str,
        content_type: ContentType,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """Store content in vector database."""
        if not self._vector_db:
            return None

        try:
            return await store_enhanced_document(
                content=content,
                content_type=content_type,
                metadata=metadata or {}
            )
        except Exception as e:
            self.logger.error("Failed to store content", extra={
                "content_type": content_type.value,
                "error": str(e)
            })
            return None

    async def search_content(
        self,
        query: str,
        content_type: Optional[ContentType] = None,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Search for similar content."""
        if not self._vector_db:
            return []

        try:
            return await search_enhanced_documents(
                query=query,
                content_type=content_type,
                limit=limit
            )
        except Exception as e:
            self.logger.error("Failed to search content", extra={
                "query": query,
                "content_type": content_type.value if content_type else None,
                "error": str(e)
            })
            return []

    async def find_similar_content(
        self,
        content: str,
        content_type: Optional[ContentType] = None,
        limit: int = 3
    ) -> List[Dict[str, Any]]:
        """Find content similar to the provided content."""
        if not self._vector_db:
            return []

        try:
            return await find_similar_content(
                content=content,
                content_type=content_type,
                limit=limit
            )
        except Exception as e:
            self.logger.error("Failed to find similar content", extra={
                "content_type": content_type.value if content_type else None,
                "error": str(e)
            })
            return []

    # Agent Operations
    def get_agent(self, agent_type: str):
        """Get an agent instance."""
        return self._agents.get(agent_type)

    def list_agents(self) -> List[str]:
        """List available agent types."""
        return list(self._agents.keys())

    def get_orchestrator(self):
        """Get the orchestrator instance."""
        return self._orchestrator

    # Workflow Execution
    async def execute_workflow(
        self,
        workflow_type: Union[WorkflowType, str],
        input_data: Dict[str, Any],
        session_id: Optional[str] = None,
        custom_options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute a predefined workflow."""
        if not self._workflow_builder:
            raise RuntimeError("Orchestration not enabled")

        start_time = datetime.now()

        try:
            # Convert string to enum if needed
            if isinstance(workflow_type, str):
                workflow_type = WorkflowType(workflow_type)

            self.logger.info("Executing workflow", extra={
                "workflow_type": workflow_type.value,
                "session_id": session_id,
                "input_keys": list(input_data.keys())
            })

            result = await self._workflow_builder.execute_workflow(
                workflow_type=workflow_type,
                input_data=input_data,
                session_id=session_id,
                custom_options=custom_options
            )

            # Update performance stats
            processing_time = (datetime.now() - start_time).total_seconds()
            self._performance_stats["requests_processed"] += 1
            self._performance_stats["total_processing_time"] += processing_time

            self.logger.info("Workflow completed", extra={
                "workflow_type": workflow_type.value,
                "session_id": session_id,
                "processing_time": processing_time,
                "success": result.success,
                "tasks_completed": len(result.task_results)
            })

            return {
                "success": result.success,
                "results": result.task_results,
                "metadata": result.metadata,
                "processing_time": processing_time,
                "errors": result.error_summary
            }

        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            self._performance_stats["errors_encountered"] += 1

            self.logger.error("Workflow execution failed", extra={
                "workflow_type": workflow_type.value if hasattr(workflow_type, 'value') else str(workflow_type),
                "session_id": session_id,
                "processing_time": processing_time,
                "error": str(e)
            })

            if self.config.enable_error_recovery:
                recovery_result = await self.error_recovery.handle_error(
                    exception=e,
                    item_id=f"workflow_{workflow_type.value}",
                    item_type=ContentType.EXECUTIVE_SUMMARY,  # Default content type for workflow errors
                    session_id=session_id or "default",
                    context={
                        "workflow_type": workflow_type.value,  # Convert enum to string
                        "input_data": input_data
                    }
                )
                if recovery_result.strategy in [RecoveryStrategy.IMMEDIATE_RETRY, RecoveryStrategy.EXPONENTIAL_BACKOFF, RecoveryStrategy.LINEAR_BACKOFF]:
                    self.logger.info("Retrying workflow after error recovery")
                    if recovery_result.delay_seconds > 0:
                        await asyncio.sleep(recovery_result.delay_seconds)
                    return await self.execute_workflow(
                        workflow_type, input_data, session_id, custom_options
                    )

            return {
                "success": False,
                "results": [],
                "metadata": {},
                "processing_time": processing_time,
                "errors": [str(e)]
            }

    # Convenience workflow methods
    async def generate_basic_cv(
        self,
        personal_info: Dict[str, Any],
        experience: List[Dict[str, Any]],
        education: List[Dict[str, Any]],
        session_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate a basic CV."""
        return await self.execute_workflow(
            WorkflowType.BASIC_CV_GENERATION,
            {
                "personal_info": personal_info,
                "experience": experience,
                "education": education,
                **kwargs
            },
            session_id
        )

    async def generate_job_tailored_cv(
        self,
        personal_info: Dict[str, Any],
        experience: List[Dict[str, Any]],
        job_description: Dict[str, Any],
        session_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate a job-tailored CV."""
        return await self.execute_workflow(
            WorkflowType.JOB_TAILORED_CV,
            {
                "personal_info": personal_info,
                "experience": experience,
                "job_description": job_description,
                **kwargs
            },
            session_id
        )

    async def optimize_cv(
        self,
        existing_cv: Dict[str, Any],
        session_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Optimize an existing CV."""
        return await self.execute_workflow(
            WorkflowType.CV_OPTIMIZATION,
            {
                "existing_cv": existing_cv,
                **kwargs
            },
            session_id
        )

    async def check_cv_quality(
        self,
        cv_content: Dict[str, Any],
        session_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Perform quality assurance on CV content."""
        return await self.execute_workflow(
            WorkflowType.QUALITY_ASSURANCE,
            {
                "cv_content": cv_content,
                **kwargs
            },
            session_id
        )

    # Statistics and Monitoring
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = self._performance_stats.copy()

        # Calculate derived metrics
        if stats["requests_processed"] > 0:
            stats["average_processing_time"] = (
                stats["total_processing_time"] / stats["requests_processed"]
            )
            stats["error_rate"] = (
                stats["errors_encountered"] / stats["requests_processed"]
            )
        else:
            stats["average_processing_time"] = 0.0
            stats["error_rate"] = 0.0

        # Add component stats
        if self._vector_db:
            try:
                vector_stats = self._vector_db.get_enhanced_stats()
                stats["vector_db"] = vector_stats
            except Exception:
                pass

        if self._orchestrator:
            try:
                orchestrator_stats = self._orchestrator.get_stats()
                stats["orchestrator"] = orchestrator_stats
            except Exception:
                pass

        return stats

    def reset_performance_stats(self):
        """Reset performance statistics."""
        self._performance_stats = {
            "requests_processed": 0,
            "total_processing_time": 0.0,
            "errors_encountered": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        self.logger.info("Performance statistics reset")

    # Health Check
    def health_check(self) -> Dict[str, Any]:
        """Perform a health check on all components."""
        health = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {}
        }

        try:
            # Check template manager
            if self._template_manager:
                template_count = len(self._template_manager.list_templates())
                health["components"]["template_manager"] = {
                    "status": "healthy",
                    "template_count": template_count
                }

            # Check vector database
            if self._vector_db:
                try:
                    vector_stats = self._vector_db.get_enhanced_stats()
                    health["components"]["vector_db"] = {
                        "status": "healthy",
                        "document_count": vector_stats.get("total_documents", 0)
                    }
                except Exception as e:
                    health["components"]["vector_db"] = {
                        "status": "unhealthy",
                        "error": str(e)
                    }
                    health["status"] = "degraded"

            # Check orchestrator
            if self._orchestrator:
                try:
                    orchestrator_stats = self._orchestrator.get_stats()
                    health["components"]["orchestrator"] = {
                        "status": "healthy",
                        "registered_agents": orchestrator_stats.get("registered_agents", 0)
                    }
                except Exception as e:
                    health["components"]["orchestrator"] = {
                        "status": "unhealthy",
                        "error": str(e)
                    }
                    health["status"] = "degraded"

            # Check agents
            healthy_agents = 0
            for agent_type, agent in self._agents.items():
                try:
                    # Simple health check - try to access agent properties
                    _ = agent.name if hasattr(agent, 'name') else agent_type
                    healthy_agents += 1
                except Exception:
                    health["status"] = "degraded"

            health["components"]["agents"] = {
                "status": "healthy" if healthy_agents == len(self._agents) else "degraded",
                "healthy_count": healthy_agents,
                "total_count": len(self._agents)
            }

        except Exception as e:
            health["status"] = "unhealthy"
            health["error"] = str(e)

        return health


# Global integration instance
_enhanced_cv_integration = None


def get_enhanced_cv_integration(config: Optional[EnhancedCVConfig] = None) -> EnhancedCVIntegration:
    """Get enhanced CV system integration instance."""
    global _enhanced_cv_integration
    if _enhanced_cv_integration is None:
        _enhanced_cv_integration = EnhancedCVIntegration(config)
    return _enhanced_cv_integration


def reset_enhanced_cv_integration():
    """Reset the global enhanced CV system integration instance."""
    global _enhanced_cv_integration
    _enhanced_cv_integration = None


# Convenience functions
async def generate_cv(
    workflow_type: Union[WorkflowType, str],
    input_data: Dict[str, Any],
    session_id: Optional[str] = None,
    config: Optional[EnhancedCVConfig] = None
) -> Dict[str, Any]:
    """Generate a CV using the specified workflow."""
    integration = get_enhanced_cv_integration(config)
    return await integration.execute_workflow(
        workflow_type, input_data, session_id
    )


def get_cv_templates(category: str = None) -> List[str]:
    """Get available CV templates."""
    integration = get_enhanced_cv_integration()
    return integration.list_templates(category)


async def search_cv_examples(
    query: str,
    content_type: Optional[ContentType] = None,
    limit: int = 5
) -> List[Dict[str, Any]]:
    """Search for CV examples."""
    integration = get_enhanced_cv_integration()
    return await integration.search_content(query, content_type, limit)


def get_system_health() -> Dict[str, Any]:
    """Get system health status."""
    integration = get_enhanced_cv_integration()
    return integration.health_check()


def get_system_stats() -> Dict[str, Any]:
    """Get system performance statistics."""
    integration = get_enhanced_cv_integration()
    return integration.get_performance_stats()