"""Integration layer for enhanced CV system components.

This module provides a unified interface for coordinating and managing
all enhanced CV generation features including agents, orchestration, templates,
vector database, and workflows.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum

from ..models.data_models import ContentType, ProcessingStatus
from ..config.logging_config import get_structured_logger
from ..config.settings import get_config
from ..services.error_recovery import get_error_recovery_service, RecoveryStrategy
from ..utils.security_utils import redact_sensitive_data
from ..core.performance_optimizer import PerformanceOptimizer
from ..core.async_optimizer import AsyncOptimizer
from ..core.caching_strategy import get_intelligent_cache_manager, CachePattern
import hashlib

# Enhanced CV system imports
from ..agents.enhanced_content_writer import EnhancedContentWriterAgent
from ..agents.specialized_agents import CVAnalysisAgent, create_cv_analysis_agent
from ..agents.quality_assurance_agent import QualityAssuranceAgent
from ..templates.content_templates import get_template_manager, ContentTemplateManager
from ..services.vector_store_service import get_vector_store_service
from ..core.enhanced_orchestrator import EnhancedOrchestrator
from ..core.state_manager import StateManager
from ..models.data_models import WorkflowType
from ..orchestration.state import AgentState


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
        if "mode" in data and isinstance(data["mode"], str):
            data["mode"] = IntegrationMode(data["mode"])
        # Convert timeout back to timedelta
        if "orchestration_timeout" in data and isinstance(
            data["orchestration_timeout"], (int, float)
        ):
            data["orchestration_timeout"] = timedelta(
                seconds=data["orchestration_timeout"]
            )
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
        self._orchestrator: Optional[EnhancedOrchestrator] = None
        self._state_manager: Optional[StateManager] = None
        self._agents: Dict[str, Any] = {}

        # Performance optimization components
        self._performance_optimizer: Optional[PerformanceOptimizer] = None
        self._async_optimizer: Optional[AsyncOptimizer] = None
        self._intelligent_cache = None

        # Performance tracking
        self._performance_stats = {
            "requests_processed": 0,
            "total_processing_time": 0.0,
            "errors_encountered": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

        # Initialize components
        self._initialize_components()

    def _initialize_components(self):
        """Initialize all enhanced CV system components."""
        try:
            # Redact sensitive data from config before logging
            redacted_config = redact_sensitive_data(asdict(self.config))
            self.logger.info(
                "Initializing enhanced CV system components",
                extra={"mode": self.config.mode.value, "config": redacted_config},
            )

            # Initialize template manager
            if self.config.enable_templates:
                self._template_manager = get_template_manager()
                self.logger.info("Template manager initialized")

            # Initialize vector database
            if self.config.enable_vector_db:
                self._vector_db = get_vector_store_service()
                self.logger.info("Vector database initialized")

            # Initialize orchestrator
            if self.config.enable_orchestration:
                self._state_manager = StateManager()
                self._orchestrator = EnhancedOrchestrator(self._state_manager)
                self.logger.info("Enhanced orchestration components initialized")

            # Initialize specialized agents
            if self.config.enable_specialized_agents:
                self._initialize_agents()
                self.logger.info("Specialized agents initialized")

            # Initialize performance optimization components
            if self.config.enable_performance_monitoring:
                self._performance_optimizer = PerformanceOptimizer()
                self._async_optimizer = AsyncOptimizer()
                self._intelligent_cache = get_intelligent_cache_manager()
                self.logger.info("Performance optimization components initialized")

            self.logger.info("Enhanced CV system integration initialized successfully")

        except Exception as e:
            self.logger.error(
                "Failed to initialize enhanced CV system components",
                extra={
                    "error": str(e),
                    "config": redact_sensitive_data(asdict(self.config)),
                },
            )
            if self.config.enable_error_recovery:
                # Note: handle_error is async, but we're in a sync context
                # For now, we'll skip the error recovery call during initialization
                self.logger.warning(
                    "Error recovery skipped during initialization due to async/sync mismatch"
                )
            raise

    def _initialize_agents(self):
        """Initialize all specialized agents."""
        try:
            # Enhanced content writer
            self._agents["enhanced_content_writer"] = EnhancedContentWriterAgent()

            # Specialized agents
            self._agents["cv_analysis"] = create_cv_analysis_agent()
            # content_optimization agent removed - was never implemented
            self._agents["quality_assurance"] = QualityAssuranceAgent()

            self.logger.info(
                "Agents initialized",
                extra={
                    "agent_count": len(self._agents),
                    "agent_types": list(self._agents.keys()),
                },
            )

        except Exception as e:
            self.logger.error("Failed to initialize agents", extra={"error": str(e)})
            raise

    # Template Management
    def get_template(
        self, template_id: str, category: str = None
    ) -> Optional[Dict[str, Any]]:
        """Get a content template."""
        if not self._template_manager:
            return None

        try:
            return self._template_manager.get_template(template_id, category)
        except Exception as e:
            self.logger.error(
                "Failed to get template",
                extra={
                    "template_id": template_id,
                    "category": category,
                    "error": str(e),
                },
            )
            return None

    def format_template(
        self, template_id: str, variables: Dict[str, Any], category: str = None
    ) -> Optional[str]:
        """Format a template with variables."""
        if not self._template_manager:
            return None

        try:
            return self._template_manager.format_template(
                template_id, variables, category
            )
        except Exception as e:
            self.logger.error(
                "Failed to format template",
                extra={
                    "template_id": template_id,
                    "category": category,
                    "error": str(e),
                },
            )
            return None

    def list_templates(self, category: str = None) -> List[str]:
        """List available templates."""
        if not self._template_manager:
            return []

        try:
            return self._template_manager.list_templates(category)
        except Exception as e:
            self.logger.error(
                "Failed to list templates",
                extra={"category": category, "error": str(e)},
            )
            return []

    # Vector Database Operations
    async def store_content(
        self,
        content: str,
        content_type: ContentType,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Store content in vector database."""
        if not self._vector_db:
            return None

        try:
            # Use vector store service to add content
            item_id = self._vector_db.add_item(
                item=content,
                content=content,
                metadata={"content_type": content_type.value, **(metadata or {})},
            )
            return item_id
        except Exception as e:
            self.logger.error(
                "Failed to store content",
                extra={"content_type": content_type.value, "error": str(e)},
            )
            return None

    async def search_content(
        self, query: str, content_type: Optional[ContentType] = None, limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Search for similar content."""
        if not self._vector_db:
            return []

        try:
            # Use vector store service to search content
            results = self._vector_db.search(
                query=query,
                n_results=limit,
                where={"content_type": content_type.value} if content_type else None,
            )
            return results
        except Exception as e:
            self.logger.error(
                "Failed to search content",
                extra={
                    "query": query,
                    "content_type": content_type.value if content_type else None,
                    "error": str(e),
                },
            )
            return []

    async def find_similar_content(
        self, content: str, content_type: Optional[ContentType] = None, limit: int = 3
    ) -> List[Dict[str, Any]]:
        """Find content similar to the provided content."""
        if not self._vector_db:
            return []

        try:
            # Use vector store service to find similar content
            results = self._vector_db.search(
                query=content,
                n_results=limit,
                where={"content_type": content_type.value} if content_type else None,
            )
            return results
        except Exception as e:
            self.logger.error(
                "Failed to find similar content",
                extra={
                    "content_type": content_type.value if content_type else None,
                    "error": str(e),
                },
            )
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

    @property
    def orchestrator(self):
        """Get the orchestrator instance as a property."""
        return self._orchestrator

    def _get_performance_context(self):
        """Get performance optimization context."""
        if self._performance_optimizer:
            return self._performance_optimizer.optimized_execution(
                operation_type="workflow_execution", expected_duration=30.0
            )
        else:
            # Return a no-op context manager if performance optimizer is not available
            from contextlib import nullcontext

            return nullcontext()

    def _get_async_context(self):
        """Get async optimization context."""
        if self._async_optimizer:
            return self._async_optimizer.optimized_context(
                max_concurrent=self.config.max_concurrent_agents,
                timeout=self.config.orchestration_timeout.total_seconds(),
            )
        else:
            # Return a no-op context manager if async optimizer is not available
            from contextlib import nullcontext

            return nullcontext()

    # Workflow Execution
    async def execute_workflow(
        self,
        workflow_type: Union[WorkflowType, str],
        input_data: AgentState,
        session_id: Optional[str] = None,
        custom_options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute a predefined workflow.

        Args:
            workflow_type: The type of workflow to execute.
            input_data: The AgentState object containing all necessary workflow data.
            session_id: Optional session ID for state management.
            custom_options: Optional custom options for the workflow.

        Returns:
            A dictionary containing the results of the workflow execution.
        """
        """Execute a predefined workflow."""
        if not self._orchestrator:
            raise RuntimeError("Orchestration not enabled")

        start_time = datetime.now()
        success = False  # Initialize success variable
        result_state = None  # Initialize result_state variable

        # Check intelligent cache first
        cache_key = None
        if self._intelligent_cache:
            cache_data = {
                "workflow_type": (
                    workflow_type.value
                    if hasattr(workflow_type, "value")
                    else str(workflow_type)
                ),
                "input_data": input_data.model_dump(),
                "custom_options": custom_options or {},
            }
            cache_key = hashlib.md5(str(cache_data).encode()).hexdigest()
            cached_result = self._intelligent_cache.get(cache_key)
            if cached_result:
                self.logger.info(
                    "Workflow result served from cache", cache_key=cache_key
                )
                self._performance_stats["cache_hits"] += 1
                return cached_result
            else:
                self._performance_stats["cache_misses"] += 1

        try:
            # Use performance optimizer for the entire workflow execution
            async with self._get_performance_context():
                # Convert string to enum if needed
                if isinstance(workflow_type, str):
                    workflow_type = WorkflowType(workflow_type)

                # Use the provided AgentState directly
                initial_agent_state = input_data

                self.logger.info(
                    "Executing workflow",
                    extra={
                        "workflow_type": workflow_type.value,
                        "session_id": session_id,
                        "input_data_type": "AgentState",
                    },
                )

                # Execute workflow with clean AgentState input
                if workflow_type in [
                    WorkflowType.BASIC_CV_GENERATION,
                    WorkflowType.JOB_TAILORED_CV,
                ]:
                    # Populate state manager with AgentState components
                    if initial_agent_state.structured_cv:
                        self._orchestrator.state_manager.set_structured_cv(
                            initial_agent_state.structured_cv
                        )
                        self.logger.info("Structured CV data set in state manager")
                        # Set job description data directly on StructuredCV.metadata if present
                        if initial_agent_state.job_description_data:
                            if hasattr(
                                self._orchestrator.state_manager.get_structured_cv(),
                                "metadata",
                            ):
                                self._orchestrator.state_manager.get_structured_cv().metadata.extra[
                                    "job_description"
                                ] = (
                                    initial_agent_state.job_description_data.model_dump()
                                )
                                self.logger.info(
                                    "Job description data set in StructuredCV.metadata.extra"
                                )
                        else:
                            self.logger.warning(
                                "Job description data missing in AgentState"
                            )
                        self.logger.info("State manager populated from AgentState")

                        # No fallback needed - AgentState is the only supported input type

                        # Initialize workflow after setting up the data
                        await self._orchestrator.initialize_workflow()

                        # Execute the full workflow with async optimization
                        async with self._get_async_context():
                            result_state = (
                                await self._orchestrator.execute_full_workflow()
                            )

                        success = not bool(
                            result_state.error_messages if result_state else True
                        )  # Assume success if no result_state
                        self.logger.info(f"Workflow success: {success}")

                    else:
                        self.logger.warning(
                            f"Workflow type {workflow_type.value} not fully implemented for AgentState input or not recognized."
                        )
                        success = False
                        result_state = None

                    # Update performance stats
                    processing_time = (datetime.now() - start_time).total_seconds()
                    self._performance_stats["requests_processed"] += 1
                    self._performance_stats["total_processing_time"] += processing_time

                    self.logger.info(
                        "Workflow completed",
                        extra={
                            "workflow_type": workflow_type.value,
                            "session_id": session_id,
                            "processing_time": processing_time,
                            "success": success,
                        },
                    )

                    # Cache successful results
                    if (
                        success
                        and result_state
                        and cache_key
                        and self._intelligent_cache
                    ):
                        workflow_result = {
                            "success": success,
                            "result_state": result_state,
                            "processing_time": processing_time,
                            "session_id": session_id,
                        }

                        self._intelligent_cache.set(
                            cache_key,
                            workflow_result,
                            ttl_hours=2,
                            tags={"workflow", workflow_type.value, "cv_generation"},
                            priority=3,
                            pattern=CachePattern.READ_HEAVY,
                        )

                    # Debug logging for final return structure
                    final_errors = result_state.error_messages if result_state else []
                self.logger.info(
                    f"Final return structure - success: {success}, errors: {final_errors}"
                )

                return {
                    "success": success,
                    "results": result_state.model_dump() if result_state else {},
                    "metadata": {
                        "workflow_type": workflow_type.value,
                        "session_id": session_id,
                    },
                    "processing_time": processing_time,
                    "errors": final_errors,
                }

        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            self._performance_stats["errors_encountered"] += 1

            self.logger.error(
                "Workflow execution failed",
                extra={
                    "workflow_type": (
                        workflow_type.value
                        if hasattr(workflow_type, "value")
                        else str(workflow_type)
                    ),
                    "session_id": session_id,
                    "processing_time": processing_time,
                    "error": str(e),
                },
            )

            if self.config.enable_error_recovery:
                recovery_result = await self.error_recovery.handle_error(
                    exception=e,
                    item_id=f"workflow_{workflow_type.value}",
                    item_type=ContentType.EXECUTIVE_SUMMARY,  # Default content type for workflow errors
                    session_id=session_id or "default",
                    context={
                        "workflow_type": workflow_type.value,  # Convert enum to string
                        "input_data": input_data,
                    },
                )
                if recovery_result.strategy in [
                    RecoveryStrategy.IMMEDIATE_RETRY,
                    RecoveryStrategy.EXPONENTIAL_BACKOFF,
                    RecoveryStrategy.LINEAR_BACKOFF,
                ]:
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
                "errors": [str(e)],
            }

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
        return await self.execute_workflow(
            WorkflowType.BASIC_CV_GENERATION,
            {
                "personal_info": personal_info,
                "experience": experience,
                "education": education,
                **kwargs,
            },
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

        return await self.execute_workflow(
            WorkflowType.JOB_TAILORED_CV,
            {
                "personal_info": personal_info,
                "experience": experience,
                "job_description": job_desc_dict,
                **kwargs,
            },
            session_id,
        )

    async def optimize_cv(
        self, existing_cv: Dict[str, Any], session_id: Optional[str] = None, **kwargs
    ) -> Dict[str, Any]:
        """Optimize an existing CV."""
        return await self.execute_workflow(
            WorkflowType.CV_OPTIMIZATION,
            {"existing_cv": existing_cv, **kwargs},
            session_id,
        )

    async def check_cv_quality(
        self, cv_content: Dict[str, Any], session_id: Optional[str] = None, **kwargs
    ) -> Dict[str, Any]:
        """Perform quality assurance on CV content."""
        return await self.execute_workflow(
            WorkflowType.QUALITY_ASSURANCE,
            {"cv_content": cv_content, **kwargs},
            session_id,
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
                # Enhanced orchestrator doesn't have get_stats method
                # Add basic orchestrator info instead
                stats["orchestrator"] = {
                    "type": "enhanced_orchestrator",
                    "status": "active",
                }
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
            "cache_misses": 0,
        }
        self.logger.info("Performance statistics reset")

    # Health Check
    def health_check(self) -> Dict[str, Any]:
        """Perform a health check on all components."""
        health = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {},
        }

        try:
            # Check template manager
            if self._template_manager:
                template_count = len(self._template_manager.list_templates())
                health["components"]["template_manager"] = {
                    "status": "healthy",
                    "template_count": template_count,
                }

            # Check vector database
            if self._vector_db:
                try:
                    vector_stats = self._vector_db.get_enhanced_stats()
                    health["components"]["vector_db"] = {
                        "status": "healthy",
                        "document_count": vector_stats.get("total_documents", 0),
                    }
                except Exception as e:
                    health["components"]["vector_db"] = {
                        "status": "unhealthy",
                        "error": str(e),
                    }
                    health["status"] = "degraded"

            # Check orchestrator
            if self._orchestrator:
                try:
                    # Simple health check for the enhanced orchestrator
                    health["components"]["orchestrator"] = {
                        "status": "healthy",
                        "type": "enhanced_orchestrator",
                    }
                except Exception as e:
                    health["components"]["orchestrator"] = {
                        "status": "unhealthy",
                        "error": str(e),
                    }
                    health["status"] = "degraded"

            # Check agents
            healthy_agents = 0
            for agent_type, agent in self._agents.items():
                try:
                    # Simple health check - try to access agent properties
                    _ = agent.name if hasattr(agent, "name") else agent_type
                    healthy_agents += 1
                except Exception:
                    health["status"] = "degraded"

            health["components"]["agents"] = {
                "status": (
                    "healthy" if healthy_agents == len(self._agents) else "degraded"
                ),
                "healthy_count": healthy_agents,
                "total_count": len(self._agents),
            }

        except Exception as e:
            health["status"] = "unhealthy"
            health["error"] = str(e)

        return health


# Global integration instance
_enhanced_cv_integration = None


def get_enhanced_cv_integration(
    config: Optional[EnhancedCVConfig] = None,
) -> EnhancedCVIntegration:
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
    config: Optional[EnhancedCVConfig] = None,
) -> Dict[str, Any]:
    """Generate a CV using the specified workflow."""
    integration = get_enhanced_cv_integration(config)
    return await integration.execute_workflow(workflow_type, input_data, session_id)


def get_cv_templates(category: str = None) -> List[str]:
    """Get available CV templates."""
    integration = get_enhanced_cv_integration()
    return integration.list_templates(category)


async def search_cv_examples(
    query: str, content_type: Optional[ContentType] = None, limit: int = 5
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
