"""Module for monitoring the health and performance of the CV generation system."""

from datetime import datetime
from typing import Any, Dict, Optional

from src.config.logging_config import get_structured_logger
from src.error_handling.exceptions import VectorStoreError, WorkflowError
from src.orchestration.cv_workflow_graph import CVWorkflowGraph
from src.services.vector_store_service import VectorStoreService
from src.templates.content_templates import ContentTemplateManager


class CVSystemMonitor:
    """Monitors the health and performance of the CV generation system components."""

    def __init__(
        self,
        template_manager: Optional[ContentTemplateManager],
        vector_db: Optional[VectorStoreService],
        orchestrator: Optional[CVWorkflowGraph],
        agents: Dict[str, Any],
    ):
        self._template_manager = template_manager
        self._vector_db = vector_db
        self._orchestrator = orchestrator
        self._agents = agents
        self.logger = get_structured_logger(__name__)

        self._performance_stats = {
            "requests_processed": 0,
            "total_processing_time": 0.0,
            "errors_encountered": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = self._performance_stats.copy()

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

        if self._vector_db:
            try:
                vector_stats = self._vector_db.get_enhanced_stats()
                stats["vector_db"] = vector_stats
            except (AttributeError, VectorStoreError) as e:
                self.logger.warning("Could not retrieve vector DB stats", error=str(e))

        if self._orchestrator:
            try:
                stats["orchestrator"] = {
                    "type": "enhanced_orchestrator",
                    "status": "active",
                }
            except (AttributeError, WorkflowError) as e:
                self.logger.warning(
                    "Could not retrieve orchestrator stats", error=str(e)
                )

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

    def health_check(self) -> Dict[str, Any]:
        """Perform a health check on all components."""
        health = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {},
        }

        try:
            if self._template_manager:
                template_count = len(self._template_manager.list_templates())
                health["components"]["template_manager"] = {
                    "status": "healthy",
                    "template_count": template_count,
                }

            if self._vector_db:
                try:
                    vector_stats = self._vector_db.get_enhanced_stats()
                    health["components"]["vector_db"] = {
                        "status": "healthy",
                        "document_count": vector_stats.get("total_documents", 0),
                    }
                except (AttributeError, VectorStoreError) as e:
                    health["components"]["vector_db"] = {
                        "status": "unhealthy",
                        "error": str(e),
                    }
                    health["status"] = "degraded"

            if self._orchestrator:
                try:
                    health["components"]["orchestrator"] = {
                        "status": "healthy",
                        "type": "enhanced_orchestrator",
                    }
                except (AttributeError, WorkflowError) as e:
                    health["components"]["orchestrator"] = {
                        "status": "unhealthy",
                        "error": str(e),
                    }
                    health["status"] = "degraded"

            healthy_agents = 0
            for agent_type, agent in self._agents.items():
                try:
                    _ = agent.name if hasattr(agent, "name") else agent_type
                    healthy_agents += 1
                except (AttributeError, TypeError) as e:
                    health["status"] = "degraded"
                    self.logger.warning(
                        f"Health check failed for agent {agent_type}", error=str(e)
                    )

            health["components"]["agents"] = {
                "status": (
                    "healthy" if healthy_agents == len(self._agents) else "degraded"
                ),
                "healthy_count": healthy_agents,
                "total_count": len(self._agents),
            }

        except (TypeError, ValueError, KeyError) as e:
            health["status"] = "unhealthy"
            health["error"] = str(e)

        return health
