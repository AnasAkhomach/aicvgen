"""Agent lifecycle manager for simplified agent management with dependency injection."""

import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

from src.config.logging_config import get_structured_logger
from src.error_handling.agent_error_handler import AgentErrorHandler as ErrorHandler
from src.core.container import get_container

logger = get_structured_logger(__name__)


@dataclass
class AgentMetrics:
    """Simple metrics for agent usage."""

    created_at: datetime = None
    last_used: datetime = None
    usage_count: int = 0

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.last_used is None:
            self.last_used = datetime.now()


class AgentLifecycleManager:
    """Simplified agent lifecycle manager using dependency injection container."""

    def __init__(self):
        self.container = get_container()
        self._lock = threading.RLock()
        self._agent_metrics: Dict[str, AgentMetrics] = {}
        self._error_handler = ErrorHandler()

        logger.info(
            "Agent lifecycle manager initialized with dependency injection container"
        )

    def get_agent(self, agent_type: str, session_id: Optional[str] = None) -> Any:
        """Get an agent instance from the container."""
        with self._lock:
            try:
                # Use the container to get the agent
                if hasattr(self.container, agent_type):
                    agent = getattr(self.container, agent_type)()

                    # Track metrics
                    agent_id = f"{agent_type}_{session_id or 'default'}"
                    if agent_id not in self._agent_metrics:
                        self._agent_metrics[agent_id] = AgentMetrics(
                            created_at=datetime.now(), last_used=datetime.now()
                        )
                    else:
                        self._agent_metrics[agent_id].last_used = datetime.now()
                        self._agent_metrics[agent_id].usage_count += 1

                    logger.debug(
                        f"Retrieved agent: {agent_type}", session_id=session_id
                    )
                    return agent
                else:
                    raise AttributeError(
                        f"Agent type '{agent_type}' not found in container"
                    )

            except Exception as e:
                logger.error(f"Failed to get agent: {agent_type}", error=str(e))
                raise

    def get_metrics(self) -> Dict[str, Any]:
        """Get simple metrics about agent usage."""
        with self._lock:
            return {
                "total_agents": len(self._agent_metrics),
                "agent_metrics": {
                    k: {
                        "created_at": v.created_at.isoformat(),
                        "last_used": v.last_used.isoformat(),
                        "usage_count": v.usage_count,
                    }
                    for k, v in self._agent_metrics.items()
                },
            }

    def cleanup(self):
        """Cleanup resources."""
        with self._lock:
            self._agent_metrics.clear()
            logger.info("Agent lifecycle manager cleaned up")


# Global instance
_lifecycle_manager_lock = threading.Lock()
_lifecycle_manager: Optional[AgentLifecycleManager] = None


def get_agent_lifecycle_manager() -> AgentLifecycleManager:
    """Get the singleton agent lifecycle manager."""
    global _lifecycle_manager
    if _lifecycle_manager is None:
        with _lifecycle_manager_lock:
            if _lifecycle_manager is None:
                _lifecycle_manager = AgentLifecycleManager()
    return _lifecycle_manager
