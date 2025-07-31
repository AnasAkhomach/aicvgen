"""Module for managing agent instances with proper lifecycle management."""

import threading
from typing import Any, Dict, List, Optional

from src.config.logging_config import get_structured_logger
from src.core.agent_lifecycle_manager import get_agent_lifecycle_manager
from src.error_handling.agent_error_handler import AgentErrorHandler

logger = get_structured_logger(__name__)


class CVAgentManager:
    """Manages agent instances with proper lifecycle management and standardized init/cleanup."""

    def __init__(self, session_id: Optional[str] = None):
        """Initialize the CV agent manager with lifecycle management.

        Args:
            session_id: Optional session identifier for agent tracking
        """
        self._session_id = session_id or "default"
        self._lifecycle_manager = get_agent_lifecycle_manager()
        self._error_handler = AgentErrorHandler()
        self._lock = threading.RLock()
        self._initialized = False

        # Initialize the manager
        self._initialize()

        logger.info(
            "CV Agent Manager initialized with lifecycle management",
            session_id=self._session_id,
        )

    def _initialize(self) -> None:
        """Initialize the agent manager with standardized procedures."""
        with self._lock:
            if self._initialized:
                return

            try:
                # Validate lifecycle manager is available
                if not self._lifecycle_manager:
                    raise RuntimeError("Agent lifecycle manager not available")

                # Mark as initialized
                self._initialized = True
                logger.debug("CV Agent Manager initialization completed")

            except Exception as e:
                logger.error("Failed to initialize CV Agent Manager", error=str(e))
                raise

    def get_agent(
        self, agent_type: str, session_id: Optional[str] = None
    ) -> Optional[Any]:
        """Get an agent instance by type with proper lifecycle management.

        Args:
            agent_type: The type of agent to retrieve
            session_id: Optional session ID override

        Returns:
            Agent instance or None if not found
        """
        if not self._initialized:
            raise RuntimeError("CV Agent Manager not properly initialized")

        effective_session_id = session_id or self._session_id

        try:
            with self._lock:
                agent = self._lifecycle_manager.get_agent(
                    agent_type=agent_type, session_id=effective_session_id
                )

                logger.debug(
                    f"Retrieved agent: {agent_type}", session_id=effective_session_id
                )
                return agent

        except Exception as e:
            logger.error(
                f"Failed to get agent: {agent_type}",
                error=str(e),
                session_id=effective_session_id,
            )
            # Use error handler for fallback behavior
            return self._error_handler.handle_general_error(
                error=e,
                agent_type=agent_type,
                context=f"get_agent:{effective_session_id}",
            )

    def list_agents(self) -> List[str]:
        """List available agent types from the container.

        Returns:
            List of available agent type names
        """
        if not self._initialized:
            raise RuntimeError("CV Agent Manager not properly initialized")

        try:
            # Get available agents from the container
            container = self._lifecycle_manager.container
            agent_types = [
                name
                for name in dir(container)
                if name.endswith("_agent") and not name.startswith("_")
            ]

            logger.debug(f"Available agent types: {agent_types}")
            return agent_types

        except Exception as e:
            logger.error("Failed to list agents", error=str(e))
            return []

    def get_metrics(self) -> Dict[str, Any]:
        """Get agent usage metrics.

        Returns:
            Dictionary containing agent metrics
        """
        if not self._initialized:
            raise RuntimeError("CV Agent Manager not properly initialized")

        try:
            return self._lifecycle_manager.get_metrics()
        except Exception as e:
            logger.error("Failed to get metrics", error=str(e))
            return {}

    def cleanup(self) -> None:
        """Cleanup resources with standardized procedures."""
        with self._lock:
            if not self._initialized:
                return

            try:
                # Cleanup lifecycle manager if needed
                if hasattr(self._lifecycle_manager, "cleanup"):
                    self._lifecycle_manager.cleanup()

                self._initialized = False
                logger.info(
                    "CV Agent Manager cleanup completed", session_id=self._session_id
                )

            except Exception as e:
                logger.error(
                    "Error during CV Agent Manager cleanup",
                    error=str(e),
                    session_id=self._session_id,
                )

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()

    @property
    def session_id(self) -> str:
        """Get the current session ID."""
        return self._session_id

    @property
    def is_initialized(self) -> bool:
        """Check if the manager is properly initialized."""
        return self._initialized
