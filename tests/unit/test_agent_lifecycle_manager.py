"""Unit tests for AgentLifecycleManager.

Tests agent lifecycle management, execution coordination,
and resource management functionality.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
from typing import Dict, Any, List

from src.core.agent_lifecycle_manager import AgentLifecycleManager
from src.agents.agent_base import EnhancedAgentBase
from src.orchestration.state import AgentState
from src.models.data_models import (
    StructuredCV,
    Section,
    Subsection,
    Item,
    JobDescriptionData,
    ProcessingStatus,
    ItemType,
    ItemStatus,
)
from src.utils.exceptions import AgentExecutionError, StateManagerError


class MockAgent(EnhancedAgentBase):
    """Mock agent for testing."""

    def __init__(
        self, name: str, should_fail: bool = False, execution_time: float = 0.1
    ):
        from src.models.data_models import AgentIO

        # Create mock schemas
        mock_schema = AgentIO(
            description="Mock schema", schema_type="object", properties={}
        )

        super().__init__(
            name=name,
            description=f"Mock agent {name}",
            input_schema=mock_schema,
            output_schema=mock_schema,
        )
        self.should_fail = should_fail
        self.execution_time = execution_time
        self.execution_count = 0
        self.last_execution_time = None

    async def _process_single_item(
        self,
        cv_data: Dict[str, Any],
        job_data: Dict[str, Any],
        item_id: str,
        research_findings: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Mock processing method."""
        self.execution_count += 1
        self.last_execution_time = datetime.now()

        # Simulate processing time
        await asyncio.sleep(self.execution_time)

        if self.should_fail:
            raise AgentExecutionError(f"Mock agent {self.name} failed")

        return {
            "processed_item_id": item_id,
            "agent_name": self.name,
            "execution_count": self.execution_count,
        }

    async def run_async(self, context: "AgentExecutionContext") -> "AgentResult":
        """Mock run_async implementation for compatibility."""
        from src.agents.agent_base import AgentResult

        try:
            result = await self._process_single_item(
                cv_data=context.cv_data or {},
                job_data=context.job_data or {},
                item_id="mock_item",
                research_findings={},
            )

            return AgentResult(
                success=True, data=result, metadata={"agent_name": self.name}
            )
        except Exception as e:
            return AgentResult(
                success=False, error=str(e), metadata={"agent_name": self.name}
            )

    async def run_as_node(self, state: AgentState) -> AgentState:
        """Mock LangGraph node implementation."""
        try:
            result = await self._process_single_item(
                cv_data=state.structured_cv.model_dump() if state.structured_cv else {},
                job_data=(
                    state.job_description_data.model_dump()
                    if state.job_description_data
                    else {}
                ),
                item_id=state.current_item_id or "default_item",
                research_findings=state.research_findings or {},
            )

            # Update state with result
            state.research_findings = state.research_findings or {}
            state.research_findings.update(result)

            return state
        except Exception as e:
            state.error_messages.append(str(e))
            return state


class TestAgentLifecycleManager:
    """Test cases for AgentLifecycleManager."""

    @pytest.fixture
    def mock_agents(self):
        """Create mock agents for testing."""
        return {
            "parser": MockAgent("parser"),
            "content_writer": MockAgent("content_writer"),
            "qa": MockAgent("qa"),
            "formatter": MockAgent("formatter"),
        }

    @pytest.fixture
    def lifecycle_manager(self, mock_agents):
        """Create an AgentLifecycleManager instance for testing."""
        manager = AgentLifecycleManager()

        # Register mock agents
        for agent_name, agent_instance in mock_agents.items():
            manager.register_agent_type(
                agent_name, lambda name=agent_name: mock_agents[name]
            )

        return manager

    @pytest.fixture
    def sample_agent_state(self):
        """Create a sample AgentState for testing."""
        return AgentState(
            session_id="test_session_123",
            structured_cv=StructuredCV(
                sections=[
                    Section(
                        name="Experience",
                        subsections=[
                            Subsection(
                                name="Software Engineer",
                                items=[
                                    Item(
                                        content="Developed web applications",
                                        status=ItemStatus.INITIAL,
                                        item_type=ItemType.BULLET_POINT,
                                        metadata={
                                            "created_at": datetime.now().isoformat()
                                        },
                                    )
                                ],
                            )
                        ],
                    )
                ]
            ),
            job_description_data=JobDescriptionData(
                raw_text="Software Engineer position",
                company_name="Test Company",
                role_title="Software Engineer",
                key_requirements=["Python", "React"],
                nice_to_have=["Docker"],
                company_info="Tech startup",
            ),
            current_item_id="item_1",
            items_to_process_queue=["item_2", "item_3"],
            research_findings={},
            user_feedback=None,
            error_messages=[],
            processing_complete=False,
        )

    def test_lifecycle_manager_initialization(self, mock_agents):
        """Test AgentLifecycleManager initialization."""
        lifecycle_manager = AgentLifecycleManager()

        assert hasattr(lifecycle_manager, "_pools")
        assert hasattr(lifecycle_manager, "_pool_configs")
        assert hasattr(lifecycle_manager, "_agent_registry")
        assert hasattr(lifecycle_manager, "_global_metrics")

    @pytest.mark.asyncio
    async def test_get_agent_success(self, lifecycle_manager, sample_agent_state):
        """Test successful agent retrieval from pool."""
        agent_name = "parser"

        # Get agent from pool
        managed_agent = lifecycle_manager.get_agent(
            agent_name, session_id="test_session"
        )

        assert managed_agent is not None
        assert managed_agent.config.agent_type == agent_name
        assert managed_agent.session_id == "test_session"

        # Return agent to pool
        lifecycle_manager.return_agent(managed_agent)
        assert managed_agent.state.value == "idle"

    def test_get_nonexistent_agent(self, lifecycle_manager):
        """Test getting a non-existent agent type."""
        # Try to get an agent type that wasn't registered
        managed_agent = lifecycle_manager.get_agent("nonexistent_agent")

        assert managed_agent is None

    def test_agent_pool_capacity(self, lifecycle_manager):
        """Test agent pool capacity limits."""
        agent_name = "parser"

        # Get multiple agents up to the max capacity (default is 5)
        agents = []
        for i in range(6):  # Try to get more than max capacity
            agent = lifecycle_manager.get_agent(agent_name, session_id=f"session_{i}")
            if agent:
                agents.append(agent)

        # Should not exceed max capacity
        assert len(agents) <= 5

        # Return all agents
        for agent in agents:
            lifecycle_manager.return_agent(agent)

    def test_dispose_agent(self, lifecycle_manager):
        """Test agent disposal."""
        agent_name = "parser"

        # Get an agent
        managed_agent = lifecycle_manager.get_agent(
            agent_name, session_id="test_session"
        )
        assert managed_agent is not None

        # Dispose the agent
        lifecycle_manager.dispose_agent(managed_agent)

        # Check that agent is marked as disposed
        assert managed_agent.state.value == "disposed"

    def test_dispose_session_agents(self, lifecycle_manager):
        """Test disposing all agents for a session."""
        session_id = "test_session_123"

        # Get multiple agents for the same session
        agents = []
        for agent_type in ["parser", "content_writer", "qa"]:
            agent = lifecycle_manager.get_agent(agent_type, session_id=session_id)
            if agent:
                agents.append(agent)

        assert len(agents) > 0

        # Dispose all session agents
        lifecycle_manager.dispose_session_agents(session_id)

        # Check that all agents are disposed
        for agent in agents:
            assert agent.state.value == "disposed"

    def test_get_statistics(self, lifecycle_manager):
        """Test getting lifecycle manager statistics."""
        # Get some agents to populate statistics
        agents = []
        for agent_type in ["parser", "content_writer"]:
            agent = lifecycle_manager.get_agent(agent_type, session_id="test_session")
            if agent:
                agents.append(agent)

        # Get statistics
        stats = lifecycle_manager.get_statistics()

        assert "global_metrics" in stats
        assert "pool_statistics" in stats
        assert "active_sessions" in stats
        assert "container_stats" in stats

        # Check that pool statistics contain our agent types
        pool_stats = stats["pool_statistics"]
        assert "parser" in pool_stats
        assert "content_writer" in pool_stats

        # Return agents
        for agent in agents:
            lifecycle_manager.return_agent(agent)

    def test_warmup_pools(self, lifecycle_manager):
        """Test warming up agent pools."""
        # Warmup pools
        lifecycle_manager.warmup_pools()

        # Check that pools exist for registered agent types
        stats = lifecycle_manager.get_statistics()
        pool_stats = stats["pool_statistics"]

        assert "parser" in pool_stats
        assert "content_writer" in pool_stats
        assert "qa" in pool_stats
        assert "formatter" in pool_stats

    def test_shutdown(self, lifecycle_manager):
        """Test lifecycle manager shutdown."""
        # Get some agents first
        agents = []
        for agent_type in ["parser", "content_writer"]:
            agent = lifecycle_manager.get_agent(agent_type, session_id="test_session")
            if agent:
                agents.append(agent)

        assert len(agents) > 0

        # Shutdown the manager
        lifecycle_manager.shutdown()

        # Check that all agents are disposed
        for agent in agents:
            assert agent.state.value == "disposed"

        # Check that pools are cleared
        stats = lifecycle_manager.get_statistics()
        pool_stats = stats["pool_statistics"]
        for pool_stat in pool_stats.values():
            assert pool_stat["current_size"] == 0

    def test_background_tasks(self, lifecycle_manager):
        """Test background task management."""
        # Start background tasks
        lifecycle_manager.start_background_tasks()

        # Verify statistics are available (background tasks are running)
        stats = lifecycle_manager.get_statistics()
        assert isinstance(stats, dict)
        assert len(stats) > 0

        # Shutdown should stop background tasks
        lifecycle_manager.shutdown()

        # Verify shutdown completed
        assert lifecycle_manager._shutdown == True
