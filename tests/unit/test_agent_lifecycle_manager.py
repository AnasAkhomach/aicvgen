"""Unit tests for AgentLifecycleManager.

Tests agent lifecycle management, execution coordination,
and resource management functionality.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
from typing import Dict, Any, List

from src.services.agent_lifecycle_manager import AgentLifecycleManager
from src.agents.enhanced_agent_base import EnhancedAgentBase
from src.orchestration.state import AgentState
from src.models.data_models import (
    StructuredCV, Section, Subsection, Item, ItemMetadata,
    JobDescriptionData, ProcessingStatus
)
from src.utils.exceptions import AgentExecutionError, StateManagerError


class MockAgent(EnhancedAgentBase):
    """Mock agent for testing."""
    
    def __init__(self, name: str, should_fail: bool = False, execution_time: float = 0.1):
        super().__init__(name=name, description=f"Mock agent {name}")
        self.should_fail = should_fail
        self.execution_time = execution_time
        self.execution_count = 0
        self.last_execution_time = None
    
    async def _process_single_item(self, cv_data: Dict[str, Any], job_data: Dict[str, Any], 
                                 item_id: str, research_findings: Dict[str, Any]) -> Dict[str, Any]:
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
            "execution_count": self.execution_count
        }
    
    async def run_as_node(self, state: AgentState) -> AgentState:
        """Mock LangGraph node implementation."""
        try:
            result = await self._process_single_item(
                cv_data=state.structured_cv.model_dump() if state.structured_cv else {},
                job_data=state.job_description_data.model_dump() if state.job_description_data else {},
                item_id=state.current_item_id or "default_item",
                research_findings=state.research_findings or {}
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
            "formatter": MockAgent("formatter")
        }

    @pytest.fixture
    def lifecycle_manager(self, mock_agents):
        """Create an AgentLifecycleManager instance for testing."""
        return AgentLifecycleManager(agents=mock_agents)

    @pytest.fixture
    def sample_agent_state(self):
        """Create a sample AgentState for testing."""
        return AgentState(
            session_id="test_session_123",
            structured_cv=StructuredCV(
                sections=[
                    Section(
                        id="section_1",
                        title="Experience",
                        subsections=[
                            Subsection(
                                id="subsection_1",
                                title="Software Engineer",
                                items=[
                                    Item(
                                        id="item_1",
                                        type="experience",
                                        content="Developed web applications",
                                        metadata=ItemMetadata(
                                            status=ProcessingStatus.PENDING,
                                            created_at=datetime.now()
                                        )
                                    )
                                ]
                            )
                        ]
                    )
                ]
            ),
            job_description_data=JobDescriptionData(
                raw_text="Software Engineer position",
                company_name="Test Company",
                role_title="Software Engineer",
                key_requirements=["Python", "React"],
                nice_to_have=["Docker"],
                company_info="Tech startup"
            ),
            current_item_id="item_1",
            items_to_process_queue=["item_2", "item_3"],
            research_findings={},
            user_feedback=None,
            error_messages=[],
            processing_complete=False
        )

    def test_lifecycle_manager_initialization(self, mock_agents):
        """Test AgentLifecycleManager initialization."""
        lifecycle_manager = AgentLifecycleManager(agents=mock_agents)
        
        assert lifecycle_manager.agents == mock_agents
        assert hasattr(lifecycle_manager, '_execution_history')
        assert hasattr(lifecycle_manager, '_agent_metrics')
        assert hasattr(lifecycle_manager, '_resource_monitor')

    @pytest.mark.asyncio
    async def test_execute_agent_success(self, lifecycle_manager, sample_agent_state):
        """Test successful agent execution."""
        agent_name = "parser"
        
        # Execute agent
        result_state = await lifecycle_manager.execute_agent(agent_name, sample_agent_state)
        
        assert result_state is not None
        assert result_state.session_id == sample_agent_state.session_id
        assert len(result_state.error_messages) == 0
        
        # Check that agent was executed
        agent = lifecycle_manager.agents[agent_name]
        assert agent.execution_count == 1
        assert agent.last_execution_time is not None

    @pytest.mark.asyncio
    async def test_execute_agent_failure(self, lifecycle_manager, sample_agent_state):
        """Test agent execution failure handling."""
        # Create a failing agent
        failing_agent = MockAgent("failing_agent", should_fail=True)
        lifecycle_manager.agents["failing_agent"] = failing_agent
        
        # Execute failing agent
        result_state = await lifecycle_manager.execute_agent("failing_agent", sample_agent_state)
        
        assert result_state is not None
        assert len(result_state.error_messages) > 0
        assert "Mock agent failing_agent failed" in result_state.error_messages[0]

    @pytest.mark.asyncio
    async def test_execute_agent_nonexistent(self, lifecycle_manager, sample_agent_state):
        """Test executing non-existent agent."""
        with pytest.raises(AgentExecutionError, match="Agent not found"):
            await lifecycle_manager.execute_agent("nonexistent_agent", sample_agent_state)

    @pytest.mark.asyncio
    async def test_execute_agent_sequence(self, lifecycle_manager, sample_agent_state):
        """Test executing a sequence of agents."""
        agent_sequence = ["parser", "content_writer", "qa"]
        
        # Execute agent sequence
        result_state = await lifecycle_manager.execute_agent_sequence(agent_sequence, sample_agent_state)
        
        assert result_state is not None
        assert result_state.session_id == sample_agent_state.session_id
        
        # Check that all agents were executed
        for agent_name in agent_sequence:
            agent = lifecycle_manager.agents[agent_name]
            assert agent.execution_count == 1

    @pytest.mark.asyncio
    async def test_execute_agent_sequence_with_failure(self, lifecycle_manager, sample_agent_state):
        """Test executing agent sequence with failure in middle."""
        # Add a failing agent
        failing_agent = MockAgent("failing_agent", should_fail=True)
        lifecycle_manager.agents["failing_agent"] = failing_agent
        
        agent_sequence = ["parser", "failing_agent", "qa"]
        
        # Execute sequence (should continue despite failure)
        result_state = await lifecycle_manager.execute_agent_sequence(
            agent_sequence, sample_agent_state, stop_on_error=False
        )
        
        assert result_state is not None
        assert len(result_state.error_messages) > 0
        
        # Check execution counts
        assert lifecycle_manager.agents["parser"].execution_count == 1
        assert lifecycle_manager.agents["failing_agent"].execution_count == 1
        assert lifecycle_manager.agents["qa"].execution_count == 1

    @pytest.mark.asyncio
    async def test_execute_agent_sequence_stop_on_error(self, lifecycle_manager, sample_agent_state):
        """Test executing agent sequence that stops on error."""
        # Add a failing agent
        failing_agent = MockAgent("failing_agent", should_fail=True)
        lifecycle_manager.agents["failing_agent"] = failing_agent
        
        agent_sequence = ["parser", "failing_agent", "qa"]
        
        # Execute sequence (should stop on error)
        result_state = await lifecycle_manager.execute_agent_sequence(
            agent_sequence, sample_agent_state, stop_on_error=True
        )
        
        assert result_state is not None
        assert len(result_state.error_messages) > 0
        
        # Check execution counts (qa should not be executed)
        assert lifecycle_manager.agents["parser"].execution_count == 1
        assert lifecycle_manager.agents["failing_agent"].execution_count == 1
        assert lifecycle_manager.agents["qa"].execution_count == 0

    @pytest.mark.asyncio
    async def test_execute_agents_parallel(self, lifecycle_manager, sample_agent_state):
        """Test executing agents in parallel."""
        agent_names = ["parser", "content_writer", "qa"]
        
        # Execute agents in parallel
        results = await lifecycle_manager.execute_agents_parallel(agent_names, sample_agent_state)
        
        assert len(results) == 3
        assert all(result is not None for result in results)
        
        # Check that all agents were executed
        for agent_name in agent_names:
            agent = lifecycle_manager.agents[agent_name]
            assert agent.execution_count == 1

    @pytest.mark.asyncio
    async def test_execute_agents_parallel_with_timeout(self, lifecycle_manager, sample_agent_state):
        """Test executing agents in parallel with timeout."""
        # Create slow agents
        slow_agent = MockAgent("slow_agent", execution_time=2.0)
        lifecycle_manager.agents["slow_agent"] = slow_agent
        
        agent_names = ["parser", "slow_agent"]
        
        # Execute with short timeout
        with pytest.raises(asyncio.TimeoutError):
            await lifecycle_manager.execute_agents_parallel(
                agent_names, sample_agent_state, timeout=0.5
            )

    def test_get_agent_metrics(self, lifecycle_manager, mock_agents):
        """Test getting agent metrics."""
        # Execute some agents first
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            sample_state = AgentState(
                session_id="test",
                structured_cv=StructuredCV(sections=[]),
                job_description_data=JobDescriptionData(
                    raw_text="test", company_name="test", role_title="test",
                    key_requirements=[], nice_to_have=[], company_info="test"
                ),
                current_item_id="test",
                items_to_process_queue=[],
                research_findings={},
                user_feedback=None,
                error_messages=[],
                processing_complete=False
            )
            
            # Execute agents
            loop.run_until_complete(lifecycle_manager.execute_agent("parser", sample_state))
            loop.run_until_complete(lifecycle_manager.execute_agent("content_writer", sample_state))
            
            # Get metrics
            metrics = lifecycle_manager.get_agent_metrics()
            
            assert isinstance(metrics, dict)
            assert "parser" in metrics
            assert "content_writer" in metrics
            
            # Check metric structure
            parser_metrics = metrics["parser"]
            assert "execution_count" in parser_metrics
            assert "last_execution_time" in parser_metrics
            assert "average_execution_time" in parser_metrics
            assert "success_rate" in parser_metrics
        finally:
            loop.close()

    def test_get_execution_history(self, lifecycle_manager):
        """Test getting execution history."""
        # Execute some agents first
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            sample_state = AgentState(
                session_id="test",
                structured_cv=StructuredCV(sections=[]),
                job_description_data=JobDescriptionData(
                    raw_text="test", company_name="test", role_title="test",
                    key_requirements=[], nice_to_have=[], company_info="test"
                ),
                current_item_id="test",
                items_to_process_queue=[],
                research_findings={},
                user_feedback=None,
                error_messages=[],
                processing_complete=False
            )
            
            # Execute agents
            loop.run_until_complete(lifecycle_manager.execute_agent("parser", sample_state))
            loop.run_until_complete(lifecycle_manager.execute_agent("content_writer", sample_state))
            
            # Get execution history
            history = lifecycle_manager.get_execution_history()
            
            assert isinstance(history, list)
            assert len(history) >= 2
            
            # Check history entry structure
            entry = history[0]
            assert "agent_name" in entry
            assert "execution_time" in entry
            assert "success" in entry
            assert "session_id" in entry
        finally:
            loop.close()

    def test_reset_agent_metrics(self, lifecycle_manager):
        """Test resetting agent metrics."""
        # Execute an agent first
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            sample_state = AgentState(
                session_id="test",
                structured_cv=StructuredCV(sections=[]),
                job_description_data=JobDescriptionData(
                    raw_text="test", company_name="test", role_title="test",
                    key_requirements=[], nice_to_have=[], company_info="test"
                ),
                current_item_id="test",
                items_to_process_queue=[],
                research_findings={},
                user_feedback=None,
                error_messages=[],
                processing_complete=False
            )
            
            # Execute agent
            loop.run_until_complete(lifecycle_manager.execute_agent("parser", sample_state))
            
            # Verify metrics exist
            metrics_before = lifecycle_manager.get_agent_metrics()
            assert len(metrics_before) > 0
            
            # Reset metrics
            lifecycle_manager.reset_agent_metrics()
            
            # Verify metrics are reset
            metrics_after = lifecycle_manager.get_agent_metrics()
            assert len(metrics_after) == 0 or all(
                metrics["execution_count"] == 0 for metrics in metrics_after.values()
            )
        finally:
            loop.close()

    @pytest.mark.asyncio
    async def test_agent_health_check(self, lifecycle_manager, sample_agent_state):
        """Test agent health check functionality."""
        # Perform health check
        health_status = await lifecycle_manager.perform_health_check()
        
        assert isinstance(health_status, dict)
        
        # Check that all agents are included
        for agent_name in lifecycle_manager.agents.keys():
            assert agent_name in health_status
            
            agent_health = health_status[agent_name]
            assert "status" in agent_health
            assert "last_check" in agent_health
            assert agent_health["status"] in ["healthy", "unhealthy", "unknown"]

    @pytest.mark.asyncio
    async def test_agent_resource_monitoring(self, lifecycle_manager, sample_agent_state):
        """Test agent resource monitoring."""
        # Execute an agent
        await lifecycle_manager.execute_agent("parser", sample_agent_state)
        
        # Get resource usage
        resource_usage = lifecycle_manager.get_resource_usage()
        
        assert isinstance(resource_usage, dict)
        assert "memory_usage" in resource_usage
        assert "cpu_usage" in resource_usage
        assert "execution_time" in resource_usage

    @pytest.mark.asyncio
    async def test_agent_lifecycle_events(self, lifecycle_manager, sample_agent_state):
        """Test agent lifecycle event handling."""
        events = []
        
        def event_handler(event_type: str, agent_name: str, **kwargs):
            events.append({
                "type": event_type,
                "agent": agent_name,
                "timestamp": datetime.now(),
                **kwargs
            })
        
        # Register event handler
        lifecycle_manager.register_event_handler(event_handler)
        
        # Execute agent
        await lifecycle_manager.execute_agent("parser", sample_agent_state)
        
        # Check that events were recorded
        assert len(events) >= 2  # At least start and end events
        
        event_types = [event["type"] for event in events]
        assert "agent_start" in event_types
        assert "agent_end" in event_types

    @pytest.mark.asyncio
    async def test_agent_retry_mechanism(self, lifecycle_manager, sample_agent_state):
        """Test agent retry mechanism on failure."""
        # Create an agent that fails initially but succeeds on retry
        class RetryableAgent(MockAgent):
            def __init__(self, name: str):
                super().__init__(name)
                self.attempt_count = 0
            
            async def _process_single_item(self, cv_data, job_data, item_id, research_findings):
                self.attempt_count += 1
                if self.attempt_count < 3:  # Fail first 2 attempts
                    raise AgentExecutionError(f"Attempt {self.attempt_count} failed")
                return await super()._process_single_item(cv_data, job_data, item_id, research_findings)
        
        retryable_agent = RetryableAgent("retryable_agent")
        lifecycle_manager.agents["retryable_agent"] = retryable_agent
        
        # Execute with retry
        result_state = await lifecycle_manager.execute_agent_with_retry(
            "retryable_agent", sample_agent_state, max_retries=3
        )
        
        assert result_state is not None
        assert len(result_state.error_messages) == 0  # Should succeed after retries
        assert retryable_agent.attempt_count == 3

    @pytest.mark.asyncio
    async def test_agent_circuit_breaker(self, lifecycle_manager, sample_agent_state):
        """Test agent circuit breaker functionality."""
        # Create a consistently failing agent
        failing_agent = MockAgent("failing_agent", should_fail=True)
        lifecycle_manager.agents["failing_agent"] = failing_agent
        
        # Execute multiple times to trigger circuit breaker
        for _ in range(5):
            try:
                await lifecycle_manager.execute_agent("failing_agent", sample_agent_state)
            except AgentExecutionError:
                pass
        
        # Check circuit breaker status
        circuit_status = lifecycle_manager.get_circuit_breaker_status("failing_agent")
        
        assert circuit_status is not None
        assert "state" in circuit_status
        assert "failure_count" in circuit_status
        assert circuit_status["failure_count"] >= 5

    def test_agent_registration_and_deregistration(self, lifecycle_manager):
        """Test dynamic agent registration and deregistration."""
        # Register new agent
        new_agent = MockAgent("new_agent")
        lifecycle_manager.register_agent("new_agent", new_agent)
        
        assert "new_agent" in lifecycle_manager.agents
        assert lifecycle_manager.agents["new_agent"] == new_agent
        
        # Deregister agent
        lifecycle_manager.deregister_agent("new_agent")
        
        assert "new_agent" not in lifecycle_manager.agents

    def test_agent_configuration_update(self, lifecycle_manager):
        """Test updating agent configuration."""
        agent_name = "parser"
        
        # Update agent configuration
        new_config = {
            "timeout": 30,
            "max_retries": 5,
            "circuit_breaker_threshold": 10
        }
        
        lifecycle_manager.update_agent_config(agent_name, new_config)
        
        # Verify configuration is updated
        config = lifecycle_manager.get_agent_config(agent_name)
        assert config["timeout"] == 30
        assert config["max_retries"] == 5
        assert config["circuit_breaker_threshold"] == 10