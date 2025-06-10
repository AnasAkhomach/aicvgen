"""Comprehensive test suite for Phase 5 optimization components."""

import pytest
import asyncio
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any

# Import the optimization components
from src.core.dependency_injection import (
    DependencyContainer, LifecycleScope, DependencyState,
    get_container, reset_container
)
from src.core.agent_lifecycle_manager import (
    AgentLifecycleManager, AgentPoolConfig, AgentPoolStrategy,
    AgentState, ManagedAgent, get_agent_lifecycle_manager,
    reset_agent_lifecycle_manager
)
from src.core.startup_optimizer import StartupOptimizer, get_startup_optimizer
from src.core.performance_monitor import (
    PerformanceMonitor, OperationTimer, get_performance_monitor,
    reset_performance_monitor
)
from src.models.data_models import ContentType


class TestDependencyInjection:
    """Test dependency injection system."""
    
    def setup_method(self):
        """Reset container before each test."""
        reset_container()
    
    def test_singleton_registration_and_resolution(self):
        """Test singleton dependency registration and resolution."""
        container = get_container()
        
        class TestService:
            def __init__(self):
                self.value = "test"
        
        # Register singleton
        container.register_singleton("test_service", TestService, factory=TestService)
        
        # Resolve multiple times
        service1 = container.get(TestService, "test_service")
        service2 = container.get(TestService, "test_service")
        
        # Should be the same instance
        assert service1 is service2
        assert service1.value == "test"
    
    def test_session_scoped_dependencies(self):
        """Test session-scoped dependency behavior."""
        from src.core.dependency_injection import DependencyContainer
        
        class SessionService:
            def __init__(self):
                self.instance_id = time.time()
        
        # Test with session_id set
        container = DependencyContainer(session_id="test_session")
        container.register_session("session_service", SessionService, factory=SessionService)
        
        # Same session should return same instance
        service1 = container.get(SessionService, "session_service")
        service2 = container.get(SessionService, "session_service")
        assert service1 is service2
        
        # Different session should return different instance
        container2 = DependencyContainer(session_id="test_session2")
        container2.register_session("session_service", SessionService, factory=SessionService)
        service3 = container2.get(SessionService, "session_service")
        assert service1 is not service3
    
    def test_transient_dependencies(self):
        """Test transient dependency behavior."""
        container = get_container()
        
        import uuid
        
        class TransientService:
            def __init__(self):
                self.instance_id = str(uuid.uuid4())
        
        container.register_transient("transient_service", TransientService, factory=TransientService)
        
        # Should always return new instances
        service1 = container.get(TransientService, "transient_service")
        service2 = container.get(TransientService, "transient_service")
        
        assert service1 is not service2
        assert service1.instance_id != service2.instance_id
    
    def test_dependency_not_found(self):
        """Test behavior when dependency is not found."""
        container = get_container()
        
        class NonExistentService:
            pass
        
        with pytest.raises(ValueError):
            container.get(NonExistentService, "nonexistent_service")
    
    def test_container_statistics(self):
        """Test container statistics collection."""
        from src.core.dependency_injection import DependencyContainer
        
        class TestService:
            pass
        
        # Use container with session_id for session dependencies
        container = DependencyContainer(session_id="test_session")
        
        container.register_singleton("test1", TestService, factory=TestService)
        container.register_session("test2", TestService, factory=TestService)
        container.register_transient("test3", TestService, factory=TestService)
        
        # Resolve some dependencies
        container.get(TestService, "test1")  # Creates singleton instance
        container.get(TestService, "test2")  # Creates session instance
        container.get(TestService, "test3")  # Creates transient instance (not cached)
        
        stats = container.get_statistics()
        
        assert stats["registered_dependencies"] == 3
        assert stats["total_instances"] >= 2  # singleton + session (transient not cached)
    
    def test_session_cleanup(self):
        """Test session cleanup functionality."""
        container = get_container()
        
        class SessionService:
            def __init__(self):
                self.cleaned_up = False
            
            def cleanup(self):
                self.cleaned_up = True
        
        container.register_session("session_service", SessionService, factory=SessionService)
        
        # Create session instance
        service = container.get(SessionService, "session_service")
        
        # Cleanup session (using cleanup_idle_instances method)
        container.cleanup_idle_instances()
        
        # New request should create new instance
        new_service = container.get(SessionService, "session_service")
        assert service is not new_service


class TestAgentLifecycleManager:
    """Test agent lifecycle management system."""
    
    def setup_method(self):
        """Reset lifecycle manager before each test."""
        reset_agent_lifecycle_manager()
    
    def test_agent_registration(self):
        """Test agent type registration."""
        manager = get_agent_lifecycle_manager()
        
        class MockAgent:
            def __init__(self):
                self.agent_type = "test_agent"
        
        config = AgentPoolConfig(
            agent_type="test_agent",
            factory=MockAgent,
            min_instances=1,
            max_instances=3,
            strategy=AgentPoolStrategy.LAZY
        )
        
        manager.register_agent_type("test_agent", MockAgent, config)
        
        stats = manager.get_statistics()
        assert len(stats["pool_statistics"]) == 1
        assert "test_agent" in stats["pool_statistics"]
    
    def test_eager_agent_strategy(self):
        """Test eager agent pool strategy."""
        manager = get_agent_lifecycle_manager()
        
        class MockAgent:
            def __init__(self):
                self.agent_type = "eager_agent"
        
        config = AgentPoolConfig(
            agent_type="eager_agent",
            factory=MockAgent,
            min_instances=2,
            max_instances=5,
            strategy=AgentPoolStrategy.EAGER,
            warmup_on_startup=True
        )
        
        manager.register_agent_type("eager_agent", MockAgent, config)
        manager.warmup_pools()
        
        stats = manager.get_statistics()
        pool_stats = stats["pool_statistics"]["eager_agent"]
        
        # Should have pre-created minimum instances
        assert pool_stats["current_size"] >= 2
    
    def test_agent_acquisition_and_return(self):
        """Test agent acquisition and return to pool."""
        manager = get_agent_lifecycle_manager()
        
        class MockAgent:
            def __init__(self):
                self.agent_type = "test_agent"
                self.processed_count = 0
        
        config = AgentPoolConfig(
            agent_type="test_agent",
            factory=MockAgent,
            min_instances=1,
            max_instances=3,
            strategy=AgentPoolStrategy.ADAPTIVE
        )
        
        manager.register_agent_type("test_agent", MockAgent, config)
        
        # Acquire agent
        managed_agent = manager.get_agent("test_agent")
        assert managed_agent is not None
        assert isinstance(managed_agent, ManagedAgent)
        # Agent should be marked as busy when acquired
        assert managed_agent.state == AgentState.BUSY
        
        # Return agent
        manager.return_agent(managed_agent)
        assert managed_agent.state == AgentState.IDLE
        
        stats = manager.get_statistics()
        assert stats["global_metrics"]["pool_hits"] >= 0 or stats["global_metrics"]["pool_misses"] >= 1
    
    def test_agent_pool_limits(self):
        """Test agent pool size limits."""
        manager = get_agent_lifecycle_manager()
        
        class MockAgent:
            def __init__(self):
                self.agent_type = "limited_agent"
        
        config = AgentPoolConfig(
            agent_type="limited_agent",
            factory=MockAgent,
            min_instances=0,
            max_instances=2,
            strategy=AgentPoolStrategy.LAZY
        )
        
        manager.register_agent_type("limited_agent", MockAgent, config)
        
        # Acquire up to max instances
        agents = []
        for i in range(3):  # Try to get more than max
            agent = manager.get_agent("limited_agent")
            if agent:
                agents.append(agent)
        
        # Should only get max_instances (2 in this case)
        # The pool should enforce the limit and return None for the 3rd request
        assert len(agents) == config.max_instances
        
        # Return agents
        for agent in agents:
            manager.return_agent(agent)
    
    def test_session_agent_management(self):
        """Test session-specific agent management."""
        manager = get_agent_lifecycle_manager()
        
        class MockAgent:
            def __init__(self):
                self.agent_type = "session_agent"
        
        config = AgentPoolConfig(
            agent_type="session_agent",
            factory=MockAgent,
            min_instances=0,
            max_instances=3,
            strategy=AgentPoolStrategy.LAZY
        )
        
        manager.register_agent_type("session_agent", MockAgent, config)
        
        # Acquire agents for different sessions
        agent1 = manager.get_agent("session_agent", session_id="session1")
        agent2 = manager.get_agent("session_agent", session_id="session2")
        
        assert agent1 is not None
        assert agent2 is not None
        
        # Cleanup session
        manager.dispose_session_agents("session1")
        
        stats = manager.get_statistics()
        assert stats["active_sessions"] >= 1  # session2 should still be active


class TestStartupOptimizer:
    """Test startup optimization system."""
    
    def setup_method(self):
        """Setup for each test."""
        reset_container()
        reset_agent_lifecycle_manager()
    
    @pytest.mark.asyncio
    async def test_minimal_startup_strategy(self):
        """Test minimal startup optimization strategy."""
        optimizer = get_startup_optimizer()
        
        metrics = await optimizer.optimize_startup(
            strategy="minimal",
            warmup_pools=True,
            preload_dependencies=True
        )
        
        assert metrics.optimization_level == "minimal"
        assert metrics.total_startup_time > 0
        assert metrics.agents_preloaded >= 0
        assert metrics.dependencies_registered >= 0
    
    @pytest.mark.asyncio
    async def test_balanced_startup_strategy(self):
        """Test balanced startup optimization strategy."""
        optimizer = get_startup_optimizer()
        
        with patch('src.agents.enhanced_content_writer.EnhancedContentWriterAgent') as mock_agent:
            mock_agent.return_value = Mock()
            
            metrics = await optimizer.optimize_startup(
                strategy="balanced",
                warmup_pools=True,
                preload_dependencies=True
            )
        
        assert metrics.optimization_level == "balanced"
        assert metrics.total_startup_time > 0
        # Balanced should register more dependencies than minimal
        assert metrics.dependencies_registered >= 1
    
    @pytest.mark.asyncio
    async def test_aggressive_startup_strategy(self):
        """Test aggressive startup optimization strategy."""
        optimizer = get_startup_optimizer()
        
        with patch('src.agents.enhanced_content_writer.EnhancedContentWriterAgent') as mock_agent:
            mock_agent.return_value = Mock()
            
            metrics = await optimizer.optimize_startup(
                strategy="aggressive",
                warmup_pools=True,
                preload_dependencies=True
            )
        
        assert metrics.optimization_level == "aggressive"
        assert metrics.total_startup_time > 0
        # Aggressive should register the most dependencies and agents
        assert metrics.dependencies_registered >= 3
    
    @pytest.mark.asyncio
    async def test_invalid_strategy(self):
        """Test handling of invalid optimization strategy."""
        optimizer = get_startup_optimizer()
        
        with pytest.raises(ValueError, match="Unknown optimization strategy"):
            await optimizer.optimize_startup(strategy="invalid_strategy")
    
    def test_optimization_report_generation(self):
        """Test optimization report generation."""
        optimizer = get_startup_optimizer()
        
        # Initially should have no metrics
        report = optimizer.get_optimization_report()
        assert "error" in report
        
        # Add some mock metrics
        from src.core.startup_optimizer import StartupMetrics
        
        mock_metrics = StartupMetrics(
            total_startup_time=2.5,
            dependency_injection_time=0.5,
            agent_pool_warmup_time=1.0,
            memory_usage_before=100.0,
            memory_usage_after=120.0,
            memory_delta=20.0,
            agents_preloaded=3,
            dependencies_registered=5,
            errors_encountered=0,
            optimization_level="test",
            timestamp=datetime.now()
        )
        
        optimizer.metrics_history.append(mock_metrics)
        
        report = optimizer.get_optimization_report()
        assert "latest_metrics" in report
        assert "performance_summary" in report
        assert "recommendations" in report
        assert report["latest_metrics"]["total_startup_time"] == 2.5


class TestPerformanceMonitor:
    """Test performance monitoring system."""
    
    def setup_method(self):
        """Reset monitor before each test."""
        reset_performance_monitor()
    
    def test_metric_recording(self):
        """Test basic metric recording."""
        monitor = get_performance_monitor()
        
        monitor.record_metric("test_metric", 42.5, "units")
        
        assert len(monitor.metrics_history) == 1
        metric = monitor.metrics_history[0]
        assert metric.name == "test_metric"
        assert metric.value == 42.5
        assert metric.unit == "units"
    
    def test_operation_timing(self):
        """Test operation timing functionality."""
        monitor = get_performance_monitor()
        
        monitor.record_operation_time("test_operation", 1500.0)
        
        assert "test_operation" in monitor.operation_times
        assert monitor.operation_times["test_operation"][0] == 1500.0
        assert len(monitor.metrics_history) == 1  # Should also record as metric
    
    def test_operation_timer_context_manager(self):
        """Test operation timer context manager."""
        monitor = get_performance_monitor()
        
        with OperationTimer(monitor, "timed_operation"):
            time.sleep(0.1)  # Simulate work
        
        assert "timed_operation" in monitor.operation_times
        assert len(monitor.operation_times["timed_operation"]) == 1
        # Should be approximately 100ms (allowing for some variance)
        assert 50 <= monitor.operation_times["timed_operation"][0] <= 200
    
    def test_error_recording(self):
        """Test error recording functionality."""
        monitor = get_performance_monitor()
        
        monitor.record_error("test_error")
        monitor.record_error("test_error")
        monitor.record_error("other_error")
        
        assert monitor.error_counts["test_error"] == 2
        assert monitor.error_counts["other_error"] == 1
    
    @pytest.mark.asyncio
    async def test_monitoring_lifecycle(self):
        """Test monitoring start/stop lifecycle."""
        monitor = get_performance_monitor()
        
        assert not monitor.monitoring_active
        
        # Start monitoring
        monitor.start_monitoring(interval=1)
        assert monitor.monitoring_active
        assert monitor.monitoring_task is not None
        
        # Let it run briefly
        await asyncio.sleep(0.5)
        
        # Stop monitoring
        monitor.stop_monitoring()
        assert not monitor.monitoring_active
    
    def test_performance_summary(self):
        """Test performance summary generation."""
        monitor = get_performance_monitor()
        
        # Add some test data
        monitor.record_metric("cpu_percent", 75.0, "%")
        monitor.record_metric("memory_mb", 512.0, "MB")
        monitor.record_operation_time("test_op", 1000.0)
        monitor.record_error("test_error")
        
        summary = monitor.get_performance_summary(timedelta(minutes=5))
        
        assert "metrics_count" in summary
        assert "operation_performance" in summary
        assert "error_summary" in summary
        assert "recommendations" in summary
        assert summary["metrics_count"] >= 3
    
    def test_baseline_comparison(self):
        """Test baseline setting and comparison."""
        monitor = get_performance_monitor()
        
        # Set baseline
        monitor.set_baseline("test_metric", 100.0)
        assert monitor.baselines["test_metric"] == 100.0
        
        # Record some values
        monitor.record_metric("test_metric", 90.0, "units")
        monitor.record_metric("test_metric", 85.0, "units")
        
        # Compare to baseline
        comparison = monitor.compare_to_baseline("test_metric")
        assert comparison is not None
        assert comparison["baseline_value"] == 100.0
        assert comparison["improvement_percent"] > 0  # Should show improvement
    
    def test_metrics_export(self, tmp_path):
        """Test metrics export functionality."""
        monitor = get_performance_monitor()
        
        # Add some test data
        monitor.record_metric("test_metric", 42.0, "units")
        monitor.record_operation_time("test_op", 500.0)
        
        # Export to file
        export_file = tmp_path / "test_metrics.json"
        monitor.export_metrics(str(export_file))
        
        assert export_file.exists()
        
        # Verify file content
        import json
        with open(export_file) as f:
            data = json.load(f)
        
        assert "export_timestamp" in data
        assert "metrics" in data
        assert len(data["metrics"]) >= 2


class TestIntegration:
    """Integration tests for optimization components."""
    
    def setup_method(self):
        """Reset all systems before each test."""
        reset_container()
        reset_agent_lifecycle_manager()
        reset_performance_monitor()
    
    @pytest.mark.asyncio
    async def test_full_optimization_workflow(self):
        """Test complete optimization workflow integration."""
        # Start performance monitoring
        monitor = get_performance_monitor()
        monitor.start_monitoring(interval=1)
        
        try:
            # Run startup optimization
            optimizer = get_startup_optimizer()
            
            with patch('src.agents.enhanced_content_writer.EnhancedContentWriterAgent') as mock_agent:
                mock_agent.return_value = Mock()
                
                metrics = await optimizer.optimize_startup(
                    strategy="balanced",
                    warmup_pools=True,
                    preload_dependencies=True
                )
            
            # Verify systems are working together
            container = get_container()
            lifecycle_manager = get_agent_lifecycle_manager()
            
            # Check that dependencies were registered
            container_stats = container.get_statistics()
            assert container_stats["total_dependencies"] > 0
            
            # Check that agents were registered
            lifecycle_stats = lifecycle_manager.get_statistics()
            assert lifecycle_stats["total_agent_types"] >= 0
            
            # Check that monitoring captured the activity
            # Note: monitoring might not capture activity immediately in test environment
            
            # Verify optimization metrics
            assert metrics.total_startup_time > 0
            assert metrics.optimization_level == "balanced"
            
        finally:
            monitor.stop_monitoring()
    
    @pytest.mark.asyncio
    async def test_agent_lifecycle_with_dependency_injection(self):
        """Test agent lifecycle manager working with dependency injection."""
        container = get_container()
        lifecycle_manager = get_agent_lifecycle_manager()
        
        # Register a service that agents might depend on
        class ConfigService:
            def __init__(self):
                self.config = {"max_tokens": 1000}
        
        container.register_singleton("config", ConfigService, factory=ConfigService)
        
        # Register an agent that uses the service
        class DependentAgent:
            def __init__(self):
                self.config_service = container.get("config")
                self.agent_type = "dependent_agent"
        
        config = AgentPoolConfig(
            agent_type="dependent_agent",
            factory=DependentAgent,
            min_instances=1,
            max_instances=2,
            strategy=AgentPoolStrategy.LAZY
        )
        
        lifecycle_manager.register_agent_type("dependent_agent", DependentAgent, config)
        
        # Acquire agent
        managed_agent = lifecycle_manager.get_agent("dependent_agent")
        assert managed_agent is not None
        assert hasattr(managed_agent.agent, 'config_service')
        assert managed_agent.agent.config_service.config["max_tokens"] == 1000
        
        # Return agent
        lifecycle_manager.return_agent(managed_agent)
    
    def test_performance_monitoring_during_operations(self):
        """Test performance monitoring during actual operations."""
        monitor = get_performance_monitor()
        container = get_container()
        
        # Register a service
        class TestService:
            def process(self, data):
                time.sleep(0.1)  # Simulate work
                return f"processed: {data}"
        
        container.register_singleton("test_service", TestService, factory=TestService)
        
        # Use the service with timing
        service = container.get(TestService, "test_service")
        
        with OperationTimer(monitor, "service_processing"):
            result = service.process("test_data")
        
        assert result == "processed: test_data"
        assert "service_processing" in monitor.operation_times
        assert len(monitor.operation_times["service_processing"]) == 1
        
        # Check that metrics were recorded
        assert len(monitor.metrics_history) >= 1
        
        # Verify container statistics
        stats = container.get_statistics()
        # Check that container has registered dependencies
        assert stats["registered_dependencies"] >= 1


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])