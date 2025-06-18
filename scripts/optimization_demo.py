#!/usr/bin/env python3
"""Demonstration script for Phase 5 optimization improvements.

This script demonstrates:
1. Agent lifecycle management with dependency injection
2. Startup performance optimization
3. Performance monitoring and analysis
4. Comparison of different optimization strategies
"""

import asyncio
import sys
import time
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.startup_optimizer import get_startup_optimizer
from core.performance_monitor import get_performance_monitor, time_operation
from core.dependency_injection import get_container, reset_container
from core.agent_lifecycle_manager import (
    get_agent_lifecycle_manager,
    reset_agent_lifecycle_manager,
)
from config.logging_config import get_structured_logger

logger = get_structured_logger(__name__)


class OptimizationDemo:
    """Demonstrates the optimization improvements."""

    def __init__(self):
        self.optimizer = get_startup_optimizer()
        self.monitor = get_performance_monitor()
        self.results = {}

    async def run_complete_demo(self) -> None:
        """Run the complete optimization demonstration."""
        print("\n" + "=" * 80)
        print("CV GENERATION SYSTEM - PHASE 5 OPTIMIZATION DEMO")
        print("=" * 80)

        try:
            # Start performance monitoring
            self.monitor.start_monitoring(interval=10)

            # Run startup optimization tests
            await self._demo_startup_optimization()

            # Demonstrate dependency injection
            await self._demo_dependency_injection()

            # Demonstrate agent lifecycle management
            await self._demo_agent_lifecycle()

            # Show performance analysis
            await self._demo_performance_analysis()

            # Generate final report
            await self._generate_final_report()

        except Exception as e:
            logger.error(f"Demo failed: {e}")
            print(f"\n❌ Demo failed: {e}")
        finally:
            self.monitor.stop_monitoring()

    async def _demo_startup_optimization(self) -> None:
        """Demonstrate startup optimization strategies."""
        print("\n🚀 STARTUP OPTIMIZATION DEMONSTRATION")
        print("-" * 50)

        strategies = ["minimal", "balanced", "aggressive", "development"]

        for strategy in strategies:
            print(f"\n📊 Testing {strategy.upper()} strategy...")

            # Reset systems for clean test
            reset_container()
            reset_agent_lifecycle_manager()

            # Run optimization
            with time_operation(f"startup_{strategy}"):
                metrics = await self.optimizer.optimize_startup(
                    strategy=strategy, warmup_pools=True, preload_dependencies=True
                )

            self.results[f"startup_{strategy}"] = metrics

            # Display results
            print(f"   ⏱️  Total time: {metrics.total_startup_time:.3f}s")
            print(f"   🔧 DI time: {metrics.dependency_injection_time:.3f}s")
            print(f"   🤖 Agent warmup: {metrics.agent_pool_warmup_time:.3f}s")
            print(f"   💾 Memory delta: {metrics.memory_delta:.1f}MB")
            print(f"   📦 Agents loaded: {metrics.agents_preloaded}")
            print(f"   🔗 Dependencies: {metrics.dependencies_registered}")

            # Brief pause between tests
            await asyncio.sleep(2)

        # Show optimization report
        report = self.optimizer.get_optimization_report()
        print("\n📈 OPTIMIZATION SUMMARY:")
        print(
            f"   Fastest startup: {report['performance_summary']['fastest_startup']:.3f}s"
        )
        print(
            f"   Slowest startup: {report['performance_summary']['slowest_startup']:.3f}s"
        )
        print(
            f"   Average startup: {report['performance_summary']['average_startup_time']:.3f}s"
        )

        print("\n💡 RECOMMENDATIONS:")
        for rec in report["recommendations"]:
            print(f"   • {rec}")

    async def _demo_dependency_injection(self) -> None:
        """Demonstrate dependency injection capabilities."""
        print("\n🔗 DEPENDENCY INJECTION DEMONSTRATION")
        print("-" * 50)

        container = get_container()

        # Register sample services with different scopes
        print("\n📝 Registering services with different scopes...")

        # Singleton service
        class ConfigService:
            def __init__(self):
                self.config = {"app_name": "CV Generator", "version": "2.0"}
                print("   🔧 ConfigService created (singleton)")

        # Session service
        class UserSessionService:
            def __init__(self):
                self.session_id = f"session_{int(time.time())}"
                print(f"   👤 UserSessionService created: {self.session_id}")

        # Transient service
        class RequestService:
            def __init__(self):
                self.request_id = f"req_{int(time.time() * 1000)}"
                print(f"   📨 RequestService created: {self.request_id}")

        # Register services
        from core.dependency_injection import LifecycleScope

        container.register_singleton("config", ConfigService, factory=ConfigService)
        container.register_session(
            "user_session", UserSessionService, factory=UserSessionService
        )
        container.register_transient("request", RequestService, factory=RequestService)

        print("\n🧪 Testing dependency resolution...")

        # Test singleton behavior
        config1 = container.get("config")
        config2 = container.get("config")
        print(f"   Singleton test: Same instance? {config1 is config2}")

        # Test session behavior
        session1 = container.get("user_session")
        session2 = container.get("user_session")
        print(f"   Session test: Same instance? {session1 is session2}")

        # Test transient behavior
        request1 = container.get("request")
        request2 = container.get("request")
        print(f"   Transient test: Different instances? {request1 is not request2}")

        # Show container statistics
        stats = container.get_statistics()
        print("\n📊 Container Statistics:")
        print(f"   Total dependencies: {stats['total_dependencies']}")
        print(f"   Active instances: {stats['active_instances']}")
        print(f"   Total resolutions: {stats['total_resolutions']}")

        for scope, count in stats["instances_by_scope"].items():
            print(f"   {scope.value} instances: {count}")

    async def _demo_agent_lifecycle(self) -> None:
        """Demonstrate agent lifecycle management."""
        print("\n🤖 AGENT LIFECYCLE MANAGEMENT DEMONSTRATION")
        print("-" * 50)

        lifecycle_manager = get_agent_lifecycle_manager()

        # Register sample agents
        print("\n📝 Registering agent types...")

        from core.agent_lifecycle_manager import AgentPoolConfig, AgentPoolStrategy
        from models.data_models import ContentType

        # Mock agent for demonstration
        class MockAgent:
            def __init__(self, agent_type: str):
                self.agent_type = agent_type
                self.created_at = datetime.now()
                print(f"   🤖 {agent_type} agent created")

            async def process(self, content: str) -> str:
                await asyncio.sleep(0.1)  # Simulate processing
                return f"Processed by {self.agent_type}: {content[:50]}..."

        # Register different agent types
        agent_configs = [
            ("content_writer", AgentPoolStrategy.EAGER, 2, 4),
            ("cv_analyzer", AgentPoolStrategy.ADAPTIVE, 1, 3),
            ("quality_checker", AgentPoolStrategy.LAZY, 0, 2),
        ]

        for agent_type, strategy, min_inst, max_inst in agent_configs:
            lifecycle_manager.register_agent_type(
                agent_type,
                lambda at=agent_type: MockAgent(at),
                AgentPoolConfig(
                    agent_type=agent_type,
                    factory=lambda at=agent_type: MockAgent(at),
                    min_instances=min_inst,
                    max_instances=max_inst,
                    strategy=strategy,
                    content_types=[ContentType.EXECUTIVE_SUMMARY],
                    warmup_on_startup=(strategy == AgentPoolStrategy.EAGER),
                ),
            )

        # Warm up pools
        print("\n🔥 Warming up agent pools...")
        lifecycle_manager.warmup_pools()

        # Test agent acquisition and usage
        print("\n🧪 Testing agent acquisition...")

        agents_in_use = []

        for i in range(5):
            with time_operation("agent_acquisition"):
                managed_agent = lifecycle_manager.get_agent(
                    "content_writer", content_type=ContentType.EXECUTIVE_SUMMARY
                )

            if managed_agent:
                agents_in_use.append(managed_agent)
                print(f"   📦 Acquired agent {i+1}: {managed_agent.agent.agent_type}")

                # Simulate some work
                result = await managed_agent.agent.process(f"Sample content {i+1}")
                print(f"   ✅ Result: {result[:60]}...")

        # Return agents to pool
        print("\n🔄 Returning agents to pool...")
        for managed_agent in agents_in_use:
            lifecycle_manager.return_agent(managed_agent)
            print(f"   ↩️  Returned {managed_agent.agent.agent_type} agent")

        # Show lifecycle statistics
        stats = lifecycle_manager.get_statistics()
        print("\n📊 Lifecycle Manager Statistics:")
        print(f"   Total agent types: {stats['total_agent_types']}")
        print(f"   Total acquisitions: {stats['total_acquisitions']}")
        print(f"   Active sessions: {stats['active_sessions']}")

        for agent_type, agent_stats in stats["agent_pools"].items():
            print(f"   {agent_type}:")
            print(f"     Active: {agent_stats['active_instances']}")
            print(f"     Available: {agent_stats['available_instances']}")
            print(f"     Total created: {agent_stats['total_created']}")

    async def _demo_performance_analysis(self) -> None:
        """Demonstrate performance monitoring and analysis."""
        print("\n📈 PERFORMANCE ANALYSIS DEMONSTRATION")
        print("-" * 50)

        # Record some sample metrics
        print("\n📊 Recording performance metrics...")

        # Simulate various operations with timing
        operations = [
            ("cv_parsing", 0.5, 2.0),
            ("content_generation", 1.0, 3.0),
            ("quality_check", 0.3, 1.0),
            ("file_processing", 0.8, 2.5),
        ]

        for operation, min_time, max_time in operations:
            for _ in range(10):
                # Simulate operation with random duration
                import random

                duration = random.uniform(min_time, max_time) * 1000  # Convert to ms

                self.monitor.record_operation_time(operation, duration)
                await asyncio.sleep(0.1)  # Brief pause

        # Record some system metrics
        self.monitor.record_metric("custom_metric_1", 85.5, "%")
        self.monitor.record_metric("custom_metric_2", 1024, "MB")

        # Get performance summary
        summary = self.monitor.get_performance_summary(timedelta(minutes=5))

        print("\n📋 Performance Summary:")

        if "system_performance" in summary:
            sys_perf = summary["system_performance"]
            if "cpu_usage" in sys_perf:
                print(f"   CPU Usage: {sys_perf['cpu_usage']['current']:.1f}%")
            if "memory_usage_mb" in sys_perf:
                print(
                    f"   Memory Usage: {sys_perf['memory_usage_mb']['current']:.1f}MB"
                )

        if "operation_performance" in summary:
            print("\n   Operation Performance:")
            for op, stats in summary["operation_performance"].items():
                print(
                    f"     {op}: {stats['average_time_ms']:.1f}ms avg ({stats['total_executions']} runs)"
                )

        if "recommendations" in summary:
            print("\n💡 Performance Recommendations:")
            for rec in summary["recommendations"]:
                print(f"   • {rec}")

        # Demonstrate baseline comparison
        print("\n🎯 Setting and comparing baselines...")

        # Set baselines for some operations
        self.monitor.set_baseline("operation_cv_parsing_time", 1500)  # 1.5 seconds
        self.monitor.set_baseline(
            "operation_content_generation_time", 2000
        )  # 2 seconds

        # Compare to baselines
        for operation in ["cv_parsing", "content_generation"]:
            comparison = self.monitor.compare_to_baseline(f"operation_{operation}_time")
            if comparison:
                improvement = comparison["improvement_percent"]
                if improvement > 0:
                    print(
                        f"   ✅ {operation}: {improvement:.1f}% improvement vs baseline"
                    )
                else:
                    print(
                        f"   ⚠️  {operation}: {abs(improvement):.1f}% slower than baseline"
                    )

    async def _generate_final_report(self) -> None:
        """Generate and display the final optimization report."""
        print("\n📄 FINAL OPTIMIZATION REPORT")
        print("=" * 50)

        # Startup optimization summary
        if hasattr(self, "results") and self.results:
            print("\n🚀 Startup Optimization Results:")

            fastest_strategy = None
            fastest_time = float("inf")

            for strategy_key, metrics in self.results.items():
                if strategy_key.startswith("startup_"):
                    strategy = strategy_key.replace("startup_", "")
                    time_taken = metrics.total_startup_time

                    print(f"   {strategy.capitalize():>12}: {time_taken:.3f}s")

                    if time_taken < fastest_time:
                        fastest_time = time_taken
                        fastest_strategy = strategy

            if fastest_strategy:
                print(
                    f"\n   🏆 Fastest strategy: {fastest_strategy.upper()} ({fastest_time:.3f}s)"
                )

        # System improvements summary
        print("\n🔧 System Improvements Implemented:")
        improvements = [
            "✅ Dependency injection system with lifecycle management",
            "✅ Agent pooling with configurable strategies (eager, adaptive, lazy)",
            "✅ Startup optimization with multiple strategies",
            "✅ Performance monitoring and analysis",
            "✅ Resource cleanup and memory management",
            "✅ Error handling and recovery mechanisms",
            "✅ Metrics collection and baseline comparison",
        ]

        for improvement in improvements:
            print(f"   {improvement}")

        # Performance metrics
        container_stats = get_container().get_statistics()
        lifecycle_stats = get_agent_lifecycle_manager().get_statistics()

        print("\n📊 Current System State:")
        print(f"   Dependencies registered: {container_stats['total_dependencies']}")
        print(f"   Active instances: {container_stats['active_instances']}")
        print(f"   Agent types registered: {lifecycle_stats['total_agent_types']}")
        print(f"   Total agent acquisitions: {lifecycle_stats['total_acquisitions']}")

        # Export performance data
        try:
            export_path = (
                Path(__file__).parent.parent
                / "logs"
                / f"optimization_demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            export_path.parent.mkdir(exist_ok=True)

            self.monitor.export_metrics(str(export_path))
            print(f"\n💾 Performance data exported to: {export_path}")
        except Exception as e:
            print(f"\n⚠️  Failed to export performance data: {e}")

        print("\n🎉 OPTIMIZATION DEMO COMPLETED SUCCESSFULLY!")
        print("\n" + "=" * 80)


async def main():
    """Main demo function."""
    demo = OptimizationDemo()
    await demo.run_complete_demo()


if __name__ == "__main__":
    # Run the demo
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Demo failed with error: {e}")
        import traceback

        traceback.print_exc()
