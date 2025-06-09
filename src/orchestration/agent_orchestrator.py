"""Agent orchestration system for coordinating multiple agents."""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import uuid

from ..models.data_models import ContentType, ProcessingStatus
from ..config.logging_config import get_structured_logger
from ..services.error_recovery import get_error_recovery_service
from ..services.progress_tracker import get_progress_tracker
from ..services.session_manager import get_session_manager
from ..agents.agent_base import EnhancedAgentBase, AgentExecutionContext, AgentResult
from ..agents.enhanced_content_writer import EnhancedContentWriterAgent
from ..agents.specialized_agents import (
    CVAnalysisAgent, ContentOptimizationAgent, QualityAssuranceAgent,
    get_agent, list_available_agents
)

logger = get_structured_logger(__name__)


class OrchestrationStrategy(Enum):
    """Different orchestration strategies."""
    SEQUENTIAL = "sequential"  # Execute agents one by one
    PARALLEL = "parallel"  # Execute compatible agents in parallel
    PIPELINE = "pipeline"  # Execute in a predefined pipeline
    ADAPTIVE = "adaptive"  # Dynamically choose strategy based on context


class AgentPriority(Enum):
    """Agent execution priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class AgentTask:
    """Represents a task to be executed by an agent."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_type: str = ""
    agent_instance: Optional[EnhancedAgentBase] = None
    context: Optional[AgentExecutionContext] = None
    priority: AgentPriority = AgentPriority.NORMAL
    dependencies: List[str] = field(default_factory=list)  # Task IDs this depends on
    timeout: Optional[timedelta] = None
    retry_count: int = 0
    max_retries: int = 3
    status: ProcessingStatus = ProcessingStatus.PENDING
    result: Optional[AgentResult] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OrchestrationPlan:
    """Represents an orchestration execution plan."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    strategy: OrchestrationStrategy = OrchestrationStrategy.SEQUENTIAL
    tasks: List[AgentTask] = field(default_factory=list)
    max_parallel_tasks: int = 3
    timeout: Optional[timedelta] = None
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OrchestrationResult:
    """Result of orchestration execution."""
    plan_id: str
    success: bool
    completed_tasks: List[AgentTask]
    failed_tasks: List[AgentTask]
    total_execution_time: timedelta
    performance_stats: Dict[str, Any]
    error_summary: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def task_results(self) -> List[Dict[str, Any]]:
        """Extract results from completed tasks."""
        from src.core.main import enum_to_value, EnumEncoder
        import json
        import logging
        
        logger = logging.getLogger(__name__)
        results = []
        
        for task in self.completed_tasks:
            if task.result and task.result.success:
                try:
                    # Convert AgentResult to dictionary format with comprehensive enum handling
                    task_result = {
                        "task_id": task.id,
                        "agent_type": task.agent_type,
                        "content": enum_to_value(task.result.output_data),
                        "confidence_score": task.result.confidence_score,
                        "processing_time": task.result.processing_time,
                        "metadata": enum_to_value(task.result.metadata)
                    }
                    
                    # Test serialization to catch any remaining enum issues
                    json.dumps(task_result, cls=EnumEncoder)
                    results.append(task_result)
                    
                except Exception as e:
                    logger.error(f"Error processing task result for task {task.id}: {e}")
                    logger.error(f"Task output_data type: {type(task.result.output_data)}")
                    logger.error(f"Task metadata type: {type(task.result.metadata)}")
                    # Add a simplified version without problematic data
                    task_result = {
                        "task_id": task.id,
                        "agent_type": task.agent_type,
                        "content": str(task.result.output_data),
                        "confidence_score": task.result.confidence_score,
                        "processing_time": task.result.processing_time,
                        "metadata": {"error": "Serialization failed", "original_error": str(e)}
                    }
                    results.append(task_result)
                    
        return results


class AgentOrchestrator:
    """Orchestrates multiple agents for complex CV generation workflows."""
    
    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id or str(uuid.uuid4())
        self.session_manager = get_session_manager()
        self.error_recovery = get_error_recovery_service()
        self.progress_tracker = get_progress_tracker()
        
        # Agent registry
        self.agents: Dict[str, EnhancedAgentBase] = {}
        self.agent_factories: Dict[str, Callable[[], EnhancedAgentBase]] = {}
        
        # Execution state
        self.active_plans: Dict[str, OrchestrationPlan] = {}
        self.execution_history: List[OrchestrationResult] = []
        
        # Performance tracking
        self.stats = {
            "plans_executed": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "average_execution_time": 0.0,
            "agent_usage_count": {},
            "error_count_by_type": {}
        }
        
        # Initialize with default agents
        self._register_default_agents()
        
        logger.info(
            "Agent orchestrator initialized",
            session_id=self.session_id,
            available_agents=list(self.agents.keys())
        )
    
    def _register_default_agents(self):
        """Register default agents and their factories."""
        # Register agent factories
        self.agent_factories.update({
            "content_writer": lambda: EnhancedContentWriterAgent(),
            "cv_analysis": lambda: get_agent("cv_analysis"),
            "content_optimization": lambda: get_agent("content_optimization"),
            "quality_assurance": lambda: get_agent("quality_assurance")
        })
        
        # Pre-instantiate commonly used agents
        self.agents["content_writer"] = self.agent_factories["content_writer"]()
    
    def register_agent(
        self,
        agent_type: str,
        agent_factory: Callable[[], EnhancedAgentBase],
        preinstantiate: bool = False
    ):
        """Register a new agent type."""
        self.agent_factories[agent_type] = agent_factory
        
        if preinstantiate:
            self.agents[agent_type] = agent_factory()
        
        logger.info("Agent registered", agent_type=agent_type, preinstantiated=preinstantiate)
    
    def get_agent(self, agent_type: str) -> Optional[EnhancedAgentBase]:
        """Get an agent instance, creating it if necessary."""
        if agent_type not in self.agents:
            if agent_type in self.agent_factories:
                self.agents[agent_type] = self.agent_factories[agent_type]()
            else:
                logger.error("Unknown agent type", agent_type=agent_type)
                return None
        
        return self.agents[agent_type]
    
    def create_task(
        self,
        agent_type: str,
        context: AgentExecutionContext,
        priority: AgentPriority = AgentPriority.NORMAL,
        dependencies: Optional[List[str]] = None,
        timeout: Optional[timedelta] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AgentTask:
        """Create a new agent task."""
        agent = self.get_agent(agent_type)
        if not agent:
            raise ValueError(f"Cannot create task for unknown agent type: {agent_type}")
        
        task = AgentTask(
            agent_type=agent_type,
            agent_instance=agent,
            context=context,
            priority=priority,
            dependencies=dependencies or [],
            timeout=timeout,
            metadata=metadata or {}
        )
        
        logger.debug(
            "Agent task created",
            task_id=task.id,
            agent_type=agent_type,
            priority=priority.name
        )
        
        return task
    
    def create_plan(
        self,
        strategy: OrchestrationStrategy = OrchestrationStrategy.SEQUENTIAL,
        tasks: Optional[List[AgentTask]] = None,
        max_parallel_tasks: int = 3,
        timeout: Optional[timedelta] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> OrchestrationPlan:
        """Create an orchestration plan."""
        plan = OrchestrationPlan(
            strategy=strategy,
            tasks=tasks or [],
            max_parallel_tasks=max_parallel_tasks,
            timeout=timeout,
            metadata=metadata or {}
        )
        
        self.active_plans[plan.id] = plan
        
        logger.info(
            "Orchestration plan created",
            plan_id=plan.id,
            strategy=strategy.value,
            task_count=len(plan.tasks)
        )
        
        return plan
    
    async def execute_plan(self, plan: OrchestrationPlan) -> OrchestrationResult:
        """Execute an orchestration plan."""
        start_time = datetime.now()
        
        try:
            # Update progress tracking
            from src.models.data_models import CVGenerationState, WorkflowStage
            state = CVGenerationState(
                session_id=self.session_id,
                current_stage=WorkflowStage.INITIALIZATION
            )
            self.progress_tracker.start_tracking(self.session_id, state)
            progress_id = f"orchestration_plan_{plan.id}"
            
            logger.info(
                "Starting orchestration plan execution",
                plan_id=plan.id,
                strategy=plan.strategy.value,
                task_count=len(plan.tasks)
            )
            
            # Execute based on strategy
            if plan.strategy == OrchestrationStrategy.SEQUENTIAL:
                completed_tasks, failed_tasks = await self._execute_sequential(plan, progress_id)
            elif plan.strategy == OrchestrationStrategy.PARALLEL:
                completed_tasks, failed_tasks = await self._execute_parallel(plan, progress_id)
            elif plan.strategy == OrchestrationStrategy.PIPELINE:
                completed_tasks, failed_tasks = await self._execute_pipeline(plan, progress_id)
            elif plan.strategy == OrchestrationStrategy.ADAPTIVE:
                completed_tasks, failed_tasks = await self._execute_adaptive(plan, progress_id)
            else:
                raise ValueError(f"Unknown orchestration strategy: {plan.strategy}")
            
            # Calculate execution time
            execution_time = datetime.now() - start_time
            
            # Create result
            result = OrchestrationResult(
                plan_id=plan.id,
                success=len(failed_tasks) == 0,
                completed_tasks=completed_tasks,
                failed_tasks=failed_tasks,
                total_execution_time=execution_time,
                performance_stats=self._calculate_performance_stats(completed_tasks, failed_tasks),
                error_summary=self._create_error_summary(failed_tasks) if failed_tasks else None
            )
            
            # Update stats
            self._update_stats(result)
            
            # Store result
            self.execution_history.append(result)
            
            # Complete progress tracking
            self.progress_tracker.record_workflow_completed(
                self.session_id, 
                {
                    "plan_id": plan.id,
                    "success": result.success,
                    "completed_tasks": len(completed_tasks),
                    "failed_tasks": len(failed_tasks),
                    "execution_time": result.total_execution_time.total_seconds()
                }
            )
            
            # Clean up
            if plan.id in self.active_plans:
                del self.active_plans[plan.id]
            
            logger.info(
                "Orchestration plan execution completed",
                plan_id=plan.id,
                success=result.success,
                completed_tasks=len(completed_tasks),
                failed_tasks=len(failed_tasks),
                execution_time=execution_time.total_seconds()
            )
            
            return result
            
        except Exception as e:
            execution_time = datetime.now() - start_time
            
            logger.error(
                "Orchestration plan execution failed",
                plan_id=plan.id,
                error=str(e),
                execution_time=execution_time.total_seconds()
            )
            
            # Create failure result
            result = OrchestrationResult(
                plan_id=plan.id,
                success=False,
                completed_tasks=[],
                failed_tasks=plan.tasks,
                total_execution_time=execution_time,
                performance_stats={},
                error_summary=str(e)
            )
            
            self.execution_history.append(result)
            
            # Clean up
            if plan.id in self.active_plans:
                del self.active_plans[plan.id]
            
            raise
    
    async def _execute_sequential(self, plan: OrchestrationPlan, progress_id: str) -> tuple:
        """Execute tasks sequentially."""
        completed_tasks = []
        failed_tasks = []
        
        # Sort tasks by priority and dependencies
        sorted_tasks = self._sort_tasks_by_dependencies(plan.tasks)
        
        for task in sorted_tasks:
            try:
                # Check if dependencies are satisfied
                if not self._are_dependencies_satisfied(task, completed_tasks):
                    logger.warning(
                        "Task dependencies not satisfied, skipping",
                        task_id=task.id,
                        dependencies=task.dependencies
                    )
                    failed_tasks.append(task)
                    continue
                
                # Execute task
                result = await self._execute_task(task)
                
                if result.success:
                    completed_tasks.append(task)
                else:
                    failed_tasks.append(task)
                
                # Update progress - using record_item_completed instead
                self.progress_tracker.record_item_completed(
                    progress_id,
                    f"task_{task.id}",
                    ContentType.ANALYSIS,
                    result.processing_time if hasattr(result, 'processing_time') else 0.0
                )
                
            except Exception as e:
                logger.error(
                    "Task execution failed",
                    task_id=task.id,
                    agent_type=task.agent_type,
                    error=str(e)
                )
                task.error = str(e)
                task.status = ProcessingStatus.FAILED
                failed_tasks.append(task)
        
        return completed_tasks, failed_tasks
    
    async def _execute_parallel(self, plan: OrchestrationPlan, progress_id: str) -> tuple:
        """Execute compatible tasks in parallel."""
        completed_tasks = []
        failed_tasks = []
        
        # Group tasks by dependency levels
        task_groups = self._group_tasks_by_dependency_level(plan.tasks)
        
        for group in task_groups:
            # Execute tasks in this group in parallel
            semaphore = asyncio.Semaphore(plan.max_parallel_tasks)
            
            async def execute_with_semaphore(task):
                async with semaphore:
                    return await self._execute_task(task)
            
            # Create tasks for parallel execution
            parallel_tasks = [
                execute_with_semaphore(task) for task in group
                if self._are_dependencies_satisfied(task, completed_tasks)
            ]
            
            # Wait for all tasks in this group to complete
            results = await asyncio.gather(*parallel_tasks, return_exceptions=True)
            
            # Process results
            for i, result in enumerate(results):
                task = group[i]
                
                if isinstance(result, Exception):
                    logger.error(
                        "Parallel task execution failed",
                        task_id=task.id,
                        error=str(result)
                    )
                    task.error = str(result)
                    task.status = ProcessingStatus.FAILED
                    failed_tasks.append(task)
                elif result.success:
                    completed_tasks.append(task)
                else:
                    failed_tasks.append(task)
                
                # Update progress - using record_item_completed instead
                self.progress_tracker.record_item_completed(
                    progress_id,
                    f"task_{task.id}",
                    ContentType.ANALYSIS,
                    result.processing_time if hasattr(result, 'processing_time') else 0.0
                )
        
        return completed_tasks, failed_tasks
    
    async def _execute_pipeline(self, plan: OrchestrationPlan, progress_id: str) -> tuple:
        """Execute tasks in a pipeline fashion."""
        # Pipeline execution is similar to sequential but with data flow
        return await self._execute_sequential(plan, progress_id)
    
    async def _execute_adaptive(self, plan: OrchestrationPlan, progress_id: str) -> tuple:
        """Adaptively choose execution strategy based on task characteristics."""
        # Analyze tasks to determine best strategy
        if len(plan.tasks) <= 3:
            # Small number of tasks, use sequential
            return await self._execute_sequential(plan, progress_id)
        elif self._has_complex_dependencies(plan.tasks):
            # Complex dependencies, use sequential
            return await self._execute_sequential(plan, progress_id)
        else:
            # Use parallel execution
            return await self._execute_parallel(plan, progress_id)
    
    async def _execute_task(self, task: AgentTask) -> AgentResult:
        """Execute a single agent task."""
        task.started_at = datetime.now()
        task.status = ProcessingStatus.IN_PROGRESS
        
        try:
            logger.info(
                "Executing agent task",
                task_id=task.id,
                agent_type=task.agent_type,
                priority=task.priority.name
            )
            
            # Execute with timeout if specified
            if task.timeout:
                result = await asyncio.wait_for(
                    task.agent_instance.execute_with_context(
                        task.context.input_data if task.context else {},
                        task.context
                    ),
                    timeout=task.timeout.total_seconds()
                )
            else:
                result = await task.agent_instance.execute_with_context(
                    task.context.input_data if task.context else {},
                    task.context
                )
            
            task.result = result
            task.status = ProcessingStatus.COMPLETED if result.success else ProcessingStatus.FAILED
            task.completed_at = datetime.now()
            
            # Update agent usage stats
            agent_type = task.agent_type
            self.stats["agent_usage_count"][agent_type] = (
                self.stats["agent_usage_count"].get(agent_type, 0) + 1
            )
            
            logger.info(
                "Agent task completed",
                task_id=task.id,
                agent_type=task.agent_type,
                success=result.success,
                execution_time=(task.completed_at - task.started_at).total_seconds()
            )
            
            return result
            
        except asyncio.TimeoutError:
            task.error = "Task execution timed out"
            task.status = ProcessingStatus.FAILED
            task.completed_at = datetime.now()
            
            logger.error(
                "Agent task timed out",
                task_id=task.id,
                agent_type=task.agent_type,
                timeout=task.timeout.total_seconds() if task.timeout else None
            )
            
            return AgentResult(
                success=False,
                output_data={},
                error_message="Task execution timed out",
                processing_time=(task.timeout or timedelta(seconds=0)).total_seconds()
            )
            
        except Exception as e:
            task.error = str(e)
            task.status = ProcessingStatus.FAILED
            task.completed_at = datetime.now()
            
            logger.error(
                "Agent task execution failed",
                task_id=task.id,
                agent_type=task.agent_type,
                error=str(e)
            )
            
            return AgentResult(
                success=False,
                output_data={},
                error_message=str(e),
                processing_time=(datetime.now() - task.started_at).total_seconds()
            )
    
    def _sort_tasks_by_dependencies(self, tasks: List[AgentTask]) -> List[AgentTask]:
        """Sort tasks by dependencies and priority."""
        # Simple topological sort
        sorted_tasks = []
        remaining_tasks = tasks.copy()
        
        while remaining_tasks:
            # Find tasks with no unresolved dependencies
            ready_tasks = [
                task for task in remaining_tasks
                if all(dep_id in [t.id for t in sorted_tasks] for dep_id in task.dependencies)
            ]
            
            if not ready_tasks:
                # Circular dependency or missing dependency
                logger.warning("Circular or missing dependencies detected")
                ready_tasks = remaining_tasks  # Add all remaining tasks
            
            # Sort ready tasks by priority
            ready_tasks.sort(key=lambda t: t.priority.value, reverse=True)
            
            # Add to sorted list
            sorted_tasks.extend(ready_tasks)
            
            # Remove from remaining
            for task in ready_tasks:
                remaining_tasks.remove(task)
        
        return sorted_tasks
    
    def _group_tasks_by_dependency_level(self, tasks: List[AgentTask]) -> List[List[AgentTask]]:
        """Group tasks by dependency level for parallel execution."""
        groups = []
        remaining_tasks = tasks.copy()
        completed_task_ids = set()
        
        while remaining_tasks:
            # Find tasks that can be executed now
            current_group = [
                task for task in remaining_tasks
                if all(dep_id in completed_task_ids for dep_id in task.dependencies)
            ]
            
            if not current_group:
                # No tasks can be executed, break circular dependency
                current_group = remaining_tasks
            
            groups.append(current_group)
            
            # Update completed task IDs
            completed_task_ids.update(task.id for task in current_group)
            
            # Remove from remaining
            for task in current_group:
                remaining_tasks.remove(task)
        
        return groups
    
    def _are_dependencies_satisfied(self, task: AgentTask, completed_tasks: List[AgentTask]) -> bool:
        """Check if task dependencies are satisfied."""
        completed_task_ids = {t.id for t in completed_tasks}
        return all(dep_id in completed_task_ids for dep_id in task.dependencies)
    
    def _has_complex_dependencies(self, tasks: List[AgentTask]) -> bool:
        """Check if tasks have complex dependency relationships."""
        total_dependencies = sum(len(task.dependencies) for task in tasks)
        return total_dependencies > len(tasks) * 0.5  # More than 50% dependency ratio
    
    def _calculate_performance_stats(self, completed_tasks: List[AgentTask], failed_tasks: List[AgentTask]) -> Dict[str, Any]:
        """Calculate performance statistics."""
        total_tasks = len(completed_tasks) + len(failed_tasks)
        
        if not completed_tasks:
            return {
                "success_rate": 0.0,
                "average_execution_time": 0.0,
                "total_tasks": total_tasks,
                "agent_performance": {}
            }
        
        # Calculate execution times
        execution_times = [
            (task.completed_at - task.started_at).total_seconds()
            for task in completed_tasks
            if task.started_at and task.completed_at
        ]
        
        # Agent performance breakdown
        agent_performance = {}
        for task in completed_tasks:
            agent_type = task.agent_type
            if agent_type not in agent_performance:
                agent_performance[agent_type] = {
                    "completed": 0,
                    "total_time": 0.0,
                    "average_time": 0.0
                }
            
            agent_performance[agent_type]["completed"] += 1
            if task.started_at and task.completed_at:
                exec_time = (task.completed_at - task.started_at).total_seconds()
                agent_performance[agent_type]["total_time"] += exec_time
        
        # Calculate averages
        for agent_type, stats in agent_performance.items():
            if stats["completed"] > 0:
                stats["average_time"] = stats["total_time"] / stats["completed"]
        
        return {
            "success_rate": len(completed_tasks) / total_tasks,
            "average_execution_time": sum(execution_times) / len(execution_times) if execution_times else 0.0,
            "total_tasks": total_tasks,
            "agent_performance": agent_performance
        }
    
    def _create_error_summary(self, failed_tasks: List[AgentTask]) -> str:
        """Create a summary of errors from failed tasks."""
        error_counts = {}
        for task in failed_tasks:
            error_type = type(task.error).__name__ if task.error else "Unknown"
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
        
        summary_parts = []
        for error_type, count in error_counts.items():
            summary_parts.append(f"{error_type}: {count}")
        
        return "; ".join(summary_parts)
    
    def _update_stats(self, result: OrchestrationResult):
        """Update orchestrator statistics."""
        self.stats["plans_executed"] += 1
        self.stats["tasks_completed"] += len(result.completed_tasks)
        self.stats["tasks_failed"] += len(result.failed_tasks)
        
        # Update average execution time
        current_avg = self.stats["average_execution_time"]
        plan_count = self.stats["plans_executed"]
        new_time = result.total_execution_time.total_seconds()
        
        self.stats["average_execution_time"] = (
            (current_avg * (plan_count - 1) + new_time) / plan_count
        )
    
    def get_orchestrator_stats(self) -> Dict[str, Any]:
        """Get orchestrator performance statistics."""
        return {
            **self.stats,
            "active_plans": len(self.active_plans),
            "execution_history_count": len(self.execution_history),
            "available_agents": list(self.agents.keys()),
            "registered_agent_types": list(self.agent_factories.keys())
        }
    
    async def cancel_plan(self, plan_id: str) -> bool:
        """Cancel an active orchestration plan."""
        if plan_id not in self.active_plans:
            logger.warning("Cannot cancel plan - not found", plan_id=plan_id)
            return False
        
        # Mark all pending tasks as cancelled
        plan = self.active_plans[plan_id]
        for task in plan.tasks:
            if task.status == ProcessingStatus.PENDING:
                task.status = ProcessingStatus.FAILED
                task.error = "Plan cancelled"
        
        # Remove from active plans
        del self.active_plans[plan_id]
        
        logger.info("Orchestration plan cancelled", plan_id=plan_id)
        return True


# Global orchestrator instance
_orchestrator = None


def get_agent_orchestrator(session_id: Optional[str] = None) -> AgentOrchestrator:
    """Get the global agent orchestrator instance."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = AgentOrchestrator(session_id)
    return _orchestrator


# Convenience functions
async def execute_agent_workflow(
    agent_tasks: List[tuple],  # (agent_type, context, priority, dependencies)
    strategy: OrchestrationStrategy = OrchestrationStrategy.SEQUENTIAL,
    session_id: Optional[str] = None
) -> OrchestrationResult:
    """Execute a workflow of agent tasks."""
    orchestrator = get_agent_orchestrator(session_id)
    
    # Create tasks
    tasks = []
    for task_info in agent_tasks:
        agent_type, context = task_info[:2]
        priority = task_info[2] if len(task_info) > 2 else AgentPriority.NORMAL
        dependencies = task_info[3] if len(task_info) > 3 else []
        
        task = orchestrator.create_task(agent_type, context, priority, dependencies)
        tasks.append(task)
    
    # Create and execute plan
    plan = orchestrator.create_plan(strategy, tasks)
    return await orchestrator.execute_plan(plan)


async def execute_cv_generation_workflow(
    cv_data: Dict[str, Any],
    job_requirements: Dict[str, Any],
    session_id: Optional[str] = None
) -> OrchestrationResult:
    """Execute a complete CV generation workflow."""
    from ..agents.agent_base import AgentExecutionContext
    
    # Create contexts for different agents
    analysis_context = AgentExecutionContext(
        session_id=session_id or str(uuid.uuid4()),
        input_data={"cv_data": cv_data, "job_description": job_requirements},
        content_type=ContentType.ANALYSIS,
        processing_options={"analyze_fit": True, "suggest_improvements": True}
    )
    
    content_context = AgentExecutionContext(
        session_id=session_id or str(uuid.uuid4()),
        input_data={"cv_data": cv_data, "job_description": job_requirements},
        content_type=ContentType.EXPERIENCE,
        processing_options={"optimize_for_job": True}
    )
    
    qa_context = AgentExecutionContext(
        session_id=session_id or str(uuid.uuid4()),
        input_data={"cv_data": cv_data},
        content_type=ContentType.QUALITY_CHECK,
        processing_options={"check_grammar": True, "check_consistency": True}
    )
    
    # Define workflow tasks
    workflow_tasks = [
        ("cv_analysis", analysis_context, AgentPriority.HIGH, []),
        ("content_writer", content_context, AgentPriority.NORMAL, []),
        ("quality_assurance", qa_context, AgentPriority.NORMAL, [])
    ]
    
    return await execute_agent_workflow(
        workflow_tasks,
        OrchestrationStrategy.PIPELINE,
        session_id
    )