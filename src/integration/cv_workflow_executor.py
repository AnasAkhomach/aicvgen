"""Module for executing CV workflows with caching, performance monitoring, and error recovery."""

import asyncio
import hashlib
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional, Tuple, Union

from src.config.logging_config import get_structured_logger
from src.core.async_optimizer import AsyncOptimizer
from src.core.caching_strategy import CachePattern, IntelligentCacheManager
from src.core.performance_optimizer import PerformanceOptimizer
from src.error_handling.exceptions import AgentExecutionError, WorkflowError
from src.models.workflow_models import ContentType, WorkflowType
from src.orchestration.graphs.main_graph import create_cv_workflow_graph_with_di
from src.orchestration.state import GlobalState, create_agent_state
from src.services.error_recovery import ErrorRecoveryService, RecoveryStrategy


@dataclass
class WorkflowDependencies:
    container: Any  # Dependency injection container
    error_recovery_service: ErrorRecoveryService
    performance_optimizer: Optional[PerformanceOptimizer]
    async_optimizer: Optional[AsyncOptimizer]
    intelligent_cache: Optional[IntelligentCacheManager]


class CVWorkflowExecutor:
    """Executes CV workflows, handling caching, performance, and error recovery."""

    def __init__(
        self,
        dependencies: WorkflowDependencies,
        session_id: str,
        enable_error_recovery: bool,
    ):
        self.container = dependencies.container
        self.error_recovery = dependencies.error_recovery_service
        self.performance_optimizer = dependencies.performance_optimizer
        self.async_optimizer = dependencies.async_optimizer
        self.intelligent_cache = dependencies.intelligent_cache
        self._session_id = session_id
        self.enable_error_recovery = enable_error_recovery
        self.logger = get_structured_logger(__name__)
        self._orchestrator = None  # Lazy initialization

        self._performance_stats = {
            "requests_processed": 0,
            "total_processing_time": 0.0,
            "errors_encountered": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

    @property
    def orchestrator(self):
        """Lazy initialization of the workflow orchestrator."""
        if self._orchestrator is None:
            self._orchestrator = create_cv_workflow_graph_with_di(self.container)
        return self._orchestrator

    @property
    def performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        return self._performance_stats

    def _get_performance_context(self):
        """Get performance optimization context."""
        if self.performance_optimizer:
            return self.performance_optimizer.optimized_execution(
                operation_name="workflow_execution",
                operation_type="workflow_execution",
                expected_duration=30.0,
            )
        return nullcontext()

    def _get_async_context(self):
        """Get async optimization context."""
        if self.async_optimizer:
            return self.async_optimizer.optimized_execution(
                operation_type="workflow_execution", operation_name="cv_workflow"
            )
        return nullcontext()

    async def _handle_caching_logic(
        self,
        workflow_type: Union[WorkflowType, str],
        input_data: GlobalState,
        custom_options: Optional[Dict[str, Any]],
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        cache_key = None
        cached_result = None
        if self.intelligent_cache:
            cache_data = {
                "workflow_type": (
                    workflow_type.value
                    if hasattr(workflow_type, "value")
                    else str(workflow_type)
                ),
                "input_data": dict(input_data),
                "custom_options": custom_options or {},
            }
            cache_key = hashlib.md5(str(cache_data).encode()).hexdigest()
            cached_result = self.intelligent_cache.get(cache_key)
            if cached_result:
                self.logger.info(
                    "Workflow result served from cache", cache_key=cache_key
                )
                self._performance_stats["cache_hits"] += 1
            else:
                self._performance_stats["cache_misses"] += 1
        return cache_key, cached_result

    def _prepare_workflow_inputs(
        self,
        workflow_type: Union[WorkflowType, str],
        initial_agent_state: GlobalState,
        session_id: Optional[str],
    ) -> Dict[str, Any]:
        """Prepare workflow inputs based on workflow type and agent state."""
        if isinstance(workflow_type, str):
            workflow_type = WorkflowType(workflow_type)

        if workflow_type in [
            WorkflowType.BASIC_CV_GENERATION,
            WorkflowType.JOB_TAILORED_CV,
            WorkflowType.COMPREHENSIVE_CV,
        ]:
            structured_cv = initial_agent_state.get("structured_cv")
            if structured_cv:
                self.logger.info("Using StructuredCV data for workflow")

                workflow_inputs = {
                    "structured_cv": structured_cv,
                    "cv_text": structured_cv.to_raw_text(),
                    "session_id": session_id or self._session_id,
                    "workflow_type": workflow_type.value,
                }

                job_description_data = initial_agent_state.get("job_description_data")
                if job_description_data:
                    workflow_inputs["job_description_data"] = job_description_data
                    self.logger.info("Job description data added to workflow inputs")
                else:
                    self.logger.warning("Job description data missing in GlobalState")

                self.logger.info("Workflow inputs prepared")
                return workflow_inputs
            else:
                self.logger.warning(
                    f"Workflow type {workflow_type.value} requires StructuredCV data in GlobalState"
                )
                raise ValueError("Missing StructuredCV data in input")
        else:
            self.logger.warning(
                f"Workflow type {workflow_type.value} not fully implemented for GlobalState input or not recognized."
            )
            raise ValueError("Unrecognized or unimplemented workflow type")

    def _process_workflow_result(
        self, workflow_result: Dict[str, Any], initial_agent_state: GlobalState
    ) -> Tuple[GlobalState, bool]:
        """Process the raw workflow result into a GlobalState and determine success."""
        if isinstance(workflow_result, dict) and "structured_cv" in workflow_result:
            structured_cv = workflow_result.get("structured_cv")
            result_state = create_agent_state(
                cv_text=structured_cv.to_raw_text()
                if structured_cv
                else initial_agent_state["cv_text"],
                structured_cv=structured_cv,
                job_description_data=workflow_result.get("job_description_data"),
                error_messages=workflow_result.get("error_messages", []),
                session_id=initial_agent_state["session_id"],
                trace_id=initial_agent_state["trace_id"],
                automated_mode=initial_agent_state["automated_mode"],
            )
        else:
            # Create a new state with the error message added
            error_messages = list(initial_agent_state["error_messages"])
            error_messages.append("Workflow execution did not return expected results")
            result_state = {**initial_agent_state, "error_messages": error_messages}

        success = not bool(result_state["error_messages"] if result_state else True)
        self.logger.info("Workflow success", extra={"success": success})
        return result_state, success

    def _update_performance_stats(
        self,
        start_time: datetime,
        workflow_type: Union[WorkflowType, str],
        session_id: Optional[str],
        success: bool,
        result_state: Optional[GlobalState],
        cache_key: Optional[str],
    ) -> None:
        """Update performance statistics and cache successful results."""
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

        if success and result_state and cache_key and self.intelligent_cache:
            workflow_result_to_cache = {
                "success": success,
                "result_state": dict(result_state),  # Store as dict for caching
                "processing_time": processing_time,
                "session_id": session_id,
            }

            self.intelligent_cache.set(
                cache_key,
                workflow_result_to_cache,
                ttl_hours=2,
                tags={"workflow", workflow_type.value, "cv_generation"},
                priority=3,
                pattern=CachePattern.READ_HEAVY,
            )

    async def _handle_workflow_error(
        self,
        e: Exception,
        start_time: datetime,
        workflow_type: Union[WorkflowType, str],
        session_id: Optional[str],
        input_data: GlobalState,
        custom_options: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Handle workflow execution errors with recovery logic."""
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

        if self.enable_error_recovery:
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
            "results": {},
            "metadata": {},
            "processing_time": processing_time,
            "errors": [str(e)],
        }

    async def execute_workflow(
        self,
        workflow_type: Union[WorkflowType, str],
        input_data: GlobalState,
        session_id: Optional[str] = None,
        custom_options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute a predefined workflow.

        Args:
            workflow_type: The type of workflow to execute.
            input_data: The GlobalState object containing all necessary workflow data.
            session_id: Optional session ID for state management.
            custom_options: Optional custom options for the workflow.

        Returns:
            A dictionary containing the results of the workflow execution.
        """
        if not self.orchestrator:
            raise RuntimeError("Orchestration not enabled")

        start_time = datetime.now()
        success = False
        result_state = None

        cache_key, cached_result = await self._handle_caching_logic(
            workflow_type, input_data, custom_options
        )
        if cached_result:
            return cached_result

        try:
            async with self._get_performance_context():
                # Convert string to enum if needed
                if isinstance(workflow_type, str):
                    workflow_type = WorkflowType(workflow_type)

                initial_agent_state = input_data

                self.logger.info(
                    "Executing workflow",
                    extra={
                        "workflow_type": workflow_type.value,
                        "session_id": session_id,
                        "input_data_type": "GlobalState",
                    },
                )

                workflow_inputs = self._prepare_workflow_inputs(
                    workflow_type, initial_agent_state, session_id
                )

                async with self._get_async_context():
                    workflow_result = await self.orchestrator.invoke(workflow_inputs)

                result_state, success = self._process_workflow_result(
                    workflow_result, initial_agent_state
                )

                self._update_performance_stats(
                    start_time,
                    workflow_type,
                    session_id,
                    success,
                    result_state,
                    cache_key,
                )

                final_errors = (
                    result_state.get("error_messages", []) if result_state else []
                )
                self.logger.info(
                    f"Final return structure - success: {success}, errors: {final_errors}"
                )

                return {
                    "success": success,
                    "results": dict(result_state) if result_state else {},
                    "metadata": {
                        "workflow_type": workflow_type.value,
                        "session_id": session_id,
                    },
                    "processing_time": (datetime.now() - start_time).total_seconds(),
                    "errors": final_errors,
                }

        except (
            WorkflowError,
            AgentExecutionError,
            RuntimeError,
            ValueError,
            TypeError,
            KeyError,
        ) as e:
            return await self._handle_workflow_error(
                e, start_time, workflow_type, session_id, input_data, custom_options
            )
