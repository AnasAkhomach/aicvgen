# """Simplified startup optimization utilities for the CV generation system."""

# import threading
# import time
# from dataclasses import dataclass
# from datetime import datetime
# from typing import  Optional


# from src.config.logging_config import get_structured_logger
# from src.core.container import get_container
# from src.utils.performance import PerformanceMonitor

# logger = get_structured_logger(__name__)


# @dataclass
# class StartupMetrics:
#     """Simplified metrics collected during startup optimization."""

#     container_initialization_time: float = 0.0
#     service_validation_time: float = 0.0
#     total_startup_time: float = 0.0
#     memory_usage_mb: float = 0.0
#     timestamp: datetime = None

#     def __post_init__(self):
#         if self.timestamp is None:
#             self.timestamp = datetime.now()


# class StartupOptimizer:
#     """Simplified startup optimizer using dependency injection container."""

#     def __init__(self):
#         self._lock = threading.Lock()
#         self._metrics_history: list[StartupMetrics] = []
#         self._is_optimizing = False

#     async def optimize_startup(self, validate_services: bool = True) -> StartupMetrics:
#         """Optimize application startup sequence."""
#         with self._lock:
#             if self._is_optimizing:
#                 logger.warning("Startup optimization already in progress")
#                 return (
#                     self._metrics_history[-1]
#                     if self._metrics_history
#                     else StartupMetrics()
#                 )

#             self._is_optimizing = True

#         try:
#             logger.info("Starting startup optimization")
#             start_time = time.time()
#             metrics = StartupMetrics()

#             # Initialize container
#             container_start = time.time()
#             container = get_container()
#             metrics.container_initialization_time = time.time() - container_start

#             # Validate core services if requested
#             if validate_services:
#                 validation_start = time.time()
#                 await self._validate_services(container)
#                 metrics.service_validation_time = time.time() - validation_start

#             # Record memory usage
#             performance_monitor = PerformanceMonitor()
#             metrics.memory_usage_mb = performance_monitor.get_memory_usage()

#             metrics.total_startup_time = time.time() - start_time
#             self._metrics_history.append(metrics)

#             logger.info(
#                 "Startup optimization completed",
#                 total_time=metrics.total_startup_time,
#                 container_time=metrics.container_initialization_time,
#                 validation_time=metrics.service_validation_time,
#                 memory_mb=metrics.memory_usage_mb,
#             )

#             return metrics

#         finally:
#             self._is_optimizing = False

#     async def _validate_services(self, container):
#         """Validate that core services can be instantiated."""
#         try:
#             # Validate key services
#             container.config()
#             container.llm_service()
#             container.vector_store_service()
#             logger.debug("Core services validated successfully")
#         except Exception as e:
#             logger.error(f"Service validation failed: {e}")
#             raise

#     def get_latest_metrics(self) -> Optional[StartupMetrics]:
#         """Get the latest startup metrics."""
#         return self._metrics_history[-1] if self._metrics_history else None

#     def get_metrics_history(self) -> list[StartupMetrics]:
#         """Get all startup metrics history."""
#         return self._metrics_history.copy()

#     def reset_metrics(self):
#         """Reset metrics history."""
#         with self._lock:
#             self._metrics_history.clear()
#             logger.info("Startup metrics reset")


# # Global instance
# _optimizer_lock = threading.Lock()
# _optimizer: Optional[StartupOptimizer] = None


# def get_startup_optimizer() -> StartupOptimizer:
#     """Get the singleton startup optimizer."""
#     global _optimizer
#     if _optimizer is None:
#         with _optimizer_lock:
#             if _optimizer is None:
#                 _optimizer = StartupOptimizer()
#     return _optimizer
