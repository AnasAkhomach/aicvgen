"""Advanced performance optimization module for the AI CV Generator.

This module provides comprehensive performance optimizations including:
- Advanced connection pooling and resource management
- Intelligent batch processing and request coalescing
- Multi-level caching with smart eviction policies
- Async execution optimization and concurrency control
- Memory management and garbage collection optimization
"""

import asyncio
import gc
import hashlib
import os
import pickle
import threading
import time
from collections import OrderedDict, defaultdict
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.config.logging_config import get_structured_logger
from src.error_handling.boundaries import CATCHABLE_EXCEPTIONS
from src.utils.decorators import create_async_sync_decorator
from src.utils.performance import get_memory_optimizer, get_performance_monitor

logger = get_structured_logger("performance_optimizer")


@dataclass
class ConnectionPoolConfig:
    """Configuration for connection pooling."""

    max_connections: int = 20
    min_connections: int = 5
    connection_timeout: float = 30.0
    idle_timeout: float = 300.0
    max_retries: int = 3
    retry_delay: float = 1.0


@dataclass
class BatchConfig:
    """Configuration for batch processing."""

    max_batch_size: int = 10
    batch_timeout: float = 2.0
    max_concurrent_batches: int = 5
    enable_request_coalescing: bool = True
    coalescing_window: float = 0.5


@dataclass
class CacheConfig:
    """Configuration for multi-level caching."""

    l1_cache_size: int = 1000
    l2_cache_size: int = 5000
    l3_cache_size: int = 10000
    default_ttl_hours: int = 24
    enable_persistence: bool = True
    persistence_interval: float = 300.0
    compression_enabled: bool = True


class AdvancedConnectionPool:
    """Advanced connection pool with health monitoring and auto-scaling."""

    def __init__(self, config: ConnectionPoolConfig):
        self.config = config
        self._pool: List[Any] = []
        self._in_use: Dict[int, Any] = {}
        self._lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(config.max_connections)
        self._health_check_task: Optional[asyncio.Task] = None
        self._stats = {
            "connections_created": 0,
            "connections_reused": 0,
            "connections_failed": 0,
            "pool_hits": 0,
            "pool_misses": 0,
        }

        logger.info("Advanced connection pool initialized", config=config)

    async def start(self):
        """Start the connection pool and health monitoring."""
        await self._initialize_min_connections()
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        logger.info("Connection pool started")

    async def stop(self):
        """Stop the connection pool and cleanup resources."""
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass

        async with self._lock:
            # Close all connections
            for conn in self._pool:
                await self._close_connection(conn)
            for conn in self._in_use.values():
                await self._close_connection(conn)

            self._pool.clear()
            self._in_use.clear()

        logger.info("Connection pool stopped")

    @asynccontextmanager
    async def get_connection(self):
        """Get a connection from the pool."""
        await self._semaphore.acquire()
        connection = None

        try:
            connection = await self._acquire_connection()
            yield connection
        finally:
            if connection:
                await self._release_connection(connection)
            self._semaphore.release()

    async def _acquire_connection(self):
        """Acquire a connection from the pool."""
        async with self._lock:
            if self._pool:
                connection = self._pool.pop()
                conn_id = id(connection)
                self._in_use[conn_id] = connection
                self._stats["pool_hits"] += 1
                self._stats["connections_reused"] += 1
                return connection
            else:
                self._stats["pool_misses"] += 1
                connection = await self._create_connection()
                conn_id = id(connection)
                self._in_use[conn_id] = connection
                return connection

    async def _release_connection(self, connection):
        """Release a connection back to the pool."""
        async with self._lock:
            conn_id = id(connection)
            if conn_id in self._in_use:
                del self._in_use[conn_id]

                if await self._is_connection_healthy(connection):
                    if len(self._pool) < self.config.max_connections:
                        self._pool.append(connection)
                    else:
                        await self._close_connection(connection)
                else:
                    await self._close_connection(connection)

    async def _create_connection(self):
        """Create a new connection."""
        try:
            # Placeholder for actual connection creation
            connection = {"created_at": time.time(), "id": id(object())}
            self._stats["connections_created"] += 1
            return connection
        except CATCHABLE_EXCEPTIONS as e:
            self._stats["connections_failed"] += 1
            logger.error("Failed to create connection", error=str(e))
            raise

    async def _close_connection(self, connection):
        """Close a connection."""
        try:
            # Placeholder for actual connection cleanup
            pass
        except CATCHABLE_EXCEPTIONS as _:  # Catch all for close errors
            logger.warning("Error closing connection", error=str(_))

    async def _is_connection_healthy(self, connection) -> bool:
        """Check if a connection is healthy."""
        try:
            # Placeholder for actual health check
            created_at = connection.get("created_at", 0)
            return time.time() - created_at < self.config.idle_timeout
        except CATCHABLE_EXCEPTIONS as _:  # Catch all for health check errors
            return False

    async def _initialize_min_connections(self):
        """Initialize minimum number of connections."""
        for _ in range(self.config.min_connections):
            try:
                connection = await self._create_connection()
                self._pool.append(connection)
            except CATCHABLE_EXCEPTIONS as e:
                logger.warning("Failed to create initial connection", error=str(e))

    async def _health_check_loop(self):
        """Periodic health check for connections."""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                await self._cleanup_unhealthy_connections()
            except (asyncio.CancelledError, RuntimeError) as _:
                break
            except CATCHABLE_EXCEPTIONS as e:  # Top-level catch for health check loop
                logger.error("Error in health check loop", error=str(e))

    async def _cleanup_unhealthy_connections(self):
        """Remove unhealthy connections from the pool."""
        async with self._lock:
            healthy_connections = []
            for conn in self._pool:
                if await self._is_connection_healthy(conn):
                    healthy_connections.append(conn)
                else:
                    await self._close_connection(conn)

            self._pool = healthy_connections

    def get_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics."""
        return {
            **self._stats,
            "pool_size": len(self._pool),
            "in_use_count": len(self._in_use),
            "total_capacity": self.config.max_connections,
        }


class IntelligentBatchProcessor:
    """Intelligent batch processor with request coalescing and optimization."""

    def __init__(self, config: BatchConfig):
        self.config = config
        self._pending_requests: Dict[
            str, List[Tuple[Any, asyncio.Future]]
        ] = defaultdict(list)
        self._batch_timers: Dict[str, asyncio.Task] = {}
        self._semaphore = asyncio.Semaphore(config.max_concurrent_batches)
        self._lock = asyncio.Lock()
        self._stats = {
            "batches_processed": 0,
            "requests_coalesced": 0,
            "total_requests": 0,
            "average_batch_size": 0.0,
        }

        logger.info("Intelligent batch processor initialized", config=config)

    async def process_request(
        self, request_key: str, request_data: Any, processor_func: Callable
    ) -> Any:
        """Process a request with intelligent batching and coalescing."""
        self._stats["total_requests"] += 1

        if not self.config.enable_request_coalescing:
            return await processor_func([request_data])

        # Create a future for this request
        future = asyncio.Future()

        async with self._lock:
            # Add request to pending batch
            self._pending_requests[request_key].append((request_data, future))

            # Start batch timer if not already running
            if request_key not in self._batch_timers:
                self._batch_timers[request_key] = asyncio.create_task(
                    self._batch_timer(request_key, processor_func)
                )

            # Process immediately if batch is full
            if len(self._pending_requests[request_key]) >= self.config.max_batch_size:
                await self._process_batch_now(request_key, processor_func)

        return await future

    async def _batch_timer(self, request_key: str, processor_func: Callable):
        """Timer for batch processing."""
        try:
            await asyncio.sleep(self.config.batch_timeout)
            async with self._lock:
                if (
                    request_key in self._pending_requests
                    and self._pending_requests[request_key]
                ):
                    await self._process_batch_now(request_key, processor_func)
        except asyncio.CancelledError:
            pass
        except CATCHABLE_EXCEPTIONS as e:
            logger.error("Error in batch timer", request_key=request_key, error=str(e))

    async def _process_batch_now(self, request_key: str, processor_func: Callable):
        """Process the current batch immediately."""
        if (
            request_key not in self._pending_requests
            or not self._pending_requests[request_key]
        ):
            return

        # Get current batch
        batch = self._pending_requests[request_key]
        self._pending_requests[request_key] = []

        # Cancel timer
        if request_key in self._batch_timers:
            self._batch_timers[request_key].cancel()
            del self._batch_timers[request_key]

        # Process batch asynchronously
        asyncio.create_task(self._execute_batch(batch, processor_func))

    async def _execute_batch(
        self, batch: List[Tuple[Any, asyncio.Future]], processor_func: Callable
    ):
        """Execute a batch of requests."""
        async with self._semaphore:
            try:
                # Extract request data
                request_data = [item[0] for item in batch]
                futures = [item[1] for item in batch]

                # Process batch
                results = await processor_func(request_data)

                # Distribute results
                if isinstance(results, list) and len(results) == len(futures):
                    for future, result in zip(futures, results):
                        if not future.done():
                            future.set_result(result)
                else:
                    # If single result, distribute to all
                    for future in futures:
                        if not future.done():
                            future.set_result(results)

                # Update stats
                self._stats["batches_processed"] += 1
                self._stats["requests_coalesced"] += len(batch) - 1
                self._update_average_batch_size(len(batch))

            except CATCHABLE_EXCEPTIONS as e:
                # Set exception for all futures
                for _, future in batch:
                    if not future.done():
                        future.set_exception(e)
                logger.error(
                    "Error processing batch", batch_size=len(batch), error=str(e)
                )

    def _update_average_batch_size(self, batch_size: int):
        """Update average batch size statistics."""
        current_avg = self._stats["average_batch_size"]
        batches_processed = self._stats["batches_processed"]

        if batches_processed == 1:
            self._stats["average_batch_size"] = batch_size
        else:
            self._stats["average_batch_size"] = (
                current_avg * (batches_processed - 1) + batch_size
            ) / batches_processed

    def get_stats(self) -> Dict[str, Any]:
        """Get batch processor statistics."""
        return self._stats.copy()


class MultiLevelCache:
    """Multi-level cache with intelligent eviction and compression."""

    def __init__(self, config: CacheConfig):
        self.config = config

        # L1: In-memory LRU cache (fastest)
        self.l1_cache: OrderedDict = OrderedDict()

        # L2: Compressed in-memory cache
        self.l2_cache: OrderedDict = OrderedDict()

        # L3: Persistent disk cache
        self.l3_cache_dir = os.path.join(os.getcwd(), "data", "cache", "l3")
        os.makedirs(self.l3_cache_dir, exist_ok=True)

        self._lock = threading.RLock()
        self._persistence_task: Optional[asyncio.Task] = None

        self._stats = {
            "l1_hits": 0,
            "l1_misses": 0,
            "l2_hits": 0,
            "l2_misses": 0,
            "l3_hits": 0,
            "l3_misses": 0,
            "evictions": 0,
            "compressions": 0,
        }

        logger.info("Multi-level cache initialized", config=config)

    def start_persistence(self):
        """Start periodic persistence task."""
        if self.config.enable_persistence:
            try:
                loop = asyncio.get_running_loop()
                self._persistence_task = loop.create_task(self._persistence_loop())
            except RuntimeError:
                # No event loop running, persistence will be manual
                logger.warning(
                    "No running event loop found, cache persistence disabled"
                )

    def stop_persistence(self):
        """Stop persistence task."""
        if self._persistence_task:
            self._persistence_task.cancel()

    def get(self, key: str) -> Optional[Any]:
        """Get value from multi-level cache."""
        with self._lock:
            # Try L1 cache first
            if key in self.l1_cache:
                self._stats["l1_hits"] += 1
                value = self.l1_cache[key]
                # Move to end (LRU)
                self.l1_cache.move_to_end(key)
                return value["data"]

            self._stats["l1_misses"] += 1

            # Try L2 cache
            if key in self.l2_cache:
                self._stats["l2_hits"] += 1
                compressed_value = self.l2_cache[key]

                # Decompress and promote to L1
                try:
                    value = self._decompress(compressed_value["data"])
                    self._set_l1(key, value, compressed_value["expiry"])
                    self.l2_cache.move_to_end(key)
                    return value
                except CATCHABLE_EXCEPTIONS as e:
                    logger.warning(
                        "Failed to decompress L2 cache entry", key=key, error=str(e)
                    )
                    del self.l2_cache[key]

            self._stats["l2_misses"] += 1

            # Try L3 cache (disk)
            try:
                value = self._get_l3(key)
                if value is not None:
                    self._stats["l3_hits"] += 1
                    # Promote to L1
                    self._set_l1(key, value["data"], value["expiry"])
                    return value["data"]
            except CATCHABLE_EXCEPTIONS as e:
                logger.warning("Failed to read L3 cache entry", key=key, error=str(e))

            self._stats["l3_misses"] += 1
            return None

    def set(self, key: str, value: Any, ttl_hours: Optional[int] = None):
        """Set value in multi-level cache."""
        ttl = ttl_hours or self.config.default_ttl_hours
        expiry = datetime.now() + timedelta(hours=ttl)

        with self._lock:
            # Always set in L1
            self._set_l1(key, value, expiry)

            # Optionally compress and store in L2
            if self.config.compression_enabled:
                try:
                    compressed_value = self._compress(value)
                    self._set_l2(key, compressed_value, expiry)
                    self._stats["compressions"] += 1
                except CATCHABLE_EXCEPTIONS as e:
                    logger.warning(
                        "Failed to compress value for L2 cache", key=key, error=str(e)
                    )

    def _set_l1(self, key: str, value: Any, expiry: datetime):
        """Set value in L1 cache with LRU eviction."""
        # Evict if full
        while len(self.l1_cache) >= self.config.l1_cache_size:
            self.l1_cache.popitem(last=False)
            self._stats["evictions"] += 1

        self.l1_cache[key] = {"data": value, "expiry": expiry}

    def _set_l2(self, key: str, compressed_value: bytes, expiry: datetime):
        """Set compressed value in L2 cache."""
        # Evict if full
        while len(self.l2_cache) >= self.config.l2_cache_size:
            self.l2_cache.popitem(last=False)
            self._stats["evictions"] += 1

        self.l2_cache[key] = {"data": compressed_value, "expiry": expiry}

    def _get_l3(self, key: str) -> Optional[Dict[str, Any]]:
        """Get value from L3 (disk) cache."""
        cache_file = os.path.join(
            self.l3_cache_dir, f"{hashlib.md5(key.encode()).hexdigest()}.pkl"
        )

        if os.path.exists(cache_file):
            try:
                with open(cache_file, "rb") as f:
                    cached_data = pickle.load(f)

                # Check expiry
                if datetime.now() < cached_data["expiry"]:
                    return cached_data
                else:
                    os.remove(cache_file)
            except CATCHABLE_EXCEPTIONS as e:
                logger.warning(
                    "Failed to read L3 cache file", file=cache_file, error=str(e)
                )
                try:
                    os.remove(cache_file)
                except OSError:
                    pass

        return None

    def _set_l3(self, key: str, value: Any, expiry: datetime):
        """Set value in L3 (disk) cache."""
        cache_file = os.path.join(
            self.l3_cache_dir, f"{hashlib.md5(key.encode()).hexdigest()}.pkl"
        )

        try:
            cached_data = {"data": value, "expiry": expiry}
            with open(cache_file, "wb") as f:
                pickle.dump(cached_data, f)
        except CATCHABLE_EXCEPTIONS as e:
            logger.warning(
                "Failed to write L3 cache file", file=cache_file, error=str(e)
            )

    def _compress(self, value: Any) -> bytes:
        """Compress value for storage."""
        import gzip

        serialized = pickle.dumps(value)
        return gzip.compress(serialized)

    def _decompress(self, compressed_value: bytes) -> Any:
        """Decompress value from storage."""
        import gzip

        serialized = gzip.decompress(compressed_value)
        return pickle.loads(serialized)

    async def _persistence_loop(self):
        """Periodic persistence of cache to disk."""
        while True:
            try:
                await asyncio.sleep(self.config.persistence_interval)
                await self._persist_to_l3()
            except asyncio.CancelledError:
                break
            except CATCHABLE_EXCEPTIONS as e:
                logger.error("Error in persistence loop", error=str(e))

    async def _persist_to_l3(self):
        """Persist L1 and L2 cache entries to L3."""
        with self._lock:
            # Persist L1 entries
            for key, entry in list(self.l1_cache.items()):
                if datetime.now() < entry["expiry"]:
                    self._set_l3(key, entry["data"], entry["expiry"])

            # Persist L2 entries (decompress first)
            for key, entry in list(self.l2_cache.items()):
                if datetime.now() < entry["expiry"]:
                    try:
                        value = self._decompress(entry["data"])
                        self._set_l3(key, value, entry["expiry"])
                    except CATCHABLE_EXCEPTIONS as e:
                        logger.warning(
                            "Failed to persist L2 entry to L3", key=key, error=str(e)
                        )

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = sum(
                [
                    self._stats["l1_hits"],
                    self._stats["l1_misses"],
                    self._stats["l2_hits"],
                    self._stats["l2_misses"],
                    self._stats["l3_hits"],
                    self._stats["l3_misses"],
                ]
            )

            hit_rate = 0.0
            if total_requests > 0:
                total_hits = (
                    self._stats["l1_hits"]
                    + self._stats["l2_hits"]
                    + self._stats["l3_hits"]
                )
                hit_rate = (total_hits / total_requests) * 100

            return {
                **self._stats,
                "l1_size": len(self.l1_cache),
                "l2_size": len(self.l2_cache),
                "total_hit_rate": round(hit_rate, 2),
                "total_requests": total_requests,
            }


class PerformanceOptimizer:
    """Main performance optimizer coordinating all optimization strategies."""

    def __init__(
        self,
        connection_config: Optional[ConnectionPoolConfig] = None,
        batch_config: Optional[BatchConfig] = None,
        cache_config: Optional[CacheConfig] = None,
    ):
        self.connection_config = connection_config or ConnectionPoolConfig()
        self.batch_config = batch_config or BatchConfig()
        self.cache_config = cache_config or CacheConfig()

        # Initialize components
        self.connection_pool = AdvancedConnectionPool(self.connection_config)
        self.batch_processor = IntelligentBatchProcessor(self.batch_config)
        self.cache = MultiLevelCache(self.cache_config)

        # Performance monitoring
        self.performance_monitor = get_performance_monitor()
        self.memory_optimizer = get_memory_optimizer()

        # Global optimization state
        self._optimization_active = False
        self._optimization_task: Optional[asyncio.Task] = None

        logger.info("Performance optimizer initialized")

    async def start(self):
        """Start all optimization components."""
        await self.connection_pool.start()
        self.cache.start_persistence()

        self._optimization_active = True
        self._optimization_task = asyncio.create_task(self._optimization_loop())

        logger.info("Performance optimizer started")

    async def stop(self):
        """Stop all optimization components."""
        self._optimization_active = False

        if self._optimization_task:
            self._optimization_task.cancel()
            try:
                await self._optimization_task
            except asyncio.CancelledError:
                pass

        await self.connection_pool.stop()
        self.cache.stop_persistence()

        logger.info("Performance optimizer stopped")

    @asynccontextmanager
    async def optimized_execution(self, operation_name: str, **metadata):
        """Context manager for optimized execution with monitoring."""
        async with self.performance_monitor.measure_async(operation_name, **metadata):
            # Auto memory optimization before heavy operations
            if operation_name in ["llm_call", "agent_execution", "workflow_execution"]:
                self.memory_optimizer.auto_optimize()

            yield

            # Trigger garbage collection after heavy operations
            if operation_name in ["llm_call", "agent_execution"]:
                gc.collect()

    async def _optimization_loop(self):
        """Continuous optimization loop."""
        while self._optimization_active:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                await self._run_optimizations()
            except asyncio.CancelledError:
                break
            except CATCHABLE_EXCEPTIONS as e:
                logger.error("Error in optimization loop", error=str(e))

    async def _run_optimizations(self):
        """Run periodic optimizations."""
        # Memory optimization
        memory_stats = self.memory_optimizer.optimize_memory()

        # Cache cleanup
        await self._cleanup_expired_cache_entries()

        # Performance analysis
        perf_stats = self.performance_monitor.get_overall_stats()

        logger.info(
            "Periodic optimization completed",
            memory_freed_mb=memory_stats.get("memory_freed_mb", 0),
            cache_stats=self.cache.get_stats(),
            performance_stats=perf_stats,
        )

    async def _cleanup_expired_cache_entries(self):
        """Clean up expired cache entries."""
        try:
            # Clean L1 cache
            with self.cache._lock:
                expired_keys = []
                for key, entry in self.cache.l1_cache.items():
                    if datetime.now() > entry["expiry"]:
                        expired_keys.append(key)

                for key in expired_keys:
                    del self.cache.l1_cache[key]

                # Clean L2 cache
                expired_keys = []
                for key, entry in self.cache.l2_cache.items():
                    if datetime.now() > entry["expiry"]:
                        expired_keys.append(key)

                for key in expired_keys:
                    del self.cache.l2_cache[key]

            logger.debug("Cache cleanup completed")

        except CATCHABLE_EXCEPTIONS as e:
            logger.warning("Error during cache cleanup", error=str(e))

    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        return {
            "connection_pool": self.connection_pool.get_stats(),
            "batch_processor": self.batch_processor.get_stats(),
            "cache": self.cache.get_stats(),
            "performance_monitor": self.performance_monitor.get_overall_stats(),
            "memory": self.memory_optimizer.get_memory_info(),
        }


# Global optimizer instance and accessors (placed immediately after PerformanceOptimizer)
_global_optimizer = None


def get_performance_optimizer():
    """Get the global performance optimizer instance."""
    global _global_optimizer
    if _global_optimizer is None:
        _global_optimizer = PerformanceOptimizer()
    return _global_optimizer


def reset_performance_optimizer():
    """Reset the global performance optimizer."""
    global _global_optimizer
    _global_optimizer = None


# Decorator for automatic performance optimization
def optimize_performance(operation_name: str = None, **metadata):
    """Decorator for automatic performance optimization."""

    def create_async_wrapper(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            optimizer = get_performance_optimizer()
            name = operation_name or func.__name__

            async with optimizer.optimized_execution(name, **metadata):
                return await func(*args, **kwargs)

        return async_wrapper

    def create_sync_wrapper(func):
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # For sync functions, just use performance monitoring
            monitor = get_performance_monitor()
            with monitor.measure_sync(operation_name or func.__name__, **metadata):
                return func(*args, **kwargs)

        return sync_wrapper

    return create_async_sync_decorator(create_async_wrapper, create_sync_wrapper)
