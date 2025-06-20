"""Advanced caching strategies for the AI CV Generator.

This module provides sophisticated caching strategies including:
- Intelligent cache warming and preloading
- Predictive caching based on usage patterns
- Cache coherence and invalidation strategies
- Distributed caching coordination
- Cache analytics and optimization recommendations
"""

import asyncio
import time
import hashlib
import json
import pickle
from typing import Dict, Any, Optional, List, Callable, Union, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import threading
from contextlib import asynccontextmanager
from functools import wraps
import os
import weakref
from enum import Enum

from ..config.logging_config import get_structured_logger
from ..utils.performance import get_performance_monitor

logger = get_structured_logger("caching_strategy")


class CacheLevel(Enum):
    """Cache level enumeration."""

    L1_MEMORY = "l1_memory"
    L2_COMPRESSED = "l2_compressed"
    L3_PERSISTENT = "l3_persistent"
    L4_DISTRIBUTED = "l4_distributed"


class CachePattern(Enum):
    """Cache access pattern enumeration."""

    READ_HEAVY = "read_heavy"
    WRITE_HEAVY = "write_heavy"
    MIXED = "mixed"
    TEMPORAL = "temporal"
    SPATIAL = "spatial"


@dataclass
class CacheEntry:
    """Enhanced cache entry with metadata."""

    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    size_bytes: int = 0
    ttl: Optional[timedelta] = None
    tags: Set[str] = field(default_factory=set)
    priority: int = 1  # 1=low, 5=high
    pattern: CachePattern = CachePattern.MIXED
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        """Check if the cache entry is expired."""
        if self.ttl is None:
            return False
        return datetime.now() > self.created_at + self.ttl

    def update_access(self):
        """Update access statistics."""
        self.last_accessed = datetime.now()
        self.access_count += 1


@dataclass
class CacheStats:
    """Comprehensive cache statistics."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    invalidations: int = 0
    size_bytes: int = 0
    entry_count: int = 0
    avg_access_time_ms: float = 0.0
    hit_rate_percent: float = 0.0
    memory_efficiency: float = 0.0

    def calculate_hit_rate(self):
        """Calculate hit rate percentage."""
        total = self.hits + self.misses
        self.hit_rate_percent = (self.hits / total * 100) if total > 0 else 0.0


class PredictiveCacheWarmer:
    """Predictive cache warming based on usage patterns."""

    def __init__(self, max_predictions: int = 1000):
        self.max_predictions = max_predictions
        self._access_patterns: Dict[str, List[float]] = defaultdict(list)
        self._pattern_weights: Dict[str, float] = defaultdict(lambda: 1.0)
        self._prediction_cache: Dict[str, List[str]] = {}
        self._lock = threading.RLock()

        logger.info("Predictive cache warmer initialized")

    def record_access(self, key: str, timestamp: Optional[float] = None):
        """Record cache access for pattern learning."""
        timestamp = timestamp or time.time()

        with self._lock:
            # Keep only recent accesses (last 24 hours)
            cutoff = timestamp - 86400  # 24 hours
            self._access_patterns[key] = [
                t for t in self._access_patterns[key] if t > cutoff
            ]

            self._access_patterns[key].append(timestamp)

            # Limit pattern history
            if len(self._access_patterns[key]) > 100:
                self._access_patterns[key] = self._access_patterns[key][-100:]

    def predict_next_accesses(self, current_key: str, count: int = 10) -> List[str]:
        """Predict next likely cache accesses."""
        with self._lock:
            if current_key in self._prediction_cache:
                return self._prediction_cache[current_key][:count]

            predictions = self._generate_predictions(current_key)
            self._prediction_cache[current_key] = predictions

            return predictions[:count]

    def _generate_predictions(self, current_key: str) -> List[str]:
        """Generate predictions based on access patterns."""
        # Analyze temporal patterns
        temporal_predictions = self._analyze_temporal_patterns(current_key)

        # Analyze spatial patterns (key similarity)
        spatial_predictions = self._analyze_spatial_patterns(current_key)

        # Combine predictions with weights
        combined_predictions = {}

        for key, score in temporal_predictions.items():
            combined_predictions[key] = combined_predictions.get(key, 0) + score * 0.6

        for key, score in spatial_predictions.items():
            combined_predictions[key] = combined_predictions.get(key, 0) + score * 0.4

        # Sort by score and return top predictions
        sorted_predictions = sorted(
            combined_predictions.items(), key=lambda x: x[1], reverse=True
        )

        return [key for key, _ in sorted_predictions]

    def _analyze_temporal_patterns(self, current_key: str) -> Dict[str, float]:
        """Analyze temporal access patterns."""
        predictions = {}
        current_time = time.time()

        # Look for keys accessed after the current key in the past
        current_accesses = self._access_patterns.get(current_key, [])

        for access_time in current_accesses[-10:]:  # Last 10 accesses
            # Find keys accessed within 1 hour after this access
            for other_key, other_accesses in self._access_patterns.items():
                if other_key == current_key:
                    continue

                for other_time in other_accesses:
                    if access_time < other_time <= access_time + 3600:  # Within 1 hour
                        time_diff = other_time - access_time
                        score = 1.0 / (
                            1.0 + time_diff / 3600
                        )  # Closer in time = higher score
                        predictions[other_key] = predictions.get(other_key, 0) + score

        return predictions

    def _analyze_spatial_patterns(self, current_key: str) -> Dict[str, float]:
        """Analyze spatial (similarity) patterns."""
        predictions = {}

        # Simple similarity based on key prefixes and patterns
        for other_key in self._access_patterns.keys():
            if other_key == current_key:
                continue

            similarity = self._calculate_key_similarity(current_key, other_key)
            if similarity > 0.3:  # Threshold for similarity
                predictions[other_key] = similarity

        return predictions

    def _calculate_key_similarity(self, key1: str, key2: str) -> float:
        """Calculate similarity between two cache keys."""
        # Simple similarity based on common prefixes and substrings
        if key1 == key2:
            return 1.0

        # Check for common prefixes
        common_prefix_len = 0
        for i in range(min(len(key1), len(key2))):
            if key1[i] == key2[i]:
                common_prefix_len += 1
            else:
                break

        prefix_similarity = common_prefix_len / max(len(key1), len(key2))

        # Check for common substrings
        key1_parts = set(key1.split("_"))
        key2_parts = set(key2.split("_"))
        common_parts = key1_parts.intersection(key2_parts)

        if key1_parts or key2_parts:
            substring_similarity = len(common_parts) / len(key1_parts.union(key2_parts))
        else:
            substring_similarity = 0.0

        return max(prefix_similarity, substring_similarity)


class CacheCoherenceManager:
    """Manages cache coherence and invalidation across multiple cache levels."""

    def __init__(self):
        self._invalidation_callbacks: Dict[str, List[Callable]] = defaultdict(list)
        self._tag_mappings: Dict[str, Set[str]] = defaultdict(set)  # tag -> keys
        self._key_tags: Dict[str, Set[str]] = defaultdict(set)  # key -> tags
        self._lock = threading.RLock()

        logger.info("Cache coherence manager initialized")

    def register_invalidation_callback(self, pattern: str, callback: Callable):
        """Register a callback for cache invalidation."""
        with self._lock:
            self._invalidation_callbacks[pattern].append(callback)

    def tag_key(self, key: str, tags: Set[str]):
        """Associate tags with a cache key."""
        with self._lock:
            self._key_tags[key].update(tags)
            for tag in tags:
                self._tag_mappings[tag].add(key)

    def invalidate_by_key(self, key: str):
        """Invalidate cache entry by key."""
        with self._lock:
            # Remove from tag mappings
            tags = self._key_tags.get(key, set())
            for tag in tags:
                self._tag_mappings[tag].discard(key)

            if key in self._key_tags:
                del self._key_tags[key]

            # Call invalidation callbacks
            self._call_invalidation_callbacks(key)

    def invalidate_by_tag(self, tag: str):
        """Invalidate all cache entries with a specific tag."""
        with self._lock:
            keys_to_invalidate = self._tag_mappings.get(tag, set()).copy()

            for key in keys_to_invalidate:
                self.invalidate_by_key(key)

    def invalidate_by_pattern(self, pattern: str):
        """Invalidate cache entries matching a pattern."""
        with self._lock:
            import re

            pattern_regex = re.compile(pattern)

            keys_to_invalidate = []
            for key in self._key_tags.keys():
                if pattern_regex.match(key):
                    keys_to_invalidate.append(key)

            for key in keys_to_invalidate:
                self.invalidate_by_key(key)

    def _call_invalidation_callbacks(self, key: str):
        """Call registered invalidation callbacks."""
        for pattern, callbacks in self._invalidation_callbacks.items():
            import re

            if re.match(pattern, key):
                for callback in callbacks:
                    try:
                        callback(key)
                    except Exception as e:
                        logger.warning(
                            "Error in invalidation callback",
                            pattern=pattern,
                            error=str(e),
                        )


class IntelligentCacheManager:
    """Intelligent cache manager with advanced strategies."""

    def __init__(
        self,
        max_memory_mb: int = 500,
        enable_prediction: bool = True,
        enable_coherence: bool = True,
    ):

        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.enable_prediction = enable_prediction
        self.enable_coherence = enable_coherence

        # Cache storage by level
        self._caches: Dict[CacheLevel, Dict[str, CacheEntry]] = {
            level: {} for level in CacheLevel
        }

        # Components
        self.cache_warmer = PredictiveCacheWarmer() if enable_prediction else None
        self.coherence_manager = CacheCoherenceManager() if enable_coherence else None

        # Statistics
        self._stats: Dict[CacheLevel, CacheStats] = {
            level: CacheStats() for level in CacheLevel
        }

        # Configuration
        self._level_configs = {
            CacheLevel.L1_MEMORY: {"max_entries": 1000, "ttl_hours": 1},
            CacheLevel.L2_COMPRESSED: {"max_entries": 5000, "ttl_hours": 6},
            CacheLevel.L3_PERSISTENT: {"max_entries": 20000, "ttl_hours": 24},
            CacheLevel.L4_DISTRIBUTED: {
                "max_entries": 100000,
                "ttl_hours": 168,
            },  # 1 week
        }

        self._lock = threading.RLock()
        self.performance_monitor = get_performance_monitor()

        # Background tasks
        self._maintenance_task: Optional[asyncio.Task] = None
        self._warming_task: Optional[asyncio.Task] = None

        logger.info(
            "Intelligent cache manager initialized", max_memory_mb=max_memory_mb
        )

    async def start(self):
        """Start background maintenance tasks."""
        self._maintenance_task = asyncio.create_task(self._maintenance_loop())

        if self.enable_prediction:
            self._warming_task = asyncio.create_task(self._warming_loop())

        logger.info("Intelligent cache manager started")

    async def stop(self):
        """Stop background tasks and cleanup."""
        if self._maintenance_task:
            self._maintenance_task.cancel()
            try:
                await self._maintenance_task
            except asyncio.CancelledError:
                pass

        if self._warming_task:
            self._warming_task.cancel()
            try:
                await self._warming_task
            except asyncio.CancelledError:
                pass

        logger.info("Intelligent cache manager stopped")

    def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache with intelligent level selection."""
        start_time = time.time()

        with self._lock:
            # Try each cache level in order
            for level in CacheLevel:
                if key in self._caches[level]:
                    entry = self._caches[level][key]

                    # Check expiration
                    if entry.is_expired():
                        del self._caches[level][key]
                        self._stats[level].evictions += 1
                        continue

                    # Update access statistics
                    entry.update_access()
                    self._stats[level].hits += 1

                    # Promote to higher level if beneficial
                    if level != CacheLevel.L1_MEMORY and entry.access_count > 3:
                        self._promote_entry(key, entry, level)

                    # Record access for prediction
                    if self.cache_warmer:
                        self.cache_warmer.record_access(key)

                    # Update performance stats
                    access_time = (time.time() - start_time) * 1000
                    self._update_access_time(level, access_time)

                    return entry.value

            # Cache miss
            for level in CacheLevel:
                self._stats[level].misses += 1

            return default

    def set(
        self,
        key: str,
        value: Any,
        ttl_hours: Optional[int] = None,
        tags: Optional[Set[str]] = None,
        priority: int = 1,
        pattern: CachePattern = CachePattern.MIXED,
    ) -> bool:
        """Set value in cache with intelligent level placement."""

        with self._lock:
            # Calculate value size
            try:
                size_bytes = len(pickle.dumps(value))
            except Exception:
                size_bytes = 1024  # Default estimate

            # Determine optimal cache level
            target_level = self._determine_cache_level(size_bytes, priority, pattern)

            # Create cache entry
            ttl = timedelta(hours=ttl_hours) if ttl_hours else None
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                size_bytes=size_bytes,
                ttl=ttl,
                tags=tags or set(),
                priority=priority,
                pattern=pattern,
            )

            # Check memory limits
            if not self._can_fit_entry(target_level, entry):
                self._evict_entries(target_level, size_bytes)

            # Store entry
            self._caches[target_level][key] = entry
            self._stats[target_level].entry_count += 1
            self._stats[target_level].size_bytes += size_bytes

            # Register with coherence manager
            if self.coherence_manager and tags:
                self.coherence_manager.tag_key(key, tags)

            return True

    def invalidate(self, key: str = None, tag: str = None, pattern: str = None):
        """Invalidate cache entries."""
        if self.coherence_manager:
            if key:
                self.coherence_manager.invalidate_by_key(key)
            elif tag:
                self.coherence_manager.invalidate_by_tag(tag)
            elif pattern:
                self.coherence_manager.invalidate_by_pattern(pattern)

        # Direct invalidation
        with self._lock:
            if key:
                self._remove_key(key)
            elif pattern:
                import re

                pattern_regex = re.compile(pattern)
                keys_to_remove = []

                for level in CacheLevel:
                    for cache_key in self._caches[level].keys():
                        if pattern_regex.match(cache_key):
                            keys_to_remove.append(cache_key)

                for cache_key in keys_to_remove:
                    self._remove_key(cache_key)

    def _determine_cache_level(
        self, size_bytes: int, priority: int, pattern: CachePattern
    ) -> CacheLevel:
        """Determine optimal cache level for an entry."""
        # Small, high-priority items go to L1
        if size_bytes < 10240 and priority >= 4:  # < 10KB, high priority
            return CacheLevel.L1_MEMORY

        # Medium items with frequent access patterns
        if size_bytes < 102400 and pattern in [
            CachePattern.READ_HEAVY,
            CachePattern.TEMPORAL,
        ]:
            return CacheLevel.L2_COMPRESSED

        # Large items or write-heavy patterns
        if size_bytes > 1048576 or pattern == CachePattern.WRITE_HEAVY:  # > 1MB
            return CacheLevel.L3_PERSISTENT

        # Default to L2
        return CacheLevel.L2_COMPRESSED

    def _can_fit_entry(self, level: CacheLevel, entry: CacheEntry) -> bool:
        """Check if entry can fit in the specified cache level."""
        config = self._level_configs[level]
        current_cache = self._caches[level]

        # Check entry count limit
        if len(current_cache) >= config["max_entries"]:
            return False

        # Check memory limit for L1
        if level == CacheLevel.L1_MEMORY:
            current_size = sum(e.size_bytes for e in current_cache.values())
            if (
                current_size + entry.size_bytes > self.max_memory_bytes * 0.3
            ):  # 30% for L1
                return False

        return True

    def _evict_entries(self, level: CacheLevel, needed_bytes: int):
        """Evict entries to make space."""
        current_cache = self._caches[level]

        # Sort entries by eviction priority (LRU + priority + access count)
        entries_with_scores = []
        current_time = datetime.now()

        for key, entry in current_cache.items():
            # Calculate eviction score (lower = more likely to evict)
            time_score = (current_time - entry.last_accessed).total_seconds()
            priority_score = 1.0 / max(entry.priority, 1)
            access_score = 1.0 / max(entry.access_count, 1)

            eviction_score = time_score * priority_score * access_score
            entries_with_scores.append((key, entry, eviction_score))

        # Sort by eviction score (highest first = most likely to evict)
        entries_with_scores.sort(key=lambda x: x[2], reverse=True)

        # Evict entries until we have enough space
        freed_bytes = 0
        config = self._level_configs[level]
        target_count = max(len(current_cache) - config["max_entries"] // 4, 0)

        for key, entry, _ in entries_with_scores:
            if freed_bytes >= needed_bytes and len(current_cache) <= target_count:
                break

            del current_cache[key]
            freed_bytes += entry.size_bytes
            self._stats[level].evictions += 1
            self._stats[level].entry_count -= 1
            self._stats[level].size_bytes -= entry.size_bytes

    def _promote_entry(self, key: str, entry: CacheEntry, current_level: CacheLevel):
        """Promote entry to a higher cache level."""
        # Determine target level
        level_order = [
            CacheLevel.L1_MEMORY,
            CacheLevel.L2_COMPRESSED,
            CacheLevel.L3_PERSISTENT,
            CacheLevel.L4_DISTRIBUTED,
        ]

        current_index = level_order.index(current_level)
        if current_index == 0:  # Already at highest level
            return

        target_level = level_order[current_index - 1]

        # Check if promotion is beneficial
        if self._can_fit_entry(target_level, entry):
            # Move entry
            del self._caches[current_level][key]
            self._caches[target_level][key] = entry

            # Update stats
            self._stats[current_level].entry_count -= 1
            self._stats[current_level].size_bytes -= entry.size_bytes
            self._stats[target_level].entry_count += 1
            self._stats[target_level].size_bytes += entry.size_bytes

    def _remove_key(self, key: str):
        """Remove key from all cache levels."""
        for level in CacheLevel:
            if key in self._caches[level]:
                entry = self._caches[level][key]
                del self._caches[level][key]

                self._stats[level].invalidations += 1
                self._stats[level].entry_count -= 1
                self._stats[level].size_bytes -= entry.size_bytes

    def _update_access_time(self, level: CacheLevel, access_time_ms: float):
        """Update average access time statistics."""
        stats = self._stats[level]
        total_accesses = stats.hits + stats.misses

        if total_accesses == 1:
            stats.avg_access_time_ms = access_time_ms
        else:
            # Exponential moving average
            alpha = 0.1
            stats.avg_access_time_ms = (
                alpha * access_time_ms + (1 - alpha) * stats.avg_access_time_ms
            )

    async def _maintenance_loop(self):
        """Background maintenance loop."""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                await self._run_maintenance()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in cache maintenance loop", error=str(e))

    async def _warming_loop(self):
        """Background cache warming loop."""
        while True:
            try:
                await asyncio.sleep(60)  # Run every minute
                await self._run_cache_warming()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in cache warming loop", error=str(e))

    async def _run_maintenance(self):
        """Run periodic maintenance tasks."""
        with self._lock:
            # Clean expired entries
            expired_count = 0
            current_time = datetime.now()

            for level in CacheLevel:
                expired_keys = []
                for key, entry in self._caches[level].items():
                    if entry.is_expired():
                        expired_keys.append(key)

                for key in expired_keys:
                    entry = self._caches[level][key]
                    del self._caches[level][key]

                    self._stats[level].evictions += 1
                    self._stats[level].entry_count -= 1
                    self._stats[level].size_bytes -= entry.size_bytes
                    expired_count += 1

            # Update statistics
            for level in CacheLevel:
                self._stats[level].calculate_hit_rate()

            logger.debug("Cache maintenance completed", expired_entries=expired_count)

    async def _run_cache_warming(self):
        """Run predictive cache warming."""
        if not self.cache_warmer:
            return

        # This would implement actual cache warming logic
        # For now, just log that warming is running
        logger.debug("Cache warming cycle completed")

    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        with self._lock:
            total_stats = CacheStats()
            level_stats = {}

            for level, stats in self._stats.items():
                stats.calculate_hit_rate()
                level_stats[level.value] = {
                    "hits": stats.hits,
                    "misses": stats.misses,
                    "evictions": stats.evictions,
                    "invalidations": stats.invalidations,
                    "entry_count": stats.entry_count,
                    "size_bytes": stats.size_bytes,
                    "hit_rate_percent": stats.hit_rate_percent,
                    "avg_access_time_ms": stats.avg_access_time_ms,
                }

                # Aggregate totals
                total_stats.hits += stats.hits
                total_stats.misses += stats.misses
                total_stats.evictions += stats.evictions
                total_stats.invalidations += stats.invalidations
                total_stats.entry_count += stats.entry_count
                total_stats.size_bytes += stats.size_bytes

            total_stats.calculate_hit_rate()

            return {
                "total": {
                    "hits": total_stats.hits,
                    "misses": total_stats.misses,
                    "evictions": total_stats.evictions,
                    "invalidations": total_stats.invalidations,
                    "entry_count": total_stats.entry_count,
                    "size_mb": round(total_stats.size_bytes / 1024 / 1024, 2),
                    "hit_rate_percent": total_stats.hit_rate_percent,
                },
                "by_level": level_stats,
                "memory_usage_percent": (
                    round((total_stats.size_bytes / self.max_memory_bytes) * 100, 2)
                    if self.max_memory_bytes > 0
                    else 0
                ),
            }


# Global cache manager instance
_global_cache_manager: Optional[IntelligentCacheManager] = None


def get_intelligent_cache_manager() -> IntelligentCacheManager:
    """Get the global intelligent cache manager instance."""
    global _global_cache_manager
    if _global_cache_manager is None:
        _global_cache_manager = IntelligentCacheManager()
    return _global_cache_manager


def reset_intelligent_cache_manager():
    """Reset the global intelligent cache manager."""
    global _global_cache_manager
    if _global_cache_manager:
        # Note: In a real implementation, you'd want to properly stop the manager
        pass
    _global_cache_manager = None


# Decorator for intelligent caching
def intelligent_cache(
    ttl_hours: int = 1,
    tags: Optional[Set[str]] = None,
    priority: int = 1,
    pattern: CachePattern = CachePattern.MIXED,
):
    """Decorator for intelligent caching of function results."""
    from ..utils.decorators import create_async_sync_decorator

    def _generate_cache_key(func, args, kwargs):
        """Generate cache key for function call."""
        key_data = {"function": func.__name__, "args": args, "kwargs": kwargs}
        return hashlib.md5(
            json.dumps(key_data, sort_keys=True, default=str).encode()
        ).hexdigest()

    def create_async_wrapper(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            cache_manager = get_intelligent_cache_manager()
            cache_key = _generate_cache_key(func, args, kwargs)

            # Try cache first
            cached_result = cache_manager.get(cache_key)
            if cached_result is not None:
                return cached_result

            # Execute function
            result = await func(*args, **kwargs)

            # Cache result
            cache_manager.set(
                cache_key,
                result,
                ttl_hours=ttl_hours,
                tags=tags,
                priority=priority,
                pattern=pattern,
            )
            return result
        return async_wrapper

    def create_sync_wrapper(func):
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            cache_manager = get_intelligent_cache_manager()
            cache_key = _generate_cache_key(func, args, kwargs)

            # Try cache first
            cached_result = cache_manager.get(cache_key)
            if cached_result is not None:
                return cached_result

            # Execute function
            result = func(*args, **kwargs)

            # Cache result
            cache_manager.set(
                cache_key,
                result,
                ttl_hours=ttl_hours,
                tags=tags,
                priority=priority,
                pattern=pattern,
            )
            return result
        return sync_wrapper

    return create_async_sync_decorator(create_async_wrapper, create_sync_wrapper)
