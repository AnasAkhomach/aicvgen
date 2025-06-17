"""Unit tests for caching strategy module."""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta

from src.core.caching_strategy import (
    IntelligentCache,
    MemoryCache,
    CompressedCache,
    PersistentCache,
    DistributedCache,
    PredictiveCache,
    CacheCoherence,
    CacheAnalytics,
    intelligent_cache,
)


class TestMemoryCache:
    """Test cases for MemoryCache."""

    @pytest.fixture
    def cache(self):
        return MemoryCache(max_size=100, ttl=60)

    def test_basic_operations(self, cache):
        """Test basic cache operations."""
        # Test set and get
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

        # Test miss
        assert cache.get("missing_key") is None

        # Test delete
        cache.delete("key1")
        assert cache.get("key1") is None

    def test_ttl_expiration(self, cache):
        """Test TTL expiration."""
        with patch("time.time", side_effect=[0, 0, 61]):
            cache.set("ttl_key", "ttl_value")

            # Should be available immediately
            assert cache.get("ttl_key") == "ttl_value"

            # Should be expired after TTL
            assert cache.get("ttl_key") is None

    def test_size_limit(self, cache):
        """Test cache size limits."""
        # Fill cache to capacity
        for i in range(100):
            cache.set(f"key_{i}", f"value_{i}")

        assert cache.size() == 100

        # Add one more item (should evict oldest)
        cache.set("new_key", "new_value")
        assert cache.size() == 100
        assert cache.get("new_key") == "new_value"
        assert cache.get("key_0") is None  # Should be evicted

    def test_clear(self, cache):
        """Test cache clearing."""
        cache.set("key1", "value1")
        cache.set("key2", "value2")

        cache.clear()
        assert cache.size() == 0
        assert cache.get("key1") is None
        assert cache.get("key2") is None


class TestCompressedCache:
    """Test cases for CompressedCache."""

    @pytest.fixture
    def cache(self):
        return CompressedCache(max_size=50, compression_threshold=10)

    def test_compression(self, cache):
        """Test data compression."""
        large_data = "x" * 100  # Large enough to trigger compression

        cache.set("large_key", large_data)
        retrieved_data = cache.get("large_key")

        assert retrieved_data == large_data
        assert cache._compressed_count > 0

    def test_no_compression_small_data(self, cache):
        """Test that small data is not compressed."""
        small_data = "small"

        cache.set("small_key", small_data)
        retrieved_data = cache.get("small_key")

        assert retrieved_data == small_data

    def test_compression_stats(self, cache):
        """Test compression statistics."""
        # Add some data to trigger compression
        cache.set("large1", "x" * 100)
        cache.set("large2", "y" * 100)
        cache.set("small", "small")

        stats = cache.get_stats()
        assert "compressed_items" in stats
        assert "compression_ratio" in stats
        assert stats["compressed_items"] >= 2


class TestPersistentCache:
    """Test cases for PersistentCache."""

    @pytest.fixture
    def cache(self, tmp_path):
        cache_file = tmp_path / "test_cache.db"
        return PersistentCache(str(cache_file), max_size=100)

    def test_persistence(self, cache):
        """Test data persistence."""
        cache.set("persistent_key", "persistent_value")

        # Create new cache instance with same file
        new_cache = PersistentCache(cache._cache_file, max_size=100)

        assert new_cache.get("persistent_key") == "persistent_value"

    def test_sync_operations(self, cache):
        """Test sync operations."""
        cache.set("sync_key", "sync_value")
        cache.sync()

        # Data should be persisted
        assert cache.get("sync_key") == "sync_value"

    def test_cleanup_expired(self, cache):
        """Test cleanup of expired entries."""
        with patch("time.time", side_effect=[0, 0, 61]):
            cache.set("expire_key", "expire_value", ttl=60)

            # Should be available initially
            assert cache.get("expire_key") == "expire_value"

            # Should be cleaned up after expiration
            cache._cleanup_expired()
            assert cache.get("expire_key") is None


class TestDistributedCache:
    """Test cases for DistributedCache."""

    @pytest.fixture
    def cache(self):
        return DistributedCache(nodes=["node1", "node2", "node3"])

    def test_node_selection(self, cache):
        """Test consistent node selection."""
        node1 = cache._get_node("key1")
        node2 = cache._get_node("key1")  # Same key should go to same node

        assert node1 == node2
        assert node1 in cache._nodes

    def test_replication(self, cache):
        """Test data replication."""
        with patch.object(cache, "_store_on_node") as mock_store:
            cache.set("replicated_key", "replicated_value")

            # Should store on primary and replica nodes
            assert mock_store.call_count >= 2

    def test_failover(self, cache):
        """Test failover to replica nodes."""
        with patch.object(cache, "_get_from_node", side_effect=[None, "backup_value"]):
            value = cache.get("failover_key")
            assert value == "backup_value"


class TestPredictiveCache:
    """Test cases for PredictiveCache."""

    @pytest.fixture
    def cache(self):
        return PredictiveCache(max_size=100, prediction_window=60)

    def test_access_pattern_learning(self, cache):
        """Test access pattern learning."""
        # Simulate access pattern
        for _ in range(5):
            cache.get("pattern_key")

        patterns = cache._access_patterns
        assert "pattern_key" in patterns
        assert patterns["pattern_key"]["count"] == 5

    def test_predictive_loading(self, cache):
        """Test predictive cache loading."""
        # Set up access pattern
        cache._access_patterns["predict_key"] = {
            "count": 10,
            "last_access": time.time(),
            "intervals": [60, 60, 60],  # Regular 60-second intervals
        }

        with patch.object(cache, "_load_data") as mock_load:
            cache._predict_and_load()
            # Should attempt to load predicted keys
            mock_load.assert_called()

    def test_prediction_accuracy(self, cache):
        """Test prediction accuracy tracking."""
        cache._record_prediction("test_key", True)
        cache._record_prediction("test_key2", False)

        stats = cache.get_stats()
        assert "prediction_accuracy" in stats
        assert 0 <= stats["prediction_accuracy"] <= 1


class TestCacheCoherence:
    """Test cases for CacheCoherence."""

    @pytest.fixture
    def coherence(self):
        return CacheCoherence()

    def test_invalidation_tracking(self, coherence):
        """Test invalidation tracking."""
        coherence.invalidate("test_key", "cache1")

        assert "test_key" in coherence._invalidations
        assert "cache1" in coherence._invalidations["test_key"]

    def test_version_management(self, coherence):
        """Test version management."""
        coherence.update_version("versioned_key", 1)
        assert coherence.get_version("versioned_key") == 1

        coherence.update_version("versioned_key", 2)
        assert coherence.get_version("versioned_key") == 2

    def test_consistency_check(self, coherence):
        """Test consistency checking."""
        coherence.update_version("consistent_key", 1)

        # Same version should be consistent
        assert coherence.is_consistent("consistent_key", 1)

        # Different version should be inconsistent
        assert not coherence.is_consistent("consistent_key", 0)


class TestCacheAnalytics:
    """Test cases for CacheAnalytics."""

    @pytest.fixture
    def analytics(self):
        return CacheAnalytics()

    def test_hit_rate_calculation(self, analytics):
        """Test hit rate calculation."""
        analytics.record_hit("key1")
        analytics.record_hit("key1")
        analytics.record_miss("key2")

        hit_rate = analytics.get_hit_rate()
        assert hit_rate == 2 / 3  # 2 hits out of 3 total accesses

    def test_popular_keys_tracking(self, analytics):
        """Test popular keys tracking."""
        analytics.record_hit("popular_key")
        analytics.record_hit("popular_key")
        analytics.record_hit("popular_key")
        analytics.record_hit("other_key")

        popular_keys = analytics.get_popular_keys(limit=1)
        assert popular_keys[0][0] == "popular_key"
        assert popular_keys[0][1] == 3

    def test_performance_metrics(self, analytics):
        """Test performance metrics."""
        analytics.record_operation("get", 0.1)
        analytics.record_operation("set", 0.2)
        analytics.record_operation("get", 0.15)

        metrics = analytics.get_performance_metrics()
        assert "avg_get_time" in metrics
        assert "avg_set_time" in metrics
        assert metrics["avg_get_time"] == 0.125  # (0.1 + 0.15) / 2


class TestIntelligentCache:
    """Test cases for IntelligentCache."""

    @pytest.fixture
    def cache(self):
        return IntelligentCache()

    @pytest.mark.asyncio
    async def test_initialization(self, cache):
        """Test cache initialization."""
        await cache.start()
        assert cache._memory_cache is not None
        assert cache._compressed_cache is not None
        assert cache._analytics is not None

        await cache.stop()

    @pytest.mark.asyncio
    async def test_intelligent_storage(self, cache):
        """Test intelligent storage selection."""
        await cache.start()

        # Small data should go to memory cache
        await cache.set("small_key", "small_value")
        value = await cache.get("small_key")
        assert value == "small_value"

        # Large data should go to compressed cache
        large_data = "x" * 1000
        await cache.set("large_key", large_data)
        value = await cache.get("large_key")
        assert value == large_data

        await cache.stop()

    @pytest.mark.asyncio
    async def test_cache_promotion(self, cache):
        """Test cache level promotion."""
        await cache.start()

        # Set data in compressed cache
        await cache.set("promote_key", "promote_value")

        # Access multiple times to trigger promotion
        for _ in range(5):
            await cache.get("promote_key")

        # Should be promoted to memory cache
        stats = await cache.get_stats()
        assert stats["memory_cache"]["size"] > 0

        await cache.stop()

    @pytest.mark.asyncio
    async def test_cache_eviction(self, cache):
        """Test intelligent cache eviction."""
        await cache.start()

        # Fill memory cache
        for i in range(150):  # More than memory cache capacity
            await cache.set(f"key_{i}", f"value_{i}")

        # Some items should be evicted to compressed cache
        stats = await cache.get_stats()
        assert stats["compressed_cache"]["size"] > 0

        await cache.stop()

    @pytest.mark.asyncio
    async def test_predictive_loading(self, cache):
        """Test predictive loading."""
        await cache.start()

        # Create access pattern
        for _ in range(3):
            await cache.get("predictive_key")
            await asyncio.sleep(0.01)

        # Trigger prediction
        await cache._run_predictions()

        stats = await cache.get_stats()
        assert "predictions_made" in stats

        await cache.stop()

    def test_cache_decorator(self):
        """Test intelligent cache decorator."""
        call_count = 0

        @intelligent_cache(ttl=60, key_func=lambda x: f"test_{x}")
        def expensive_function(value):
            nonlocal call_count
            call_count += 1
            return f"result_{value}"

        # First call should execute function
        result1 = expensive_function("test")
        assert result1 == "result_test"
        assert call_count == 1

        # Second call should use cache
        result2 = expensive_function("test")
        assert result2 == "result_test"
        assert call_count == 1  # Should not increment

    @pytest.mark.asyncio
    async def test_async_cache_decorator(self):
        """Test async intelligent cache decorator."""
        call_count = 0

        @intelligent_cache(ttl=60, key_func=lambda x: f"async_test_{x}")
        async def expensive_async_function(value):
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.01)
            return f"async_result_{value}"

        # First call should execute function
        result1 = await expensive_async_function("test")
        assert result1 == "async_result_test"
        assert call_count == 1

        # Second call should use cache
        result2 = await expensive_async_function("test")
        assert result2 == "async_result_test"
        assert call_count == 1  # Should not increment

    @pytest.mark.asyncio
    async def test_comprehensive_stats(self, cache):
        """Test comprehensive statistics."""
        await cache.start()

        # Perform various operations
        await cache.set("stats_key1", "value1")
        await cache.set("stats_key2", "value2")
        await cache.get("stats_key1")
        await cache.get("missing_key")

        stats = await cache.get_stats()

        expected_keys = [
            "memory_cache",
            "compressed_cache",
            "persistent_cache",
            "total_operations",
            "hit_rate",
            "predictions_made",
            "cache_levels",
            "popular_keys",
        ]

        for key in expected_keys:
            assert key in stats

        await cache.stop()


if __name__ == "__main__":
    pytest.main([__file__])
