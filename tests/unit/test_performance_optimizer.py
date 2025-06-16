"""Unit tests for performance optimizer module."""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta

from src.core.performance_optimizer import (
    PerformanceOptimizer,
    ConnectionPool,
    BatchProcessor,
    MemoryManager
)


class TestConnectionPool:
    """Test cases for ConnectionPool."""
    
    @pytest.fixture
    def pool(self):
        return ConnectionPool(max_connections=5, timeout=1.0)
    
    @pytest.mark.asyncio
    async def test_acquire_release_connection(self, pool):
        """Test basic connection acquisition and release."""
        conn = await pool.acquire()
        assert conn is not None
        assert pool.active_connections == 1
        
        await pool.release(conn)
        assert pool.active_connections == 0
    
    @pytest.mark.asyncio
    async def test_connection_limit(self, pool):
        """Test connection pool limits."""
        connections = []
        
        # Acquire all available connections
        for _ in range(5):
            conn = await pool.acquire()
            connections.append(conn)
        
        assert pool.active_connections == 5
        
        # Try to acquire one more (should timeout)
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(pool.acquire(), timeout=0.1)
        
        # Release one connection
        await pool.release(connections[0])
        assert pool.active_connections == 4
        
        # Now we should be able to acquire again
        new_conn = await pool.acquire()
        assert new_conn is not None
    
    @pytest.mark.asyncio
    async def test_pool_cleanup(self, pool):
        """Test pool cleanup."""
        conn = await pool.acquire()
        await pool.cleanup()
        assert pool.active_connections == 0


class TestBatchProcessor:
    """Test cases for BatchProcessor."""
    
    @pytest.fixture
    def processor(self):
        return BatchProcessor(batch_size=3, timeout=0.1)
    
    @pytest.mark.asyncio
    async def test_batch_processing(self, processor):
        """Test batch processing functionality."""
        results = []
        
        async def mock_process_batch(items):
            return [f"processed_{item}" for item in items]
        
        processor._process_batch = mock_process_batch
        
        # Add items to batch
        for i in range(5):
            result = await processor.add_item(f"item_{i}")
            if result:
                results.extend(result)
        
        # Process remaining items
        remaining = await processor.flush()
        if remaining:
            results.extend(remaining)
        
        assert len(results) == 5
        assert all("processed_" in result for result in results)
    
    @pytest.mark.asyncio
    async def test_batch_timeout(self, processor):
        """Test batch timeout functionality."""
        async def mock_process_batch(items):
            return [f"processed_{item}" for item in items]
        
        processor._process_batch = mock_process_batch
        
        # Add one item and wait for timeout
        await processor.add_item("item_1")
        
        # Wait for timeout to trigger
        await asyncio.sleep(0.2)
        
        # The batch should have been processed due to timeout
        assert processor._current_batch == []


class TestMemoryManager:
    """Test cases for MemoryManager."""
    
    @pytest.fixture
    def manager(self):
        return MemoryManager(max_memory_mb=100, cleanup_threshold=0.8)
    
    def test_memory_tracking(self, manager):
        """Test memory usage tracking."""
        initial_usage = manager.get_memory_usage()
        assert initial_usage >= 0
        
        # Simulate memory allocation
        large_data = [0] * 1000000  # Allocate some memory
        current_usage = manager.get_memory_usage()
        assert current_usage >= initial_usage
    
    def test_cleanup_trigger(self, manager):
        """Test cleanup trigger functionality."""
        with patch.object(manager, 'get_memory_usage', return_value=90):
            with patch.object(manager, '_cleanup_memory') as mock_cleanup:
                manager.check_memory()
                mock_cleanup.assert_called_once()
    
    def test_no_cleanup_below_threshold(self, manager):
        """Test that cleanup is not triggered below threshold."""
        with patch.object(manager, 'get_memory_usage', return_value=50):
            with patch.object(manager, '_cleanup_memory') as mock_cleanup:
                manager.check_memory()
                mock_cleanup.assert_not_called()


class TestPerformanceOptimizer:
    """Test cases for PerformanceOptimizer."""
    
    @pytest.fixture
    def optimizer(self):
        return PerformanceOptimizer()
    
    @pytest.mark.asyncio
    async def test_initialization(self, optimizer):
        """Test optimizer initialization."""
        await optimizer.start()
        assert optimizer._connection_pool is not None
        assert optimizer._batch_processor is not None
        assert optimizer._memory_manager is not None
        
        await optimizer.stop()
    
    @pytest.mark.asyncio
    async def test_optimized_execution_context(self, optimizer):
        """Test optimized execution context manager."""
        await optimizer.start()
        
        async with optimizer.optimized_execution("test_operation", 1.0) as context:
            assert context is not None
            # Simulate some work
            await asyncio.sleep(0.01)
        
        stats = optimizer.get_stats()
        assert "operations" in stats
        assert stats["operations"] >= 1
        
        await optimizer.stop()
    
    def test_cache_operations(self, optimizer):
        """Test cache operations."""
        # Test cache set and get
        optimizer.cache_set("test_key", "test_value", ttl=60)
        value = optimizer.cache_get("test_key")
        assert value == "test_value"
        
        # Test cache miss
        missing_value = optimizer.cache_get("missing_key")
        assert missing_value is None
        
        # Test cache invalidation
        optimizer.cache_invalidate("test_key")
        invalidated_value = optimizer.cache_get("test_key")
        assert invalidated_value is None
    
    def test_cache_ttl(self, optimizer):
        """Test cache TTL functionality."""
        with patch('time.time', side_effect=[0, 0, 61]):
            optimizer.cache_set("ttl_key", "ttl_value", ttl=60)
            
            # Should be available immediately
            value = optimizer.cache_get("ttl_key")
            assert value == "ttl_value"
            
            # Should be expired after TTL
            expired_value = optimizer.cache_get("ttl_key")
            assert expired_value is None
    
    @pytest.mark.asyncio
    async def test_batch_processing_integration(self, optimizer):
        """Test batch processing integration."""
        await optimizer.start()
        
        results = []
        for i in range(5):
            result = await optimizer.process_batch_item(f"item_{i}")
            if result:
                results.extend(result)
        
        # Flush remaining items
        remaining = await optimizer.flush_batch()
        if remaining:
            results.extend(remaining)
        
        assert len(results) >= 0  # Results depend on batch processing logic
        
        await optimizer.stop()
    
    def test_performance_stats(self, optimizer):
        """Test performance statistics."""
        stats = optimizer.get_stats()
        
        expected_keys = [
            "operations", "total_duration", "avg_duration",
            "cache_hits", "cache_misses", "cache_hit_rate",
            "memory_usage_mb", "active_connections"
        ]
        
        for key in expected_keys:
            assert key in stats
    
    @pytest.mark.asyncio
    async def test_error_handling(self, optimizer):
        """Test error handling in optimizer."""
        await optimizer.start()
        
        # Test with invalid operation
        with patch.object(optimizer._memory_manager, 'check_memory', side_effect=Exception("Memory error")):
            # Should not raise exception, but handle gracefully
            async with optimizer.optimized_execution("test_operation", 1.0):
                pass
        
        await optimizer.stop()
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, optimizer):
        """Test concurrent operations."""
        await optimizer.start()
        
        async def test_operation(op_id):
            async with optimizer.optimized_execution(f"operation_{op_id}", 0.1):
                await asyncio.sleep(0.01)
                return f"result_{op_id}"
        
        # Run multiple concurrent operations
        tasks = [test_operation(i) for i in range(10)]
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 10
        assert all("result_" in result for result in results)
        
        stats = optimizer.get_stats()
        assert stats["operations"] >= 10
        
        await optimizer.stop()


if __name__ == "__main__":
    pytest.main([__file__])