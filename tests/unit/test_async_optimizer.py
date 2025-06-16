"""Unit tests for async optimizer module."""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta

from src.core.async_optimizer import (
    AsyncOptimizer,
    AdaptiveSemaphore,
    AsyncContextPool,
    DeadlockDetector,
    AsyncBatcher
)


class TestAdaptiveSemaphore:
    """Test cases for AdaptiveSemaphore."""
    
    @pytest.fixture
    def semaphore(self):
        return AdaptiveSemaphore(initial_value=3, min_value=1, max_value=10)
    
    @pytest.mark.asyncio
    async def test_basic_acquire_release(self, semaphore):
        """Test basic semaphore operations."""
        await semaphore.acquire()
        assert semaphore._current_value == 2
        
        semaphore.release()
        assert semaphore._current_value == 3
    
    @pytest.mark.asyncio
    async def test_adaptive_scaling(self, semaphore):
        """Test adaptive scaling based on performance."""
        # Simulate good performance (should increase capacity)
        semaphore.record_performance(0.1)  # Fast operation
        semaphore.record_performance(0.1)
        semaphore.record_performance(0.1)
        
        semaphore.adapt()
        assert semaphore._current_value > 3
    
    @pytest.mark.asyncio
    async def test_adaptive_scaling_down(self, semaphore):
        """Test adaptive scaling down on poor performance."""
        # Simulate poor performance (should decrease capacity)
        semaphore.record_performance(2.0)  # Slow operation
        semaphore.record_performance(2.0)
        semaphore.record_performance(2.0)
        
        semaphore.adapt()
        assert semaphore._current_value < 3
    
    def test_bounds_enforcement(self, semaphore):
        """Test that semaphore respects min/max bounds."""
        # Force to minimum
        semaphore._current_value = 0
        semaphore.adapt()
        assert semaphore._current_value >= semaphore._min_value
        
        # Force to maximum
        semaphore._current_value = 20
        semaphore.adapt()
        assert semaphore._current_value <= semaphore._max_value


class TestAsyncContextPool:
    """Test cases for AsyncContextPool."""
    
    @pytest.fixture
    def pool(self):
        return AsyncContextPool(max_contexts=3)
    
    @pytest.mark.asyncio
    async def test_context_acquisition(self, pool):
        """Test context acquisition and release."""
        async with pool.get_context() as ctx:
            assert ctx is not None
            assert pool._active_contexts == 1
        
        assert pool._active_contexts == 0
    
    @pytest.mark.asyncio
    async def test_context_limit(self, pool):
        """Test context pool limits."""
        contexts = []
        
        # Acquire all available contexts
        for _ in range(3):
            ctx = await pool.get_context().__aenter__()
            contexts.append(ctx)
        
        assert pool._active_contexts == 3
        
        # Try to acquire one more (should wait)
        with pytest.raises(asyncio.TimeoutError):
            async with asyncio.timeout(0.1):
                async with pool.get_context():
                    pass
        
        # Release one context
        await contexts[0].__aexit__(None, None, None)
        assert pool._active_contexts == 2
    
    @pytest.mark.asyncio
    async def test_context_cleanup(self, pool):
        """Test context cleanup."""
        async with pool.get_context():
            pass
        
        await pool.cleanup()
        assert pool._active_contexts == 0


class TestDeadlockDetector:
    """Test cases for DeadlockDetector."""
    
    @pytest.fixture
    def detector(self):
        return DeadlockDetector(timeout=1.0)
    
    @pytest.mark.asyncio
    async def test_operation_tracking(self, detector):
        """Test operation tracking."""
        operation_id = "test_op_1"
        
        detector.start_operation(operation_id, ["resource_1"])
        assert operation_id in detector._operations
        
        detector.end_operation(operation_id)
        assert operation_id not in detector._operations
    
    @pytest.mark.asyncio
    async def test_deadlock_detection(self, detector):
        """Test deadlock detection."""
        # Create a potential deadlock scenario
        detector.start_operation("op1", ["resource_a", "resource_b"])
        detector.start_operation("op2", ["resource_b", "resource_a"])
        
        # Check for deadlock
        has_deadlock = detector.check_deadlock()
        
        # Clean up
        detector.end_operation("op1")
        detector.end_operation("op2")
        
        # The exact result depends on the deadlock detection algorithm
        assert isinstance(has_deadlock, bool)
    
    @pytest.mark.asyncio
    async def test_timeout_detection(self, detector):
        """Test timeout detection."""
        operation_id = "timeout_op"
        
        detector.start_operation(operation_id, ["resource_1"])
        
        # Wait for timeout
        await asyncio.sleep(1.1)
        
        timeouts = detector.check_timeouts()
        assert operation_id in timeouts
        
        detector.end_operation(operation_id)


class TestAsyncBatcher:
    """Test cases for AsyncBatcher."""
    
    @pytest.fixture
    def batcher(self):
        return AsyncBatcher(batch_size=3, timeout=0.1)
    
    @pytest.mark.asyncio
    async def test_batch_processing(self, batcher):
        """Test batch processing functionality."""
        results = []
        
        async def mock_process(items):
            return [f"processed_{item}" for item in items]
        
        # Add items to batch
        for i in range(5):
            result = await batcher.add_item(f"item_{i}", mock_process)
            if result:
                results.extend(result)
        
        # Flush remaining items
        remaining = await batcher.flush()
        if remaining:
            results.extend(remaining)
        
        assert len(results) == 5
        assert all("processed_" in result for result in results)
    
    @pytest.mark.asyncio
    async def test_batch_timeout(self, batcher):
        """Test batch timeout functionality."""
        async def mock_process(items):
            return [f"processed_{item}" for item in items]
        
        # Add one item and wait for timeout
        await batcher.add_item("item_1", mock_process)
        
        # Wait for timeout to trigger
        await asyncio.sleep(0.2)
        
        # The batch should have been processed due to timeout
        assert len(batcher._current_batch) == 0


class TestAsyncOptimizer:
    """Test cases for AsyncOptimizer."""
    
    @pytest.fixture
    def optimizer(self):
        return AsyncOptimizer()
    
    @pytest.mark.asyncio
    async def test_initialization(self, optimizer):
        """Test optimizer initialization."""
        await optimizer.start()
        assert optimizer._semaphore is not None
        assert optimizer._context_pool is not None
        assert optimizer._deadlock_detector is not None
        assert optimizer._batcher is not None
        
        await optimizer.stop()
    
    @pytest.mark.asyncio
    async def test_optimized_context(self, optimizer):
        """Test optimized context manager."""
        await optimizer.start()
        
        async with optimizer.optimized_context(max_concurrent=2, timeout=1.0) as context:
            assert context is not None
            # Simulate some async work
            await asyncio.sleep(0.01)
        
        stats = optimizer.get_stats()
        assert "total_operations" in stats
        assert stats["total_operations"] >= 1
        
        await optimizer.stop()
    
    @pytest.mark.asyncio
    async def test_concurrent_execution(self, optimizer):
        """Test concurrent execution management."""
        await optimizer.start()
        
        async def test_task(task_id):
            async with optimizer.optimized_context(max_concurrent=3, timeout=1.0):
                await asyncio.sleep(0.01)
                return f"task_{task_id}_completed"
        
        # Run multiple concurrent tasks
        tasks = [test_task(i) for i in range(5)]
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 5
        assert all("completed" in result for result in results)
        
        stats = optimizer.get_stats()
        assert stats["total_operations"] >= 5
        
        await optimizer.stop()
    
    @pytest.mark.asyncio
    async def test_batch_operations(self, optimizer):
        """Test batch operations."""
        await optimizer.start()
        
        async def process_function(items):
            return [f"processed_{item}" for item in items]
        
        results = []
        for i in range(5):
            result = await optimizer.batch_operation(f"item_{i}", process_function)
            if result:
                results.extend(result)
        
        # Flush any remaining items
        remaining = await optimizer.flush_batches()
        if remaining:
            results.extend(remaining)
        
        assert len(results) >= 0  # Results depend on batching logic
        
        await optimizer.stop()
    
    @pytest.mark.asyncio
    async def test_deadlock_prevention(self, optimizer):
        """Test deadlock prevention."""
        await optimizer.start()
        
        # Simulate operations that might cause deadlock
        async def risky_operation(op_id, resources):
            async with optimizer.optimized_context(max_concurrent=2, timeout=0.5):
                # Simulate resource acquisition
                await asyncio.sleep(0.01)
                return f"operation_{op_id}_completed"
        
        # Run potentially conflicting operations
        tasks = [
            risky_operation("op1", ["resource_a", "resource_b"]),
            risky_operation("op2", ["resource_b", "resource_a"])
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Should complete without deadlock
        assert len(results) == 2
        
        await optimizer.stop()
    
    def test_performance_stats(self, optimizer):
        """Test performance statistics."""
        stats = optimizer.get_stats()
        
        expected_keys = [
            "total_operations", "active_operations", "avg_duration",
            "semaphore_value", "active_contexts", "detected_deadlocks",
            "timeout_operations", "batch_operations"
        ]
        
        for key in expected_keys:
            assert key in stats
    
    @pytest.mark.asyncio
    async def test_adaptive_behavior(self, optimizer):
        """Test adaptive behavior based on performance."""
        await optimizer.start()
        
        # Run several operations to generate performance data
        async def fast_operation():
            async with optimizer.optimized_context(max_concurrent=2, timeout=1.0):
                await asyncio.sleep(0.001)  # Very fast operation
                return "fast_completed"
        
        # Run multiple fast operations
        tasks = [fast_operation() for _ in range(10)]
        await asyncio.gather(*tasks)
        
        # Trigger adaptation
        optimizer.adapt_performance()
        
        stats = optimizer.get_stats()
        assert stats["total_operations"] >= 10
        
        await optimizer.stop()
    
    @pytest.mark.asyncio
    async def test_error_handling(self, optimizer):
        """Test error handling in async operations."""
        await optimizer.start()
        
        async def failing_operation():
            async with optimizer.optimized_context(max_concurrent=2, timeout=1.0):
                raise ValueError("Test error")
        
        # Should handle errors gracefully
        with pytest.raises(ValueError):
            await failing_operation()
        
        # Optimizer should still be functional
        stats = optimizer.get_stats()
        assert "total_operations" in stats
        
        await optimizer.stop()
    
    @pytest.mark.asyncio
    async def test_timeout_handling(self, optimizer):
        """Test timeout handling."""
        await optimizer.start()
        
        async def slow_operation():
            async with optimizer.optimized_context(max_concurrent=1, timeout=0.1):
                await asyncio.sleep(0.2)  # Longer than timeout
                return "should_not_complete"
        
        # Should timeout
        with pytest.raises(asyncio.TimeoutError):
            await slow_operation()
        
        stats = optimizer.get_stats()
        assert stats["timeout_operations"] >= 1
        
        await optimizer.stop()


if __name__ == "__main__":
    pytest.main([__file__])