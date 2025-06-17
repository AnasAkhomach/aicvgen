"""Unit tests for performance optimization utilities."""

import pytest
import asyncio
import time
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from src.utils.performance import (
    PerformanceMetrics,
    PerformanceMonitor,
    MemoryOptimizer,
    BatchProcessor,
    get_performance_monitor,
    get_memory_optimizer,
    get_batch_processor,
    monitor_performance,
    auto_memory_optimize,
)


class TestPerformanceMetrics:
    """Test PerformanceMetrics dataclass."""

    def test_performance_metrics_creation(self):
        """Test creating performance metrics."""
        metrics = PerformanceMetrics(
            operation_name="test_operation",
            start_time=1000.0,
            end_time=1001.5,
            duration=1.5,
            memory_before=100.0,
            memory_after=105.0,
            memory_delta=5.0,
            cpu_percent=25.5,
        )

        assert metrics.operation_name == "test_operation"
        assert metrics.duration == 1.5
        assert metrics.duration_ms == 1500.0
        assert metrics.memory_delta == 5.0
        assert metrics.success is True
        assert metrics.error_message is None

    def test_performance_metrics_with_error(self):
        """Test creating performance metrics with error."""
        metrics = PerformanceMetrics(
            operation_name="failed_operation",
            start_time=1000.0,
            end_time=1001.0,
            duration=1.0,
            memory_before=100.0,
            memory_after=100.0,
            memory_delta=0.0,
            cpu_percent=15.0,
            success=False,
            error_message="Test error",
        )

        assert metrics.success is False
        assert metrics.error_message == "Test error"


class TestPerformanceMonitor:
    """Test PerformanceMonitor class."""

    @pytest.fixture
    def monitor(self):
        """Create a performance monitor for testing."""
        return PerformanceMonitor(max_history=10)

    @patch("src.utils.performance.psutil.Process")
    @patch("src.utils.performance.psutil.cpu_percent")
    def test_monitor_initialization(self, mock_cpu_percent, mock_process, monitor):
        """Test monitor initialization."""
        assert monitor.max_history == 10
        assert len(monitor.metrics_history) == 0
        assert len(monitor.operation_stats) == 0
        assert "llm_call" in monitor.thresholds
        assert "agent_execution" in monitor.thresholds

    @patch("src.utils.performance.psutil.Process")
    @patch("src.utils.performance.psutil.cpu_percent")
    def test_measure_sync_success(self, mock_cpu_percent, mock_process, monitor):
        """Test synchronous measurement context manager."""
        mock_process.return_value.memory_info.return_value.rss = (
            100 * 1024 * 1024
        )  # 100MB
        mock_cpu_percent.return_value = 25.0

        with monitor.measure_sync("test_operation", test_param="value"):
            time.sleep(0.01)  # Small delay

        assert len(monitor.metrics_history) == 1
        metrics = monitor.metrics_history[0]
        assert metrics.operation_name == "test_operation"
        assert metrics.success is True
        assert metrics.duration > 0
        assert "test_param" in metrics.metadata

    @patch("src.utils.performance.psutil.Process")
    @patch("src.utils.performance.psutil.cpu_percent")
    def test_measure_sync_failure(self, mock_cpu_percent, mock_process, monitor):
        """Test synchronous measurement with exception."""
        mock_process.return_value.memory_info.return_value.rss = 100 * 1024 * 1024
        mock_cpu_percent.return_value = 25.0

        with pytest.raises(ValueError):
            with monitor.measure_sync("failing_operation"):
                raise ValueError("Test error")

        assert len(monitor.metrics_history) == 1
        metrics = monitor.metrics_history[0]
        assert metrics.operation_name == "failing_operation"
        assert metrics.success is False
        assert metrics.error_message == "Test error"

    @patch("src.utils.performance.psutil.Process")
    @patch("src.utils.performance.psutil.cpu_percent")
    @pytest.mark.asyncio
    async def test_measure_async_success(self, mock_cpu_percent, mock_process, monitor):
        """Test asynchronous measurement context manager."""
        mock_process.return_value.memory_info.return_value.rss = 100 * 1024 * 1024
        mock_cpu_percent.return_value = 25.0

        async with monitor.measure_async("async_operation", test_param="async_value"):
            await asyncio.sleep(0.01)

        assert len(monitor.metrics_history) == 1
        metrics = monitor.metrics_history[0]
        assert metrics.operation_name == "async_operation"
        assert metrics.success is True
        assert metrics.duration > 0
        assert "test_param" in metrics.metadata

    def test_get_operation_stats(self, monitor):
        """Test getting operation statistics."""
        # Add some mock metrics
        monitor.operation_stats["test_op"] = [1.0, 2.0, 1.5]

        stats = monitor.get_operation_stats("test_op")

        assert stats["operation"] == "test_op"
        assert stats["count"] == 3
        assert stats["total_duration"] == 4.5
        assert stats["average_duration"] == 1.5
        assert stats["min_duration"] == 1.0
        assert stats["max_duration"] == 2.0
        assert stats["last_duration"] == 1.5

    def test_get_operation_stats_empty(self, monitor):
        """Test getting stats for non-existent operation."""
        stats = monitor.get_operation_stats("nonexistent")

        assert stats["operation"] == "nonexistent"
        assert stats["count"] == 0

    @patch("src.utils.performance.psutil.Process")
    @patch("src.utils.performance.psutil.cpu_percent")
    def test_get_overall_stats(self, mock_cpu_percent, mock_process, monitor):
        """Test getting overall statistics."""
        mock_process.return_value.memory_info.return_value.rss = 100 * 1024 * 1024
        mock_cpu_percent.return_value = 25.0

        # Add some mock metrics
        for i in range(5):
            metrics = PerformanceMetrics(
                operation_name=f"op_{i}",
                start_time=1000.0 + i,
                end_time=1001.0 + i,
                duration=1.0,
                memory_before=100.0,
                memory_after=100.0,
                memory_delta=0.0,
                cpu_percent=25.0,
                success=i < 4,  # One failure
            )
            monitor.metrics_history.append(metrics)

        stats = monitor.get_overall_stats()

        assert stats["total_operations"] == 5
        assert stats["successful_operations"] == 4
        assert stats["failed_operations"] == 1
        assert stats["success_rate_percent"] == 80.0
        assert stats["average_duration_seconds"] == 1.0
        assert "uptime_seconds" in stats

    def test_export_metrics(self, monitor):
        """Test exporting metrics to file."""
        # Add some mock data
        monitor.operation_stats["test_op"] = [1.0, 2.0]

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            filepath = f.name

        try:
            monitor.export_metrics(filepath)

            # Verify file was created and contains expected data
            assert os.path.exists(filepath)

            with open(filepath, "r") as f:
                import json

                data = json.load(f)

            assert "export_timestamp" in data
            assert "overall_stats" in data
            assert "operation_stats" in data
            assert "recent_metrics" in data
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)

    def test_clear_metrics(self, monitor):
        """Test clearing metrics."""
        # Add some data
        monitor.operation_stats["test_op"] = [1.0, 2.0]
        monitor.metrics_history.append(
            PerformanceMetrics(
                operation_name="test",
                start_time=1000.0,
                end_time=1001.0,
                duration=1.0,
                memory_before=100.0,
                memory_after=100.0,
                memory_delta=0.0,
                cpu_percent=25.0,
            )
        )

        monitor.clear_metrics()

        assert len(monitor.metrics_history) == 0
        assert len(monitor.operation_stats) == 0


class TestMemoryOptimizer:
    """Test MemoryOptimizer class."""

    @pytest.fixture
    def optimizer(self):
        """Create a memory optimizer for testing."""
        return MemoryOptimizer()

    @patch("src.utils.performance.psutil.Process")
    @patch("src.utils.performance.psutil.virtual_memory")
    def test_get_memory_info(self, mock_virtual_memory, mock_process, optimizer):
        """Test getting memory information."""
        # Mock memory info
        mock_process.return_value.memory_info.return_value.rss = (
            100 * 1024 * 1024
        )  # 100MB
        mock_process.return_value.memory_info.return_value.vms = (
            200 * 1024 * 1024
        )  # 200MB
        mock_process.return_value.memory_percent.return_value = 5.0
        mock_virtual_memory.return_value.available = 1000 * 1024 * 1024  # 1GB
        mock_virtual_memory.return_value.total = 2000 * 1024 * 1024  # 2GB

        memory_info = optimizer.get_memory_info()

        assert memory_info["rss_mb"] == 100.0
        assert memory_info["vms_mb"] == 200.0
        assert memory_info["percent"] == 5.0
        assert memory_info["available_mb"] == 1000.0
        assert memory_info["total_mb"] == 2000.0

    @patch("src.utils.performance.psutil.Process")
    @patch("src.utils.performance.psutil.virtual_memory")
    def test_should_run_gc_memory_threshold(
        self, mock_virtual_memory, mock_process, optimizer
    ):
        """Test GC trigger based on memory threshold."""
        mock_process.return_value.memory_info.return_value.rss = (
            150 * 1024 * 1024
        )  # 150MB > threshold
        mock_process.return_value.memory_info.return_value.vms = 200 * 1024 * 1024
        mock_process.return_value.memory_percent.return_value = 7.5
        mock_virtual_memory.return_value.available = 1000 * 1024 * 1024
        mock_virtual_memory.return_value.total = 2000 * 1024 * 1024

        assert optimizer.should_run_gc() is True

    @patch("src.utils.performance.psutil.Process")
    @patch("src.utils.performance.psutil.virtual_memory")
    def test_should_run_gc_time_threshold(
        self, mock_virtual_memory, mock_process, optimizer
    ):
        """Test GC trigger based on time threshold."""
        mock_process.return_value.memory_info.return_value.rss = (
            50 * 1024 * 1024
        )  # 50MB < threshold
        mock_process.return_value.memory_info.return_value.vms = 100 * 1024 * 1024
        mock_process.return_value.memory_percent.return_value = 2.5
        mock_virtual_memory.return_value.available = 1000 * 1024 * 1024
        mock_virtual_memory.return_value.total = 2000 * 1024 * 1024

        # Set last GC time to past
        optimizer.last_gc_time = time.time() - 400  # 400 seconds ago > 300s interval

        assert optimizer.should_run_gc() is True

    @patch("src.utils.performance.gc.collect")
    @patch("src.utils.performance.psutil.Process")
    @patch("src.utils.performance.psutil.virtual_memory")
    def test_optimize_memory(
        self, mock_virtual_memory, mock_process, mock_gc_collect, optimizer
    ):
        """Test memory optimization."""
        # Mock memory before optimization
        mock_process.return_value.memory_info.return_value.rss = (
            150 * 1024 * 1024
        )  # 150MB
        mock_process.return_value.memory_info.return_value.vms = 200 * 1024 * 1024
        mock_process.return_value.memory_percent.return_value = 7.5
        mock_virtual_memory.return_value.available = 1000 * 1024 * 1024
        mock_virtual_memory.return_value.total = 2000 * 1024 * 1024

        # Mock GC collecting 100 objects
        mock_gc_collect.return_value = 100

        # Mock memory after optimization (reduced)
        def memory_side_effect():
            if mock_gc_collect.called:
                mock_process.return_value.memory_info.return_value.rss = (
                    140 * 1024 * 1024
                )  # 140MB
            return mock_process.return_value.memory_info.return_value

        mock_process.return_value.memory_info.side_effect = memory_side_effect

        result = optimizer.optimize_memory()

        assert result["objects_collected"] == 100
        assert result["memory_before_mb"] == 150.0
        assert "memory_after_mb" in result
        assert "memory_freed_mb" in result
        assert "timestamp" in result
        mock_gc_collect.assert_called_once()


class TestBatchProcessor:
    """Test BatchProcessor class."""

    @pytest.fixture
    def processor(self):
        """Create a batch processor for testing."""
        return BatchProcessor(batch_size=3, max_concurrent=2)

    @pytest.mark.asyncio
    async def test_process_batch_async(self, processor):
        """Test asynchronous batch processing."""

        # Mock processor function
        async def mock_processor(item):
            await asyncio.sleep(0.01)  # Simulate work
            return f"processed_{item}"

        items = [1, 2, 3, 4, 5, 6, 7]
        results = await processor.process_batch_async(items, mock_processor)

        assert len(results) == 7
        assert results[0] == "processed_1"
        assert results[6] == "processed_7"

    @pytest.mark.asyncio
    async def test_process_batch_async_with_progress(self, processor):
        """Test batch processing with progress callback."""
        progress_calls = []

        async def progress_callback(progress, current_batch, total_batches):
            progress_calls.append((progress, current_batch, total_batches))

        async def mock_processor(item):
            return f"processed_{item}"

        items = [1, 2, 3, 4, 5, 6]
        await processor.process_batch_async(items, mock_processor, progress_callback)

        assert len(progress_calls) == 2  # 6 items / 3 batch_size = 2 batches
        assert progress_calls[0] == (50.0, 1, 2)
        assert progress_calls[1] == (100.0, 2, 2)

    @pytest.mark.asyncio
    async def test_process_batch_async_with_exceptions(self, processor):
        """Test batch processing with exceptions."""

        async def mock_processor(item):
            if item == 3:
                raise ValueError(f"Error processing {item}")
            return f"processed_{item}"

        items = [1, 2, 3, 4, 5]
        results = await processor.process_batch_async(items, mock_processor)

        assert len(results) == 5
        assert results[0] == "processed_1"
        assert results[1] == "processed_2"
        assert isinstance(results[2], ValueError)
        assert results[3] == "processed_4"
        assert results[4] == "processed_5"


class TestGlobalInstances:
    """Test global instance functions."""

    def test_get_performance_monitor(self):
        """Test getting global performance monitor."""
        monitor1 = get_performance_monitor()
        monitor2 = get_performance_monitor()

        assert monitor1 is monitor2  # Should be same instance
        assert isinstance(monitor1, PerformanceMonitor)

    def test_get_memory_optimizer(self):
        """Test getting global memory optimizer."""
        optimizer1 = get_memory_optimizer()
        optimizer2 = get_memory_optimizer()

        assert optimizer1 is optimizer2  # Should be same instance
        assert isinstance(optimizer1, MemoryOptimizer)

    def test_get_batch_processor(self):
        """Test getting global batch processor."""
        processor1 = get_batch_processor()
        processor2 = get_batch_processor()

        assert processor1 is processor2  # Should be same instance
        assert isinstance(processor1, BatchProcessor)


class TestDecorators:
    """Test performance monitoring decorators."""

    @patch("src.utils.performance.get_performance_monitor")
    def test_monitor_performance_sync(self, mock_get_monitor):
        """Test performance monitoring decorator for sync functions."""
        mock_monitor = Mock()
        mock_get_monitor.return_value = mock_monitor

        @monitor_performance("test_function")
        def test_func(x, y):
            return x + y

        result = test_func(1, 2)

        assert result == 3
        mock_monitor.measure_sync.assert_called_once_with("test_function")

    @patch("src.utils.performance.get_performance_monitor")
    @pytest.mark.asyncio
    async def test_monitor_performance_async(self, mock_get_monitor):
        """Test performance monitoring decorator for async functions."""
        mock_monitor = Mock()
        mock_get_monitor.return_value = mock_monitor

        @monitor_performance("test_async_function")
        async def test_async_func(x, y):
            return x + y

        result = await test_async_func(1, 2)

        assert result == 3
        mock_monitor.measure_async.assert_called_once_with("test_async_function")

    @patch("src.utils.performance.get_memory_optimizer")
    def test_auto_memory_optimize_sync(self, mock_get_optimizer):
        """Test auto memory optimization decorator for sync functions."""
        mock_optimizer = Mock()
        mock_optimizer.get_memory_info.return_value = {
            "rss_mb": 150.0
        }  # Above threshold
        mock_get_optimizer.return_value = mock_optimizer

        @auto_memory_optimize(threshold_mb=100)
        def test_func():
            return "result"

        result = test_func()

        assert result == "result"
        mock_optimizer.auto_optimize.assert_called_once()

    @patch("src.utils.performance.get_memory_optimizer")
    @pytest.mark.asyncio
    async def test_auto_memory_optimize_async(self, mock_get_optimizer):
        """Test auto memory optimization decorator for async functions."""
        mock_optimizer = Mock()
        mock_optimizer.get_memory_info.return_value = {
            "rss_mb": 150.0
        }  # Above threshold
        mock_get_optimizer.return_value = mock_optimizer

        @auto_memory_optimize(threshold_mb=100)
        async def test_async_func():
            return "async_result"

        result = await test_async_func()

        assert result == "async_result"
        mock_optimizer.auto_optimize.assert_called_once()
