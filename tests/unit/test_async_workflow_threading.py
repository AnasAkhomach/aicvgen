#!/usr/bin/env python3
"""
Test for the event loop fix in the Streamlit application.

This test verifies that the async workflow functions can be properly
executed in a thread-based approach to avoid "Event loop is closed" errors.
"""

import asyncio
import threading
import pytest
from unittest.mock import Mock, patch, MagicMock

# Import the function we're testing
from app import run_async_workflow_in_thread


class TestEventLoopFix:
    """Test cases for the event loop fix."""

    def test_run_async_workflow_in_thread_success(self):
        """Test that async workflow runs successfully in thread."""

        # Create a mock async function
        async def mock_async_workflow(arg1, arg2, kwarg1=None):
            """Mock async workflow function."""
            await asyncio.sleep(0.01)  # Simulate async work
            return f"Success: {arg1}, {arg2}, {kwarg1}"

        # Run the async function in thread
        thread = run_async_workflow_in_thread(
            mock_async_workflow, "test_arg1", "test_arg2", kwarg1="test_kwarg"
        )

        # Wait for thread to complete
        thread.join(timeout=5.0)

        # Verify thread completed successfully
        assert not thread.is_alive()

    def test_run_async_workflow_in_thread_with_exception(self):
        """Test that exceptions in async workflow are handled properly."""

        # Create a mock async function that raises an exception
        async def mock_failing_workflow():
            """Mock async workflow that fails."""
            await asyncio.sleep(0.01)
            raise ValueError("Test exception")

        # Mock the logger to capture error messages
        with patch("app.logger") as mock_logger:
            # Run the failing async function in thread
            thread = run_async_workflow_in_thread(mock_failing_workflow)

            # Wait for thread to complete
            thread.join(timeout=5.0)

            # Verify thread completed and error was logged
            assert not thread.is_alive()
            mock_logger.error.assert_called_once()
            error_call_args = mock_logger.error.call_args[0][0]
            assert "Error in workflow thread" in error_call_args
            assert "Test exception" in error_call_args

    def test_event_loop_isolation(self):
        """Test that each thread gets its own event loop."""
        loop_ids = []

        async def capture_loop_id():
            """Capture the current event loop ID."""
            loop = asyncio.get_event_loop()
            loop_ids.append(id(loop))
            await asyncio.sleep(0.01)

        # Run multiple workflows in parallel threads
        threads = []
        for _ in range(3):
            thread = run_async_workflow_in_thread(capture_loop_id)
            threads.append(thread)

        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=5.0)
            assert not thread.is_alive()

        # Verify each thread had a different event loop
        assert len(loop_ids) == 3
        assert len(set(loop_ids)) == 3  # All loop IDs should be unique

    @patch("app.threading.Thread")
    def test_thread_creation_parameters(self, mock_thread_class):
        """Test that threads are created with correct parameters."""
        mock_thread_instance = Mock()
        mock_thread_class.return_value = mock_thread_instance

        async def mock_workflow():
            """Mock workflow function."""
            pass

        # Run the function
        result = run_async_workflow_in_thread(mock_workflow)

        # Verify thread was created with daemon=True
        mock_thread_class.assert_called_once()
        call_kwargs = mock_thread_class.call_args[1]
        assert call_kwargs["daemon"] is True

        # Verify thread.start() was called
        mock_thread_instance.start.assert_called_once()

        # Verify the function returns the thread instance
        assert result == mock_thread_instance

    def test_workflow_manager_integration(self):
        """Test integration with workflow manager trigger_workflow_step."""
        # Create mock workflow manager
        mock_manager = Mock()
        mock_manager.trigger_workflow_step = Mock()

        # Make trigger_workflow_step an async function
        async def mock_trigger_workflow_step(session_id, state):
            """Mock trigger workflow step."""
            await asyncio.sleep(0.01)
            return f"Processed: {session_id}"

        mock_manager.trigger_workflow_step = mock_trigger_workflow_step

        # Test the integration
        thread = run_async_workflow_in_thread(
            mock_manager.trigger_workflow_step, "test_session_id", "test_state"
        )

        # Wait for completion
        thread.join(timeout=5.0)
        assert not thread.is_alive()

    def test_multiple_concurrent_workflows(self):
        """Test that multiple workflows can run concurrently."""
        results = []

        async def mock_workflow(workflow_id):
            """Mock workflow that simulates work."""
            await asyncio.sleep(0.1)  # Simulate some work
            results.append(f"Workflow {workflow_id} completed")

        # Start multiple workflows concurrently
        threads = []
        for i in range(5):
            thread = run_async_workflow_in_thread(mock_workflow, i)
            threads.append(thread)

        # Wait for all to complete
        for thread in threads:
            thread.join(timeout=10.0)
            assert not thread.is_alive()

        # Verify all workflows completed
        assert len(results) == 5
        for i in range(5):
            assert f"Workflow {i} completed" in results
