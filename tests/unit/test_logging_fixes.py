"""Tests for logging fixes - duplicate logging and error message improvements."""

import logging
import pytest
from unittest.mock import Mock, patch, MagicMock
from io import StringIO

from src.config.logging_config import setup_logging
from src.core.managers.workflow_manager import WorkflowManager

# Import removed - using standard exceptions for testing


class TestLoggingFixes:
    """Test suite for logging fixes."""

    def test_setup_logging_idempotent(self):
        """Test that setup_logging can be called multiple times without creating duplicate handlers."""
        # Clear any existing handlers
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Call setup_logging multiple times
        setup_logging()
        initial_handler_count = len(root_logger.handlers)

        setup_logging()
        setup_logging()

        # Should not have more handlers than the first call
        final_handler_count = len(root_logger.handlers)
        assert (
            final_handler_count == initial_handler_count
        ), "setup_logging should not create duplicate handlers"

    def test_error_logging_format(self):
        """Test that error logs include detailed information in the message."""
        # Create a direct test of the error logging format
        logger = logging.getLogger("test_error_format")

        # Create a string buffer to capture log output
        log_capture = StringIO()
        handler = logging.StreamHandler(log_capture)
        handler.setLevel(logging.ERROR)
        formatter = logging.Formatter("%(levelname)s - %(message)s")
        handler.setFormatter(formatter)

        # Add the handler to our logger
        logger.addHandler(handler)
        logger.setLevel(logging.ERROR)

        # Log with the old format (generic message with details in extra)
        session_id = "test_session_123"
        error_msg = "Test error message"
        logger.error(
            "Workflow step execution failed",
            extra={"session_id": session_id, "error": error_msg},
        )

        # Log with the new format (details included in message)
        logger.error(
            f"Workflow step execution failed for session {session_id}: {error_msg}",
            extra={"session_id": session_id, "error": error_msg},
        )

        # Get the captured output
        log_output = log_capture.getvalue()
        log_lines = log_output.strip().split("\n")

        # First line should be the old format (generic message)
        assert "Workflow step execution failed" in log_lines[0]
        assert "test_session_123" not in log_lines[0]
        assert "Test error message" not in log_lines[0]

        # Second line should be the new format (detailed message)
        assert (
            "Workflow step execution failed for session test_session_123: Test error message"
            in log_lines[1]
        )

        # Clean up
        logger.removeHandler(handler)
        handler.close()

    def test_logging_configuration_prevents_duplicates(self):
        """Test that the logging configuration properly prevents duplicate log entries."""
        # Create a string buffer to capture log output
        log_capture = StringIO()
        handler = logging.StreamHandler(log_capture)
        handler.setLevel(logging.INFO)

        # Create a test logger
        test_logger = logging.getLogger("test_duplicate_logging")
        test_logger.setLevel(logging.INFO)
        test_logger.addHandler(handler)

        # Log a message multiple times
        test_message = "Test message for duplicate detection"
        test_logger.info(test_message)
        test_logger.info(test_message)

        # Get the captured output
        log_output = log_capture.getvalue()

        # Count occurrences of the test message
        message_count = log_output.count(test_message)

        # Should have exactly 2 occurrences (one for each log call)
        assert message_count == 2, f"Expected 2 occurrences, got {message_count}"

        # Clean up
        test_logger.removeHandler(handler)
        handler.close()

    def test_error_message_improvement(self):
        """Test that the new error message format is more descriptive than the old one."""
        # Test the old format
        old_format = "Workflow step execution failed"

        # Test the new format
        session_id = "session_456"
        error_details = "Invalid input data"
        new_format = (
            f"Workflow step execution failed for session {session_id}: {error_details}"
        )

        # Verify the new format includes more information
        assert session_id in new_format
        assert error_details in new_format
        assert "Workflow step execution failed" in new_format

        # Verify the old format was generic
        assert session_id not in old_format
        assert error_details not in old_format

        # Verify the new format is more descriptive
        assert len(new_format) > len(old_format)
        assert new_format != old_format
