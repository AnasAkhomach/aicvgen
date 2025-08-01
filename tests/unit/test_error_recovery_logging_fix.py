"""Tests for error recovery service logging fix.

This test verifies that the ErrorRecoveryService._record_error method
no longer passes session_id as a keyword argument to logger.error().
"""

import logging
import pytest
from datetime import datetime
from unittest.mock import Mock, patch

from src.services.error_recovery import ErrorRecoveryService, ErrorContext, ErrorType
from src.models.data_models import ContentType


class TestErrorRecoveryLoggingFix:
    """Test class for error recovery logging fixes."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_logger = Mock(spec=logging.Logger)
        self.error_recovery = ErrorRecoveryService(logger=self.mock_logger)

    def test_record_error_does_not_pass_session_id_as_kwarg(self):
        """Test that _record_error doesn't pass session_id as keyword argument."""
        # Create test error context
        error_context = ErrorContext(
            error_type=ErrorType.VALIDATION_ERROR,
            error_message="Test error message",
            item_id="test_item",
            item_type=ContentType.EXPERIENCE,
            session_id="test_session_123",
            timestamp=datetime.now(),
            retry_count=1,
        )

        # Call _record_error
        self.error_recovery._record_error(error_context)

        # Verify logger.error was called with only the message (no kwargs)
        self.mock_logger.error.assert_called_once()
        call_args = self.mock_logger.error.call_args

        # Should have only positional arguments (the message)
        assert len(call_args[0]) == 1  # Only the message
        assert len(call_args[1]) == 0  # No keyword arguments

        # Verify the message contains all the information
        message = call_args[0][0]
        assert "validation_error" in message
        assert "test_session_123" in message
        assert "test_item" in message
        assert "Test error message" in message
        assert "1" in message

    @pytest.mark.asyncio
    async def test_handle_error_integration(self):
        """Test that handle_error works without logging errors."""
        # Mock the _record_error method to ensure it's called
        with patch.object(self.error_recovery, "_record_error") as mock_record:
            result = await self.error_recovery.handle_error(
                Exception("Test error"),
                "test_session",
                "test_item",
                ErrorType.PARSING_ERROR,
            )

            # Verify _record_error was called
            mock_record.assert_called_once()

            # Verify the result is a RecoveryAction
            assert hasattr(result, "strategy")
            assert hasattr(result, "should_continue")
