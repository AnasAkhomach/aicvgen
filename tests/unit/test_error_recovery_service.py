"""Unit tests for ErrorRecoveryService.

Tests error classification, recovery strategy determination,
circuit breaker functionality, and fallback content generation.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta
from typing import Dict, Any

from src.services.error_recovery import (
    ErrorRecoveryService,
    ErrorType,
    RecoveryStrategy,
    ErrorContext,
    RecoveryAction,
    CircuitBreakerState
)
from src.models.data_models import ContentType, Item, ItemMetadata, ProcessingStatus
from src.utils.exceptions import (
    ValidationError,
    LLMResponseParsingError,
    WorkflowPreconditionError,
    RateLimitError,
    NetworkError,
    TimeoutError
)


class TestErrorRecoveryService:
    """Test cases for ErrorRecoveryService."""

    @pytest.fixture
    def error_recovery_service(self):
        """Create an ErrorRecoveryService instance for testing."""
        return ErrorRecoveryService()

    @pytest.fixture
    def sample_error_context(self):
        """Create a sample error context for testing."""
        return ErrorContext(
            session_id="test_session_123",
            item_id="test_item_456",
            item_type=ContentType.QUALIFICATION,
            error_type=ErrorType.API_ERROR,
            error_message="Test API error",
            retry_count=1,
            timestamp=datetime.now(),
            metadata={"field": "software engineering"}
        )

    @pytest.fixture
    def sample_item(self):
        """Create a sample item for testing."""
        return Item(
            id="test_item_456",
            type="qualification",
            content="Original content",
            metadata=ItemMetadata(
                status=ProcessingStatus.IN_PROGRESS,
                created_at=datetime.now()
            )
        )

    def test_error_classification_by_type(self, error_recovery_service):
        """Test error classification based on exception type."""
        # Test validation error
        validation_error = ValidationError("Invalid input")
        assert error_recovery_service.classify_error(validation_error) == ErrorType.VALIDATION_ERROR

        # Test parsing error
        parsing_error = LLMResponseParsingError("Failed to parse response")
        assert error_recovery_service.classify_error(parsing_error) == ErrorType.PARSING_ERROR

        # Test rate limit error
        rate_limit_error = RateLimitError("Rate limit exceeded")
        assert error_recovery_service.classify_error(rate_limit_error) == ErrorType.RATE_LIMIT

        # Test network error
        network_error = NetworkError("Connection failed")
        assert error_recovery_service.classify_error(network_error) == ErrorType.NETWORK_ERROR

        # Test timeout error
        timeout_error = TimeoutError("Request timed out")
        assert error_recovery_service.classify_error(timeout_error) == ErrorType.TIMEOUT_ERROR

    def test_error_classification_by_message(self, error_recovery_service):
        """Test error classification based on error message content."""
        # Test rate limit detection in message
        generic_error = Exception("Rate limit exceeded. Please try again later.")
        assert error_recovery_service.classify_error(generic_error) == ErrorType.RATE_LIMIT

        # Test API error detection
        api_error = Exception("API key invalid or expired")
        assert error_recovery_service.classify_error(api_error) == ErrorType.API_ERROR

        # Test timeout detection
        timeout_error = Exception("Request timeout after 30 seconds")
        assert error_recovery_service.classify_error(timeout_error) == ErrorType.TIMEOUT_ERROR

        # Test unknown error
        unknown_error = Exception("Some random error")
        assert error_recovery_service.classify_error(unknown_error) == ErrorType.UNKNOWN_ERROR

    def test_recovery_strategy_determination(self, error_recovery_service, sample_error_context):
        """Test recovery strategy determination for different error types."""
        # Test rate limit strategy
        sample_error_context.error_type = ErrorType.RATE_LIMIT
        action = error_recovery_service.determine_recovery_action(
            sample_error_context, retry_count=0
        )
        assert action.strategy == RecoveryStrategy.RATE_LIMIT_BACKOFF
        assert action.delay_seconds == 60.0
        assert action.max_retries == 5

        # Test API error strategy
        sample_error_context.error_type = ErrorType.API_ERROR
        action = error_recovery_service.determine_recovery_action(
            sample_error_context, retry_count=0
        )
        assert action.strategy == RecoveryStrategy.EXPONENTIAL_BACKOFF
        assert action.delay_seconds == 2.0
        assert action.max_retries == 3

        # Test validation error strategy
        sample_error_context.error_type = ErrorType.VALIDATION_ERROR
        action = error_recovery_service.determine_recovery_action(
            sample_error_context, retry_count=0
        )
        assert action.strategy == RecoveryStrategy.MANUAL_INTERVENTION
        assert action.max_retries == 0

    def test_max_retries_exceeded_fallback(self, error_recovery_service, sample_error_context):
        """Test fallback action when max retries are exceeded."""
        # Test with content type that has fallback template
        sample_error_context.error_type = ErrorType.API_ERROR
        action = error_recovery_service.determine_recovery_action(
            sample_error_context, retry_count=5  # Exceeds max retries (3)
        )
        assert action.strategy == RecoveryStrategy.FALLBACK_CONTENT
        assert action.fallback_content is not None
        assert "software engineering" in action.fallback_content

    def test_circuit_breaker_functionality(self, error_recovery_service, sample_error_context):
        """Test circuit breaker state management."""
        # Set up for system error that triggers circuit breaker
        sample_error_context.error_type = ErrorType.SYSTEM_ERROR
        
        # Record multiple failures to trigger circuit breaker
        for i in range(6):  # Exceed failure threshold (5)
            error_recovery_service._record_error(sample_error_context)
        
        # Check if circuit breaker should be triggered
        should_break = error_recovery_service._should_circuit_break(
            ErrorType.SYSTEM_ERROR, 
            sample_error_context.session_id
        )
        assert should_break is True

        # Test recovery action with circuit breaker
        action = error_recovery_service.determine_recovery_action(
            sample_error_context, retry_count=0
        )
        assert action.strategy == RecoveryStrategy.CIRCUIT_BREAKER
        assert action.should_continue is False

    def test_fallback_content_generation(self, error_recovery_service, sample_error_context):
        """Test fallback content generation for different content types."""
        # Test qualification fallback
        sample_error_context.item_type = ContentType.QUALIFICATION
        content = error_recovery_service._generate_fallback_content(
            ContentType.QUALIFICATION, sample_error_context
        )
        assert "software engineering" in content
        assert "qualification" in content.lower()

        # Test experience fallback
        sample_error_context.item_type = ContentType.EXPERIENCE
        content = error_recovery_service._generate_fallback_content(
            ContentType.EXPERIENCE, sample_error_context
        )
        assert "software engineering" in content
        assert "experience" in content.lower()

        # Test project fallback
        sample_error_context.item_type = ContentType.PROJECT
        content = error_recovery_service._generate_fallback_content(
            ContentType.PROJECT, sample_error_context
        )
        assert "software engineering" in content
        assert "project" in content.lower()

    @pytest.mark.asyncio
    async def test_execute_recovery_action_fallback_content(self, error_recovery_service, sample_item):
        """Test execution of fallback content recovery action."""
        action = RecoveryAction(
            strategy=RecoveryStrategy.FALLBACK_CONTENT,
            fallback_content="• Relevant qualification in software engineering",
            should_continue=True
        )
        
        result = await error_recovery_service.execute_recovery_action(
            action, sample_item, "test_session_123"
        )
        
        assert result is True
        assert sample_item.content == "• Relevant qualification in software engineering"
        assert sample_item.metadata.status == ProcessingStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_execute_recovery_action_skip_item(self, error_recovery_service, sample_item):
        """Test execution of skip item recovery action."""
        action = RecoveryAction(
            strategy=RecoveryStrategy.SKIP_ITEM,
            should_continue=True
        )
        
        result = await error_recovery_service.execute_recovery_action(
            action, sample_item, "test_session_123"
        )
        
        assert result is True
        assert sample_item.metadata.status == ProcessingStatus.SKIPPED

    @pytest.mark.asyncio
    async def test_execute_recovery_action_circuit_breaker(self, error_recovery_service, sample_item):
        """Test execution of circuit breaker recovery action."""
        action = RecoveryAction(
            strategy=RecoveryStrategy.CIRCUIT_BREAKER,
            should_continue=False
        )
        
        result = await error_recovery_service.execute_recovery_action(
            action, sample_item, "test_session_123"
        )
        
        assert result is False

    def test_success_recording_resets_circuit_breaker(self, error_recovery_service):
        """Test that recording success resets circuit breaker state."""
        session_id = "test_session_123"
        
        # Create a circuit breaker in open state
        breaker_key = f"{session_id}_{ErrorType.SYSTEM_ERROR.value}"
        error_recovery_service.circuit_breakers[breaker_key] = CircuitBreakerState(
            failure_count=5,
            is_open=False,  # Start in closed state for half-open testing
            half_open_attempts=0
        )
        
        # Record success
        error_recovery_service.record_success(
            "test_item", ContentType.QUALIFICATION, session_id
        )
        
        # Check that failure count is reset
        breaker = error_recovery_service.circuit_breakers[breaker_key]
        assert breaker.failure_count == 0

    def test_error_summary_generation(self, error_recovery_service, sample_error_context):
        """Test error summary generation for a session."""
        # Record some errors
        for i in range(3):
            sample_error_context.retry_count = i
            error_recovery_service._record_error(sample_error_context)
        
        # Generate summary
        summary = error_recovery_service.get_error_summary(sample_error_context.session_id)
        
        assert summary["total_errors"] == 3
        assert ErrorType.API_ERROR.value in summary["error_types"]
        assert summary["error_types"][ErrorType.API_ERROR.value] == 3
        assert len(summary["recent_errors"]) == 3

    def test_delay_calculation_exponential_backoff(self, error_recovery_service, sample_error_context):
        """Test delay calculation for exponential backoff strategy."""
        base_action = RecoveryAction(
            strategy=RecoveryStrategy.EXPONENTIAL_BACKOFF,
            delay_seconds=2.0,
            max_retries=3
        )
        
        # Test different retry counts
        delay_0 = error_recovery_service._calculate_delay(base_action, 0, sample_error_context)
        delay_1 = error_recovery_service._calculate_delay(base_action, 1, sample_error_context)
        delay_2 = error_recovery_service._calculate_delay(base_action, 2, sample_error_context)
        
        # Exponential backoff should increase delays
        assert delay_0 == 2.0
        assert delay_1 == 4.0  # 2 * 2^1
        assert delay_2 == 8.0  # 2 * 2^2

    def test_delay_calculation_linear_backoff(self, error_recovery_service, sample_error_context):
        """Test delay calculation for linear backoff strategy."""
        base_action = RecoveryAction(
            strategy=RecoveryStrategy.LINEAR_BACKOFF,
            delay_seconds=5.0,
            max_retries=3
        )
        
        # Test different retry counts
        delay_0 = error_recovery_service._calculate_delay(base_action, 0, sample_error_context)
        delay_1 = error_recovery_service._calculate_delay(base_action, 1, sample_error_context)
        delay_2 = error_recovery_service._calculate_delay(base_action, 2, sample_error_context)
        
        # Linear backoff should increase delays linearly
        assert delay_0 == 5.0
        assert delay_1 == 10.0  # 5 * (1 + 1)
        assert delay_2 == 15.0  # 5 * (2 + 1)

    def test_retry_after_extraction_from_message(self, error_recovery_service, sample_error_context):
        """Test extraction of retry-after value from error message."""
        # Test retry-after pattern
        sample_error_context.error_message = "Rate limit exceeded. Retry after 120 seconds."
        retry_after = error_recovery_service._extract_retry_after_from_message(sample_error_context)
        assert retry_after == 120.0
        
        # Test wait pattern
        sample_error_context.error_message = "Please wait 45 seconds before retrying."
        retry_after = error_recovery_service._extract_retry_after_from_message(sample_error_context)
        assert retry_after == 45.0
        
        # Test no pattern match
        sample_error_context.error_message = "Generic error message"
        retry_after = error_recovery_service._extract_retry_after_from_message(sample_error_context)
        assert retry_after is None

    def test_retry_after_extraction_from_metadata(self, error_recovery_service, sample_error_context):
        """Test extraction of retry-after value from error metadata."""
        sample_error_context.metadata["retry_after"] = 90
        retry_after = error_recovery_service._extract_retry_after_from_message(sample_error_context)
        assert retry_after == 90.0