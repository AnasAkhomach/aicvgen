"""Error recovery service for CV generation workflow.

This module provides comprehensive error handling, recovery strategies,
and resilience mechanisms for the individual item processing workflow.
"""

import asyncio
import logging
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from src.constants.error_constants import ErrorConstants
from src.error_handling.exceptions import (
    AgentExecutionError,
    ConfigurationError,
    LLMResponseParsingError,
    NetworkError,
    OperationTimeoutError,
    RateLimitError,
    StateManagerError,
    ValidationError,
    WorkflowPreconditionError,
)
from src.models.data_models import ContentType, Item, ItemStatus
from src.utils.retry_predicates import is_transient_error

# Additional exception types for comprehensive error handling


class ErrorType(Enum):
    """Types of errors that can occur during processing."""

    RATE_LIMIT = "rate_limit"
    API_ERROR = "api_error"
    NETWORK_ERROR = "network_error"
    VALIDATION_ERROR = "validation_error"
    TIMEOUT_ERROR = "timeout_error"
    PARSING_ERROR = "parsing_error"
    CONTENT_ERROR = "content_error"
    SYSTEM_ERROR = "system_error"
    UNKNOWN_ERROR = "unknown_error"


class RecoveryStrategy(Enum):
    """Recovery strategies for different error types."""

    IMMEDIATE_RETRY = "immediate_retry"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    RATE_LIMIT_BACKOFF = "rate_limit_backoff"
    FALLBACK_CONTENT = "fallback_content"
    SKIP_ITEM = "skip_item"
    MANUAL_INTERVENTION = "manual_intervention"
    CIRCUIT_BREAKER = "circuit_breaker"


@dataclass
class ErrorContext:
    """Context information for an error."""

    error_type: ErrorType
    error_message: str
    item_id: str
    item_type: ContentType
    session_id: str
    timestamp: datetime
    stack_trace: Optional[str] = None
    retry_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    original_exception: Optional[Exception] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "error_type": self.error_type.value,
            "error_message": self.error_message,
            "item_id": self.item_id,
            "item_type": self.item_type.value,
            "session_id": self.session_id,
            "timestamp": self.timestamp.isoformat(),
            "stack_trace": self.stack_trace,
            "retry_count": self.retry_count,
            "metadata": self.metadata,
        }


@dataclass
class RecoveryAction:
    """Recovery action to be taken for an error."""

    strategy: RecoveryStrategy
    delay_seconds: float = 0.0
    max_retries: int = 3
    fallback_content: Optional[str] = None
    should_continue: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CircuitBreakerState:
    """State of a circuit breaker."""

    failure_count: int = 0
    last_failure_time: Optional[datetime] = None
    is_open: bool = False
    half_open_attempts: int = 0

    # Configuration
    failure_threshold: int = ErrorConstants.CIRCUIT_BREAKER_FAILURE_THRESHOLD
    timeout_seconds: int = ErrorConstants.CIRCUIT_BREAKER_TIMEOUT_SECONDS
    half_open_max_attempts: int = ErrorConstants.CIRCUIT_BREAKER_HALF_OPEN_MAX_ATTEMPTS


class ErrorRecoveryService:
    """Service for handling errors and implementing recovery strategies."""

    def __init__(self, logger: logging.Logger):
        """Initialize ErrorRecoveryService with injected dependencies.

        Args:
            logger: Logger instance for error recovery operations
        """
        self.logger = logger

        # Error tracking
        self.error_history: Dict[str, List[ErrorContext]] = {}
        self.circuit_breakers: Dict[str, CircuitBreakerState] = {}

        # Recovery configuration
        self.error_strategies = self._initialize_error_strategies()
        self.max_error_history = ErrorConstants.MAX_ERROR_HISTORY_SIZE

        # Fallback content templates
        self.fallback_templates = self._initialize_fallback_templates()

    def _initialize_error_strategies(self) -> Dict[ErrorType, RecoveryAction]:
        """Initialize default recovery strategies for different error types."""
        return {
            ErrorType.RATE_LIMIT: RecoveryAction(
                strategy=RecoveryStrategy.RATE_LIMIT_BACKOFF,
                delay_seconds=ErrorConstants.RATE_LIMIT_RECOVERY_DELAY,
                max_retries=ErrorConstants.RATE_LIMIT_MAX_RETRIES,
            ),
            ErrorType.API_ERROR: RecoveryAction(
                strategy=RecoveryStrategy.EXPONENTIAL_BACKOFF,
                delay_seconds=ErrorConstants.API_ERROR_RECOVERY_DELAY,
                max_retries=ErrorConstants.API_ERROR_MAX_RETRIES,
            ),
            ErrorType.NETWORK_ERROR: RecoveryAction(
                strategy=RecoveryStrategy.EXPONENTIAL_BACKOFF,
                delay_seconds=ErrorConstants.NETWORK_ERROR_RECOVERY_DELAY,
                max_retries=ErrorConstants.NETWORK_ERROR_MAX_RETRIES,
            ),
            ErrorType.TIMEOUT_ERROR: RecoveryAction(
                strategy=RecoveryStrategy.LINEAR_BACKOFF,
                delay_seconds=ErrorConstants.TIMEOUT_ERROR_RECOVERY_DELAY,
                max_retries=ErrorConstants.TIMEOUT_ERROR_MAX_RETRIES,
            ),
            ErrorType.VALIDATION_ERROR: RecoveryAction(
                strategy=RecoveryStrategy.MANUAL_INTERVENTION,
                max_retries=ErrorConstants.VALIDATION_ERROR_MAX_RETRIES,
            ),
            ErrorType.PARSING_ERROR: RecoveryAction(
                strategy=RecoveryStrategy.FALLBACK_CONTENT,
                max_retries=ErrorConstants.PARSING_ERROR_MAX_RETRIES,
            ),
            ErrorType.CONTENT_ERROR: RecoveryAction(
                strategy=RecoveryStrategy.FALLBACK_CONTENT,
                max_retries=ErrorConstants.CONTENT_ERROR_MAX_RETRIES,
            ),
            ErrorType.SYSTEM_ERROR: RecoveryAction(
                strategy=RecoveryStrategy.CIRCUIT_BREAKER,
                delay_seconds=ErrorConstants.SYSTEM_ERROR_RECOVERY_DELAY,
                max_retries=ErrorConstants.SYSTEM_ERROR_MAX_RETRIES,
            ),
            ErrorType.UNKNOWN_ERROR: RecoveryAction(
                strategy=RecoveryStrategy.EXPONENTIAL_BACKOFF,
                delay_seconds=ErrorConstants.UNKNOWN_ERROR_RECOVERY_DELAY,
                max_retries=ErrorConstants.UNKNOWN_ERROR_MAX_RETRIES,
            ),
        }

    def _initialize_fallback_templates(self) -> Dict[ContentType, str]:
        """Initialize fallback content templates."""
        return {
            ContentType.QUALIFICATION: "• Relevant qualification in {field}",
            ContentType.EXPERIENCE: "• Experience in {field} with demonstrated results",
            ContentType.PROJECT: "• Project involving {field} with successful outcomes",
            ContentType.EXECUTIVE_SUMMARY: "Experienced professional with expertise in {field} and proven track record of success.",
        }

    def classify_error(
        self, exception: Exception, context: Dict[str, Any] = None
    ) -> ErrorType:
        """Classify an exception into an error type.

        Uses centralized error classification utilities for consistent categorization.
        """
        # --- Type-based classification (most robust) ---
        if isinstance(exception, (WorkflowPreconditionError, ValidationError)):
            return ErrorType.VALIDATION_ERROR
        if isinstance(exception, LLMResponseParsingError):
            return ErrorType.PARSING_ERROR
        if isinstance(exception, AgentExecutionError):
            return ErrorType.SYSTEM_ERROR
        if isinstance(exception, ConfigurationError):
            return ErrorType.SYSTEM_ERROR
        if isinstance(exception, StateManagerError):
            return ErrorType.SYSTEM_ERROR

        # --- Use direct exception type checks ---
        if isinstance(exception, RateLimitError):
            return ErrorType.RATE_LIMIT
        if isinstance(exception, NetworkError):
            return ErrorType.NETWORK_ERROR
        if isinstance(exception, OperationTimeoutError):
            return ErrorType.TIMEOUT_ERROR

        # Default to system error for unknown exceptions
        return ErrorType.SYSTEM_ERROR

    async def handle_error(
        self,
        exception: Exception,
        item_id: str,
        item_type: ContentType,
        session_id: str,
        retry_count: int = 0,
        context: Dict[str, Any] = None,
    ) -> RecoveryAction:
        """Handle an error and determine the recovery action."""

        # Classify the error
        error_type = self.classify_error(exception, context)

        # Create error context
        error_context = ErrorContext(
            error_type=error_type,
            error_message=str(exception),
            item_id=item_id,
            item_type=item_type,
            session_id=session_id,
            timestamp=datetime.now(),
            stack_trace=traceback.format_exc(),
            retry_count=retry_count,
            metadata=context or {},
            original_exception=exception,
        )

        # Record the error
        self._record_error(error_context)

        # Check circuit breaker
        if self._should_circuit_break(error_type, session_id):
            return RecoveryAction(
                strategy=RecoveryStrategy.CIRCUIT_BREAKER,
                should_continue=False,
                metadata={"reason": "Circuit breaker open"},
            )

        # Get base recovery strategy
        base_action = self.error_strategies.get(
            error_type, self.error_strategies[ErrorType.UNKNOWN_ERROR]
        )

        # Check if we've exceeded max retries
        if retry_count >= base_action.max_retries:
            return self._get_final_fallback_action(error_context)

        # Calculate delay based on strategy
        delay = self._calculate_delay(base_action, retry_count, error_context)

        # Create recovery action
        recovery_action = RecoveryAction(
            strategy=base_action.strategy,
            delay_seconds=delay,
            max_retries=base_action.max_retries,
            should_continue=True,
            metadata={
                "error_type": error_type.value,
                "retry_count": retry_count,
                "original_delay": base_action.delay_seconds,
            },
        )

        # Add fallback content if needed
        if base_action.strategy == RecoveryStrategy.FALLBACK_CONTENT:
            recovery_action.fallback_content = self._generate_fallback_content(
                item_type, error_context
            )

        self.logger.info(
            f"Error recovery action determined: {recovery_action.strategy.value}",
            session_id=session_id,
            item_id=item_id,
            error_type=error_type.value,
            retry_count=retry_count,
            delay_seconds=delay,
        )

        return recovery_action

    def _record_error(self, error_context: ErrorContext):
        """Record an error in the history."""
        session_id = error_context.session_id

        if session_id not in self.error_history:
            self.error_history[session_id] = []

        self.error_history[session_id].append(error_context)

        # Limit history size
        if len(self.error_history[session_id]) > self.max_error_history:
            self.error_history[session_id] = self.error_history[session_id][
                -self.max_error_history :
            ]

        # Update circuit breaker
        self._update_circuit_breaker(error_context)

        # Log the error
        self.logger.error(
            f"Error recorded: {error_context.error_type.value} - Session: {session_id}, Item: {error_context.item_id}, Message: {error_context.error_message}, Retry: {error_context.retry_count}"
        )

    def _should_circuit_break(self, error_type: ErrorType, session_id: str) -> bool:
        """Check if circuit breaker should be triggered."""
        if error_type not in [ErrorType.SYSTEM_ERROR, ErrorType.API_ERROR]:
            return False

        breaker_key = f"{session_id}_{error_type.value}"
        breaker = self.circuit_breakers.get(breaker_key)

        if not breaker:
            return False

        # Check if circuit is open
        if breaker.is_open:
            # Check if timeout has passed
            if breaker.last_failure_time:
                time_since_failure = datetime.now() - breaker.last_failure_time
                if time_since_failure.total_seconds() > breaker.timeout_seconds:
                    # Move to half-open state
                    breaker.is_open = False
                    breaker.half_open_attempts = 0
                    return False
            return True

        return False

    def _update_circuit_breaker(self, error_context: ErrorContext):
        """Update circuit breaker state based on error."""
        if error_context.error_type not in [
            ErrorType.SYSTEM_ERROR,
            ErrorType.API_ERROR,
        ]:
            return

        breaker_key = f"{error_context.session_id}_{error_context.error_type.value}"

        if breaker_key not in self.circuit_breakers:
            self.circuit_breakers[breaker_key] = CircuitBreakerState()

        breaker = self.circuit_breakers[breaker_key]
        breaker.failure_count += 1
        breaker.last_failure_time = error_context.timestamp

        # Check if we should open the circuit
        if breaker.failure_count >= breaker.failure_threshold:
            breaker.is_open = True

            self.logger.warning(
                f"Circuit breaker opened for {error_context.error_type.value}",
                session_id=error_context.session_id,
                failure_count=breaker.failure_count,
            )

    def _calculate_delay(
        self, action: RecoveryAction, retry_count: int, error_context: ErrorContext
    ) -> float:
        """Calculate delay based on recovery strategy."""
        base_delay = action.delay_seconds

        if action.strategy == RecoveryStrategy.IMMEDIATE_RETRY:
            return 0.0

        elif action.strategy == RecoveryStrategy.LINEAR_BACKOFF:
            return base_delay * (retry_count + 1)

        elif action.strategy == RecoveryStrategy.EXPONENTIAL_BACKOFF:
            return base_delay * (2**retry_count)

        elif action.strategy == RecoveryStrategy.RATE_LIMIT_BACKOFF:
            # Extract retry-after from error if available
            retry_after = self._extract_retry_after(error_context)
            if retry_after:
                return retry_after
            return base_delay * (retry_count + 1)

        return base_delay

    def _extract_retry_after(self, error_context: ErrorContext) -> Optional[float]:
        """Extract retry-after value from rate limit error using explicit exception attributes."""
        # First check metadata for explicit retry_after value
        if "retry_after" in error_context.metadata:
            return float(error_context.metadata["retry_after"])

        # Check if the original exception has retry_after information
        if error_context.original_exception is not None:
            original_exc = error_context.original_exception
            if hasattr(original_exc, "retry_after"):
                return float(original_exc.retry_after)
            if hasattr(original_exc, "headers") and original_exc.headers:
                # Check both 'Retry-After' and 'retry-after' headers
                for header_key in ["Retry-After", "retry-after"]:
                    if header_key in original_exc.headers:
                        return float(original_exc.headers[header_key])

        # Fallback for rate limit errors: provide default backoff
        if error_context.error_type == ErrorType.RATE_LIMIT:
            return 60.0

        return None

    def _get_final_fallback_action(self, error_context: ErrorContext) -> RecoveryAction:
        """Get final fallback action when max retries exceeded."""
        # Try fallback content first
        if error_context.item_type in self.fallback_templates:
            return RecoveryAction(
                strategy=RecoveryStrategy.FALLBACK_CONTENT,
                fallback_content=self._generate_fallback_content(
                    error_context.item_type, error_context
                ),
                should_continue=True,
                metadata={"reason": "Max retries exceeded, using fallback"},
            )

        # Otherwise skip the item
        return RecoveryAction(
            strategy=RecoveryStrategy.SKIP_ITEM,
            should_continue=True,
            metadata={"reason": "Max retries exceeded, skipping item"},
        )

    def _generate_fallback_content(
        self, item_type: ContentType, error_context: ErrorContext
    ) -> str:
        """Generate fallback content for an item."""
        template = self.fallback_templates.get(item_type, "• Relevant content")

        # Try to extract field from metadata or use generic
        field = error_context.metadata.get("field", "relevant field")

        try:
            return template.format(field=field)
        except KeyError:
            return template

    async def get_fallback_content(
        self,
        content_type: ContentType,
        error_message: str = "",
        item_id: str = "",
        **kwargs,
    ) -> str:
        """
        Public method to get fallback content for a given content type.

        Args:
            content_type: The type of content to generate fallback for
            error_message: The error message that triggered fallback
            item_id: Optional item ID for context
            **kwargs: Additional context data

        Returns:
            Fallback content string
        """
        # Create a minimal error context for template processing
        error_context = ErrorContext(
            error_type=ErrorType.CONTENT_ERROR,
            error_message=error_message,
            item_id=item_id,
            item_type=content_type,
            session_id=kwargs.get("session_id", ""),
            timestamp=datetime.now(),
            stack_trace="",
            retry_count=0,
            metadata=kwargs,
        )

        return self._generate_fallback_content(content_type, error_context)

    async def execute_recovery_action(
        self, action: RecoveryAction, item: Item, session_id: str
    ) -> bool:
        """Execute a recovery action."""

        if action.delay_seconds > 0:
            self.logger.info(
                f"Waiting {action.delay_seconds} seconds before recovery action - Session: {session_id}, Item: {item.id}, Strategy: {action.strategy.value}"
            )
            await asyncio.sleep(action.delay_seconds)

        if action.strategy == RecoveryStrategy.FALLBACK_CONTENT:
            if action.fallback_content:
                item.content = action.fallback_content
                item.metadata.status = ItemStatus.COMPLETED
                item.metadata.completed_at = datetime.now()

                self.logger.info(
                    f"Applied fallback content - Session: {session_id}, Item: {item.id}"
                )
                return True

        elif action.strategy == RecoveryStrategy.SKIP_ITEM:
            item.metadata.status = ItemStatus.SKIPPED
            item.metadata.completed_at = datetime.now()

            self.logger.info(
                f"Skipped item due to recovery action - Session: {session_id}, Item: {item.id}"
            )
            return True

        elif action.strategy == RecoveryStrategy.CIRCUIT_BREAKER:
            self.logger.warning(
                f"Circuit breaker triggered, stopping processing - Session: {session_id}"
            )
            return False

        # For retry strategies, return True to continue processing
        return action.should_continue

    def record_success(self, item_id: str, item_type: ContentType, session_id: str):
        """Record a successful processing to update circuit breakers."""
        # Reset circuit breakers on success
        for error_type in [ErrorType.SYSTEM_ERROR, ErrorType.API_ERROR]:
            breaker_key = f"{session_id}_{error_type.value}"
            if breaker_key in self.circuit_breakers:
                breaker = self.circuit_breakers[breaker_key]
                if not breaker.is_open:
                    breaker.failure_count = 0
                elif breaker.half_open_attempts < breaker.half_open_max_attempts:
                    breaker.half_open_attempts += 1
                    if breaker.half_open_attempts >= breaker.half_open_max_attempts:
                        # Close the circuit
                        breaker.is_open = False
                        breaker.failure_count = 0
                        breaker.half_open_attempts = 0

                        self.logger.info(
                            f"Circuit breaker closed for {error_type.value} - Session: {session_id}"
                        )

    def get_error_summary(self, session_id: str) -> Dict[str, Any]:
        """Get error summary for a session."""
        errors = self.error_history.get(session_id, [])

        if not errors:
            return {"total_errors": 0, "error_types": {}, "recent_errors": []}

        # Count errors by type
        error_counts = {}
        for error in errors:
            error_type = error.error_type.value
            error_counts[error_type] = error_counts.get(error_type, 0) + 1

        # Get recent errors
        recent_errors = [error.to_dict() for error in errors[-10:]]

        # Get circuit breaker states
        circuit_states = {}
        for key, breaker in self.circuit_breakers.items():
            if key.startswith(session_id):
                error_type = key.split("_", 1)[1]
                circuit_states[error_type] = {
                    "is_open": breaker.is_open,
                    "failure_count": breaker.failure_count,
                    "last_failure": (
                        breaker.last_failure_time.isoformat()
                        if breaker.last_failure_time
                        else None
                    ),
                }

        return {
            "total_errors": len(errors),
            "error_types": error_counts,
            "recent_errors": recent_errors,
            "circuit_breakers": circuit_states,
        }

    def cleanup_session(self, session_id: str):
        """Clean up error tracking data for a session."""
        if session_id in self.error_history:
            del self.error_history[session_id]

        # Clean up circuit breakers
        keys_to_remove = [
            key for key in self.circuit_breakers.keys() if key.startswith(session_id)
        ]
        for key in keys_to_remove:
            del self.circuit_breakers[key]

        self.logger.info(f"Cleaned up error tracking data - Session: {session_id}")
