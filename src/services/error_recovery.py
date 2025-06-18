"""Error recovery service for CV generation workflow.

This module provides comprehensive error handling, recovery strategies,
and resilience mechanisms for the individual item processing workflow.
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import traceback
import json

from ..config.logging_config import get_structured_logger
from ..models.data_models import ProcessingStatus, ContentType, Item
from ..orchestration.state import AgentState
from ..utils.exceptions import (
    AicvgenError,
    WorkflowPreconditionError,
    LLMResponseParsingError,
    AgentExecutionError,
    ConfigurationError,
    StateManagerError,
    ValidationError,
)


# Additional exception types for comprehensive error handling
class RateLimitError(AicvgenError):
    """Raised when rate limits are exceeded."""

    pass


class NetworkError(AicvgenError):
    """Raised when network-related errors occur."""

    pass


class TimeoutError(AicvgenError):
    """Raised when operations timeout."""

    pass


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
    failure_threshold: int = 5
    timeout_seconds: int = 60
    half_open_max_attempts: int = 3


class ErrorRecoveryService:
    """Service for handling errors and implementing recovery strategies."""

    def __init__(self):
        self.logger = get_structured_logger("error_recovery")

        # Error tracking
        self.error_history: Dict[str, List[ErrorContext]] = {}
        self.circuit_breakers: Dict[str, CircuitBreakerState] = {}

        # Recovery configuration
        self.error_strategies = self._initialize_error_strategies()
        self.max_error_history = 100

        # Fallback content templates
        self.fallback_templates = self._initialize_fallback_templates()

    def _initialize_error_strategies(self) -> Dict[ErrorType, RecoveryAction]:
        """Initialize default recovery strategies for different error types."""
        return {
            ErrorType.RATE_LIMIT: RecoveryAction(
                strategy=RecoveryStrategy.RATE_LIMIT_BACKOFF,
                delay_seconds=60.0,
                max_retries=5,
            ),
            ErrorType.API_ERROR: RecoveryAction(
                strategy=RecoveryStrategy.EXPONENTIAL_BACKOFF,
                delay_seconds=2.0,
                max_retries=3,
            ),
            ErrorType.NETWORK_ERROR: RecoveryAction(
                strategy=RecoveryStrategy.EXPONENTIAL_BACKOFF,
                delay_seconds=5.0,
                max_retries=4,
            ),
            ErrorType.TIMEOUT_ERROR: RecoveryAction(
                strategy=RecoveryStrategy.LINEAR_BACKOFF,
                delay_seconds=10.0,
                max_retries=2,
            ),
            ErrorType.VALIDATION_ERROR: RecoveryAction(
                strategy=RecoveryStrategy.MANUAL_INTERVENTION, max_retries=0
            ),
            ErrorType.PARSING_ERROR: RecoveryAction(
                strategy=RecoveryStrategy.FALLBACK_CONTENT, max_retries=1
            ),
            ErrorType.CONTENT_ERROR: RecoveryAction(
                strategy=RecoveryStrategy.FALLBACK_CONTENT, max_retries=1
            ),
            ErrorType.SYSTEM_ERROR: RecoveryAction(
                strategy=RecoveryStrategy.CIRCUIT_BREAKER,
                delay_seconds=30.0,
                max_retries=2,
            ),
            ErrorType.UNKNOWN_ERROR: RecoveryAction(
                strategy=RecoveryStrategy.EXPONENTIAL_BACKOFF,
                delay_seconds=5.0,
                max_retries=2,
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

        Uses type-based classification first (most robust), then falls back to
        string-based classification for generic errors.
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
        if isinstance(exception, RateLimitError):
            return ErrorType.RATE_LIMIT
        if isinstance(exception, NetworkError):
            return ErrorType.NETWORK_ERROR
        if isinstance(exception, TimeoutError):
            return ErrorType.TIMEOUT_ERROR

        # --- String-based classification (fallback for generic errors) ---
        error_message = str(exception).lower()

        # Rate limiting errors
        if any(
            keyword in error_message
            for keyword in ["rate limit", "too many requests", "quota exceeded", "429"]
        ):
            return ErrorType.RATE_LIMIT

        # API errors
        if any(
            keyword in error_message
            for keyword in [
                "api error",
                "invalid api key",
                "unauthorized",
                "401",
                "403",
            ]
        ):
            return ErrorType.API_ERROR

        # Network errors
        if any(
            keyword in error_message
            for keyword in ["connection", "network", "timeout", "dns", "unreachable"]
        ):
            return ErrorType.NETWORK_ERROR

        # Timeout errors
        if any(
            keyword in error_message
            for keyword in ["timeout", "timed out", "deadline exceeded"]
        ):
            return ErrorType.TIMEOUT_ERROR

        # Validation errors
        if any(
            keyword in error_message
            for keyword in [
                "validation",
                "invalid input",
                "bad request",
                "400",
                "data is missing",
                "cannot initialize workflow",
                "required to initialize",
            ]
        ):
            return ErrorType.VALIDATION_ERROR

        # Parsing errors
        if any(
            keyword in error_message
            for keyword in ["json", "parse", "decode", "format", "syntax"]
        ):
            return ErrorType.PARSING_ERROR

        # Content errors
        if any(
            keyword in error_message
            for keyword in [
                "content",
                "empty response",
                "no content",
                "invalid content",
            ]
        ):
            return ErrorType.CONTENT_ERROR

        # System errors
        if any(
            keyword in error_message
            for keyword in ["system", "internal server error", "500", "503"]
        ):
            return ErrorType.SYSTEM_ERROR

        return ErrorType.UNKNOWN_ERROR

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
            f"Error recorded: {error_context.error_type.value}",
            session_id=session_id,
            item_id=error_context.item_id,
            error_message=error_context.error_message,
            retry_count=error_context.retry_count,
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
        """Extract retry-after value from rate limit error."""
        # Try to extract from error message
        error_message = error_context.error_message.lower()

        # Look for common patterns
        import re

        patterns = [
            r"retry[\s-]?after[:\s]+(\d+)",
            r"wait[\s]+(\d+)[\s]*seconds?",
            r"try again in[\s]+(\d+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, error_message)
            if match:
                return float(match.group(1))

        # Check metadata
        if "retry_after" in error_context.metadata:
            return float(error_context.metadata["retry_after"])

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

    async def execute_recovery_action(
        self, action: RecoveryAction, item: Item, session_id: str
    ) -> bool:
        """Execute a recovery action."""

        if action.delay_seconds > 0:
            self.logger.info(
                f"Waiting {action.delay_seconds} seconds before recovery action",
                session_id=session_id,
                item_id=item.id,
                strategy=action.strategy.value,
            )
            await asyncio.sleep(action.delay_seconds)

        if action.strategy == RecoveryStrategy.FALLBACK_CONTENT:
            if action.fallback_content:
                item.content = action.fallback_content
                item.metadata.status = ProcessingStatus.COMPLETED
                item.metadata.completed_at = datetime.now()

                self.logger.info(
                    "Applied fallback content", session_id=session_id, item_id=item.id
                )
                return True

        elif action.strategy == RecoveryStrategy.SKIP_ITEM:
            item.metadata.status = ProcessingStatus.SKIPPED
            item.metadata.completed_at = datetime.now()

            self.logger.info(
                "Skipped item due to recovery action",
                session_id=session_id,
                item_id=item.id,
            )
            return True

        elif action.strategy == RecoveryStrategy.CIRCUIT_BREAKER:
            self.logger.warning(
                "Circuit breaker triggered, stopping processing", session_id=session_id
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
                            f"Circuit breaker closed for {error_type.value}",
                            session_id=session_id,
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

        self.logger.info("Cleaned up error tracking data", session_id=session_id)


# Global error recovery service instance
_global_error_recovery_service: Optional[ErrorRecoveryService] = None


def get_error_recovery_service() -> ErrorRecoveryService:
    """Get the global error recovery service instance."""
    global _global_error_recovery_service
    if _global_error_recovery_service is None:
        _global_error_recovery_service = ErrorRecoveryService()
    return _global_error_recovery_service


def reset_error_recovery_service():
    """Reset the global error recovery service (useful for testing)."""
    global _global_error_recovery_service
    _global_error_recovery_service = None
