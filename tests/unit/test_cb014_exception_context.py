"""Tests for CB-014: Enhanced exception hierarchy with proper context preservation."""

import pytest
from datetime import datetime
from unittest.mock import patch

from src.error_handling.exceptions import (
    AicvgenError,
    AgentExecutionError,
    ConfigurationError,
    LLMResponseParsingError,
    NetworkError,
    RateLimitError,
    ServiceInitializationError,
    TemplateFormattingError,
    ValidationError,
    VectorStoreError,
    WorkflowError,
    WorkflowPreconditionError,
)
from src.error_handling.models import ErrorCategory, ErrorContext, ErrorSeverity


class TestAicvgenErrorBase:
    """Test the base AicvgenError class functionality."""

    def test_basic_initialization(self):
        """Test basic error initialization with default values."""
        error = AicvgenError("Test error message")

        assert error.message == "Test error message"
        assert error.category == ErrorCategory.UNKNOWN
        assert error.severity == ErrorSeverity.MEDIUM
        assert error.context is not None
        assert isinstance(error.context, ErrorContext)
        assert error.original_exception is None

    def test_initialization_with_context(self):
        """Test error initialization with custom context."""
        context = ErrorContext(
            component="test_component",
            operation="test_operation",
            session_id="test_session",
        )

        error = AicvgenError(
            "Test error",
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.HIGH,
            context=context,
        )

        assert error.category == ErrorCategory.VALIDATION
        assert error.severity == ErrorSeverity.HIGH
        assert error.context.component == "test_component"
        assert error.context.operation == "test_operation"
        assert error.context.session_id == "test_session"

    def test_to_structured_error(self):
        """Test conversion to structured error format."""
        error = AicvgenError(
            "Test error",
            category=ErrorCategory.NETWORK,
            severity=ErrorSeverity.CRITICAL,
        )

        structured = error.to_structured_error()

        assert structured.message == "Test error"
        assert structured.category == ErrorCategory.NETWORK
        assert structured.severity == ErrorSeverity.CRITICAL
        assert structured.context == error.context

    def test_with_context_updates(self):
        """Test context updates using with_context method."""
        error = AicvgenError("Test error")

        updated_error = error.with_context(
            component="updated_component", custom_field="custom_value"
        )

        assert updated_error.context.component == "updated_component"
        assert updated_error.context.additional_data["custom_field"] == "custom_value"

    def test_stack_trace_preservation(self):
        """Test that stack traces are properly preserved."""
        try:
            raise ValueError("Original error")
        except ValueError as e:
            error = AicvgenError("Wrapped error", original_exception=e)

            assert error.original_exception == e
            assert error.stack_trace is not None


class TestSpecificExceptions:
    """Test specific exception classes for proper context handling."""

    def test_agent_execution_error(self):
        """Test AgentExecutionError context preservation."""
        error = AgentExecutionError("test_agent", "Agent failed to process")

        assert "Agent 'test_agent' failed: Agent failed to process" in str(error)
        assert error.category == ErrorCategory.AGENT_LIFECYCLE
        assert error.severity == ErrorSeverity.HIGH
        assert error.context.component == "test_agent"
        assert error.context.additional_data["agent_name"] == "test_agent"

    def test_llm_response_parsing_error(self):
        """Test LLMResponseParsingError with raw response context."""
        raw_response = "Invalid JSON response" * 20  # Long response
        error = LLMResponseParsingError(
            "Failed to parse response", raw_response=raw_response
        )

        assert error.category == ErrorCategory.PARSING
        assert error.severity == ErrorSeverity.HIGH
        assert error.context.additional_data["raw_response"] == raw_response
        assert error.context.additional_data["response_length"] == len(raw_response)
        assert "Raw response snippet:" in str(error)

    def test_configuration_error(self):
        """Test ConfigurationError with config key context."""
        error = ConfigurationError("Missing API key", config_key="GEMINI_API_KEY")

        assert error.category == ErrorCategory.CONFIGURATION
        assert error.severity == ErrorSeverity.HIGH
        assert error.context.additional_data["config_key"] == "GEMINI_API_KEY"

    def test_service_initialization_error(self):
        """Test ServiceInitializationError with service context."""
        error = ServiceInitializationError("llm_service", "Failed to connect to API")

        assert "Service 'llm_service' initialization failed" in str(error)
        assert error.category == ErrorCategory.CONFIGURATION
        assert error.severity == ErrorSeverity.CRITICAL
        assert error.context.component == "llm_service"
        assert error.context.additional_data["service_name"] == "llm_service"

    def test_validation_error(self):
        """Test ValidationError with field context."""
        error = ValidationError("Invalid email format", field_name="email")

        assert error.category == ErrorCategory.VALIDATION
        assert error.severity == ErrorSeverity.MEDIUM
        assert error.context.additional_data["field_name"] == "email"

    def test_template_formatting_error(self):
        """Test TemplateFormattingError with missing keys context."""
        missing_keys = ["name", "email", "phone"]
        error = TemplateFormattingError(
            "Template formatting failed", missing_keys=missing_keys
        )

        assert error.category == ErrorCategory.GENERATION
        assert error.severity == ErrorSeverity.HIGH
        assert error.context.additional_data["missing_keys"] == missing_keys

    def test_rate_limit_error(self):
        """Test RateLimitError with retry context."""
        error = RateLimitError("API rate limit exceeded", retry_after=60)

        assert error.category == ErrorCategory.RATE_LIMIT
        assert error.severity == ErrorSeverity.MEDIUM
        assert error.context.additional_data["retry_after"] == 60

    def test_network_error(self):
        """Test NetworkError with status code context."""
        error = NetworkError("Connection failed", status_code=503)

        assert error.category == ErrorCategory.NETWORK
        assert error.severity == ErrorSeverity.HIGH
        assert error.context.additional_data["status_code"] == 503

    def test_vector_store_error(self):
        """Test VectorStoreError with operation context."""
        error = VectorStoreError("Failed to insert document", operation="insert")

        assert error.category == ErrorCategory.DATABASE
        assert error.severity == ErrorSeverity.HIGH
        assert error.context.operation == "insert"

    def test_workflow_error(self):
        """Test WorkflowError with workflow step context."""
        error = WorkflowError("Step execution failed", workflow_step="cv_analysis")

        assert error.category == ErrorCategory.AGENT_LIFECYCLE
        assert error.severity == ErrorSeverity.HIGH
        assert error.context.additional_data["workflow_step"] == "cv_analysis"

    def test_workflow_precondition_error(self):
        """Test WorkflowPreconditionError with missing data context."""
        error = WorkflowPreconditionError(
            "Missing required input", missing_data="cv_text"
        )

        assert error.category == ErrorCategory.VALIDATION
        assert error.severity == ErrorSeverity.MEDIUM
        assert error.context.additional_data["missing_data"] == "cv_text"


class TestErrorContextIntegration:
    """Test integration between exceptions and error context models."""

    def test_error_context_timestamp(self):
        """Test that error context includes timestamp."""
        before_time = datetime.now()
        error = AicvgenError("Test error")
        after_time = datetime.now()

        assert before_time <= error.context.timestamp <= after_time

    def test_error_context_id_generation(self):
        """Test that each error gets a unique ID."""
        error1 = AicvgenError("Error 1")
        error2 = AicvgenError("Error 2")

        assert error1.context.error_id != error2.context.error_id
        assert len(error1.context.error_id) > 0
        assert len(error2.context.error_id) > 0

    def test_additional_data_preservation(self):
        """Test that additional data is properly preserved."""
        error = AicvgenError(
            "Test error",
            custom_field="custom_value",
            numeric_field=42,
            list_field=[1, 2, 3],
        )

        assert error.additional_data["custom_field"] == "custom_value"
        assert error.additional_data["numeric_field"] == 42
        assert error.additional_data["list_field"] == [1, 2, 3]

    def test_inheritance_chain_preservation(self):
        """Test that inheritance chain is properly maintained."""
        error = ValidationError("Test validation error")

        assert isinstance(error, ValidationError)
        assert isinstance(error, ValueError)
        assert isinstance(error, AicvgenError)
        assert isinstance(error, Exception)

    def test_structured_error_conversion_completeness(self):
        """Test that structured error conversion preserves all data."""
        original_exception = ValueError("Original error")

        # Create error in an actual exception context to get proper stack trace
        try:
            raise original_exception
        except ValueError:
            error = AicvgenError(
                "Wrapper error",
                category=ErrorCategory.PARSING,
                severity=ErrorSeverity.CRITICAL,
                original_exception=original_exception,
            )

        structured = error.to_structured_error()

        assert structured.message == error.message
        assert structured.category == error.category
        assert structured.severity == error.severity
        assert structured.context == error.context
        assert structured.original_exception == original_exception
        assert structured.stack_trace == error.stack_trace
        # Verify stack trace is captured when in exception context
        assert structured.stack_trace is not None
        assert "Traceback" in structured.stack_trace
