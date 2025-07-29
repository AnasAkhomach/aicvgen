"""Integration tests for CB-014: Consistent Error Handling Contracts.

This test suite validates that all components across the application
follow consistent error handling contracts and patterns.
"""


from unittest.mock import Mock, patch

from src.agents.agent_base import AgentBase
from src.error_handling.agent_error_handler import AgentErrorHandler
from src.error_handling.boundaries import StreamlitErrorBoundary
from src.error_handling.exceptions import (
    AgentExecutionError,
    AicvgenError,
    ConfigurationError,
    NetworkError,
    OperationTimeoutError,
    RateLimitError,
    ValidationError,
    VectorStoreError,
)
from src.error_handling.models import ErrorCategory, ErrorContext, ErrorSeverity
from src.orchestration.state import AgentState


class TestErrorHandlingContracts:
    """Test suite for validating error handling contracts across the application."""

    def test_custom_exception_hierarchy(self):
        """Test that custom exceptions follow proper hierarchy and structure."""
        # Test base exception
        base_error = AicvgenError(
            "Test error",
            category=ErrorCategory.UNKNOWN,
            severity=ErrorSeverity.HIGH,
        )
        assert isinstance(base_error, Exception)
        assert base_error.message == "Test error"
        assert base_error.category == ErrorCategory.UNKNOWN
        assert base_error.severity == ErrorSeverity.HIGH
        assert base_error.context is not None

        # Test specific exceptions inherit from base
        agent_error = AgentExecutionError("test_agent", "Agent failed")
        assert isinstance(agent_error, AicvgenError)
        assert agent_error.category == ErrorCategory.AGENT_LIFECYCLE

        config_error = ConfigurationError("Config invalid")
        assert isinstance(config_error, AicvgenError)
        assert config_error.category == ErrorCategory.CONFIGURATION

        validation_error = ValidationError("Validation failed")
        assert isinstance(validation_error, AicvgenError)
        assert validation_error.category == ErrorCategory.VALIDATION

    def test_structured_error_conversion(self):
        """Test that exceptions can be converted to structured errors."""
        error = AgentExecutionError(
            "test_agent",
            "Test agent error",
        )

        structured = error.to_structured_error()
        assert "Test agent error" in structured.message
        assert structured.category == ErrorCategory.AGENT_LIFECYCLE
        assert structured.severity == ErrorSeverity.HIGH
        assert structured.context.component == "test_agent"

    def test_agent_error_handler_integration(self):
        """Test that AgentErrorHandler provides consistent error handling."""
        # Test validation error handling
        validation_error = ValidationError("Invalid input")
        result = AgentErrorHandler.handle_validation_error(
            validation_error, "test_agent"
        )

        assert not result.success
        assert "Invalid input" in result.error_message
        assert result.metadata["validation_error"] is True
        assert result.metadata["error_type"] == "ValidationError"

        # Test general error handling
        general_error = Exception("Unexpected error")
        result = AgentErrorHandler.handle_general_error(general_error, "test_agent")

        assert not result.success
        assert result.error_message == "Unexpected error"
        assert result.metadata["error_type"] == "Exception"

    def test_agent_error_handler_decorator(self):
        """Test that the error handling decorator works correctly."""
        from src.error_handling.agent_error_handler import with_error_handling

        @with_error_handling("test_agent")
        def test_method():
            # Simulate an error
            raise ValidationError("Test validation error")

        # The decorator should catch the error and return a structured response
        result = test_method()

        assert not result.success
        assert "Test validation error" in result.error_message
        assert result.metadata["validation_error"] is True

    def test_streamlit_error_boundary(self):
        """Test that Streamlit error boundary handles errors correctly."""
        from src.error_handling.boundaries import error_boundary

        # Test context manager
        with patch("streamlit.warning") as mock_st_warning:
            # The error_boundary context manager catches and handles the error
            with error_boundary(component_name="test_component"):
                raise ValidationError("Test error")

            # Should have called streamlit.warning (default severity is MEDIUM)
            mock_st_warning.assert_called_once()
            call_args = mock_st_warning.call_args[0][0]
            assert "test_component" in call_args

    def test_error_boundary_decorator(self):
        """Test that error boundary decorator works correctly."""
        from src.error_handling.boundaries import handle_api_errors

        @handle_api_errors
        def test_function():
            raise NetworkError("API connection failed")

        with patch("streamlit.error") as mock_st_error:
            # The decorator should catch the error and not re-raise it
            result = test_function()
            assert result is None  # Decorator returns None on error

            # Should have called streamlit.error
            mock_st_error.assert_called_once()
            call_args = mock_st_error.call_args[0][0]
            assert "connection" in call_args.lower()

    def test_error_context_propagation(self):
        """Test that error context is properly propagated."""
        context = ErrorContext(
            component="test_component",
            operation="test_operation",
            user_id="test_user",
            session_id="test_session",
            additional_data={"key": "value"},
        )

        error = AgentExecutionError(
            "test_agent", "Test error with context", context=context
        )

        # Test context is preserved (AgentExecutionError sets component to agent_name)
        assert error.context.component == "test_agent"
        assert error.context.operation == "test_operation"
        assert error.context.user_id == "test_user"
        assert error.context.session_id == "test_session"
        assert error.context.additional_data["key"] == "value"

        # Test context can be updated
        error.with_context(new_key="new_value")
        assert error.context.additional_data["new_key"] == "new_value"

    def test_error_severity_levels(self):
        """Test that error severity levels are properly categorized."""
        # Test different severity levels
        low_error = ValidationError("Minor validation issue")
        low_error.severity = ErrorSeverity.LOW
        medium_error = AgentExecutionError("test_agent", "Agent processing issue")
        medium_error.severity = ErrorSeverity.MEDIUM
        high_error = ConfigurationError("Critical config error")
        high_error.severity = ErrorSeverity.HIGH
        critical_error = VectorStoreError("Database corruption")
        critical_error.severity = ErrorSeverity.CRITICAL

        assert low_error.severity == ErrorSeverity.LOW
        assert medium_error.severity == ErrorSeverity.MEDIUM
        assert high_error.severity == ErrorSeverity.HIGH
        assert critical_error.severity == ErrorSeverity.CRITICAL

        # Test structured error messages vary by severity
        low_structured = low_error.to_structured_error()
        critical_structured = critical_error.to_structured_error()

        assert low_structured.severity == ErrorSeverity.LOW
        assert critical_structured.severity == ErrorSeverity.CRITICAL

    def test_catchable_exceptions_consistency(self):
        """Test that CATCHABLE_EXCEPTIONS tuple is properly defined."""
        from src.error_handling.exceptions import CATCHABLE_EXCEPTIONS

        # Should include common exceptions that should be handled gracefully
        assert ValueError in CATCHABLE_EXCEPTIONS
        assert TypeError in CATCHABLE_EXCEPTIONS
        assert KeyError in CATCHABLE_EXCEPTIONS
        assert AttributeError in CATCHABLE_EXCEPTIONS
        assert IOError in CATCHABLE_EXCEPTIONS

        # Should be a tuple for performance
        assert isinstance(CATCHABLE_EXCEPTIONS, tuple)

        # Test that it can be used in except clauses
        try:
            raise ValueError("Test error")
        except CATCHABLE_EXCEPTIONS as e:
            assert isinstance(e, ValueError)

    def test_error_handling_integration_flow(self):
        """Test complete error handling flow from exception to user display."""
        # Simulate a complete error flow
        original_error = Exception("Database connection failed")

        # 1. Wrap in custom exception with context
        context = ErrorContext(
            component="vector_store_service",
            operation="search",
            session_id="test_session",
        )

        wrapped_error = VectorStoreError(
            "Failed to search vector store",
            context=context,
        )
        wrapped_error.original_exception = original_error

        # 2. Convert to structured error
        structured = wrapped_error.to_structured_error()

        # 3. Verify all information is preserved
        assert structured.message == "Failed to search vector store"
        assert structured.category == ErrorCategory.DATABASE
        assert structured.severity == ErrorSeverity.HIGH
        assert structured.context.component == "vector_store_service"
        assert structured.context.operation == "search"
        assert structured.original_exception == original_error

        # 4. Test user-friendly message generation
        user_message = structured.user_message
        assert "database" in user_message.lower()
