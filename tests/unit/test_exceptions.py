"""Unit tests for custom exception hierarchy.

Tests the custom exception classes defined in src.utils.exceptions to ensure
proper inheritance, error handling, and type-based classification.
"""

import pytest
from src.utils.exceptions import (
    AicvgenError,
    WorkflowPreconditionError,
    LLMResponseParsingError,
    AgentExecutionError,
    ConfigurationError,
    StateManagerError,
    ValidationError,
    RateLimitError,
    NetworkError,
    TimeoutError
)


class TestAicvgenError:
    """Test the base AicvgenError class."""
    
    def test_base_exception_inheritance(self):
        """Test that AicvgenError inherits from Exception."""
        error = AicvgenError("Test error")
        assert isinstance(error, Exception)
        assert str(error) == "Test error"
    
    def test_base_exception_can_be_raised(self):
        """Test that AicvgenError can be raised and caught."""
        with pytest.raises(AicvgenError) as exc_info:
            raise AicvgenError("Base error test")
        assert str(exc_info.value) == "Base error test"


class TestWorkflowPreconditionError:
    """Test the WorkflowPreconditionError class."""
    
    def test_inheritance(self):
        """Test proper inheritance from ValueError and AicvgenError."""
        error = WorkflowPreconditionError("Missing data")
        assert isinstance(error, ValueError)
        assert isinstance(error, AicvgenError)
        assert isinstance(error, Exception)
    
    def test_error_message(self):
        """Test error message handling."""
        message = "Job description data is required to initialize workflow."
        error = WorkflowPreconditionError(message)
        assert str(error) == message


class TestLLMResponseParsingError:
    """Test the LLMResponseParsingError class."""
    
    def test_inheritance(self):
        """Test proper inheritance from ValueError and AicvgenError."""
        error = LLMResponseParsingError("Parse failed")
        assert isinstance(error, ValueError)
        assert isinstance(error, AicvgenError)
        assert isinstance(error, Exception)
    
    def test_basic_message(self):
        """Test basic error message without raw response."""
        message = "Failed to parse LLM response"
        error = LLMResponseParsingError(message)
        assert str(error) == message
        assert error.raw_response == ""
    
    def test_message_with_raw_response(self):
        """Test error message with raw response snippet."""
        message = "Failed to parse JSON"
        raw_response = "This is a very long response that should be truncated" * 10
        error = LLMResponseParsingError(message, raw_response)
        
        # Check that raw_response is stored
        assert error.raw_response == raw_response
        
        # Check that error message includes snippet
        error_str = str(error)
        assert message in error_str
        assert "Raw response snippet:" in error_str
        assert len(error_str) < len(raw_response) + len(message) + 50  # Should be truncated
    
    def test_raw_response_truncation(self):
        """Test that raw response is truncated to 200 characters."""
        message = "Parse error"
        raw_response = "x" * 300  # 300 character response
        error = LLMResponseParsingError(message, raw_response)
        
        error_str = str(error)
        # Should contain truncated response (200 chars + "...")
        assert "x" * 200 + "..." in error_str
        assert "x" * 250 not in error_str  # Should not contain full response


class TestAgentExecutionError:
    """Test the AgentExecutionError class."""
    
    def test_inheritance(self):
        """Test proper inheritance from AicvgenError."""
        error = AgentExecutionError("TestAgent", "Execution failed")
        assert isinstance(error, AicvgenError)
        assert isinstance(error, Exception)
    
    def test_agent_name_storage(self):
        """Test that agent name is stored correctly."""
        agent_name = "ParserAgent"
        message = "Failed to parse content"
        error = AgentExecutionError(agent_name, message)
        
        assert error.agent_name == agent_name
        assert agent_name in str(error)
        assert message in str(error)
    
    def test_formatted_message(self):
        """Test that error message is properly formatted."""
        agent_name = "ContentWriter"
        message = "LLM timeout"
        error = AgentExecutionError(agent_name, message)
        
        expected = f"Agent '{agent_name}' failed: {message}"
        assert str(error) == expected


class TestOtherExceptions:
    """Test other exception classes for basic functionality."""
    
    def test_configuration_error(self):
        """Test ConfigurationError inheritance and basic functionality."""
        error = ConfigurationError("Invalid config")
        assert isinstance(error, AicvgenError)
        assert str(error) == "Invalid config"
    
    def test_state_manager_error(self):
        """Test StateManagerError inheritance and basic functionality."""
        error = StateManagerError("State corruption")
        assert isinstance(error, AicvgenError)
        assert str(error) == "State corruption"
    
    def test_validation_error(self):
        """Test ValidationError inheritance and basic functionality."""
        error = ValidationError("Invalid data")
        assert isinstance(error, ValueError)
        assert isinstance(error, AicvgenError)
        assert str(error) == "Invalid data"
    
    def test_rate_limit_error(self):
        """Test RateLimitError inheritance and basic functionality."""
        error = RateLimitError("Rate limit exceeded")
        assert isinstance(error, AicvgenError)
        assert str(error) == "Rate limit exceeded"
    
    def test_network_error(self):
        """Test NetworkError inheritance and basic functionality."""
        error = NetworkError("Connection failed")
        assert isinstance(error, AicvgenError)
        assert str(error) == "Connection failed"
    
    def test_timeout_error(self):
        """Test TimeoutError inheritance and basic functionality."""
        error = TimeoutError("Operation timed out")
        assert isinstance(error, AicvgenError)
        assert str(error) == "Operation timed out"


class TestExceptionHierarchy:
    """Test the overall exception hierarchy and type checking."""
    
    def test_all_exceptions_inherit_from_base(self):
        """Test that all custom exceptions inherit from AicvgenError."""
        exceptions = [
            WorkflowPreconditionError("test"),
            LLMResponseParsingError("test"),
            AgentExecutionError("agent", "test"),
            ConfigurationError("test"),
            StateManagerError("test"),
            ValidationError("test"),
            RateLimitError("test"),
            NetworkError("test"),
            TimeoutError("test")
        ]
        
        for exception in exceptions:
            assert isinstance(exception, AicvgenError)
            assert isinstance(exception, Exception)
    
    def test_value_error_inheritance(self):
        """Test that appropriate exceptions inherit from ValueError."""
        value_error_exceptions = [
            WorkflowPreconditionError("test"),
            LLMResponseParsingError("test"),
            ValidationError("test")
        ]
        
        for exception in value_error_exceptions:
            assert isinstance(exception, ValueError)
    
    def test_exception_catching(self):
        """Test that exceptions can be caught by their base types."""
        # Test catching by AicvgenError
        with pytest.raises(AicvgenError):
            raise WorkflowPreconditionError("test")
        
        # Test catching by ValueError
        with pytest.raises(ValueError):
            raise LLMResponseParsingError("test")
        
        # Test catching by specific type
        with pytest.raises(AgentExecutionError):
            raise AgentExecutionError("agent", "test")