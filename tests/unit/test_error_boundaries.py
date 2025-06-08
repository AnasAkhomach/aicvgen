#!/usr/bin/env python3
"""
Unit Tests for Error Boundaries

Tests the error boundary functionality for Streamlit components.
"""

import pytest
import unittest
from unittest.mock import patch, MagicMock
import streamlit as st

from src.utils.error_boundaries import (
    StreamlitErrorBoundary,
    error_boundary,
    safe_streamlit_component,
    handle_api_errors,
    handle_file_operations,
    handle_data_processing,
    ErrorSeverity,
    ErrorRecovery,
    create_error_report
)


class TestStreamlitErrorBoundary(unittest.TestCase):
    """Test the StreamlitErrorBoundary class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.boundary = StreamlitErrorBoundary(
            component_name="test_component",
            show_error_details=True,
            severity=ErrorSeverity.MEDIUM
        )
    
    @patch('streamlit.warning')
    @patch('streamlit.info')
    def test_handle_error_medium_severity(self, mock_info, mock_warning):
        """Test error handling for medium severity errors."""
        test_error = ValueError("Test error message")
        
        self.boundary._handle_error(test_error, "test_function", (), {})
        
        # Check that warning was called
        mock_warning.assert_called_once()
        mock_info.assert_called_once()
        
        # Check warning message contains component name
        warning_call = mock_warning.call_args[0][0]
        self.assertIn("test_component", warning_call)
    
    @patch('streamlit.error')
    def test_handle_error_critical_severity(self, mock_error):
        """Test error handling for critical severity errors."""
        boundary = StreamlitErrorBoundary(
            component_name="critical_component",
            severity=ErrorSeverity.CRITICAL
        )
        
        test_error = RuntimeError("Critical error")
        boundary._handle_error(test_error, "test_function", (), {})
        
        # Check that error was called twice (main message + error ID)
        self.assertEqual(mock_error.call_count, 2)
    
    def test_generate_error_id(self):
        """Test error ID generation."""
        error_id = self.boundary._generate_error_id()
        
        self.assertIsInstance(error_id, str)
        self.assertTrue(error_id.startswith("ERR_test_component_"))
        self.assertTrue(len(error_id) > 20)  # Should include timestamp
    
    @patch('streamlit.warning')
    def test_decorator_functionality(self, mock_warning):
        """Test that the decorator properly catches and handles errors."""
        @self.boundary
        def failing_function():
            raise ValueError("Test error")
        
        result = failing_function()
        
        # Function should return None when error occurs
        self.assertIsNone(result)
        
        # Warning should be displayed
        mock_warning.assert_called_once()


class TestErrorBoundaryDecorators(unittest.TestCase):
    """Test error boundary decorators."""
    
    @patch('streamlit.error')
    def test_handle_api_errors_connection_error(self, mock_error):
        """Test API error handler for connection errors."""
        @handle_api_errors
        def api_function():
            raise ConnectionError("Connection failed")
        
        result = api_function()
        
        self.assertIsNone(result)
        mock_error.assert_called_once()
        error_message = mock_error.call_args[0][0]
        self.assertIn("Connection Error", error_message)
    
    @patch('streamlit.error')
    def test_handle_api_errors_timeout(self, mock_error):
        """Test API error handler for timeout errors."""
        @handle_api_errors
        def api_function():
            raise TimeoutError("Request timed out")
        
        result = api_function()
        
        self.assertIsNone(result)
        mock_error.assert_called_once()
        error_message = mock_error.call_args[0][0]
        self.assertIn("Timeout Error", error_message)
    
    @patch('streamlit.error')
    def test_handle_file_operations_file_not_found(self, mock_error):
        """Test file operations error handler for file not found."""
        @handle_file_operations
        def file_function():
            raise FileNotFoundError("File not found")
        
        result = file_function()
        
        self.assertIsNone(result)
        mock_error.assert_called_once()
        error_message = mock_error.call_args[0][0]
        self.assertIn("File Not Found", error_message)
    
    @patch('streamlit.error')
    def test_handle_data_processing_key_error(self, mock_error):
        """Test data processing error handler for key errors."""
        @handle_data_processing
        def data_function():
            raise KeyError("Missing key")
        
        result = data_function()
        
        self.assertIsNone(result)
        mock_error.assert_called_once()
        error_message = mock_error.call_args[0][0]
        self.assertIn("Data Error", error_message)


class TestErrorBoundaryContextManager(unittest.TestCase):
    """Test error boundary context manager."""
    
    @patch('streamlit.warning')
    def test_context_manager_catches_errors(self, mock_warning):
        """Test that context manager catches and handles errors."""
        with error_boundary("test_context"):
            raise ValueError("Test error in context")
        
        mock_warning.assert_called_once()
    
    def test_context_manager_allows_success(self):
        """Test that context manager allows successful execution."""
        result = None
        
        with error_boundary("test_context"):
            result = "success"
        
        self.assertEqual(result, "success")


class TestErrorRecovery(unittest.TestCase):
    """Test error recovery strategies."""
    
    def test_retry_with_backoff_success_on_retry(self):
        """Test retry mechanism succeeds on second attempt."""
        call_count = 0
        
        def flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("Temporary failure")
            return "success"
        
        result = ErrorRecovery.retry_with_backoff(flaky_function, max_retries=3)
        
        self.assertEqual(result, "success")
        self.assertEqual(call_count, 2)
    
    def test_retry_with_backoff_max_retries_exceeded(self):
        """Test retry mechanism fails after max retries."""
        def always_failing_function():
            raise ValueError("Always fails")
        
        with self.assertRaises(ValueError):
            ErrorRecovery.retry_with_backoff(always_failing_function, max_retries=2)
    
    def test_fallback_chain_first_succeeds(self):
        """Test fallback chain when first function succeeds."""
        def first_function():
            return "first_success"
        
        def second_function():
            return "second_success"
        
        result = ErrorRecovery.fallback_chain(first_function, second_function)
        
        self.assertEqual(result, "first_success")
    
    def test_fallback_chain_second_succeeds(self):
        """Test fallback chain when first fails but second succeeds."""
        def first_function():
            raise ValueError("First fails")
        
        def second_function():
            return "second_success"
        
        result = ErrorRecovery.fallback_chain(first_function, second_function)
        
        self.assertEqual(result, "second_success")
    
    def test_fallback_chain_all_fail(self):
        """Test fallback chain when all functions fail."""
        def first_function():
            raise ValueError("First fails")
        
        def second_function():
            raise RuntimeError("Second fails")
        
        with self.assertRaises(RuntimeError):  # Should raise the last error
            ErrorRecovery.fallback_chain(first_function, second_function)


class TestErrorReporting(unittest.TestCase):
    """Test error reporting functionality."""
    
    def test_create_error_report(self):
        """Test error report creation."""
        test_error = ValueError("Test error")
        context = {"component": "test", "user_id": "123"}
        
        report = create_error_report(test_error, context)
        
        # Check required fields
        self.assertIn("timestamp", report)
        self.assertIn("error_type", report)
        self.assertIn("error_message", report)
        self.assertIn("stack_trace", report)
        self.assertIn("context", report)
        
        # Check values
        self.assertEqual(report["error_type"], "ValueError")
        self.assertEqual(report["error_message"], "Test error")
        self.assertEqual(report["context"], context)
        self.assertIsInstance(report["timestamp"], str)
        self.assertIsInstance(report["stack_trace"], str)


class TestSafeStreamlitComponent(unittest.TestCase):
    """Test safe Streamlit component decorator."""
    
    @patch('streamlit.info')
    def test_safe_component_low_severity(self, mock_info):
        """Test safe component with low severity error."""
        @safe_streamlit_component("test_component", severity=ErrorSeverity.LOW)
        def test_function():
            raise ValueError("Low severity error")
        
        result = test_function()
        
        self.assertIsNone(result)
        mock_info.assert_called_once()
    
    def test_safe_component_success(self):
        """Test safe component with successful execution."""
        @safe_streamlit_component("test_component")
        def test_function():
            return "success"
        
        result = test_function()
        
        self.assertEqual(result, "success")


if __name__ == '__main__':
    unittest.main()