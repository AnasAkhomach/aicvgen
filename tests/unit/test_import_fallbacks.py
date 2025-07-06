"""Unit tests for standardized import fallback utilities.

This test suite validates that the import fallback system provides
consistent behavior across all optional dependencies.
"""

import pytest
from unittest.mock import patch, MagicMock
import logging

from src.utils.import_fallbacks import (
    OptionalDependency,
    safe_import,
    safe_import_from,
    get_security_utils,
    get_weasyprint,
    get_google_exceptions,
    get_dotenv
)


class TestOptionalDependency:
    """Test the OptionalDependency context manager."""
    
    def test_successful_import(self):
        """Test that successful imports work correctly."""
        with OptionalDependency("test_dep", "test feature") as dep:
            # This should not raise an exception
            import os  # Use a module that definitely exists
            dep.available = True
            
        assert dep.available is True
    
    def test_failed_import_with_logging(self, caplog):
        """Test that failed imports are logged appropriately."""
        with caplog.at_level(logging.WARNING):
            with OptionalDependency("nonexistent_module", "test feature") as dep:
                raise ImportError("Module not found")
                
        assert dep.available is False
        assert "Optional dependency 'nonexistent_module' not available" in caplog.text
        assert "test feature" in caplog.text
    
    def test_failed_import_silent(self, caplog):
        """Test that silent mode suppresses logging."""
        with caplog.at_level(logging.WARNING):
            with OptionalDependency("nonexistent_module", "test feature", silent=True) as dep:
                raise ImportError("Module not found")
                
        assert dep.available is False
        assert len(caplog.records) == 0
    
    def test_import_module_method(self):
        """Test the import_module method."""
        with OptionalDependency("test_dep") as dep:
            result = dep.import_module("os")
            
        assert result is not None
        assert dep.available is True
        assert dep.module is not None


class TestSafeImport:
    """Test the safe_import function."""
    
    def test_successful_import(self):
        """Test successful module import."""
        module, available = safe_import("os", "Operating System", silent=True)
        
        assert module is not None
        assert available is True
    
    def test_failed_import_with_fallback(self, caplog):
        """Test failed import with fallback value."""
        fallback_value = "fallback"
        
        with caplog.at_level(logging.WARNING):
            module, available = safe_import(
                "nonexistent_module",
                "Test Module",
                "test feature",
                fallback_value
            )
        
        assert module == fallback_value
        assert available is False
        assert "Optional dependency 'Test Module' not available" in caplog.text
    
    def test_failed_import_silent(self, caplog):
        """Test that silent mode works."""
        with caplog.at_level(logging.WARNING):
            module, available = safe_import(
                "nonexistent_module",
                silent=True
            )
        
        assert module is None
        assert available is False
        assert len(caplog.records) == 0


class TestSafeImportFrom:
    """Test the safe_import_from function."""
    
    def test_successful_single_import(self):
        """Test successful import of single item."""
        result, available = safe_import_from("os", "path", silent=True)
        
        assert result is not None
        assert available is True
    
    def test_successful_multiple_import(self):
        """Test successful import of multiple items."""
        result, available = safe_import_from("os", ["path", "environ"], silent=True)
        
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert available is True
    
    def test_failed_import_with_fallback_factory(self, caplog):
        """Test failed import with fallback factory."""
        def fallback_factory():
            return "fallback_result"
        
        with caplog.at_level(logging.WARNING):
            result, available = safe_import_from(
                "nonexistent_module",
                "nonexistent_item",
                "Test Module",
                "test feature",
                fallback_factory
            )
        
        assert result == "fallback_result"
        assert available is False
        assert "Optional dependency 'Test Module' not available" in caplog.text


class TestSpecificImportFunctions:
    """Test the specific import functions for known dependencies."""
    
    def test_get_security_utils(self):
        """Test that get_security_utils returns callable functions."""
        redact_sensitive_data, redact_log_message = get_security_utils()
        
        assert callable(redact_sensitive_data)
        assert callable(redact_log_message)
        
        # Test that fallback functions work
        test_data = "sensitive data"
        test_message = "log message"
        
        assert redact_sensitive_data(test_data) == test_data
        assert redact_log_message(test_message) == test_message
    
    @patch('src.utils.import_fallbacks.safe_import')
    def test_get_weasyprint_available(self, mock_safe_import):
        """Test get_weasyprint when WeasyPrint is available."""
        mock_module = MagicMock()
        mock_safe_import.return_value = (mock_module, True)
        
        module, available = get_weasyprint()
        
        assert module == mock_module
        assert available is True
        mock_safe_import.assert_called_once_with(
            module_path="weasyprint",
            dependency_name="WeasyPrint",
            feature_description="PDF generation will be disabled",
            fallback_value=None
        )
    
    @patch('src.utils.import_fallbacks.safe_import')
    def test_get_weasyprint_unavailable(self, mock_safe_import):
        """Test get_weasyprint when WeasyPrint is not available."""
        mock_safe_import.return_value = (None, False)
        
        module, available = get_weasyprint()
        
        assert module is None
        assert available is False
    
    @patch('src.utils.import_fallbacks.safe_import')
    def test_get_google_exceptions(self, mock_safe_import):
        """Test get_google_exceptions function."""
        mock_module = MagicMock()
        mock_safe_import.return_value = (mock_module, True)
        
        module, available = get_google_exceptions()
        
        assert module == mock_module
        assert available is True
        mock_safe_import.assert_called_once_with(
            module_path="google.api_core.exceptions",
            dependency_name="Google API Core",
            feature_description="Google-specific error handling will use generic patterns",
            fallback_value=None
        )
    
    @patch('src.utils.import_fallbacks.safe_import_from')
    def test_get_dotenv_available(self, mock_safe_import_from):
        """Test get_dotenv when python-dotenv is available."""
        mock_load_dotenv = MagicMock()
        mock_safe_import_from.return_value = (mock_load_dotenv, True)
        
        load_dotenv, available = get_dotenv()
        
        assert load_dotenv == mock_load_dotenv
        assert available is True
    
    @patch('src.utils.import_fallbacks.safe_import_from')
    def test_get_dotenv_unavailable(self, mock_safe_import_from):
        """Test get_dotenv when python-dotenv is not available."""
        fallback_func = lambda *args, **kwargs: None
        mock_safe_import_from.return_value = (fallback_func, False)
        
        load_dotenv, available = get_dotenv()
        
        assert callable(load_dotenv)
        assert available is False
        
        # Test that fallback function works without error
        load_dotenv("test.env")


class TestIntegration:
    """Integration tests for the import fallback system."""
    
    def test_consistent_logging_levels(self, caplog):
        """Test that all functions use consistent logging levels."""
        with caplog.at_level(logging.WARNING):
            # Test multiple functions that should fail
            safe_import("nonexistent1", "Test1", "feature1")
            safe_import_from("nonexistent2", "item", "Test2", "feature2")
        
        # All should log at WARNING level
        for record in caplog.records:
            assert record.levelno == logging.WARNING
    
    def test_consistent_message_format(self, caplog):
        """Test that all functions use consistent message formats."""
        with caplog.at_level(logging.WARNING):
            safe_import("nonexistent1", "Test Module", "test feature")
        
        assert len(caplog.records) == 1
        message = caplog.records[0].message
        assert "Optional dependency 'Test Module' not available" in message
        assert "test feature" in message