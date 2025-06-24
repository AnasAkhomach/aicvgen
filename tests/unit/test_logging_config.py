"""Unit tests for the environment-aware logging configuration."""

import unittest
import logging
import os
from unittest.mock import patch, MagicMock

# Temporarily set env var for testing
# In a real test suite, this might be handled by a fixture
os.environ["APP_ENV"] = "development"
from src.config.logging_config import setup_logging


class TestLoggingConfig(unittest.TestCase):
    """Test suite for logging_config.py."""

    def setUp(self):
        """Reset logging handlers before each test."""
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        root_logger.handlers.clear()
        root_logger.setLevel(logging.WARNING)

    def test_setup_logging_routes_by_env(self):
        """Verify that setup_logging configures the correct handlers based on APP_ENV."""
        from src.config.logging_config import setup_logging

        # Test for development
        with patch.dict(os.environ, {"APP_ENV": "development"}):
            self.setUp()
            setup_logging()
            root_logger = logging.getLogger()
            # Should have at least one FileHandler in development
            found_file = any(
                "FileHandler" in str(type(h)) for h in root_logger.handlers
            )
            self.assertTrue(found_file)
        # Test for production
        with patch.dict(os.environ, {"APP_ENV": "production"}):
            self.setUp()
            setup_logging()
            root_logger = logging.getLogger()
            found_json = any(
                hasattr(h, "formatter")
                and h.formatter
                and "JsonFormatter" in str(type(h.formatter))
                for h in root_logger.handlers
            )
            self.assertTrue(found_json)

    @patch("logging.basicConfig")
    def test_setup_logging_dev_mode(self, mock_basic_config):
        """Verify the development logging configuration."""
        with patch.dict(os.environ, {"APP_ENV": "development"}):
            setup_logging(log_level=logging.DEBUG)
            # Check that basicConfig was called with the correct arguments
            _, kwargs = mock_basic_config.call_args
            self.assertEqual(kwargs["level"], logging.DEBUG)
            self.assertIn("StreamHandler", str(kwargs["handlers"]))
            self.assertIn("FileHandler", str(kwargs["handlers"]))

    def test_setup_logging_prod_mode_real_handler(self):
        """Verify the production (JSON) logging configuration with a real handler."""
        from src.config.logging_config import setup_logging

        with patch.dict(os.environ, {"APP_ENV": "production"}):
            self.setUp()  # Clear handlers again
            setup_logging(log_level=logging.INFO)
            root_logger = logging.getLogger()
            # Accept 1 or more handlers, just check at least one is a StreamHandler with JsonFormatter
            found_json = False
            for handler in root_logger.handlers:
                if (
                    hasattr(handler, "formatter")
                    and handler.formatter
                    and "JsonFormatter" in str(type(handler.formatter))
                ):
                    found_json = True
            self.assertTrue(found_json)


if __name__ == "__main__":
    unittest.main()
