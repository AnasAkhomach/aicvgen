"""Unit tests for the environment-aware logging configuration."""

import unittest
import logging
import os
from unittest.mock import patch, MagicMock
import tempfile

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
            handler.close()
            root_logger.removeHandler(handler)
        root_logger.handlers.clear()
        root_logger.setLevel(logging.WARNING)

    def tearDown(self):
        """Close and remove any handlers created during a test."""
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            handler.close()
            root_logger.removeHandler(handler)

    def test_setup_logging_routes_by_env(self):
        """Verify that setup_logging configures the correct handlers based on APP_ENV."""
        from src.config.logging_config import setup_logging

        # Test for development
        with patch.dict(os.environ, {"APP_ENV": "development"}):
            with tempfile.TemporaryDirectory() as tmpdir:
                with patch("src.config.logging_config.get_config") as mock_get_config:
                    mock_config = MagicMock()
                    mock_config.logging.log_directory = tmpdir
                    mock_config.logging.main_log_file = "app.log"
                    mock_config.logging.error_log_file = "error.log"
                    mock_get_config.return_value = mock_config
                    setup_logging()
                    root_logger = logging.getLogger()
                    # Should have at least one FileHandler in development
                    found_file = any(
                        "FileHandler" in str(type(h)) for h in root_logger.handlers
                    )
                    self.assertTrue(found_file)
        # Test for production
        with patch.dict(os.environ, {"APP_ENV": "production"}):
            setup_logging()
            root_logger = logging.getLogger()
            found_json = any(
                hasattr(h, "formatter")
                and h.formatter
                and "JsonFormatter" in str(type(h.formatter))
                for h in root_logger.handlers
            )
            self.assertTrue(found_json)

    @patch("src.config.logging_config.get_config")
    def test_setup_logging_dev_mode(self, mock_get_config):
        """Verify the development logging configuration."""
        # Arrange
        mock_config = MagicMock()
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config.logging.log_directory = tmpdir
            mock_config.logging.main_log_file = "app.log"
            mock_config.logging.error_log_file = "error.log"
            mock_get_config.return_value = mock_config

            with patch.dict(os.environ, {"APP_ENV": "development"}):
                # Act
                setup_logging(log_level=logging.DEBUG)

                # Assert
                root_logger = logging.getLogger()
                self.assertEqual(logging.DEBUG, root_logger.level)

                handler_types = [type(h).__name__ for h in root_logger.handlers]
                self.assertIn("StreamHandler", handler_types)
                self.assertEqual(handler_types.count("FileHandler"), 2)
                self.assertEqual(len(root_logger.handlers), 3)

                # Verify error handler is configured correctly
                error_handler = next(
                    (
                        h
                        for h in root_logger.handlers
                        if isinstance(h, logging.FileHandler)
                        and h.level == logging.ERROR
                    ),
                    None,
                )
                self.assertIsNotNone(
                    error_handler,
                    "Error file handler not found or level not set to ERROR",
                )

    def test_setup_logging_prod_mode_real_handler(self):
        """Verify the production (JSON) logging configuration with a real handler."""
        from src.config.logging_config import setup_logging

        with patch.dict(os.environ, {"APP_ENV": "production"}):
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
