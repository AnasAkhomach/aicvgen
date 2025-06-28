"""Unit tests for logging configuration functionality."""

import logging
import tempfile
from pathlib import Path
from unittest.mock import patch


class TestLoggingConfig:
    """Unit tests for logging configuration functions."""

    def test_setup_production_logging_creates_file_handlers(self):
        """Test that production logging setup creates both console and file handlers."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock the config
            with patch("src.config.logging_config.get_config") as mock_config:

                class MockLoggingConfig:
                    log_directory = temp_dir
                    main_log_file = "app.log"
                    error_log_file = "error.log"

                class MockConfig:
                    logging = MockLoggingConfig()

                mock_config.return_value = MockConfig()

                # Import and test the function
                from src.config.logging_config import _setup_production_logging

                # Clear any existing handlers
                root_logger = logging.getLogger()
                root_logger.handlers.clear()

                try:
                    # Setup production logging
                    _setup_production_logging(logging.INFO)

                    # Verify handlers were created
                    handlers = root_logger.handlers
                    assert (
                        len(handlers) == 3
                    ), "Should have console, main file, and error file handlers"

                    # Verify handler types
                    handler_types = [type(h).__name__ for h in handlers]
                    assert (
                        "StreamHandler" in handler_types
                    ), "Should have StreamHandler for console"
                    assert (
                        handler_types.count("FileHandler") == 2
                    ), "Should have two FileHandler instances"

                    # Verify log directories were created
                    assert Path(temp_dir).exists()
                    assert (Path(temp_dir) / "error").exists()

                finally:
                    # Close and remove all handlers to release file locks
                    for handler in root_logger.handlers[:]:
                        handler.close()
                        root_logger.removeHandler(handler)

    def test_setup_production_logging_handles_io_error(self):
        """Test that production logging gracefully handles IO errors."""
        # Mock get_config to return a config that will cause IO error
        with patch("src.config.logging_config.get_config") as mock_config:

            class MockLoggingConfig:
                log_directory = "/nonexistent/readonly/path"
                main_log_file = "app.log"
                error_log_file = "error.log"

            class MockConfig:
                logging = MockLoggingConfig()

            mock_config.return_value = MockConfig()

            # Mock Path.mkdir to raise PermissionError
            with patch(
                "pathlib.Path.mkdir",
                side_effect=PermissionError("Mock permission error"),
            ):
                from src.config.logging_config import _setup_production_logging

                # Clear any existing handlers
                root_logger = logging.getLogger()
                root_logger.handlers.clear()

                # This should not raise an exception
                _setup_production_logging(logging.INFO)

                # Should fall back to console-only logging
                handlers = root_logger.handlers
                assert (
                    len(handlers) == 1
                ), "Should have only console handler as fallback"
                assert isinstance(
                    handlers[0], logging.StreamHandler
                ), "Should be StreamHandler"

    def test_production_logging_uses_correct_formatters(self):
        """Test that production logging uses JSON formatter for console and standard for files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("src.config.logging_config.get_config") as mock_config:

                class MockLoggingConfig:
                    log_directory = temp_dir
                    main_log_file = "app.log"
                    error_log_file = "error.log"

                class MockConfig:
                    logging = MockLoggingConfig()

                mock_config.return_value = MockConfig()

                from src.config.logging_config import _setup_production_logging
                from pythonjsonlogger import jsonlogger

                # Clear any existing handlers
                root_logger = logging.getLogger()
                root_logger.handlers.clear()

                try:
                    # Setup production logging
                    _setup_production_logging(logging.INFO)

                    handlers = root_logger.handlers

                    # Find console and file handlers
                    console_handler = None
                    file_handlers = []

                    for handler in handlers:
                        if isinstance(
                            handler, logging.StreamHandler
                        ) and not isinstance(handler, logging.FileHandler):
                            console_handler = handler
                        elif isinstance(handler, logging.FileHandler):
                            file_handlers.append(handler)

                    # Verify console handler uses JSON formatter
                    assert console_handler is not None, "Should have console handler"
                    assert isinstance(
                        console_handler.formatter, jsonlogger.JsonFormatter
                    ), "Console handler should use JsonFormatter"

                    # Verify file handlers use standard formatter
                    assert len(file_handlers) == 2, "Should have two file handlers"
                    for handler in file_handlers:
                        assert isinstance(
                            handler.formatter, logging.Formatter
                        ), "File handlers should use standard Formatter"
                        assert not isinstance(
                            handler.formatter, jsonlogger.JsonFormatter
                        ), "File handlers should not use JsonFormatter"

                finally:
                    # Close and remove all handlers to release file locks
                    for handler in root_logger.handlers[:]:
                        handler.close()
                        root_logger.removeHandler(handler)
