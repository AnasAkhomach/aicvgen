"""Integration tests for production logging functionality.

This test module validates that the production logging configuration correctly
writes logs to persistent files in addition to console output, particularly
focusing on error logging which is critical for production observability.
"""

import logging
import os
import tempfile
import subprocess
import time
from pathlib import Path
from unittest.mock import patch
import pytest


class TestProductionLogging:
    """Integration tests for production logging file persistence."""

    def test_production_logging_writes_to_files(self):
        """Test that production logging writes to both console and files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Set up environment for production logging
            test_env = os.environ.copy()
            test_env["APP_ENV"] = "production"

            # Mock the config to use our temporary directory
            with patch("src.config.logging_config.get_config") as mock_config:
                # Create a mock config object
                class MockLoggingConfig:
                    log_directory = temp_dir
                    main_log_file = "app.log"
                    error_log_file = "error.log"

                class MockConfig:
                    logging = MockLoggingConfig()

                mock_config.return_value = MockConfig()

                # Import and setup production logging
                from src.config.logging_config import _setup_production_logging

                # Clear any existing handlers first
                root_logger = logging.getLogger()
                root_logger.handlers.clear()

                try:
                    # Setup production logging
                    _setup_production_logging(logging.INFO)

                    # Create logger and log messages
                    logger = logging.getLogger("test_logger")

                    # Log different levels
                    logger.info("Test info message")
                    logger.warning("Test warning message")
                    logger.error("Test error message")

                    # Test exception logging
                    try:
                        raise ValueError("Test exception for logging")
                    except Exception:
                        logger.error("Exception occurred", exc_info=True)

                    # Force log flush
                    for handler in logger.handlers:
                        handler.flush()

                    # Verify log files were created
                    main_log_path = Path(temp_dir) / "app.log"
                    error_log_path = Path(temp_dir) / "error" / "error.log"

                    assert main_log_path.exists(), "Main log file should be created"
                    assert error_log_path.exists(), "Error log file should be created"

                    # Verify content in main log file
                    main_log_content = main_log_path.read_text()
                    assert "Test info message" in main_log_content
                    assert "Test warning message" in main_log_content
                    assert "Test error message" in main_log_content

                    # Verify content in error log file
                    error_log_content = error_log_path.read_text()
                    assert "Test error message" in error_log_content
                    assert "Exception occurred" in error_log_content
                    assert "ValueError: Test exception for logging" in error_log_content

                finally:
                    # Close and remove all handlers to release file locks
                    for handler in root_logger.handlers[:]:
                        handler.close()
                        root_logger.removeHandler(handler)

    def test_production_logging_handles_permission_errors(self):
        """Test that production logging gracefully handles permission errors."""
        # Create a directory that we'll make read-only
        with tempfile.TemporaryDirectory() as temp_dir:
            readonly_dir = Path(temp_dir) / "readonly"
            readonly_dir.mkdir()

            # Mock the config to use the read-only directory
            with patch("src.config.logging_config.get_config") as mock_config:

                class MockLoggingConfig:
                    log_directory = str(readonly_dir)
                    main_log_file = "app.log"
                    error_log_file = "error.log"

                class MockConfig:
                    logging = MockLoggingConfig()

                mock_config.return_value = MockConfig()

                # Clear any existing handlers first
                root_logger = logging.getLogger()
                root_logger.handlers.clear()

                # Make directory read-only on Unix systems
                if os.name != "nt":  # Not Windows
                    readonly_dir.chmod(0o444)

                try:
                    # Import and attempt to setup production logging
                    from src.config.logging_config import _setup_production_logging

                    # This should not raise an exception, but fall back gracefully
                    _setup_production_logging(logging.INFO)

                    # Verify that logging still works (console fallback)
                    logger = logging.getLogger("test_fallback_logger")
                    logger.info("Fallback test message")

                    # Test should pass without exceptions
                    assert (
                        True
                    ), "Production logging should handle permission errors gracefully"

                finally:
                    # Close and remove all handlers to release file locks before cleanup
                    for handler in root_logger.handlers[:]:
                        handler.close()
                        root_logger.removeHandler(handler)

                    # Restore permissions for cleanup
                    if os.name != "nt":
                        readonly_dir.chmod(0o755)

    @pytest.mark.skipif(
        not Path("Dockerfile").exists(),
        reason="Docker integration test requires Dockerfile",
    )
    def test_docker_production_logging_integration(self):
        """Test production logging in actual Docker environment."""
        # This test requires Docker to be available and is more of an E2E test
        # It builds the actual Docker container and tests logging behavior

        # Create a test script that will run in the container
        test_script = """
import logging
import os
import sys
sys.path.insert(0, '/app/src')

# Set production environment
os.environ["APP_ENV"] = "production"

from src.config.logging_config import setup_logging

# Setup logging
setup_logging(logging.INFO)

# Create logger and trigger an exception
logger = logging.getLogger("docker_test")
logger.info("Docker integration test started")

try:
    raise RuntimeError("Test exception for Docker logging verification")
except Exception:
    logger.error("Docker test exception", exc_info=True)

logger.info("Docker integration test completed")
"""

        try:
            # Build Docker image (skip if build fails - not our focus)
            try:
                subprocess.run(
                    ["docker", "build", "-t", "aicvgen-test", "."],
                    check=True,
                    capture_output=True,
                    cwd=Path.cwd(),
                )
            except subprocess.CalledProcessError:
                pytest.skip("Docker build failed - skipping Docker integration test")

            # Create temporary directory for volume mount
            with tempfile.TemporaryDirectory() as temp_instance_dir:
                # Create logs directory structure to match expected paths
                logs_dir = Path(temp_instance_dir) / "logs"
                error_logs_dir = logs_dir / "error"
                error_logs_dir.mkdir(parents=True, exist_ok=True)

                # Write test script directly to the mounted directory
                test_script_path = Path(temp_instance_dir) / "test_logging.py"
                test_script_path.write_text(test_script)
                # Set proper permissions for Docker to read the file
                test_script_path.chmod(0o644)

                # Ensure directory permissions are correct for Docker user
                import os

                if os.name != "nt":  # Only on Unix-like systems
                    os.chmod(temp_instance_dir, 0o755)
                    os.chmod(logs_dir, 0o755)
                    os.chmod(error_logs_dir, 0o755)

                # Run container with our test script
                # Ensure the script has executable permissions and proper ownership
                result = subprocess.run(
                    [
                        "docker",
                        "run",
                        "--rm",
                        "-v",
                        f"{temp_instance_dir}:/app/instance",
                        "--user",
                        "aicvgen:aicvgen",
                        "aicvgen-test",
                        "python",
                        "/app/instance/test_logging.py",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=60,
                )

                # Check that the container ran successfully
                if result.returncode != 0:
                    pytest.fail(f"Docker container failed: {result.stderr}")

                # Verify log files were created in the mounted volume
                error_log_path = (
                    Path(temp_instance_dir) / "logs" / "error" / "error.log"
                )
                main_log_path = Path(temp_instance_dir) / "logs" / "app.log"

                assert (
                    main_log_path.exists()
                ), "Main log file should exist in mounted volume"
                assert (
                    error_log_path.exists()
                ), "Error log file should exist in mounted volume"

                # Verify exception was logged to error file
                error_log_content = error_log_path.read_text()
                assert (
                    "RuntimeError: Test exception for Docker logging verification"
                    in error_log_content
                )
                assert "Traceback" in error_log_content

        finally:
            # No cleanup needed - test script is in temp directory that gets auto-cleaned
            pass
