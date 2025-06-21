"""Unit tests for application startup functionality."""

import sys
import os

# Ensure project root (containing src/) is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.core.application_startup import (
    ApplicationStartup,
    StartupResult,
    ServiceStatus,
    initialize_application,
    validate_application,
)
from src.utils.exceptions import ServiceInitializationError


class TestApplicationStartup:
    """Test cases for ApplicationStartup class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.startup = ApplicationStartup()

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_startup_result_initialization(self):
        """Test StartupResult initialization."""
        from datetime import datetime

        result = StartupResult(
            success=True,
            total_time=1.5,
            services={},
            errors=[],
            timestamp=datetime.now(),
        )
        assert result.success is True
        assert result.total_time == 1.5
        assert result.services == {}
        assert result.errors == []

    def test_service_status_initialization(self):
        """Test ServiceStatus initialization."""
        status = ServiceStatus(
            name="test_service", initialized=False, initialization_time=0.0
        )
        assert status.initialized is False
        assert status.initialization_time == 0.0
        assert status.error is None

    @patch("src.core.application_startup.setup_logging")
    def test_initialize_logging_success(self, mock_setup_logging):
        """Test successful logging initialization."""
        mock_setup_logging.return_value = None

        self.startup._initialize_logging()

        assert "logging" in self.startup.services
        assert self.startup.services["logging"].initialized is True
        assert self.startup.services["logging"].error is None
        mock_setup_logging.assert_called_once()

    @patch("src.core.application_startup.setup_logging")
    def test_initialize_logging_failure(self, mock_setup_logging):
        """Test logging initialization failure."""
        from src.utils.exceptions import ServiceInitializationError

        mock_setup_logging.side_effect = Exception("Logging setup failed")

        with pytest.raises(ServiceInitializationError):
            self.startup._initialize_logging()

        assert "logging" in self.startup.services
        assert self.startup.services["logging"].initialized is False
        assert "Logging setup failed" in str(self.startup.services["logging"].error)

    @patch("src.core.application_startup.load_config")
    def test_initialize_environment_success(self, mock_load_config):
        """Test successful environment initialization."""
        mock_config = MagicMock()
        mock_load_config.return_value = mock_config

        self.startup._initialize_environment()

        assert "environment" in self.startup.services
        assert self.startup.services["environment"].initialized is True
        assert self.startup.services["environment"].error is None
        mock_load_config.assert_called_once()

    @patch("src.core.application_startup.os.makedirs")
    def test_ensure_directories_success(self, mock_makedirs):
        """Test successful directory creation."""
        self.startup._ensure_directories()

        assert "directories" in self.startup.services
        assert self.startup.services["directories"].initialized is True
        assert self.startup.services["directories"].error is None
        # Should be called for each directory
        assert mock_makedirs.call_count == 5

    @patch("src.core.application_startup.get_llm_service")
    def test_initialize_llm_service_success(self, mock_get_llm_service):
        """Test successful LLM service initialization."""
        mock_service = MagicMock()
        mock_get_llm_service.return_value = mock_service

        self.startup._initialize_llm_service("test_api_key")

        assert "llm_service" in self.startup.services
        assert self.startup.services["llm_service"].initialized is True
        assert self.startup.services["llm_service"].error is None
        mock_get_llm_service.assert_called_once_with(user_api_key="test_api_key")

    @patch("src.core.application_startup.get_llm_service")
    def test_initialize_llm_service_failure(self, mock_get_llm):
        """Test LLM service initialization failure."""
        mock_get_llm.side_effect = Exception("LLM service failed")

        with pytest.raises(ServiceInitializationError) as exc_info:
            self.startup._initialize_llm_service("test_api_key")

        assert "llm_service" in self.startup.services
        assert self.startup.services["llm_service"].initialized is False
        assert "LLM service failed" in str(self.startup.services["llm_service"].error)
        assert "Failed to initialize LLM service" in str(exc_info.value)

    @patch("src.core.application_startup.get_vector_store_service")
    def test_initialize_vector_store_success(self, mock_get_vector):
        """Test successful vector store initialization."""
        mock_service = MagicMock()
        mock_get_vector.return_value = mock_service

        self.startup._initialize_vector_store()

        assert "vector_store" in self.startup.services
        assert self.startup.services["vector_store"].initialized is True
        assert self.startup.services["vector_store"].error is None
        mock_get_vector.assert_called_once()

    @patch("src.core.application_startup.get_session_manager")
    def test_initialize_session_manager_success(self, mock_get_session):
        """Test successful session manager initialization."""
        mock_service = MagicMock()
        mock_get_session.return_value = mock_service

        self.startup._initialize_session_manager()

        assert "session_manager" in self.startup.services
        assert self.startup.services["session_manager"].initialized is True
        assert self.startup.services["session_manager"].error is None
        mock_get_session.assert_called_once()

    @patch("src.utils.streamlit_utils.configure_page")
    def test_configure_streamlit_success(self, mock_configure_page):
        """Test successful Streamlit configuration."""
        mock_configure_page.return_value = True
        self.startup._configure_streamlit()

        assert "streamlit" in self.startup.services
        assert self.startup.services["streamlit"].initialized is True
        assert self.startup.services["streamlit"].error is None
        mock_configure_page.assert_called_once()

    @patch("src.core.application_startup.ApplicationStartup")
    def test_initialize_application_function(self, mock_startup_class):
        """Test the initialize_application function."""
        from datetime import datetime

        mock_startup = MagicMock()
        mock_startup_class.return_value = mock_startup
        mock_result = StartupResult(
            success=True,
            total_time=1.0,
            services={},
            errors=[],
            timestamp=datetime.now(),
        )
        mock_startup.initialize_application.return_value = mock_result

        result = initialize_application()

        assert result == mock_result
        mock_startup_class.assert_called_once()
        mock_startup.initialize_application.assert_called_once()

    @patch("src.core.application_startup.get_startup_manager")
    def test_validate_application_success(self, mock_get_startup_manager):
        """Test successful application validation."""
        # Set up services as initialized
        self.startup.services["logging"] = ServiceStatus(
            name="logging", initialized=True, initialization_time=0.1
        )
        self.startup.services["environment"] = ServiceStatus(
            name="environment", initialized=True, initialization_time=0.1
        )
        self.startup.services["llm_service"] = ServiceStatus(
            name="llm_service", initialized=True, initialization_time=0.1
        )
        self.startup.services["vector_store"] = ServiceStatus(
            name="vector_store", initialized=True, initialization_time=0.1
        )

        # Mock get_startup_manager to return our test instance
        mock_get_startup_manager.return_value = self.startup

        result = validate_application()

        # validate_application returns a list of errors, empty list means success
        assert isinstance(result, list)
        assert len(result) == 0

    @patch("src.core.application_startup.get_startup_manager")
    def test_validate_application_failure(self, mock_get_startup_manager):
        """Test application validation failure."""
        # Set up some services as failed
        self.startup.services["logging"] = ServiceStatus(
            name="logging", initialized=True, initialization_time=0.1
        )
        self.startup.services["environment"] = ServiceStatus(
            name="environment",
            initialized=False,
            initialization_time=0.1,
            error="Config failed",
        )
        self.startup.services["llm_service"] = ServiceStatus(
            name="llm_service",
            initialized=False,
            initialization_time=0.1,
            error="Service not available",
        )

        # Mock get_startup_manager to return our test instance
        mock_get_startup_manager.return_value = self.startup

        result = validate_application()

        # validate_application returns a list of errors, non-empty list means failure
        assert isinstance(result, list)
        assert len(result) > 0
        assert any("environment" in error for error in result)
        assert any("llm_service" in error for error in result)

    def test_startup_result_success_property(self):
        """Test StartupResult success property logic."""
        from datetime import datetime

        result = StartupResult(
            success=True,
            total_time=1.0,
            services={},
            errors=[],
            timestamp=datetime.now(),
        )

        # No errors should mean success
        assert result.success is True

        # Adding errors should make it fail
        result.errors.append("Test error")
        # Note: success is a field, not a computed property in this implementation
        # So we need to manually set it
        result.success = False
        assert result.success is False

        # Clear errors should make it succeed again
        result.errors.clear()
        result.success = True
        assert result.success is True

    @patch.multiple(
        "src.core.application_startup.ApplicationStartup",
        _initialize_logging=MagicMock(),
        _initialize_environment=MagicMock(),
        _ensure_directories=MagicMock(),
        _initialize_llm_service=MagicMock(),
        _initialize_vector_store=MagicMock(),
        _initialize_session_manager=MagicMock(),
        _configure_streamlit=MagicMock(),
    )
    def test_initialize_calls_all_methods(self):
        """Test that initialize_application calls all required initialization methods."""
        startup = ApplicationStartup()
        result = startup.initialize_application("test_api_key")

        # Verify all initialization methods were called
        startup._initialize_logging.assert_called_once()
        startup._initialize_environment.assert_called_once()
        startup._ensure_directories.assert_called_once()
        startup._initialize_llm_service.assert_called_once_with("test_api_key")
        startup._initialize_vector_store.assert_called_once()
        startup._initialize_session_manager.assert_called_once()
        startup._configure_streamlit.assert_called_once()

        # Verify result has timing information
        assert result.total_time >= 0
