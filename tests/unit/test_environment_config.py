#!/usr/bin/env python3
"""
Unit Tests for Environment Configuration

Tests the environment configuration system including different environments,
configuration loading, and environment variable overrides.
"""

import pytest
import unittest
import os
from unittest.mock import patch, MagicMock
from pathlib import Path
import tempfile
import logging

from src.config.environment import (
    Environment,
    AppConfig,
    DatabaseConfig,
    LoggingConfig,
    SecurityConfig,
    UIConfig,
    PerformanceConfig,
    get_environment,
    load_config,
    _override_from_env
)


class TestEnvironmentEnum(unittest.TestCase):
    """Test the Environment enum."""
    
    def test_environment_values(self):
        """Test that all expected environment values exist."""
        self.assertEqual(Environment.DEVELOPMENT.value, "development")
        self.assertEqual(Environment.TESTING.value, "testing")
        self.assertEqual(Environment.PRODUCTION.value, "production")
    
    def test_environment_from_string(self):
        """Test creating Environment from string values."""
        self.assertEqual(Environment("development"), Environment.DEVELOPMENT)
        self.assertEqual(Environment("testing"), Environment.TESTING)
        self.assertEqual(Environment("production"), Environment.PRODUCTION)


class TestDatabaseConfig(unittest.TestCase):
    """Test DatabaseConfig dataclass."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = DatabaseConfig()
        
        self.assertEqual(config.vector_db_path, "data/enhanced_vector_db")
        self.assertEqual(config.session_db_path, "data/sessions")
        self.assertTrue(config.backup_enabled)
        self.assertEqual(config.backup_interval_hours, 24)
        self.assertEqual(config.max_backups, 7)
    
    def test_custom_values(self):
        """Test custom configuration values."""
        config = DatabaseConfig(
            vector_db_path="custom/vector/path",
            backup_enabled=False,
            max_backups=3
        )
        
        self.assertEqual(config.vector_db_path, "custom/vector/path")
        self.assertFalse(config.backup_enabled)
        self.assertEqual(config.max_backups, 3)


class TestLoggingConfig(unittest.TestCase):
    """Test LoggingConfig dataclass."""
    
    def test_default_values(self):
        """Test default logging configuration."""
        config = LoggingConfig()
        
        self.assertEqual(config.level, "INFO")
        self.assertTrue(config.log_to_file)
        self.assertTrue(config.log_to_console)
        self.assertEqual(config.max_file_size_mb, 10)
        self.assertEqual(config.backup_count, 5)
        self.assertTrue(config.structured_logging)
        self.assertFalse(config.performance_logging)
    
    def test_get_log_level(self):
        """Test log level conversion to logging constants."""
        config = LoggingConfig(level="DEBUG")
        self.assertEqual(config.get_log_level(), logging.DEBUG)
        
        config = LoggingConfig(level="INFO")
        self.assertEqual(config.get_log_level(), logging.INFO)
        
        config = LoggingConfig(level="WARNING")
        self.assertEqual(config.get_log_level(), logging.WARNING)
        
        config = LoggingConfig(level="ERROR")
        self.assertEqual(config.get_log_level(), logging.ERROR)
    
    def test_invalid_log_level(self):
        """Test handling of invalid log level."""
        config = LoggingConfig(level="INVALID")
        # Should default to INFO for invalid levels
        self.assertEqual(config.get_log_level(), logging.INFO)


class TestSecurityConfig(unittest.TestCase):
    """Test SecurityConfig dataclass."""
    
    def test_default_values(self):
        """Test default security configuration."""
        config = SecurityConfig()
        
        self.assertTrue(config.api_key_validation)
        self.assertTrue(config.rate_limiting_enabled)
        self.assertEqual(config.session_timeout_minutes, 60)
        self.assertEqual(config.max_file_upload_size_mb, 10)
        self.assertEqual(config.allowed_file_types, [".txt", ".md", ".pdf", ".docx"])
    
    def test_custom_security_settings(self):
        """Test custom security configuration."""
        config = SecurityConfig(
            api_key_validation=False,
            session_timeout_minutes=30,
            allowed_file_types=[".txt", ".md"]
        )
        
        self.assertFalse(config.api_key_validation)
        self.assertEqual(config.session_timeout_minutes, 30)
        self.assertEqual(config.allowed_file_types, [".txt", ".md"])


class TestUIConfig(unittest.TestCase):
    """Test UIConfig dataclass."""
    
    def test_default_values(self):
        """Test default UI configuration."""
        config = UIConfig()
        
        self.assertEqual(config.page_title, "AI CV Generator")
        self.assertEqual(config.page_icon, "📄")
        self.assertEqual(config.layout, "wide")
        self.assertEqual(config.sidebar_state, "expanded")
        self.assertEqual(config.theme_primary_color, "#1f77b4")
        self.assertFalse(config.show_debug_info)
        self.assertEqual(config.auto_save_interval_seconds, 30)


class TestPerformanceConfig(unittest.TestCase):
    """Test PerformanceConfig dataclass."""
    
    def test_default_values(self):
        """Test default performance configuration."""
        config = PerformanceConfig()
        
        self.assertTrue(config.enable_caching)
        self.assertEqual(config.cache_ttl_seconds, 3600)
        self.assertEqual(config.max_concurrent_requests, 10)
        self.assertEqual(config.request_timeout_seconds, 300)
        self.assertFalse(config.enable_profiling)
        self.assertIsNone(config.memory_limit_mb)


class TestAppConfig(unittest.TestCase):
    """Test AppConfig dataclass and its post-initialization."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
    
    def tearDown(self):
        """Clean up test fixtures."""
        os.chdir(self.original_cwd)
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_development_environment_adjustments(self):
        """Test configuration adjustments for development environment."""
        config = AppConfig(
            environment=Environment.DEVELOPMENT,
            debug=True,
            testing=False
        )
        
        # Check development-specific settings
        self.assertTrue(config.logging.performance_logging)
        self.assertTrue(config.ui.show_debug_info)
        self.assertTrue(config.performance.enable_profiling)
    
    def test_testing_environment_adjustments(self):
        """Test configuration adjustments for testing environment."""
        config = AppConfig(
            environment=Environment.TESTING,
            debug=True,
            testing=True
        )
        
        # Check testing-specific settings
        self.assertFalse(config.logging.log_to_console)
        self.assertFalse(config.database.backup_enabled)
        self.assertFalse(config.performance.enable_caching)
    
    def test_production_environment_adjustments(self):
        """Test configuration adjustments for production environment."""
        config = AppConfig(
            environment=Environment.PRODUCTION,
            debug=False,
            testing=False
        )
        
        # Check production-specific settings
        self.assertFalse(config.debug)
        self.assertEqual(config.logging.level, "WARNING")
        self.assertFalse(config.ui.show_debug_info)
        self.assertFalse(config.performance.enable_profiling)
    
    def test_directory_creation(self):
        """Test that required directories are created."""
        config = AppConfig(
            environment=Environment.DEVELOPMENT,
            debug=True,
            testing=False
        )
        
        # Check that directories were created
        self.assertTrue(config.data_dir.exists())
        self.assertTrue(config.logs_dir.exists())
    
    def test_custom_paths(self):
        """Test custom path configuration."""
        custom_data_dir = self.temp_dir / "custom_data"
        custom_logs_dir = self.temp_dir / "custom_logs"
        
        config = AppConfig(
            environment=Environment.DEVELOPMENT,
            debug=True,
            testing=False,
            data_dir=custom_data_dir,
            logs_dir=custom_logs_dir
        )
        
        self.assertEqual(config.data_dir, custom_data_dir)
        self.assertEqual(config.logs_dir, custom_logs_dir)
        self.assertTrue(custom_data_dir.exists())
        self.assertTrue(custom_logs_dir.exists())


class TestEnvironmentDetection(unittest.TestCase):
    """Test environment detection from environment variables."""
    
    def test_get_environment_development(self):
        """Test getting development environment."""
        with patch.dict(os.environ, {'APP_ENV': 'development'}):
            env = get_environment()
            self.assertEqual(env, Environment.DEVELOPMENT)
    
    def test_get_environment_testing(self):
        """Test getting testing environment."""
        with patch.dict(os.environ, {'APP_ENV': 'testing'}):
            env = get_environment()
            self.assertEqual(env, Environment.TESTING)
    
    def test_get_environment_production(self):
        """Test getting production environment."""
        with patch.dict(os.environ, {'APP_ENV': 'production'}):
            env = get_environment()
            self.assertEqual(env, Environment.PRODUCTION)
    
    def test_get_environment_default(self):
        """Test default environment when APP_ENV is not set."""
        with patch.dict(os.environ, {}, clear=True):
            env = get_environment()
            self.assertEqual(env, Environment.DEVELOPMENT)
    
    def test_get_environment_invalid(self):
        """Test handling of invalid environment value."""
        with patch.dict(os.environ, {'APP_ENV': 'invalid'}):
            env = get_environment()
            self.assertEqual(env, Environment.DEVELOPMENT)
    
    def test_get_environment_case_insensitive(self):
        """Test that environment detection is case insensitive."""
        with patch.dict(os.environ, {'APP_ENV': 'PRODUCTION'}):
            env = get_environment()
            self.assertEqual(env, Environment.PRODUCTION)


class TestConfigurationLoading(unittest.TestCase):
    """Test configuration loading functionality."""
    
    def test_load_config_basic(self):
        """Test basic configuration loading."""
        with patch.dict(os.environ, {'APP_ENV': 'development', 'DEBUG': 'true', 'TESTING': 'false'}):
            config = load_config()
            
            self.assertEqual(config.environment, Environment.DEVELOPMENT)
            self.assertTrue(config.debug)
            self.assertFalse(config.testing)
    
    def test_load_config_testing_mode(self):
        """Test configuration loading in testing mode."""
        with patch.dict(os.environ, {
            'APP_ENV': 'testing',
            'DEBUG': 'false',
            'TESTING': 'true'
        }):
            config = load_config()
            
            self.assertEqual(config.environment, Environment.TESTING)
            self.assertFalse(config.debug)
            self.assertTrue(config.testing)


class TestEnvironmentVariableOverrides(unittest.TestCase):
    """Test environment variable overrides for configuration."""
    
    def test_log_level_override(self):
        """Test log level override from environment variable."""
        config = AppConfig(
            environment=Environment.DEVELOPMENT,
            debug=True,
            testing=False
        )
        
        with patch.dict(os.environ, {'LOG_LEVEL': 'ERROR'}):
            _override_from_env(config)
            
        self.assertEqual(config.logging.level, 'ERROR')
    
    def test_log_to_file_override(self):
        """Test log to file override from environment variable."""
        config = AppConfig(
            environment=Environment.DEVELOPMENT,
            debug=True,
            testing=False
        )
        
        with patch.dict(os.environ, {'LOG_TO_FILE': 'false'}):
            _override_from_env(config)
            
        self.assertFalse(config.logging.log_to_file)
    
    def test_caching_override(self):
        """Test caching override from environment variable."""
        config = AppConfig(
            environment=Environment.DEVELOPMENT,
            debug=True,
            testing=False
        )
        
        with patch.dict(os.environ, {'ENABLE_CACHING': 'false'}):
            _override_from_env(config)
            
        self.assertFalse(config.performance.enable_caching)
    
    def test_request_timeout_override(self):
        """Test request timeout override from environment variable."""
        config = AppConfig(
            environment=Environment.DEVELOPMENT,
            debug=True,
            testing=False
        )
        
        with patch.dict(os.environ, {'REQUEST_TIMEOUT': '600'}):
            _override_from_env(config)
            
        self.assertEqual(config.performance.request_timeout_seconds, 600)
    
    def test_session_timeout_override(self):
        """Test session timeout override from environment variable."""
        config = AppConfig(
            environment=Environment.DEVELOPMENT,
            debug=True,
            testing=False
        )
        
        with patch.dict(os.environ, {'SESSION_TIMEOUT_MINUTES': '120'}):
            _override_from_env(config)
            
        self.assertEqual(config.security.session_timeout_minutes, 120)
    
    def test_invalid_numeric_override(self):
        """Test handling of invalid numeric environment variables."""
        config = AppConfig(
            environment=Environment.DEVELOPMENT,
            debug=True,
            testing=False
        )
        
        original_timeout = config.performance.request_timeout_seconds
        
        with patch.dict(os.environ, {'REQUEST_TIMEOUT': 'invalid'}):
            _override_from_env(config)
            
        # Should remain unchanged for invalid values
        self.assertEqual(config.performance.request_timeout_seconds, original_timeout)


if __name__ == '__main__':
    unittest.main()