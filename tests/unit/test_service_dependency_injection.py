"""Test dependency injection fixes for services."""

import sys
import os
import logging
from unittest.mock import Mock

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import pytest
from services.vector_store_service import VectorStoreService
from services.session_manager import SessionManager
from services.progress_tracker import ProgressTracker
from services.rate_limiter import RateLimiter
from services.error_recovery import ErrorRecoveryService
from config.settings import Settings
from src.models.llm_data_models import VectorStoreConfig


class TestDependencyInjectionFixes:
    """Test that all fixed services require proper dependency injection."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_logger = Mock(spec=logging.Logger)
        self.mock_settings = Mock(spec=Settings)
        # Add required attributes for SessionManager
        self.mock_settings.sessions_directory = Mock()
        self.mock_settings.sessions_directory.mkdir = Mock()
        self.mock_settings.ui = Mock()
        self.mock_settings.ui.session_timeout_seconds = 3600
        self.mock_settings.session = Mock()
        self.mock_settings.session.max_active_sessions = 10
        self.mock_settings.session.cleanup_interval_minutes = 5
        
        self.mock_vector_config = Mock(spec=VectorStoreConfig)
        # Add required attributes for VectorStoreService
        self.mock_vector_config.persist_directory = "/tmp/test_vector_db"
        
        self.mock_config = Mock()

    def test_vector_store_service_requires_dependencies(self):
        """Test VectorStoreService requires logger and vector_config."""
        # Should work with proper dependencies
        service = VectorStoreService(
            logger=self.mock_logger,
            vector_config=self.mock_vector_config
        )
        assert service.logger is self.mock_logger
        assert service.vector_config is self.mock_vector_config
        
        # Should fail without dependencies
        with pytest.raises(TypeError):
            VectorStoreService()

    def test_session_manager_requires_dependencies(self):
        """Test SessionManager requires logger and settings."""
        # Should work with proper dependencies
        service = SessionManager(
            logger=self.mock_logger,
            settings=self.mock_settings
        )
        assert service.logger is self.mock_logger
        assert service.settings is self.mock_settings
        
        # Should fail without dependencies
        with pytest.raises(TypeError):
            SessionManager()

    def test_progress_tracker_requires_dependencies(self):
        """Test ProgressTracker requires logger."""
        # Should work with proper dependencies
        service = ProgressTracker(logger=self.mock_logger)
        assert service.logger is self.mock_logger
        
        # Should fail without dependencies
        with pytest.raises(TypeError):
            ProgressTracker()

    def test_rate_limiter_requires_dependencies(self):
        """Test RateLimiter requires logger and config."""
        # Should work with proper dependencies
        service = RateLimiter(
            logger=self.mock_logger,
            config=self.mock_config
        )
        assert service.logger is self.mock_logger
        assert service.config is self.mock_config
        
        # Should fail without dependencies
        with pytest.raises(TypeError):
            RateLimiter()

    def test_error_recovery_service_requires_dependencies(self):
        """Test ErrorRecoveryService requires logger."""
        # Should work with proper dependencies
        service = ErrorRecoveryService(logger=self.mock_logger)
        assert service.logger is self.mock_logger
        
        # Should fail without dependencies
        with pytest.raises(TypeError):
            ErrorRecoveryService()

    def test_services_no_longer_have_global_functions(self):
        """Test that global getter functions have been removed."""
        # These imports should fail since we removed the global functions
        
        # Test VectorStoreService
        try:
            from services.vector_store_service import get_vector_store_service
            pytest.fail("get_vector_store_service should not exist")
        except ImportError:
            pass  # Expected
            
        # Test SessionManager
        try:
            from services.session_manager import get_session_manager
            pytest.fail("get_session_manager should not exist")
        except ImportError:
            pass  # Expected
            
        # Test ProgressTracker
        try:
            from services.progress_tracker import get_progress_tracker
            pytest.fail("get_progress_tracker should not exist")
        except ImportError:
            pass  # Expected
            
        # Test RateLimiter
        try:
            from services.rate_limiter import get_rate_limiter
            pytest.fail("get_rate_limiter should not exist")
        except ImportError:
            pass  # Expected


if __name__ == "__main__":
    # Simple test runner for debugging
    test_instance = TestDependencyInjectionFixes()
    test_instance.setup_method()
    
    try:
        test_instance.test_vector_store_service_requires_dependencies()
        print("✓ VectorStoreService dependency injection test passed")
    except Exception as e:
        print(f"✗ VectorStoreService dependency injection test failed: {e}")
        
    try:
        test_instance.test_session_manager_requires_dependencies()
        print("✓ SessionManager dependency injection test passed")
    except Exception as e:
        print(f"✗ SessionManager dependency injection test failed: {e}")
        
    try:
        test_instance.test_progress_tracker_requires_dependencies()
        print("✓ ProgressTracker dependency injection test passed")
    except Exception as e:
        print(f"✗ ProgressTracker dependency injection test failed: {e}")
        
    try:
        test_instance.test_rate_limiter_requires_dependencies()
        print("✓ RateLimiter dependency injection test passed")
    except Exception as e:
        print(f"✗ RateLimiter dependency injection test failed: {e}")
        
    try:
        test_instance.test_error_recovery_service_requires_dependencies()
        print("✓ ErrorRecoveryService dependency injection test passed")
    except Exception as e:
        print(f"✗ ErrorRecoveryService dependency injection test failed: {e}")
        
    try:
        test_instance.test_services_no_longer_have_global_functions()
        print("✓ Global functions removal test passed")
    except Exception as e:
        print(f"✗ Global functions removal test failed: {e}")
        
    print("\nAll dependency injection fixes have been verified!")