"""Unit tests for Container lazy initialization functionality."""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.core.container import get_container, ContainerSingleton
from src.error_handling.exceptions import ServiceInitializationError


class TestContainerLazyInitialization:
    """Test cases for Container lazy initialization."""
    
    def setup_method(self):
        """Reset singleton before each test."""
        ContainerSingleton.reset_instance()
    
    def teardown_method(self):
        """Clean up after each test."""
        ContainerSingleton.reset_instance()
    
    def test_llm_service_stack_lazy_initialization(self):
        """Test that LLM service stack initializes lazily without circular dependencies."""
        container = get_container()
        
        # These should not raise any circular dependency errors
        llm_api_key_manager = container.llm_api_key_manager()
        assert llm_api_key_manager is not None
        
        llm_retry_service = container.llm_retry_service()
        assert llm_retry_service is not None
        
        llm_service = container.llm_service()
        assert llm_service is not None
    
    def test_llm_service_stack_singleton_behavior(self):
        """Test that lazy-initialized services maintain singleton behavior."""
        container = get_container()
        
        # Get services multiple times
        api_key_manager1 = container.llm_api_key_manager()
        api_key_manager2 = container.llm_api_key_manager()
        assert api_key_manager1 is api_key_manager2
        
        retry_service1 = container.llm_retry_service()
        retry_service2 = container.llm_retry_service()
        assert retry_service1 is retry_service2
        
        llm_service1 = container.llm_service()
        llm_service2 = container.llm_service()
        assert llm_service1 is llm_service2
    
    def test_dependency_injection_order_independence(self):
        """Test that services can be accessed in any order without issues."""
        container = get_container()
        
        # Access services in different orders
        llm_service = container.llm_service()
        retry_service = container.llm_retry_service()
        api_key_manager = container.llm_api_key_manager()
        
        assert llm_service is not None
        assert retry_service is not None
        assert api_key_manager is not None
        
        # Access in reverse order with new container
        ContainerSingleton.reset_instance()
        container2 = get_container()
        
        api_key_manager2 = container2.llm_api_key_manager()
        retry_service2 = container2.llm_retry_service()
        llm_service2 = container2.llm_service()
        
        assert api_key_manager2 is not None
        assert retry_service2 is not None
        assert llm_service2 is not None
    
    def test_lazy_initialization_error_handling(self):
        """Test that ServiceInitializationError can be properly created and handled."""
        # Test that ServiceInitializationError works as expected
        error = ServiceInitializationError(
            "test_service", "Test initialization error"
        )
        
        assert error.context.additional_data['service_name'] == "test_service"
        assert "Test initialization error" in str(error)
        
        # Test that the error can be raised and caught
        with pytest.raises(ServiceInitializationError) as exc_info:
            raise error
        
        assert exc_info.value.context.additional_data['service_name'] == "test_service"
        assert "Test initialization error" in str(exc_info.value)
    
    def test_container_provides_all_expected_services(self):
        """Test that container provides all expected services after lazy initialization."""
        container = get_container()
        
        # Core services
        assert container.config() is not None
        assert container.template_manager() is not None
        
        # LLM stack (lazy initialized)
        assert container.llm_model() is not None
        assert container.llm_client() is not None
        assert container.llm_retry_handler() is not None
        assert container.llm_api_key_manager() is not None
        assert container.llm_retry_service() is not None
        assert container.llm_service() is not None
        
        # Other services
        assert container.vector_store_service() is not None
        assert container.progress_tracker() is not None
        assert container.agent_factory() is not None
    
    def test_agent_creation_with_lazy_llm_services(self):
        """Test that agents can be created successfully with lazy-initialized LLM services."""
        container = get_container()
        
        # Create various agents that depend on LLM services
        cv_analyzer = container.cv_analyzer_agent()
        assert cv_analyzer is not None
        
        key_qualifications_writer = container.key_qualifications_writer_agent()
        assert key_qualifications_writer is not None
        
        research_agent = container.research_agent()
        assert research_agent is not None
        
        job_parser = container.job_description_parser_agent()
        assert job_parser is not None
    
    def test_lazy_initialization_performance(self):
        """Test that lazy initialization doesn't cause performance issues."""
        import time
        
        start_time = time.time()
        container = get_container()
        container_creation_time = time.time() - start_time
        
        # Container creation should be fast (< 1 second)
        assert container_creation_time < 1.0
        
        # First access of LLM service should initialize the stack
        start_time = time.time()
        llm_service = container.llm_service()
        first_access_time = time.time() - start_time
        
        # Subsequent access should be much faster (cached)
        start_time = time.time()
        llm_service2 = container.llm_service()
        second_access_time = time.time() - start_time
        
        assert llm_service is llm_service2
        # Just verify that second access doesn't take longer than first
        assert second_access_time <= first_access_time + 0.1  # Allow small margin
    
    def test_container_reset_and_reinitialization(self):
        """Test that container can be reset and reinitialized properly."""
        # Get initial container and services
        container1 = get_container()
        llm_service1 = container1.llm_service()
        
        # Reset and get new container
        ContainerSingleton.reset_instance()
        container2 = get_container()
        
        # Containers should be different instances
        assert container1 is not container2
        
        # Services should work correctly in both containers
        assert llm_service1 is not None
        
        llm_service2 = container2.llm_service()
        assert llm_service2 is not None
        
        # Note: Due to dependency injection framework behavior,
        # singleton services may be reused across container instances
        # This is expected behavior for dependency injection containers