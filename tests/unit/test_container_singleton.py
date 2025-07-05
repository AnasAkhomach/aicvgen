"""Unit tests for Container singleton behavior."""

import pytest
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from core.container import Container, ContainerSingleton, get_container


class TestContainerSingleton:
    """Test cases for Container singleton enforcement."""
    
    def setup_method(self):
        """Reset singleton before each test."""
        ContainerSingleton.reset_instance()
    
    def teardown_method(self):
        """Clean up after each test."""
        ContainerSingleton.reset_instance()
    
    def test_direct_instantiation_raises_error(self):
        """Test that direct Container instantiation raises RuntimeError."""
        with pytest.raises(RuntimeError, match="Container cannot be instantiated directly"):
            Container()
    
    def test_get_container_returns_singleton(self):
        """Test that get_container() returns the same instance."""
        container1 = get_container()
        container2 = get_container()
        
        assert container1 is container2
        assert isinstance(container1, Container)
    
    def test_container_singleton_get_instance_works(self):
        """Test that ContainerSingleton.get_instance() works correctly."""
        container1 = ContainerSingleton.get_instance()
        container2 = ContainerSingleton.get_instance()
        
        assert container1 is container2
        assert isinstance(container1, Container)
    
    def test_reset_instance_allows_new_creation(self):
        """Test that reset_instance allows creating a new singleton."""
        container1 = get_container()
        ContainerSingleton.reset_instance()
        container2 = get_container()
        
        # Should be different instances after reset
        assert container1 is not container2
        assert isinstance(container1, Container)
        assert isinstance(container2, Container)
    
    def test_container_provides_expected_services(self):
        """Test that the singleton container provides expected services."""
        container = get_container()
        
        # Test core services
        config = container.config()
        assert config is not None
        
        llm_service = container.llm_service()
        assert llm_service is not None
        
        # Test singleton behavior for services
        config2 = container.config()
        assert config is config2