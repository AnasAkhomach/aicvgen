"""
Simple integration test that avoids the logging issue.
"""

import pytest
from unittest.mock import patch


class TestSimpleIntegration:
    """Simple integration tests to validate basic functionality."""

    def test_basic_imports(self):
        """Test that basic imports work without critical errors."""
        try:
            from src.config.settings import get_config

            config = get_config()
            assert config is not None
            print("✅ Configuration import successful")
        except Exception as e:
            pytest.fail(f"Configuration import failed: {str(e)}")

    def test_dependency_injection_basic(self):
        """Test basic dependency injection without full agent initialization."""
        try:
            from src.core.dependency_injection import DependencyContainer

            container = DependencyContainer()
            assert container is not None
            print("✅ Basic DI container creation successful")
        except Exception as e:
            pytest.fail(f"DI container creation failed: {str(e)}")

    def test_model_imports(self):
        """Test that all data models can be imported."""
        try:
            from src.models.data_models import StructuredCV, JobDescriptionData
            from src.orchestration.state import AgentState
            from src.models.agent_output_models import ParserAgentOutput

            # Test basic model creation
            assert AgentState is not None
            assert StructuredCV is not None
            assert JobDescriptionData is not None
            assert ParserAgentOutput is not None

            print("✅ Model imports successful")
        except Exception as e:
            pytest.fail(f"Model imports failed: {str(e)}")

    def test_agent_base_imports(self):
        """Test that agent base classes can be imported."""
        try:
            from src.agents.agent_base import EnhancedAgentBase, AgentResult

            assert EnhancedAgentBase is not None
            assert AgentResult is not None
            print("✅ Agent base imports successful")
        except Exception as e:
            pytest.fail(f"Agent base imports failed: {str(e)}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
