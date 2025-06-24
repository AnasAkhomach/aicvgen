#!/usr/bin/env python3
"""
Simple test to isolate the Settings hanging issue.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def test_settings_creation():
    """Test Settings creation directly."""
    print("Testing Settings creation...")
    try:
        from src.config.environment import load_config

        load_config()
        print("Environment loaded")

        from src.config.settings import Settings

        print("Creating Settings...")
        settings = Settings()
        print("Settings created successfully")
        print(f"LLM model: {settings.llm_settings.default_model}")
        return True
    except Exception as e:
        print(f"Settings creation failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_settings_from_container():
    """Test Settings creation from a fresh container."""
    print("\nTesting Settings from fresh container...")
    try:
        from src.core.dependency_injection import DependencyContainer
        from src.config.settings import Settings

        # Create a simple container
        container = DependencyContainer()

        # Register Settings directly
        container.register_singleton("settings", Settings, factory=Settings)

        print("Getting settings from container...")
        settings = container.get_by_name("settings")
        print("Settings retrieved successfully")
        return True
    except Exception as e:
        print(f"Container test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run tests."""
    print("=== Settings Isolation Test ===\n")

    success1 = test_settings_creation()
    success2 = test_settings_from_container()

    if success1 and success2:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")


if __name__ == "__main__":
    main()
