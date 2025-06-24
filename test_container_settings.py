#!/usr/bin/env python3
"""
Test dependency container settings retrieval in isolation.
"""

import os
import sys
import time
import threading
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def run_with_timeout(func, timeout_seconds=10):
    """Run a function with timeout to prevent hanging."""
    result = [None]
    exception = [None]

    def target():
        try:
            result[0] = func()
        except Exception as e:
            exception[0] = e

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout_seconds)

    if thread.is_alive():
        print(f"   ⚠ Operation timed out after {timeout_seconds} seconds")
        return None

    if exception[0]:
        raise exception[0]

    return result[0]


def test_settings_from_container():
    """Test getting settings from the dependency container."""
    print("Testing Settings retrieval from DI container...")

    try:
        from src.config.environment import load_config
        from src.core.dependency_injection import get_container, register_core_services

        load_config()
        print("  Environment loaded")

        print("  Creating container...")
        container = get_container()
        print("  Container created")

        print("  Registering core services...")

        def register_services():
            register_core_services(container)
            return True

        result = run_with_timeout(register_services, timeout_seconds=10)
        if result:
            print("  Core services registered")
        else:
            print("  ✗ Core services registration timed out")
            return False

        print("  Getting settings from container...")

        def get_settings():
            return container.get_by_name("settings")

        settings = run_with_timeout(get_settings, timeout_seconds=5)
        if settings:
            print(f"  ✓ Settings retrieved: {type(settings)}")
            print(f"  ✓ LLM model: {settings.llm_settings.default_model}")
            return True
        else:
            print("  ✗ Settings retrieval timed out")
            return False

    except Exception as e:
        print(f"  ✗ Failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run the test."""
    print("=== Dependency Container Settings Debug ===\n")
    test_settings_from_container()


if __name__ == "__main__":
    main()
