#!/usr/bin/env python3
"""
Test individual dependency resolution to find circular dependencies.
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


def test_individual_dependencies():
    """Test getting each dependency individually."""
    print("Testing individual dependency resolution...")

    try:
        from src.core.dependency_injection import get_container, register_core_services
        from src.config.environment import load_config

        load_config()

        container = get_container()
        register_core_services(container)

        dependencies = [
            "settings",
            "RateLimiter",
            "AdvancedCache",
            "ErrorHandler",
            "PerformanceOptimizer",
            "ContentTemplateManager",
            "ErrorRecoveryService",
        ]

        for dep_name in dependencies:
            print(f"  Getting {dep_name}...")
            try:

                def get_dep():
                    return container.get_by_name(dep_name)

                dep = run_with_timeout(get_dep, timeout_seconds=5)
                if dep is not None:
                    print(f"    ✓ {dep_name} resolved successfully")
                else:
                    print(f"    ✗ {dep_name} timed out")
                    return False
            except Exception as e:
                print(f"    ✗ {dep_name} failed: {e}")
                return False

        print("  All dependencies resolved successfully!")
        return True

    except Exception as e:
        print(f"  Failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run the test."""
    print("=== Dependency Resolution Debug ===\n")
    test_individual_dependencies()


if __name__ == "__main__":
    main()
