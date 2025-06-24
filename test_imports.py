#!/usr/bin/env python3
"""
Minimal import test to identify the exact problematic module.
This script tests imports one by one to isolate the hanging issue.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_import(module_name, description):
    """Test a single import and report results."""
    print(f"Testing import: {description}")
    try:
        if module_name == "streamlit":
            import streamlit as st
        elif module_name == "src.config.environment":
            from src.config.environment import load_config
        elif module_name == "src.models.data_models":
            from src.models.data_models import AgentExecutionLog
        elif module_name == "src.core.dependency_injection":
            from src.core.dependency_injection import get_container
        elif module_name == "src.core.state_manager":
            from src.core.state_manager import StateManager
        elif module_name == "src.core":
            from src.core import StateManager
        elif module_name == "src.core.main":
            from src.core.main import main
        else:
            exec(f"import {module_name}")

        print(f"✅ SUCCESS: {description}")
        return True
    except Exception as e:
        print(f"❌ FAILED: {description} - Error: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Test imports step by step."""
    print("🔍 Testing imports step by step...")
    print("=" * 50)

    # Test basic imports first
    test_import("json", "Basic JSON import")
    test_import("logging", "Basic logging import")
    test_import("pathlib", "Basic pathlib import")

    # Test Streamlit
    test_import("streamlit", "Streamlit import")

    # Test our configuration modules
    test_import("src.config.environment", "Environment config")

    # Test data models (this might hang due to complex imports)
    print("\n⚠️  CRITICAL TEST: Data models (this might hang)")
    success = test_import("src.models.data_models", "Data models")
    if not success:
        print("🛑 Stopping here - data models failed")
        return

    # Test core modules
    print("\n⚠️  CRITICAL TEST: Core modules")
    test_import("src.core.state_manager", "State manager")
    test_import("src.core.dependency_injection", "Dependency injection")
    test_import("src.core", "Core module (__init__.py)")
    test_import("src.core.main", "Main module")

    print("\n🎉 All imports completed!")


if __name__ == "__main__":
    main()
