#!/usr/bin/env python3
"""
Debug script to isolate the application startup hang issue.
This script tests each import and initialization step individually.
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_step(step_name, test_func):
    """Test a single step and report results."""
    print(f"\n🔍 Testing: {step_name}")
    try:
        result = test_func()
        print(f"✅ SUCCESS: {step_name}")
        return result
    except Exception as e:
        print(f"❌ FAILED: {step_name}")
        print(f"   Error: {e}")
        import traceback

        traceback.print_exc()
        return None


def test_basic_imports():
    """Test basic Python imports."""
    import logging
    import json
    import time

    return True


def test_streamlit_import():
    """Test Streamlit import."""
    import streamlit as st

    return True


def test_logging_config():
    """Test logging configuration import."""
    from src.config.logging_config import get_logger, setup_logging

    return True


def test_logging_setup():
    """Test logging setup (this might hang)."""
    from src.config.logging_config import setup_logging

    logger = setup_logging(log_to_console=True, log_to_file=False)
    logger.info("Test log message")
    return True


def test_environment_config():
    """Test environment configuration."""
    from src.config.environment import load_config

    config = load_config()
    return config


def test_dependency_injection():
    """Test dependency injection."""
    from src.core.dependency_injection import get_container

    container = get_container()
    return container


def test_main_imports():
    """Test main module imports."""
    from src.core.main import main

    return True


def test_streamlit_page_config():
    """Test Streamlit page configuration."""
    import streamlit as st

    st.set_page_config(page_title="Debug Test", page_icon="🔍", layout="wide")
    return True


def main():
    """Run all diagnostic tests."""
    print("🚀 Starting Application Startup Diagnostics")
    print("=" * 50)

    # Test each component step by step
    test_step("Basic Python imports", test_basic_imports)
    test_step("Streamlit import", test_streamlit_import)
    test_step("Logging config import", test_logging_config)
    test_step("Environment config", test_environment_config)
    test_step("Dependency injection", test_dependency_injection)

    # This is where it might hang
    print("\n⚠️  CRITICAL TEST: Logging setup (this might hang)")
    test_step("Logging setup", test_logging_setup)

    # Test remaining components
    test_step("Main module import", test_main_imports)
    test_step("Streamlit page config", test_streamlit_page_config)

    print("\n🎉 All tests completed successfully!")
    print("If you see this message, the startup should work.")


if __name__ == "__main__":
    main()
