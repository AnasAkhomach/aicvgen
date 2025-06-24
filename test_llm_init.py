#!/usr/bin/env python3
"""
Simple test script to isolate LLM service initialization issues.
This script tests the LLM service initialization step by step to identify where it hangs.
"""

import os
import sys
import time
import threading
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def test_imports():
    """Test if we can import all necessary modules."""
    print("1. Testing imports...")
    try:
        from src.config.logging_config import setup_logging, get_logger

        print("   ✓ Logging imports OK")

        from src.config.settings import Settings

        print("   ✓ Settings import OK")

        import google.generativeai as genai

        print("   ✓ Google GenerativeAI import OK")

        from src.services.llm_service import EnhancedLLMService

        print("   ✓ LLM service import OK")

        from src.core.dependency_injection import get_container, build_llm_service

        print("   ✓ Dependency injection imports OK")

        return True
    except Exception as e:
        print(f"   ✗ Import failed: {e}")
        return False


def test_basic_config():
    """Test basic configuration loading."""
    print("2. Testing basic configuration...")
    try:
        from src.config.settings import Settings

        settings = Settings()
        print(f"   ✓ Settings loaded")
        print(f"   ✓ LLM model: {settings.llm_settings.default_model}")

        # Check for API keys
        api_key = None
        if (
            hasattr(settings.llm, "gemini_api_key_primary")
            and settings.llm.gemini_api_key_primary
        ):
            api_key = settings.llm.gemini_api_key_primary[:10] + "..."
            print(f"   ✓ Primary API key found: {api_key}")
        elif (
            hasattr(settings.llm, "gemini_api_key_fallback")
            and settings.llm.gemini_api_key_fallback
        ):
            api_key = settings.llm.gemini_api_key_fallback[:10] + "..."
            print(f"   ✓ Fallback API key found: {api_key}")
        else:
            print("   ✗ No API key found")
            return False

        return True
    except Exception as e:
        print(f"   ✗ Configuration failed: {e}")
        return False


def test_genai_configure():
    """Test Google GenerativeAI configuration in isolation."""
    print("3. Testing Google GenerativeAI configuration...")
    try:
        import google.generativeai as genai
        from src.config.settings import Settings

        settings = Settings()

        # Get API key
        api_key = None
        if (
            hasattr(settings.llm, "gemini_api_key_primary")
            and settings.llm.gemini_api_key_primary
        ):
            api_key = settings.llm.gemini_api_key_primary
        elif (
            hasattr(settings.llm, "gemini_api_key_fallback")
            and settings.llm.gemini_api_key_fallback
        ):
            api_key = settings.llm.gemini_api_key_fallback

        if not api_key:
            print("   ✗ No API key available")
            return False

        # Test with timeout
        def configure_genai():
            genai.configure(api_key=api_key)
            return True

        result = run_with_timeout(configure_genai, timeout_seconds=5)
        if result:
            print("   ✓ Google GenerativeAI configured successfully")
            return True
        else:
            print("   ✗ Configuration failed")
            return False

    except Exception as e:
        print(f"   ✗ GenAI configuration failed: {e}")
        return False


def test_model_creation():
    """Test model creation in isolation."""
    print("4. Testing model creation...")
    try:
        import google.generativeai as genai
        from src.config.settings import Settings

        settings = Settings()

        def create_model():
            model = genai.GenerativeModel(settings.llm_settings.default_model)
            return model

        model = run_with_timeout(create_model, timeout_seconds=5)
        if model:
            print(
                f"   ✓ Model created successfully: {settings.llm_settings.default_model}"
            )
            return True
        else:
            print("   ✗ Model creation failed")
            return False

    except Exception as e:
        print(f"   ✗ Model creation failed: {e}")
        return False


def test_dependency_container():
    """Test dependency container initialization."""
    print("5. Testing dependency container...")
    try:
        from src.core.dependency_injection import get_container, register_core_services

        container = get_container()
        print("   ✓ Container created")

        register_core_services(container)
        print("   ✓ Core services registered")

        # Test getting basic services
        settings = container.get_by_name("settings")
        print("   ✓ Settings retrieved from container")

        return True
    except Exception as e:
        print(f"   ✗ Container test failed: {e}")
        return False


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


def test_llm_service_factory():
    """Test the actual LLM service factory."""
    print("6. Testing LLM service factory...")
    try:
        from src.core.dependency_injection import (
            get_container,
            register_core_services,
            build_llm_service,
        )

        container = get_container()
        register_core_services(container)

        # Test the factory directly
        def build_service():
            service = build_llm_service(container)
            return service

        service = run_with_timeout(build_service, timeout_seconds=15)
        if service:
            print("   ✓ LLM service factory completed successfully")
            return True
        else:
            print("   ✗ LLM service factory timed out")
            return False

    except Exception as e:
        print(f"   ✗ LLM service factory failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all tests in sequence."""
    print("=== LLM Service Initialization Debug ===\n")

    # Load environment
    try:
        from src.config.environment import load_config

        load_config()
        print("Environment loaded\n")
    except Exception as e:
        print(f"Failed to load environment: {e}\n")
        return False

    tests = [
        test_imports,
        test_basic_config,
        test_genai_configure,
        test_model_creation,
        test_dependency_container,
        test_llm_service_factory,  # Add the new test
    ]

    for i, test_func in enumerate(tests, 1):
        try:
            success = test_func()
            if not success:
                print(f"\n❌ Test {i} failed. Stopping here.")
                return False
            print()
        except Exception as e:
            print(f"\n❌ Test {i} failed with exception: {e}")
            return False

    print("✅ All tests passed! LLM service should initialize properly.")
    return True


if __name__ == "__main__":
    main()
