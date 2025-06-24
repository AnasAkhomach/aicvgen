#!/usr/bin/env python3
"""
Simple test to verify logging configuration doesn't hang.
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_logging():
    """Test logging configuration."""
    print("Testing logging configuration...")

    try:
        # Test basic logging import
        from src.config.logging_config import get_logger, setup_logging

        print("✅ Logging imports successful")

        # Test setup_logging with minimal config
        logger = setup_logging(log_to_console=True, log_to_file=False)
        print("✅ Logging setup successful")

        # Test basic logging
        logger.info("Test log message")
        print("✅ Basic logging successful")

        # Test get_logger
        test_logger = get_logger("test")
        test_logger.info("Test logger message")
        print("✅ get_logger successful")

        return True

    except Exception as e:
        print(f"❌ Logging test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🔍 Testing logging configuration...")
    success = test_logging()
    if success:
        print("🎉 Logging test completed successfully!")
    else:
        print("💥 Logging test failed!")
        sys.exit(1)
