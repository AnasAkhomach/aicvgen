#!/usr/bin/env python3
"""Simple test runner script to handle Python path issues."""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Now run pytest
import pytest

if __name__ == "__main__":
    # Run the new test files
    exit_code = pytest.main([
        "tests/unit/test_llm_service_comprehensive.py",
        "tests/unit/test_item_processor_simplified.py",
        "-v",
        "--tb=short"
    ])
    sys.exit(exit_code)