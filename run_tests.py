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
    # Run the relevant test files for the state and parser changes
    exit_code = pytest.main(
        [
            "tests/unit/test_orchestration_state.py",
            "tests/unit/test_parser_agent.py",
            "-v",
            "--tb=short",
        ]
    )
    sys.exit(exit_code)
