import sys
import os

# Ensure the src directory is on sys.path for all tests
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)
