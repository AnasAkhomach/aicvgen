#!/usr/bin/env python3
"""
Debug script to reproduce the KeyError in the test
"""

import sys
import traceback
sys.path.insert(0, '.')

try:
    from tests.e2e.test_complete_cv_generation import TestCompleteCVGeneration
    
    # Create test instance
    test_instance = TestCompleteCVGeneration()
    
    # Try to run the test method
    print("Running test_complete_cv_generation_happy_path...")
    test_instance.test_complete_cv_generation_happy_path('software_engineer')
    print("Test completed successfully!")
    
except Exception as e:
    print(f"Error occurred: {e}")
    print("\nFull traceback:")
    traceback.print_exc()