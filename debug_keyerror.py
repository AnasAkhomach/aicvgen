#!/usr/bin/env python3
"""
Debug script to check ContentType enum values and trace PROFESSIONAL_SUMMARY usage
"""

import sys
import importlib
sys.path.insert(0, '.')

try:
    from src.models.data_models import ContentType
    
    print("Available ContentType values:")
    for content_type in ContentType:
        print(f"  {content_type.name} = {content_type.value}")
    
    # Check if PROFESSIONAL_SUMMARY exists
    try:
        prof_summary = ContentType.PROFESSIONAL_SUMMARY
        print(f"\nERROR: ContentType.PROFESSIONAL_SUMMARY still exists = {prof_summary}")
        print(f"Value: {prof_summary.value}")
    except AttributeError:
        print("\nGOOD: ContentType.PROFESSIONAL_SUMMARY does not exist (this is expected)")
    
    # Check if EXECUTIVE_SUMMARY exists
    try:
        exec_summary = ContentType.EXECUTIVE_SUMMARY
        print(f"GOOD: ContentType.EXECUTIVE_SUMMARY = {exec_summary}")
        print(f"Value: {exec_summary.value}")
    except AttributeError:
        print("ERROR: ContentType.EXECUTIVE_SUMMARY does not exist (this is a problem)")
    
    # Test creating a dictionary with the enum values
    print("\nTesting dictionary access with enum values:")
    test_dict = {
        "executive_summary": "test content",
        "qualification": "test qual",
        "experience": "test exp"
    }
    
    # Test accessing with string value
    try:
        result = test_dict[ContentType.EXECUTIVE_SUMMARY.value]
        print(f"SUCCESS: test_dict[ContentType.EXECUTIVE_SUMMARY.value] = {result}")
    except KeyError as e:
        print(f"ERROR: KeyError when accessing with .value: {e}")
    
    # Test accessing with enum directly (this should fail)
    try:
        result = test_dict[ContentType.EXECUTIVE_SUMMARY]
        print(f"UNEXPECTED: test_dict[ContentType.EXECUTIVE_SUMMARY] = {result}")
    except KeyError as e:
        print(f"EXPECTED: KeyError when accessing with enum directly: {e}")
        
except Exception as e:
    print(f"Error importing ContentType: {e}")
    import traceback
    traceback.print_exc()

# Check if there are any other modules that might define PROFESSIONAL_SUMMARY
print("\nChecking for other modules that might define PROFESSIONAL_SUMMARY...")
try:
    # Check if it's defined in any other imported modules
    import inspect
    for name, obj in globals().items():
        if hasattr(obj, 'PROFESSIONAL_SUMMARY'):
            print(f"Found PROFESSIONAL_SUMMARY in {name}: {obj.PROFESSIONAL_SUMMARY}")
except Exception as e:
    print(f"Error checking globals: {e}")