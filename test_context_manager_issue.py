#!/usr/bin/env python3
"""
Test to verify the async context manager issue.
"""

import sys
import os
import asyncio
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.core.performance_optimizer import PerformanceOptimizer

async def test_context_manager():
    """Test the optimized_execution context manager."""
    print("=== Testing optimized_execution context manager ===")
    
    try:
        # Initialize performance optimizer
        optimizer = PerformanceOptimizer()
        print("✓ Performance optimizer initialized")
        
        # Test the context manager correctly
        print("\nTesting correct usage:")
        async with optimizer.optimized_execution("test_operation"):
            print("✓ Inside context manager - this should work")
            result = "test_result"
        
        print(f"✓ Context manager completed successfully: {result}")
        
        # Test what happens if we try to await the context manager
        print("\nTesting incorrect usage (awaiting context manager):")
        try:
            # This should fail - you can't await a context manager
            result = await optimizer.optimized_execution("test_operation")
            print(f"✗ Unexpected success: {result}")
        except Exception as e:
            print(f"✓ Expected error when awaiting context manager: {str(e)}")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_context_manager())
    print(f"\nTest result: {'PASSED' if success else 'FAILED'}")