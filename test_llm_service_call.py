#!/usr/bin/env python3
"""
Test to verify the get_llm_service() call issue.
"""

import sys
import os
import asyncio
import inspect
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.services.llm_service import get_llm_service

async def test_llm_service_call():
    """Test how get_llm_service() should be called."""
    print("=== Testing get_llm_service() call ===")
    
    # Check if get_llm_service is a coroutine function
    print(f"get_llm_service is coroutine function: {inspect.iscoroutinefunction(get_llm_service)}")
    
    # Try calling it without await (this should fail or return a coroutine)
    try:
        result_sync = get_llm_service()
        print(f"Sync call result type: {type(result_sync).__name__}")
        print(f"Sync call result: {result_sync}")
        
        if inspect.iscoroutine(result_sync):
            print("✓ get_llm_service() returns a coroutine - needs await!")
            result_sync.close()  # Clean up the coroutine
        else:
            print("✗ get_llm_service() does not return a coroutine")
            
    except Exception as e:
        print(f"Sync call failed: {str(e)}")
    
    # Try calling it with await
    try:
        result_async = await get_llm_service()
        print(f"Async call result type: {type(result_async).__name__}")
        print(f"Async call successful: {result_async is not None}")
        return True
        
    except Exception as e:
        print(f"Async call failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_llm_service_call())
    print(f"\nTest result: {'PASSED' if success else 'FAILED'}")