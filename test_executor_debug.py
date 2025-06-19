#!/usr/bin/env python3
"""
Test to debug the executor initialization issue.
"""

import sys
import os
import asyncio
import traceback
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.services.llm_service import EnhancedLLMService

async def test_executor_debug():
    """Debug the executor initialization."""
    print("=== Debugging Executor Initialization ===")
    
    try:
        # Initialize LLM service
        print("1. Initializing LLM service...")
        llm_service = EnhancedLLMService()
        print("   ✓ LLM service initialized")
        
        # Check executor state
        print(f"\n2. Checking executor state...")
        print(f"   - Executor type: {type(llm_service.executor).__name__}")
        print(f"   - Executor is None: {llm_service.executor is None}")
        print(f"   - Executor max_workers: {getattr(llm_service.executor, '_max_workers', 'N/A')}")
        print(f"   - Executor shutdown: {getattr(llm_service.executor, '_shutdown', 'N/A')}")
        
        # Test a simple executor task
        print("\n3. Testing executor with simple task...")
        loop = asyncio.get_event_loop()
        
        def simple_task():
            return "Hello from executor"
        
        try:
            result = await loop.run_in_executor(llm_service.executor, simple_task)
            print(f"   ✓ Executor test successful: {result}")
        except Exception as e:
            print(f"   ✗ Executor test failed: {str(e)}")
            traceback.print_exc()
            return False
        
        # Now test the actual LLM generation
        print("\n4. Testing LLM generation...")
        try:
            response = await llm_service.generate_content(
                prompt="Say 'Hello test' exactly.",
                session_id="test_session"
            )
            print(f"   - Response success: {response.success}")
            if not response.success:
                print(f"   - Error: {response.error_message}")
                print(f"   - Content: {response.content}")
                
                # Check executor state again after failure
                print(f"\n5. Checking executor state after failure...")
                print(f"   - Executor is None: {llm_service.executor is None}")
                print(f"   - Executor shutdown: {getattr(llm_service.executor, '_shutdown', 'N/A')}")
                
        except Exception as e:
            print(f"   ✗ LLM generation failed: {str(e)}")
            traceback.print_exc()
            
            # Check executor state after exception
            print(f"\n5. Checking executor state after exception...")
            print(f"   - Executor is None: {llm_service.executor is None}")
            print(f"   - Executor shutdown: {getattr(llm_service.executor, '_shutdown', 'N/A')}")
            
            return False
        
        return True
        
    except Exception as e:
        print(f"\n✗ Test setup failed: {str(e)}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_executor_debug())
    print(f"\nTest result: {'PASSED' if success else 'FAILED'}")