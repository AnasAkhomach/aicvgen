#!/usr/bin/env python3
"""
Detailed test to trace the exact source of the None await issue.
"""

import sys
import os
import asyncio
import traceback
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.services.llm_service import EnhancedLLMService

async def test_detailed_debug():
    """Detailed debugging of the await issue."""
    print("=== Detailed Debug Test ===")
    
    try:
        # Initialize LLM service
        print("1. Initializing LLM service...")
        llm_service = EnhancedLLMService()
        print(f"   ✓ LLM service initialized")
        print(f"   - LLM model: {type(llm_service.llm).__name__ if llm_service.llm else 'None'}")
        print(f"   - Executor: {type(llm_service.executor).__name__ if llm_service.executor else 'None'}")
        print(f"   - Performance optimizer: {type(llm_service.performance_optimizer).__name__ if hasattr(llm_service, 'performance_optimizer') and llm_service.performance_optimizer else 'None'}")
        
        # Test each component individually
        print("\n2. Testing performance optimizer...")
        try:
            async with llm_service.performance_optimizer.optimized_execution("test_operation"):
                print("   ✓ Performance optimizer context manager works")
        except Exception as e:
            print(f"   ✗ Performance optimizer failed: {str(e)}")
            traceback.print_exc()
            return False
        
        print("\n3. Testing loop.run_in_executor...")
        try:
            loop = asyncio.get_event_loop()
            print(f"   - Event loop: {type(loop).__name__}")
            print(f"   - Executor: {type(llm_service.executor).__name__ if llm_service.executor else 'None'}")
            
            # Test a simple function call
            def simple_test():
                return "test_result"
            
            result = await loop.run_in_executor(llm_service.executor, simple_test)
            print(f"   ✓ Simple executor test passed: {result}")
            
        except Exception as e:
            print(f"   ✗ Executor test failed: {str(e)}")
            traceback.print_exc()
            return False
        
        print("\n4. Testing _generate_with_timeout in executor...")
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                llm_service.executor, 
                llm_service._generate_with_timeout, 
                "Test prompt", 
                "test_session", 
                "test_trace"
            )
            print(f"   ✓ _generate_with_timeout in executor passed: {type(result).__name__}")
            
        except Exception as e:
            print(f"   ✗ _generate_with_timeout in executor failed: {str(e)}")
            traceback.print_exc()
            return False
        
        print("\n5. Testing full generate_content method...")
        try:
            response = await llm_service.generate_content(
                prompt="Say 'Hello test' exactly.",
                session_id="test_session"
            )
            print(f"   Response type: {type(response).__name__}")
            print(f"   Success: {response.success}")
            print(f"   Content: {response.content[:100] if response.content else 'None'}...")
            
            if response.success:
                print("   ✓ Full generate_content test passed")
                return True
            else:
                print(f"   ✗ Generate_content returned unsuccessful response: {response.content}")
                return False
                
        except Exception as e:
            print(f"   ✗ Full generate_content failed: {str(e)}")
            print(f"   - Error type: {type(e).__name__}")
            traceback.print_exc()
            return False
            
    except Exception as e:
        print(f"\n✗ Test setup failed: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_detailed_debug())
    print(f"\nTest result: {'PASSED' if success else 'FAILED'}")