#!/usr/bin/env python3
"""
Test to trace the exact location of the await error.
"""

import sys
import os
import asyncio
import traceback
import logging
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Set up detailed logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

from src.services.llm_service import EnhancedLLMService

async def test_trace_await_error():
    """Trace the exact location of the await error."""
    print("=== Tracing Await Error ===")
    
    try:
        # Initialize LLM service
        print("1. Initializing LLM service...")
        llm_service = EnhancedLLMService()
        print("   ✓ LLM service initialized")
        
        # Test with very detailed error tracking
        print("\n2. Testing generate_content with detailed tracing...")
        
        try:
            # Add some debugging to see exactly where the error occurs
            print("   - About to call generate_content...")
            response = await llm_service.generate_content(
                prompt="Say 'Hello test' exactly.",
                session_id="test_session"
            )
            print(f"   - generate_content returned: {type(response).__name__}")
            print(f"   - Response success: {response.success}")
            
            if not response.success:
                print(f"   - Error message: {response.error_message}")
                print(f"   - Content: {response.content}")
            
        except Exception as e:
            print(f"   ✗ Exception caught: {type(e).__name__}: {str(e)}")
            print("   - Full traceback:")
            traceback.print_exc()
            
            # Try to get more details about the exception
            if hasattr(e, '__cause__') and e.__cause__:
                print(f"   - Caused by: {type(e.__cause__).__name__}: {str(e.__cause__)}")
            
            if hasattr(e, '__context__') and e.__context__:
                print(f"   - Context: {type(e.__context__).__name__}: {str(e.__context__)}")
            
            return False
        
        return True
        
    except Exception as e:
        print(f"\n✗ Test setup failed: {str(e)}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_trace_await_error())
    print(f"\nTest result: {'PASSED' if success else 'FAILED'}")