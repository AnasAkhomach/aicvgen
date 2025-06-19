#!/usr/bin/env python3
"""
Focused test to debug the specific await issue.
"""

import sys
import os
import asyncio
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.services.llm_service import EnhancedLLMService

async def test_await_issue():
    """Test the specific await issue."""
    print("=== Testing Await Issue ===")
    
    try:
        # Initialize LLM service
        print("1. Initializing LLM service...")
        llm_service = EnhancedLLMService()
        print(f"   ✓ LLM service initialized: {type(llm_service).__name__}")
        print(f"   - LLM model: {type(llm_service.llm).__name__ if llm_service.llm else 'None'}")
        print(f"   - Executor: {type(llm_service.executor).__name__ if llm_service.executor else 'None'}")
        
        # Test the specific method that's failing
        print("\n2. Testing _generate_with_timeout directly...")
        try:
            result = await llm_service._generate_with_timeout("Test prompt")
            print(f"   ✓ _generate_with_timeout returned: {type(result).__name__}")
            print(f"   - Has text attribute: {hasattr(result, 'text')}")
            if hasattr(result, 'text'):
                print(f"   - Text content: {result.text[:50] if result.text else 'None'}...")
        except Exception as e:
            print(f"   ✗ _generate_with_timeout failed: {str(e)}")
            print(f"   - Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            return False
        
        # Test the async generate_content method
        print("\n3. Testing generate_content async method...")
        try:
            response = await llm_service.generate_content(
                prompt="Say 'Hello test' exactly.",
                session_id="test_session"
            )
            print(f"   ✓ generate_content returned: {type(response).__name__}")
            print(f"   - Success: {response.success}")
            print(f"   - Content: {response.content[:100] if response.content else 'None'}...")
        except Exception as e:
            print(f"   ✗ generate_content failed: {str(e)}")
            print(f"   - Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            return False
            
        print("\n=== All tests passed! ===")
        return True
        
    except Exception as e:
        print(f"\n✗ Test setup failed: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_await_issue())
    print(f"\nTest result: {'PASSED' if success else 'FAILED'}")